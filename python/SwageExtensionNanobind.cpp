// python/SwageExtensionNanobind.cpp
//===- SwageExtensionNanobind.cpp - swage dialect python module -----------===//
//
// Exposes registration of the dialects the semantic level composes with.
//
//===----------------------------------------------------------------------===//

#include "swage-c/Codegen.h"
#include "swage-c/Dialects.h"
#include "swage-c/Target.h"
#include "swage/Python/BuildIdentity.h"

#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/Math.h"
#include "mlir-c/Dialect/MemRef.h"
#include "mlir-c/Dialect/Vector.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/vector.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <dlfcn.h>

namespace nb = nanobind;

namespace {

enum class PTXKind {
  Fixed,
  Segmented,
  Fused,
  Persistent,
  SplitPartial,
  SplitMerge
};

/// CUDA driver entry points resolved at runtime. The extension must not
/// link against libcuda: CPU-only builds and CI have no driver, and the
/// Python ctypes wrapper stays as the fallback dispatch path.
struct CudaLauncher {
  using LaunchFn = int (*)(void *, unsigned, unsigned, unsigned, unsigned,
                           unsigned, unsigned, unsigned, void *, void **,
                           void **);
  using ErrorTextFn = int (*)(int, const char **);
  using CurrentContextFn = int (*)(void **);
  using ContextIdFn = int (*)(void *, unsigned long long *);
  using EventRecordFn = int (*)(void *, void *);
  using StreamIsCapturingFn = int (*)(void *, int *);
  LaunchFn launch = nullptr;
  ErrorTextFn errorName = nullptr;
  ErrorTextFn errorString = nullptr;
  CurrentContextFn currentContext = nullptr;
  // Context ids exist from CUDA 12. An older driver has only handles.
  ContextIdFn contextId = nullptr;
  EventRecordFn eventRecord = nullptr;
  StreamIsCapturingFn streamIsCapturing = nullptr;
};

const CudaLauncher &cudaLauncher() {
  static const CudaLauncher launcher = [] {
    CudaLauncher resolved;
    void *library = dlopen("libcuda.so.1", RTLD_NOW);
    if (!library)
      return resolved;
    resolved.launch = reinterpret_cast<CudaLauncher::LaunchFn>(
        dlsym(library, "cuLaunchKernel"));
    resolved.errorName = reinterpret_cast<CudaLauncher::ErrorTextFn>(
        dlsym(library, "cuGetErrorName"));
    resolved.errorString = reinterpret_cast<CudaLauncher::ErrorTextFn>(
        dlsym(library, "cuGetErrorString"));
    resolved.currentContext = reinterpret_cast<CudaLauncher::CurrentContextFn>(
        dlsym(library, "cuCtxGetCurrent"));
    resolved.contextId = reinterpret_cast<CudaLauncher::ContextIdFn>(
        dlsym(library, "cuCtxGetId"));
    resolved.eventRecord = reinterpret_cast<CudaLauncher::EventRecordFn>(
        dlsym(library, "cuEventRecord"));
    resolved.streamIsCapturing =
        reinterpret_cast<CudaLauncher::StreamIsCapturingFn>(
            dlsym(library, "cuStreamIsCapturing"));
    return resolved;
  }();
  return launcher;
}

void checkCUDAResult(const CudaLauncher &launcher, const char *operation,
                     int result) {
  if (result == 0)
    return;
  const char *name = nullptr;
  const char *text = nullptr;
  if (launcher.errorName)
    launcher.errorName(result, &name);
  if (launcher.errorString)
    launcher.errorString(result, &text);
  throw std::runtime_error(std::string("CUDA Driver ") + operation +
                           " failed: " + (name ? name : "unknown") + " (" +
                           std::to_string(result) +
                           "): " + (text ? text : "unknown"));
}

/// The context that is current on this thread, as `_CudaDriver` names it:
/// its id where the driver has context ids, and its handle otherwise.
uint64_t currentContextIdentity(const CudaLauncher &launcher) {
  unsigned long long identifier = 0;
  // A null context asks for the id of the current context.
  if (launcher.contextId && launcher.contextId(nullptr, &identifier) == 0)
    return identifier;
  void *context = nullptr;
  checkCUDAResult(launcher, "cuCtxGetCurrent",
                  launcher.currentContext(&context));
  return reinterpret_cast<uint64_t>(context);
}

MlirModule unwrapModule(nb::object moduleObject) {
  std::optional<nb::object> capsule =
      nb::detail::mlirApiObjectToCapsule(moduleObject);
  if (!capsule)
    throw nb::type_error("module must be an mlir_swage.ir.Module");
  MlirModule module = mlirPythonCapsuleToModule(capsule->ptr());
  if (mlirModuleIsNull(module))
    throw nb::type_error("module must be an mlir_swage.ir.Module");
  return module;
}

/// Keeps two compiles off one MLIR context.
///
/// A compile runs without the GIL, and swage-c/Codegen.h requires that no
/// other thread uses the context of the module until the call returns. Two
/// compiles of one context, for example of the same module from two threads,
/// would break that rule and abort inside MLIR, so the second one waits
/// here. Compiles on different contexts do not wait for each other.
///
/// The guard covers compiles only. While a compile runs, a Python thread
/// that parses into the same context, builds IR in it, or loads a dialect
/// into it is a data race the guard cannot see. The upstream bindings do not
/// meet this case, because their PassManager.run keeps the GIL, so it is an
/// obligation of this binding's callers: give each thread its own context,
/// or keep a shared context idle until the compile returns.
///
/// The guard is taken after the GIL is released and dropped before the GIL is
/// taken back, so a thread never waits for one while it holds the other. An
/// entry point of this module that keeps the GIL and runs passes on a
/// caller's context is not covered either: it must take this guard if it can
/// overlap a compile of the same context.
class ContextUse {
public:
  explicit ContextUse(MlirContext context) : context(context.ptr) {
    std::unique_lock<std::mutex> lock(state().mutex);
    state().released.wait(
        lock, [&] { return state().busy.insert(this->context).second; });
  }
  ContextUse(const ContextUse &) = delete;
  ContextUse &operator=(const ContextUse &) = delete;
  ~ContextUse() {
    {
      std::lock_guard<std::mutex> lock(state().mutex);
      state().busy.erase(context);
    }
    state().released.notify_all();
  }

private:
  struct State {
    std::mutex mutex;
    std::condition_variable released;
    std::set<const void *> busy;
  };
  static State &state() {
    static State shared;
    return shared;
  }

  const void *context;
};

/// Returns the lowered module, the PTX, and the launch contract of the
/// kernel as JSON. `blockSize` and `useTaskIds` reach only the kinds whose C
/// entry point takes them, the fixed and the segmented one. The other kinds
/// compile at a width the C API fixes, so their callers pass neither.
std::tuple<std::string, std::string, std::string>
compilePTX(nb::object moduleObject, std::string kernelName, std::string target,
           PTXKind kind, int64_t blockSize = 0, bool useTaskIds = false) {
  MlirModule module = unwrapModule(moduleObject);
  MlirContext context = mlirModuleGetContext(module);

  std::string lowered;
  std::string ptx;
  std::string contract;
  std::string message;
  auto store = [](MlirStringRef value, void *output) {
    static_cast<std::string *>(output)->assign(value.data, value.length);
  };
  MlirStringRef kernel =
      mlirStringRefCreate(kernelName.data(), kernelName.size());
  MlirStringRef chip = mlirStringRefCreate(target.data(), target.size());
  MlirLogicalResult result;
  {
    // A cold compile takes milliseconds and never calls back into Python, so
    // the other Python threads run meanwhile. Everything in this scope is
    // plain C++ on locals. The diagnostics handler is attached inside the
    // guard: attached earlier, it would also collect the diagnostics of the
    // compile this thread is waiting for.
    nb::gil_scoped_release release;
    ContextUse use(context);
    mlir::python::CollectDiagnosticsToStringScope diagnostics(context);
    switch (kind) {
    case PTXKind::Fixed:
      result =
          swageCompileFixedBlockToPTX(module, kernel, blockSize, chip, store,
                                      &lowered, store, &ptx, store, &contract);
      break;
    case PTXKind::Segmented:
      result = swageCompileSegmentedReductionToPTX(
          module, kernel, blockSize, chip, useTaskIds, store, &lowered, store,
          &ptx, store, &contract);
      break;
    case PTXKind::Fused:
      result = swageCompileFusedSegmentedReductionToPTX(
          module, kernel, chip, store, &lowered, store, &ptx, store, &contract);
      break;
    case PTXKind::Persistent:
      result = swageCompilePersistentSegmentedReductionToPTX(
          module, kernel, chip, store, &lowered, store, &ptx, store, &contract);
      break;
    case PTXKind::SplitPartial:
      result = swageCompileSplitPartialReductionToPTX(
          module, kernel, chip, store, &lowered, store, &ptx, store, &contract);
      break;
    case PTXKind::SplitMerge:
      result = swageCompileSplitMergeReductionToPTX(
          module, kernel, chip, store, &lowered, store, &ptx, store, &contract);
      break;
    }
    message = diagnostics.takeMessage();
  }
  if (mlirLogicalResultIsFailure(result))
    throw nb::value_error(message.c_str());
  return {std::move(lowered), std::move(ptx), std::move(contract)};
}

/// The storage of the arguments of one host call: one eight-byte slot per
/// argument, and the pointer to each slot that the packed call reads.
struct alignas(uint64_t) ArgumentSlot {
  std::array<unsigned char, sizeof(uint64_t)> bytes{};
};

struct ArgumentBuffer {
  std::vector<ArgumentSlot> slots;
  std::vector<void *> parameters;
};

/// The width in bits of an argument of contract kind `kind`.
unsigned argumentBitWidth(const std::string &kind, size_t index) {
  if (kind == "i1")
    return 1;
  if (kind == "i8")
    return 8;
  if (kind == "i16" || kind == "f16" || kind == "bf16")
    return 16;
  if (kind == "i32" || kind == "f32")
    return 32;
  if (kind == "ptr" || kind == "i64" || kind == "f64")
    return 64;
  throw nb::value_error(("argument " + std::to_string(index) +
                         " has the unknown contract kind '" + kind + "'")
                            .c_str());
}

/// The raw bits of one argument, given as a Python integer that fits the
/// width of its kind.
uint64_t parseRawArgument(PyObject *value, const std::string &kind,
                          size_t index, unsigned bitWidth) {
  if (!PyLong_CheckExact(value))
    throw nb::type_error(
        ("argument " + std::to_string(index) + " must be a Python integer")
            .c_str());
  unsigned long long converted = PyLong_AsUnsignedLongLong(value);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    throw nb::value_error(
        (kind + " argument " + std::to_string(index) + " is out of range")
            .c_str());
  }
  auto raw = static_cast<uint64_t>(converted);
  if (bitWidth < 64 && raw >= (uint64_t{1} << bitWidth))
    throw nb::value_error(
        (kind + " argument " + std::to_string(index) + " is out of range")
            .c_str());
  return raw;
}

/// Pack the arguments of one host call. A value keeps its low bytes, which
/// are the value only on a little-endian host.
ArgumentBuffer packArguments(const std::vector<std::string> &argumentKinds,
                             nb::sequence argumentValues) {
  const uint16_t probe = 1;
  if (*reinterpret_cast<const unsigned char *>(&probe) != 1)
    throw std::runtime_error("host calls require a little-endian host");
  const size_t valueCount = nb::len(argumentValues);
  if (argumentKinds.size() != valueCount)
    throw nb::value_error(("the call has " +
                           std::to_string(argumentKinds.size()) +
                           " argument kinds and " + std::to_string(valueCount) +
                           " argument values")
                              .c_str());
  ArgumentBuffer buffer;
  buffer.slots.resize(argumentKinds.size());
  buffer.parameters.resize(argumentKinds.size());
  for (size_t index = 0; index < argumentKinds.size(); ++index) {
    const std::string &kind = argumentKinds[index];
    unsigned bitWidth = argumentBitWidth(kind, index);
    uint64_t raw =
        parseRawArgument(argumentValues[index].ptr(), kind, index, bitWidth);
    std::memcpy(buffer.slots[index].bytes.data(), &raw, (bitWidth + 7) / 8);
    buffer.parameters[index] = buffer.slots[index].bytes.data();
  }
  return buffer;
}

std::array<uint32_t, 3> parseGeometry(nb::sequence value, const char *label) {
  if (nb::len(value) != 3)
    throw nb::value_error(
        (std::string(label) + " must contain exactly three axes").c_str());
  std::array<uint32_t, 3> geometry;
  for (size_t index = 0; index < geometry.size(); ++index) {
    PyObject *axis = value[index].ptr();
    if (!PyLong_CheckExact(axis))
      throw nb::type_error(
          (std::string(label) + " axes must be Python integers").c_str());
    unsigned long long converted = PyLong_AsUnsignedLongLong(axis);
    if (PyErr_Occurred() || converted == 0 ||
        converted > std::numeric_limits<uint32_t>::max()) {
      PyErr_Clear();
      throw nb::value_error(
          (std::string(label) + " axes must be positive u32 values").c_str());
    }
    geometry[index] = static_cast<uint32_t>(converted);
  }
  return geometry;
}

void launchCUDAKernel(const std::vector<std::string> &argumentKinds,
                      nb::sequence argumentValues, nb::sequence gridValue,
                      nb::sequence blockValue, int64_t sharedMemory,
                      uint64_t stream, uint64_t function) {
  const CudaLauncher &launcher = cudaLauncher();
  if (!launcher.launch)
    throw std::runtime_error("CUDA Driver library libcuda.so.1 is unavailable");
  std::array<uint32_t, 3> grid = parseGeometry(gridValue, "grid");
  std::array<uint32_t, 3> block = parseGeometry(blockValue, "block");
  uint64_t threads = static_cast<uint64_t>(block[0]) * block[1] * block[2];
  if (threads > 1024)
    throw nb::value_error("block may contain at most 1024 threads");
  if (sharedMemory < 0 || sharedMemory > int64_t(UINT32_MAX))
    throw nb::value_error("shared_memory must be a u32");

  ArgumentBuffer arguments = packArguments(argumentKinds, argumentValues);
  int result = launcher.launch(
      reinterpret_cast<void *>(function), grid[0], grid[1], grid[2], block[0],
      block[1], block[2], static_cast<unsigned>(sharedMemory),
      reinterpret_cast<void *>(stream), arguments.parameters.data(), nullptr);
  checkCUDAResult(launcher, "cuLaunchKernel", result);
}

nb::object checkedObject(PyObject *value) {
  if (!value)
    throw nb::python_error();
  return nb::steal<nb::object>(value);
}

nb::object callOne(nb::handle function, nb::handle argument) {
  return checkedObject(PyObject_CallOneArg(function.ptr(), argument.ptr()));
}

nb::object callTwo(nb::handle function, nb::handle first, nb::handle second) {
  PyObject *arguments[] = {first.ptr(), second.ptr()};
  return checkedObject(
      PyObject_Vectorcall(function.ptr(), arguments, 2, nullptr));
}

PyCFunction noArgsTensorMethod(nb::handle method, PyTypeObject *tensorType) {
  if (Py_TYPE(method.ptr()) != &PyMethodDescr_Type ||
      !PyType_IsSubtype(tensorType, PyDescr_TYPE(method.ptr())))
    return nullptr;
  PyMethodDef *definition =
      reinterpret_cast<PyMethodDescrObject *>(method.ptr())->d_method;
  return definition->ml_flags == METH_NOARGS ? definition->ml_meth : nullptr;
}

bool hasTensorInstanceState(PyObject *tensor) {
#if PY_VERSION_HEX >= 0x030D0000
  if (Py_TYPE(tensor)->tp_flags & Py_TPFLAGS_MANAGED_DICT) {
    // Inspect managed values without materializing an otherwise absent dict.
    int state = PyObject_VisitManagedDict(
        tensor, [](PyObject *, void *) { return 1; }, nullptr);
    if (state < 0)
      throw nb::python_error();
    return state != 0;
  }
#endif
  PyObject **dictionary = _PyObject_GetDictPtr(tensor);
  if (!dictionary && PyErr_Occurred())
    throw nb::python_error();
  return dictionary && *dictionary && PyDict_GET_SIZE(*dictionary) != 0;
}

/// Bound methods retain a lock, never the entry or either residency cache.
struct CachedPythonLock {
  nb::object object;
  nb::object acquire;
  nb::object release;

  CachedPythonLock() = default;
  explicit CachedPythonLock(nb::object lock)
      : object(std::move(lock)), acquire(object.attr("acquire")),
        release(object.attr("release")) {}

  int traverse(visitproc visit, void *arg) const {
    Py_VISIT(object.ptr());
    Py_VISIT(acquire.ptr());
    Py_VISIT(release.ptr());
    return 0;
  }
};

class TryPythonLock {
public:
  explicit TryPythonLock(const CachedPythonLock &lock)
      : lock(lock), held(callOne(lock.acquire, nb::handle(Py_False))
                             .is(nb::handle(Py_True))) {}
  TryPythonLock(const TryPythonLock &) = delete;
  TryPythonLock &operator=(const TryPythonLock &) = delete;
  ~TryPythonLock() {
    if (!held)
      return;
    PyObject *result = PyObject_CallNoArgs(lock.release.ptr());
    if (!result)
      PyErr_WriteUnraisable(lock.release.ptr());
    Py_XDECREF(result);
  }
  explicit operator bool() const { return held; }

private:
  const CachedPythonLock &lock;
  bool held;
};

/// A guarded, fixed-ABI dispatch specialization, not a module lifetime lease.
/// In particular, no bound OrderedDict method may be retained here: it would
/// keep an evicted cache (and all its loaded modules) alive.
class FixedCUDALaunch {
public:
  FixedCUDALaunch(nb::object entryRef, nb::object artifact, nb::object kernel,
                  nb::object torch, nb::object stream, nb::object runtime,
                  nb::object cudaBackend, nb::object dtype)
      : entryRef(std::move(entryRef)), artifact(std::move(artifact)),
        stream(std::move(stream)) {
    static constexpr const char *attributeNames[] = {
        "_artifact_cache", "_loaded_functions",
        "_retired_loaded", "_cache_lock",
        "_cuda_lock",      "_compiler_identity",
        "_identity_cache", "_logger",
        "_driver",         "driver",
        "cache_resident",  "unloading",
        "unloaded",        "unload_blocked",
        "context",         "function",
        "pending_streams", "is_cuda",
        "dtype",           "ndim",
        "requires_grad"};
    static_assert(std::size(attributeNames) == AttributeCount);
    for (size_t i = 0; i < names.size(); ++i)
      names[i] = checkedObject(PyUnicode_InternFromString(attributeNames[i]));

    if (!PyWeakref_CheckRef(this->entryRef.ptr()) ||
        !PyModule_Check(runtime.ptr()) || !PyModule_Check(cudaBackend.ptr()))
      throw nb::type_error("fixed CUDA launch requires an entry weakref and "
                           "runtime modules");
    nb::object entry = dereference(this->entryRef);
    if (entry.is_none())
      throw nb::value_error("fixed CUDA launch entry is no longer alive");
    runtimeRef = weakref(runtime);
    cudaBackendRef = weakref(cudaBackend);
    nb::object artifactCache = attribute(runtime, ArtifactCache);
    nb::object loadedCache = attribute(cudaBackend, LoadedCache);
    nb::object orderedDict =
        nb::module_::import_("collections").attr("OrderedDict");
    if (Py_TYPE(artifactCache.ptr()) !=
            reinterpret_cast<PyTypeObject *>(orderedDict.ptr()) ||
        Py_TYPE(loadedCache.ptr()) !=
            reinterpret_cast<PyTypeObject *>(orderedDict.ptr()))
      throw nb::type_error("fixed CUDA launch requires OrderedDict caches");
    artifactCacheRef = weakref(artifactCache);
    loadedCacheRef = weakref(loadedCache);
    moveToEnd = orderedDict.attr("move_to_end");
    driverRef = weakref(attribute(entry, EntryDriver));
    // Construction runs under entry.lock and _cuda_lock in the Python owner.
    // Never acquire _cache_lock here; a concurrent native hit may hold it.
    artifactLock = CachedPythonLock(attribute(runtime, CacheLock));
    cudaLock = CachedPythonLock(attribute(cudaBackend, CudaLock));
    identity = attribute(runtime, IdentityCache);
    logger = attribute(runtime, Logger);
    loggingEnabled = logger.attr("isEnabledFor");
    debugLevel = nb::int_(10);
    artifactKey = this->artifact.attr("key");
    loadedKey = entry.attr("key");
    contextValue = attribute(entry, Context);
    functionValue = attribute(entry, Function);
    context = nb::cast<uint64_t>(contextValue);
    function = nb::cast<uint64_t>(functionValue);
    rawStreamValue = this->stream.attr("cuda_stream");
    rawStream = nb::cast<uint64_t>(rawStreamValue);
    if (rawStream) {
      nb::object events = entry.attr("events");
      if (!PyDict_CheckExact(events.ptr()))
        throw nb::type_error("fixed CUDA launch requires stream events");
      PyObject *value = dictItem(events, rawStreamValue);
      if (!value)
        throw nb::value_error("fixed CUDA launch stream has no event");
      event = nb::cast<uint64_t>(nb::handle(value));
      if (!event)
        throw nb::value_error("fixed CUDA launch requires a live event");
    }
    deviceValue = this->stream.attr("device").attr("index");
    device = nb::cast<int32_t>(deviceValue);

    nb::sequence parameterNames =
        nb::cast<nb::sequence>(kernel.attr("parameter_names"));
    if (nb::len(parameterNames) != parameters.size())
      throw nb::value_error("fixed CUDA launch requires five parameters");
    for (size_t i = 0; i < parameters.size(); ++i) {
      parameters[i] = nb::borrow<nb::object>(parameterNames[i]);
      if (!PyUnicode_CheckExact(parameters[i].ptr()))
        throw nb::type_error("fixed CUDA parameter names must be strings");
    }
    nb::object contract = this->artifact.attr("contract");
    nb::object geometry = contract.attr("launch").attr("block");
    if (!PyTuple_CheckExact(geometry.ptr()) ||
        PyTuple_GET_SIZE(geometry.ptr()) != 3 ||
        !positiveI32(PyTuple_GET_ITEM(geometry.ptr(), 0), block) ||
        block > 1024 ||
        nb::cast<int>(nb::handle(PyTuple_GET_ITEM(geometry.ptr(), 1))) != 1 ||
        nb::cast<int>(nb::handle(PyTuple_GET_ITEM(geometry.ptr(), 2))) != 1)
      throw nb::value_error("fixed CUDA launch requires a fixed block");

    tensorType = torch.attr("Tensor");
    if (!PyType_Check(tensorType.ptr()))
      throw nb::type_error("torch.Tensor must be a type");
    elementDtype = std::move(dtype);
    elementBytes = nb::cast<uint64_t>(elementDtype.attr("itemsize"));
    numel = tensorType.attr("numel");
    isContiguous = tensorType.attr("is_contiguous");
    isNeg = tensorType.attr("is_neg");
    isConj = tensorType.attr("is_conj");
    getDevice = tensorType.attr("get_device");
    dataPtr = tensorType.attr("data_ptr");
    recordStream = tensorType.attr("record_stream");
    // Metadata queries must not run arbitrary Python between validation and
    // pointer extraction. Allocator recording runs only after enqueue.
    bool ordinaryMethods = Py_TYPE(isContiguous.ptr()) == &PyMethodDescr_Type;
    auto *tensorClass = reinterpret_cast<PyTypeObject *>(tensorType.ptr());
    numelMethod = noArgsTensorMethod(numel, tensorClass);
    negativeMethod = noArgsTensorMethod(isNeg, tensorClass);
    conjugateMethod = noArgsTensorMethod(isConj, tensorClass);
    deviceMethod = noArgsTensorMethod(getDevice, tensorClass);
    pointerMethod = noArgsTensorMethod(dataPtr, tensorClass);
    ordinaryMethods &= numelMethod && negativeMethod && conjugateMethod &&
                       deviceMethod && pointerMethod;
    for (Attribute name : {IsCUDA, Dtype, NDim, RequiresGrad}) {
      size_t index = name - IsCUDA;
      nb::object &descriptor = tensorDescriptors[index];
      descriptor = attribute(tensorType, name);
      if (Py_TYPE(descriptor.ptr()) != &PyGetSetDescr_Type ||
          !PyType_IsSubtype(tensorClass, PyDescr_TYPE(descriptor.ptr()))) {
        ordinaryMethods = false;
        continue;
      }
      tensorGetters[index] =
          reinterpret_cast<PyGetSetDescrObject *>(descriptor.ptr())->d_getset;
      ordinaryMethods &= tensorGetters[index]->get != nullptr;
    }
    if (ordinaryMethods && tensorClass->tp_getattro == PyObject_GenericGetAttr)
      tensorTypeVersion = tensorClass->tp_version_tag;

    isInBadFork = torch.attr("cuda").attr("_is_in_bad_fork");
    currentDevice = torch.attr("_C").attr("_cuda_getDevice");
    currentRawStream = torch.attr("_C").attr("_cuda_getCurrentRawStream");
    isCapturing = torch.attr("_C").attr("_cuda_isCurrentStreamCapturing");
    torchFunctionMode =
        torch.attr("_C").attr("_is_torch_function_mode_enabled");
    // A kernel store is invisible to autograd, so every launch advances the
    // version counter of its output, as the Python launch does. The public
    // increment_version wraps one tensor in a tuple for
    // torch._C._increment_version; calling that directly saves a Python
    // frame per launch. It is used only when it takes a tuple, which an empty
    // one shows without touching any tensor; a single tensor would be read
    // as the sequence of its elements.
    incrementVersion =
        torch.attr("autograd").attr("graph").attr("increment_version");
    nb::object versionCounter =
        nb::getattr(torch.attr("_C"), "_increment_version", nb::none());
    if (PyCallable_Check(versionCounter.ptr())) {
      nb::object empty = checkedObject(PyTuple_New(0));
      if (PyObject *probed =
              PyObject_CallOneArg(versionCounter.ptr(), empty.ptr())) {
        Py_DECREF(probed);
        incrementVersionTuple = std::move(versionCounter);
      } else {
        PyErr_Clear();
      }
    }

    // Ordinary attribute reads stay live. A changed class invalidates the
    // shortcut rather than silently accepting different lookup semantics.
    entryType = nb::borrow<nb::object>(
        reinterpret_cast<PyObject *>(Py_TYPE(entry.ptr())));
    if (Py_TYPE(entry.ptr())->tp_getattro == PyObject_GenericGetAttr &&
        Py_TYPE(entryType.ptr()) == &PyType_Type)
      entryTypeVersion = Py_TYPE(entry.ptr())->tp_version_tag;
  }

  bool call(nb::handle arguments, nb::handle constexprs, nb::handle grid) {
    if (!entryRef || !entryTypeVersion || !tensorTypeVersion ||
        reinterpret_cast<PyTypeObject *>(tensorType.ptr())->tp_version_tag !=
            tensorTypeVersion ||
        !exactStringDict(arguments, 4) || !exactStringDict(constexprs, 1) ||
        !PyTuple_CheckExact(grid.ptr()) || PyTuple_GET_SIZE(grid.ptr()) != 1)
      return false;
    uint32_t requestedBlock, count, blocks;
    if (!positiveI32(dictItem(constexprs, parameters[4]), requestedBlock) ||
        requestedBlock != block ||
        !positiveI32(dictItem(arguments, parameters[3]), count) ||
        !positiveI32(PyTuple_GET_ITEM(grid.ptr(), 0), blocks) ||
        blocks != (uint64_t(count) + block - 1) / block)
      return false;

    // Pin dictionary values before any Torch call can release the GIL.
    // No tensor or exported view is retained beyond this call.
    std::array<nb::object, 3> tensors;
    for (size_t i = 0; i < tensors.size(); ++i) {
      PyObject *tensor = dictItem(arguments, parameters[i]);
      if (!tensor ||
          Py_TYPE(tensor) !=
              reinterpret_cast<PyTypeObject *>(tensorType.ptr()) ||
          hasTensorInstanceState(tensor))
        return false;
      tensors[i] = nb::borrow<nb::object>(tensor);
    }

    nb::object entry = dereference(entryRef);
    nb::object runtime = dereference(runtimeRef);
    nb::object cudaBackend = dereference(cudaBackendRef);
    nb::object artifactCache = dereference(artifactCacheRef);
    nb::object loadedCache = dereference(loadedCacheRef);
    nb::object driver = dereference(driverRef);
    if (entry.is_none() || runtime.is_none() || cudaBackend.is_none() ||
        artifactCache.is_none() || loadedCache.is_none() || driver.is_none())
      return false;

    // A cached launcher implies CUDA was initialized in this process. Forked
    // children must go through PyTorch's public initialization error path,
    // without making even a current-device or stream query here.
    if (nb::cast<bool>(checkedObject(PyObject_CallNoArgs(isInBadFork.ptr()))) ||
        nb::cast<bool>(
            checkedObject(PyObject_CallNoArgs(torchFunctionMode.ptr()))))
      return false;

    // Validate every tensor and all scalar/device/geometry constraints before
    // requesting any raw pointer. Metadata is never cached by tensor identity.
    for (const nb::object &tensor : tensors) {
      if (!tensorProperty(tensor, IsCUDA).is(nb::handle(Py_True)) ||
          !tensorProperty(tensor, RequiresGrad).is(nb::handle(Py_False)) ||
          !tensorProperty(tensor, Dtype).is(elementDtype) ||
          nb::cast<int32_t>(tensorProperty(tensor, NDim)) != 1 ||
          !callOne(isContiguous, tensor).is(nb::handle(Py_True)) ||
          !checkedObject(negativeMethod(tensor.ptr(), nullptr))
               .is(nb::handle(Py_False)) ||
          !checkedObject(conjugateMethod(tensor.ptr(), nullptr))
               .is(nb::handle(Py_False)) ||
          nb::cast<int64_t>(checkedObject(numelMethod(tensor.ptr(), nullptr))) <
              int64_t(count) ||
          nb::cast<int32_t>(
              checkedObject(deviceMethod(tensor.ptr(), nullptr))) != device)
        return false;
    }
    const CudaLauncher &launcher = cudaLauncher();
    // The capture query comes after the current-stream comparison, so the
    // stream it asks about is the current one.
    if (nb::cast<int32_t>(checkedObject(
            PyObject_CallNoArgs(currentDevice.ptr()))) != device ||
        nb::cast<uint64_t>(callOne(currentRawStream, deviceValue)) !=
            rawStream ||
        streamCapturing(launcher) ||
        nb::cast<bool>(callOne(loggingEnabled, debugLevel)))
      return false;

    if (reinterpret_cast<PyTypeObject *>(tensorType.ptr())->tp_version_tag !=
            tensorTypeVersion ||
        nb::cast<bool>(
            checkedObject(PyObject_CallNoArgs(torchFunctionMode.ptr()))))
      return false;
    uint64_t pointers[3];
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (hasTensorInstanceState(tensors[i].ptr()))
        return false;
      // data_ptr includes storage offsets; logical metadata was checked above.
      nb::object pointer =
          checkedObject(pointerMethod(tensors[i].ptr(), nullptr));
      if (!PyLong_CheckExact(pointer.ptr()))
        return false;
      pointers[i] = PyLong_AsUnsignedLongLong(pointer.ptr());
      if (PyErr_Occurred()) {
        if (!PyErr_ExceptionMatches(PyExc_OverflowError))
          throw nb::python_error();
        PyErr_Clear();
        return false;
      }
      if (!pointers[i])
        return false;
    }
    uint64_t activeBytes = uint64_t(count) * elementBytes;
    for (uint64_t input : {pointers[0], pointers[1]}) {
      uint64_t distance =
          input < pointers[2] ? pointers[2] - input : input - pointers[2];
      if (distance && distance < activeBytes)
        return false;
    }

    if (!launcher.launch || !launcher.currentContext ||
        (rawStream && !launcher.eventRecord))
      return false;
    std::exception_ptr failure;
    {
      // Native enqueues keep the GIL and both residency locks. Eviction and
      // retirement cannot begin until enqueue finishes; no per-entry lock
      // is needed to protect a cache-resident module or its event handles.
      TryPythonLock artifactGuard(artifactLock);
      if (!artifactGuard)
        return false;
      TryPythonLock cudaGuard(cudaLock);
      if (!cudaGuard)
        return false;
      // Torch calls above may release the GIL. Do not trust earlier residency,
      // identity, lock, or driver observations after any such call.
      if (Py_TYPE(entry.ptr()) !=
              reinterpret_cast<PyTypeObject *>(entryType.ptr()) ||
          Py_TYPE(entry.ptr())->tp_version_tag != entryTypeVersion ||
          reinterpret_cast<PyTypeObject *>(tensorType.ptr())->tp_version_tag !=
              tensorTypeVersion ||
          !moduleState(runtime, cudaBackend, artifactCache, loadedCache,
                       driver) ||
          !attribute(entry, EntryDriver).is(driver) ||
          !attribute(entry, Context).is(contextValue) ||
          !attribute(entry, Function).is(functionValue) ||
          !attribute(entry, CacheResident).is(nb::handle(Py_True)) ||
          !attribute(entry, Unloading).is(nb::handle(Py_False)) ||
          !attribute(entry, Unloaded).is(nb::handle(Py_False)) ||
          !attribute(entry, UnloadBlocked).is(nb::handle(Py_False)) ||
          dictItem(artifactCache, artifactKey) != artifact.ptr() ||
          dictItem(loadedCache, loadedKey) != entry.ptr())
        return false;
      PyObject *retired = moduleItem(cudaBackend, RetiredLoaded);
      if (!retired || !PyDict_CheckExact(retired) || PyDict_GET_SIZE(retired))
        return false;
      if (currentContextIdentity(launcher) != context)
        return false;
      if (PyDict_GET_SIZE(artifactCache.ptr()) > 1)
        callTwo(moveToEnd, artifactCache, artifactKey);
      if (PyDict_GET_SIZE(loadedCache.ptr()) > 1)
        callTwo(moveToEnd, loadedCache, loadedKey);
      // Only the context-owned legacy stream is safe to fence later.
      // Externally owned streams may be destroyed immediately after use.
      if (!rawStream) {
        nb::object pendingStreams = attribute(entry, PendingStreams);
        if (!PySet_CheckExact(pendingStreams.ptr()))
          return false;
        if (PySet_Add(pendingStreams.ptr(), rawStreamValue.ptr()) < 0)
          throw nb::python_error();
      }

      int32_t scalar = static_cast<int32_t>(count);
      void *parameters[] = {&pointers[0], &pointers[1], &pointers[2], &scalar};
      // After this point there is no False path, including enqueue errors.
      // The GIL remains held across the driver enqueue.
      try {
        checkCUDAResult(launcher, "cuLaunchKernel",
                        launcher.launch(reinterpret_cast<void *>(function),
                                        blocks, 1, 1, block, 1, 1, 0,
                                        reinterpret_cast<void *>(rawStream),
                                        parameters, nullptr));
        if (rawStream)
          checkCUDAResult(
              launcher, "cuEventRecord",
              launcher.eventRecord(reinterpret_cast<void *>(event),
                                   reinterpret_cast<void *>(rawStream)));
      } catch (...) {
        failure = std::current_exception();
        // No future operation may safely reuse a caller-owned stream after
        // an enqueue/fence error. Retain this module through context teardown.
        if (rawStream &&
            PyObject_SetAttr(entry.ptr(), names[UnloadBlocked].ptr(), Py_True) <
                0)
          PyErr_WriteUnraisable(entry.ptr());
      }
    }
    // Mirror the Python finally path even on an enqueue error, and
    // attempt all three allocator records if any one record itself fails.
    for (const nb::object &tensor : tensors) {
      try {
        callTwo(recordStream, tensor, stream);
      } catch (...) {
        if (!failure)
          failure = std::current_exception();
      }
    }
    if (failure)
      std::rethrow_exception(failure);
    if (incrementVersionTuple)
      callOne(incrementVersionTuple,
              checkedObject(PyTuple_Pack(1, tensors[2].ptr())));
    else
      callOne(incrementVersion, tensors[2]);
    return true;
  }

  /// Whether the launch stream captures a CUDA graph, asked of the driver.
  ///
  /// PyTorch's `_cuda_isCurrentStreamCapturing` asks the runtime the same
  /// question about the current stream, which the caller has just compared
  /// with `rawStream`; the driver answers it without a Python call. A
  /// capture that is active or invalidated, or a query that fails, counts as
  /// capturing, which hands the launch to the Python path. Without the driver
  /// entry point the PyTorch query answers.
  bool streamCapturing(const CudaLauncher &launcher) const {
    if (!launcher.streamIsCapturing)
      return nb::cast<bool>(
          checkedObject(PyObject_CallNoArgs(isCapturing.ptr())));
    int status = 0;
    return launcher.streamIsCapturing(reinterpret_cast<void *>(rawStream),
                                      &status) != 0 ||
           status != 0;
  }

  static int traverse(PyObject *self, visitproc visit, void *arg) {
    Py_VISIT(Py_TYPE(self));
    if (!nb::inst_ready(self))
      return 0;
    const auto &value = *nb::inst_ptr<FixedCUDALaunch>(self);
    for (PyObject *reference :
         {value.entryRef.ptr(),        value.artifact.ptr(),
          value.entryType.ptr(),       value.runtimeRef.ptr(),
          value.cudaBackendRef.ptr(),  value.artifactCacheRef.ptr(),
          value.loadedCacheRef.ptr(),  value.driverRef.ptr(),
          value.identity.ptr(),        value.stream.ptr(),
          value.tensorType.ptr(),      value.elementDtype.ptr(),
          value.numel.ptr(),           value.isContiguous.ptr(),
          value.isNeg.ptr(),           value.isConj.ptr(),
          value.getDevice.ptr(),       value.dataPtr.ptr(),
          value.recordStream.ptr(),    value.isInBadFork.ptr(),
          value.currentDevice.ptr(),   value.currentRawStream.ptr(),
          value.isCapturing.ptr(),     value.logger.ptr(),
          value.loggingEnabled.ptr(),  value.debugLevel.ptr(),
          value.moveToEnd.ptr(),       value.artifactKey.ptr(),
          value.loadedKey.ptr(),       value.contextValue.ptr(),
          value.functionValue.ptr(),   value.rawStreamValue.ptr(),
          value.deviceValue.ptr(),     value.torchFunctionMode.ptr(),
          value.incrementVersion.ptr()})
      Py_VISIT(reference);
    Py_VISIT(value.incrementVersionTuple.ptr());
    for (const nb::object &name : value.names)
      Py_VISIT(name.ptr());
    for (const nb::object &parameter : value.parameters)
      Py_VISIT(parameter.ptr());
    for (const nb::object &descriptor : value.tensorDescriptors)
      Py_VISIT(descriptor.ptr());
    for (const CachedPythonLock *lock :
         {&value.artifactLock, &value.cudaLock}) {
      int result = lock->traverse(visit, arg);
      if (result)
        return result;
    }
    return 0;
  }

  static int clear(PyObject *self) {
    if (nb::inst_ready(self))
      *nb::inst_ptr<FixedCUDALaunch>(self) = FixedCUDALaunch();
    return 0;
  }

private:
  FixedCUDALaunch() = default;
  enum Attribute {
    ArtifactCache,
    LoadedCache,
    RetiredLoaded,
    CacheLock,
    CudaLock,
    CompilerIdentity,
    IdentityCache,
    Logger,
    Driver,
    EntryDriver,
    CacheResident,
    Unloading,
    Unloaded,
    UnloadBlocked,
    Context,
    Function,
    PendingStreams,
    IsCUDA,
    Dtype,
    NDim,
    RequiresGrad,
    AttributeCount
  };

  static nb::object weakref(nb::handle value) {
    return checkedObject(PyWeakref_NewRef(value.ptr(), nullptr));
  }
  static nb::object dereference(nb::handle reference) {
#if PY_VERSION_HEX >= 0x030D0000
    PyObject *value = nullptr;
    int alive = PyWeakref_GetRef(reference.ptr(), &value);
    if (alive < 0)
      throw nb::python_error();
    return alive ? nb::steal<nb::object>(value) : nb::none();
#else
    PyObject *value = PyWeakref_GetObject(reference.ptr());
    if (!value)
      throw nb::python_error();
    return nb::borrow<nb::object>(value);
#endif
  }
  nb::object attribute(nb::handle object, Attribute name) const {
    return checkedObject(PyObject_GetAttr(object.ptr(), names[name].ptr()));
  }
  nb::object tensorProperty(nb::handle tensor, Attribute name) const {
    PyGetSetDef *definition = tensorGetters[name - IsCUDA];
    return checkedObject(definition->get(tensor.ptr(), definition->closure));
  }
  static PyObject *dictItem(nb::handle dictionary, nb::handle key) {
    PyObject *value = PyDict_GetItemWithError(dictionary.ptr(), key.ptr());
    if (!value && PyErr_Occurred())
      throw nb::python_error();
    return value;
  }
  PyObject *moduleItem(nb::handle module, Attribute name) const {
    return dictItem(nb::handle(PyModule_GetDict(module.ptr())), names[name]);
  }
  static bool exactStringDict(nb::handle value, Py_ssize_t size) {
    if (!PyDict_CheckExact(value.ptr()) || PyDict_GET_SIZE(value.ptr()) != size)
      return false;
    Py_ssize_t position = 0;
    PyObject *key, *item;
    while (PyDict_Next(value.ptr(), &position, &key, &item))
      if (!PyUnicode_CheckExact(key))
        return false;
    return true;
  }
  static bool positiveI32(PyObject *value, uint32_t &converted) {
    if (!value || !PyLong_CheckExact(value))
      return false;
    int overflow = 0;
    long long number = PyLong_AsLongLongAndOverflow(value, &overflow);
    if (PyErr_Occurred())
      throw nb::python_error();
    if (overflow || number <= 0 || number > INT32_MAX)
      return false;
    converted = static_cast<uint32_t>(number);
    return true;
  }
  bool moduleState(nb::handle runtime, nb::handle cudaBackend,
                   nb::handle artifactCache, nb::handle loadedCache,
                   nb::handle driver) const {
    return moduleItem(runtime, ArtifactCache) == artifactCache.ptr() &&
           moduleItem(cudaBackend, LoadedCache) == loadedCache.ptr() &&
           moduleItem(runtime, CacheLock) == artifactLock.object.ptr() &&
           moduleItem(cudaBackend, CudaLock) == cudaLock.object.ptr() &&
           moduleItem(runtime, IdentityCache) == identity.ptr() &&
           PyTuple_CheckExact(identity.ptr()) &&
           PyTuple_GET_SIZE(identity.ptr()) == 2 &&
           moduleItem(runtime, CompilerIdentity) ==
               PyTuple_GET_ITEM(identity.ptr(), 0) &&
           moduleItem(runtime, Logger) == logger.ptr() &&
           moduleItem(cudaBackend, Driver) == driver.ptr();
  }

  std::array<nb::object, AttributeCount> names;
  std::array<nb::object, 5> parameters;
  nb::object entryType;
  uint32_t entryTypeVersion = 0, tensorTypeVersion = 0;
  nb::object entryRef, artifact, runtimeRef, cudaBackendRef;
  nb::object artifactCacheRef, loadedCacheRef, driverRef, identity, stream;
  nb::object tensorType, elementDtype, numel, isContiguous, isNeg, isConj;
  nb::object getDevice, dataPtr, recordStream;
  std::array<nb::object, 4> tensorDescriptors;
  std::array<PyGetSetDef *, 4> tensorGetters{};
  PyCFunction numelMethod = nullptr, deviceMethod = nullptr;
  PyCFunction negativeMethod = nullptr, conjugateMethod = nullptr;
  PyCFunction pointerMethod = nullptr;
  nb::object isInBadFork, currentDevice, currentRawStream, isCapturing;
  nb::object torchFunctionMode, incrementVersion, incrementVersionTuple;
  nb::object logger, loggingEnabled, debugLevel, moveToEnd;
  nb::object artifactKey, loadedKey, contextValue, functionValue;
  nb::object rawStreamValue, deviceValue;
  CachedPythonLock artifactLock, cudaLock;
  uint64_t context = 0, function = 0, rawStream = 0, event = 0;
  uint64_t elementBytes = 0;
  int32_t device = 0;
  uint32_t block = 0;
};

PyType_Slot fixedCUDALaunchSlots[] = {
    {Py_tp_traverse, reinterpret_cast<void *>(FixedCUDALaunch::traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(FixedCUDALaunch::clear)},
    {0, nullptr}};

/// The entry and the argument kinds of a host-call contract.
std::pair<std::string, std::vector<std::string>>
parseContractABI(const std::string &contract) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(contract);
  if (!parsed)
    throw std::runtime_error("invalid host launch contract: " +
                             llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  const llvm::json::Array *arguments =
      object ? object->getArray("arguments") : nullptr;
  std::optional<llvm::StringRef> entry =
      object ? object->getString("entry") : std::nullopt;
  if (!entry || !arguments)
    throw std::runtime_error(
        "invalid host launch contract: it has no entry or no arguments");
  std::vector<std::string> kinds;
  kinds.reserve(arguments->size());
  for (const llvm::json::Value &value : *arguments) {
    const llvm::json::Object *argument = value.getAsObject();
    std::optional<llvm::StringRef> kind =
        argument ? argument->getString("kind") : std::nullopt;
    if (!kind)
      throw std::runtime_error(
          "invalid host launch contract: an argument has no kind");
    kinds.push_back(kind->str());
  }
  return {entry->str(), std::move(kinds)};
}

/// A native executable of the fixed elementwise kernel, owned by Python.
class BoundHostExecutable {
public:
  explicit BoundHostExecutable(SwageHostExecutable handle) : handle(handle) {}
  BoundHostExecutable(const BoundHostExecutable &) = delete;
  BoundHostExecutable &operator=(const BoundHostExecutable &) = delete;
  BoundHostExecutable(BoundHostExecutable &&other) noexcept
      : handle(other.handle), entryName(std::move(other.entryName)),
        expectedKinds(std::move(other.expectedKinds)) {
    other.handle.ptr = nullptr;
  }
  BoundHostExecutable &operator=(BoundHostExecutable &&other) noexcept {
    if (this == &other)
      return *this;
    swageHostExecutableDestroy(handle);
    handle = other.handle;
    entryName = std::move(other.entryName);
    expectedKinds = std::move(other.expectedKinds);
    other.handle.ptr = nullptr;
    return *this;
  }
  ~BoundHostExecutable() { swageHostExecutableDestroy(handle); }

  /// Record the entry and the argument kinds of the contract.
  void setABI(std::string entry, std::vector<std::string> kinds) {
    entryName = std::move(entry);
    expectedKinds = std::move(kinds);
  }

  const std::string &entry() const { return entryName; }

  /// Run the executable once. The kinds must be those of the contract, and
  /// each value is the raw bits of its argument: a buffer address or a
  /// scalar. The call runs without the GIL.
  void invoke(const std::vector<std::string> &argumentKinds,
              nb::sequence argumentValues) {
    if (argumentKinds != expectedKinds)
      throw nb::value_error(
          "the argument kinds of a host call differ from its contract");
    ArgumentBuffer arguments = packArguments(argumentKinds, argumentValues);
    std::string error;
    auto store = [](MlirStringRef value, void *output) {
      static_cast<std::string *>(output)->assign(value.data, value.length);
    };
    MlirLogicalResult result;
    {
      nb::gil_scoped_release release;
      result = swageHostExecutableInvoke(
          handle, arguments.parameters.data(),
          static_cast<intptr_t>(arguments.parameters.size()), store, &error);
    }
    if (mlirLogicalResultIsFailure(result))
      throw std::runtime_error(error.empty() ? "the host call failed" : error);
  }

private:
  SwageHostExecutable handle;
  std::string entryName;
  std::vector<std::string> expectedKinds;
};

/// Compile the fixed elementwise kernel for the host. Returns the lowered
/// module, the executable, and the launch contract as JSON. The compile runs
/// without the GIL and under the guard of its context, like compilePTX.
std::tuple<std::string, BoundHostExecutable, std::string>
compileFixedHost(nb::object moduleObject, std::string kernelName,
                 int64_t blockSize) {
  MlirModule module = unwrapModule(moduleObject);
  MlirContext context = mlirModuleGetContext(module);
  std::string lowered;
  std::string contract;
  std::string message;
  auto store = [](MlirStringRef value, void *output) {
    static_cast<std::string *>(output)->assign(value.data, value.length);
  };
  MlirStringRef kernel =
      mlirStringRefCreate(kernelName.data(), kernelName.size());
  SwageHostExecutable handle{nullptr};
  {
    nb::gil_scoped_release release;
    ContextUse use(context);
    mlir::python::CollectDiagnosticsToStringScope diagnostics(context);
    handle = swageCompileFixedBlockToHost(module, kernel, blockSize, store,
                                          &lowered, store, &contract);
    message = diagnostics.takeMessage();
  }
  if (swageHostExecutableIsNull(handle))
    throw nb::value_error(message.c_str());
  // The executable is owned before anything below can throw.
  BoundHostExecutable executable(handle);
  auto [entry, kinds] = parseContractABI(contract);
  executable.setABI(std::move(entry), std::move(kinds));
  return {std::move(lowered), std::move(executable), std::move(contract)};
}

std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::vector<int32_t>,
           std::vector<int32_t>>
materializeSegmentedPlan(nb::object moduleObject, const std::string &kernelName,
                         const std::vector<int64_t> &offsets,
                         int64_t valueCount, int64_t segmentCount,
                         int64_t warpMaxElements, int64_t ctaChunkElements) {
  MlirModule module = unwrapModule(moduleObject);
  // The plan call reads a module in the caller's context, which the runtime
  // shares between threads; keep it apart from a compile that released the
  // GIL.
  ContextUse use(mlirModuleGetContext(module));
  mlir::python::CollectDiagnosticsToStringScope diagnostics(
      mlirModuleGetContext(module));
  std::vector<int32_t> warp;
  std::vector<int32_t> cta;
  std::vector<int32_t> partial;
  std::vector<int32_t> merge;
  auto store = [](const int32_t *taskIds, intptr_t taskCount, void *output) {
    auto &tasks = *static_cast<std::vector<int32_t> *>(output);
    if (taskCount)
      tasks.assign(taskIds, taskIds + taskCount);
  };
  MlirLogicalResult result = swageMaterializeSegmentedPlan(
      module, mlirStringRefCreate(kernelName.data(), kernelName.size()),
      offsets.data(), static_cast<intptr_t>(offsets.size()), valueCount,
      segmentCount, warpMaxElements, ctaChunkElements, store, &warp, store,
      &cta, store, &partial, store, &merge);
  if (mlirLogicalResultIsFailure(result))
    throw nb::value_error(diagnostics.takeMessage().c_str());
  return {std::move(warp), std::move(cta), std::move(partial),
          std::move(merge)};
}

using PlanOffsets =
    nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using PlanRecords = nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>;

/// Hands one record vector to Python as an int32 array that owns it, so no
/// Python integer is created per record.
PlanRecords takePlanRecords(std::vector<int32_t> &&records) {
  auto *owned = new std::vector<int32_t>(std::move(records));
  // An empty vector may have no storage; the array still needs an address.
  owned->reserve(1);
  nb::capsule owner(owned, [](void *storage) noexcept {
    delete static_cast<std::vector<int32_t> *>(storage);
  });
  return PlanRecords(owned->data(), {owned->size()}, owner);
}

/// The Python entry point of materializeSegmentedPlan. Offsets arrive as one
/// host int32 buffer and the four record arrays leave as buffers. The C API
/// classifies int64 offsets, so they are widened here in one pass.
std::tuple<PlanRecords, PlanRecords, PlanRecords, PlanRecords>
materializeSegmentedPlanBuffers(nb::object moduleObject,
                                const std::string &kernelName,
                                PlanOffsets offsets, int64_t valueCount,
                                int64_t segmentCount, int64_t warpMaxElements,
                                int64_t ctaChunkElements) {
  std::vector<int64_t> wideOffsets(offsets.data(),
                                   offsets.data() + offsets.shape(0));
  auto [warp, cta, partial, merge] = materializeSegmentedPlan(
      std::move(moduleObject), kernelName, wideOffsets, valueCount,
      segmentCount, warpMaxElements, ctaChunkElements);
  return {takePlanRecords(std::move(warp)), takePlanRecords(std::move(cta)),
          takePlanRecords(std::move(partial)),
          takePlanRecords(std::move(merge))};
}

/// The Python entry point of swageClassifySegments: the records of one
/// layout in one int32 array, laid out as SwageTaskRecordsCallback states,
/// then the warp, CTA, partial, and merge counts. It takes no module and
/// touches no MLIR context, so it needs no ContextUse. The GIL stays held:
/// the call lasts microseconds, which is less than releasing and retaking
/// the GIL costs when threads contend for it. The classifier reads the
/// offsets twice, to count and then to write, so the buffer must not change
/// during the call; the runtime passes a host copy that nothing else holds.
std::tuple<PlanRecords, intptr_t, intptr_t, intptr_t, intptr_t>
classifySegments(PlanOffsets offsets, int64_t valueCount, int64_t segmentCount,
                 int64_t warpMaxElements, int64_t ctaChunkElements) {
  struct Classification {
    std::vector<int32_t> records;
    intptr_t warpCount = 0;
    intptr_t ctaCount = 0;
    intptr_t partialCount = 0;
    intptr_t mergeCount = 0;
    std::string error;
  } classification;
  auto store = [](const int32_t *records, intptr_t warpCount, intptr_t ctaCount,
                  intptr_t partialCount, intptr_t mergeCount, void *output) {
    auto &result = *static_cast<Classification *>(output);
    result.records.assign(records, records + warpCount + ctaCount +
                                       3 * partialCount + 3 * mergeCount);
    result.warpCount = warpCount;
    result.ctaCount = ctaCount;
    result.partialCount = partialCount;
    result.mergeCount = mergeCount;
  };
  auto fail = [](MlirStringRef message, void *output) {
    static_cast<Classification *>(output)->error.assign(message.data,
                                                        message.length);
  };
  MlirLogicalResult result = swageClassifySegments(
      offsets.data(), static_cast<intptr_t>(offsets.shape(0)), valueCount,
      segmentCount, warpMaxElements, ctaChunkElements, store, &classification,
      fail, &classification);
  if (mlirLogicalResultIsFailure(result))
    throw nb::value_error(classification.error.c_str());
  return {takePlanRecords(std::move(classification.records)),
          classification.warpCount, classification.ctaCount,
          classification.partialCount, classification.mergeCount};
}

/// Lets a `swage` frontend that is already imported check these bindings
/// against itself. It reads only the build identity, which is all the module
/// holds when this runs. The frontend owns the rule and raises on a mismatch,
/// which fails the import. A frontend from before the check existed cannot
/// refuse anything, so its use is reported with one warning. Nothing is checked
/// when `swage` is not imported: the bindings are usable on their own, and
/// `swage` checks bindings that were loaded first when it reaches for them.
void verifyLoadedFrontend(nb::handle bindings) {
  nb::object modules = nb::module_::import_("sys").attr("modules");
  nb::object frontend = modules.attr("get")("swage");
  if (frontend.is_none())
    return;
  nb::object runtime = nb::getattr(frontend, "_runtime", nb::none());
  if (nb::hasattr(runtime, "_verify_bindings")) {
    runtime.attr("_verify_bindings")(bindings);
    return;
  }
  nb::object file = nb::getattr(frontend, "__file__", nb::none());
  std::string message =
      "the swage package at " + nb::cast<std::string>(nb::str(file)) +
      " predates the check that pairs a frontend with its bindings; the "
      "mlir_swage bindings were built for swage " SWAGE_BUILD_VERSION
      " at revision " SWAGE_BUILD_REVISION
      ", and nothing verified that this frontend matches them";
  if (PyErr_WarnEx(PyExc_RuntimeWarning, message.c_str(), 1) < 0)
    throw nb::python_error();
}

} // namespace

NB_MODULE(_swageDialectsNanobind, m) {
  auto swageM = m.def_submodule("swage");

  // What this extension was built from: the `swage` version, the source
  // revision of its checkout and the digest of the frontend sources beside
  // it, and the LLVM release it was compiled and linked against. `swage`
  // refuses bindings built for another version or beside a frontend that
  // does not fit, and its environment report prints the version, the
  // revision, and the LLVM release.
  swageM.attr("__version__") = SWAGE_BUILD_VERSION;
  swageM.attr("__source_revision__") = SWAGE_BUILD_REVISION;
  swageM.attr("__frontend_digest__") = SWAGE_BUILD_FRONTEND;
  swageM.attr("__llvm_version__") = LLVM_VERSION_STRING;

  // Before anything is registered, so a refused import leaves nothing
  // behind and the next attempt is refused for the same reason.
  verifyLoadedFrontend(swageM);

  // The target description the lowerings and the C API read, as plain
  // values, so the host takes block widths, claim batches, and planning
  // defaults from the compiler instead of repeating them.
  const SwageTargetDescription target = swageGetTargetDescription();
  swageM.def("_target_description", [target]() {
    auto text = [](MlirStringRef value) {
      return std::string(value.data, value.length);
    };
    nb::list processors;
    for (intptr_t index = 0; index < target.processorCount; ++index)
      processors.append(target.processors[index]);
    nb::dict record;
    record["name"] = text(target.name);
    record["triple"] = text(target.triple);
    record["processor_prefix"] = text(target.processorPrefix);
    record["processors"] = nb::tuple(processors);
    record["subgroup_width"] = target.subgroupWidth;
    record["max_block_threads"] = target.maxBlockThreads;
    record["cta_block_threads"] = target.ctaBlockThreads;
    record["split_block_threads"] = target.splitBlockThreads;
    record["persistent_block_threads"] = target.persistentBlockThreads;
    record["persistent_partial_claim"] = target.persistentPartialClaim;
    record["persistent_warp_claim"] = target.persistentWarpClaim;
    record["default_warp_max_elements"] = target.defaultWarpMaxElements;
    record["default_cta_chunk_elements"] = target.defaultCtaChunkElements;
    return record;
  });

  nb::class_<FixedCUDALaunch>(swageM, "_FixedCUDALaunch",
                              nb::is_weak_referenceable(),
                              nb::type_slots(fixedCUDALaunchSlots))
      .def(nb::init<nb::object, nb::object, nb::object, nb::object, nb::object,
                    nb::object, nb::object, nb::object>(),
           nb::arg("entry_ref"), nb::arg("artifact"), nb::arg("kernel"),
           nb::arg("torch"), nb::arg("stream"), nb::arg("runtime"),
           nb::arg("cuda_backend"), nb::arg("dtype"))
      .def("__call__", &FixedCUDALaunch::call, nb::arg("arguments"),
           nb::arg("constexprs"), nb::arg("grid"));

  // The GIL is deliberately held across cuLaunchKernel: the enqueue is
  // microseconds, the driver never re-enters Python, and releasing it per
  // launch makes contended multithreaded dispatch an order of magnitude
  // slower through GIL reacquisition convoys.
  swageM.def("_launch_cuda_kernel", &launchCUDAKernel,
             nb::arg("argument_kinds"), nb::arg("argument_values"),
             nb::arg("grid"), nb::arg("block"), nb::arg("shared_memory"),
             nb::arg("stream"), nb::arg("function"));

  swageM.def(
      "register_dialects",
      [](MlirContext context, bool load) {
        MlirDialectHandle handles[] = {
            mlirGetDialectHandle__swage__(),  mlirGetDialectHandle__func__(),
            mlirGetDialectHandle__arith__(),  mlirGetDialectHandle__math__(),
            mlirGetDialectHandle__memref__(), mlirGetDialectHandle__vector__()};
        for (MlirDialectHandle handle : handles) {
          mlirDialectHandleRegisterDialect(handle, context);
          if (load)
            mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  // `!swage.segment<T>`. A type that reaches Python from the parser or from
  // an operation arrives as this class through its registered type ID.
  auto segmentType = mlir::python::nanobind_adaptors::mlir_type_subclass(
      swageM, "SegmentType", swageTypeIsASegment, swageSegmentTypeGetTypeID);
  segmentType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType elementType) {
        mlir::python::CollectDiagnosticsToStringScope diagnostics(
            mlirTypeGetContext(elementType));
        MlirType segment = swageSegmentTypeGet(elementType);
        if (mlirTypeIsNull(segment))
          throw nb::value_error(diagnostics.takeMessage().c_str());
        return cls(segment);
      },
      nb::arg("cls"), nb::arg("element_type"));
  segmentType.def_property_readonly("element_type", [](MlirType self) {
    return swageSegmentTypeGetElementType(self);
  });

  nb::class_<BoundHostExecutable>(swageM, "_HostExecutable")
      .def_prop_ro("entry", &BoundHostExecutable::entry)
      .def("invoke", &BoundHostExecutable::invoke, nb::arg("argument_kinds"),
           nb::arg("argument_values"));
  swageM.def("_compile_fixed_host", &compileFixedHost, nb::arg("module"),
             nb::arg("kernel_name"), nb::arg("block_size"));

  // Each compile function returns the lowered module, the PTX, and the
  // launch contract of the kernel as JSON.
  swageM.def(
      "_compile_ptx",
      [](nb::object module, std::string kernelName, int64_t blockSize,
         std::string target) {
        return compilePTX(module, std::move(kernelName), std::move(target),
                          PTXKind::Fixed, blockSize);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("block_size"),
      nb::arg("target"));
  swageM.def(
      "_compile_segmented_reduction_ptx",
      [](nb::object module, std::string kernelName, int64_t blockSize,
         std::string target, bool useTaskIds) {
        return compilePTX(module, std::move(kernelName), std::move(target),
                          PTXKind::Segmented, blockSize, useTaskIds);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("block_size"),
      nb::arg("target"), nb::arg("use_task_ids") = false);
  swageM.def(
      "_compile_fused_segmented_reduction_ptx",
      [](nb::object module, std::string kernelName, std::string target) {
        return compilePTX(module, std::move(kernelName), std::move(target),
                          PTXKind::Fused);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("target"));
  swageM.def(
      "_compile_persistent_segmented_reduction_ptx",
      [](nb::object module, std::string kernelName, std::string target) {
        return compilePTX(module, std::move(kernelName), std::move(target),
                          PTXKind::Persistent);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("target"));
  swageM.def(
      "_compile_split_partial_reduction_ptx",
      [](nb::object module, std::string kernelName, std::string target) {
        return compilePTX(module, std::move(kernelName), std::move(target),
                          PTXKind::SplitPartial);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("target"));
  swageM.def(
      "_compile_split_merge_reduction_ptx",
      [](nb::object module, std::string kernelName, std::string target) {
        return compilePTX(module, std::move(kernelName), std::move(target),
                          PTXKind::SplitMerge);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("target"));
  // Offsets are a contiguous rank-one host int32 buffer and the records come
  // back as four int32 arrays. Nothing is converted: a list, a tuple, or a
  // buffer of another dtype, rank, or layout is a TypeError.
  swageM.def("_materialize_segmented_plan", &materializeSegmentedPlanBuffers,
             nb::arg("module"), nb::arg("kernel_name"),
             nb::arg("offsets").noconvert(), nb::arg("value_count"),
             nb::arg("segment_count"),
             nb::arg("warp_max_elements") = target.defaultWarpMaxElements,
             nb::arg("cta_chunk_elements") = target.defaultCtaChunkElements);
  // The relative work of the element programs of a module, or None when a
  // region holds an operation without a weight.
  swageM.def(
      "_element_work",
      [](nb::object moduleObject) -> nb::object {
        MlirModule module = unwrapModule(moduleObject);
        // The estimate reads a module in the caller's context; keep it apart
        // from a compile of that context that released the GIL.
        ContextUse use(mlirModuleGetContext(module));
        int64_t work = swageEstimateElementWork(module);
        if (work < 0)
          return nb::none();
        return nb::int_(work);
      },
      nb::arg("module"));
  // The same offsets buffer and limits without a module. A program is
  // admitted once through `_materialize_segmented_plan`; this classifies
  // each of its layouts.
  swageM.def("_classify_segments", &classifySegments,
             nb::arg("offsets").noconvert(), nb::arg("value_count"),
             nb::arg("segment_count"),
             nb::arg("warp_max_elements") = target.defaultWarpMaxElements,
             nb::arg("cta_chunk_elements") = target.defaultCtaChunkElements);
}
