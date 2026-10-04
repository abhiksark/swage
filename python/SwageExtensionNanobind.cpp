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
  LaunchFn launch = nullptr;
  ErrorTextFn errorName = nullptr;
  ErrorTextFn errorString = nullptr;
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
    return resolved;
  }();
  return launcher;
}

void launchKernel(uint64_t function, int64_t gridX, int64_t blockX,
                  uint64_t stream, std::vector<uint64_t> &pointers,
                  std::vector<int32_t> &scalars) {
  constexpr size_t maxArguments = 16;
  const CudaLauncher &launcher = cudaLauncher();
  if (!launcher.launch)
    throw std::runtime_error("CUDA Driver library libcuda.so.1 is unavailable");
  if (gridX <= 0 || gridX > int64_t(UINT32_MAX))
    throw nb::value_error("grid_x must be a positive u32");
  static const int64_t maxBlockThreads =
      swageGetTargetDescription().maxBlockThreads;
  if (blockX <= 0 || blockX > maxBlockThreads)
    throw nb::value_error(
        ("block_x must be in 1.." + std::to_string(maxBlockThreads)).c_str());
  if (pointers.size() + scalars.size() > maxArguments)
    throw nb::value_error("too many kernel arguments");
  std::array<void *, maxArguments> parameters;
  size_t index = 0;
  for (uint64_t &pointer : pointers)
    parameters[index++] = &pointer;
  for (int32_t &scalar : scalars)
    parameters[index++] = &scalar;
  int result = launcher.launch(
      reinterpret_cast<void *>(function), static_cast<unsigned>(gridX), 1, 1,
      static_cast<unsigned>(blockX), 1, 1, 0, reinterpret_cast<void *>(stream),
      parameters.data(), nullptr);
  if (result == 0)
    return;
  const char *name = nullptr;
  const char *text = nullptr;
  if (launcher.errorName)
    launcher.errorName(result, &name);
  if (launcher.errorString)
    launcher.errorString(result, &text);
  throw std::runtime_error(std::string("CUDA Driver cuLaunchKernel failed: ") +
                           (name ? name : "unknown") + " (" +
                           std::to_string(result) +
                           "): " + (text ? text : "unknown"));
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

  // What this extension was built from: the `swage` version and the source
  // revision of its checkout, and the LLVM release it was compiled and
  // linked against. `swage` refuses bindings built for another version, and
  // its environment report prints all three.
  swageM.attr("__version__") = SWAGE_BUILD_VERSION;
  swageM.attr("__source_revision__") = SWAGE_BUILD_REVISION;
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

  // The GIL is deliberately held across cuLaunchKernel: the enqueue is
  // microseconds, the driver never re-enters Python, and releasing it per
  // launch makes contended multithreaded dispatch an order of magnitude
  // slower through GIL reacquisition convoys.
  swageM.def(
      "_launch_kernel",
      [](uint64_t function, int64_t gridX, int64_t blockX, uint64_t stream,
         std::vector<uint64_t> pointers, std::vector<int32_t> scalars) {
        launchKernel(function, gridX, blockX, stream, pointers, scalars);
      },
      nb::arg("function"), nb::arg("grid_x"), nb::arg("block_x"),
      nb::arg("stream"), nb::arg("pointers"), nb::arg("scalars"));

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
