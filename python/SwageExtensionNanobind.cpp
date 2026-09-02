// python/SwageExtensionNanobind.cpp
//===- SwageExtensionNanobind.cpp - swage dialect python module -----------===//
//
// Exposes registration of the dialects the semantic level composes with.
//
//===----------------------------------------------------------------------===//

#include "swage-c/Codegen.h"
#include "swage-c/Dialects.h"

#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/Math.h"
#include "mlir-c/Dialect/MemRef.h"
#include "mlir-c/Dialect/Vector.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/vector.h"

#include <array>
#include <cstdint>
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
  if (blockX <= 0 || blockX > 1024)
    throw nb::value_error("block_x must be in 1..1024");
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

std::pair<std::string, std::string>
compilePTX(nb::object moduleObject, std::string kernelName, int64_t blockSize,
           std::string target, PTXKind kind, bool useTaskIds = false) {
  MlirModule module = unwrapModule(moduleObject);

  std::string lowered;
  std::string ptx;
  auto store = [](MlirStringRef value, void *output) {
    static_cast<std::string *>(output)->assign(value.data, value.length);
  };
  mlir::python::CollectDiagnosticsToStringScope diagnostics(
      mlirModuleGetContext(module));
  MlirStringRef kernel =
      mlirStringRefCreate(kernelName.data(), kernelName.size());
  MlirStringRef chip = mlirStringRefCreate(target.data(), target.size());
  MlirLogicalResult result;
  switch (kind) {
  case PTXKind::Fixed:
    result = swageCompileFixedBlockToPTX(module, kernel, blockSize, chip, store,
                                         &lowered, store, &ptx);
    break;
  case PTXKind::Segmented:
    result = swageCompileSegmentedReductionToPTX(module, kernel, blockSize,
                                                 chip, useTaskIds, store,
                                                 &lowered, store, &ptx);
    break;
  case PTXKind::Fused:
    result = swageCompileFusedSegmentedReductionToPTX(
        module, kernel, chip, store, &lowered, store, &ptx);
    break;
  case PTXKind::Persistent:
    result = swageCompilePersistentSegmentedReductionToPTX(
        module, kernel, chip, store, &lowered, store, &ptx);
    break;
  case PTXKind::SplitPartial:
    result = swageCompileSplitPartialReductionToPTX(module, kernel, chip, store,
                                                    &lowered, store, &ptx);
    break;
  case PTXKind::SplitMerge:
    result = swageCompileSplitMergeReductionToPTX(module, kernel, chip, store,
                                                  &lowered, store, &ptx);
    break;
  }
  if (mlirLogicalResultIsFailure(result))
    throw nb::value_error(diagnostics.takeMessage().c_str());
  return {std::move(lowered), std::move(ptx)};
}

std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::vector<int32_t>,
           std::vector<int32_t>>
materializeSegmentedPlan(nb::object moduleObject,
                         const std::vector<int64_t> &offsets,
                         int64_t valueCount, int64_t segmentCount,
                         int64_t warpMaxElements, int64_t ctaChunkElements) {
  MlirModule module = unwrapModule(moduleObject);
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
      module, offsets.data(), static_cast<intptr_t>(offsets.size()), valueCount,
      segmentCount, warpMaxElements, ctaChunkElements, store, &warp, store,
      &cta, store, &partial, store, &merge);
  if (mlirLogicalResultIsFailure(result))
    throw nb::value_error(diagnostics.takeMessage().c_str());
  return {std::move(warp), std::move(cta), std::move(partial),
          std::move(merge)};
}

} // namespace

NB_MODULE(_swageDialectsNanobind, m) {
  auto swageM = m.def_submodule("swage");

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

  swageM.def(
      "_compile_ptx",
      [](nb::object module, std::string kernelName, int64_t blockSize,
         std::string target) {
        return compilePTX(module, std::move(kernelName), blockSize,
                          std::move(target), PTXKind::Fixed);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("block_size"),
      nb::arg("target"));
  swageM.def(
      "_compile_segmented_reduction_ptx",
      [](nb::object module, std::string kernelName, int64_t blockSize,
         std::string target, bool useTaskIds) {
        return compilePTX(module, std::move(kernelName), blockSize,
                          std::move(target), PTXKind::Segmented, useTaskIds);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("block_size"),
      nb::arg("target"), nb::arg("use_task_ids") = false);
  swageM.def(
      "_compile_fused_segmented_reduction_ptx",
      [](nb::object module, std::string kernelName, std::string target) {
        return compilePTX(module, std::move(kernelName), 128, std::move(target),
                          PTXKind::Fused);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("target"));
  swageM.def(
      "_compile_persistent_segmented_reduction_ptx",
      [](nb::object module, std::string kernelName, std::string target) {
        return compilePTX(module, std::move(kernelName), 512, std::move(target),
                          PTXKind::Persistent);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("target"));
  swageM.def(
      "_compile_split_partial_reduction_ptx",
      [](nb::object module, std::string kernelName, std::string target) {
        return compilePTX(module, std::move(kernelName), 128, std::move(target),
                          PTXKind::SplitPartial);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("target"));
  swageM.def(
      "_compile_split_merge_reduction_ptx",
      [](nb::object module, std::string kernelName, std::string target) {
        return compilePTX(module, std::move(kernelName), 128, std::move(target),
                          PTXKind::SplitMerge);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("target"));
  swageM.def("_materialize_segmented_plan", &materializeSegmentedPlan,
             nb::arg("module"), nb::arg("offsets"), nb::arg("value_count"),
             nb::arg("segment_count"), nb::arg("warp_max_elements") = 32,
             nb::arg("cta_chunk_elements") = 4096);
}
