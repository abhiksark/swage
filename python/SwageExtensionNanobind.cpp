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
#include "nanobind/stl/vector.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

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
           std::string target, bool segmented, bool useTaskIds,
           bool fusedMixed = false) {
  MlirModule module = unwrapModule(moduleObject);

  std::string lowered;
  std::string ptx;
  auto store = [](MlirStringRef value, void *output) {
    static_cast<std::string *>(output)->assign(value.data, value.length);
  };
  mlir::python::CollectDiagnosticsToStringScope diagnostics(
      mlirModuleGetContext(module));
  MlirLogicalResult result =
      fusedMixed
          ? swageCompileFusedSegmentedReductionToPTX(
                module,
                mlirStringRefCreate(kernelName.data(), kernelName.size()),
                mlirStringRefCreate(target.data(), target.size()), store,
                &lowered, store, &ptx)
      : segmented
          ? swageCompileSegmentedReductionToPTX(
                module,
                mlirStringRefCreate(kernelName.data(), kernelName.size()),
                blockSize, mlirStringRefCreate(target.data(), target.size()),
                useTaskIds, store, &lowered, store, &ptx)
          : swageCompileFixedBlockToPTX(
                module,
                mlirStringRefCreate(kernelName.data(), kernelName.size()),
                blockSize, mlirStringRefCreate(target.data(), target.size()),
                store, &lowered, store, &ptx);
  if (mlirLogicalResultIsFailure(result))
    throw nb::value_error(diagnostics.takeMessage().c_str());
  return {std::move(lowered), std::move(ptx)};
}

std::pair<std::vector<int32_t>, std::vector<int32_t>> materializeSegmentedPlan(
    nb::object moduleObject, const std::vector<int64_t> &offsets,
    int64_t valueCount, int64_t segmentCount, int64_t warpMaxElements) {
  MlirModule module = unwrapModule(moduleObject);
  mlir::python::CollectDiagnosticsToStringScope diagnostics(
      mlirModuleGetContext(module));
  std::vector<int32_t> warp;
  std::vector<int32_t> cta;
  auto store = [](const int32_t *taskIds, intptr_t taskCount, void *output) {
    auto &tasks = *static_cast<std::vector<int32_t> *>(output);
    if (taskCount)
      tasks.assign(taskIds, taskIds + taskCount);
  };
  MlirLogicalResult result = swageMaterializeSegmentedPlan(
      module, offsets.data(), static_cast<intptr_t>(offsets.size()), valueCount,
      segmentCount, warpMaxElements, store, &warp, store, &cta);
  if (mlirLogicalResultIsFailure(result))
    throw nb::value_error(diagnostics.takeMessage().c_str());
  return {std::move(warp), std::move(cta)};
}

} // namespace

NB_MODULE(_swageDialectsNanobind, m) {
  auto swageM = m.def_submodule("swage");

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
                          std::move(target), false, false);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("block_size"),
      nb::arg("target"));
  swageM.def(
      "_compile_segmented_reduction_ptx",
      [](nb::object module, std::string kernelName, int64_t blockSize,
         std::string target, bool useTaskIds) {
        return compilePTX(module, std::move(kernelName), blockSize,
                          std::move(target), true, useTaskIds);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("block_size"),
      nb::arg("target"), nb::arg("use_task_ids") = false);
  swageM.def(
      "_compile_fused_segmented_reduction_ptx",
      [](nb::object module, std::string kernelName, std::string target) {
        return compilePTX(module, std::move(kernelName), 128, std::move(target),
                          true, true, true);
      },
      nb::arg("module"), nb::arg("kernel_name"), nb::arg("target"));
  swageM.def("_materialize_segmented_plan", &materializeSegmentedPlan,
             nb::arg("module"), nb::arg("offsets"), nb::arg("value_count"),
             nb::arg("segment_count"), nb::arg("warp_max_elements") = 32);
}
