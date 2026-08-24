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

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

enum class PTXKind { Fixed, Segmented, Fused, SplitPartial, SplitMerge };

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
