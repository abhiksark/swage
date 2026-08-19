//===- SwageExtensionNanobind.cpp - swage dialect python module -----------===//
//
// Exposes registration of the dialects the semantic level composes with.
//
//===----------------------------------------------------------------------===//

#include "swage-c/Dialects.h"

#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/Math.h"
#include "mlir-c/Dialect/MemRef.h"
#include "mlir-c/Dialect/Vector.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

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
}
