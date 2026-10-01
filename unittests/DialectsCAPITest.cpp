// unittests/DialectsCAPITest.cpp
//===- DialectsCAPITest.cpp - Swage dialect C API tests -------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage-c/Dialects.h"

#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace {

MlirStringRef ref(const std::string &text) {
  return mlirStringRefCreate(text.data(), text.size());
}

std::string str(MlirStringRef text) {
  return std::string(text.data, text.length);
}

/// A context with a handler that collects diagnostics instead of printing.
class Session {
public:
  Session() : context(mlirContextCreate()) {
    mlirContextAttachDiagnosticHandler(context, collect, &diagnostics, nullptr);
  }
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  ~Session() { mlirContextDestroy(context); }

  void load(MlirDialectHandle handle) {
    mlirDialectHandleRegisterDialect(handle, context);
    mlirDialectHandleLoadDialect(handle, context);
  }

  MlirType parseType(const std::string &text) {
    return mlirTypeParseGet(context, ref(text));
  }

  /// Expects exactly one diagnostic, and that it contains `message`.
  void expectOnly(const std::string &message) const {
    ASSERT_EQ(diagnostics.size(), 1U);
    EXPECT_NE(diagnostics.front().find(message), std::string::npos)
        << diagnostics.front();
  }

  MlirContext context;
  std::vector<std::string> diagnostics;

private:
  static MlirLogicalResult collect(MlirDiagnostic diagnostic, void *userData) {
    std::string text;
    mlirDiagnosticPrint(
        diagnostic,
        +[](MlirStringRef part, void *data) {
          static_cast<std::string *>(data)->append(part.data, part.length);
        },
        &text);
    static_cast<std::vector<std::string> *>(userData)->push_back(text);
    return mlirLogicalResultSuccess();
  }
};

TEST(DialectsCAPITest, ThePlanHandleRegistersThePlanningDialect) {
  Session session;
  MlirDialectHandle handle = mlirGetDialectHandle__swage_plan__();
  EXPECT_EQ(str(mlirDialectHandleGetNamespace(handle)), "swage_plan");
  ASSERT_FALSE(mlirContextIsRegisteredOperation(session.context,
                                                ref("swage_plan.classify")));

  session.load(handle);

  EXPECT_TRUE(mlirContextIsRegisteredOperation(session.context,
                                               ref("swage_plan.classify")));
}

TEST(DialectsCAPITest, ThePlanHandleLetsACallerParsePlanningIR) {
  Session session;
  session.load(mlirGetDialectHandle__func__());
  session.load(mlirGetDialectHandle__swage_plan__());

  MlirModule module = mlirModuleCreateParse(session.context, ref(R"mlir(
module {
  func.func private @semantic_sum(
      memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32)

  func.func @classify(%offsets: memref<?xi32>, %value_count: i32,
                      %segment_count: i32) -> !swage_plan.task_range {
    %tasks = swage_plan.classify %offsets, %value_count, %segment_count {
        cta_chunk_elements = 4096 : i32, kernel = @semantic_sum,
        policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>],
        warp_max_elements = 32 : i32}
        : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}
)mlir"));

  ASSERT_FALSE(mlirModuleIsNull(module));
  EXPECT_TRUE(session.diagnostics.empty());
  mlirModuleDestroy(module);
}

TEST(DialectsCAPITest, BuildsTheSegmentTypeTheParserBuilds) {
  Session session;
  session.load(mlirGetDialectHandle__swage__());
  MlirType f32 = mlirF32TypeGet(session.context);

  MlirType segment = swageSegmentTypeGet(f32);

  ASSERT_FALSE(mlirTypeIsNull(segment));
  EXPECT_TRUE(mlirTypeEqual(segment, session.parseType("!swage.segment<f32>")));
  EXPECT_TRUE(mlirTypeEqual(swageSegmentTypeGetElementType(segment), f32));
  EXPECT_TRUE(session.diagnostics.empty());
}

TEST(DialectsCAPITest, TellsSegmentTypesFromOtherTypes) {
  Session session;
  session.load(mlirGetDialectHandle__swage__());
  MlirType i32 = mlirIntegerTypeGet(session.context, 32);
  MlirType segment = swageSegmentTypeGet(i32);

  EXPECT_TRUE(swageTypeIsASegment(segment));
  EXPECT_TRUE(swageTypeIsASegment(session.parseType("!swage.segment<f64>")));
  EXPECT_FALSE(swageTypeIsASegment(i32));
  EXPECT_FALSE(swageTypeIsASegment(session.parseType("memref<?xi32>")));
  EXPECT_TRUE(
      mlirTypeIDEqual(swageSegmentTypeGetTypeID(), mlirTypeGetTypeID(segment)));
  EXPECT_FALSE(
      mlirTypeIDEqual(swageSegmentTypeGetTypeID(), mlirTypeGetTypeID(i32)));
}

TEST(DialectsCAPITest, RefusesASegmentOfANonScalarElement) {
  Session session;
  session.load(mlirGetDialectHandle__swage__());

  MlirType segment = swageSegmentTypeGet(session.parseType("memref<2xf32>"));

  EXPECT_TRUE(mlirTypeIsNull(segment));
  session.expectOnly("segment element type must be an integer or float type, "
                     "got 'memref<2xf32>'");
}

TEST(DialectsCAPITest, RefusesASegmentBeforeTheDialectIsLoaded) {
  Session session;

  MlirType segment = swageSegmentTypeGet(mlirF32TypeGet(session.context));

  EXPECT_TRUE(mlirTypeIsNull(segment));
  session.expectOnly("the swage dialect is not loaded in this context");
}

} // namespace
