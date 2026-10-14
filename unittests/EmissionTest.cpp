// unittests/EmissionTest.cpp
//===- EmissionTest.cpp - Kernel emission helper tests --------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/SwagePlanToGPU/Emission.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "gtest/gtest.h"

#include <cmath>
#include <set>
#include <string>
#include <tuple>

namespace mlir::swage {
namespace {

/// What one reduction kind lowers to: its identity, the operation that
/// combines an accumulator with an element, and the block-wide reduction.
struct KindLowering {
  double identity;
  Type identityType;
  std::string combine;
  gpu::AllReduceOperation blockReduction;
};

KindLowering loweringOf(MLIRContext &context, ReductionKind kind,
                        Type elementType) {
  Block block;
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(&block);
  Location loc = builder.getUnknownLoc();

  Value identity = identityFor(builder, loc, kind, elementType);
  APFloat value(0.0f);
  EXPECT_TRUE(matchPattern(identity, m_ConstantFloat(&value)));
  // The value is read in double: an f32 infinity or zero converts exactly.
  bool losesInformation = false;
  value.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                &losesInformation);
  EXPECT_FALSE(losesInformation);
  Value combined = combine(builder, loc, kind, identity, identity);
  return {value.convertToDouble(), identity.getType(),
          combined.getDefiningOp()->getName().getStringRef().str(),
          allReduceOperationFor(kind)};
}

// A kind that shared a branch with another kind would compute that kind's
// reduction under its own name, so each kind is held to its own row here and
// no two kinds may share one. The identity has the element type of the
// reduction: an f32 identity in an f64 reduction would not verify.
TEST(EmissionTest, EveryReductionKindLowersToItsOwnIdentityAndCombine) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();
  const double infinity = INFINITY;
  for (Type elementType :
       {Type(Float32Type::get(&context)), Type(Float64Type::get(&context))}) {
    std::set<std::tuple<double, std::string, gpu::AllReduceOperation>> seen;
    for (unsigned value = 0; value <= getMaxEnumValForReductionKind();
         ++value) {
      std::optional<ReductionKind> kind = symbolizeReductionKind(value);
      ASSERT_TRUE(kind.has_value());
      SCOPED_TRACE(stringifyReductionKind(*kind).str());
      KindLowering lowering = loweringOf(context, *kind, elementType);
      EXPECT_EQ(lowering.identityType, elementType);
      switch (*kind) {
      case ReductionKind::Sum:
        EXPECT_EQ(lowering.identity, 0.0);
        EXPECT_EQ(lowering.combine, "arith.addf");
        EXPECT_EQ(lowering.blockReduction, gpu::AllReduceOperation::ADD);
        break;
      case ReductionKind::Max:
        EXPECT_EQ(lowering.identity, -infinity);
        EXPECT_EQ(lowering.combine, "arith.maximumf");
        EXPECT_EQ(lowering.blockReduction, gpu::AllReduceOperation::MAXIMUMF);
        break;
      case ReductionKind::Min:
        EXPECT_EQ(lowering.identity, infinity);
        EXPECT_EQ(lowering.combine, "arith.minimumf");
        EXPECT_EQ(lowering.blockReduction, gpu::AllReduceOperation::MINIMUMF);
        break;
      }
      EXPECT_TRUE(seen.emplace(lowering.identity, lowering.combine,
                               lowering.blockReduction)
                      .second)
          << "shares its lowering with another kind";
    }
  }
}

} // namespace
} // namespace mlir::swage
