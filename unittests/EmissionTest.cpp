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
  std::string combine;
  gpu::AllReduceOperation blockReduction;
};

KindLowering loweringOf(ReductionKind kind) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();
  Block block;
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(&block);
  Location loc = builder.getUnknownLoc();

  Value identity = identityFor(builder, loc, kind);
  APFloat value(0.0f);
  EXPECT_TRUE(matchPattern(identity, m_ConstantFloat(&value)));
  Value combined = combine(builder, loc, kind, identity, identity);
  return {value.convertToDouble(),
          combined.getDefiningOp()->getName().getStringRef().str(),
          allReduceOperationFor(kind)};
}

// A kind that shared a branch with another kind would compute that kind's
// reduction under its own name, so each kind is held to its own row here and
// no two kinds may share one.
TEST(EmissionTest, EveryReductionKindLowersToItsOwnIdentityAndCombine) {
  const double infinity = INFINITY;
  std::set<std::tuple<double, std::string, gpu::AllReduceOperation>> seen;
  for (unsigned value = 0; value <= getMaxEnumValForReductionKind(); ++value) {
    std::optional<ReductionKind> kind = symbolizeReductionKind(value);
    ASSERT_TRUE(kind.has_value());
    SCOPED_TRACE(stringifyReductionKind(*kind).str());
    KindLowering lowering = loweringOf(*kind);
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

} // namespace
} // namespace mlir::swage
