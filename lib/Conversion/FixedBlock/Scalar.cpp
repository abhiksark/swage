//===- Scalar.cpp - Fixed vector-add scalar arithmetic -----------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "Analysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"

#include <cassert>

using namespace mlir;

namespace mlir::swage::detail {
namespace {

// FP8 storage is always i8. These scalar conversions need neither FP8 LLVM
// types nor newer GPU instructions, and are identical on the host and GPU.
Value decodeFP8(OpBuilder &builder, Location loc, Value storage,
                unsigned mantissaBits, unsigned bias) {
  auto integer = [&](uint32_t value) -> Value {
    return arith::ConstantIntOp::create(builder, loc, value, 32);
  };
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  Value bits = arith::ExtUIOp::create(builder, loc, i32, storage);
  Value magnitude = arith::AndIOp::create(builder, loc, bits, integer(0x7f));
  Value sign = arith::AndIOp::create(builder, loc, bits, integer(0x80));
  sign = arith::ShLIOp::create(builder, loc, sign, integer(24));

  Value normalBits = arith::ShLIOp::create(builder, loc, magnitude,
                                           integer(23 - mantissaBits));
  normalBits = arith::AddIOp::create(builder, loc, normalBits,
                                     integer((127 - bias) << 23));

  // FP8 subnormals are exact multiples of 2^(1-bias-mantissaBits), all
  // representable as normal f32 values. Converting zero here yields +0;
  // installing the original sign below recovers -0 as well.
  Value mantissa = arith::AndIOp::create(builder, loc, magnitude,
                                         integer((1u << mantissaBits) - 1));
  Value subnormal = arith::UIToFPOp::create(builder, loc, f32, mantissa);
  Value quantum = arith::BitcastOp::create(
      builder, loc, f32, integer((128 - bias - mantissaBits) << 23));
  subnormal = arith::MulFOp::create(builder, loc, subnormal, quantum);
  Value subnormalBits = arith::BitcastOp::create(builder, loc, i32, subnormal);
  Value isSubnormal =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult, magnitude,
                            integer(1u << mantissaBits));
  Value decoded = arith::SelectOp::create(builder, loc, isSubnormal,
                                          subnormalBits, normalBits);

  if (mantissaBits == 3) {
    // E4M3FN has no infinity: exponent 15 is finite except for magnitude 127.
    Value isNaN = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                        magnitude, integer(0x7f));
    decoded = arith::SelectOp::create(builder, loc, isNaN, integer(0x7fc00000),
                                      decoded);
  } else {
    // E5M2 exponent 31 encodes infinity (mantissa zero) or a NaN.
    Value specialBits = arith::ShLIOp::create(builder, loc, mantissa,
                                              integer(23 - mantissaBits));
    specialBits =
        arith::OrIOp::create(builder, loc, specialBits, integer(0x7f800000));
    Value isSpecial = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::uge, magnitude, integer(0x7c));
    decoded =
        arith::SelectOp::create(builder, loc, isSpecial, specialBits, decoded);
  }
  decoded = arith::OrIOp::create(builder, loc, decoded, sign);
  return arith::BitcastOp::create(builder, loc, f32, decoded);
}

Value encodeFP8(OpBuilder &builder, Location loc, Value value,
                unsigned mantissaBits, unsigned bias) {
  auto integer = [&](uint32_t number) -> Value {
    return arith::ConstantIntOp::create(builder, loc, number, 32);
  };
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  Value bits = arith::BitcastOp::create(builder, loc, i32, value);
  Value sign = arith::ShRUIOp::create(builder, loc, bits, integer(24));
  sign = arith::AndIOp::create(builder, loc, sign, integer(0x80));
  Value magnitude =
      arith::AndIOp::create(builder, loc, bits, integer(0x7fffffff));

  // Normal rounding retains the low bit of the destination significand.
  // Adding half an ulp minus one plus that bit rounds ties to even, including
  // carries into the next exponent and the finite/overflow boundary.
  unsigned shift = 23 - mantissaBits;
  Value retained =
      arith::ShRUIOp::create(builder, loc, magnitude, integer(shift));
  Value odd = arith::AndIOp::create(builder, loc, retained, integer(1));
  Value rounded = arith::AddIOp::create(builder, loc, magnitude,
                                        integer((1u << (shift - 1)) - 1));
  rounded = arith::AddIOp::create(builder, loc, rounded, odd);
  rounded = arith::ShRUIOp::create(builder, loc, rounded, integer(shift));
  rounded = arith::SubIOp::create(builder, loc, rounded,
                                  integer((127 - bias) << mantissaBits));

  // At this power of two, an f32 ulp is exactly one FP8 subnormal quantum.
  // The ordinary (non-fastmath) f32 add performs RNE, and subtracting its
  // bit pattern recovers the rounded mantissa, including zero/min-normal.
  Value magicBits = integer((127 - bias + shift + 1) << 23);
  Value magic = arith::BitcastOp::create(builder, loc, f32, magicBits);
  Value absolute = arith::BitcastOp::create(builder, loc, f32, magnitude);
  Value subnormal = arith::AddFOp::create(builder, loc, absolute, magic);
  Value subnormalBits = arith::BitcastOp::create(builder, loc, i32, subnormal);
  Value subnormalMagnitude =
      arith::SubIOp::create(builder, loc, subnormalBits, magicBits);
  Value isSubnormal =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult, magnitude,
                            integer((128 - bias) << 23));
  Value encoded = arith::SelectOp::create(builder, loc, isSubnormal,
                                          subnormalMagnitude, rounded);

  // Do not saturate. E4M3FN overflow (including input infinity/NaN) becomes
  // NaN; E5M2 overflow becomes infinity, with an explicit NaN override.
  Value overflow = integer(mantissaBits == 3 ? 0x7f : 0x7c);
  Value isOverflow = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::uge, encoded, overflow);
  encoded =
      arith::SelectOp::create(builder, loc, isOverflow, overflow, encoded);
  if (mantissaBits == 2) {
    Value isNaN = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt,
                                        magnitude, integer(0x7f800000));
    encoded =
        arith::SelectOp::create(builder, loc, isNaN, integer(0x7e), encoded);
  }
  encoded = arith::OrIOp::create(builder, loc, encoded, sign);
  return arith::TruncIOp::create(builder, loc, builder.getI8Type(), encoded);
}

} // namespace

void buildFixedScalarAdd(OpBuilder &builder, Location loc, Type elementType,
                         Value xBase, Value yBase, Value outputBase,
                         Value offset) {
  bool isFP8 = isa<Float8E4M3FNType, Float8E5M2Type>(elementType);
  assert((isFP8 || elementType.isF16() || elementType.isF32()) &&
         "fixed vector-add element type was not admitted");
  Type storageType = isFP8 ? builder.getI8Type() : elementType;
  Type pointer = xBase.getType();
  Value xAddress =
      LLVM::GEPOp::create(builder, loc, pointer, storageType, xBase, offset);
  Value yAddress =
      LLVM::GEPOp::create(builder, loc, pointer, storageType, yBase, offset);
  Value outputAddress = LLVM::GEPOp::create(builder, loc, pointer, storageType,
                                            outputBase, offset);
  Value x = LLVM::LoadOp::create(builder, loc, storageType, xAddress);
  Value y = LLVM::LoadOp::create(builder, loc, storageType, yAddress);
  unsigned mantissaBits = isa<Float8E4M3FNType>(elementType) ? 3 : 2;
  unsigned bias = mantissaBits == 3 ? 7 : 15;
  if (isFP8) {
    x = decodeFP8(builder, loc, x, mantissaBits, bias);
    y = decodeFP8(builder, loc, y, mantissaBits, bias);
  } else if (elementType.isF16()) {
    x = arith::ExtFOp::create(builder, loc, builder.getF32Type(), x);
    y = arith::ExtFOp::create(builder, loc, builder.getF32Type(), y);
  }
  Value sum = arith::AddFOp::create(builder, loc, x, y);
  if (isFP8)
    sum = encodeFP8(builder, loc, sum, mantissaBits, bias);
  else if (elementType.isF16())
    sum = arith::TruncFOp::create(builder, loc, elementType, sum);
  LLVM::StoreOp::create(builder, loc, sum, outputAddress);
}

} // namespace mlir::swage::detail
