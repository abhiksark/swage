//===- KernelContract.h - Physical kernel ABI contract ---------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_SUPPORT_KERNELCONTRACT_H
#define SWAGE_SUPPORT_KERNELCONTRACT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::swage {

inline constexpr llvm::StringLiteral kernelContractAttrName =
    "swage.kernel_contract";
inline constexpr int64_t kernelContractVersion = 2;

enum class KernelBackend { CUDA, CPU };
enum class KernelLaunchModel { SPMDGrid, HostCall };
enum class KernelArgumentKind {
  Pointer,
  I1,
  I8,
  I16,
  I32,
  I64,
  F16,
  BF16,
  F32,
  F64
};
enum class KernelArgumentOrigin { User, Derived, Plan, Scratch };
enum class KernelArgumentAccess { Read, Write, ReadWrite };

struct KernelArgument {
  KernelArgumentKind kind;
  KernelArgumentOrigin origin;
  std::optional<uint32_t> sourceIndex;
  std::string key;
  std::optional<KernelArgumentAccess> access;

  static KernelArgument userPointer(uint32_t sourceIndex,
                                    KernelArgumentAccess access);
  static KernelArgument userScalar(KernelArgumentKind kind,
                                   uint32_t sourceIndex);
  static KernelArgument keyedPointer(KernelArgumentOrigin origin,
                                     llvm::StringRef key,
                                     KernelArgumentAccess access);
  static KernelArgument keyedScalar(KernelArgumentKind kind,
                                    KernelArgumentOrigin origin,
                                    llvm::StringRef key);

  bool operator==(const KernelArgument &other) const;
};

struct KernelLaunch {
  KernelLaunchModel model = KernelLaunchModel::SPMDGrid;
  std::optional<std::array<int32_t, 3>> block = std::array<int32_t, 3>{1, 1, 1};

  bool operator==(const KernelLaunch &other) const;
};

struct KernelContract {
  int64_t version = kernelContractVersion;
  KernelBackend backend = KernelBackend::CUDA;
  std::string entry;
  KernelLaunch launch;
  std::vector<KernelArgument> arguments;

  bool operator==(const KernelContract &other) const;
};

/// Validate all version-two schema and binding invariants.
llvm::Error validateKernelContract(const KernelContract &contract);

/// Parse strict version-two JSON. Unknown or context-inapplicable fields fail.
llvm::Expected<KernelContract> parseKernelContractJSON(llvm::StringRef json);

/// Return compact canonical JSON with stable field ordering.
std::string serializeKernelContractJSON(const KernelContract &contract);

/// Return the lowercase SHA-256 of the canonical serialization.
std::string digestKernelContract(const KernelContract &contract);

/// Convert between the owning contract and its temporary MLIR dictionary form.
DictionaryAttr buildKernelContractAttr(Builder &builder,
                                       const KernelContract &contract);
llvm::Expected<KernelContract> parseKernelContractAttr(Attribute attribute);

/// Derive a concrete function input list from the same ordered specification
/// used to build the contract.
SmallVector<Type>
getKernelArgumentTypes(MLIRContext *context,
                       llvm::ArrayRef<KernelArgument> arguments);

/// Validate backend-neutral contract and concrete function ABI invariants.
llvm::Error
validateKernelContractAgainstFunction(const KernelContract &contract,
                                      llvm::StringRef entry, FunctionType type);

/// Validate a CUDA contract against the compiled entry ABI and reqntid.
llvm::Error validateCUDAKernelContract(const KernelContract &contract,
                                       llvm::StringRef entry, FunctionType type,
                                       llvm::ArrayRef<int32_t> requiredBlock);

/// Validate a host-call contract against the compiled entry ABI.
llvm::Error validateHostKernelContract(const KernelContract &contract,
                                       llvm::StringRef entry,
                                       FunctionType type);

} // namespace mlir::swage

#endif // SWAGE_SUPPORT_KERNELCONTRACT_H
