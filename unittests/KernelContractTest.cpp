// unittests/KernelContractTest.cpp
#include "swage/Support/KernelContract.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <string>

namespace mlir::swage {
namespace {

KernelContract fixedContract() {
  KernelContract contract;
  contract.backend = KernelBackend::CUDA;
  contract.entry = "add_kernel";
  contract.launch = {KernelLaunchModel::SPMDGrid,
                     std::array<int32_t, 3>{128, 1, 1}};
  contract.arguments = {
      KernelArgument::userPointer(0, KernelArgumentAccess::Read),
      KernelArgument::userPointer(1, KernelArgumentAccess::Read),
      KernelArgument::userPointer(2, KernelArgumentAccess::Write),
      KernelArgument::userScalar(KernelArgumentKind::I32, 3)};
  return contract;
}

KernelContract hostContract() {
  KernelContract contract = fixedContract();
  contract.backend = KernelBackend::CPU;
  contract.launch = {KernelLaunchModel::HostCall, std::nullopt};
  return contract;
}

void expectParseError(llvm::StringRef json, llvm::StringRef expected) {
  llvm::Expected<KernelContract> parsed = parseKernelContractJSON(json);
  if (parsed)
    FAIL() << "expected contract parsing to fail";
  std::string message = llvm::toString(parsed.takeError());
  EXPECT_NE(message.find(expected.str()), std::string::npos) << message;
}

void expectValidationError(KernelContract contract, llvm::StringRef expected) {
  llvm::Error error = validateKernelContract(contract);
  if (!error)
    FAIL() << "expected contract validation to fail";
  std::string message = llvm::toString(std::move(error));
  EXPECT_NE(message.find(expected.str()), std::string::npos) << message;
}

TEST(KernelContractTest, SerializesCanonicalCUDAJSONAndRoundTrips) {
  KernelContract contract = fixedContract();
  constexpr llvm::StringLiteral expected =
      R"({"version":2,"backend":"cuda","entry":"add_kernel","launch":{"model":"spmd-grid","block":[128,1,1]},"arguments":[{"kind":"ptr","origin":"user","source_index":0,"access":"read"},{"kind":"ptr","origin":"user","source_index":1,"access":"read"},{"kind":"ptr","origin":"user","source_index":2,"access":"write"},{"kind":"i32","origin":"user","source_index":3}]})";

  std::string canonical = serializeKernelContractJSON(contract);

  EXPECT_EQ(canonical, expected);
  llvm::Expected<KernelContract> parsed = parseKernelContractJSON(canonical);
  if (!parsed)
    FAIL() << llvm::toString(parsed.takeError());
  EXPECT_EQ(*parsed, contract);
}

TEST(KernelContractTest, SerializesCanonicalHostJSONAndRoundTrips) {
  KernelContract contract = hostContract();
  constexpr llvm::StringLiteral expected =
      R"({"version":2,"backend":"cpu","entry":"add_kernel","launch":{"model":"host-call"},"arguments":[{"kind":"ptr","origin":"user","source_index":0,"access":"read"},{"kind":"ptr","origin":"user","source_index":1,"access":"read"},{"kind":"ptr","origin":"user","source_index":2,"access":"write"},{"kind":"i32","origin":"user","source_index":3}]})";

  std::string canonical = serializeKernelContractJSON(contract);

  EXPECT_EQ(canonical, expected);
  llvm::Expected<KernelContract> parsed = parseKernelContractJSON(canonical);
  if (!parsed)
    FAIL() << llvm::toString(parsed.takeError());
  EXPECT_EQ(*parsed, contract);
}

TEST(KernelContractTest, NormalizesReorderedAndWhitespaceJSON) {
  constexpr llvm::StringLiteral input =
      R"( { "arguments" : [], "launch" : { "model" : "host-call" }, "entry" : "k", "backend" : "cpu", "version" : 2 } )";
  llvm::Expected<KernelContract> parsed = parseKernelContractJSON(input);
  if (!parsed)
    FAIL() << llvm::toString(parsed.takeError());

  EXPECT_EQ(
      serializeKernelContractJSON(*parsed),
      R"({"version":2,"backend":"cpu","entry":"k","launch":{"model":"host-call"},"arguments":[]})");
}

TEST(KernelContractTest, DigestIsStableOverCanonicalSerialization) {
  KernelContract contract = fixedContract();

  EXPECT_EQ(digestKernelContract(contract),
            "d2d9ec3d43f1b96444de1237935e25c6e4a21b0829c96b7ee8e032dc3afbfc18");
  llvm::Expected<KernelContract> parsed =
      parseKernelContractJSON(serializeKernelContractJSON(contract));
  if (!parsed)
    FAIL() << llvm::toString(parsed.takeError());
  EXPECT_EQ(digestKernelContract(*parsed), digestKernelContract(contract));
}

TEST(KernelContractTest, DictionaryAttributeRoundTripsBothLaunchModels) {
  MLIRContext context;
  Builder builder(&context);
  for (const KernelContract &contract : {fixedContract(), hostContract()}) {
    DictionaryAttr attribute = buildKernelContractAttr(builder, contract);
    llvm::Expected<KernelContract> parsed = parseKernelContractAttr(attribute);

    if (!parsed)
      FAIL() << llvm::toString(parsed.takeError());
    EXPECT_EQ(*parsed, contract);
  }
}

TEST(KernelContractTest, RejectsMalformedDictionaryAttribute) {
  MLIRContext context;
  Builder builder(&context);
  DictionaryAttr malformed = builder.getDictionaryAttr(
      {builder.getNamedAttr("version", builder.getI64IntegerAttr(2)),
       builder.getNamedAttr("backend", builder.getStringAttr("cpu")),
       builder.getNamedAttr("entry", builder.getStringAttr("k")),
       builder.getNamedAttr("launch",
                            builder.getDictionaryAttr({builder.getNamedAttr(
                                "model", builder.getStringAttr("host-call"))})),
       builder.getNamedAttr("arguments", builder.getStringAttr("not-array"))});

  llvm::Expected<KernelContract> parsed = parseKernelContractAttr(malformed);

  if (parsed)
    FAIL() << "expected contract attribute parsing to fail";
  EXPECT_NE(llvm::toString(parsed.takeError()).find("array arguments"),
            std::string::npos);
}

TEST(KernelContractTest, RejectsUnknownMissingAndInapplicableJSONFields) {
  expectParseError(
      R"({"version":2,"backend":"cpu","entry":"k","launch":{"model":"host-call"},"arguments":[],"extra":0})",
      "unknown field 'extra'");
  expectParseError(
      R"({"version":2,"backend":"cpu","entry":"k","arguments":[]})",
      "requires version, backend, entry, launch, and arguments");
  expectParseError(
      R"({"version":2,"backend":"cuda","entry":"k","launch":{"model":"spmd-grid","block":[1,1,1]},"arguments":[{"kind":"i32","origin":"user","source_index":0,"access":"read"}]})",
      "scalar argument 0 must not have access");
  expectParseError(
      R"({"version":2,"backend":"cuda","entry":"k","launch":{"model":"spmd-grid","block":[1,1,1]},"arguments":[{"kind":"ptr","origin":"plan","key":"p"}]})",
      "pointer argument 0 requires a string access");
  expectParseError(
      R"({"version":2,"backend":"cpu","entry":"k","launch":{"model":"host-call","block":[1,1,1]},"arguments":[]})",
      "host-call launch must not have block");
}

TEST(KernelContractTest, RejectsUnknownVersionBackendKindOriginAndAccess) {
  expectParseError(
      R"({"version":1,"backend":"cpu","entry":"k","launch":{"model":"host-call"},"arguments":[]})",
      "unsupported kernel contract version 1");
  expectParseError(
      R"({"version":2,"backend":"metal","entry":"k","launch":{"model":"host-call"},"arguments":[]})",
      "unknown kernel backend 'metal'");
  expectParseError(
      R"({"version":2,"backend":"cuda","entry":"k","launch":{"model":"spmd-grid","block":[1,1,1]},"arguments":[{"kind":"u64","origin":"user","source_index":0}]})",
      "unknown kernel argument kind 'u64'");
  expectParseError(
      R"({"version":2,"backend":"cuda","entry":"k","launch":{"model":"spmd-grid","block":[1,1,1]},"arguments":[{"kind":"i32","origin":"runtime","key":"n"}]})",
      "unknown kernel argument origin 'runtime'");
  expectParseError(
      R"({"version":2,"backend":"cuda","entry":"k","launch":{"model":"spmd-grid","block":[1,1,1]},"arguments":[{"kind":"ptr","origin":"user","source_index":0,"access":"execute"}]})",
      "unknown kernel argument access 'execute'");
}

TEST(KernelContractTest, RejectsInvalidAndDuplicateBindings) {
  expectParseError(
      R"({"version":2,"backend":"cuda","entry":"k","launch":{"model":"spmd-grid","block":[1,1,1]},"arguments":[{"kind":"i32","origin":"user","source_index":-1}]})",
      "nonnegative u32 source_index");
  KernelContract duplicateSource = fixedContract();
  duplicateSource.arguments[1].sourceIndex = 0;
  expectValidationError(std::move(duplicateSource),
                        "duplicate user source_index 0");

  KernelContract duplicateKey = hostContract();
  duplicateKey.entry = "k";
  duplicateKey.arguments = {
      KernelArgument::keyedScalar(KernelArgumentKind::I32,
                                  KernelArgumentOrigin::Derived, "count"),
      KernelArgument::keyedScalar(KernelArgumentKind::I32,
                                  KernelArgumentOrigin::Plan, "count")};
  expectValidationError(std::move(duplicateKey),
                        "duplicate non-user binding key 'count'");
}

TEST(KernelContractTest, AllowsSparseSemanticSourceIndexes) {
  KernelContract contract = fixedContract();
  contract.entry = "sum__merge";
  contract.launch.block = std::array<int32_t, 3>{512, 1, 1};
  contract.arguments = {
      KernelArgument::keyedPointer(KernelArgumentOrigin::Scratch, "partials",
                                   KernelArgumentAccess::Read),
      KernelArgument::userPointer(2, KernelArgumentAccess::Write),
      KernelArgument::keyedPointer(KernelArgumentOrigin::Plan, "merge_ranges",
                                   KernelArgumentAccess::Read),
      KernelArgument::keyedScalar(KernelArgumentKind::I32,
                                  KernelArgumentOrigin::Derived,
                                  "partial_task_count"),
      KernelArgument::keyedScalar(KernelArgumentKind::I32,
                                  KernelArgumentOrigin::Derived,
                                  "merge_task_count")};

  if (llvm::Error error = validateKernelContract(contract))
    FAIL() << llvm::toString(std::move(error));
}

TEST(KernelContractTest, MapsEveryPhysicalArgumentType) {
  MLIRContext context;
  context.getOrLoadDialect<LLVM::LLVMDialect>();
  std::vector<KernelArgument> arguments;
  for (KernelArgumentKind kind :
       {KernelArgumentKind::Pointer, KernelArgumentKind::I1,
        KernelArgumentKind::I8, KernelArgumentKind::I16,
        KernelArgumentKind::I32, KernelArgumentKind::I64,
        KernelArgumentKind::F16, KernelArgumentKind::BF16,
        KernelArgumentKind::F32, KernelArgumentKind::F64}) {
    if (kind == KernelArgumentKind::Pointer)
      arguments.push_back(
          KernelArgument::userPointer(0, KernelArgumentAccess::Read));
    else
      arguments.push_back(KernelArgument::userScalar(kind, arguments.size()));
  }

  SmallVector<Type> types = getKernelArgumentTypes(&context, arguments);

  ASSERT_EQ(types.size(), 10u);
  EXPECT_TRUE(isa<LLVM::LLVMPointerType>(types[0]));
  EXPECT_TRUE(types[1].isSignlessInteger(1));
  EXPECT_TRUE(types[2].isSignlessInteger(8));
  EXPECT_TRUE(types[3].isSignlessInteger(16));
  EXPECT_TRUE(types[4].isSignlessInteger(32));
  EXPECT_TRUE(types[5].isSignlessInteger(64));
  EXPECT_TRUE(types[6].isF16());
  EXPECT_TRUE(types[7].isBF16());
  EXPECT_TRUE(types[8].isF32());
  EXPECT_TRUE(types[9].isF64());
}

TEST(KernelContractTest, ValidatesFunctionTypeBackendAndBlock) {
  MLIRContext context;
  context.getOrLoadDialect<LLVM::LLVMDialect>();
  KernelContract contract = fixedContract();
  FunctionType matching = FunctionType::get(
      &context, getKernelArgumentTypes(&context, contract.arguments), {});

  if (llvm::Error error = validateCUDAKernelContract(contract, "add_kernel",
                                                     matching, {128, 1, 1}))
    FAIL() << llvm::toString(std::move(error));

  FunctionType wrongType = FunctionType::get(
      &context,
      {LLVM::LLVMPointerType::get(&context),
       LLVM::LLVMPointerType::get(&context),
       LLVM::LLVMPointerType::get(&context), IntegerType::get(&context, 64)},
      {});
  llvm::Error typeError = validateCUDAKernelContract(contract, "add_kernel",
                                                     wrongType, {128, 1, 1});
  if (!typeError)
    FAIL() << "expected function type validation to fail";
  EXPECT_NE(llvm::toString(std::move(typeError))
                .find("argument 3 kind does not match function type"),
            std::string::npos);

  llvm::Error blockError =
      validateCUDAKernelContract(contract, "add_kernel", matching, {64, 1, 1});
  if (!blockError)
    FAIL() << "expected block validation to fail";
  EXPECT_NE(llvm::toString(std::move(blockError))
                .find("block does not match nvvm.reqntid"),
            std::string::npos);

  KernelContract host = hostContract();
  if (llvm::Error error =
          validateHostKernelContract(host, "add_kernel", matching))
    FAIL() << llvm::toString(std::move(error));
  llvm::Error backendError =
      validateHostKernelContract(contract, "add_kernel", matching);
  if (!backendError)
    FAIL() << "expected backend validation to fail";
  EXPECT_NE(llvm::toString(std::move(backendError)).find("host-call"),
            std::string::npos);
}

} // namespace
} // namespace mlir::swage
