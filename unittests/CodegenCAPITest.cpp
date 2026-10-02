// unittests/CodegenCAPITest.cpp
//===- CodegenCAPITest.cpp - Swage code generation C API tests ------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// Exercises include/swage-c/Codegen.h the way a C embedder would: through the
// MLIR C API only, with a diagnostic handler attached to the context. Each
// test pins one sentence of the contract stated in that header.
//
//===----------------------------------------------------------------------===//

#include "swage-c/Codegen.h"
#include "swage-c/Dialects.h"

#include "mlir-c/Diagnostics.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/GPU.h"
#include "mlir-c/Dialect/Math.h"
#include "mlir-c/Dialect/MemRef.h"
#include "mlir-c/Dialect/Vector.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr const char *fixedVectorAdd = R"mlir(
module {
  func.func @add_kernel(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    %pid = swage.program_id 0
    %block = arith.constant 128 : index
    %base = arith.muli %pid, %block : index
    %lane = vector.step : vector<128xindex>
    %base_vector = vector.broadcast %base : index to vector<128xindex>
    %offsets = arith.addi %base_vector, %lane : vector<128xindex>
    %n_index = arith.index_cast %n : i32 to index
    %n_vector = vector.broadcast %n_index : index to vector<128xindex>
    %mask = arith.cmpi slt, %offsets, %n_vector : vector<128xindex>
    %zero = arith.constant 0.0 : f32
    %passthrough = vector.broadcast %zero : f32 to vector<128xf32>
    %c0 = arith.constant 0 : index
    %lhs = vector.gather %x[%c0] [%offsets], %mask, %passthrough
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
          into vector<128xf32>
    %rhs = vector.gather %y[%c0] [%offsets], %mask, %passthrough
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
          into vector<128xf32>
    %sum = arith.addf %lhs, %rhs : vector<128xf32>
    vector.scatter %output[%c0] [%offsets], %mask, %sum
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
    return
  }
}
)mlir";

constexpr const char *segmentedSum = R"mlir(
module {
  func.func @segmented_sum(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
)mlir";

/// The segmented kernel next to a function that holds no Swage operation.
constexpr const char *segmentedSumWithBystander = R"mlir(
module {
  func.func @bystander() {
    return
  }
  func.func @segmented_sum(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
)mlir";

/// Two segment functions in one module: `@segmented_sum` and a copy of it
/// named `@other_sum`.
std::string twoKernels() {
  std::string first = segmentedSum;
  std::string second = first.substr(first.find("  func.func"));
  first.erase(first.rfind('}'));
  std::string name = "@segmented_sum";
  second.replace(second.find(name), name.size(), "@other_sum");
  return first + second;
}

/// `program` with `text` inserted before its first function.
std::string beforeFirstFunction(std::string program, const std::string &text) {
  program.insert(program.find("func.func"), text);
  return program;
}

MlirStringRef ref(const std::string &text) {
  return mlirStringRefCreate(text.data(), text.size());
}

bool contains(const std::string &text, const std::string &part) {
  return text.find(part) != std::string::npos;
}

/// One line per diagnostic, for a readable failure message.
std::string joined(const std::vector<std::string> &diagnostics) {
  std::string text;
  for (const std::string &diagnostic : diagnostics)
    text += "\n  " + diagnostic;
  return text;
}

bool anyContains(const std::vector<std::string> &diagnostics,
                 const std::string &message) {
  for (const std::string &diagnostic : diagnostics)
    if (contains(diagnostic, message))
      return true;
  return false;
}

/// What one compile call reported back: its result, the two strings, and how
/// often each callback ran.
struct Compiled {
  bool succeeded = false;
  std::string lowered;
  std::string ptx;
  int loweredCalls = 0;
  int ptxCalls = 0;
};

struct StringSink {
  std::string *text;
  int *calls;
};

void storeString(MlirStringRef value, void *userData) {
  auto *sink = static_cast<StringSink *>(userData);
  sink->text->assign(value.data, value.length);
  ++*sink->calls;
}

/// One compile entry point with its fixed arguments bound, so every entry
/// point can run through the same checks.
using CompileFn = std::function<MlirLogicalResult(
    MlirModule, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback, void *, SwageStringCallback, void *)>;

struct EntryPoint {
  const char *name;
  const char *program;
  const char *kernelName;
  /// The `.entry` symbol of the PTX the entry point emits.
  const char *entry;
  CompileFn compile;
};

std::vector<EntryPoint> entryPoints() {
  return {
      {"fixed block", fixedVectorAdd, "add_kernel", "add_kernel",
       [](MlirModule module, MlirStringRef kernel, MlirStringRef target,
          SwageStringCallback lowered, void *loweredData,
          SwageStringCallback ptx, void *ptxData) {
         return swageCompileFixedBlockToPTX(module, kernel, 128, target,
                                            lowered, loweredData, ptx, ptxData);
       }},
      {"segmented", segmentedSum, "segmented_sum", "segmented_sum",
       [](MlirModule module, MlirStringRef kernel, MlirStringRef target,
          SwageStringCallback lowered, void *loweredData,
          SwageStringCallback ptx, void *ptxData) {
         return swageCompileSegmentedReductionToPTX(module, kernel, 128, target,
                                                    false, lowered, loweredData,
                                                    ptx, ptxData);
       }},
      {"segmented with task ids", segmentedSum, "segmented_sum",
       "segmented_sum",
       [](MlirModule module, MlirStringRef kernel, MlirStringRef target,
          SwageStringCallback lowered, void *loweredData,
          SwageStringCallback ptx, void *ptxData) {
         return swageCompileSegmentedReductionToPTX(module, kernel, 32, target,
                                                    true, lowered, loweredData,
                                                    ptx, ptxData);
       }},
      {"fused", segmentedSum, "segmented_sum", "segmented_sum",
       swageCompileFusedSegmentedReductionToPTX},
      {"persistent", segmentedSum, "segmented_sum", "segmented_sum",
       swageCompilePersistentSegmentedReductionToPTX},
      {"split partial", segmentedSum, "segmented_sum", "segmented_sum__partial",
       swageCompileSplitPartialReductionToPTX},
      {"split merge", segmentedSum, "segmented_sum", "segmented_sum__merge",
       swageCompileSplitMergeReductionToPTX},
  };
}

/// An MLIR context set up the way the header asks of a caller: the dialects
/// of the input registered and loaded, and a handler collecting diagnostics.
class Session {
public:
  Session() : context(mlirContextCreate()) {
    for (MlirDialectHandle handle :
         {mlirGetDialectHandle__swage__(), mlirGetDialectHandle__func__(),
          mlirGetDialectHandle__arith__(), mlirGetDialectHandle__math__(),
          mlirGetDialectHandle__memref__(), mlirGetDialectHandle__vector__()}) {
      mlirDialectHandleRegisterDialect(handle, context);
      mlirDialectHandleLoadDialect(handle, context);
    }
    mlirContextAttachDiagnosticHandler(context, collect, &diagnostics, nullptr);
  }
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  ~Session() {
    for (MlirModule module : modules)
      mlirModuleDestroy(module);
    mlirContextDestroy(context);
  }

  MlirModule parse(const std::string &text) {
    MlirModule module = mlirModuleCreateParse(context, ref(text));
    if (!mlirModuleIsNull(module))
      modules.push_back(module);
    return module;
  }

  Compiled compile(const CompileFn &function, MlirModule module,
                   const std::string &kernelName, const std::string &target) {
    Compiled compiled;
    StringSink lowered{&compiled.lowered, &compiled.loweredCalls};
    StringSink ptx{&compiled.ptx, &compiled.ptxCalls};
    compiled.succeeded = mlirLogicalResultIsSuccess(
        function(module, ref(kernelName), ref(target), storeString, &lowered,
                 storeString, &ptx));
    return compiled;
  }

  Compiled compile(const EntryPoint &entryPoint,
                   const std::string &target = "sm_86") {
    return compile(entryPoint.compile, parse(entryPoint.program),
                   entryPoint.kernelName, target);
  }

  std::string print(MlirModule module) {
    std::string text;
    mlirOperationPrint(
        mlirModuleGetOperation(module),
        +[](MlirStringRef part, void *userData) {
          static_cast<std::string *>(userData)->append(part.data, part.length);
        },
        &text);
    return text;
  }

  bool isRegistered(const std::string &operationName) {
    return mlirContextIsRegisteredOperation(context, ref(operationName));
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

  std::vector<MlirModule> modules;
};

/// A failed call leaves a diagnostic that names the cause and runs neither
/// callback.
void expectRejected(const Compiled &compiled, const Session &session,
                    const std::string &message) {
  EXPECT_FALSE(compiled.succeeded);
  EXPECT_TRUE(anyContains(session.diagnostics, message))
      << "expected '" << message << "' among:" << joined(session.diagnostics);
  EXPECT_EQ(compiled.loweredCalls, 0);
  EXPECT_EQ(compiled.ptxCalls, 0);
}

TEST(CodegenCAPITest, EveryEntryPointCompilesItsKernel) {
  for (const EntryPoint &entryPoint : entryPoints()) {
    SCOPED_TRACE(entryPoint.name);
    Session session;

    Compiled compiled = session.compile(entryPoint);

    ASSERT_TRUE(compiled.succeeded);
    EXPECT_TRUE(session.diagnostics.empty()) << joined(session.diagnostics);
    EXPECT_EQ(compiled.loweredCalls, 1);
    EXPECT_EQ(compiled.ptxCalls, 1);
    EXPECT_TRUE(contains(compiled.lowered, "gpu.module"));
    EXPECT_FALSE(contains(compiled.lowered, "swage."));
    EXPECT_TRUE(contains(compiled.ptx, ".target sm_86"));
    EXPECT_TRUE(contains(compiled.ptx,
                         std::string(".entry ") + entryPoint.entry + "("));
  }
}

TEST(CodegenCAPITest, ACallLeavesItsModuleUnchanged) {
  for (const EntryPoint &entryPoint : entryPoints()) {
    SCOPED_TRACE(entryPoint.name);
    Session session;
    MlirModule module = session.parse(entryPoint.program);
    const std::string before = session.print(module);

    Compiled compiled = session.compile(entryPoint.compile, module,
                                        entryPoint.kernelName, "sm_86");

    ASSERT_TRUE(compiled.succeeded);
    EXPECT_EQ(session.print(module), before);
    EXPECT_TRUE(mlirOperationVerify(mlirModuleGetOperation(module)));
  }
}

TEST(CodegenCAPITest, ACompileLoadsTheLoweringDialectsIntoTheCallersContext) {
  Session session;
  const char *const lowered[] = {"gpu.module", "scf.if", "llvm.func",
                                 "nvvm.barrier0"};
  for (const char *operation : lowered)
    ASSERT_FALSE(session.isRegistered(operation)) << operation;

  ASSERT_TRUE(session.compile(entryPoints().front()).succeeded);

  for (const char *operation : lowered)
    EXPECT_TRUE(session.isRegistered(operation)) << operation;
}

TEST(CodegenCAPITest, ASegmentedCompileLoadsThePlanningDialect) {
  Session session;
  ASSERT_FALSE(session.isRegistered("swage_plan.tasks"));

  // The direct kernel is planned and then converted, so its compile builds
  // plan operations in the context of the module.
  ASSERT_TRUE(session.compile(entryPoints()[1]).succeeded);

  EXPECT_TRUE(session.isRegistered("swage_plan.tasks"));
}

TEST(CodegenCAPITest, RepeatedCompilesInOneContextGiveTheSamePTX) {
  for (const EntryPoint &entryPoint : entryPoints()) {
    SCOPED_TRACE(entryPoint.name);
    Session session;
    MlirModule module = session.parse(entryPoint.program);

    Compiled first = session.compile(entryPoint.compile, module,
                                     entryPoint.kernelName, "sm_86");
    Compiled second = session.compile(entryPoint.compile, module,
                                      entryPoint.kernelName, "sm_86");

    ASSERT_TRUE(first.succeeded);
    ASSERT_TRUE(second.succeeded);
    EXPECT_EQ(first.lowered, second.lowered);
    EXPECT_EQ(first.ptx, second.ptx);
  }
}

TEST(CodegenCAPITest, ANullCallbackIsReportedOnTheModule) {
  for (const EntryPoint &entryPoint : entryPoints()) {
    SCOPED_TRACE(entryPoint.name);
    for (bool dropLowered : {true, false}) {
      Session session;
      Compiled compiled;
      StringSink lowered{&compiled.lowered, &compiled.loweredCalls};
      StringSink ptx{&compiled.ptx, &compiled.ptxCalls};

      compiled.succeeded = mlirLogicalResultIsSuccess(entryPoint.compile(
          session.parse(entryPoint.program), ref(entryPoint.kernelName),
          ref("sm_86"), dropLowered ? nullptr : storeString, &lowered,
          dropLowered ? storeString : nullptr, &ptx));

      expectRejected(compiled, session,
                     dropLowered ? "loweredCallback must not be null"
                                 : "ptxCallback must not be null");
    }
  }
}

TEST(CodegenCAPITest, ANullModuleFailsWithoutACrash) {
  for (const EntryPoint &entryPoint : entryPoints()) {
    SCOPED_TRACE(entryPoint.name);
    Session session;

    Compiled compiled = session.compile(entryPoint.compile, MlirModule{nullptr},
                                        entryPoint.kernelName, "sm_86");

    // There is no context to report on, so this is the one silent failure.
    EXPECT_FALSE(compiled.succeeded);
    EXPECT_EQ(compiled.loweredCalls + compiled.ptxCalls, 0);
    EXPECT_TRUE(session.diagnostics.empty()) << joined(session.diagnostics);
  }
}

TEST(CodegenCAPITest, RejectsTargetsOutsideTheAdmittedProcessors) {
  // Targets the syntax refuses, then processors the pinned LLVM lacks.
  const char *const malformed[] = {"",       "sm_8",       "sm_79",
                                   "sm_130", "compute_86", "sm_8x"};
  const char *const unpinned[] = {"sm_85", "sm_99", "sm_129"};
  std::vector<std::pair<std::string, std::string>> cases;
  for (const char *target : malformed)
    cases.emplace_back(target,
                       std::string("target must match sm_<major><minor> and be "
                                   "sm_80 or newer, got '") +
                           target + "'");
  for (const char *target : unpinned)
    cases.emplace_back(target,
                       std::string("target ") + target +
                           " is not a processor supported by the pinned LLVM");

  for (const EntryPoint &entryPoint : entryPoints()) {
    for (const auto &[target, message] : cases) {
      SCOPED_TRACE(std::string(entryPoint.name) + " for '" + target + "'");
      Session session;

      Compiled compiled = session.compile(entryPoint, target);

      expectRejected(compiled, session, message);
    }
  }
}

TEST(CodegenCAPITest, RejectsBlockSizesNoDeviceLaunches) {
  struct Case {
    int64_t blockSize;
    const char *message;
  };
  const Case cases[] = {
      {0, "block_size must be a positive integer, got 0"},
      {-128, "block_size must be a positive integer, got -128"},
      {1025, "block_size must be at most 1024, got 1025"},
  };
  for (const Case &testCase : cases) {
    SCOPED_TRACE(testCase.blockSize);
    for (bool segmented : {false, true}) {
      Session session;
      Compiled compiled;
      StringSink lowered{&compiled.lowered, &compiled.loweredCalls};
      StringSink ptx{&compiled.ptx, &compiled.ptxCalls};

      MlirLogicalResult result =
          segmented ? swageCompileSegmentedReductionToPTX(
                          session.parse(segmentedSum), ref("segmented_sum"),
                          testCase.blockSize, ref("sm_86"), false, storeString,
                          &lowered, storeString, &ptx)
                    : swageCompileFixedBlockToPTX(
                          session.parse(fixedVectorAdd), ref("add_kernel"),
                          testCase.blockSize, ref("sm_86"), storeString,
                          &lowered, storeString, &ptx);
      compiled.succeeded = mlirLogicalResultIsSuccess(result);

      expectRejected(compiled, session, testCase.message);
    }
  }
}

TEST(CodegenCAPITest, RejectsAKernelNameTheModuleDoesNotDefine) {
  for (const EntryPoint &entryPoint : entryPoints()) {
    SCOPED_TRACE(entryPoint.name);
    Session session;

    Compiled compiled =
        session.compile(entryPoint.compile, session.parse(entryPoint.program),
                        "absent", "sm_86");

    expectRejected(
        compiled, session,
        "kernel_name 'absent' does not name a function of the module");
  }
}

TEST(CodegenCAPITest, RejectsAKernelNameThatIsNotASegmentFunction) {
  Session session;

  Compiled compiled = session.compile(swageCompileFusedSegmentedReductionToPTX,
                                      session.parse(segmentedSumWithBystander),
                                      "bystander", "sm_86");

  expectRejected(
      compiled, session,
      "function names @bystander, which holds no Swage segment operation");
}

TEST(CodegenCAPITest, CompilesTheNamedKernelOfAModuleWithSeveral) {
  for (const EntryPoint &entryPoint : entryPoints()) {
    if (entryPoint.program != segmentedSum)
      continue;
    SCOPED_TRACE(entryPoint.name);
    Session session;
    MlirModule module = session.parse(twoKernels());
    ASSERT_FALSE(mlirModuleIsNull(module));
    std::string entry = entryPoint.entry;
    std::string otherEntry = entry;
    otherEntry.replace(0, std::strlen("segmented_sum"), "other_sum");

    Compiled alone = session.compile(entryPoint);
    Compiled first =
        session.compile(entryPoint.compile, module, "segmented_sum", "sm_86");
    Compiled second =
        session.compile(entryPoint.compile, module, "other_sum", "sm_86");

    ASSERT_TRUE(first.succeeded) << joined(session.diagnostics);
    ASSERT_TRUE(second.succeeded) << joined(session.diagnostics);
    // The kernel does not depend on what else the module holds.
    EXPECT_EQ(first.ptx, alone.ptx);
    EXPECT_FALSE(contains(first.ptx, "other_sum"));
    EXPECT_TRUE(contains(second.ptx, ".entry " + otherEntry + "("));
    EXPECT_FALSE(contains(second.ptx, "segmented_sum"));
    // The function that was not named stays in the lowered module as it was.
    EXPECT_TRUE(contains(first.lowered, "func.func @other_sum("));
    EXPECT_TRUE(contains(first.lowered, "swage.reduce"));
  }
}

TEST(CodegenCAPITest, RejectsFunctionNamesPTXCannotPrint) {
  Session session;
  std::string program = fixedVectorAdd;
  program.replace(program.find("@add_kernel"), std::strlen("@add_kernel"),
                  "@\"add.kernel\"");

  Compiled compiled =
      session.compile(entryPoints().front().compile, session.parse(program),
                      "add.kernel", "sm_86");

  expectRejected(compiled, session,
                 "function name 'add.kernel' is not a valid PTX identifier");
}

TEST(CodegenCAPITest, RejectsAModuleThatFailsVerification) {
  Session session;
  MlirModule module = session.parse(fixedVectorAdd);
  MlirOperation function =
      mlirBlockGetFirstOperation(mlirModuleGetBody(module));
  MlirOperation terminator = mlirBlockGetTerminator(
      mlirRegionGetFirstBlock(mlirOperationGetRegion(function, 0)));
  mlirOperationRemoveFromParent(terminator);
  mlirOperationDestroy(terminator);

  Compiled compiled = session.compile(entryPoints().front().compile, module,
                                      "add_kernel", "sm_86");

  expectRejected(compiled, session, "block with no terminator");
}

TEST(CodegenCAPITest, ReportsTheDiagnosticOfAFailedLowering) {
  Session session;
  Compiled compiled;
  StringSink lowered{&compiled.lowered, &compiled.loweredCalls};
  StringSink ptx{&compiled.ptx, &compiled.ptxCalls};

  compiled.succeeded = mlirLogicalResultIsSuccess(swageCompileFixedBlockToPTX(
      session.parse(fixedVectorAdd), ref("add_kernel"), 64, ref("sm_86"),
      storeString, &lowered, storeString, &ptx));

  expectRejected(compiled, session,
                 "vector width 128 does not match requested block size 64");
}

TEST(CodegenCAPITest, RejectsANamedMemorySpaceWithoutAborting) {
  Session session;
  std::string program = fixedVectorAdd;
  const std::string buffer = "%y: memref<?xf32>";
  program.replace(program.find(buffer), buffer.size(),
                  "%y: memref<?xf32, \"device\">");
  const std::string gather = "%rhs = vector.gather";
  const std::string gatherType = ": memref<?xf32>";
  program.replace(program.find(gatherType, program.find(gather)),
                  gatherType.size(), ": memref<?xf32, \"device\">");

  Compiled compiled =
      session.compile(entryPoints().front().compile, session.parse(program),
                      "add_kernel", "sm_86");

  expectRejected(compiled, session,
                 "only default-memory-space pointers are supported, got "
                 "'memref<?xf32, \"device\">'");
}

TEST(CodegenCAPITest, RejectsAModuleThatDefinesTheSymbolOfTheKernelModule) {
  for (const EntryPoint &entryPoint : entryPoints()) {
    if (entryPoint.program != segmentedSum)
      continue;
    SCOPED_TRACE(entryPoint.name);
    Session session;
    MlirDialectHandle gpu = mlirGetDialectHandle__gpu__();
    mlirDialectHandleRegisterDialect(gpu, session.context);
    mlirDialectHandleLoadDialect(gpu, session.context);
    std::string symbol = std::string(entryPoint.entry) + "_module";

    Compiled compiled = session.compile(
        entryPoint.compile,
        session.parse(beforeFirstFunction(
            segmentedSum, "gpu.module @" + symbol + " {\n  }\n  ")),
        "segmented_sum", "sm_86");

    expectRejected(compiled, session,
                   "lowering @segmented_sum creates @" + symbol +
                       ", which the module already defines");
  }
}

TEST(CodegenCAPITest, RejectsAKernelThatTheModuleRefersTo) {
  Session session;
  std::string program = segmentedSum;
  program.insert(program.rfind('}'), R"mlir(
  func.func @caller(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    call @segmented_sum(
        %values, %offsets, %output, %value_count, %segment_count)
        : (memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32) -> ()
    return
  }
)mlir");

  Compiled compiled =
      session.compile(swageCompileFusedSegmentedReductionToPTX,
                      session.parse(program), "segmented_sum", "sm_86");

  expectRejected(compiled, session,
                 "segment function @segmented_sum is referenced 1 times; "
                 "lowering it to a GPU kernel removes it, so it must have no "
                 "symbol use");
}

TEST(CodegenCAPITest, SelectsTheKernelModuleBesideAnotherGPUModule) {
  for (const EntryPoint &entryPoint : entryPoints()) {
    SCOPED_TRACE(entryPoint.name);
    Session session;
    MlirDialectHandle gpu = mlirGetDialectHandle__gpu__();
    mlirDialectHandleRegisterDialect(gpu, session.context);
    mlirDialectHandleLoadDialect(gpu, session.context);

    Compiled compiled = session.compile(
        entryPoint.compile,
        session.parse(beforeFirstFunction(entryPoint.program,
                                          "gpu.module @earlier {\n  }\n  ")),
        entryPoint.kernelName, "sm_86");

    ASSERT_TRUE(compiled.succeeded) << joined(session.diagnostics);
    EXPECT_TRUE(
        contains(compiled.ptx, std::string(".entry ") + entryPoint.entry));
  }
}

/// What one plan call reported back, one flat record list per callback.
struct Plan {
  bool succeeded = false;
  std::vector<int32_t> warp;
  std::vector<int32_t> cta;
  std::vector<int32_t> partial;
  std::vector<int32_t> merge;
  int calls = 0;
};

struct RecordSink {
  std::vector<int32_t> *records;
  int *calls;
};

void storeRecords(const int32_t *records, intptr_t count, void *userData) {
  auto *sink = static_cast<RecordSink *>(userData);
  sink->records->assign(records, records + count);
  ++*sink->calls;
}

/// Callbacks to pass for the warp, CTA, partial, and merge records. A test
/// clears one to model a caller that forgot it.
struct PlanCallbacks {
  SwageTaskIdsCallback warp = storeRecords;
  SwageTaskIdsCallback cta = storeRecords;
  SwageTaskIdsCallback partial = storeRecords;
  SwageTaskIdsCallback merge = storeRecords;
};

Plan materialize(Session &session, const int64_t *offsets, intptr_t offsetCount,
                 int64_t valueCount, int64_t segmentCount,
                 PlanCallbacks callbacks = {},
                 const std::string &program = segmentedSum,
                 const std::string &kernelName = "segmented_sum") {
  Plan plan;
  RecordSink warp{&plan.warp, &plan.calls};
  RecordSink cta{&plan.cta, &plan.calls};
  RecordSink partial{&plan.partial, &plan.calls};
  RecordSink merge{&plan.merge, &plan.calls};
  plan.succeeded = mlirLogicalResultIsSuccess(swageMaterializeSegmentedPlan(
      session.parse(program), ref(kernelName), offsets, offsetCount, valueCount,
      segmentCount, 32, 4096, callbacks.warp, &warp, callbacks.cta, &cta,
      callbacks.partial, &partial, callbacks.merge, &merge));
  return plan;
}

void expectRejected(const Plan &plan, const Session &session,
                    const std::string &message) {
  EXPECT_FALSE(plan.succeeded);
  EXPECT_TRUE(anyContains(session.diagnostics, message))
      << "expected '" << message << "' among:" << joined(session.diagnostics);
  EXPECT_EQ(plan.calls, 0);
}

TEST(CodegenCAPITest, MaterializesOneRecordListPerPolicy) {
  Session session;
  // A warp segment, a CTA segment, and one segment of three chunks.
  const int64_t offsets[] = {0, 32, 132, 8325};

  Plan plan = materialize(session, offsets, 4, 8325, 3);

  ASSERT_TRUE(plan.succeeded);
  EXPECT_TRUE(session.diagnostics.empty()) << joined(session.diagnostics);
  EXPECT_EQ(plan.calls, 4);
  EXPECT_EQ(plan.warp, (std::vector<int32_t>{0}));
  EXPECT_EQ(plan.cta, (std::vector<int32_t>{1}));
  EXPECT_EQ(plan.partial,
            (std::vector<int32_t>{132, 4228, 4228, 8324, 8324, 8325}));
  EXPECT_EQ(plan.merge, (std::vector<int32_t>{2, 0, 3}));
}

TEST(CodegenCAPITest, APlanCallLeavesItsContextUnchanged) {
  Session session;
  const int64_t offsets[] = {0};
  ASSERT_FALSE(session.isRegistered("swage_plan.tasks"));

  // A plan call admits the program and classifies the offsets. It builds no
  // IR, so it loads no dialect.
  ASSERT_TRUE(materialize(session, offsets, 1, 0, 0).succeeded);

  EXPECT_FALSE(session.isRegistered("swage_plan.tasks"));
}

TEST(CodegenCAPITest, APlanCallReportsRejectedArguments) {
  const int64_t offsets[] = {0, 4};
  {
    Session session;
    expectRejected(materialize(session, offsets, -1, 4, 1), session,
                   "offsetCount must not be negative, got -1");
  }
  {
    Session session;
    expectRejected(materialize(session, nullptr, 2, 4, 1), session,
                   "offsets must not be null when offsetCount is 2");
  }
  for (int dropped = 0; dropped < 4; ++dropped) {
    SCOPED_TRACE(dropped);
    Session session;
    PlanCallbacks callbacks;
    SwageTaskIdsCallback *slots[] = {&callbacks.warp, &callbacks.cta,
                                     &callbacks.partial, &callbacks.merge};
    *slots[dropped] = nullptr;
    expectRejected(materialize(session, offsets, 2, 4, 1, callbacks), session,
                   "warpCallback, ctaCallback, partialCallback, and "
                   "mergeCallback must not be null");
  }
}

TEST(CodegenCAPITest, APlanCallAdmitsTheNamedKernelOfAModuleWithSeveral) {
  const int64_t offsets[] = {0, 32, 132};
  for (const char *kernel : {"segmented_sum", "other_sum"}) {
    SCOPED_TRACE(kernel);
    Session session;

    Plan plan =
        materialize(session, offsets, 3, 132, 2, {}, twoKernels(), kernel);

    ASSERT_TRUE(plan.succeeded) << joined(session.diagnostics);
    EXPECT_EQ(plan.warp, (std::vector<int32_t>{0}));
    EXPECT_EQ(plan.cta, (std::vector<int32_t>{1}));
  }
}

TEST(CodegenCAPITest, APlanCallRejectsAKernelNameThatIsNotASegmentFunction) {
  const int64_t offsets[] = {0, 4};
  {
    Session session;
    expectRejected(materialize(session, offsets, 2, 4, 1, {},
                               segmentedSumWithBystander, "bystander"),
                   session,
                   "function names @bystander, which holds no Swage segment "
                   "operation");
  }
  {
    Session session;
    expectRejected(materialize(session, offsets, 2, 4, 1, {},
                               segmentedSumWithBystander, "absent"),
                   session,
                   "kernel_name 'absent' does not name a function of the "
                   "module");
  }
}

TEST(CodegenCAPITest, APlanCallOnANullModuleFailsWithoutACrash) {
  const int64_t offsets[] = {0, 4};
  Plan plan;
  RecordSink sink{&plan.warp, &plan.calls};

  plan.succeeded = mlirLogicalResultIsSuccess(swageMaterializeSegmentedPlan(
      MlirModule{nullptr}, ref("segmented_sum"), offsets, 2, 4, 1, 32, 4096,
      storeRecords, &sink, storeRecords, &sink, storeRecords, &sink,
      storeRecords, &sink));

  EXPECT_FALSE(plan.succeeded);
  EXPECT_EQ(plan.calls, 0);
}

TEST(CodegenCAPITest, APlanCallReportsInvalidMetadata) {
  Session session;
  const int64_t offsets[] = {0, 4, 2};

  expectRejected(materialize(session, offsets, 3, 4, 2), session,
                 "offsets must be nondecreasing");
}

/// What one swageClassifySegments call left with its two callbacks.
struct Classification {
  bool succeeded = false;
  std::vector<int32_t> records;
  std::vector<intptr_t> counts;
  int recordCalls = 0;
  std::vector<std::string> errors;
};

void storeClassification(const int32_t *records, intptr_t warpCount,
                         intptr_t ctaCount, intptr_t partialCount,
                         intptr_t mergeCount, void *userData) {
  auto *classification = static_cast<Classification *>(userData);
  classification->records.assign(records, records + warpCount + ctaCount +
                                              3 * partialCount +
                                              3 * mergeCount);
  classification->counts = {warpCount, ctaCount, partialCount, mergeCount};
  ++classification->recordCalls;
}

void storeClassificationError(MlirStringRef message, void *userData) {
  static_cast<Classification *>(userData)->errors.emplace_back(message.data,
                                                               message.length);
}

Classification
classify(const int32_t *offsets, intptr_t offsetCount, int64_t valueCount,
         int64_t segmentCount, int64_t warpMaxElements = 32,
         int64_t ctaChunkElements = 4096,
         SwageTaskRecordsCallback recordsCallback = storeClassification,
         SwageStringCallback errorCallback = storeClassificationError) {
  Classification classification;
  classification.succeeded = mlirLogicalResultIsSuccess(
      swageClassifySegments(offsets, offsetCount, valueCount, segmentCount,
                            warpMaxElements, ctaChunkElements, recordsCallback,
                            &classification, errorCallback, &classification));
  return classification;
}

void expectRejected(const Classification &classification,
                    const std::string &message) {
  EXPECT_FALSE(classification.succeeded);
  EXPECT_EQ(classification.recordCalls, 0);
  EXPECT_EQ(classification.errors, (std::vector<std::string>{message}));
}

TEST(CodegenCAPITest, EstimatesTheWorkOfTheElementPrograms) {
  Session session;
  // The identity reduction costs nothing.
  EXPECT_EQ(swageEstimateElementWork(session.parse(segmentedSum)), 0);

  // One unit per multiply, eight per exp2, sixteen per division, over the
  // map and the reduction together; the constant and the yields are free.
  std::string program = segmentedSum;
  std::string reduction = "    %sum = swage.reduce %segment kind<sum>";
  program.replace(program.find(reduction), reduction.size(),
                  R"mlir(    %scaled = swage.map %segment
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%element: f32):
      %half = arith.constant 0.5 : f32
      %product = arith.mulf %element, %half : f32
      %exponential = math.exp2 %product : f32
      swage.yield %exponential : f32
    }
    %sum = swage.reduce %scaled kind<sum>)mlir");
  std::string identity = "      swage.yield %value : f32";
  program.replace(program.rfind(identity), identity.size(),
                  "      %quotient = arith.divf %value, %value : f32\n"
                  "      swage.yield %quotient : f32");
  MlirModule weighted = session.parse(program);
  ASSERT_FALSE(mlirModuleIsNull(weighted)) << joined(session.diagnostics);
  EXPECT_EQ(swageEstimateElementWork(weighted), 1 + 8 + 16);

  // An operation without a weight gives no estimate.
  std::string unweighted = segmentedSum;
  unweighted.replace(unweighted.rfind(identity), identity.size(),
                     "      %root = math.sqrt %value : f32\n"
                     "      swage.yield %root : f32");
  EXPECT_EQ(swageEstimateElementWork(session.parse(unweighted)), -1);
  EXPECT_EQ(swageEstimateElementWork(MlirModule{nullptr}), -1);
  EXPECT_TRUE(session.diagnostics.empty()) << joined(session.diagnostics);
}

TEST(CodegenCAPITest, AClassifyCallGivesTheRecordsOfThePlanWithoutAModule) {
  // A warp segment, a CTA segment, and one segment of three chunks. No
  // context exists when the classification runs.
  const int32_t offsets[] = {0, 32, 132, 8325};

  Classification classification = classify(offsets, 4, 8325, 3);

  ASSERT_TRUE(classification.succeeded);
  EXPECT_EQ(classification.recordCalls, 1);
  EXPECT_TRUE(classification.errors.empty());
  EXPECT_EQ(classification.counts, (std::vector<intptr_t>{1, 1, 3, 1}));

  Session session;
  const int64_t wideOffsets[] = {0, 32, 132, 8325};
  Plan plan = materialize(session, wideOffsets, 4, 8325, 3);
  ASSERT_TRUE(plan.succeeded);
  std::vector<int32_t> expected = plan.warp;
  expected.insert(expected.end(), plan.cta.begin(), plan.cta.end());
  expected.insert(expected.end(), plan.partial.begin(), plan.partial.end());
  expected.insert(expected.end(), plan.merge.begin(), plan.merge.end());
  // The three partial tasks belong to the one merge.
  expected.insert(expected.end(), {0, 0, 0});
  EXPECT_EQ(classification.records, expected);
  EXPECT_EQ(classification.records,
            (std::vector<int32_t>{0, 1, 132, 4228, 4228, 8324, 8324, 8325, 2, 0,
                                  3, 0, 0, 0}));
}

TEST(CodegenCAPITest, AClassifyCallOfNoSegmentsGivesNoRecords) {
  const int32_t offsets[] = {0};

  Classification classification = classify(offsets, 1, 0, 0);

  ASSERT_TRUE(classification.succeeded);
  EXPECT_EQ(classification.recordCalls, 1);
  EXPECT_TRUE(classification.records.empty());
  EXPECT_EQ(classification.counts, (std::vector<intptr_t>{0, 0, 0, 0}));
}

TEST(CodegenCAPITest, AClassifyCallReportsRejectedArguments) {
  const int32_t offsets[] = {0, 4};

  expectRejected(classify(offsets, -1, 4, 1),
                 "offsetCount must not be negative, got -1");
  expectRejected(classify(nullptr, 2, 4, 1),
                 "offsets must not be null when offsetCount is 2");
  expectRejected(classify(offsets, 2, 4, 1, 32, 4096, nullptr),
                 "recordsCallback must not be null");
}

TEST(CodegenCAPITest, AClassifyCallReportsInvalidMetadataToItsErrorCallback) {
  const int32_t decreasing[] = {0, 4, 2};
  const int32_t offsets[] = {0, 4};

  expectRejected(classify(decreasing, 3, 4, 2),
                 "offsets must be nondecreasing");
  expectRejected(classify(offsets, 2, 3, 1),
                 "final offset must not exceed value count");
  expectRejected(classify(offsets, 2, 4, 2),
                 "offset count must equal segment count plus one");
  expectRejected(classify(offsets, 2, 4, 1, 33, 32),
                 "warp max elements must not exceed CTA chunk elements");
}

TEST(CodegenCAPITest, AFailedClassifyCallNeedsNoErrorCallback) {
  const int32_t decreasing[] = {0, 4, 2};

  Classification classification =
      classify(decreasing, 3, 4, 2, 32, 4096, storeClassification, nullptr);

  EXPECT_FALSE(classification.succeeded);
  EXPECT_EQ(classification.recordCalls, 0);
  EXPECT_TRUE(classification.errors.empty());
}

TEST(CodegenCAPITest, ClassifyCallsMayRunAtTheSameTime) {
  // The call takes no context and keeps no state, so threads share nothing
  // but the offsets they read.
  constexpr int threadCount = 4;
  std::vector<int32_t> offsets = {0};
  for (int32_t segment = 0; segment < 20000; ++segment)
    offsets.push_back(offsets.back() + (segment * 37) % 6000);
  const intptr_t offsetCount = static_cast<intptr_t>(offsets.size());
  const Classification expected =
      classify(offsets.data(), offsetCount, offsets.back(), offsetCount - 1);
  ASSERT_TRUE(expected.succeeded);

  std::vector<Classification> results(threadCount);
  std::vector<std::thread> threads;
  for (int index = 0; index < threadCount; ++index)
    threads.emplace_back([&, index] {
      for (int repeat = 0; repeat < 20; ++repeat)
        results[index] = classify(offsets.data(), offsetCount, offsets.back(),
                                  offsetCount - 1);
    });
  for (std::thread &thread : threads)
    thread.join();

  for (const Classification &result : results) {
    ASSERT_TRUE(result.succeeded);
    EXPECT_EQ(result.counts, expected.counts);
    EXPECT_EQ(result.records, expected.records);
  }
}

TEST(CodegenCAPITest, CompilesOnSeparateContextsMayRunAtTheSameTime) {
  // The header allows concurrent calls only on modules of different
  // contexts. Each thread owns its context for its whole life, and every
  // result must equal the one a single thread produces.
  constexpr int threadCount = 4;
  constexpr int rounds = 8;
  std::vector<std::string> expected;
  for (const EntryPoint &entryPoint : entryPoints()) {
    Session session;
    Compiled compiled = session.compile(entryPoint);
    ASSERT_TRUE(compiled.succeeded);
    expected.push_back(compiled.ptx);
  }

  std::vector<int> mismatches(threadCount, 0);
  std::vector<std::thread> threads;
  for (int thread = 0; thread < threadCount; ++thread) {
    threads.emplace_back([&, thread] {
      const std::vector<EntryPoint> kinds = entryPoints();
      for (int round = 0; round < rounds; ++round) {
        Session session;
        for (size_t kind = 0; kind < kinds.size(); ++kind) {
          Compiled compiled = session.compile(kinds[kind]);
          if (!compiled.succeeded || compiled.ptx != expected[kind] ||
              !session.diagnostics.empty())
            ++mismatches[thread];
        }
      }
    });
  }
  for (std::thread &thread : threads)
    thread.join();

  EXPECT_EQ(mismatches, std::vector<int>(threadCount, 0));
}

} // namespace
