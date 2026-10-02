// lib/Conversion/SwageToPlan/SwageToPlan.cpp
//===- SwageToPlan.cpp - Segment functions to plan functions --------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/SwageToPlan/SwageToPlan.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "swage/Conversion/SwageToPlan/Admission.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "swage/Dialect/SwagePlan/IR/KernelLayout.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanDialect.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"
#include "swage/Target/TargetDescription.h"

namespace mlir::swage {
namespace {

using swage_plan::KernelArgument;
using swage_plan::KernelKind;
using swage_plan::KernelLayout;
using swage_plan::SwagePlanDialect;
using swage_plan::TaskPolicy;

/// The role a kernel argument has in the segment function, for the five
/// arguments the function declares. The arguments a schedule adds have none.
std::optional<ArgumentRole> roleOf(KernelArgument argument) {
  switch (argument) {
  case KernelArgument::Values:
    return ArgumentRole::Values;
  case KernelArgument::Offsets:
    return ArgumentRole::Offsets;
  case KernelArgument::Output:
    return ArgumentRole::Output;
  case KernelArgument::ValueCount:
    return ArgumentRole::ValueCount;
  case KernelArgument::SegmentCount:
    return ArgumentRole::SegmentCount;
  default:
    return std::nullopt;
  }
}

/// Where the segment function takes the argument of `role`.
unsigned sourceIndexOf(const SegmentABI &abi, ArgumentRole role) {
  switch (role) {
  case ArgumentRole::Values:
    return abi.values;
  case ArgumentRole::Offsets:
    return abi.offsets;
  case ArgumentRole::Output:
    return abi.output;
  case ArgumentRole::ValueCount:
    return abi.valueCount;
  case ArgumentRole::SegmentCount:
    return abi.segmentCount;
  }
  llvm_unreachable("unknown argument role");
}

/// Replace an admitted segment function by its plan function.
///
/// The plan function takes the parameters of the kernel in the order of its
/// layout, whatever order the segment function declared its arguments in.
/// Its task operation absorbs what no consumer pattern can lower on its own:
/// the segment id, the segment construction, and the scalar store. The
/// reductions move into the task region in program order, and a map store
/// follows them, which is the order the kernel runs them in.
void buildPlanFunction(func::FuncOp source, SegmentProgramAnalysis &analysis,
                       const PlanOptions &options,
                       const TargetDescription &target) {
  fuseAdmittedMaps(analysis);

  MLIRContext *context = source.getContext();
  Location loc = source.getLoc();
  FunctionType sourceType = source.getFunctionType();
  bool useTaskIds = options.schedule == PlanSchedule::TaskIds;
  const KernelLayout layout = swage_plan::kernelLayout(
      useTaskIds ? KernelKind::TaskIds : KernelKind::Direct);
  // A task buffer is a buffer of the word the offsets hold, and a task
  // count is a count like the two the function declares.
  Type wordBuffer = sourceType.getInput(analysis.abi.offsets);
  Type word = sourceType.getInput(analysis.abi.valueCount);
  SmallVector<Type> inputs;
  for (KernelArgument argument : layout.arguments()) {
    if (std::optional<ArgumentRole> role = roleOf(argument))
      inputs.push_back(sourceType.getInput(sourceIndexOf(analysis.abi, *role)));
    else
      inputs.push_back(swage_plan::isBuffer(argument) ? wordBuffer : word);
  }

  OpBuilder builder(context);
  builder.setInsertionPoint(source);
  auto plan = func::FuncOp::create(builder, loc, source.getName(),
                                   FunctionType::get(context, inputs, {}));
  plan->setAttr(
      SwagePlanDialect::getBlockThreadsAttrName(),
      builder.getI32IntegerAttr(static_cast<int32_t>(options.blockThreads)));
  for (auto [index, argument] : llvm::enumerate(layout.arguments()))
    if (std::optional<ArgumentRole> role = roleOf(argument))
      plan.setArgAttr(static_cast<unsigned>(index),
                      SwageDialect::getRoleAttrName(),
                      ArgumentRoleAttr::get(context, *role));
  Block *entry = plan.addEntryBlock();
  auto argument = [&](KernelArgument parameter) {
    return Value(entry->getArgument(layout.indexOf(parameter)));
  };

  // The task-id kernel reduces within one subgroup exactly when a block is
  // one subgroup. The direct kernel always reduces across the block.
  TaskPolicy policy = useTaskIds && options.blockThreads == target.subgroupWidth
                          ? TaskPolicy::Warp
                          : TaskPolicy::CTA;
  bool storesScalar = analysis.mapStores.empty();
  builder.setInsertionPointToEnd(entry);
  auto tasks = swage_plan::TasksOp::create(
      builder, loc, argument(KernelArgument::Values),
      argument(KernelArgument::Offsets), argument(KernelArgument::ValueCount),
      argument(KernelArgument::SegmentCount),
      useTaskIds ? argument(KernelArgument::TaskIds) : Value(),
      useTaskIds ? argument(KernelArgument::TaskCount) : Value(),
      storesScalar ? argument(KernelArgument::Output) : Value(), policy);
  func::ReturnOp::create(builder, loc);

  MakeSegmentOp segment = analysis.segments.front();
  Block *body = builder.createBlock(&tasks.getBody(), {},
                                    {segment.getResult().getType()}, {loc});
  for (ReduceOp reduction : analysis.reductions)
    reduction->moveBefore(body, body->end());
  for (MapStoreOp mapStore : analysis.mapStores) {
    mapStore->moveBefore(body, body->end());
    mapStore.getOutputMutable().assign(argument(KernelArgument::Output));
  }
  segment.getResult().replaceAllUsesWith(body->getArgument(0));
  swage_plan::YieldOp::create(
      builder, loc,
      storesScalar ? analysis.storedReduction.getResult() : Value());
  source.erase();
}

LogicalResult parseSchedule(StringRef text, PlanSchedule &schedule) {
  if (text == "direct")
    schedule = PlanSchedule::Direct;
  else if (text == "task-ids")
    schedule = PlanSchedule::TaskIds;
  else
    return failure();
  return success();
}

class SwageToPlanPass
    : public PassWrapper<SwageToPlanPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwageToPlanPass)

  SwageToPlanPass() = default;
  SwageToPlanPass(const SwageToPlanPass &other) : PassWrapper(other) {
    schedule = other.schedule.getValue();
    blockThreads = other.blockThreads.getValue();
    selectedFunction = other.selectedFunction.getValue();
  }

  StringRef getArgument() const final { return "swage-to-plan"; }
  StringRef getDescription() const final {
    return "Replace every segment function by the plan function of the "
           "kernel a schedule selects";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, SwagePlanDialect>();
  }

  void runOnOperation() final {
    PlanOptions options;
    if (failed(parseSchedule(schedule, options.schedule))) {
      getOperation().emitError() << "schedule must be direct or task-ids, got '"
                                 << schedule.getValue() << "'";
      return signalPassFailure();
    }
    options.blockThreads = blockThreads;
    options.function = selectedFunction;
    if (failed(planSegmentFunctions(getOperation(), options, nvidiaTarget())))
      signalPassFailure();
  }

private:
  Option<std::string> schedule{
      *this, "schedule",
      llvm::cl::desc("The kernel to plan: direct (one block per segment) or "
                     "task-ids (one block per task of a task buffer)"),
      llvm::cl::init("direct")};
  Option<int64_t> blockThreads{
      *this, "block-threads",
      llvm::cl::desc("Launch width of the kernel in threads"),
      llvm::cl::init(nvidiaTarget().ctaBlockThreads)};
  Option<std::string> selectedFunction{
      *this, "function",
      llvm::cl::desc("Plan only this function instead of every function that "
                     "holds Swage operations")};
};

} // namespace

LogicalResult planSegmentFunctions(ModuleOp module, const PlanOptions &options,
                                   const TargetDescription &target) {
  if (!target.admitsBlockThreads(options.blockThreads))
    return module.emitError()
           << "block-threads must be a launch width the target admits, from "
              "1 to "
           << target.maxBlockThreads
           << " threads with a power-of-two subgroup count, got "
           << options.blockThreads;
  FailureOr<SmallVector<func::FuncOp>> functions =
      findSegmentFunctions(module, options.function);
  if (failed(functions))
    return failure();
  // Every function is admitted before any is changed, so a rejected module
  // is left as it was.
  SmallVector<SegmentProgramAnalysis, 1> analyses(functions->size());
  for (auto [function, analysis] : llvm::zip(*functions, analyses)) {
    if (failed(analyzeSegmentProgram(function, analysis)))
      return failure();
    // A task buffer comes from host classification, which describes one
    // capture-free reduction whose result is stored per segment.
    if (options.schedule == PlanSchedule::TaskIds &&
        failed(verifyPlanningProgram(analysis)))
      return failure();
    if (failed(verifyKernelSymbols(module, function, "")))
      return failure();
  }
  for (auto [function, analysis] : llvm::zip(*functions, analyses))
    buildPlanFunction(function, analysis, options, target);
  return success();
}

LogicalResult admitTaskProgram(ModuleOp module, StringRef function) {
  if (function.empty())
    return module.emitError("the name of the function to admit is empty");
  FailureOr<SmallVector<func::FuncOp>> functions =
      findSegmentFunctions(module, function);
  if (failed(functions))
    return failure();
  SegmentProgramAnalysis analysis;
  if (failed(analyzeSegmentProgram(functions->front(), analysis)))
    return failure();
  return verifyPlanningProgram(analysis);
}

std::unique_ptr<Pass> createSwageToPlanPass() {
  return std::make_unique<SwageToPlanPass>();
}

void registerSwageToPlanPass() { PassRegistration<SwageToPlanPass>(); }

} // namespace mlir::swage
