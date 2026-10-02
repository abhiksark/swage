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

/// What the planner needs to know about one schedule that plans a kernel.
struct KernelSchedule {
  /// The name of the schedule in the `schedule` option.
  StringLiteral name;
  /// The kernel, which fixes the parameter list of the plan function.
  KernelKind kind;
  /// What the kernel name adds to the name of the segment function.
  StringLiteral suffix;
  /// Whether host classification feeds the kernel, which describes one
  /// capture-free reduction whose result is stored per segment.
  bool needsTaskProgram;
  /// Whether `block-threads` gives the launch width. The target fixes the
  /// width of every other kernel.
  bool takesBlockThreads;
};

/// The schedules that plan a kernel. The sequential schedule plans none and
/// is handled apart.
std::optional<KernelSchedule> kernelSchedule(PlanSchedule schedule) {
  switch (schedule) {
  case PlanSchedule::Direct:
    return KernelSchedule{"direct", KernelKind::Direct, "", false, true};
  case PlanSchedule::TaskIds:
    return KernelSchedule{"task-ids", KernelKind::TaskIds, "", true, true};
  case PlanSchedule::SplitPartial:
    return KernelSchedule{"split-partial", KernelKind::SplitPartial,
                          "__partial", true, false};
  case PlanSchedule::Sequential:
    return std::nullopt;
  }
  llvm_unreachable("unknown plan schedule");
}

/// The launch width of the kernel `schedule` plans.
int64_t blockThreadsOf(PlanSchedule schedule, const PlanOptions &options,
                       const TargetDescription &target) {
  if (schedule == PlanSchedule::SplitPartial)
    return target.splitBlockThreads;
  return options.blockThreads;
}

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

/// The type of one parameter of a plan function. An argument the segment
/// function declares keeps its type. Scratch holds elements, like the
/// output; every other buffer a schedule adds holds words, like the
/// offsets; and every count it adds is a count like the two the function
/// declares.
Type parameterType(KernelArgument argument, FunctionType source,
                   const SegmentABI &abi) {
  if (std::optional<ArgumentRole> role = roleOf(argument))
    return source.getInput(sourceIndexOf(abi, *role));
  if (argument == KernelArgument::Scratch)
    return source.getInput(abi.output);
  if (swage_plan::isBuffer(argument))
    return source.getInput(abi.offsets);
  return source.getInput(abi.valueCount);
}

/// Move the consumers of an admitted program into the region of its task
/// operation and end the region. The reductions come first, in program
/// order, and a map store follows them, which is the order every lowering
/// runs them in.
void fillTaskRegion(Operation *task, SegmentProgramAnalysis &analysis,
                    Value output) {
  Location loc = task->getLoc();
  MakeSegmentOp segment = analysis.segments.front();
  OpBuilder builder(task->getContext());
  Block *body = builder.createBlock(&task->getRegion(0), {},
                                    {segment.getResult().getType()}, {loc});
  for (ReduceOp reduction : analysis.reductions)
    reduction->moveBefore(body, body->end());
  for (MapStoreOp mapStore : analysis.mapStores) {
    mapStore->moveBefore(body, body->end());
    mapStore.getOutputMutable().assign(output);
  }
  segment.getResult().replaceAllUsesWith(body->getArgument(0));
  swage_plan::YieldOp::create(builder, loc,
                              analysis.mapStores.empty()
                                  ? analysis.storedReduction.getResult()
                                  : Value());
}

/// Plan an admitted segment function for the CPU oracle, in place. The
/// function keeps its signature, its roles, and its callers. Its segment
/// id, segment construction, and scalar store become one sequential task
/// operation ahead of the return.
void buildSequentialPlan(func::FuncOp function,
                         SegmentProgramAnalysis &analysis) {
  fuseAdmittedMaps(analysis);

  const SegmentABI &abi = analysis.abi;
  Value output = function.getArgument(abi.output);
  OpBuilder builder(function.getContext());
  builder.setInsertionPoint(analysis.returns.front());
  auto tasks = swage_plan::TasksOp::create(
      builder, function.getLoc(), function.getArgument(abi.values),
      function.getArgument(abi.offsets), function.getArgument(abi.valueCount),
      function.getArgument(abi.segmentCount), Value(), Value(),
      analysis.mapStores.empty() ? output : Value(), TaskPolicy::Sequential);
  fillTaskRegion(tasks, analysis, output);
  for (memref::StoreOp store : analysis.stores)
    store.erase();
  analysis.segments.front().erase();
  analysis.segmentIds.front().erase();
}

/// Replace an admitted segment function by the plan function of one kernel.
///
/// The plan function takes the parameters of the kernel in the order of its
/// layout, whatever order the segment function declared its arguments in.
/// Its task operation absorbs what no consumer pattern can lower on its own:
/// the segment id, the segment construction, and the scalar store.
void buildKernelPlan(func::FuncOp source, SegmentProgramAnalysis &analysis,
                     PlanSchedule schedule, const PlanOptions &options,
                     const TargetDescription &target) {
  fuseAdmittedMaps(analysis);

  const KernelSchedule kernel = *kernelSchedule(schedule);
  MLIRContext *context = source.getContext();
  Location loc = source.getLoc();
  const KernelLayout layout = swage_plan::kernelLayout(kernel.kind);
  SmallVector<Type> inputs;
  for (KernelArgument argument : layout.arguments())
    inputs.push_back(
        parameterType(argument, source.getFunctionType(), analysis.abi));

  OpBuilder builder(context);
  builder.setInsertionPoint(source);
  int64_t blockThreads = blockThreadsOf(schedule, options, target);
  auto plan = func::FuncOp::create(builder, loc,
                                   (source.getName() + kernel.suffix).str(),
                                   FunctionType::get(context, inputs, {}));
  plan->setAttr(SwagePlanDialect::getBlockThreadsAttrName(),
                builder.getI32IntegerAttr(static_cast<int32_t>(blockThreads)));
  for (auto [index, argument] : llvm::enumerate(layout.arguments()))
    if (std::optional<ArgumentRole> role = roleOf(argument))
      plan.setArgAttr(static_cast<unsigned>(index),
                      SwageDialect::getRoleAttrName(),
                      ArgumentRoleAttr::get(context, *role));
  Block *entry = plan.addEntryBlock();
  auto argument = [&](KernelArgument parameter) {
    return Value(entry->getArgument(layout.indexOf(parameter)));
  };

  builder.setInsertionPointToEnd(entry);
  Operation *task = nullptr;
  Value output;
  if (schedule == PlanSchedule::SplitPartial) {
    // A partial task reduces one chunk into its scratch slot.
    task = swage_plan::PartialTasksOp::create(
        builder, loc, argument(KernelArgument::Values),
        argument(KernelArgument::ValueCount),
        argument(KernelArgument::PartialRanges),
        argument(KernelArgument::PartialCount),
        argument(KernelArgument::Scratch));
  } else {
    bool useTaskIds = schedule == PlanSchedule::TaskIds;
    // The task-id kernel reduces within one subgroup exactly when a block is
    // one subgroup. The direct kernel always reduces across the block.
    TaskPolicy policy = useTaskIds && blockThreads == target.subgroupWidth
                            ? TaskPolicy::Warp
                            : TaskPolicy::CTA;
    output = argument(KernelArgument::Output);
    task = swage_plan::TasksOp::create(
        builder, loc, argument(KernelArgument::Values),
        argument(KernelArgument::Offsets), argument(KernelArgument::ValueCount),
        argument(KernelArgument::SegmentCount),
        useTaskIds ? argument(KernelArgument::TaskIds) : Value(),
        useTaskIds ? argument(KernelArgument::TaskCount) : Value(),
        analysis.mapStores.empty() ? output : Value(), policy);
  }
  func::ReturnOp::create(builder, loc);
  fillTaskRegion(task, analysis, output);
  source.erase();
}

std::optional<PlanSchedule> parseSchedule(StringRef text) {
  if (text == "sequential")
    return PlanSchedule::Sequential;
  for (PlanSchedule schedule : {PlanSchedule::Direct, PlanSchedule::TaskIds,
                                PlanSchedule::SplitPartial})
    if (text == kernelSchedule(schedule)->name)
      return schedule;
  return std::nullopt;
}

class SwageToPlanPass
    : public PassWrapper<SwageToPlanPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwageToPlanPass)

  SwageToPlanPass() = default;
  SwageToPlanPass(const SwageToPlanPass &other) : PassWrapper(other) {
    schedules = other.schedules;
    blockThreads = other.blockThreads.getValue();
    selectedFunction = other.selectedFunction.getValue();
  }

  StringRef getArgument() const final { return "swage-to-plan"; }
  StringRef getDescription() const final {
    return "Replace every segment function by the plan functions of the "
           "kernels the schedules select";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, SwagePlanDialect>();
  }

  void runOnOperation() final {
    PlanOptions options;
    for (const std::string &name : schedules) {
      std::optional<PlanSchedule> schedule = parseSchedule(name);
      if (!schedule) {
        getOperation().emitError()
            << "schedule must be direct, task-ids, split-partial, or "
               "sequential, got '"
            << name << "'";
        return signalPassFailure();
      }
      options.schedules.push_back(*schedule);
    }
    if (options.schedules.empty())
      options.schedules.push_back(PlanSchedule::Direct);
    options.blockThreads = blockThreads;
    options.function = selectedFunction;
    if (failed(planSegmentFunctions(getOperation(), options, nvidiaTarget())))
      signalPassFailure();
  }

private:
  ListOption<std::string> schedules{
      *this, "schedule",
      llvm::cl::desc(
          "The kernels to plan, one plan function each: direct (one block "
          "per segment, the default), task-ids (one block per task of a "
          "task buffer), split-partial (one block per chunk of a long "
          "segment), or sequential (no kernel: the CPU oracle, alone)")};
  Option<int64_t> blockThreads{
      *this, "block-threads",
      llvm::cl::desc("Launch width of the direct and task-ids kernels in "
                     "threads; the target fixes every other width"),
      llvm::cl::init(nvidiaTarget().ctaBlockThreads)};
  Option<std::string> selectedFunction{
      *this, "function",
      llvm::cl::desc("Plan only this function instead of every function that "
                     "holds Swage operations")};
};

/// Require a list of schedules that names each kernel once. A sequential
/// plan keeps its function, so it stands alone.
LogicalResult verifySchedules(ModuleOp module, const PlanOptions &options,
                              const TargetDescription &target) {
  ArrayRef<PlanSchedule> schedules = options.schedules;
  if (schedules.empty())
    return module.emitError("the planner needs at least one schedule");
  if (llvm::is_contained(schedules, PlanSchedule::Sequential)) {
    if (schedules.size() != 1)
      return module.emitError(
          "the sequential schedule plans a function in place and keeps it, "
          "so it cannot share a schedule list with a kernel");
    return success();
  }
  for (auto [index, schedule] : llvm::enumerate(schedules)) {
    const KernelSchedule kernel = *kernelSchedule(schedule);
    for (PlanSchedule earlier : schedules.take_front(index))
      if (kernelSchedule(earlier)->suffix == kernel.suffix)
        return module.emitError()
               << "schedules " << kernelSchedule(earlier)->name << " and "
               << kernel.name << " both name their kernel @<function>"
               << kernel.suffix << "; a schedule list names each kernel once";
    if (kernel.takesBlockThreads &&
        !target.admitsBlockThreads(options.blockThreads))
      return module.emitError()
             << "block-threads must be a launch width the target admits, "
                "from 1 to "
             << target.maxBlockThreads
             << " threads with a power-of-two subgroup count, got "
             << options.blockThreads;
  }
  return success();
}

} // namespace

LogicalResult planSegmentFunctions(ModuleOp module, const PlanOptions &options,
                                   const TargetDescription &target) {
  if (failed(verifySchedules(module, options, target)))
    return failure();
  ArrayRef<PlanSchedule> schedules = options.schedules;
  bool sequential = schedules.front() == PlanSchedule::Sequential;
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
    if (sequential)
      continue;
    if (llvm::any_of(schedules,
                     [](PlanSchedule schedule) {
                       return kernelSchedule(schedule)->needsTaskProgram;
                     }) &&
        failed(verifyPlanningProgram(analysis)))
      return failure();
    // A kernel replaces its function. The oracle keeps it, so its callers
    // stay and no symbol is created.
    for (PlanSchedule schedule : schedules)
      if (failed(verifyKernelSymbols(module, function,
                                     kernelSchedule(schedule)->suffix)))
        return failure();
  }
  for (auto [function, analysis] : llvm::zip(*functions, analyses)) {
    if (sequential) {
      buildSequentialPlan(function, analysis);
      continue;
    }
    // One plan function per schedule, in the order of the list. Each but the
    // last is built from a copy of the segment function, which admission
    // accepts because it accepted the original.
    for (PlanSchedule schedule : schedules.drop_back()) {
      func::FuncOp copy = function.clone();
      function->getBlock()->getOperations().insert(function->getIterator(),
                                                   copy);
      SegmentProgramAnalysis copyAnalysis;
      [[maybe_unused]] LogicalResult admitted =
          analyzeSegmentProgram(copy, copyAnalysis);
      assert(succeeded(admitted) && "a copy of an admitted function");
      buildKernelPlan(copy, copyAnalysis, schedule, options, target);
    }
    buildKernelPlan(function, analysis, schedules.back(), options, target);
  }
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
