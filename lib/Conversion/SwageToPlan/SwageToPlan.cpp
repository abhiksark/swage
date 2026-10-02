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
#include "mlir/IR/IRMapping.h"
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
  case PlanSchedule::FusedMixed:
    return KernelSchedule{"fused-mixed", KernelKind::FusedMixed, "", true,
                          false};
  case PlanSchedule::SplitPartial:
    return KernelSchedule{"split-partial", KernelKind::SplitPartial,
                          "__partial", true, false};
  case PlanSchedule::SplitMerge:
    return KernelSchedule{"split-merge", KernelKind::SplitMerge, "__merge",
                          true, false};
  case PlanSchedule::Persistent:
    return KernelSchedule{"persistent", KernelKind::Persistent, "", true,
                          false};
  case PlanSchedule::Sequential:
    return std::nullopt;
  }
  llvm_unreachable("unknown plan schedule");
}

/// The launch width of the kernel `schedule` plans.
int64_t blockThreadsOf(PlanSchedule schedule, const PlanOptions &options,
                       const TargetDescription &target) {
  if (schedule == PlanSchedule::SplitPartial ||
      schedule == PlanSchedule::SplitMerge)
    return target.splitBlockThreads;
  if (schedule == PlanSchedule::FusedMixed)
    return target.ctaBlockThreads;
  if (schedule == PlanSchedule::Persistent)
    return target.persistentBlockThreads;
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

/// Give a task operation its merge region: an identity reduction, of the
/// kind of the program, over the bound range of scratch. Scratch holds
/// completed partial reductions, so the element program of the function
/// stays behind with the function and the merge never runs it.
void fillMergeRegion(Region &region, SegmentProgramAnalysis &analysis) {
  Location loc = region.getParentOp()->getLoc();
  ReduceOp reduction = analysis.reductions.front();
  Type element = reduction.getResult().getType();
  OpBuilder builder(loc.getContext());
  Block *body = builder.createBlock(
      &region, {}, {analysis.segments.front().getResult().getType()}, {loc});
  auto merged = ReduceOp::create(builder, loc, element, body->getArgument(0),
                                 ValueRange(), reduction.getKind());
  swage_plan::YieldOp::create(builder, loc, merged.getResult());
  Block *identity =
      builder.createBlock(&merged.getBody(), {}, {element}, {loc});
  YieldOp::create(builder, loc, identity->getArgument(0));
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
  if (schedule == PlanSchedule::Persistent) {
    output = argument(KernelArgument::Output);
    task = swage_plan::PersistentTasksOp::create(
        builder, loc, argument(KernelArgument::Values),
        argument(KernelArgument::Offsets), argument(KernelArgument::ValueCount),
        argument(KernelArgument::SegmentCount),
        argument(KernelArgument::WarpIds),
        argument(KernelArgument::WarpTaskCount),
        argument(KernelArgument::CtaIds),
        argument(KernelArgument::CtaTaskCount),
        argument(KernelArgument::PartialRanges),
        argument(KernelArgument::PartialMergeIds),
        argument(KernelArgument::PartialCount),
        argument(KernelArgument::MergeRecords),
        argument(KernelArgument::MergeCount), argument(KernelArgument::Scratch),
        argument(KernelArgument::Counters), output);
  } else if (schedule == PlanSchedule::FusedMixed) {
    output = argument(KernelArgument::Output);
    task = swage_plan::FusedTasksOp::create(
        builder, loc, argument(KernelArgument::Values),
        argument(KernelArgument::Offsets), argument(KernelArgument::ValueCount),
        argument(KernelArgument::SegmentCount),
        argument(KernelArgument::TaskIds),
        argument(KernelArgument::WarpTaskCount),
        argument(KernelArgument::CtaTaskCount), output);
  } else if (schedule == PlanSchedule::SplitMerge) {
    // A merge task reduces the partial results of one split segment.
    task = swage_plan::MergeTasksOp::create(
        builder, loc, argument(KernelArgument::Scratch),
        argument(KernelArgument::PartialCount),
        argument(KernelArgument::MergeRecords),
        argument(KernelArgument::MergeCount),
        argument(KernelArgument::SegmentCount),
        argument(KernelArgument::Output));
  } else if (schedule == PlanSchedule::SplitPartial) {
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
  if (schedule == PlanSchedule::SplitMerge) {
    fillMergeRegion(task->getRegion(0), analysis);
    source.erase();
    return;
  }
  fillTaskRegion(task, analysis, output);
  // A warp task and a block task run the same program, so the block region
  // of a fused task operation is a copy of its warp region.
  if (schedule == PlanSchedule::FusedMixed) {
    IRMapping mapping;
    task->getRegion(0).cloneInto(&task->getRegion(1), mapping);
  }
  // A block task, a partial task, and a warp task of the queue kernel run
  // the program too, and its merge combines their partial results.
  if (auto persistent = dyn_cast<swage_plan::PersistentTasksOp>(task)) {
    for (Region *region : {&persistent.getPartial(), &persistent.getWarp()}) {
      IRMapping mapping;
      persistent.getCta().cloneInto(region, mapping);
    }
    fillMergeRegion(persistent.getMerge(), analysis);
  }
  source.erase();
}

/// The name of `schedule` in the `schedule` option.
StringRef nameOf(PlanSchedule schedule) {
  std::optional<KernelSchedule> kernel = kernelSchedule(schedule);
  return kernel ? StringRef(kernel->name) : StringRef("sequential");
}

std::optional<PlanSchedule> parseSchedule(StringRef text) {
  if (text == "sequential")
    return PlanSchedule::Sequential;
  for (PlanSchedule schedule :
       {PlanSchedule::Direct, PlanSchedule::TaskIds, PlanSchedule::FusedMixed,
        PlanSchedule::SplitPartial, PlanSchedule::SplitMerge,
        PlanSchedule::Persistent})
    if (text == kernelSchedule(schedule)->name)
      return schedule;
  return std::nullopt;
}

class SwageToPlanPass
    : public PassWrapper<SwageToPlanPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwageToPlanPass)

  SwageToPlanPass() = default;
  // Pass::clone copies the option values after it copies the pass.
  SwageToPlanPass(const SwageToPlanPass &other)
      : PassWrapper(other), target(other.target) {}
  SwageToPlanPass(const PlanOptions &options, const TargetDescription &target)
      : target(&target) {
    SmallVector<std::string> names;
    for (PlanSchedule schedule : options.schedules)
      names.push_back(nameOf(schedule).str());
    schedules = names;
    blockThreads = options.blockThreads;
    selectedFunction = options.function.str();
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
            << "schedule must be direct, task-ids, fused-mixed, "
               "split-partial, split-merge, persistent, or sequential, got '"
            << name << "'";
        return signalPassFailure();
      }
      options.schedules.push_back(*schedule);
    }
    if (options.schedules.empty())
      options.schedules.push_back(PlanSchedule::Direct);
    options.blockThreads = blockThreads;
    options.function = selectedFunction;
    if (failed(planSegmentFunctions(getOperation(), options, *target)))
      signalPassFailure();
  }

private:
  const TargetDescription *target = &nvidiaTarget();
  ListOption<std::string> schedules{
      *this, "schedule",
      llvm::cl::desc(
          "The kernels to plan, one plan function each: direct (one block "
          "per segment, the default), task-ids (one block per task of a "
          "task buffer), fused-mixed (warp tasks and block tasks in one "
          "launch), split-partial (one block per chunk of a long segment), "
          "split-merge (one block per split segment, over its partial "
          "results), persistent (resident blocks that drain task queues), "
          "or sequential (no kernel: the CPU oracle, alone)")};
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
    if (llvm::is_contained(schedules, PlanSchedule::Persistent) &&
        failed(verifyPersistentProgram(analysis)))
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

std::unique_ptr<Pass> createSwageToPlanPass(const PlanOptions &options,
                                            const TargetDescription &target) {
  return std::make_unique<SwageToPlanPass>(options, target);
}

void registerSwageToPlanPass() { PassRegistration<SwageToPlanPass>(); }

} // namespace mlir::swage
