// test/Conversion/SwagePlanToSCF/pipeline.mlir
// The oracle as two passes of one pipeline: the planner writes a sequential
// plan and the conversion lowers it, which gives the module the one-step
// oracle flag gives.
//
// RUN: swage-opt --swage-segmented-reduction-to-scf \
// RUN:   %S/../SwageToCPU/segmented-sum.mlir > %t.sum
// RUN: swage-opt --swage-to-plan='schedule=sequential' --swage-plan-to-scf \
// RUN:   %S/../SwageToCPU/segmented-sum.mlir | diff %t.sum -
// RUN: swage-opt --swage-segmented-reduction-to-scf \
// RUN:   %S/../SwageToCPU/ragged-softmax.mlir > %t.softmax
// RUN: swage-opt --swage-to-plan='schedule=sequential' --swage-plan-to-scf \
// RUN:   %S/../SwageToCPU/ragged-softmax.mlir | diff %t.softmax -
// RUN: swage-opt --swage-segmented-reduction-to-scf \
// RUN:   %S/../SwageToCPU/two-kernels.mlir > %t.two
// RUN: swage-opt --swage-to-plan='schedule=sequential' --swage-plan-to-scf \
// RUN:   %S/../SwageToCPU/two-kernels.mlir | diff %t.two -
