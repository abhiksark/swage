// test/Conversion/SwagePlanToGPU/persistent-pipeline.mlir
// The persistent queue kernel as two passes of one pipeline: the plan
// converts to the module the one-step persistent flag gives.
//
// RUN: swage-opt \
// RUN:   --swage-segmented-reduction-to-gpu='block-size=512 persistent' \
// RUN:   %S/../SwageToGPU/persistent.mlir > %t.persistent
// RUN: swage-opt --swage-to-plan='schedule=persistent' --swage-plan-to-gpu \
// RUN:   %S/../SwageToGPU/persistent.mlir | diff %t.persistent -
