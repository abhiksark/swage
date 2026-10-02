// include/swage/Dialect/SwagePlan/IR/TaskRecords.h
//===- TaskRecords.h - Layouts of the host task records --------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// The records host classification writes and a split kernel reads, as data.
// A record is a run of i32 words in a task buffer. The classifier fills the
// words by these fields and a lowering loads them by the same fields, so
// the two cannot disagree about a stride.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_DIALECT_SWAGEPLAN_IR_TASKRECORDS_H
#define SWAGE_DIALECT_SWAGEPLAN_IR_TASKRECORDS_H

namespace mlir::swage_plan {

/// One chunk of a split segment: a half-open range of value indices. The
/// partial task with index `t` reads record `t` and writes scratch slot `t`.
namespace partial_record {
enum Field : unsigned { Begin, End, Words };
} // namespace partial_record

/// One split segment: the segment that receives the result, and the
/// half-open range of scratch slots its partial tasks wrote.
namespace merge_record {
enum Field : unsigned { Segment, PartialBegin, PartialEnd, Words };
} // namespace merge_record

} // namespace mlir::swage_plan

#endif // SWAGE_DIALECT_SWAGEPLAN_IR_TASKRECORDS_H
