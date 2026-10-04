// include/swage-c/Runtime.h
//===- Runtime.h - Swage runtime library C API --------------------*- C -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// The part of Swage that a process needs to run kernels that were compiled
// elsewhere: the classifier that turns segment offsets into task records,
// and a function that enqueues one kernel through the CUDA driver. The
// library that implements this header is plain C and links against the C
// library and the dynamic loader only. It includes no LLVM or MLIR header
// and loads `libcuda.so.1` at the first launch, not when it is loaded.
//
// The classifier produces the records swageClassifySegments of Codegen.h
// produces for the same offsets and limits, and refuses what it refuses with
// the same reason.
//
// Threads
//   Every function may run on any threads at the same time. The classifier
//   keeps no state. The launch function resolves the driver once and reads
//   only its arguments afterwards.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_C_RUNTIME_H
#define SWAGE_C_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SWAGE_RUNTIME_EXPORTED __attribute__((visibility("default")))

/// The version of the functions below. It changes when a signature or the
/// layout of the records changes.
#define SWAGE_RUNTIME_ABI_VERSION 1

/// Returns SWAGE_RUNTIME_ABI_VERSION of the library that was loaded.
SWAGE_RUNTIME_EXPORTED int32_t swageRuntimeAbiVersion(void);

/// Validates one layout and counts its task records.
///
/// `offsets` holds `offsetCount` i32 values, which must equal
/// `segmentCount + 1`. They must start at zero, never decrease, and end at
/// or below `valueCount`. A segment of at most `warpMaxElements` elements is
/// warp work, a longer one of at most `ctaChunkElements` elements is CTA
/// work, and a longer one is split into partial tasks of at most
/// `ctaChunkElements` elements and one merge.
///
/// Returns null and stores four counts in `counts`: the warp segments, the
/// CTA segments, the partial tasks, and the merges. Otherwise returns the
/// reason the input is refused, a string that lives as long as the library,
/// and leaves `counts` unchanged.
SWAGE_RUNTIME_EXPORTED const char *
swageRuntimeCountTasks(const int32_t *offsets, int64_t offsetCount,
                       int64_t valueCount, int64_t segmentCount,
                       int64_t warpMaxElements, int64_t ctaChunkElements,
                       int64_t *counts);

/// Writes the task records of a layout that swageRuntimeCountTasks admitted.
///
/// `counts` holds the four counts that call stored for the same offsets and
/// limits, and the offsets must not have changed since. `records` must have
/// room for `counts[0] + counts[1] + 3 * counts[2] + 3 * counts[3]` values.
/// The caller owns that buffer and decides its capacity: comparing the sum
/// with the capacity before this call is how a caller bounds its memory.
///
/// The buffer receives the warp segment ids, then the CTA segment ids, then
/// one [begin, end] pair per partial task, then one
/// [segment_id, partial_begin, partial_end] triple per merge, then for each
/// partial task the index of its merge triple.
SWAGE_RUNTIME_EXPORTED void
swageRuntimeWriteTasks(const int32_t *offsets, int64_t segmentCount,
                       int64_t warpMaxElements, int64_t ctaChunkElements,
                       const int64_t *counts, int32_t *records);

/// Results of swageRuntimeLaunch that do not come from the driver.
#define SWAGE_RUNTIME_ERROR_DRIVER (-1)
#define SWAGE_RUNTIME_ERROR_GRID (-2)
#define SWAGE_RUNTIME_ERROR_BLOCK (-3)
#define SWAGE_RUNTIME_ERROR_ARGUMENTS (-4)

/// Enqueues one kernel on `stream` with a grid of `gridX` blocks of `blockX`
/// threads.
///
/// `function` is a CUfunction and `stream` a CUstream, or zero for the
/// default stream; the CUDA context that loaded the function must be current
/// on the calling thread. The kernel receives the `pointerCount` values of
/// `pointers` as device pointers, followed by the `scalarCount` values of
/// `scalars` as i32, at most 16 arguments in all. Both arrays are read
/// during the call only.
///
/// Returns 0 when the kernel was enqueued. A positive result is the CUresult
/// of cuLaunchKernel. A negative result is one of the codes above: the
/// driver library or its launch function is unavailable, `gridX` is not in
/// 1..UINT32_MAX, `blockX` is not in 1..1024, or the argument counts are
/// negative or exceed 16. Nothing is enqueued for a nonzero result.
SWAGE_RUNTIME_EXPORTED int32_t swageRuntimeLaunch(
    uint64_t function, int64_t gridX, int64_t blockX, uint64_t stream,
    const uint64_t *pointers, int32_t pointerCount, const int32_t *scalars,
    int32_t scalarCount);

/// Describes a nonzero result of swageRuntimeLaunch. `name` and `text`
/// receive strings that live as long as the library or the driver, or null
/// for a CUresult that the driver does not describe.
SWAGE_RUNTIME_EXPORTED void
swageRuntimeDescribe(int32_t result, const char **name, const char **text);

#ifdef __cplusplus
}
#endif

#endif // SWAGE_C_RUNTIME_H
