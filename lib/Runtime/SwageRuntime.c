// lib/Runtime/SwageRuntime.c
//===- SwageRuntime.c - Swage runtime library -----------------------------===//
//
// The record classifier and the kernel launcher for a process that runs
// kernels compiled elsewhere. This file must stay free of LLVM and MLIR: it
// includes the C library, the dynamic loader, and its own header only.
//
// The classifier is a second implementation of classifyTaskRecords in
// lib/Dialect/SwagePlan/IR/TaskClassifier.cpp. unittests/RuntimeTest.cpp
// holds the two to the same records and the same refusals.
//
//===----------------------------------------------------------------------===//

#include "swage-c/Runtime.h"

#include <dlfcn.h>
#include <stdatomic.h>
#include <stddef.h>

#define SWAGE_I32_MAX INT64_C(2147483647)

int32_t swageRuntimeAbiVersion(void) { return SWAGE_RUNTIME_ABI_VERSION; }

static int outsideI32(int64_t value) {
  return value < 0 || value > SWAGE_I32_MAX;
}

/// Names the first offset that is negative or below the one before it. The
/// caller has seen one, so the walk finds it.
static const char *firstOffsetError(const int32_t *offsets,
                                    int64_t offsetCount) {
  int64_t previous = 0;
  for (int64_t index = 0; index < offsetCount; ++index) {
    if (offsets[index] < 0)
      return "offset must be a nonnegative i32 value";
    if (offsets[index] < previous)
      break;
    previous = offsets[index];
  }
  return "offsets must be nondecreasing";
}

const char *swageRuntimeCountTasks(const int32_t *offsets, int64_t offsetCount,
                                   int64_t valueCount, int64_t segmentCount,
                                   int64_t warpMaxElements,
                                   int64_t ctaChunkElements, int64_t *counts) {
  if (outsideI32(valueCount))
    return "value count must be a nonnegative i32 value";
  if (outsideI32(segmentCount))
    return "segment count must be a nonnegative i32 value";
  if (outsideI32(warpMaxElements))
    return "warp max elements must be a nonnegative i32 value";
  if (outsideI32(ctaChunkElements))
    return "CTA chunk elements must be a nonnegative i32 value";
  if (warpMaxElements == 0)
    return "warp max elements must be positive";
  if (ctaChunkElements == 0)
    return "CTA chunk elements must be positive";
  if (warpMaxElements > ctaChunkElements)
    return "warp max elements must not exceed CTA chunk elements";
  if (offsetCount != segmentCount + 1)
    return "offset count must equal segment count plus one";
  if (!offsets)
    return "offsets must not be null";

  // One walk validates and counts. A negative or decreasing offset is a
  // decrease, or is the first offset itself, because every offset before the
  // first invalid one is at least zero. Valid offsets lie in [0, i32 max],
  // so their differences fit in i32; the counts of an invalid walk are never
  // used. The body has no branch and stays in 32-bit values so that it
  // vectorizes.
  const int32_t warpLimit = (int32_t)warpMaxElements;
  const int32_t chunkLimit = (int32_t)ctaChunkElements;
  uint32_t disorder = offsets[0] < 0;
  uint32_t warpSegments = 0;
  uint32_t splitSegments = 0;
  for (int64_t segment = 0; segment < segmentCount; ++segment) {
    const int32_t begin = offsets[segment];
    const int32_t end = offsets[segment + 1];
    const int32_t length = (int32_t)((uint32_t)end - (uint32_t)begin);
    disorder |= end < begin;
    warpSegments += length <= warpLimit;
    splitSegments += length > chunkLimit;
  }
  if (disorder)
    return firstOffsetError(offsets, offsetCount);
  if (offsets[0] != 0)
    return "offsets must start at zero";
  if (offsets[segmentCount] > valueCount)
    return "final offset must not exceed value count";

  // The partial tasks need a division per split segment, so they are
  // counted in a second walk, and only when a segment is split.
  uint64_t partialTasks = 0;
  if (splitSegments) {
    for (int64_t segment = 0; segment < segmentCount; ++segment) {
      const int64_t length = (int64_t)offsets[segment + 1] - offsets[segment];
      if (length > ctaChunkElements)
        partialTasks += (uint64_t)(length / ctaChunkElements) +
                        (uint64_t)(length % ctaChunkElements != 0);
    }
  }
  // One record per unsplit segment, per partial task, and per merge.
  if ((uint64_t)segmentCount + partialTasks > (uint64_t)SWAGE_I32_MAX)
    return "descriptor count must fit in i32";
  counts[0] = warpSegments;
  counts[1] = segmentCount - warpSegments - splitSegments;
  counts[2] = (int64_t)partialTasks;
  counts[3] = splitSegments;
  return NULL;
}

void swageRuntimeWriteTasks(const int32_t *offsets, int64_t segmentCount,
                            int64_t warpMaxElements, int64_t ctaChunkElements,
                            const int64_t *counts, int32_t *records) {
  int32_t *warp = records;
  int32_t *cta = warp + counts[0];
  int32_t *partial = cta + counts[1];
  int32_t *merge = partial + 2 * counts[2];
  int32_t *partialMerge = merge + 3 * counts[3];
  int32_t scratchIndex = 0;
  int32_t mergeIndex = 0;
  int64_t begin = 0;
  for (int64_t segment = 0; segment < segmentCount; ++segment) {
    const int64_t end = offsets[segment + 1];
    const int64_t length = end - begin;
    if (length <= warpMaxElements) {
      *warp++ = (int32_t)segment;
    } else if (length <= ctaChunkElements) {
      *cta++ = (int32_t)segment;
    } else {
      *merge++ = (int32_t)segment;
      *merge++ = scratchIndex;
      for (int64_t chunkBegin = begin; chunkBegin < end;
           chunkBegin += ctaChunkElements) {
        const int64_t chunkEnd = chunkBegin + ctaChunkElements;
        *partial++ = (int32_t)chunkBegin;
        *partial++ = (int32_t)(chunkEnd < end ? chunkEnd : end);
        *partialMerge++ = mergeIndex;
        ++scratchIndex;
      }
      *merge++ = scratchIndex;
      ++mergeIndex;
    }
    begin = end;
  }
}

typedef int (*SwageLaunchFn)(void *, unsigned, unsigned, unsigned, unsigned,
                             unsigned, unsigned, unsigned, void *, void **,
                             void **);
typedef int (*SwageErrorTextFn)(int, const char **);

// The driver entry points, resolved at the first launch. The library does
// not link against libcuda: a host without a driver still classifies. Two
// threads that both resolve store the same addresses.
static _Atomic(SwageLaunchFn) driverLaunch;
static _Atomic(SwageErrorTextFn) driverErrorName;
static _Atomic(SwageErrorTextFn) driverErrorString;

static SwageLaunchFn resolveDriver(void) {
  SwageLaunchFn launch = atomic_load(&driverLaunch);
  if (launch)
    return launch;
  void *library = dlopen("libcuda.so.1", RTLD_NOW);
  if (!library)
    return NULL;
  atomic_store(&driverErrorName,
               (SwageErrorTextFn)dlsym(library, "cuGetErrorName"));
  atomic_store(&driverErrorString,
               (SwageErrorTextFn)dlsym(library, "cuGetErrorString"));
  launch = (SwageLaunchFn)dlsym(library, "cuLaunchKernel");
  atomic_store(&driverLaunch, launch);
  return launch;
}

#define SWAGE_MAX_ARGUMENTS 16

int32_t swageRuntimeLaunch(uint64_t function, int64_t gridX, int64_t blockX,
                           uint64_t stream, const uint64_t *pointers,
                           int32_t pointerCount, const int32_t *scalars,
                           int32_t scalarCount) {
  if (gridX <= 0 || gridX > INT64_C(0xFFFFFFFF))
    return SWAGE_RUNTIME_ERROR_GRID;
  if (blockX <= 0 || blockX > 1024)
    return SWAGE_RUNTIME_ERROR_BLOCK;
  if (pointerCount < 0 || scalarCount < 0 ||
      pointerCount > SWAGE_MAX_ARGUMENTS ||
      scalarCount > SWAGE_MAX_ARGUMENTS - pointerCount)
    return SWAGE_RUNTIME_ERROR_ARGUMENTS;
  SwageLaunchFn launch = resolveDriver();
  if (!launch)
    return SWAGE_RUNTIME_ERROR_DRIVER;
  // The driver takes the address of each argument and copies the value
  // before it returns. It does not write through these addresses, although
  // its parameter type is not const.
  void *parameters[SWAGE_MAX_ARGUMENTS];
  int32_t index = 0;
  for (int32_t pointer = 0; pointer < pointerCount; ++pointer)
    parameters[index++] = (void *)(uintptr_t)&pointers[pointer];
  for (int32_t scalar = 0; scalar < scalarCount; ++scalar)
    parameters[index++] = (void *)(uintptr_t)&scalars[scalar];
  return launch((void *)(uintptr_t)function, (unsigned)gridX, 1, 1,
                (unsigned)blockX, 1, 1, 0, (void *)(uintptr_t)stream,
                parameters, NULL);
}

void swageRuntimeDescribe(int32_t result, const char **name,
                          const char **text) {
  *name = NULL;
  *text = NULL;
  switch (result) {
  case SWAGE_RUNTIME_ERROR_DRIVER:
    *name = "SWAGE_RUNTIME_ERROR_DRIVER";
    *text = "CUDA Driver library libcuda.so.1 is unavailable";
    return;
  case SWAGE_RUNTIME_ERROR_GRID:
    *name = "SWAGE_RUNTIME_ERROR_GRID";
    *text = "grid_x must be a positive u32";
    return;
  case SWAGE_RUNTIME_ERROR_BLOCK:
    *name = "SWAGE_RUNTIME_ERROR_BLOCK";
    *text = "block_x must be in 1..1024";
    return;
  case SWAGE_RUNTIME_ERROR_ARGUMENTS:
    *name = "SWAGE_RUNTIME_ERROR_ARGUMENTS";
    *text = "too many kernel arguments";
    return;
  default:
    break;
  }
  SwageErrorTextFn errorName = atomic_load(&driverErrorName);
  SwageErrorTextFn errorString = atomic_load(&driverErrorString);
  if (errorName)
    errorName(result, name);
  if (errorString)
    errorString(result, text);
}
