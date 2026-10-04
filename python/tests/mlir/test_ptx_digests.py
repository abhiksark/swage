# python/tests/mlir/test_ptx_digests.py
"""Digest gate over the text of every kernel the compiler emits.

Each kernel in the matrix below is compiled, and the SHA-256 of its lowered
MLIR and of its PTX is compared with `ptx_digests.json` beside this file.
Nothing is launched, so the gate needs the native bindings and no GPU.

A digest that moves means the emitted text of a kernel changed. That is the
point of the gate: a refactoring of a lowering must keep every byte, and a
change that is meant to alter kernels has to say so by regenerating the
file. Regeneration is legitimate in two cases only: a change of the LLVM
pin, and a change to a lowering or to code generation that states which
kernels change and why. Run this from the repository root, after building:

    PYTHONPATH=$PWD/python:$PWD/build/python_packages \
        python3 python/tests/mlir/test_ptx_digests.py --write

The command compiles the whole matrix with the native package on the path
and then replaces `python/tests/mlir/ptx_digests.json`. It writes no other
file, and it leaves the old file in place if any compile fails. Without
`--write` it only reports what differs.

The digests cover the text, not the behavior. They say nothing about a
kernel that still compiles to the same bytes and is wrong, and nothing about
a processor on which the PTX was never run.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile

import pytest
from mlir_swage import ir
from mlir_swage._mlir_libs import _swageDialectsNanobind as extension
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from mlir_swage.dialects import swage
from reduction_programs import reduction_module
from swage import _segmented_qualification as qualification
from test_target_compile import _ADMITTED as _PROCESSORS

_DATA = pathlib.Path(__file__).with_name("ptx_digests.json")
_REGENERATE = (
    "PYTHONPATH=$PWD/python:$PWD/build/python_packages "
    "python3 python/tests/mlir/test_ptx_digests.py --write"
)

# `_PROCESSORS` is the list `test_target_compile.py` keeps equal to the
# processors the code generation C API admits, so a newly admitted processor
# fails here as a missing kernel until the data is regenerated.
#
# The programs that were in the matrix when the gate was added are compiled
# for every processor. A program that was added later is compiled for the
# oldest admitted processor and for the one the kernels are executed on,
# unless its entry below says otherwise.
_NEWER_PROCESSORS = (80, 86)

_FIXED_VECTOR_ADD = """
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
"""

# The storage types of a fixed elementwise kernel.
_FIXED_ELEMENTS = ("f32", "f16", "f8E4M3FN", "f8E5M2")


def _fixed_elementwise(operation, element):
    """Return the fixed vector program of one operation and storage type.

    Args:
        operation: "add" or "multiply".
        element: An MLIR float type in `_FIXED_ELEMENTS`.

    Returns:
        The text of `_FIXED_VECTOR_ADD` with that operation and element type.
    """
    text = _FIXED_VECTOR_ADD.replace("f32", element)
    if operation == "multiply":
        text = text.replace("@add_kernel", "@multiply_kernel")
        text = text.replace("arith.addf", "arith.mulf")
    return text


# The schedules of a planned reduction: the variant name, the native compile
# function, and its options. The direct kernel is compiled at the block
# width of a CTA task and at the widest tile the split stages use, and the
# task-ID kernel at the warp and the CTA width.
_REDUCTION_VARIANTS = (
    ("direct-128", "_compile_segmented_reduction_ptx", {"block_size": 128}),
    ("direct-512", "_compile_segmented_reduction_ptx", {"block_size": 512}),
    (
        "task-ids-32",
        "_compile_segmented_reduction_ptx",
        {"block_size": 32, "use_task_ids": True},
    ),
    (
        "task-ids-128",
        "_compile_segmented_reduction_ptx",
        {"block_size": 128, "use_task_ids": True},
    ),
    ("fused-mixed", "_compile_fused_segmented_reduction_ptx", {}),
    ("split-partial", "_compile_split_partial_reduction_ptx", {}),
    ("split-merge", "_compile_split_merge_reduction_ptx", {}),
)
_PERSISTENT_VARIANT = (
    "persistent", "_compile_persistent_segmented_reduction_ptx", {},
)

# The reduction programs: a kind of `segment_reduce`, an element transform of
# `reduction_programs.reduction_module`, an element type, and the processors
# each is compiled for. Together they cover every admitted kind and element
# type, a region with arithmetic, a chain of maps, and `math.exp2`.
#
# The f64 sum and maximum are compiled for every processor although they
# were added later: the backend selects other instructions for f64, and an
# f64 maximum is expanded into a compare and selects, so their text is the
# one most likely to differ between processors.
_REDUCTIONS = (
    ("sum", "identity", "f32", _PROCESSORS),
    ("sum", "square", "f32", _PROCESSORS),
    ("sum", "maps", "f32", _PROCESSORS),
    ("sum", "exp2", "f32", _PROCESSORS),
    ("max", "identity", "f32", _PROCESSORS),
    ("max", "maps", "f32", _PROCESSORS),
    ("min", "identity", "f32", _NEWER_PROCESSORS),
    ("min", "maps", "f32", _NEWER_PROCESSORS),
    ("sum", "identity", "f64", _PROCESSORS),
    ("max", "identity", "f64", _PROCESSORS),
    ("min", "identity", "f64", _NEWER_PROCESSORS),
    ("sum", "square", "f64", _NEWER_PROCESSORS),
    ("max", "maps", "f64", _NEWER_PROCESSORS),
    # A mean is a sum with a division after the combination of each task.
    # Its merge kernel also reads the range records of the partial tasks.
    ("mean", "identity", "f32", _NEWER_PROCESSORS),
    ("mean", "identity", "f64", _NEWER_PROCESSORS),
)


def _programs():
    """Return the matrix as programs, each with its kernels.

    Returns:
        A dict from program name to a tuple of the semantic module text, the
        kernel name, the variants to compile, and the processors to compile
        them for. A variant is a name, a native compile function name, and
        that function's options.
    """
    programs = {
        "fixed-add": (
            _FIXED_VECTOR_ADD,
            "add_kernel",
            (("block-128", "_compile_ptx", {"block_size": 128}),),
            _PROCESSORS,
        ),
    }
    # The other fixed elementwise kernels: the multiply, and both operations
    # on half and on the two eight-bit float formats, which are stored in
    # their own width and computed in f32.
    for operation in ("add", "multiply"):
        for element in _FIXED_ELEMENTS:
            if (operation, element) == ("add", "f32"):
                continue
            programs[f"fixed-{operation}-{element}"] = (
                _fixed_elementwise(operation, element),
                f"{operation}_kernel",
                (("block-128", "_compile_ptx", {"block_size": 128}),),
                _NEWER_PROCESSORS,
            )
    for kind, transform, element, processors in _REDUCTIONS:
        variants = _REDUCTION_VARIANTS
        # The persistent queue admits the f32 identity sum only.
        if (kind, transform, element) == ("sum", "identity", "f32"):
            variants += (_PERSISTENT_VARIANT,)
        suffix = "" if element == "f32" else f"-{element}"
        programs[f"{kind}-{transform}{suffix}"] = (
            reduction_module(kind, transform, element),
            qualification._reduction_kernel(kind, element),
            variants,
            processors,
        )
    # The reductions over rank-two values. Each has one kernel, the direct
    # schedule, which the runtime compiles at the CTA block width.
    for element in ("f32", "f64"):
        for kind in ("sum", "max", "min", "mean"):
            suffix = "" if element == "f32" else f"-{element}"
            programs[f"{kind}-columns{suffix}"] = (
                qualification._semantic_module(kind, element, 2),
                qualification._reduction_kernel(kind, element, 2),
                (
                    (
                        "direct-128",
                        "_compile_segmented_reduction_ptx",
                        {"block_size": 128},
                    ),
                ),
                _NEWER_PROCESSORS,
            )
    # The one program with a `map_store` terminal, as the runtime compiles
    # it: one block per segment, at its default width and at one warp.
    programs["ragged-softmax"] = (
        qualification._SOFTMAX_MODULE,
        "ragged_softmax",
        (
            (
                "direct-128",
                "_compile_segmented_reduction_ptx",
                {"block_size": 128},
            ),
            (
                "direct-32",
                "_compile_segmented_reduction_ptx",
                {"block_size": 32},
            ),
        ),
        _PROCESSORS,
    )
    # The softmax over rank-two values: the three stages of a column in
    # one thread, in the one kernel of a rank-two program.
    programs["ragged-softmax-columns"] = (
        qualification._softmax_text(2),
        "ragged_softmax_r2",
        (
            (
                "direct-128",
                "_compile_segmented_reduction_ptx",
                {"block_size": 128},
            ),
        ),
        _NEWER_PROCESSORS,
    )
    return programs


_PROGRAMS = _programs()
# Every program with each processor it is compiled for.
_CELLS = [
    (program, processor)
    for program, (_, _, _, processors) in _PROGRAMS.items()
    for processor in processors
]


def _sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _compile(program, processor):
    """Compile every kernel of one program for one processor.

    Args:
        program: A key of `_PROGRAMS`.
        processor: The number of an admitted `sm_` processor.

    Returns:
        A dict from `program/variant/sm_N` to the pair of digests of the
        lowered MLIR and of the PTX.
    """
    text, kernel_name, variants, _ = _PROGRAMS[program]
    target = f"sm_{processor}"
    digests = {}
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(text)
        for variant, function, options in variants:
            lowered, ptx, _ = getattr(native_swage, function)(
                module, kernel_name=kernel_name, target=target, **options
            )
            digests[f"{program}/{variant}/{target}"] = [
                _sha256(lowered), _sha256(ptx),
            ]
    return digests


def _compile_matrix():
    """Compile the whole matrix and return its digests, sorted by key."""
    digests = {}
    for program, processor in _CELLS:
        digests.update(_compile(program, processor))
    return dict(sorted(digests.items()))


def _recorded():
    """Return the committed data, or an empty record when it is missing."""
    if not _DATA.is_file():
        return {"llvm": None, "pairs": 0, "digests": {}}
    return json.loads(_DATA.read_text())


def _describe(key, recorded, actual):
    """Say which half of one pair moved."""
    if recorded is None:
        return f"{key}: not in the data file"
    lowered = recorded[0] != actual[0]
    ptx = recorded[1] != actual[1]
    if lowered and ptx:
        return f"{key}: the lowered MLIR and the PTX differ"
    if ptx:
        return f"{key}: the PTX differs, the lowered MLIR is unchanged"
    return f"{key}: the lowered MLIR differs, the PTX is unchanged"


def _guidance(recorded_llvm):
    """Return the part of a failure message that says what to do."""
    lines = [
        "A digest that moved means the emitted text of a kernel changed.",
        "That is a regression unless the change is intended. Regenerating",
        "the data is legitimate only with a change of the LLVM pin, or with",
        "a change to a lowering or to code generation that states which",
        "kernels change and why. In that case run, from the repository",
        "root after building, and commit the file with the change:",
        f"  {_REGENERATE}",
        f"It rewrites {_DATA.name} beside this test and nothing else.",
    ]
    if recorded_llvm is None:
        lines.insert(0, f"{_DATA} is missing.")
    elif recorded_llvm != native_swage.__llvm_version__:
        lines.insert(
            0,
            f"The data was generated with LLVM {recorded_llvm}; this build "
            f"links LLVM {native_swage.__llvm_version__}.",
        )
    return "\n".join(lines)


@pytest.mark.parametrize(("program", "processor"), _CELLS)
def test_kernel_text_matches_the_recorded_digests(program, processor):
    """Compile one program for one processor and compare every digest."""
    recorded = _recorded()
    actual = _compile(program, processor)

    moved = [
        _describe(key, recorded["digests"].get(key), pair)
        for key, pair in actual.items()
        if recorded["digests"].get(key) != pair
    ]

    assert not moved, (
        f"{len(moved)} of {len(actual)} kernels of {program!r} on "
        f"sm_{processor} do not match {_DATA.name}:\n  "
        + "\n  ".join(moved)
        + "\n"
        + _guidance(recorded["llvm"])
    )


def test_the_data_file_holds_exactly_the_matrix():
    """Refuse a data file with stale or missing kernels."""
    recorded = _recorded()
    expected = {
        f"{program}/{variant}/sm_{processor}"
        for program, (_, _, variants, processors) in _PROGRAMS.items()
        for variant, _, _ in variants
        for processor in processors
    }
    stale = sorted(set(recorded["digests"]) - expected)
    missing = sorted(expected - set(recorded["digests"]))

    assert not stale and not missing, (
        f"{_DATA.name} does not hold the matrix this test compiles: "
        f"{len(missing)} kernels are missing and {len(stale)} are no "
        f"longer compiled (first missing: {missing[:1]}, first stale: "
        f"{stale[:1]}).\n" + _guidance(recorded["llvm"])
    )
    assert recorded["pairs"] == len(expected) == len(recorded["digests"])


def _render(digests):
    """Render the data file: a short header, then one pair per line."""
    lines = [
        "{",
        f' "llvm": {json.dumps(native_swage.__llvm_version__)},',
        f' "pairs": {len(digests)},',
        ' "digests": {',
    ]
    rows = [
        f"  {json.dumps(key)}: {json.dumps(pair)}"
        for key, pair in digests.items()
    ]
    lines.append(",\n".join(rows))
    lines.extend([" }", "}"])
    return "\n".join(lines) + "\n"


def _main(arguments):
    """Report what differs from the data file and optionally rewrite it.

    Args:
        arguments: Command-line arguments without the program name.

    Returns:
        The process exit status: 0 when the file matches or was written, 1
        when it differs and was left alone.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"replace {_DATA.name} with the digests of this build",
    )
    options = parser.parse_args(arguments)

    recorded = _recorded()
    digests = _compile_matrix()
    old = recorded["digests"]
    lowered = sum(key in old and old[key][0] != digests[key][0]
                  for key in digests)
    ptx = sum(key in old and old[key][1] != digests[key][1]
              for key in digests)
    added = len(set(digests) - set(old))
    removed = len(set(old) - set(digests))
    print(f"native extension: {extension.__file__}")
    print(f"linked LLVM: {native_swage.__llvm_version__}")
    print(
        f"compiled {len(digests)} kernels: {lowered} lowered MLIR digests "
        f"and {ptx} PTX digests differ from {_DATA.name}, {added} kernels "
        f"are new, {removed} are gone"
    )
    unchanged = not (lowered or ptx or added or removed)
    if not options.write:
        if not unchanged:
            print(f"left {_DATA} unchanged; pass --write to replace it")
        return 0 if unchanged else 1

    # Written beside the target and renamed, so an interrupted run leaves
    # the old file whole.
    handle, temporary = tempfile.mkstemp(
        dir=_DATA.parent, prefix=_DATA.name, suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(_render(digests))
        os.replace(temporary, _DATA)
    except BaseException:
        os.unlink(temporary)
        raise
    print(f"wrote {_DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
