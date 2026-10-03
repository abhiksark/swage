# python/tests/mlir/artifact_child.py
"""Run the public segmented calls from an artifact in a bare process.

`test_artifact.py` starts this script in a process of its own, with
`SWAGE_ARTIFACT_DIR` set and with a copy of the pure `swage` package on the
path. The script runs every case it is given and reports what the process
loaded:

    python artifact_child.py CASES RESULTS [importable]

CASES is a `torch.save` file of `{name: (kind, values, offsets)}` with host
tensors, where kind is a kind of `segment_reduce` or `"softmax"`. A case
of four entries, `(kind, values, offsets, upstream)`, also records a
gradient: its result is `(result, first, second)`, the first derivative
for the upstream gradient and the derivative of the summed squares of the
first derivative with respect to that upstream gradient. RESULTS receives
the host result of every case, the files mapped into the process, and what
the driver launches with.

Without the third argument the script makes `mlir_swage` unimportable
first. With `importable` it leaves the bindings on the path, checks that
the segmented calls did not import them, and then launches the fixed
vector add, which does need them, to show that both run in one process.
"""

import importlib.abc
import os
import re
import sys

# The file names a compiler library of Swage, LLVM, or MLIR is mapped under.
COMPILER_LIBRARY = re.compile(
    r"LLVM|MLIR|mlir|SwagePythonCAPI|swageDialects|nanobind"
)


class _NoNativeBindings(importlib.abc.MetaPathFinder):
    """Refuse every import of `mlir_swage`, wherever it could be found."""

    def find_spec(self, name, path=None, target=None):
        """Raise for `mlir_swage` and its submodules; pass on the rest."""
        if name == "mlir_swage" or name.startswith("mlir_swage."):
            raise ModuleNotFoundError(
                f"{name} is unimportable in this process", name=name
            )
        return None


def mapped_files():
    """Return the files that are mapped into this process."""
    with open("/proc/self/maps", encoding="utf-8") as maps:
        return sorted(
            {
                fields[5]
                for fields in (line.split(None, 5) for line in maps)
                if len(fields) == 6 and fields[5].startswith("/")
            }
        )


def _launch_vector_add(swage, torch):
    """Compile and launch the fixed vector add, which needs the bindings.

    Returns:
        The output and the expected sum, on the host.
    """
    import swage.language as sl

    @swage.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
        pid = sl.program_id(0)
        offsets = pid * BLOCK + sl.arange(0, BLOCK)
        mask = offsets < n
        x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
        sl.store(output_ptr + offsets, x + y, mask=mask)

    count, block = 1025, 64
    x = torch.arange(count, dtype=torch.float32, device="cuda")
    y = torch.full((count,), 0.5, device="cuda")
    output = torch.full((count,), float("nan"), device="cuda")
    add_kernel.launch(
        arguments={"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": count},
        constexprs={"BLOCK": block},
        grid=((count + block - 1) // block,),
    )
    return output.cpu(), (x + y).cpu()


def _compiler_libraries():
    """Return the mapped files that belong to the compiler."""
    return [
        path.strip()
        for path in mapped_files()
        if COMPILER_LIBRARY.search(os.path.basename(path.strip()))
    ]


def main(cases_path, results_path, bindings="blocked"):
    """Run every case from the selected artifact and save the report."""
    if bindings == "blocked":
        sys.meta_path.insert(0, _NoNativeBindings())
    import swage
    import torch
    from swage import _artifact, _runtime

    results = {}
    for name, (kind, values, offsets, *upstream) in torch.load(
        cases_path
    ).items():
        values, offsets = values.cuda(), offsets.cuda()
        if upstream:
            values.requires_grad_()
        if kind == "softmax":
            result = swage.segment_softmax(values, offsets)
        else:
            result = swage.segment_reduce(values, offsets, kind)
        if not upstream:
            results[name] = result.cpu()
            continue
        weight = upstream[0].cuda().requires_grad_()
        (first,) = torch.autograd.grad(
            result, values, weight, create_graph=True
        )
        (second,) = torch.autograd.grad((first * first).sum(), weight)
        results[name] = (
            result.detach().cpu(),
            first.detach().cpu(),
            second.cpu(),
        )

    mapped = [path.strip() for path in mapped_files()]
    compiler = [
        path
        for path in mapped
        if COMPILER_LIBRARY.search(os.path.basename(path))
    ]
    assert "mlir_swage" not in sys.modules, "mlir_swage was imported"
    assert not compiler, f"compiler libraries are mapped: {compiler}"
    launcher = _runtime._get_driver()._native_launch
    report = {
        "results": results,
        "mapped": mapped,
        "swage_file": swage.__file__,
        "path": list(sys.path),
        "modules": sorted(
            name for name in sys.modules if name.startswith("mlir")
        ),
        "launches_with_the_runtime_library": (
            getattr(launcher, "__self__", None) is _artifact.selected()
        ),
    }
    if bindings == "importable":
        # The same process can still compile: the public launch imports the
        # bindings, and the driver keeps the launcher of the artifact.
        output, expected = _launch_vector_add(swage, torch)
        launcher = _runtime._get_driver()._native_launch
        report["vector_add"] = (output, expected)
        report["compiler_mapped_after_the_launch"] = _compiler_libraries()
        report["launcher_after_the_launch_is_the_runtime_library"] = (
            getattr(launcher, "__self__", None) is _artifact.selected()
        )
    torch.save(report, results_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
