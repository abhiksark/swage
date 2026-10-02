# python/tests/mlir/artifact_child.py
"""Run the public segmented calls from an artifact in a bare process.

`test_artifact.py` starts this script in a process of its own, with
`SWAGE_ARTIFACT_DIR` set and with a copy of the pure `swage` package as the
only Swage code on the path. The script makes `mlir_swage` unimportable,
runs every case it is given, and reports what the process loaded:

    python artifact_child.py CASES RESULTS

CASES is a `torch.save` file of `{name: (kind, values, offsets)}` with host
tensors, where kind is `"sum"`, `"max"`, or `"softmax"`. RESULTS receives
the host result of every case, the files mapped into the process, and what
the driver launches with.
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


def main(cases_path, results_path):
    """Run every case from the selected artifact and save the report."""
    sys.meta_path.insert(0, _NoNativeBindings())
    import swage
    import torch
    from swage import _artifact, _runtime

    results = {}
    for name, (kind, values, offsets) in torch.load(cases_path).items():
        values, offsets = values.cuda(), offsets.cuda()
        if kind == "softmax":
            result = swage.segment_softmax(values, offsets)
        else:
            result = swage.segment_reduce(values, offsets, kind)
        results[name] = result.cpu()

    mapped = [path.strip() for path in mapped_files()]
    compiler = [
        path
        for path in mapped
        if COMPILER_LIBRARY.search(os.path.basename(path))
    ]
    assert "mlir_swage" not in sys.modules, "mlir_swage was imported"
    assert not compiler, f"compiler libraries are mapped: {compiler}"
    launcher = _runtime._get_driver()._native_launch
    torch.save(
        {
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
        },
        results_path,
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
