# scripts/smoke_installed_wheel.py
"""Exercise a native wheel outside its checkout, including cache reuse."""

import argparse
import json
import logging
import os
import pathlib
import sysconfig

import swage as sw
import swage.language as sl
from swage import env


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
    """Add equal-dtype vectors without reading or writing beyond the mask."""
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


@sw.jit
def multiply_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
    """Multiply equal-dtype vectors without accessing masked-off elements."""
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x * y, mask=mask)


class _CacheEvents(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.counts = {"memory-hit": 0, "persistent-hit": 0, "compile": 0}

    def emit(self, record):
        event = record.getMessage().split()[0]
        if event in self.counts:
            self.counts[event] += 1


def main(argv=None):
    """Verify installed paths, provenance, emission, results and cache reuse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expected-revision")
    parser.add_argument("--require-persistent-hit", action="store_true")
    args = parser.parse_args(argv)
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("installed-wheel smoke requires PYTHONPATH unset")
    checkout = pathlib.Path(__file__).resolve().parents[1]
    if (
        checkout / "pyproject.toml"
    ).is_file() and pathlib.Path.cwd().is_relative_to(checkout):
        raise RuntimeError("run installed-wheel smoke outside the checkout")
    # These imports must resolve from the wheel, never from a build tree.
    from mlir_swage import ir

    sites = {
        pathlib.Path(sysconfig.get_path(key)).resolve()
        for key in ("purelib", "platlib")
    }
    paths = {"swage": sw.__file__, "mlir_swage": ir.__file__}
    for name, filename in paths.items():
        if not any(
            pathlib.Path(filename).resolve().is_relative_to(p) for p in sites
        ):
            raise RuntimeError(
                f"{name} did not import from this environment's wheel"
            )
    report = env.report()
    native = report["native"]
    if not native["available"] or native["error"]:
        raise RuntimeError("native wheel health failed")
    if native["package_version"] != sw.__version__:
        raise RuntimeError("native and Python package versions differ")
    if native["llvm_version"] != "llvmorg-22.1.8":
        raise RuntimeError("native wheel does not identify the pinned LLVM")
    if (
        args.expected_revision
        and native["source_revision"] != args.expected_revision
    ):
        raise RuntimeError("native wheel source revision differs from expected")
    if not report["backends"][args.backend]["available"]:
        raise RuntimeError(report["backends"][args.backend]["reason"])
    if args.require_persistent_hit and args.backend != "cuda":
        raise ValueError("only CUDA artifacts have a persistent cache")
    import torch

    observer = _CacheEvents()
    logger = logging.getLogger("swage.runtime")
    old_level = logger.level
    logger.addHandler(observer)
    logger.setLevel(logging.DEBUG)
    sizes = (0, 1, 127, 128, 129, 4097)
    dtype_names = ("float32", "float16", "float8_e4m3fn", "float8_e5m2")
    operations = (
        ("add", add_kernel, lambda x, y: x.float() + y.float()),
        ("multiply", multiply_kernel, lambda x, y: x.float() * y.float()),
    )
    cache_coverage = {}
    try:
        for operation_name, kernel, reference in operations:
            for dtype_name in dtype_names:
                before = observer.counts.copy()
                dtype = getattr(torch, dtype_name)
                for n in sizes:
                    values = torch.arange(
                        n, device=args.backend, dtype=torch.float32
                    ) % 13
                    x = values.to(dtype)
                    y = (values * 0.5).to(dtype)
                    output = torch.empty_like(x)
                    expected = reference(x, y).to(dtype)
                    arguments = {
                        "x_ptr": x,
                        "y_ptr": y,
                        "output_ptr": output,
                        "n": n,
                    }
                    module = kernel.emit_mlir(
                        arguments=arguments,
                        constexprs={"BLOCK": 128},
                    )
                    if not module.operation.verify():
                        raise RuntimeError(
                            "wheel emitted invalid semantic MLIR"
                        )
                    for _ in range(2):
                        # All-one bits encode a NaN in every supported dtype.
                        output.view(torch.uint8).fill_(0xFF)
                        kernel.launch(
                            arguments=arguments,
                            constexprs={"BLOCK": 128},
                            grid=((n + 127) // 128,),
                            backend=args.backend,
                        )
                        if args.backend == "cuda":
                            torch.cuda.synchronize()
                        torch.testing.assert_close(
                            output.float(),
                            expected.float(),
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                        )
                events = {
                    name: observer.counts[name] - before[name]
                    for name in observer.counts
                }
                cache_coverage[f"{operation_name}:{dtype_name}"] = events
                if events["memory-hit"] == 0:
                    raise RuntimeError(
                        f"{operation_name} {dtype_name} did not reuse the "
                        "process artifact"
                    )
                if args.require_persistent_hit and (
                    events["persistent-hit"] == 0 or events["compile"]
                ):
                    raise RuntimeError(
                        f"{operation_name} {dtype_name} did not reuse the "
                        "persistent artifact"
                    )
    finally:
        logger.removeHandler(observer)
        logger.setLevel(old_level)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "backend": args.backend,
                "native": native,
                "sizes": sizes,
                "dtypes": dtype_names,
                "operations": tuple(name for name, _, _ in operations),
                "cache_events": observer.counts,
                "cache_coverage": cache_coverage,
                "passed": True,
                "imports": paths,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
