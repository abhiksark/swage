# examples/fixed_vector_add.py
"""Emit semantic MLIR and execute the canonical fixed vector-add kernel."""

import argparse

import swage as sw
import swage.language as sl
import torch


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
    """Add two vectors elementwise under a bounds mask."""
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


def main():
    """Print MLIR and verify the explicitly selected CPU or CUDA backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "float8_e4m3fn", "float8_e5m2"),
        default="float32",
    )
    args = parser.parse_args()
    backend = args.backend
    dtype = getattr(torch, args.dtype)
    n = 1025
    block = 128
    x = torch.randn(n, device=backend, dtype=torch.float32).to(dtype)
    y = torch.randn(n, device=backend, dtype=torch.float32).to(dtype)
    output = torch.empty_like(x)
    arguments = {
        "x_ptr": x,
        "y_ptr": y,
        "output_ptr": output,
        "n": n,
    }

    module = add_kernel.emit_mlir(
        arguments=arguments,
        constexprs={"BLOCK": block},
    )
    print("=== Semantic MLIR ===")
    print(module)

    add_kernel.launch(
        arguments=arguments,
        constexprs={"BLOCK": block},
        grid=((n + block - 1) // block,),
        backend=backend,
    )
    if backend == "cuda":
        torch.cuda.synchronize()
    expected = (x.float() + y.float()).to(dtype)
    torch.testing.assert_close(
        output.float(), expected.float(), rtol=0, atol=0, equal_nan=True
    )
    print(
        f"=== {backend.upper()} {args.dtype} result matches "
        "FP32 addition rounded to the output dtype ==="
    )


if __name__ == "__main__":
    main()
