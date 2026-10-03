# examples/fixed_vector_multiply.py
"""Emit MLIR and execute the canonical fixed vector-multiply kernel."""

import argparse

import swage as sw
import swage.language as sl
import torch


@sw.jit
def multiply_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
    """Multiply two vectors elementwise under a bounds mask."""
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x * y, mask=mask)


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
    dtype = getattr(torch, args.dtype)
    n, block = 1025, 128
    x = torch.randn(n, device=args.backend, dtype=torch.float32).to(dtype)
    y = torch.randn(n, device=args.backend, dtype=torch.float32).to(dtype)
    output = torch.empty_like(x)
    arguments = {"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": n}

    print("=== Semantic MLIR ===")
    print(
        multiply_kernel.emit_mlir(
            arguments=arguments, constexprs={"BLOCK": block}
        )
    )
    multiply_kernel.launch(
        arguments=arguments,
        constexprs={"BLOCK": block},
        grid=((n + block - 1) // block,),
        backend=args.backend,
    )
    if args.backend == "cuda":
        torch.cuda.synchronize()
    expected = (x.float() * y.float()).to(dtype)
    torch.testing.assert_close(
        output.float(), expected.float(), rtol=0, atol=0, equal_nan=True
    )
    print(
        f"=== {args.backend.upper()} {args.dtype} result matches "
        "FP32 multiplication rounded to the output dtype ==="
    )


if __name__ == "__main__":
    main()
