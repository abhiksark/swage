# examples/fixed_vector_add.py
"""Emit semantic MLIR and execute the M3 fixed vector-add kernel."""

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
    """Print semantic MLIR, launch on CUDA, and verify the result."""
    n = 1025
    block = 128
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
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
    )
    torch.testing.assert_close(output, torch.add(x, y))
    print("=== CUDA result matches torch.add ===")


if __name__ == "__main__":
    main()
