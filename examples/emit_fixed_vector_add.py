# examples/emit_fixed_vector_add.py
"""Emit semantic MLIR for the canonical fixed vector-add kernel.

This example stops after compile-only emission. It needs the native build
and nothing else: no GPU and no PyTorch. The explicit signature replaces the
tensor metadata that `examples/fixed_vector_add.py` infers from CUDA tensors.
"""

import swage as sw
import swage.language as sl


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
    """Print the verified semantic MLIR module."""
    module = add_kernel.emit_mlir(
        signature={
            "x_ptr": sl.pointer(sl.float32),
            "y_ptr": sl.pointer(sl.float32),
            "output_ptr": sl.pointer(sl.float32),
            "n": sl.int32,
        },
        constexprs={"BLOCK": 128},
    )
    print("=== Semantic MLIR ===")
    print(module)


if __name__ == "__main__":
    main()
