# tests/typing/fixed_vector_add.py
"""The canonical fixed kernel and both explicit supported launch backends."""

from typing import Any

import swage as sw
import swage.language as sl


@sw.jit
def add_kernel(
    x_ptr: Any,
    y_ptr: Any,
    output_ptr: Any,
    n: int,
    BLOCK: sl.constexpr,
) -> None:
    """Add bounded vectors using the public symbolic DSL."""
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


def use_fixed_kernel(x: Any, y: Any, output: Any, n: int) -> None:
    """Type-check without importing Torch or executing a native kernel."""
    arguments = {"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": n}
    constexprs = {"BLOCK": 128}
    grid = ((n + 127) // 128,)
    add_kernel.emit_mlir(arguments=arguments, constexprs=constexprs)
    for dtype in (sl.float32, sl.float16, sl.float8_e4m3fn, sl.float8_e5m2):
        add_kernel.emit_mlir(
            signature={
                "x_ptr": sl.pointer(dtype),
                "y_ptr": sl.pointer(dtype),
                "output_ptr": sl.pointer(dtype),
                "n": sl.int32,
            },
            constexprs=constexprs,
        )
    add_kernel.launch(
        arguments=arguments,
        constexprs=constexprs,
        grid=grid,
        backend="cpu",
    )
    add_kernel.launch(
        arguments=arguments,
        constexprs=constexprs,
        grid=grid,
        backend="cuda",
    )
