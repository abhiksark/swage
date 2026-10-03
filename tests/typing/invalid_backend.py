# tests/typing/invalid_backend.py
"""Reject an unsupported backend through the same decorated public kernel."""

from typing import Any

from fixed_vector_add import add_kernel


def use_unsupported_backend(x: Any, y: Any, output: Any, n: int) -> None:
    """This sole argument-type error is intentional and checked by pytest."""
    add_kernel.launch(
        arguments={"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": n},
        constexprs={"BLOCK": 128},
        grid=((n + 127) // 128,),
        backend="rocm",
    )
