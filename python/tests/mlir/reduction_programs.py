# python/tests/mlir/reduction_programs.py
"""Semantic programs shared by static codegen and runtime qualification."""

from swage._segmented_programs import _reduction_kernel, _semantic_module


def reduction_module(kind, transform, element="f32"):
    """Build simple and compute-heavy native reduction qualification IR.

    Args:
        kind: The reduction kind.
        transform: The element program, by name.
        element: The element type, `"f32"` or `"f64"`. The f64 program is
            the f32 program with every f32 replaced, under the kernel name
            `_reduction_kernel(kind, "f64")`.
    """
    if element != "f32":
        return (
            reduction_module(kind, transform)
            .replace("f32", element)
            .replace(
                f"@{_reduction_kernel(kind)}(",
                f"@{_reduction_kernel(kind, element)}(",
            )
        )
    module = _semantic_module(kind)
    if transform in (
        "exp2",
        "exp2_chain",
        "exp2_pair",
        "rational4",
        "rational8",
        "affine4",
        "affine16",
        "affine32",
    ):
        lines = [
            "%half = arith.constant -0.5 : f32",
            "%quarter = arith.constant 0.25 : f32",
            "%eighth = arith.constant 0.125 : f32",
            "%one = arith.constant 1.0 : f32",
        ]
        value = "%value"
        steps = {
            "exp2": 1,
            "exp2_pair": 2,
            "rational4": 4,
            "affine4": 2,
            "affine16": 8,
            "affine32": 16,
        }.get(transform, 8)
        for index in range(steps):
            result = f"%result{index}"
            if transform == "exp2":
                lines.append(f"{result} = math.exp2 {value} : f32")
            elif transform in ("exp2_chain", "exp2_pair"):
                lines.extend(
                    [
                        f"%scaled{index} = arith.mulf {value}, %half : f32",
                        f"{result} = math.exp2 %scaled{index} : f32",
                    ]
                )
            elif transform.startswith("affine"):
                lines.extend(
                    [
                        f"%scaled{index} = arith.mulf {value}, %half : f32",
                        f"{result} = arith.addf %scaled{index}, %eighth : f32",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"%square{index} = arith.mulf {value}, {value} : f32",
                        f"%scaled{index} = arith.mulf %square{index}, "
                        "%quarter : f32",
                        f"%denom{index} = arith.addf %scaled{index}, "
                        "%one : f32",
                        f"%numer{index} = arith.addf {value}, %eighth : f32",
                        f"{result} = arith.divf %numer{index}, "
                        f"%denom{index} : f32",
                    ]
                )
            value = result
        lines.append(f"swage.yield {value} : f32")
        return module.replace(
            "swage.yield %value : f32", "\n      ".join(lines)
        )
    if transform == "square":
        return module.replace(
            "swage.yield %value : f32",
            "%square = arith.mulf %value, %value : f32\n"
            "      swage.yield %square : f32",
        )
    if transform == "maps":
        return module.replace(
            "%result = swage.reduce %segment",
            """%shifted = swage.map %segment
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32):
      %one = arith.constant 1.0 : f32
      %shift = arith.addf %value, %one : f32
      swage.yield %shift : f32
    }
    %mapped = swage.map %shifted
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32):
      %two = arith.constant 2.0 : f32
      %scaled = arith.mulf %value, %two : f32
      swage.yield %scaled : f32
    }
    %result = swage.reduce %mapped""",
        )
    assert transform == "identity"
    return module
