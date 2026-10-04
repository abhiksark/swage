# python/swage/_segmented_programs.py
"""Canonical segment programs of the private runner and the public calls.

A program is MLIR text with one segment function whose parameters carry a
`swage.role`. The compiler numbers the user arguments of every kernel of a
program by the parameter order of that function, so the runner passes its
tensors and counts in the order `_parameter_roles` reads.
"""

import functools
import re

# The element types a reduction admits, as the MLIR type of an element.
_ELEMENTS = ("f32", "f64")
_ROLE = re.compile(r"\{swage\.role = #swage\.role<(\w+)>\}")


def _element_dtype(torch, element):
    """Return the tensor dtype of one element type of `_ELEMENTS`."""
    return {"f32": torch.float32, "f64": torch.float64}[element]


def _element_of(torch, values):
    """Return the element type of a values tensor, or None.

    None stands for a dtype that no reduction admits, and for a `values`
    that is not a tensor. The caller validates both.
    """
    dtype = getattr(values, "dtype", None)
    return {torch.float32: "f32", torch.float64: "f64"}.get(dtype)


def _reduction_kernel(kind, element="f32", rank=1):
    """Return the name of the kernel function of one reduction program.

    The name is `segmented_<kind>[_f64][_r2]`. The f32 programs over
    rank-one values are unsuffixed, which keeps the names and the program
    texts they had before f64 and rank two were admitted.
    """
    return (
        f"segmented_{kind}"
        + ("" if element == "f32" else f"_{element}")
        + ("" if rank == 1 else "_r2")
    )


def _program_element(module_text):
    """Return the element type a segment program declares for its values.

    The private prepared and one-shot launches take a program as text. The
    values and the output a caller passes must have the element type of
    that program, so it is read from the declaration of the values.

    Raises:
        ValueError: The text declares no values argument of an element type
            in `_ELEMENTS`.
    """
    match = re.search(
        r"memref<(?:\?x)+(\w+)> \{swage\.role = #swage\.role<values>\}",
        module_text,
    )
    if match is None or match[1] not in _ELEMENTS:
        raise ValueError(
            "the segment program declares no swage.role<values> argument of "
            "an element type in " + ", ".join(_ELEMENTS)
        )
    return match[1]


@functools.lru_cache(maxsize=128)
def _parameter_roles(module_text):
    """Return the roles of the parameters of a segment function, in order.

    A user argument of a kernel names the parameter it binds by its index
    in the segment function, so the runner orders the tensors and counts it
    passes by these roles. The text holds one segment function, and every
    parameter of it has a role.
    """
    return tuple(_ROLE.findall(module_text))


def _semantic_module(kind, element="f32", rank=1):
    """Return the canonical private qualification module.

    A mean is a sum, the extent of the segment, and one division: the
    reduction kind of its program is `sum`, and the division runs once per
    segment, after the reduction.

    A program over rank-two values reduces one column of one segment per
    instance: `swage.segment_id 1` is the column, `make_segment` binds it,
    and the function declares the number of columns as `feature_count`.

    Args:
        kind: `"sum"`, `"max"`, `"min"`, or `"mean"`.
        element: The element type of the values and the result, `"f32"` or
            `"f64"`.
        rank: The rank of the values and of the output, one or two.
    """
    if kind not in {"sum", "max", "min", "mean"}:
        raise ValueError(
            "reduction kind must be 'sum', 'max', 'min', or 'mean'"
        )
    if element not in _ELEMENTS:
        raise ValueError("reduction element type must be 'f32' or 'f64'")
    if rank not in (1, 2):
        raise ValueError("reduction rank must be one or two")
    reduced, epilogue, stored = kind, "", "%result"
    if kind == "mean":
        reduced, stored = "sum", "%mean"
        epilogue = f"""
    %extent = swage.extent %segment : !swage.segment<{element}>
    %count = arith.index_cast %extent : index to i32
    %divisor = arith.sitofp %count : i32 to {element}
    %mean = arith.divf %result, %divisor : {element}"""
    if rank == 2:
        return f"""
module {{
  func.func @{_reduction_kernel(kind, element, rank)}(
      %values: memref<?x?x{element}> {{swage.role = #swage.role<values>}},
      %offsets: memref<?xi32> {{swage.role = #swage.role<offsets>}},
      %output: memref<?x?x{element}> {{swage.role = #swage.role<output>}},
      %value_count: i32 {{swage.role = #swage.role<value_count>}},
      %segment_count: i32 {{swage.role = #swage.role<segment_count>}},
      %feature_count: i32 {{swage.role = #swage.role<feature_count>}}) {{
    %sid = swage.segment_id 0
    %column = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid column(%column)
        : memref<?x?x{element}>, memref<?xi32>, index, index
          -> !swage.segment<{element}>
    %result = swage.reduce %segment kind<{reduced}>
        : !swage.segment<{element}> -> {element} {{
    ^bb0(%value: {element}):
      swage.yield %value : {element}
    }}{epilogue}
    memref.store {stored}, %output[%sid, %column] : memref<?x?x{element}>
    return
  }}
}}
"""
    return f"""
module {{
  func.func @{_reduction_kernel(kind, element)}(
      %values: memref<?x{element}> {{swage.role = #swage.role<values>}},
      %offsets: memref<?xi32> {{swage.role = #swage.role<offsets>}},
      %output: memref<?x{element}> {{swage.role = #swage.role<output>}},
      %value_count: i32 {{swage.role = #swage.role<value_count>}},
      %segment_count: i32 {{swage.role = #swage.role<segment_count>}}) {{
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?x{element}>, memref<?xi32>, index -> !swage.segment<{element}>
    %result = swage.reduce %segment kind<{reduced}>
        : !swage.segment<{element}> -> {element} {{
    ^bb0(%value: {element}):
      swage.yield %value : {element}
    }}{epilogue}
    memref.store {stored}, %output[%sid] : memref<?x{element}>
    return
  }}
}}
"""


def _softmax_text(rank):
    """Return the softmax program over values of rank one or two.

    Over rank-two values a program instance is one segment and one column,
    as for a reduction: the softmax normalizes each column over the rows of
    its segment and writes it to the same rows and column of the output.
    """
    rows = "memref<?xf32>" if rank == 1 else "memref<?x?xf32>"
    name = "ragged_softmax" if rank == 1 else "ragged_softmax_r2"
    features = column = bound = index = ""
    if rank == 2:
        features = (
            ",\n      %feature_count: i32 "
            "{swage.role = #swage.role<feature_count>}"
        )
        column = "\n    %column = swage.segment_id 1"
        bound, index = " column(%column)", ", index"
    closing = features + ") {"
    return f"""
module {{
  func.func @{name}(
      %values: {rows} {{swage.role = #swage.role<values>}},
      %offsets: memref<?xi32> {{swage.role = #swage.role<offsets>}},
      %output: {rows} {{swage.role = #swage.role<output>}},
      %value_count: i32 {{swage.role = #swage.role<value_count>}},
      %segment_count: i32 {{swage.role = #swage.role<segment_count>}}{closing}
    %sid = swage.segment_id 0{column}
    %segment = swage.make_segment %values, %offsets, %sid{bound}
        : {rows}, memref<?xi32>, index{index} -> !swage.segment<f32>
    %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {{
    ^bb0(%value: f32):
      swage.yield %value : f32
    }}
    %shifted = swage.map %segment captures(%max : f32)
        : !swage.segment<f32> -> !swage.segment<f32> {{
    ^bb0(%value: f32, %m: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      swage.yield %exponential : f32
    }}
    %total = swage.reduce %shifted kind<sum> : !swage.segment<f32> -> f32 {{
    ^bb0(%element: f32):
      swage.yield %element : f32
    }}
    swage.map_store %segment, %output captures(%max, %total : f32, f32)
        : !swage.segment<f32>, {rows} {{
    ^bb0(%value: f32, %m: f32, %t: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      %normalized = arith.divf %exponential, %t : f32
      swage.yield %normalized : f32
    }}
    return
  }}
}}
"""


# The rank-one program, whose text every artifact and digest records.
_SOFTMAX_MODULE = _softmax_text(1)
