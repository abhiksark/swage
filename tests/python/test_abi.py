# tests/python/test_abi.py
"""LLVM-free tests for compiler-generated kernel contract binding."""

import json
import math
from dataclasses import FrozenInstanceError

import pytest
from swage import _abi


def _contract(
    arguments,
    *,
    entry="kernel",
    backend="cuda",
    block=(128, 1, 1),
):
    launch = {"model": "host-call"}
    if backend == "cuda":
        launch = {"model": "spmd-grid", "block": list(block)}
    return json.dumps(
        {
            "version": 2,
            "backend": backend,
            "entry": entry,
            "launch": launch,
            "arguments": arguments,
        },
        separators=(",", ":"),
    )


def _pointer(origin, *, source_index=None, key=None, access="read"):
    argument = {"kind": "ptr", "origin": origin}
    if source_index is not None:
        argument["source_index"] = source_index
    if key is not None:
        argument["key"] = key
    argument["access"] = access
    return argument


def _scalar(kind, origin, *, source_index=None, key=None):
    argument = {"kind": kind, "origin": origin}
    if source_index is not None:
        argument["source_index"] = source_index
    if key is not None:
        argument["key"] = key
    return argument


@pytest.mark.parametrize("backend", ["cuda", "cpu"])
def test_contract_round_trip_is_canonical_and_immutable(backend):
    """Retain both launch forms in values that callers cannot mutate."""
    raw = _contract(
        [
            _pointer("user", source_index=0),
            _scalar("i32", "derived", key="value_count"),
        ],
        backend=backend,
    )

    contract = _abi.parse_kernel_contract(raw)

    assert _abi.serialize_kernel_contract(contract) == raw
    assert contract.backend == backend
    assert contract.launch.block == ((128, 1, 1) if backend == "cuda" else None)
    assert contract.arguments[1].key == "value_count"
    with pytest.raises(FrozenInstanceError):
        setattr(contract, "entry", "other")


@pytest.mark.parametrize(
    "raw",
    [
        _contract([]).replace('"version":2', '"version":1'),
        _contract([]).replace('"arguments":[]', '"arguments":[],"extra":1'),
        _contract([_scalar("u64", "user", source_index=0)]),
        _contract([_scalar("i32", "unknown", key="n")]),
        _contract([_pointer("user", source_index=0)]) + " ",
        _contract(
            [
                _pointer("user", source_index=0),
                _pointer("user", source_index=0),
            ]
        ),
        _contract(
            [
                _scalar("i32", "derived", key="count"),
                _scalar("i32", "plan", key="count"),
            ]
        ),
        _contract([], backend="cpu").replace("host-call", "spmd-grid"),
        _contract([]).replace("spmd-grid", "host-call"),
        _contract([]).replace('"block":[128,1,1]', '"block":[128,1]'),
    ],
)
def test_contract_parser_rejects_ambiguous_schema(raw):
    """Reject schema drift instead of guessing a physical ABI."""
    with pytest.raises(ValueError):
        _abi.parse_kernel_contract(raw)


def test_binding_preserves_contract_order_and_defers_pointer_reads():
    """Resolve every origin before crossing any tensor pointer boundary."""
    reads = []

    class _Tensor:
        def __init__(self, pointer):
            self.pointer = pointer

        def data_ptr(self):
            reads.append(self.pointer)
            return self.pointer

    raw = _contract(
        [
            _pointer("plan", key="tasks"),
            _pointer("user", source_index=1, access="write"),
            _scalar("i32", "derived", key="task_count"),
            _pointer("scratch", key="partial", access="readwrite"),
            _pointer("user", source_index=0),
        ]
    )
    contract = _abi.parse_kernel_contract(raw)
    bound = _abi.bind_kernel_contract(
        contract,
        user=(_Tensor(0x10), _Tensor(0x20)),
        derived={"task_count": lambda: 7},
        plan={"tasks": _Tensor(0x30)},
        scratch={"partial": _Tensor(0x40)},
    )

    assert reads == []
    assert _abi.materialize_launch_arguments(bound) == (
        ("ptr", "ptr", "i32", "ptr", "ptr"),
        (0x30, 0x20, 7, 0x40, 0x10),
    )
    assert reads == [0x30, 0x20, 0x40, 0x10]


def test_parser_mirrors_compiler_string_and_numeric_bounds():
    """Accept labels and structural geometry while enforcing integer bounds."""
    parsed = _abi.parse_kernel_contract(
        _contract(
            [
                {
                    "kind": "ptr",
                    "origin": "plan",
                    "key": "plan-buffer",
                    "access": "read",
                }
            ],
            entry="kernel.with.dot",
            block=(1025, 1, 1),
        )
    )
    assert parsed.entry == "kernel.with.dot"
    assert parsed.arguments[0].key == "plan-buffer"
    assert parsed.launch.block == (1025, 1, 1)

    too_large_source = _contract([_scalar("i32", "user", source_index=1 << 32)])
    with pytest.raises(ValueError, match="nonnegative u32"):
        _abi.parse_kernel_contract(too_large_source)


def test_binding_supports_sparse_semantic_source_indexes():
    """Bind a split entry that consumes only semantic output index two."""

    class _Tensor:
        def data_ptr(self):
            return 0x30

    contract = _abi.parse_kernel_contract(
        _contract([_pointer("user", source_index=2, access="write")])
    )

    bound = _abi.bind_kernel_contract(
        contract, user=(object(), object(), _Tensor())
    )

    assert _abi.materialize_launch_arguments(bound) == (("ptr",), (0x30,))
    with pytest.raises(ValueError, match="referenced source index"):
        _abi.bind_kernel_contract(contract, user=(object(),))


def test_binding_mismatch_precedes_derived_and_pointer_reads():
    """Report missing or extra bindings without touching supplied values."""
    touched = []
    contract = _abi.parse_kernel_contract(
        _contract(
            [
                _pointer("user", source_index=0),
                _scalar("i32", "derived", key="count"),
            ]
        )
    )

    with pytest.raises(ValueError, match="extra unexpected"):
        _abi.bind_kernel_contract(
            contract,
            user=(object(),),
            derived={"count": lambda: touched.append(True), "unexpected": 1},
        )

    assert touched == []


def test_materialization_maps_every_scalar_to_canonical_raw_bits():
    """Use one unsigned bit-pattern convention for every physical scalar."""
    arguments = (
        _abi.BoundArgument("i1", 1),
        _abi.BoundArgument("i8", 0xFF),
        _abi.BoundArgument("i16", 0xFFFF),
        _abi.BoundArgument("i32", 0xFFFFFFFF),
        _abi.BoundArgument("i64", 0xFFFFFFFFFFFFFFFF),
        _abi.BoundArgument("f16", 1.5),
        _abi.BoundArgument("bf16", 1.5),
        _abi.BoundArgument("f32", 1.5),
        _abi.BoundArgument("f64", 1.5),
    )

    kinds, values = _abi.materialize_launch_arguments(arguments)

    assert kinds == (
        "i1",
        "i8",
        "i16",
        "i32",
        "i64",
        "f16",
        "bf16",
        "f32",
        "f64",
    )
    assert values == (
        1,
        0xFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x3E00,
        0x3FC0,
        0x3FC00000,
        0x3FF8000000000000,
    )


def test_bf16_rounds_ties_to_even_and_quiets_nan():
    """Keep bfloat conversion deterministic at rounding and NaN edges."""
    _, rounded = _abi.materialize_launch_arguments(
        (
            _abi.BoundArgument("bf16", float.fromhex("0x1.01p+0")),
            _abi.BoundArgument("bf16", math.nan),
        )
    )
    assert rounded[0] == 0x3F80
    assert rounded[1] & 0x7F80 == 0x7F80
    assert rounded[1] & 0x0040


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("i1", True),
        ("i1", -1),
        ("i1", 2),
        ("i8", 1 << 8),
        ("i16", 1 << 16),
        ("i32", 1 << 32),
        ("i64", 1 << 64),
        ("f16", 1),
        ("f16", 1e100),
        ("bf16", 1),
        ("bf16", 1e100),
        ("f32", 1),
        ("f32", 1e100),
        ("f64", 1),
    ],
)
def test_materialization_rejects_wrong_types_and_overflow(kind, value):
    """Reject values outside each exact physical scalar domain."""
    with pytest.raises((TypeError, ValueError), match=rf"{kind} argument"):
        _abi.materialize_launch_arguments((_abi.BoundArgument(kind, value),))
