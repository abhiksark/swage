# python/swage/_abi.py
"""Strict internal representation and binding for compiled kernel contracts."""

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_VERSION = 2
_BACKENDS = frozenset({"cuda", "cpu"})
_LAUNCH_MODELS = frozenset({"spmd-grid", "host-call"})
_KINDS = frozenset(
    {"ptr", "i1", "i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64"}
)
_INTEGER_WIDTHS = {"i1": 1, "i8": 8, "i16": 16, "i32": 32, "i64": 64}
_FLOAT_FORMATS = {"f16": ("e", "H"), "f32": ("f", "I"), "f64": ("d", "Q")}
_ORIGINS = frozenset({"user", "derived", "plan", "scratch"})
_ACCESS = frozenset({"read", "write", "readwrite"})
_I32_MAX = (1 << 31) - 1
_U64_MAX = (1 << 64) - 1
_U32_MAX = (1 << 32) - 1


@dataclass(frozen=True, slots=True)
class KernelArgument:
    """One ordered physical kernel argument."""

    kind: str
    origin: str
    source_index: int | None = None
    key: str | None = None
    access: str | None = None

    def to_json(self) -> dict[str, object]:
        """Return this argument in canonical field order."""
        value: dict[str, object] = {
            "kind": self.kind,
            "origin": self.origin,
        }
        if self.source_index is not None:
            value["source_index"] = self.source_index
        if self.key is not None:
            value["key"] = self.key
        if self.access is not None:
            value["access"] = self.access
        return value


@dataclass(frozen=True, slots=True)
class KernelLaunch:
    """Backend-specific physical launch model."""

    model: str
    block: tuple[int, int, int] | None = None

    def to_json(self) -> dict[str, object]:
        """Return this launch in canonical field order."""
        value: dict[str, object] = {"model": self.model}
        if self.block is not None:
            value["block"] = list(self.block)
        return value


@dataclass(frozen=True, slots=True)
class KernelContract:
    """Versioned physical launch contract emitted by the compiler."""

    version: int
    backend: str
    entry: str
    launch: KernelLaunch
    arguments: tuple[KernelArgument, ...]

    def to_json(self) -> dict[str, object]:
        """Return this contract in canonical field order."""
        return {
            "version": self.version,
            "backend": self.backend,
            "entry": self.entry,
            "launch": self.launch.to_json(),
            "arguments": [argument.to_json() for argument in self.arguments],
        }


@dataclass(frozen=True, slots=True)
class BoundArgument:
    """One contract-tagged value awaiting native ABI materialization."""

    kind: str
    value: object


def _require_exact_fields(value, expected, label):
    fields = set(value)
    if fields != expected:
        missing = sorted(expected - fields)
        unknown = sorted(fields - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise ValueError(f"{label} fields are invalid: {'; '.join(details)}")


def _parse_argument(value, index):
    label = f"contract argument {index}"
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    kind = value.get("kind")
    origin = value.get("origin")
    if kind not in _KINDS:
        raise ValueError(f"{label} has unknown kind {kind!r}")
    if origin not in _ORIGINS:
        raise ValueError(f"{label} has unknown origin {origin!r}")

    binding_field = "source_index" if origin == "user" else "key"
    expected = {"kind", "origin", binding_field}
    if kind == "ptr":
        expected.add("access")
    _require_exact_fields(value, expected, label)

    source_index = value.get("source_index")
    key = value.get("key")
    access = value.get("access")
    if origin == "user":
        if (
            type(source_index) is not int
            or source_index < 0
            or source_index > _U32_MAX
        ):
            raise ValueError(f"{label} source_index must be a nonnegative u32")
    elif not isinstance(key, str) or not key:
        raise ValueError(f"{label} key must be a nonempty string")
    if kind == "ptr" and access not in _ACCESS:
        raise ValueError(f"{label} has unknown access {access!r}")
    return KernelArgument(kind, origin, source_index, key, access)


def _parse_launch(value):
    if not isinstance(value, dict):
        raise ValueError("contract launch must be an object")
    model = value.get("model")
    if model not in _LAUNCH_MODELS:
        raise ValueError(f"contract launch has unknown model {model!r}")
    if model == "host-call":
        _require_exact_fields(value, {"model"}, "contract launch")
        return KernelLaunch(model)

    _require_exact_fields(value, {"model", "block"}, "contract launch")
    block = value["block"]
    if (
        not isinstance(block, list)
        or len(block) != 3
        or any(
            type(axis) is not int or axis <= 0 or axis > _I32_MAX
            for axis in block
        )
    ):
        raise ValueError("contract block must contain three positive i32s")
    return KernelLaunch(model, tuple(block))


def parse_kernel_contract(raw):
    """Parse canonical JSON and reject any schema ambiguity."""
    if not isinstance(raw, str):
        raise TypeError("kernel contract must be a JSON string")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("kernel contract is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("kernel contract must be an object")
    _require_exact_fields(
        value,
        {"version", "backend", "entry", "launch", "arguments"},
        "contract",
    )
    if type(value["version"]) is not int or value["version"] != _VERSION:
        raise ValueError(
            f"unsupported kernel contract version {value['version']!r}"
        )
    backend = value["backend"]
    if backend not in _BACKENDS:
        raise ValueError(f"contract has unknown backend {backend!r}")
    entry = value["entry"]
    if not isinstance(entry, str) or not entry:
        raise ValueError("contract entry must be a nonempty string")
    launch = _parse_launch(value["launch"])
    if backend == "cuda" and launch.model != "spmd-grid":
        raise ValueError("CUDA contract requires spmd-grid launch")
    if backend == "cpu" and launch.model != "host-call":
        raise ValueError("CPU contract requires host-call launch")
    arguments_value = value["arguments"]
    if not isinstance(arguments_value, list):
        raise ValueError("contract arguments must be an array")
    arguments = tuple(
        _parse_argument(argument, index)
        for index, argument in enumerate(arguments_value)
    )

    source_indexes = [
        argument.source_index
        for argument in arguments
        if argument.origin == "user"
    ]
    if len(set(source_indexes)) != len(source_indexes):
        raise ValueError("contract contains duplicate user source indexes")
    keys = [argument.key for argument in arguments if argument.key is not None]
    if len(set(keys)) != len(keys):
        raise ValueError("contract contains duplicate binding keys")

    contract = KernelContract(
        version=value["version"],
        backend=backend,
        entry=entry,
        launch=launch,
        arguments=arguments,
    )
    if serialize_kernel_contract(contract) != raw:
        raise ValueError("kernel contract JSON is not canonical")
    return contract


def serialize_kernel_contract(contract):
    """Serialize a contract deterministically without insignificant spaces."""
    if not isinstance(contract, KernelContract):
        raise TypeError("contract must be a KernelContract")
    return json.dumps(contract.to_json(), separators=(",", ":"))


def kernel_contract_digest(raw):
    """Return the SHA-256 digest of canonical contract JSON."""
    parse_kernel_contract(raw)
    return hashlib.sha256(raw.encode()).hexdigest()


def _binding_map(value, origin):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{origin} bindings must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{origin} binding keys must be strings")
    return value


def bind_kernel_contract(
    contract, *, user, derived=None, plan=None, scratch=None
):
    """Bind all contract origins without reading tensor data pointers."""
    if not isinstance(contract, KernelContract):
        raise TypeError("contract must be a KernelContract")
    if not isinstance(user, Sequence) or isinstance(user, (str, bytes)):
        raise TypeError("user bindings must be an ordered sequence")
    source_indexes = []
    for argument in contract.arguments:
        if argument.origin == "user":
            if argument.source_index is None:
                raise ValueError("user contract argument has no source index")
            source_indexes.append(argument.source_index)
    if source_indexes and len(user) <= max(source_indexes):
        raise ValueError(
            "user bindings do not cover every referenced source index"
        )
    maps = {
        "derived": _binding_map(derived, "derived"),
        "plan": _binding_map(plan, "plan"),
        "scratch": _binding_map(scratch, "scratch"),
    }
    expected = {origin: set() for origin in maps}
    for argument in contract.arguments:
        if argument.origin != "user":
            expected[argument.origin].add(argument.key)
    for origin, bindings in maps.items():
        missing = expected[origin] - set(bindings)
        extra = set(bindings) - expected[origin]
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {', '.join(sorted(missing))}")
            if extra:
                details.append(f"extra {', '.join(sorted(extra))}")
            raise ValueError(
                f"{origin} bindings do not match contract: {'; '.join(details)}"
            )

    bound = []
    for argument in contract.arguments:
        if argument.origin == "user":
            if argument.source_index is None:
                raise ValueError("user contract argument has no source index")
            value = user[argument.source_index]
        else:
            value = maps[argument.origin][argument.key]
            if argument.origin == "derived" and callable(value):
                value = value()
        bound.append(BoundArgument(argument.kind, value))
    return tuple(bound)


def _materialize_float(kind, value, index):
    if type(value) is not float:
        raise TypeError(f"{kind} argument {index} must be an exact float")
    if kind == "bf16":
        try:
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
        except OverflowError as error:
            raise ValueError(
                f"bf16 argument {index} is out of range"
            ) from error
        if bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF:
            return (bits >> 16) | 0x0040
        return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16

    pack_format, unpack_format = _FLOAT_FORMATS[kind]
    try:
        packed = struct.pack(f"<{pack_format}", value)
    except OverflowError as error:
        raise ValueError(f"{kind} argument {index} is out of range") from error
    return struct.unpack(f"<{unpack_format}", packed)[0]


def materialize_launch_arguments(arguments):
    """Convert validated bound values into ordered native raw bit patterns."""
    kinds = []
    values = []
    for index, argument in enumerate(arguments):
        if not isinstance(argument, BoundArgument):
            raise TypeError(f"bound argument {index} must be a BoundArgument")
        value = argument.value
        if argument.kind == "ptr":
            if type(value) is not int:
                data_ptr = getattr(value, "data_ptr", None)
                if not callable(data_ptr):
                    raise TypeError(
                        f"pointer argument {index} must be an integer or tensor"
                    )
                value = data_ptr()
            if type(value) is not int or not 0 <= value <= _U64_MAX:
                raise ValueError(f"pointer argument {index} must be a u64")
        elif argument.kind in _INTEGER_WIDTHS:
            width = _INTEGER_WIDTHS[argument.kind]
            if type(value) is not int or not 0 <= value < 1 << width:
                raise ValueError(
                    f"{argument.kind} argument {index} is out of range"
                )
        elif argument.kind == "bf16" or argument.kind in _FLOAT_FORMATS:
            value = _materialize_float(argument.kind, value, index)
        else:
            raise ValueError(f"bound argument {index} has unknown kind")
        kinds.append(argument.kind)
        values.append(value)
    return tuple(kinds), tuple(values)
