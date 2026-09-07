# python/swage/language.pyi
"""Descriptors and symbolic values, not a second DSL type system."""

from typing import Any, TypeAlias

constexpr: TypeAlias = Any

class _ScalarType: ...

class _PointerType:
    element_type: _ScalarType

float16: _ScalarType
float8_e4m3fn: _ScalarType
float8_e5m2: _ScalarType
float32: _ScalarType
int32: _ScalarType

def pointer(element_type: _ScalarType) -> _PointerType: ...
def program_id(axis: int) -> Any: ...
def arange(start: int, end: int) -> Any: ...
def load(pointer_value: Any, *, mask: Any = ..., other: Any = ...) -> Any: ...
def store(pointer_value: Any, value: Any, *, mask: Any = ...) -> None: ...
