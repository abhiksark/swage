# python/swage/_frontend.py
"""Compile a restricted Python AST directly into MLIR operations."""

import ast
import functools
import hashlib
import inspect
import math
import struct
import textwrap
from collections.abc import Mapping
from typing import NamedTuple

from . import language

_INDEX_MIN = -(1 << 63)
_INDEX_MAX = (1 << 63) - 1
_OPERATORS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.MatMult: "@",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
}
_INSTALLATION = (
    "docs/getting-started/installation.md in "
    "https://github.com/abhiksark/swage"
)


class CompilationError(Exception):
    """A source-located error in a Swage kernel definition."""


class _Value(NamedTuple):
    """An MLIR value, its kind, and its compile-time index range.

    `value` is None while a body is checked without the native package.
    `bounds` is the inclusive `(low, high)` range of an index value whose
    operands are all known at compile time, and None for every other value.
    """

    value: object
    kind: str
    bounds: tuple | None = None


class _Address(NamedTuple):
    """A transient base buffer and vector offset pair."""

    base: object
    offsets: object


class _Kernel:
    """Captured, non-executing kernel source."""

    def __init__(self, function):
        self.python_function = function
        self.filename = inspect.getsourcefile(function) or "<unknown>"
        try:
            source, self.source_line = inspect.getsourcelines(function)
        except (OSError, TypeError) as error:
            raise CompilationError(
                f"{self.filename}:1:1: {function.__name__}: "
                "source is unavailable"
            ) from error
        self.source_indent = len(source[0]) - len(source[0].lstrip())
        try:
            parsed = ast.parse(textwrap.dedent("".join(source)))
        except SyntaxError as error:
            line = self.source_line + (error.lineno or 1) - 1
            column = (error.offset or 1) + self.source_indent
            raise CompilationError(
                f"{self.filename}:{line}:{column}: {function.__name__}: "
                f"{error.msg}"
            ) from error
        if len(parsed.body) != 1 or not isinstance(
            parsed.body[0], ast.FunctionDef
        ):
            raise CompilationError(
                f"{self.filename}:{self.source_line}:"
                f"{self.source_indent + 1}: "
                f"{function.__name__}: expected one function definition"
            )
        self.function = parsed.body[0]
        self._plain_parameters = False
        self.language_names = _language_names(function, self.function)
        # The names bound to the language module are part of what the source
        # means, so two kernels with one spelling and different bindings
        # must not share a compiled artifact.
        self.source_digest = hashlib.sha256(
            (
                ast.dump(self.function, include_attributes=False)
                + repr(sorted(self.language_names))
            ).encode()
        ).hexdigest()
        self.parameter_names = [
            argument.arg for argument in self.function.args.args
        ]
        self.constexpr_names = {
            argument.arg
            for argument in self.function.args.args
            if self._is_constexpr_annotation(argument.annotation)
        }
        functools.update_wrapper(self, function)
        if not (self.__name__.isascii() and self.__name__.isidentifier()):
            self._raise(
                self.function,
                "kernel name must be an ASCII identifier",
            )
        for decorator in self.function.decorator_list:
            is_jit = (
                isinstance(decorator, ast.Name) and decorator.id == "jit"
            ) or (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "jit"
            )
            if not is_jit:
                self._raise(
                    decorator,
                    "only @swage.jit may decorate a kernel",
                )

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"Swage kernel '{self.__name__}' is not directly callable; "
            "use kernel.launch()"
        )

    def launch(self, *, arguments, constexprs, grid):
        """Asynchronously launch the canonical fixed vector-add subset."""
        from ._runtime import launch

        return launch(
            self,
            arguments=arguments,
            constexprs=constexprs,
            grid=grid,
        )

    def emit_mlir(self, *, signature=None, arguments=None, constexprs):
        """Emit and return a live native MLIR module for this kernel.

        The body is checked against the kernel language before the native
        package is imported. A body outside the language therefore raises
        `CompilationError` on an install that has only the pure Python
        package, and a missing native package is reported only for a body
        that passed the check.
        """
        runtime_types, static_values = self._validate_inputs(
            signature, arguments, constexprs
        )
        _Checker(self, runtime_types, static_values).check()
        try:
            from mlir_swage import ir
            from mlir_swage.dialects import arith, func, swage, vector
        except ImportError as error:
            raise RuntimeError(
                "Swage emit_mlir() requires the build-tree mlir_swage "
                "bindings, which the swage-compiler wheel does not include; "
                f"kernel '{self.__name__}' passed the language check. See "
                f"{_INSTALLATION} for the native build"
            ) from error

        emitter = _Emitter(
            self,
            runtime_types,
            static_values,
            ir,
            arith,
            func,
            swage,
            vector,
        )
        return emitter.emit()

    def _validate_inputs(self, signature, arguments, constexprs):
        runtime_values, runtime_label = self._select_runtime_inputs(
            signature, arguments
        )
        self._validate_input_mappings(
            runtime_values, runtime_label, constexprs
        )
        parameters, constexpr_names, runtime_parameters = (
            self._partition_parameters(
                runtime_values, runtime_label, constexprs
            )
        )
        if signature is not None:
            runtime_types = self._validate_signature(
                signature, runtime_parameters
            )
        else:
            runtime_types = self._infer_signature(
                arguments, runtime_parameters
            )
        self._validate_constexprs(parameters, constexpr_names, constexprs)
        return runtime_types, dict(constexprs)

    def _select_runtime_inputs(self, signature, arguments):
        """Select exactly one explicit or inferred runtime input mode."""
        if (signature is None) == (arguments is None):
            self._raise(
                self.function,
                "exactly one of signature or arguments is required",
            )
        if signature is not None:
            return signature, "signature"
        return arguments, "arguments"

    def _validate_input_mappings(
        self, runtime_values, runtime_label, constexprs
    ):
        """Validate mapping containers and key types in diagnostic order."""
        if not isinstance(runtime_values, Mapping):
            self._raise(
                self.function, f"{runtime_label} must be a mapping"
            )
        if not isinstance(constexprs, Mapping):
            self._raise(self.function, "constexprs must be a mapping")
        if any(not isinstance(key, str) for key in runtime_values):
            self._raise(
                self.function, f"{runtime_label} keys must be strings"
            )
        if any(not isinstance(key, str) for key in constexprs):
            self._raise(self.function, "constexprs keys must be strings")

    def _partition_parameters(
        self, runtime_values, runtime_label, constexprs
    ):
        """Validate and return the declared runtime/constexpr partition."""
        self._require_plain_parameters()
        parameters = self.parameter_names
        constexpr_names = self.constexpr_names
        runtime_parameters = [
            name for name in parameters if name not in constexpr_names
        ]
        runtime_names = set(runtime_parameters)
        runtime_value_names = set(runtime_values)
        supplied_constexprs = set(constexprs)

        misplaced = runtime_value_names & constexpr_names
        if misplaced:
            name = sorted(misplaced)[0]
            self._raise(
                self.function,
                f"constexpr parameter '{name}' must be passed in constexprs",
            )
        misplaced = supplied_constexprs & runtime_names
        if misplaced:
            name = sorted(misplaced)[0]
            self._raise(
                self.function,
                f"runtime parameter '{name}' must be passed in "
                f"{runtime_label}",
            )
        self._require_keys(
            runtime_label, runtime_value_names, runtime_names
        )
        self._require_keys(
            "constexprs", supplied_constexprs, constexpr_names
        )
        return parameters, constexpr_names, runtime_parameters

    def _validate_constexprs(self, parameters, constexpr_names, constexprs):
        """Validate static values in source parameter order."""
        for name in parameters:
            if name in constexpr_names:
                self._validate_constexpr(name, constexprs[name])

    def _validate_constexpr(self, name, value):
        """Validate one constexpr value and its MLIR integer bounds."""
        if name == "BLOCK" and (type(value) is not int or value <= 0):
            self._raise(
                self.function,
                "constexpr 'BLOCK' must be a positive integer",
            )
        if type(value) is not int:
            self._raise(
                self.function,
                f"constexpr '{name}' must be an integer",
            )
        if not -(1 << 63) <= value <= (1 << 63) - 1:
            if name != "BLOCK":
                self._raise(
                    self.function,
                    f"constexpr '{name}' must fit signed 64-bit",
                )
            self._raise(
                self.function,
                "constexpr 'BLOCK' must fit a signed 64-bit MLIR dimension",
            )

    def _validate_signature(self, signature, runtime_parameters):
        for name in runtime_parameters:
            value = signature[name]
            if value is language.int32:
                continue
            if (
                isinstance(value, language._PointerType)
                and value.element_type is language.float32
            ):
                continue
            self._raise(
                self.function,
                f"unsupported type for parameter '{name}'",
            )
        return dict(signature)

    def _infer_signature(self, arguments, runtime_parameters):
        try:
            import torch
            float32 = torch.float32
            strided = torch.strided
            tensor_type = torch.Tensor
        except Exception:
            self._raise(
                self.function,
                "PyTorch metadata inference requires "
                "'swage-compiler[pytorch]'",
            )

        signature = {}
        for name in runtime_parameters:
            value = arguments[name]
            if type(value) is int and -(1 << 31) <= value < (1 << 31):
                signature[name] = language.int32
                continue
            if not isinstance(value, tensor_type):
                self._raise(
                    self.function,
                    f"unsupported argument for parameter '{name}'",
                )
            try:
                layout = value.layout
                dtype = value.dtype
                rank = value.dim()
                device_type = value.device.type
                contiguous = value.is_contiguous()
            except Exception:
                self._raise(
                    self.function,
                    f"could not read PyTorch metadata for parameter "
                    f"'{name}'; install 'swage-compiler[pytorch]'",
                )
            if layout != strided:
                reason = f"layout {layout}"
            elif dtype != float32:
                reason = f"dtype {dtype}"
            elif rank != 1:
                reason = f"rank {rank}"
            elif device_type not in {"cpu", "cuda"}:
                reason = f"device type '{device_type}'"
            elif not contiguous:
                reason = "non-contiguous"
            else:
                signature[name] = language.pointer(language.float32)
                continue
            self._raise(
                self.function,
                f"unsupported argument for parameter '{name}': {reason}",
            )
        return signature

    def _require_plain_parameters(self):
        """Reject every parameter form the kernel ABI does not express.

        `launch` calls this on every launch. The parsed source never
        changes, so a parameter list that passed is not walked again.
        """
        if self._plain_parameters:
            return
        syntax_arguments = self.function.args
        if syntax_arguments.posonlyargs:
            self._raise(
                syntax_arguments.posonlyargs[0],
                "positional-only parameters are unsupported",
            )
        if syntax_arguments.kwonlyargs:
            self._raise(
                syntax_arguments.kwonlyargs[0],
                "keyword-only parameters are unsupported",
            )
        if syntax_arguments.vararg:
            self._raise(
                syntax_arguments.vararg,
                "variadic positional parameters are unsupported",
            )
        if syntax_arguments.kwarg:
            self._raise(
                syntax_arguments.kwarg,
                "variadic keyword parameters are unsupported",
            )
        parameters = syntax_arguments.args
        if syntax_arguments.defaults:
            defaults = syntax_arguments.defaults
            name = parameters[len(parameters) - len(defaults)].arg
            self._raise(
                defaults[0],
                f"parameter '{name}' has a default value; defaults are "
                "unsupported, so pass every value when the kernel is "
                "emitted or launched",
            )
        for parameter in parameters:
            annotation = parameter.annotation
            if annotation is None or parameter.arg in self.constexpr_names:
                continue
            self._raise(
                annotation,
                f"unsupported annotation '{ast.unparse(annotation)}' on "
                f"parameter '{parameter.arg}'; the only accepted annotation "
                "is constexpr through a name bound to the swage.language "
                "module, such as sl.constexpr",
            )
        if self.function.returns is not None:
            self._raise(
                self.function.returns,
                "return annotations are unsupported; a kernel returns "
                "nothing",
            )
        self._plain_parameters = True

    def _is_constexpr_annotation(self, node):
        """Tell whether an annotation is `constexpr` on the language module."""
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.language_names
            and node.attr == "constexpr"
        )

    def _require_keys(self, label, actual, expected):
        if actual == expected:
            return
        details = []
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"extra: {', '.join(extra)}")
        parameter_kind = (
            "runtime parameters" if label in {"signature", "arguments"} else
            "constexpr parameters"
        )
        self._raise(
            self.function,
            f"{label} keys must match {parameter_kind}; {'; '.join(details)}",
        )

    def _raise(self, node, reason):
        line = self.source_line + node.lineno - 1
        column = node.col_offset + self.source_indent + 1
        raise CompilationError(
            f"{self.filename}:{line}:{column}: {self.__name__}: {reason}"
        )


class _Checker:
    """Walks a kernel body, enforces the language, and tracks value kinds.

    Nothing in this class needs the native package, and every diagnostic is
    raised here. `_Emitter` repeats the same walk and overrides only the
    `_build_*` hooks, which create nothing in this class, so the check and
    the emission accept the same bodies and report the same text.
    """

    def __init__(self, kernel, runtime_types, constexprs):
        self.kernel = kernel
        self.runtime_types = runtime_types
        self.constexprs = constexprs
        self.block = constexprs.get("BLOCK")
        self.symbols = {}

    def check(self):
        """Raise `CompilationError` unless the body is in the language."""
        body = self._body()
        self._bind_arguments([None] * len(self._runtime_parameters()))
        for statement in body:
            self._statement(statement)

    def _body(self):
        """Return the statements to walk, without a leading docstring."""
        body = self.kernel.function.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and type(body[0].value.value) is str
        ):
            body = body[1:]
        for statement in body[:-1]:
            if isinstance(statement, ast.Return):
                self._error(
                    statement,
                    "empty return must be the final statement",
                )
        return body

    def _runtime_parameters(self):
        return [
            argument
            for argument in self.kernel.function.args.args
            if argument.arg in self.runtime_types
        ]

    def _bind_arguments(self, values):
        for syntax, value in zip(
            self._runtime_parameters(), values, strict=True
        ):
            declared = self.runtime_types[syntax.arg]
            kind = "i32" if declared is language.int32 else "pointer"
            self.symbols[syntax.arg] = _Value(value, kind)

    def _statement(self, node):
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(
                node.targets[0], ast.Name
            ):
                self._error(node, "only single-name assignments are supported")
            target = node.targets[0]
            if target.id in self.constexprs:
                self._error(
                    target,
                    f"cannot assign to constexpr parameter '{target.id}'",
                )
            self.symbols[target.id] = self._expression(node.value)
            return
        if isinstance(node, ast.Expr):
            if (
                isinstance(node.value, ast.Call)
                and self._language_call(node.value) == "store"
            ):
                self._store(node.value)
                return
            value = self._expression(node.value)
            if value is not None:
                self._error(node, "only sl.store may be used as a statement")
            return
        if isinstance(node, ast.Return):
            if node.value is not None:
                self._error(node, "return values are unsupported")
            self._build_return(node)
            return
        self._error(node, f"unsupported statement '{type(node).__name__}'")

    def _expression(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.symbols:
                return self.symbols[node.id]
            if node.id in self.constexprs:
                return self._index_constant(self.constexprs[node.id], node)
            self._error(node, f"unknown name '{node.id}'")
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return self._index_constant(node.value, node)
        if isinstance(node, ast.Constant) and type(node.value) is float:
            self._error(
                node,
                f"float literal {ast.unparse(node)} is unsupported here; a "
                "float literal is accepted only as the other= value of load",
            )
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        self._error(node, f"unsupported expression '{type(node).__name__}'")

    def _binary(self, node):
        left = self._expression(node.left)
        right = self._expression(node.right)
        if isinstance(left, _Value) and left.kind == "pointer":
            if (
                isinstance(node.op, ast.Add)
                and isinstance(right, _Value)
                and right.kind == "index_vector"
            ):
                return _Address(left.value, right.value)
            self._error(node, "pointers support only addition with offsets")
        if not isinstance(left, _Value) or not isinstance(right, _Value):
            self._error(node, "unsupported binary operands")
        operator = _OPERATORS.get(type(node.op), type(node.op).__name__)
        if "f32_vector" in {left.kind, right.kind}:
            if not isinstance(node.op, ast.Add):
                self._error(
                    node,
                    f"'{operator}' is not supported on f32 vectors; only "
                    "'+' is",
                )
            if left.kind != right.kind:
                self._error(
                    node,
                    "'+' on an f32 vector requires another f32 vector",
                )
            return _Value(self._build_add_f32(left, right, node), "f32_vector")
        if not isinstance(node.op, (ast.Add, ast.Mult)):
            self._error(
                node,
                f"unsupported binary operator '{operator}'; index "
                "arithmetic supports only '+' and '*'",
            )
        left, right = self._broadcast_index_pair(left, right, node)
        bounds = self._index_bounds(left, right, node, operator)
        if isinstance(node.op, ast.Add):
            result = self._build_add_index(left, right, node)
        else:
            result = self._build_mul_index(left, right, node)
        return _Value(result, left.kind, bounds)

    def _broadcast_index_pair(self, left, right, node):
        allowed = {"index", "index_vector"}
        if left.kind not in allowed or right.kind not in allowed:
            self._error(node, "integer arithmetic requires index operands")
        if left.kind == right.kind:
            return left, right
        if left.kind == "index":
            left = _Value(
                self._build_index_broadcast(left, node),
                "index_vector",
                left.bounds,
            )
        else:
            right = _Value(
                self._build_index_broadcast(right, node),
                "index_vector",
                right.bounds,
            )
        return left, right

    def _index_bounds(self, left, right, node, operator):
        """Range-check index arithmetic on compile-time operands.

        MLIR index arithmetic wraps at 64 bits where Python integers do
        not. A result whose operands are all literals, constexpr values, or
        `arange` lanes has a range known here, and a range that leaves
        signed 64-bit is rejected. An operand that depends on a program
        coordinate is known only at run time, so its results are unchecked.
        """
        if left.bounds is None or right.bounds is None:
            return None
        if isinstance(node.op, ast.Add):
            low = left.bounds[0] + right.bounds[0]
            high = left.bounds[1] + right.bounds[1]
        else:
            products = [a * b for a in left.bounds for b in right.bounds]
            low, high = min(products), max(products)
        if low < _INDEX_MIN or high > _INDEX_MAX:
            self._error(
                node,
                f"'{operator}' on compile-time index operands leaves signed "
                f"64-bit; the result ranges from {low} to {high}",
            )
        return low, high

    def _compare(self, node):
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Lt):
            self._error(node, "only a single '<' comparison is supported")
        reason = "comparison requires index offsets and i32"
        left = self._require_value(
            self._expression(node.left), node.left, reason
        )
        right = self._require_value(
            self._expression(node.comparators[0]),
            node.comparators[0],
            reason,
        )
        if left.kind != "index_vector" or right.kind not in {"i32", "index"}:
            self._error(node, reason)
        return _Value(self._build_less_than(left, right, node), "bool_vector")

    def _call(self, node):
        name = self._language_call(node)
        callee = ast.unparse(node.func)
        handlers = {
            "program_id": self._program_id,
            "arange": self._arange,
            "load": self._load,
        }
        if name == "store":
            self._error(
                node,
                f"{callee} is only supported as an expression statement",
            )
        if name not in handlers:
            self._error(
                node,
                f"unsupported call '{callee}'; a kernel can call only "
                "program_id, arange, load, and store, each through a name "
                "bound to the swage.language module, such as sl.load",
            )
        return handlers[name](node, callee)

    def _language_call(self, node):
        """Return the attribute a call takes from the language module."""
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in self.kernel.language_names
        ):
            return function.attr
        return None

    def _program_id(self, node, callee):
        if node.keywords or len(node.args) != 1:
            self._error(node, f"{callee} expects one axis literal")
        axis = node.args[0]
        if (
            not isinstance(axis, ast.Constant)
            or type(axis.value) is not int
            or axis.value < 0
        ):
            self._error(axis, "program_id axis must be a nonnegative integer")
        if axis.value > (1 << 31) - 1:
            self._error(axis, "program_id axis must fit signed i32")
        return _Value(self._build_program_id(axis.value, node), "index")

    def _arange(self, node, callee):
        if node.keywords or len(node.args) != 2:
            self._error(node, f"{callee} expects start and end")
        start, end = node.args
        if (
            not isinstance(start, ast.Constant)
            or type(start.value) is not int
            or start.value != 0
            or not isinstance(end, ast.Name)
            or end.id != "BLOCK"
        ):
            self._error(
                node,
                f"{ast.unparse(node)} is unsupported; the start must be the "
                "literal 0 and the end must be the compile-time parameter "
                "named BLOCK",
            )
        if self.block is None:
            self._error(node, "BLOCK is required for vector operations")
        return _Value(
            self._build_arange(node), "index_vector", (0, self.block - 1)
        )

    def _load(self, node, callee):
        keywords = self._keywords(node)
        if len(node.args) != 1 or set(keywords) != {"mask", "other"}:
            self._error(
                node,
                f"{callee} requires one address, mask=..., and other=...; "
                "both keywords are required",
            )
        address = self._expression(node.args[0])
        mask = self._require_value(
            self._expression(keywords["mask"]),
            keywords["mask"],
            f"{callee} mask must be a vector",
        )
        if not isinstance(address, _Address):
            self._error(node.args[0], f"{callee} requires pointer + offsets")
        if mask.kind != "bool_vector":
            self._error(keywords["mask"], f"{callee} mask must be a vector")
        other = self._float32_literal(keywords["other"], callee)
        return _Value(
            self._build_load(address, mask, other, node), "f32_vector"
        )

    def _float32_literal(self, node, callee):
        """Return the `other` literal, rejecting what float32 cannot hold.

        A float literal rounds to the nearest float32, as a float32 constant
        does in any language. It is rejected when the result is not finite
        or when a nonzero literal rounds to zero. An integer literal is
        rejected unless float32 represents it exactly.
        """
        literal = node
        negative = isinstance(node, ast.UnaryOp) and isinstance(
            node.op, ast.USub
        )
        if negative:
            literal = node.operand
        if (
            not isinstance(literal, ast.Constant)
            or type(literal.value) not in {int, float}
        ):
            self._error(node, f"{callee} other must be a numeric literal")
        value = -literal.value if negative else literal.value
        written = f"{callee} other={ast.unparse(node)}"
        try:
            exact = float(value)
            rounded = struct.unpack("<f", struct.pack("<f", exact))[0]
        except OverflowError:
            rounded = None
        if rounded is None:
            self._error(node, f"{written} is outside the float32 range")
        if not math.isfinite(exact):
            self._error(node, f"{callee} other must be a finite literal")
        if value != 0 and rounded == 0:
            self._error(node, f"{written} rounds to zero in float32")
        if type(value) is int and rounded != value:
            self._error(
                node,
                f"{written} is an integer that float32 cannot hold exactly",
            )
        return exact

    def _store(self, node):
        callee = ast.unparse(node.func)
        keywords = self._keywords(node)
        if len(node.args) != 2 or set(keywords) != {"mask"}:
            self._error(
                node, f"{callee} expects address, value, and mask=..."
            )
        address = self._expression(node.args[0])
        reason = f"{callee} requires float values and a mask"
        value = self._require_value(
            self._expression(node.args[1]), node.args[1], reason
        )
        mask = self._require_value(
            self._expression(keywords["mask"]), keywords["mask"], reason
        )
        if not isinstance(address, _Address):
            self._error(node.args[0], f"{callee} requires pointer + offsets")
        if value.kind != "f32_vector" or mask.kind != "bool_vector":
            self._error(node, reason)
        self._build_store(address, value, mask, node)
        return None

    def _keywords(self, node):
        if any(keyword.arg is None for keyword in node.keywords):
            self._error(node, "keyword expansion is unsupported")
        names = [keyword.arg for keyword in node.keywords]
        if len(names) != len(set(names)):
            self._error(node, "duplicate keyword argument")
        return {keyword.arg: keyword.value for keyword in node.keywords}

    def _index_constant(self, value, node):
        if not _INDEX_MIN <= value <= _INDEX_MAX:
            self._error(node, "integer literal must fit signed 64-bit")
        return _Value(
            self._build_index_constant(value, node), "index", (value, value)
        )

    def _require_value(self, value, node, reason):
        if not isinstance(value, _Value):
            self._error(node, reason)
        return value

    def _error(self, node, reason):
        self.kernel._raise(node, reason)

    # The hooks below create MLIR in `_Emitter` and nothing here. They must
    # not raise a diagnostic: a body that reaches a hook is already accepted.

    def _build_return(self, node):
        return None

    def _build_index_constant(self, value, node):
        return None

    def _build_index_broadcast(self, scalar, node):
        return None

    def _build_add_f32(self, left, right, node):
        return None

    def _build_add_index(self, left, right, node):
        return None

    def _build_mul_index(self, left, right, node):
        return None

    def _build_less_than(self, left, right, node):
        return None

    def _build_program_id(self, axis, node):
        return None

    def _build_arange(self, node):
        return None

    def _build_load(self, address, mask, other, node):
        return None

    def _build_store(self, address, value, mask, node):
        return None


class _Emitter(_Checker):
    """Direct AST-to-MLIR emitter for the fixed-block slice.

    The walk and its diagnostics live in `_Checker`. This class supplies the
    MLIR module around the walk and the operation each hook creates.
    """

    def __init__(
        self,
        kernel,
        runtime_types,
        constexprs,
        ir,
        arith,
        func,
        swage,
        vector,
    ):
        super().__init__(kernel, runtime_types, constexprs)
        self.ir = ir
        self.arith = arith
        self.func = func
        self.swage = swage
        self.vector = vector

    def emit(self):
        """Build, verify, and return the MLIR module for the kernel."""
        body = self._body()
        context = self.ir.Context()
        with context:
            self.swage.register_dialects(context)
            self.f32 = self.ir.F32Type.get()
            self.i32 = self.ir.IntegerType.get_signless(32)
            self.index = self.ir.IndexType.get()
            location = self._location(self.kernel.function)
            with location:
                module = self.ir.Module.create(location)
                argument_types = self._argument_types()
                with self.ir.InsertionPoint(module.body):
                    function = self.func.FuncOp(
                        self.kernel.__name__,
                        (argument_types, []),
                        loc=location,
                    )
                with self.ir.InsertionPoint(function.add_entry_block()):
                    self._bind_arguments(function.arguments)
                    for statement in body:
                        self._statement(statement)
                    if not body or not isinstance(body[-1], ast.Return):
                        self.func.ReturnOp([], loc=location)
            try:
                verified = module.operation.verify()
            except self.ir.MLIRError:
                self._error(
                    self.kernel.function,
                    "emitted MLIR failed verification",
                )
            if not verified:
                self._error(
                    self.kernel.function,
                    "emitted MLIR failed verification",
                )
            return module

    def _argument_types(self):
        dynamic = self.ir.ShapedType.get_dynamic_size()
        types = []
        for argument in self._runtime_parameters():
            declared = self.runtime_types[argument.arg]
            if declared is language.int32:
                types.append(self.i32)
            else:
                types.append(self.ir.MemRefType.get([dynamic], self.f32))
        return types

    def _build_return(self, node):
        self.func.ReturnOp([], loc=self._location(node))

    def _build_index_constant(self, value, node):
        return self.arith.ConstantOp(
            self.index, value, loc=self._location(node)
        ).result

    def _build_index_broadcast(self, scalar, node):
        return self.vector.BroadcastOp(
            self._index_vector_type(), scalar.value, loc=self._location(node)
        ).result

    def _build_add_f32(self, left, right, node):
        return self.arith.AddFOp(
            left.value, right.value, loc=self._location(node)
        ).result

    def _build_add_index(self, left, right, node):
        return self.arith.AddIOp(
            left.value, right.value, loc=self._location(node)
        ).result

    def _build_mul_index(self, left, right, node):
        return self.arith.MulIOp(
            left.value, right.value, loc=self._location(node)
        ).result

    def _build_less_than(self, left, right, node):
        location = self._location(node)
        bound = right.value
        if right.kind == "i32":
            bound = self.arith.IndexCastOp(
                self.index, bound, loc=location
            ).result
        bound = self.vector.BroadcastOp(
            self._index_vector_type(), bound, loc=location
        ).result
        return self.arith.CmpIOp(
            self.arith.CmpIPredicate.slt,
            left.value,
            bound,
            loc=location,
        ).result

    def _build_program_id(self, axis, node):
        return self.swage.ProgramIdOp(
            self.index, axis, loc=self._location(node)
        ).result

    def _build_arange(self, node):
        return self.vector.StepOp(
            self._index_vector_type(), loc=self._location(node)
        ).result

    def _build_load(self, address, mask, other, node):
        location = self._location(node)
        constant = self.arith.ConstantOp(self.f32, other, loc=location).result
        pass_through = self.vector.BroadcastOp(
            self._float_vector_type(), constant, loc=location
        ).result
        zero = self.arith.ConstantOp(self.index, 0, loc=location).result
        return self.vector.GatherOp(
            self._float_vector_type(),
            address.base,
            [zero],
            address.offsets,
            mask.value,
            pass_through,
            loc=location,
        ).result

    def _build_store(self, address, value, mask, node):
        location = self._location(node)
        zero = self.arith.ConstantOp(self.index, 0, loc=location).result
        self.vector.ScatterOp(
            None,
            address.base,
            [zero],
            address.offsets,
            mask.value,
            value.value,
            loc=location,
        )

    def _index_vector_type(self):
        return self.ir.VectorType.get([self.block], self.index)

    def _float_vector_type(self):
        return self.ir.VectorType.get([self.block], self.f32)

    def _location(self, node):
        line = self.kernel.source_line + node.lineno - 1
        child = self.ir.Location.file(
            self.kernel.filename,
            line,
            node.col_offset + self.kernel.source_indent + 1,
        )
        return self.ir.Location.name(self.kernel.__name__, child)


def _language_names(function, syntax):
    """Return the names a kernel uses that are bound to `swage.language`.

    A name counts when the function sees the module object through its
    closure or its globals and the kernel neither takes the name as a
    parameter nor assigns to it. The match is on the object, so any import
    name works and a different object spelled `sl` does not.

    Args:
        function: The Python function being captured.
        syntax: Its parsed `ast.FunctionDef`.

    Returns:
        A frozenset of names.
    """
    loaded = set()
    local = set()
    for node in ast.walk(syntax):
        if isinstance(node, ast.arg):
            local.add(node.arg)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            else:
                local.add(node.id)
    cells = dict(
        zip(function.__code__.co_freevars, function.__closure__ or ())
    )
    names = set()
    for name in loaded - local:
        if name in cells:
            try:
                bound = cells[name].cell_contents
            except ValueError:
                continue
        else:
            bound = function.__globals__.get(name)
        if bound is language:
            names.add(name)
    return frozenset(names)


def jit(function):
    """Capture a Python function as a non-executing Swage kernel."""
    return _Kernel(function)


__all__ = ["CompilationError", "jit"]
