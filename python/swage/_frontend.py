# python/swage/_frontend.py
"""Compile a restricted Python AST directly into MLIR operations."""

import ast
import functools
import inspect
import textwrap
from collections.abc import Mapping
from typing import NamedTuple

from . import language


class CompilationError(Exception):
    """A source-located error in a Swage kernel definition."""


class _Value(NamedTuple):
    """An MLIR value and the small type fact needed while emitting."""

    value: object
    kind: str


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
        """Asynchronously launch the M3 fixed vector-add subset."""
        from ._runtime import launch

        return launch(
            self,
            arguments=arguments,
            constexprs=constexprs,
            grid=grid,
        )

    def emit_mlir(self, *, signature=None, arguments=None, constexprs):
        """Emit and return a live native MLIR module for this kernel."""
        runtime_types, static_values = self._validate_inputs(
            signature, arguments, constexprs
        )
        try:
            from mlir_swage import ir
            from mlir_swage.dialects import arith, func, swage, vector
        except ImportError as error:
            raise RuntimeError(
                "Swage emit_mlir() requires the build-tree "
                "mlir_swage bindings"
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
        if (signature is None) == (arguments is None):
            self._raise(
                self.function,
                "exactly one of signature or arguments is required",
            )
        runtime_values = signature if signature is not None else arguments
        runtime_label = "signature" if signature is not None else "arguments"
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

        parameters = [argument.arg for argument in syntax_arguments.args]
        constexpr_names = {
            argument.arg
            for argument in syntax_arguments.args
            if _is_constexpr_annotation(argument.annotation)
        }
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

        if signature is not None:
            runtime_types = self._validate_signature(
                signature, runtime_parameters
            )
        else:
            runtime_types = self._infer_signature(
                arguments, runtime_parameters
            )

        for name in parameters:
            if name not in constexpr_names:
                continue
            value = constexprs[name]
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
                    "constexpr 'BLOCK' must fit a signed 64-bit MLIR "
                    "dimension",
                )
        return runtime_types, dict(constexprs)

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


class _Emitter:
    """Minimal direct AST-to-MLIR emitter for the fixed-block slice."""

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
        self.kernel = kernel
        self.runtime_types = runtime_types
        self.constexprs = constexprs
        self.ir = ir
        self.arith = arith
        self.func = func
        self.swage = swage
        self.vector = vector
        self.symbols = {}

    def emit(self):
        for statement in self.kernel.function.body[:-1]:
            if isinstance(statement, ast.Return):
                self._error(
                    statement,
                    "empty return must be the final statement",
                )
        context = self.ir.Context()
        with context:
            self.swage.register_dialects(context)
            self.f32 = self.ir.F32Type.get()
            self.i32 = self.ir.IntegerType.get_signless(32)
            self.index = self.ir.IndexType.get()
            self.block = self.constexprs.get("BLOCK")
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
                    for statement in self.kernel.function.body:
                        self._statement(statement)
                    if not self.kernel.function.body or not isinstance(
                        self.kernel.function.body[-1], ast.Return
                    ):
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
        for argument in self.kernel.function.args.args:
            if argument.arg not in self.runtime_types:
                continue
            declared = self.runtime_types[argument.arg]
            if declared is language.int32:
                types.append(self.i32)
            else:
                types.append(self.ir.MemRefType.get([dynamic], self.f32))
        return types

    def _bind_arguments(self, arguments):
        runtime_arguments = (
            argument
            for argument in self.kernel.function.args.args
            if argument.arg in self.runtime_types
        )
        for syntax, value in zip(runtime_arguments, arguments, strict=True):
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
                and self._symbolic_call_name(node.value) == "store"
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
            self.func.ReturnOp([], loc=self._location(node))
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
        location = self._location(node)
        if isinstance(node.op, ast.Add):
            if left.kind == right.kind == "f32_vector":
                result = self.arith.AddFOp(
                    left.value, right.value, loc=location
                ).result
                return _Value(result, "f32_vector")
            left, right = self._broadcast_index_pair(left, right, node)
            result = self.arith.AddIOp(
                left.value, right.value, loc=location
            ).result
            return _Value(result, left.kind)
        if isinstance(node.op, ast.Mult):
            left, right = self._broadcast_index_pair(left, right, node)
            result = self.arith.MulIOp(
                left.value, right.value, loc=location
            ).result
            return _Value(result, left.kind)
        operator = type(node.op).__name__
        self._error(node, f"unsupported binary operator '{operator}'")

    def _broadcast_index_pair(self, left, right, node):
        allowed = {"index", "index_vector"}
        if left.kind not in allowed or right.kind not in allowed:
            self._error(node, "integer arithmetic requires index operands")
        if left.kind == right.kind:
            return left, right
        vector_type = self._index_vector_type(node)
        location = self._location(node)
        if left.kind == "index":
            left = _Value(
                self.vector.BroadcastOp(
                    vector_type, left.value, loc=location
                ).result,
                "index_vector",
            )
        else:
            right = _Value(
                self.vector.BroadcastOp(
                    vector_type, right.value, loc=location
                ).result,
                "index_vector",
            )
        return left, right

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
        location = self._location(node)
        if right.kind == "i32":
            right = _Value(
                self.arith.IndexCastOp(
                    self.index, right.value, loc=location
                ).result,
                "index",
            )
        right = self.vector.BroadcastOp(
            self._index_vector_type(node), right.value, loc=location
        ).result
        result = self.arith.CmpIOp(
            self.arith.CmpIPredicate.slt,
            left.value,
            right,
            loc=location,
        ).result
        return _Value(result, "bool_vector")

    def _call(self, node):
        name = self._symbolic_call_name(node)
        handlers = {
            "program_id": self._program_id,
            "arange": self._arange,
            "load": self._load,
        }
        if name == "store":
            self._error(
                node,
                "sl.store is only supported as an expression statement",
            )
        if name not in handlers:
            self._error(node, "unsupported call")
        return handlers[name](node)

    @staticmethod
    def _symbolic_call_name(node):
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "sl"
        ):
            return function.attr
        return None

    def _program_id(self, node):
        if node.keywords or len(node.args) != 1:
            self._error(node, "sl.program_id expects one axis literal")
        axis = node.args[0]
        if (
            not isinstance(axis, ast.Constant)
            or type(axis.value) is not int
            or axis.value < 0
        ):
            self._error(axis, "program_id axis must be a nonnegative integer")
        if axis.value > (1 << 31) - 1:
            self._error(axis, "program_id axis must fit signed i32")
        result = self.swage.ProgramIdOp(
            self.index, axis.value, loc=self._location(node)
        ).result
        return _Value(result, "index")

    def _arange(self, node):
        if node.keywords or len(node.args) != 2:
            self._error(node, "sl.arange expects start and end")
        start, end = node.args
        if (
            not isinstance(start, ast.Constant)
            or type(start.value) is not int
            or start.value != 0
            or not isinstance(end, ast.Name)
            or end.id != "BLOCK"
        ):
            self._error(node, "only sl.arange(0, BLOCK) is supported")
        result = self.vector.StepOp(
            self._index_vector_type(node), loc=self._location(node)
        ).result
        return _Value(result, "index_vector")

    def _load(self, node):
        keywords = self._keywords(node)
        if len(node.args) != 1 or set(keywords) != {"mask", "other"}:
            self._error(
                node,
                "sl.load expects address, mask=..., and other=...",
            )
        address = self._expression(node.args[0])
        mask = self._require_value(
            self._expression(keywords["mask"]),
            keywords["mask"],
            "sl.load mask must be a vector",
        )
        other_node = keywords["other"]
        if not isinstance(address, _Address):
            self._error(node.args[0], "sl.load requires pointer + offsets")
        if mask.kind != "bool_vector":
            self._error(keywords["mask"], "sl.load mask must be a vector")
        if (
            not isinstance(other_node, ast.Constant)
            or type(other_node.value) not in {int, float}
        ):
            self._error(other_node, "sl.load other must be a numeric literal")
        location = self._location(node)
        other = self.arith.ConstantOp(
            self.f32, float(other_node.value), loc=location
        ).result
        pass_through = self.vector.BroadcastOp(
            self._float_vector_type(node), other, loc=location
        ).result
        zero = self.arith.ConstantOp(self.index, 0, loc=location).result
        result = self.vector.GatherOp(
            self._float_vector_type(node),
            address.base,
            [zero],
            address.offsets,
            mask.value,
            pass_through,
            loc=location,
        ).result
        return _Value(result, "f32_vector")

    def _store(self, node):
        keywords = self._keywords(node)
        if len(node.args) != 2 or set(keywords) != {"mask"}:
            self._error(node, "sl.store expects address, value, and mask=...")
        address = self._expression(node.args[0])
        reason = "sl.store requires float values and a mask"
        value = self._require_value(
            self._expression(node.args[1]), node.args[1], reason
        )
        mask = self._require_value(
            self._expression(keywords["mask"]), keywords["mask"], reason
        )
        if not isinstance(address, _Address):
            self._error(node.args[0], "sl.store requires pointer + offsets")
        if value.kind != "f32_vector" or mask.kind != "bool_vector":
            self._error(node, "sl.store requires float values and a mask")
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
        return None

    def _keywords(self, node):
        if any(keyword.arg is None for keyword in node.keywords):
            self._error(node, "keyword expansion is unsupported")
        names = [keyword.arg for keyword in node.keywords]
        if len(names) != len(set(names)):
            self._error(node, "duplicate keyword argument")
        return {keyword.arg: keyword.value for keyword in node.keywords}

    def _index_constant(self, value, node):
        if not -(1 << 63) <= value <= (1 << 63) - 1:
            self._error(node, "integer literal must fit signed 64-bit")
        result = self.arith.ConstantOp(
            self.index, value, loc=self._location(node)
        ).result
        return _Value(result, "index")

    def _require_value(self, value, node, reason):
        if not isinstance(value, _Value):
            self._error(node, reason)
        return value

    def _index_vector_type(self, node):
        if self.block is None:
            self._error(node, "BLOCK is required for vector operations")
        return self.ir.VectorType.get([self.block], self.index)

    def _float_vector_type(self, node):
        if self.block is None:
            self._error(node, "BLOCK is required for vector operations")
        return self.ir.VectorType.get([self.block], self.f32)

    def _location(self, node):
        line = self.kernel.source_line + node.lineno - 1
        child = self.ir.Location.file(
            self.kernel.filename,
            line,
            node.col_offset + self.kernel.source_indent + 1,
        )
        return self.ir.Location.name(self.kernel.__name__, child)

    def _error(self, node, reason):
        self.kernel._raise(node, reason)


def _is_constexpr_annotation(node):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sl"
        and node.attr == "constexpr"
    )


def jit(function):
    """Capture a Python function as a non-executing Swage kernel."""
    return _Kernel(function)


__all__ = ["CompilationError", "jit"]
