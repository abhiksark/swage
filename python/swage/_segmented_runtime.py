# python/swage/_segmented_runtime.py
"""Compile, load, bind, and enqueue the kernels of segment programs.

A kernel is compiled at most once per process into a `_Kernel`, which keeps
its PTX and the launch contract the compiler gave it. `_cuda_backend` loads
it once per CUDA context and hands out leases on the module, which keep it
loaded; a prepared launch holds its leases for as long as it lives. The
arguments of a launch are bound by the contract: the user arguments by the
roles of the segment function, and the derived counts, plan records, and
scratch buffers by their contract keys.

The PTX memo is keyed by the native compile function, the semantic module
text, and every code generation option (kernel name, block size, target).
It is bounded: it keeps `_runtime._CACHE_LIMIT` kernels and then forgets
its oldest, which is compiled again on its next use. A hit takes no lock. A
miss takes the process-wide cold-path lock, which serializes native
compiles; see `_runtime._compile_lock`. The private segmented path has no
persistent cache.
"""

import types

from . import _abi, _artifact, _cuda_backend, _runtime
from . import _segmented_programs as _programs

_target_record = None
# The entry suffix of the kernel each native compile function produces,
# beside the name of the segment function.
_ENTRY_SUFFIXES = {
    "_compile_split_partial_reduction_ptx": "__partial",
    "_compile_split_merge_reduction_ptx": "__merge",
}
_memo_lock = _runtime._compile_lock
_ptx_memo = _runtime._BoundedCache(_runtime._CACHE_LIMIT)
# The NVPTX processor of each CUDA device index. A device keeps its compute
# capability for as long as the process runs.
_targets = {}


def _target_description():
    """Return the target description of the compiler, read once per process.

    Block widths, the subgroup width, claim batches, and the planning
    defaults live in the native target description, which the lowerings and
    the code generation C API read too, so the host cannot disagree with a
    kernel. The native package is imported here and not when this module is,
    which keeps `import swage` free of it.

    A selected artifact answers instead of the native package, which is then
    not imported: it gives the widths its kernels were compiled for and the
    planning limits its programs were admitted under.

    Returns:
        A namespace with one attribute per field of the description, for
        example `subgroup_width`, `cta_block_threads`, and
        `default_cta_chunk_elements`. An artifact gives the fields the
        kernels it holds are launched with, and no others.

    Raises:
        RuntimeError: `SWAGE_ARTIFACT_DIR` names an artifact that cannot be
            used. The bindings are not tried instead.
    """
    artifact = _artifact.selected()
    if artifact is not None:
        return artifact.target_description
    global _target_record
    if _target_record is None:
        from mlir_swage._mlir_libs._swageDialectsNanobind import (
            swage as native_swage,
        )

        _target_record = types.SimpleNamespace(
            **native_swage._target_description()
        )
    return _target_record


def _target(torch, device_index):
    """Return the NVPTX processor name of one CUDA device, looked up once."""
    target = _targets.get(device_index)
    if target is None:
        major, minor = torch.cuda.get_device_capability(device_index)
        target = _targets[device_index] = f"sm_{major}{minor}"
    return target


def _native_swage():
    """Return what classifies, admits, and compiles for the planned path.

    A selected artifact stands in for the native `swage` bindings, which
    are then not imported: the artifact answers with kernels that were
    compiled ahead of time. Without one, the bindings are imported.

    Raises:
        RuntimeError: `SWAGE_ARTIFACT_DIR` names an artifact that cannot be
            used. The bindings are not tried instead.
    """
    artifact = _artifact.selected()
    if artifact is not None:
        return artifact
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    return native_swage


class _Kernel:
    """One compiled kernel and the launch contract the compiler gave it.

    It is what `_cuda_backend` loads: `image` is the PTX, and the PTX text
    is also the identity, so a kernel is loaded once per CUDA context.

    Attributes:
        image: The PTX text.
        contract_json: The launch contract as canonical JSON.
        contract: The parsed launch contract.
    """

    __slots__ = ("image", "contract_json", "contract")

    def __init__(self, image, contract_json, contract):
        """Hold one kernel whose contract was parsed and checked."""
        self.image = image
        self.contract_json = contract_json
        self.contract = contract

    @property
    def identity(self):
        """Return what tells one loaded kernel from another: its PTX."""
        return self.image


def _checked_kernel(compiler, ptx, contract_json, options):
    """Parse the contract of a fresh kernel and check it against its request.

    The launch takes its block from the contract, so a contract for another
    entry or block would launch the kernel wrongly; it is refused here.

    Raises:
        RuntimeError: The contract is malformed, or names another entry,
            backend, or block than the request.
    """
    contract = _runtime._parse_compiler_contract(contract_json)
    entry = options.get("kernel_name", "") + _ENTRY_SUFFIXES.get(compiler, "")
    if contract.backend != "cuda":
        _runtime._contract_error("backend is not cuda")
    if contract.entry != entry:
        _runtime._contract_error(
            f"entry {contract.entry!r} does not match the requested {entry!r}"
        )
    block_size = options.get("block_size")
    if block_size is not None and contract.launch.block != (block_size, 1, 1):
        _runtime._contract_error("block does not match the requested size")
    return _Kernel(ptx, contract_json, contract)


def _compile_once(compile_ptx, module_text, *, module=None, **options):
    """Compile one kernel at most once per process.

    Args:
        compile_ptx: Native compile function, looked up by the caller at call
            time. It is part of the key, so a replaced function is called
            instead of being served an earlier result.
        module_text: Semantic module text that identifies the program.
        module: The same program already parsed in an active MLIR context.
            When omitted, the text is parsed only on a miss.
        **options: Keyword arguments of the compile function: the kernel
            name, the target, the block size, and any other code generation
            option. Each one is part of the key.

    Returns:
        The `_Kernel`. A failed compile is not kept. A kernel compiled
        before is returned without taking a lock, so it never waits for
        another thread's compile. While an artifact is selected, the kernel
        comes from the artifact and nothing is compiled.

    Raises:
        RuntimeError: SWAGE_NO_COMPILE=1 is set and this process does not
            hold the kernel, the selected artifact cannot answer the
            request, or the contract does not fit the request. Nothing is
            parsed or compiled.
        ValueError: SWAGE_NO_COMPILE has a value other than 0 or 1.
    """
    key = (compile_ptx, module_text, tuple(sorted(options.items())))
    kernel = _ptx_memo.get(key)
    if kernel is not None:
        return kernel
    with _memo_lock:
        kernel = _ptx_memo.get(key)
        if kernel is not None:
            return kernel
        compiler = getattr(compile_ptx, "__name__", "")
        artifact = _artifact.selected()
        if artifact is not None:
            ptx, contract_json = artifact.kernel(compiler, module_text, options)
        else:
            if _runtime._switch_on("SWAGE_NO_COMPILE"):
                raise _compile_refusal(options)
            if module is None:
                from mlir_swage import ir
                from mlir_swage.dialects import swage

                with ir.Context() as context:
                    swage.register_dialects(context)
                    _, ptx, contract_json = compile_ptx(
                        ir.Module.parse(module_text), **options
                    )
            else:
                _, ptx, contract_json = compile_ptx(module, **options)
        kernel = _ptx_memo[key] = _checked_kernel(
            compiler, ptx, contract_json, options
        )
    return kernel


def _compile_refusal(options):
    """Return the error for a kernel this process may not compile.

    The public launch can answer a miss from the persistent cache. This
    path keeps its kernels in the process only, so a kernel it does not
    hold stays unavailable while compiling is switched off.

    Args:
        options: The code generation options of the refused compile.
    """
    held_for = " and ".join(
        f"{label} {options[name]}"
        for name, label in (("block_size", "block size"), ("target", "target"))
        if name in options
    )
    return _runtime._compile_refusal(
        options.get("kernel_name"),
        f"this process does not hold it for {held_for}, and the private "
        "segmented path has no persistent cache",
    )


def _lease(torch, kernel):
    """Load `kernel` in the current context, or find it loaded, and lease it.

    The caller releases the lease once it no longer launches the kernel; a
    prepared launch keeps it for as long as it lives.
    """
    return _cuda_backend._load_artifact(
        kernel,
        _cuda_backend._get_driver(),
        capturing=_cuda_backend.is_current_stream_capturing(torch),
    )


def _user_arguments(module_text, **by_role):
    """Order the user values of a launch by the parameters of the program.

    Args:
        module_text: The program text.
        **by_role: The value of each parameter by its role: data pointers
            for the buffers and integers for the counts.
    """
    return tuple(
        by_role[role] for role in _programs._parameter_roles(module_text)
    )


def _bind(kernel, user, named):
    """Return the native launch arguments of one kernel, bound by contract.

    Args:
        kernel: The `_Kernel` to launch.
        user: The user values in the parameter order of the program.
        named: Derived counts, plan record pointers, and scratch pointers,
            by contract key. It may hold values the kernel does not take;
            the contract picks the ones it names.

    Raises:
        ValueError: The contract names a value that `named` lacks, or a
            value is out of the range of its kind.
    """
    picked = {"derived": {}, "plan": {}, "scratch": {}}
    for argument in kernel.contract.arguments:
        if argument.origin != "user" and argument.key in named:
            picked[argument.origin][argument.key] = named[argument.key]
    return _abi.materialize_launch_arguments(
        _abi.bind_kernel_contract(kernel.contract, user=user, **picked)
    )


def _enqueue(torch, lease, kernel, arguments, blocks, stream):
    """Enqueue one launch of a leased kernel on `stream`.

    Args:
        torch: The PyTorch module.
        lease: The lease on the loaded kernel.
        kernel: The `_Kernel`, whose contract gives the block.
        arguments: What `_bind` returned for this launch.
        blocks: The number of blocks of the one-dimensional grid.
        stream: The PyTorch stream to enqueue on.
    """
    _cuda_backend._launch_loaded(
        lease.entry,
        kernel.contract,
        arguments,
        (blocks, 1, 1),
        stream.cuda_stream,
        capturing=_cuda_backend.is_current_stream_capturing(torch),
    )
