# benchmarks/results/fixed-dispatch-a6000-sm86-3d0fe92/characterize_callable.py
"""Compare live native method lookup with the preserved binding proposal."""

import json
from types import MethodType

import torch
from swage import _benchmark


class LookupChanged(RuntimeError):
    """A caller-installed native launcher method refused the launch."""


def main():
    """Report real CUDA behavior without changing source files."""
    kernel = _benchmark._fixed_vector_add_kernel()
    launch, output, expected, x, y = _benchmark._inputs(torch, kernel, 129, 256)
    launch()
    launch()
    torch.cuda.synchronize()
    owner = kernel._cuda_fast_launch()
    assert owner is not None
    launcher_type = type(owner)
    original = launcher_type.__call__
    bound = MethodType(original, owner)
    calls = []
    result = {
        "launcher_type": str(launcher_type),
        "source": "installed baseline qualification",
    }

    def refuse(self, *args):
        calls.append(self)
        raise LookupChanged("native launcher lookup changed")

    try:
        launcher_type.__call__ = refuse
    except (AttributeError, TypeError) as error:
        result.update(class_mutable=False, refusal=str(error))
    else:
        try:
            output.fill_(-1)
            before = output._version
            try:
                launch()
            except LookupChanged:
                result["ordinary_lookup_refuses"] = True
            else:
                result["ordinary_lookup_refuses"] = False
            result["ordinary_output_unchanged"] = bool(
                torch.all(output == -1).item()
            )
            result["ordinary_version_unchanged"] = output._version == before
            arguments = {"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": 129}
            result["bound_launch_accepted"] = bound(
                arguments, {"BLOCK": 256}, (1,)
            )
            torch.cuda.synchronize()
            result["bound_output_correct"] = bool(torch.equal(output, expected))
            result["bound_version_step"] = output._version - before
            result["override_calls"] = len(calls)
            result["class_mutable"] = True
        finally:
            launcher_type.__call__ = original
    print(json.dumps(result))


if __name__ == "__main__":
    main()
