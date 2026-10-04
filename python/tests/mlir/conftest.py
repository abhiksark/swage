# python/tests/mlir/conftest.py
"""Shared fixtures for the native binding and CUDA tests."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_kernel_cache(tmp_path_factory):
    """Keep the run away from the user's persistent kernel cache.

    Without SWAGE_CACHE_DIR the runtime caches compiled kernels under the
    user's cache directory. A test run would then write there, and a kernel
    left by an earlier run would be loaded from disk instead of compiled.
    The whole run therefore shares one fresh cache directory, which
    processes spawned by a test inherit.

    A caller that sets SWAGE_CACHE_DIR chose that directory deliberately,
    for example a workflow that inspects it after the run, so a value that
    is already set is left alone.

    Args:
        tmp_path_factory: Session-scoped pytest factory for temporary paths.

    Yields:
        None, once the cache directory is settled for the session.
    """
    if os.environ.get("SWAGE_CACHE_DIR"):
        yield
        return
    with pytest.MonkeyPatch.context() as patch:
        cache_dir = tmp_path_factory.mktemp("swage-cache")
        patch.setenv("SWAGE_CACHE_DIR", str(cache_dir))
        yield


@pytest.fixture(scope="session", autouse=True)
def _shared_identity_ids():
    """Upload the shared segment ids before any test replaces a constructor.

    The private runner uploads one tensor of segment ids per CUDA device at
    its first preparation and keeps it for the process. A test that replaces
    `torch.tensor` to observe or to fake the uploads of a preparation would
    otherwise see that upload too, or leave its fake behind as the shared
    ids, depending on which test runs first.
    """
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        from swage import _segmented_plan as _plan

        _plan._identity_ids(torch, torch.device("cuda", 0), 1)
