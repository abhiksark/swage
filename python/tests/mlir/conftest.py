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
