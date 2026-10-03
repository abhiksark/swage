# python/swage/_errors.py
"""Stable exceptions for compiler and runtime environment failures."""


class SwageError(RuntimeError):
    """Base class for Swage compiler and runtime failures."""


class CompilationError(SwageError):
    """A source-located error in a Swage kernel definition."""


class BackendUnavailableError(SwageError):
    """An unavailable environment prerequisite, not a compilation failure."""

    def __init__(self, message, *, code, backend, remediation):
        self.code = code
        self.backend = backend
        self.remediation = remediation
        super().__init__(f"{message}; {remediation}")
