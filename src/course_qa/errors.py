"""Typed operational errors used by the terminal exit policy."""


class IndexStateError(ValueError):
    """No usable or compatible active snapshot (exit 3)."""


class MissingIndexError(FileNotFoundError):
    """Preserve the missing-file API while assigning missing indexes exit 3."""


class SourceReadError(ValueError):
    """A selected source cannot be extracted completely (exit 4)."""
