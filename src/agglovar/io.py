"""I/O utilities."""

__all__ = [
    'PlainOrGzReader',
]

from dataclasses import dataclass, field
import gzip
from pathlib import Path
from typing import Any, Optional


@dataclass
class PlainOrGzReader:
    """Read a plain or a gzipped file using context guard.

    :param file_path: File path or file name.
    :param mode: File open mode.
    :param is_gz: If `None`, determine if file is gzipped based on the file name. Set to `True` to force
        reading gzipped, `False` to force reading plain.

    Example::

        with PlainOrGzReader('path/to/file.gz'): ...
    """

    file_path: str | Path
    mode: str = 'rt'
    is_gz: Optional[bool] = field(default=None)
    _file_handle: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Post initialization."""
        if self.is_gz is None:
            self.is_gz = str(self.file_path).lower().endswith('.gz')

    def __enter__(self):
        """Enter context."""
        if self.is_gz:
            self._file_handle = gzip.open(Path(self.file_path), self.mode)
        else:
            self._file_handle = open(Path(self.file_path), self.mode)

        return self._file_handle

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context."""
        if self._file_handle is not None:
            self._file_handle.__exit__()
            self._file_handle = None
