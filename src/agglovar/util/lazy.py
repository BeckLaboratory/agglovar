"""Shared materialization helpers for LazyFrame-based join pipelines.

Chunked joins re-evaluate their input ``LazyFrame`` graphs every time the
inner loop calls ``.collect()`` on a derived expression. When the source is a
``scan_parquet`` (or any frame with ``with_row_index`` blocking predicate
pushdown) this means a full source scan per chunk. The fix is to materialise
each input table exactly once before the chunk loop. ``materialize_pair`` is
the single-source-of-truth for that policy across :mod:`agglovar.pairwise`
and :mod:`agglovar.bed`.

The ``temp_dir`` argument has the same meaning everywhere it appears:

* ``False`` (default): collect both tables into memory and yield as
  ``LazyFrame`` (``df.collect().lazy()``). Lowest latency; uses RAM
  proportional to the input size.
* ``True``: sink both tables to temporary parquet files in the system temp
  directory and yield ``scan_parquet`` over them. Files are unlinked when
  the context exits (even on error).
* ``str`` or ``pathlib.Path``: same as ``True`` but in the given directory.
"""

from __future__ import annotations

__all__ = [
    'materialize_pair',
]

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator

import polars as pl


@contextmanager
def materialize_pair(
        df_a: pl.LazyFrame,
        df_b: pl.LazyFrame,
        temp_dir: bool | str | Path = False,
        prefix: str = 'agglovar_lazy_',
) -> Iterator[tuple[pl.LazyFrame, pl.LazyFrame]]:
    """Yield ``(df_a, df_b)`` materialised once according to ``temp_dir``.

    :param df_a: First lazy table.
    :param df_b: Second lazy table.
    :param temp_dir: Materialisation policy. See module docstring.
    :param prefix: Filename prefix used for temp parquet files (only when
        ``temp_dir`` is truthy).

    :yields: A pair of ``LazyFrame`` instances backed by the materialised data.
    """
    if temp_dir is False:
        yield df_a.collect().lazy(), df_b.collect().lazy()
        return

    temp_dir_path = None if temp_dir is True else Path(temp_dir)

    with tempfile.NamedTemporaryFile(
        prefix=f'{prefix}a_', suffix='.parquet', dir=temp_dir_path, delete=False,
    ) as f:
        path_a = Path(f.name)

    with tempfile.NamedTemporaryFile(
        prefix=f'{prefix}b_', suffix='.parquet', dir=temp_dir_path, delete=False,
    ) as f:
        path_b = Path(f.name)

    try:
        df_a.sink_parquet(path_a)
        df_b.sink_parquet(path_b)
        yield pl.scan_parquet(path_a), pl.scan_parquet(path_b)
    finally:
        path_a.unlink(missing_ok=True)
        path_b.unlink(missing_ok=True)
