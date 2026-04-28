"""Brute-force Python reference implementations for ``agglovar.bed``.

These mirror the *intended* behaviour of the bed module so tests can compare
the polars implementations against an independent oracle. They are slow
(O(N*M)) and only suitable for the small synthetic datasets used in tests.

Conventions (BED half-open ``[pos, end)``):

- Two records overlap or touch when ``pos_b - distance <= end_a`` and
  ``end_b + distance >= pos_a`` (the same predicate the polars
  ``pairwise_join_iter`` filter uses). Touching at ``distance=0`` is included.
- ``output distance = max(pos_a, pos_b) - min(end_a, end_b)``: negative for
  proper overlap, zero for touching, positive for gaps within ``distance``.
- ``as_proportion`` divides the covered base count of ``[pos_a, end_a)`` by
  ``end_a - pos_a``; zero-length A intervals produce ``NaN``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Optional

import polars as pl


@dataclass(frozen=True)
class JoinRow:
    """One row of the expected ``pairwise_join`` table."""

    index_a: int
    index_b: int
    chrom: str
    pos: int
    end: int
    distance: int


def _coord_cols(name: Optional[str | tuple[str, str, str]]) -> tuple[str, str, str]:
    """Resolve a ``CoordCol``-like spec to ``(chrom, pos, end)`` column names.

    Mirrors :func:`agglovar.bed.col.get_coord_cols` for oracle inputs.
    """
    if name is None or name == 'ref':
        return ('chrom', 'pos', 'end')
    if name == 'qry':
        return ('qry_id', 'qry_pos', 'qry_end')
    chrom, pos, end = name
    return (chrom, pos, end)


def _rows(
        df: pl.DataFrame | pl.LazyFrame,
        cols: tuple[str, str, str],
) -> list[tuple[str, int, int]]:
    """Materialize ``(chrom, pos, end)`` triples from an input frame."""
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    chrom_col, pos_col, end_col = cols
    return list(
        df.select(pl.col(chrom_col), pl.col(pos_col), pl.col(end_col)).iter_rows()
    )


def expected_join(
        df_a: pl.DataFrame | pl.LazyFrame,
        df_b: pl.DataFrame | pl.LazyFrame,
        distance: int = 0,
        col_names_a: Optional[str | tuple[str, str, str]] = None,
        col_names_b: Optional[str | tuple[str, str, str]] = None,
) -> list[JoinRow]:
    """Return all join records as a list of :class:`JoinRow`.

    Applies the final-filter predicate ``B.pos - distance <= A.end and
    B.end + distance >= A.pos``. The list is sorted by
    ``(index_a, index_b)`` for deterministic comparison.
    """
    cols_a = _coord_cols(col_names_a)
    cols_b = _coord_cols(col_names_b)

    rows_a = _rows(df_a, cols_a)
    rows_b = _rows(df_b, cols_b)

    out: list[JoinRow] = []

    for i_a, (chrom_a, pos_a, end_a) in enumerate(rows_a):
        if chrom_a is None or pos_a is None or end_a is None:
            continue

        for i_b, (chrom_b, pos_b, end_b) in enumerate(rows_b):
            if chrom_a != chrom_b:
                continue
            if pos_b is None or end_b is None:
                continue

            if pos_b - distance > end_a:
                continue
            if end_b + distance < pos_a:
                continue

            inter_pos = max(pos_a, pos_b)
            inter_end = min(end_a, end_b)
            output_distance = inter_pos - inter_end

            out_pos = min(inter_pos, inter_end)
            out_end = max(inter_pos, inter_end)

            out.append(
                JoinRow(
                    index_a=i_a,
                    index_b=i_b,
                    chrom=chrom_a,
                    pos=out_pos,
                    end=out_end,
                    distance=output_distance,
                )
            )

    out.sort(key=lambda r: (r.index_a, r.index_b))
    return out


def expected_join_pairs(
        df_a: pl.DataFrame | pl.LazyFrame,
        df_b: pl.DataFrame | pl.LazyFrame,
        distance: int = 0,
        col_names_a: Optional[str | tuple[str, str, str]] = None,
        col_names_b: Optional[str | tuple[str, str, str]] = None,
) -> set[tuple[int, int]]:
    """Return the set of ``(index_a, index_b)`` pairs from the expected join."""
    return {
        (row.index_a, row.index_b)
        for row in expected_join(df_a, df_b, distance, col_names_a, col_names_b)
    }


def expected_as_bool(
        df_a: pl.DataFrame | pl.LazyFrame,
        df_b: pl.DataFrame | pl.LazyFrame,
        distance: int = 0,
        negate: bool = False,
        col_names_a: Optional[str | tuple[str, str, str]] = None,
        col_names_b: Optional[str | tuple[str, str, str]] = None,
) -> list[bool]:
    """Per-row boolean indicating whether each row in ``df_a`` has a hit.

    Order matches input row order of ``df_a``.
    """
    cols_a = _coord_cols(col_names_a)
    cols_b = _coord_cols(col_names_b)

    rows_a = _rows(df_a, cols_a)
    rows_b = _rows(df_b, cols_b)

    by_chrom: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chrom_b, pos_b, end_b in rows_b:
        if chrom_b is None or pos_b is None or end_b is None:
            continue
        by_chrom[chrom_b].append((pos_b, end_b))

    hit_val = (not negate)
    miss_val = negate
    out: list[bool] = []

    for chrom_a, pos_a, end_a in rows_a:
        if chrom_a is None or pos_a is None or end_a is None:
            out.append(miss_val)
            continue

        hit = False
        for pos_b, end_b in by_chrom.get(chrom_a, ()):
            if pos_b - distance <= end_a and end_b + distance >= pos_a:
                hit = True
                break

        out.append(hit_val if hit else miss_val)

    return out


def _merge_intervals(
        intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping/touching half-open intervals; assume non-empty list."""
    intervals = sorted(intervals)
    out: list[tuple[int, int]] = []
    cur_pos, cur_end = intervals[0]

    for pos, end in intervals[1:]:
        if pos <= cur_end:
            cur_end = max(cur_end, end)
        else:
            out.append((cur_pos, cur_end))
            cur_pos, cur_end = pos, end

    out.append((cur_pos, cur_end))
    return out


def expected_as_proportion(
        df_a: pl.DataFrame | pl.LazyFrame,
        df_b: pl.DataFrame | pl.LazyFrame,
        col_names_a: Optional[str | tuple[str, str, str]] = None,
        col_names_b: Optional[str | tuple[str, str, str]] = None,
) -> list[Optional[float]]:
    """Per-row proportion of ``[pos_a, end_a)`` covered by union of B intervals.

    For zero-length A intervals returns ``NaN`` (matches the polars
    implementation's ``0 / 0`` behaviour). Rows with null ``pos`` or ``end``
    in A are dropped (matches the current ``as_proportion`` behaviour).
    """
    cols_a = _coord_cols(col_names_a)
    cols_b = _coord_cols(col_names_b)

    rows_a = _rows(df_a, cols_a)
    rows_b = _rows(df_b, cols_b)

    by_chrom_b: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chrom_b, pos_b, end_b in rows_b:
        if chrom_b is None or pos_b is None or end_b is None:
            continue
        if end_b > pos_b:
            by_chrom_b[chrom_b].append((pos_b, end_b))

    merged_b: dict[str, list[tuple[int, int]]] = {
        chrom: _merge_intervals(intervals)
        for chrom, intervals in by_chrom_b.items()
    }

    out: list[Optional[float]] = []

    for chrom_a, pos_a, end_a in rows_a:
        if chrom_a is None or pos_a is None or end_a is None:
            continue

        length = end_a - pos_a

        covered = 0
        for pos_b, end_b in merged_b.get(chrom_a, ()):
            inter_pos = max(pos_a, pos_b)
            inter_end = min(end_a, end_b)
            if inter_end > inter_pos:
                covered += inter_end - inter_pos

        if length <= 0:
            out.append(math.nan)
        else:
            out.append(covered / length)

    return out


@dataclass(frozen=True)
class MergedRegion:
    """One row of the expected ``merge_depth`` table."""

    chrom: str
    pos: int
    end: int
    max_depth: int


def expected_merge_depth(
        df: pl.DataFrame | pl.LazyFrame,
        distance: int = 0,
        col_names: Optional[str | tuple[str, str, str]] = None,
) -> list[MergedRegion]:
    """Reference merge_depth: pad ends by ``distance``, sweep per chrom.

    Returns rows sorted by ``(chrom, pos, end)`` to match
    :func:`agglovar.bed.merge.merge_depth`.
    """
    if distance < 0:
        raise ValueError('Oracle does not handle distance < 0 for merge_depth')

    cols = _coord_cols(col_names)
    rows = _rows(df, cols)

    by_chrom: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chrom, pos, end in rows:
        if chrom is None or pos is None or end is None:
            continue
        by_chrom[chrom].append((pos, end))

    out: list[MergedRegion] = []

    for chrom, intervals in by_chrom.items():
        events: list[tuple[int, int, int]] = []
        for pos, end in intervals:
            events.append((pos, 0, +1))
            events.append((end + distance, 1, -1))

        # Sort: by location, then +1 before -1 so adjacent intervals merge
        # (matches melt_depth's depth-descending sort).
        events.sort(key=lambda x: (x[0], x[1]))

        depth = 0
        region_start: Optional[int] = None
        region_max_depth = 0
        region_max_loc: Optional[int] = None

        for loc, _kind, delta in events:
            depth += delta

            if delta > 0:
                if region_start is None:
                    region_start = loc
                if depth > region_max_depth:
                    region_max_depth = depth

            if region_start is not None and (region_max_loc is None or loc > region_max_loc):
                region_max_loc = loc

            if depth == 0 and region_start is not None:
                out.append(
                    MergedRegion(
                        chrom=chrom,
                        pos=region_start,
                        end=region_max_loc - distance,
                        max_depth=region_max_depth,
                    )
                )
                region_start = None
                region_max_depth = 0
                region_max_loc = None

    out.sort(key=lambda r: (r.chrom, r.pos, r.end))
    return out
