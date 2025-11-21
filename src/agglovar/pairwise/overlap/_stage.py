"""One merge stage in a pairwise overlap strategy."""

__all__ = [
    'PairwiseOverlapStage'
]

from typing import Optional

import polars as pl

from ...meta.decorators import immutable
from ...meta.descriptors import BoundedFloat, CheckedBool, BoundedInt

from ._const import (
    EXPR_OVERLAP_RO,
    EXPR_SZRO,
    EXPR_OFFSET_DIST,
    EXPR_OFFSET_PROP,
)


@immutable
class PairwiseOverlapStage():
    """One join strategy implemented by overlap.

    :ivar ro_min: Minimum reciprocal overlap for allowed matches. If 0.0, then any overlap matches.
    :ivar size_ro_min: Reciprocal length proportion of allowed matches. If `match_prop_min` is set
        and the value of this parameter is `None` or is less than `match_prop_min`, then it is set
        to `match_prop_min` since this value represents the lower-bound of allowed match
        proportions.
    :ivar offset_max: Maximum offset allowed (minimum of start or end position distance).
    :ivar offset_prop_max: Maximum size-offset (offset / varlen) allowed.
    :ivar match_ref: "REF" column must match between two variants.
    :ivar match_alt: "ALT" column must match between two variants.
    :ivar match_prop_min: Minimum matched base proportion in alignment or None to not match.
    """
    ro_min: Optional[float] = BoundedFloat(min_val=0.0, max_val=1.0)
    size_ro_min: Optional[float] = BoundedFloat(min_val=(0.0, False), max_val=1.0)
    offset_max: Optional[int] = BoundedInt(0)
    offset_prop_max: Optional[float] = BoundedFloat(0.0)
    seg_ro_min: Optional[float] = BoundedFloat(min_val=0.0, max_val=1.0)
    match_ref: bool = CheckedBool()
    match_alt: bool = CheckedBool()
    match_prop_min: Optional[float]

    # Table and Join Control
    join_predicates: tuple[pl.Expr, ...]
    join_filters: tuple[pl.Expr, ...]
    chunk_range: tuple[
        tuple[str, str, tuple[pl.Expr, ...]],
        ...
    ]

    def __init__(
            self,
            *,
            ro_min: Optional[float] = None,
            size_ro_min: Optional[float] = None,
            offset_max: Optional[int] = None,
            offset_prop_max: Optional[float] = None,
            seg_ro_min: Optional[float] = None,
            match_ref: bool = False,
            match_alt: bool = False,
            match_prop_min: Optional[float] = None,
    ) -> None:
        self.ro_min = ro_min
        self.size_ro_min = size_ro_min
        self.offset_max = offset_max
        self.offset_prop_max = offset_prop_max
        self.match_ref = match_ref
        self.match_alt = match_alt
        self.seg_ro_min = seg_ro_min
        self.match_prop_min = match_prop_min

        # Set join control containers
        join_predicates_list: list[pl.Expr] = []
        join_filters_list = []
        chunk_range_dict: dict[tuple[str, str], list[pl.Expr]] = {}

        # Set ranges and expressions from parameters
        if self.ro_min is not None:
            join_predicates_list.append(
                EXPR_OVERLAP_RO >= self.ro_min
            )

            self._append_chunk_range(
                'pos_b', 'max',
                pl.col('_end_ro_a'),
                chunk_range_dict
            )

            self._append_chunk_range(
                '_end_ro_b', 'min',
                pl.col('pos_a'),
                chunk_range_dict
            )

        if self.size_ro_min is not None:
            join_predicates_list.append(
                EXPR_SZRO >= self.size_ro_min
            )

            self._append_chunk_range(
                'varlen_b', 'min',
                pl.col('varlen_a') * self.size_ro_min,
                chunk_range_dict
            )

            self._append_chunk_range(
                'varlen_b', 'max',
                pl.col('varlen_a') * (1 / self.size_ro_min),
                chunk_range_dict
            )

        if self.offset_max is not None:
            if self.offset_max == 0:
                join_predicates_list.extend([  # Very fast joins on equality
                    pl.col('pos_a') == pl.col('pos_b'),
                    pl.col('end_a') == pl.col('end_b'),
                ])
            else:
                join_predicates_list.append(
                    EXPR_OFFSET_DIST <= self.offset_max
                )

            self._append_chunk_range(
                'pos_b', 'min',
                pl.col('pos_a') - self.offset_max,
                chunk_range_dict
            )

            self._append_chunk_range(
                'end_b', 'max',
                pl.col('end_a') + self.offset_max,
                chunk_range_dict
            )

        if self.offset_prop_max is not None:
            join_predicates_list.append(
                EXPR_OFFSET_PROP <= self.offset_prop_max
            )

            self._append_chunk_range(
                'pos_b', 'min',
                pl.col('pos_a') - pl.col('varlen_a') * self.offset_prop_max,
                chunk_range_dict
            )

            self._append_chunk_range(
                'end_b', 'max',
                pl.col('end_a') + pl.col('varlen_a') * self.offset_prop_max,
                chunk_range_dict
            )

        if self.match_ref:
            join_predicates_list.append(
                pl.col('ref_a') == pl.col('ref_b')
            )

        if self.match_alt:
            join_predicates_list.append(
                pl.col('alt_a') == pl.col('alt_b')
            )

        if seg_ro_min is not None:
            join_filters_list.append(
                pl.col('seg_ro') >= self.seg_ro_min
            )

        if self.match_prop_min is not None and self.match_prop_min > 0.0:
            join_filters_list.append(
                pl.col('match_prop') >= self.match_prop_min
            )

        # Finalize
        self.join_predicates = tuple(join_predicates_list)
        self.join_filters = tuple(join_filters_list)
        self.chunk_range = tuple(
            tuple((str(col), str(limit), tuple(exprs)))
            for (col, limit), exprs in chunk_range_dict.items()
        )

    # def append_join_predicates(
    #         self,
    #         expr: Iterable[pl.Expr] | pl.Expr
    # ) -> None:
    #     """Append expressions to a list of join predicates given as arguments to pl.join_where().
    #
    #     This class will construct a list of join predicates from the constructor arguments,
    #     but additional join control may be added here.
    #
    #     .. Warning::
    #         Adding predicates may alter the join results so that they are not reproducible
    #         based on join arguments. Use with caution.
    #
    #     :param expr: An expression or list of expressions.
    #     """
    #     if isinstance(expr, pl.Expr):
    #         expr = [expr]
    #
    #     self._add_expected_cols(expr)
    #     self._join_predicates.extend(expr)

    @property
    def has_match(self) -> bool:
        return self.match_prop_min is not None

    @property
    def has_seg_ro(self) -> bool:
        return self.seg_ro_min is not None

    def _append_chunk_range(
            self,
            key: str,
            limit: str,
            expr: pl.Expr,
            chunk_range_dict: dict[tuple[str, str], list[pl.Expr]]
    ) -> None:
        """Append a rule for chunking df_b based on a subset of df_a.

        :param key: Column name in df_b without "_b" suffix (e.g. "pos" for "pos_a").
        :param limit: If the limit is "min" or "max".
        :param expr: An expression applied to df_a to determine a minimum or maximum value for column "key" in df_b.
        """
        if limit not in {'min', 'max'}:
            raise ValueError(f'Limit must be "min" or "max": {limit}')

        if not (key := key.strip() if key else None):
            raise ValueError('Key must not be empty')

        if (key, limit) not in chunk_range_dict:
            chunk_range_dict[key, limit] = []

        chunk_range_dict[key, limit].append(expr)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'ro_min={self.ro_min}, '
            f'size_ro_min={self.size_ro_min}, '
            f'offset_max={self.offset_max}, '
            f'offset_prop_max={self.offset_prop_max}, '
            f'match_ref={self.match_ref}, '
            f'match_alt={self.match_alt}, '
            f'match_prop_min={self.match_prop_min},)'
        )
