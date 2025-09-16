"""
Run and reconcile multiple pairwise intersects.
"""

from dataclasses import dataclass, field

import polars as pl

from .pair import PairwiseIntersect

# Priority order is a tuple of:
# 1. Column name
# 2. Weight
# 3. Location of weight 0.0 (weight is inversely proportional to field if not None)
DEFAULT_PRIORITY_MATCH = [
    ('ro', 0.2, None),
    ('size_ro', 0.2, None),
    ('offset_prop', 0.1, 2.0),
    ('match_prop', 0.5, None)
]

DEFAULT_PRIORITY_NOMATCH = [
    ('ro', 0.4, None),
    ('size_ro', 0.4, None),
    ('offset_prop', 0.2, 2.0),
]

@dataclass(frozen=True)
class MultiPair:
    """
    Run multiple pairwise joins and produce a unified join table.

    Attributes:
        is_locked: `True` if this object is locked and cannot be modified.
    """

    _pairwise_list: list[tuple[PairwiseIntersect, pl.Expr]] = field(default_factory=list, init=False,repr=False)
    is_locked: bool = field(default=False, init=False, repr=False)

    def __init__(self):
        pass

    def add_join(
            self,
            intersect: PairwiseIntersect,
            weight_expr: pl.Expr|list[tuple[str,float,float]]=None
    ) -> None:
        """
        Add a pairwise intersect.

        :param intersect: Intersect object.
        :param weight_expr: Expression to compute weights for prioritizing join records.
        """

        if self.is_locked:
            raise AttributeError('Pairwise intersect object is locked and cannot be modified.')

        if weight_expr is None:
            weight_expr = get_weight_expr(
                DEFAULT_PRIORITY_MATCH if intersect.has_match_prop else DEFAULT_PRIORITY_NOMATCH
            )

        elif isinstance(weight_expr, list):
            weight_expr = get_weight_expr(weight_expr)

        elif not isinstance(weight_expr, pl.Expr):
            raise TypeError(
                f'weight_expr must be an expression or list of (column, weight, max_value): {type(weight_expr)}'
            )

        self._pairwise_list.append((intersect, weight_expr))

    def lock(self) -> None:
        """Locks this object to prevent changes."""
        object.__setattr__(self, 'is_locked', True)

    def join(
            self,
            df_a: pl.DataFrame,
            df_b: pl.DataFrame,
            collect: bool=False
    ) -> pl.DataFrame:
        """
        Join all pairwise intersects.

        :return: A join table.
        """

        join_list = list()

        for i, (pairwise_join, weight_expr) in enumerate(self._pairwise_list):
            df_join = (
                pairwise_join.join(df_a, df_b)
                .with_columns(source_index=i, join_weight=weight_expr)
                .sort(['index_a', 'index_b', 'join_weight'], descending=[False, False, True])
            )

            if collect:
                df_join = df_join.collect().lazy()

            join_list.append(df_join)

        return (
            pl.merge_sorted(join_list, 'index_a')
            .sort(['index_a', 'index_b', 'join_weight', 'source_index'], descending=[False, False, True, False])
            .unique(['index_a', 'index_b'], keep='first')
        )

def get_weight_expr(
    priority: list[tuple[str,float,float]]
):
    """
    Get an expression to compute weights for prioritizing join records.

    :param priority: A list of (column, weight, max_value) tuples.

    :return: An expression to compute weights.
    """

    return pl.sum_horizontal(
        *[
            (
                pl.col(col).cast(pl.Float32) * float(weight)
            ) if max_value is not None else (
                (
                    (1 - pl.col(col).clip(0.0, max_value) / max_value).cast(pl.Float32)
                ) * float(weight)
            ) for col, weight, max_value in priority
        ]
    )
