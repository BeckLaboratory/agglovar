"""Variant utilites."""

__all__ = [
    'id_version_expr',
]

import logging

import polars as pl

logger = logging.getLogger(__name__)


def id_version_expr(
        id_col: str = 'id'
) -> pl.Expr:
    """De-duplicate IDs by appending an integer to ID strings.

    The first appearance of an ID is never modified. The second appearance of an ID gets ".1" appended, the third ".2",
    and so on.

    If any variant IDs are already versioned, then versions are stripped.

    :param id_col: ID column name.

    :returns: An expression for versioning variant IDs.
    """
    expr_id = pl.col(id_col).str.replace(r'\.[0-9]*$', '')

    # Rank on the filter count rather than the filter column itself: ranking a list column
    # yields null for a null filter, which leaves duplicate IDs unversioned, and the count
    # keeps the ordering that matters here -- variants that passed take the unversioned ID
    expr_version = (
        pl.col('filter')
        .list.len()
        .fill_null(0)
        .rank(method='ordinal')
        .over(expr_id)
        - 1
    )

    return (
        pl.when(expr_version > 0)
        .then(pl.concat_str(expr_id, pl.lit('.'), expr_version.cast(pl.String)))
        .otherwise(expr_id)
        .alias(id_col)
    )
