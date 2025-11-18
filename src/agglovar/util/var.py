"""Variant utilites."""

from typing import Any

import polars as pl

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

    expr_version = (
        pl.col('filter')
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
