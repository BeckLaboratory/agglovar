"""Polars expressions used by variant calling routines."""

__all__ = [
    'id_snv_expr',
    'id_nonsnv_expr',
    'sort_cols',
    'id_expr_noalias',
    'id_expr',
    'id_expr_for_table',
    'lead_src_expr',
]

import logging
from typing import Optional

import polars as pl

import agglovar.schema as _agg_schema

logger = logging.getLogger(__name__)

id_snv_expr = (
    pl.concat_str(
        pl.col('chrom'),
        pl.lit('-'),
        pl.col('pos') + 1,
        pl.lit('-SNV-'),
        pl.col('ref').str.to_uppercase(),
        pl.col('alt').str.to_uppercase(),
    )
)
"""Expression for generating SNV IDs."""


id_nonsnv_expr = (
    pl.concat_str(
        pl.col('chrom'),
        pl.lit('-'),
        (pl.col('pos') + 1).cast(pl.String),
        pl.lit('-'),
        pl.col('vartype').str.to_uppercase(),
        pl.lit('-'),
        pl.col('varlen').cast(pl.String)
    )
)
"""Expression for generating non-SNV IDs."""


sort_cols = (
    pl.col('chrom'),
    pl.col('pos'),
    pl.col('end'),
    pl.col('filter').list.len(),
    pl.coalesce(
        pl.col('^id$'),
        pl.lit(None),
    )
)
"""Sort columns for variant tables."""


id_expr_noalias = (
    pl.when(pl.col('vartype').str.to_uppercase() == 'SNV')
    .then(id_snv_expr)
    .otherwise(id_nonsnv_expr)
)
"""Expression for generating variant IDs."""

id_expr = id_expr_noalias.alias('id')
"""Expression for generating variant IDs including alias "id" for the result (ideal for `with_column()`).

Requires both the SNV columns (``ref``, ``alt``) and the non-SNV column (``varlen``) to be present:
Polars resolves both branches of the expression regardless of which one a row selects.  Use
:func:`id_expr_for_table` for single-type tables, which carry only the columns listed in
:data:`agglovar.schema.STANDARD_FIELDS` for their type.
"""


def id_expr_for_table(table: str, alias: Optional[str] = 'id') -> pl.Expr:
    """Get a variant ID expression for a single variant-type table.

    Single-type tables carry only the columns listed for their type in
    :data:`agglovar.schema.STANDARD_FIELDS`, so :data:`id_expr` cannot be applied to them (see its
    documentation).  This routine returns the one branch that matches the table's columns.

    :param table: Table name from :data:`agglovar.schema.STANDARD_FIELDS` (e.g. "snv", "insdel").
    :param alias: Output column name for the resulting IDs, or `None` to leave it unaliased.

    :returns: An expression yielding variant IDs for *table*.

    :raises ValueError: If *table* is not a known variant-type table.
    """
    fields = _agg_schema.STANDARD_FIELDS.get(table)

    if fields is None:
        raise ValueError(
            f'Unknown variant-type table: {table!r}: '
            f'expected one of {", ".join(sorted(_agg_schema.STANDARD_FIELDS))}'
        )

    # Tables without "varlen" (SNV) are identified by their changed bases, all others by length
    expr = id_nonsnv_expr if 'varlen' in fields else id_snv_expr

    return expr.alias(alias) if alias is not None else expr


def lead_src_expr(
        mg_src_col: str = 'mg_src',
        mg_src_lead_col: str = 'mg_src_lead',
        alias: Optional[str] = 'lead_src',
) -> pl.Expr:
    """Extract the lead source entry from a merged callset.

    A merge produces an ``mg_src`` list column with one struct per contributing source and an
    ``mg_src_lead`` column holding the integer position within that list of the lead
    (representative) source.  This routine gathers that lead entry back out as a struct, so an API
    user can read the lead source directly instead of manually indexing the list with
    ``pl.col('mg_src').list.get(pl.col('mg_src_lead'))``.

    Agglovar does not use this internally; the merge stores only the index to avoid duplicating the
    entry.  It is provided as a convenience for downstream API users.

    :param mg_src_col: Name of the ``mg_src`` list column.
    :param mg_src_lead_col: Name of the ``mg_src_lead`` index column.
    :param alias: Output column name for the resulting struct, or `None` to leave it unaliased.

    :returns: An expression yielding the lead source struct, carrying the same fields as an
        ``mg_src`` entry ("src_index", "src_name", "src_meta", "var_index", "var_id").
    """
    expr = pl.col(mg_src_col).list.get(pl.col(mg_src_lead_col))

    if alias is not None:
        expr = expr.alias(alias)

    return expr
