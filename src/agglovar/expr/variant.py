"""Polars expressions used by variant calling routines."""

__all__ = [
    'id_snv_expr',
    'id_nonsnv_expr',
    'sort_cols',
    'id_expr_noalias',
    'id_expr',
]

import polars as pl

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
"""Expression for generating variant IDs including alias "id" for the result (ideal for `with_column()`)."""
