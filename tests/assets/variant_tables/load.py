"""Utilities functions for loading variant tables."""

import os
import polars as pl

import agglovar

VAR_ASSET_DIR = 'tests/assets/variant_tables'

VAR_ADDITIONAL_FIELDS = {
    'table_name': pl.String,
    'comment': pl.String
}

def load_variant_table(
        vartype: str,
        table_name: str,
        allow_empty: bool=False
) -> pl.DataFrame:
    """Read a variant table.

    :param vartype: Variant type.
    :param table_name: Table name.
    :param allow_empty: If table is empty (i.e. no variants with "table_name"), raise an error.

    :return: Variant table.
    """

    filename = os.path.join(VAR_ASSET_DIR, f'variant_tables_{vartype}.csv')

    if not os.path.exists(filename):
        raise ValueError(f'Could not find variant table for vartype "{vartype}": {filename}')

    df = (
        pl.read_csv(
            filename, has_header=True,
            schema_overrides=agglovar.schema.VARIANT | VAR_ADDITIONAL_FIELDS
        )
        .filter(pl.col('table_name') == table_name)
        .drop(list(VAR_ADDITIONAL_FIELDS.keys()))
    )

    if not allow_empty and df.height == 0:
        raise ValueError(f'No variants for vartype "{vartype}" and table_name "{table_name}"')

    return df
