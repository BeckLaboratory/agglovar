"""Standard schema for Agglovar data."""

__all__ = [
    'VARIANT',
    'STANDARD_FIELDS',
]

from typing import Union

import polars as pl

PolarsDataType = Union[pl.DataType, type[pl.DataType]]

# Schema types for variants
VARIANT: dict[str, PolarsDataType] = {
    'chrom': pl.String,
    'pos': pl.Int64,
    'end': pl.Int64,
    'id': pl.String,
    'vartype': pl.String,
    'varlen': pl.Int64,
    'ref': pl.String,
    'alt': pl.String,
    'filter': pl.List(pl.String),
    'seq': pl.String,
}
"""Schema for variant tables."""

# Standard fields and column order for variant types
STANDARD_FIELDS: dict[str, tuple[str, ...]] = {
    'snv': ('chrom', 'pos', 'end', 'id', 'vartype', 'ref', 'alt', 'filter'),
    'insdel': ('chrom', 'pos', 'end', 'id', 'vartype', 'varlen', 'filter', 'seq'),  # INS and DEL
    'inv': ('chrom', 'pos', 'end', 'id', 'vartype', 'varlen', 'filter', 'seq'),
    'dup': ('chrom', 'pos', 'end', 'id', 'vartype', 'varlen', 'filter', 'seq'),
    'sub': ('chrom', 'pos', 'end', 'id', 'vartype', 'varlen', 'ref', 'alt', 'filter'),
    'cpx': ('chrom', 'pos', 'end', 'id', 'vartype', 'varlen', 'filter'),
}
"""Standard fields and column order for variant types. Field "insdel" is used for both INS and DEL."""
