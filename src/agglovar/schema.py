"""Standard schema for Agglovar data."""

__all__ = [
    'VARIANT',
    'STANDARD_FIELDS',
]

import polars as pl

# Schema types for variants
VARIANT: dict[str, pl.DataType] = {
    'chrom': pl.String,
    'pos': pl.Int64,
    'end': pl.Int64,
    'id': pl.String,
    'vartype': pl.String,
    'varlen': pl.Int64,
    'ref': pl.String,
    'alt': pl.String,
    'seq': pl.String,
}
"""Schema for variant tables."""

# Standard fields and column order for variant types
STANDARD_FIELDS: dict[str, tuple[str, ...]] = {
    'sv': ('chrom', 'pos', 'end', 'id', 'vartype', 'varlen'),
    'indel': ('chrom', 'pos', 'end', 'id', 'vartype', 'varlen'),
    'snv': ('chrom', 'pos', 'id', 'ref', 'alt'),
}
"""Standard fields and column order for variant types."""
