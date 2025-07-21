"""
Standard schema for Agglovar data.
"""

import polars as pl

# Schema types for variants
VARIANT = {
    'chrom': pl.String,
    'pos': pl.Int64,
    'end': pl.Int64,
    'id': pl.String,
    'svtype': pl.String,
    'svlen': pl.Int64,
    'ref': pl.String,
    'alt': pl.String,
    'seq': pl.String
}

# Standard fields and column order for variant types
STANDARD_FIELDS = {
    'sv': ['chrom', 'pos', 'end', 'id', 'svtype', 'svlen'],
    'indel': ['chrom', 'pos', 'end', 'id', 'svtype', 'svlen'],
    'snv': ['chrom', 'pos', 'id', 'ref', 'alt'],
}
