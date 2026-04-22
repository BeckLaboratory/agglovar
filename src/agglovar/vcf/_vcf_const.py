"""Constants for parsing and writing VCF files."""

__vcf_const_all__ = [
    'VCF_VERSION',
    'VCF_SOURCE',
    'VCF_TO_POLARS_TYPE',
    'VCF_BASE_FIXED_SCHEMA',
    'VCF_SAMPLE_FIXED_SCHEMA',
    'polars_to_vcf_type',
]

import polars as pl
from polars.type_aliases import PolarsDataType

from .. import schema as _agg_schema

VCF_VERSION: str = '4.2'
"""VCF format version supported by agglovar."""

VCF_SOURCE: str = 'agglovar'
"""Default source string written to VCF headers."""

VCF_BASE_FIXED_SCHEMA: dict[str, PolarsDataType] = {
    # agglovar VARIANT columns (derived during read)
    'chrom':   _agg_schema.VARIANT['chrom'],    # pl.String
    'pos':     _agg_schema.VARIANT['pos'],      # pl.Int64, 0-based
    'end':     _agg_schema.VARIANT['end'],      # pl.Int64, 0-based half-open
    'vartype': _agg_schema.VARIANT['vartype'],  # pl.String
    'varlen':  _agg_schema.VARIANT['varlen'],   # pl.Int64, null for SNV/SUB
    'ref':     _agg_schema.VARIANT['ref'],      # pl.String, non-null for SNV/SUB
    'alt':     _agg_schema.VARIANT['alt'],      # pl.String, non-null for SNV/SUB
    'filter':     pl.List(pl.String), # null = unknown; [] = PASS; [ids] = failed filters
    'seq':     _agg_schema.VARIANT['seq'],      # pl.String, non-null for INS/DEL

    # VCF raw columns
    'vcf_pos':    pl.Int64,           # 1-based, as written in the VCF file
    'vcf_id':     pl.String,          # null when VCF ID is '.'
    'vcf_ref':    pl.String,
    'vcf_alt':    pl.String,
    'vcf_qual':   pl.Float32,         # null when VCF QUAL is '.'
    'vcf_rec':    pl.Int64,           # 0-based index of originating VCF record
    'vcf_allele': pl.Int32,           # 1-based ALT index within the originating record
}
"""
Fixed-column schema for the base table returned by :func:`agglovar.vcf.iter_vcf`.

The base table contains one row per alternate allele. Columns not starting with "vcf_" derive their
types from :data:`agglovar.schema.VARIANT`. Columns starting with "vcf_" carry the raw VCF fields.

All coordinate columns (``pos``, ``end``) use 0-based half-open BED-like coordinates. ``vcf_pos``
preserves the original 1-based VCF POS so individual records can be retrieved from the file via
tools like *bcftools*.

The ``filter`` column uses the same convention as :data:`agglovar.schema.VARIANT`:
``null`` means filter status is unknown (VCF ``'.'``), ``[]`` means the record
passed all filters (VCF ``'PASS'``), and a non-empty list contains the IDs of
failed filters.
"""

VCF_SAMPLE_FIXED_SCHEMA: dict[str, PolarsDataType] = {
    'vcf_rec':    pl.Int64,   # links to base table via vcf_rec
    'vcf_sample': pl.String,  # sample name
}
"""
Fixed-column schema for the sample table returned by :func:`agglovar.vcf.iter_vcf`.

The sample table uses long format: one row per ``(vcf_rec, sample_name)`` pair.
Dynamic FORMAT columns (e.g. ``GT``, ``GQ``) for every FORMAT field declared in
the VCF header are appended after these fixed columns.
"""

VCF_TO_POLARS_TYPE: dict[str, PolarsDataType] = {
    'Integer': pl.Int64,
    'Float':   pl.Float64,
    'Flag':    pl.Boolean,
    'Character': pl.String,
    'String':  pl.String,
}
"""Map VCF INFO/FORMAT type strings to Polars data types."""


def polars_to_vcf_type(dtype: PolarsDataType) -> str:
    """
    Return a VCF type string for a Polars data type.

    All integer types map to ``"Integer"``, all float types map to ``"Float"``,
    :class:`polars.Boolean` maps to ``"Flag"``, and string-like types
    (:class:`polars.String`, :class:`polars.Categorical`, :class:`polars.Enum`) map to
    ``"String"``.

    :param dtype: A Polars data type (class or instance).

    :returns: One of ``"Integer"``, ``"Float"``, ``"Flag"``, or ``"String"``.

    :raises TypeError: If ``dtype`` has no corresponding VCF type.
    """
    if dtype.is_integer():
        return 'Integer'

    if dtype.is_float():
        return 'Float'

    if dtype == pl.Boolean:
        return 'Flag'

    if dtype in {pl.String, pl.Categorical, pl.Enum}:
        return 'String'

    raise TypeError(
        f'No VCF type mapping for Polars type {dtype!r}: '
        f'Supported types are integer, float, Boolean, String, Categorical, and Enum'
    )
