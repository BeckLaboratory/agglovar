"""Constants for parsing and writing VCF files."""

__vcf_const_all__ = [
    'VCF_VERSION',
    'VCF_SOURCE',
    'VCF_TO_POLARS_TYPE',
    'VCF_SAMPLE_FIXED_SCHEMA',
    'polars_to_vcf_type',
]

from typing import Union

import polars as pl

PolarsDataType = Union[pl.DataType, type[pl.DataType]]

VCF_VERSION: str = '4.2'
"""VCF format version supported by agglovar."""

VCF_SOURCE: str = 'agglovar'
"""Default source string written to VCF headers."""

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
