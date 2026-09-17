"""Constants for parsing and writing VCF files."""

__all__ = [
    'VCF_VERSION',
    'VCF_SOURCE',
    'VCF_TO_POLARS_TYPE',
    'VCF_PERCENT_ESCAPES',
    'VCF_SAMPLE_FIXED_SCHEMA',
    'polars_to_vcf_type',
    'vcf_percent_decode',
    'vcf_percent_encode',
]

import logging
import re
from typing import Union

import polars as pl

logger = logging.getLogger(__name__)

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

VCF_PERCENT_ESCAPES: dict[str, str] = {
    '%3A': ':',
    '%3B': ';',
    '%3D': '=',
    '%25': '%',
    '%2C': ',',
    '%0D': '\r',
    '%0A': '\n',
    '%09': '\t',
}
"""Percent escapes the VCF specification defines for INFO and FORMAT values.

Section 1.2 reserves these eight characters inside a value: they delimit fields, keys, values and
lists, so a value carrying one has to encode it.  This is the complete set -- a value may contain
any other character literally, including a ``%`` that starts no escape, which is why
:func:`vcf_percent_decode` matches these sequences rather than decoding every ``%XX``."""


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


_PERCENT_DECODE_PATTERN: re.Pattern = re.compile(
    '|'.join(VCF_PERCENT_ESCAPES), re.IGNORECASE
)
"""Matches any escape in :data:`VCF_PERCENT_ESCAPES`, case-insensitively for tolerant reading."""

_PERCENT_ENCODE_MAP: dict[str, str] = {v: k for k, v in VCF_PERCENT_ESCAPES.items()}
"""Reverse of :data:`VCF_PERCENT_ESCAPES`: the character to encode mapped to its escape."""

_PERCENT_ENCODE_PATTERN: re.Pattern = re.compile(
    f'[{re.escape("".join(_PERCENT_ENCODE_MAP))}]'
)
"""Matches any character that must be percent-encoded inside an INFO or FORMAT value."""


def vcf_percent_decode(value: str) -> str:
    """
    Decode the percent escapes the VCF specification defines for INFO and FORMAT values.

    Only the eight sequences in :data:`VCF_PERCENT_ESCAPES` are decoded, unlike
    :func:`urllib.parse.unquote`, which would decode every ``%XX``.  A value written by a
    non-conforming caller may hold a literal ``%`` followed by hex digits: ``'5%AF'`` reads back
    unchanged here, where a general decoder would replace it with a byte that was never in the
    file.  An escape left undecoded is the lesser problem.

    Replacements are not rescanned, so an encoded escape survives one round trip: ``'%253B'``
    decodes to ``'%3B'`` rather than to ``';'``.

    :param value: Value read from an INFO or FORMAT field.

    :returns: The value with its escapes replaced.
    """
    if '%' not in value:
        return value

    return _PERCENT_DECODE_PATTERN.sub(
        lambda m: VCF_PERCENT_ESCAPES[m.group(0).upper()], value
    )


def vcf_percent_encode(value: str) -> str:
    """
    Encode the characters the VCF specification reserves inside INFO and FORMAT values.

    The inverse of :func:`vcf_percent_decode`.  Each character is replaced once and replacements
    are not rescanned, so a literal ``'%'`` becomes ``'%25'`` without the escapes written around it
    being encoded a second time.

    :param value: Value to write into an INFO or FORMAT field.

    :returns: The value with its reserved characters escaped.
    """
    return _PERCENT_ENCODE_PATTERN.sub(lambda m: _PERCENT_ENCODE_MAP[m.group(0)], value)
