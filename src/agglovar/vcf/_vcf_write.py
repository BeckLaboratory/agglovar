"""VCF file writing routines."""

from __future__ import annotations

__all__ = [
    'write_vcf',
]

from pathlib import Path
from typing import Literal

import polars as pl

from ._vcf_const import VCF_SOURCE


def write_vcf(
    df: pl.DataFrame | pl.LazyFrame,
    path: str | Path,
    *,
    ref_fasta: str | Path | None = None,
    alt_format: Literal['seq', 'symbolic'] = 'seq',
    sample_name: str | None = None,
    source: str = VCF_SOURCE,
) -> None:
    """
    Write a per-allele Polars variant table to a VCF 4.2 file.

    Rows sharing the same ``vcf_rec`` are reassembled into a single multi-allelic VCF record
    with ALT alleles emitted in ``vcf_allele`` order. ``vcf_info_*`` columns are written back
    to INFO fields; the VCF header is rebuilt from column dtypes via
    :func:`agglovar.vcf.polars_to_vcf_type`.

    :param df: Variant table conforming to :data:`agglovar.vcf.VCF_BASE_FIXED_SCHEMA` (plus
        any ``vcf_info_*`` columns). :class:`polars.LazyFrame` inputs are collected internally.
    :param path: Destination path.
    :param ref_fasta: Path to an indexed reference FASTA. Required when ``alt_format`` is
        ``'symbolic'`` and REF context must be recovered from the reference.
    :param alt_format: How to emit ALT alleles:

        * ``'seq'`` (default) — write the literal sequence from ``vcf_alt``.
        * ``'symbolic'`` — write symbolic alleles (``<INS>``, ``<DEL>``, ``<INV>``,
          ``<DUP>``) with inserted/deleted sequence in ``INFO/SEQ``.

    :param sample_name: Sample column name in the output FORMAT block.
    :param source: Value for the ``##source=`` header line.

    :raises ValueError: If required columns are missing or ``alt_format`` constraints are
        violated.
    :raises FileNotFoundError: If ``ref_fasta`` is given but does not exist.
    """
    raise NotImplementedError('write_vcf: implementation pending')
