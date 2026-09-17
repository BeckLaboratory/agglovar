"""Tests for VCF header parsing."""

from __future__ import annotations

from pathlib import Path

import pysam
import pytest

from agglovar.vcf import (
    VcfHeader,
    read_vcf_header,
)


_HEADER = """##fileformat=VCFv4.2
##fileDate=20240101
##source=testsrc
##reference=hg38
##contig=<ID=chr1,length=1000000>
##ALT=<ID=SV,Description="Simple SVs">
##ALT=<ID=DUP:TANDEM,Description="Tandem Duplication">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FILTER=<ID=LowQual,Description="Low quality">
##PEDIGREE=<ID=kid,Father=dad,Mother=mom>
##SAMPLE=<ID=S1,Description="Sample one">
##customkey=customvalue
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t1000\tsnv\tA\tG\t50\tPASS\t.\tGT\t0/1
"""


@pytest.fixture
def header(tmp_path: Path) -> VcfHeader:
    """Read a header covering every record type pysam reports as ``STRUCTURED``."""
    path = tmp_path / 'header.vcf'
    path.write_text(_HEADER)

    with pysam.VariantFile(str(path)) as vcf_file:
        return read_vcf_header(vcf_file)


def test_alt_records_read(header: VcfHeader) -> None:
    """``##ALT`` lines land in the ALT field, not in metadata."""
    assert [alt.id for alt in header.alts] == ['SV', 'DUP:TANDEM']
    assert [alt.description for alt in header.alts] == ['Simple SVs', 'Tandem Duplication']


def test_structured_meta_preserved(header: VcfHeader) -> None:
    """Structured lines with no dedicated field are kept intact as metadata."""
    meta = dict(header.meta)

    assert meta['PEDIGREE'] == '<ID=kid,Father=dad,Mother=mom>'
    assert meta['SAMPLE'] == '<ID=S1,Description="Sample one">'


def test_structured_not_duplicated_in_meta(header: VcfHeader) -> None:
    """Structured lines routed to a field are not also stored as metadata."""
    assert 'ALT' not in dict(header.meta)


def test_generic_meta_read(header: VcfHeader) -> None:
    """Unstructured ``##key=value`` lines still reach metadata."""
    assert dict(header.meta)['customkey'] == 'customvalue'


@pytest.mark.parametrize('line', [
    '##ALT=<ID=SV,Description="Simple SVs">',
    '##PEDIGREE=<ID=kid,Father=dad,Mother=mom>',
    '##SAMPLE=<ID=S1,Description="Sample one">',
    '##customkey=customvalue',
])
def test_header_round_trip(header: VcfHeader, line: str) -> None:
    """Every structured and generic line is written back out."""
    assert line in str(header).splitlines()
