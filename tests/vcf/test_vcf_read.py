"""Tests for VCF reading, variant classification, and per-type table construction."""

from __future__ import annotations

from pathlib import Path
import warnings

import pytest

import agglovar
from agglovar.expr.variant import id_expr_for_table


_HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##ALT=<ID=DEL,Description="Deletion">
##ALT=<ID=DUP:TANDEM,Description="Tandem Duplication">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Difference in length between REF and ALT">
##INFO=<ID=SEQ,Number=1,Type=String,Description="Variant sequence">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FILTER=<ID=LowQual,Description="Low quality">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
"""

# One record per variant type, all internally consistent so that reading emits no warnings
_CLEAN_RECORDS = [
    'chr1\t1000\tsnv\tA\tG\t50\tPASS\t.\tGT\t0/1',
    'chr1\t1100\tdel_seq\tAT\tA\t50\tPASS\t.\tGT\t1/1',
    'chr1\t1200\tins_seq\tA\tATTTG\t50\tLowQual\t.\tGT\t0|1',
    'chr1\t1300\tsub\tACGT\tTGCA\t50\tPASS\t.\tGT\t0/1',
    'chr1\t1400\tinv\tN\t<INV>\t50\tPASS\tSVTYPE=INV;SVLEN=500;END=1900\tGT\t0/1',
    'chr1\t2000\tdup_tandem\tN\t<DUP:TANDEM>\t50\tPASS\tSVTYPE=DUP:TANDEM;END=2195;SVLEN=196\tGT\t0/1',
    'chr1\t3000\tins_sym\tN\t<INS>\t50\tPASS\tSVTYPE=INS;SVLEN=60\tGT\t0/1',
    'chr1\t4000\tcpx\tN\t<CPX>\t50\tPASS\tSVTYPE=CPX;SVLEN=100;END=4100\tGT\t0/1',
    'chr1\t5000\tbnd\tN\tN[chr2:100[\t50\tPASS\tSVTYPE=BND\tGT\t0/1',
]


def _write_vcf(path: Path, records: list[str]) -> Path:
    """Write a VCF with the shared test header and *records*, returning *path*."""
    path.write_text(_HEADER + ''.join(f'{rec}\n' for rec in records))
    return path


@pytest.fixture
def clean_vcf(tmp_path: Path) -> Path:
    """Build a VCF covering every variant type with no conflicting or ambiguous fields."""
    return _write_vcf(tmp_path / 'clean.vcf', _CLEAN_RECORDS)


@pytest.fixture
def clean_batch(clean_vcf: Path) -> agglovar.vcf.VcfBatch:
    """Read :func:`clean_vcf` and return its single batch."""
    return next(agglovar.vcf.iter_vcf(clean_vcf))


def _by_id(batch: agglovar.vcf.VcfBatch, table: str) -> dict[str, dict]:
    """Collect *table* from *batch* as a ``{vcf_id: row}`` mapping."""
    return {row['vcf_id']: row for row in getattr(batch, table).collect().to_dicts()}


# ---------------------------------------------------------------------------
# Per-type table projection

@pytest.mark.parametrize('table', list(agglovar.schema.STANDARD_FIELDS))
def test_type_table_collects(clean_batch: agglovar.vcf.VcfBatch, table: str) -> None:
    """Every per-type table collects; none references a column outside its own schema."""
    df = getattr(clean_batch, table).collect()

    assert df.height > 0, f'no rows routed to {table!r}'
    assert df['id'].null_count() == 0


@pytest.mark.parametrize('table', list(agglovar.schema.STANDARD_FIELDS))
def test_type_table_columns(clean_batch: agglovar.vcf.VcfBatch, table: str) -> None:
    """Per-type tables lead with the standard fields for their type, in order."""
    fields = agglovar.schema.STANDARD_FIELDS[table]
    columns = getattr(clean_batch, table).collect_schema().names()

    assert columns[:len(fields)] == list(fields)


def test_id_expr_for_table_matches_columns() -> None:
    """The per-table ID expression only names columns that table actually carries."""
    for table, fields in agglovar.schema.STANDARD_FIELDS.items():
        used = set(id_expr_for_table(table).meta.root_names())

        assert used <= set(fields), f'{table!r} ID expression reaches outside its schema'


def test_id_expr_for_table_rejects_unknown() -> None:
    """An unknown table name is rejected rather than silently defaulting."""
    with pytest.raises(ValueError, match='Unknown variant-type table'):
        id_expr_for_table('indel')


# ---------------------------------------------------------------------------
# Coordinates

def test_sequence_allele_coordinates(clean_batch: agglovar.vcf.VcfBatch) -> None:
    """Sequence-resolved alleles land on 0-based half-open coordinates."""
    snv = _by_id(clean_batch, 'snv')['snv']
    insdel = _by_id(clean_batch, 'insdel')
    sub = _by_id(clean_batch, 'sub')['sub']

    # POS=1000 A>G: the changed base is 0-based 999
    assert (snv['pos'], snv['end'], snv['id']) == (999, 1000, 'chr1-1000-SNV-AG')

    # POS=1100 AT>A: 1100 is the padding base, the deleted T is 0-based 1100
    deletion = insdel['del_seq']
    assert (deletion['pos'], deletion['end'], deletion['varlen']) == (1100, 1101, 1)
    assert deletion['seq'] == 'T'

    # POS=1200 A>ATTTG: inserted after the padding base at 0-based 1199
    insertion = insdel['ins_seq']
    assert (insertion['pos'], insertion['end'], insertion['varlen']) == (1200, 1201, 4)
    assert insertion['seq'] == 'TTTG'

    # POS=1300 ACGT>TGCA: varlen is len(ref) + len(alt)
    assert (sub['pos'], sub['end'], sub['varlen']) == (1299, 1303, 8)


def test_symbolic_allele_coordinates(clean_batch: agglovar.vcf.VcfBatch) -> None:
    """Symbolic alleles start after the REF padding base."""
    inv = _by_id(clean_batch, 'inv')['inv']
    ins = _by_id(clean_batch, 'insdel')['ins_sym']

    # POS=1400 <INV> SVLEN=500: the inverted span is 0-based [1400, 1900)
    assert (inv['pos'], inv['end'], inv['varlen']) == (1400, 1900, 500)

    # A symbolic INS is a point event carrying its length in varlen
    assert (ins['pos'], ins['end'], ins['varlen']) == (3000, 3001, 60)


def test_vcf_pos_is_one_based(clean_batch: agglovar.vcf.VcfBatch) -> None:
    """``vcf_pos`` reproduces the POS column verbatim."""
    snv = _by_id(clean_batch, 'snv')['snv']
    inv = _by_id(clean_batch, 'inv')['inv']

    assert snv['vcf_pos'] == 1000
    assert inv['vcf_pos'] == 1400


@pytest.mark.parametrize('table', list(agglovar.schema.STANDARD_FIELDS))
def test_end_follows_pos(clean_batch: agglovar.vcf.VcfBatch, table: str) -> None:
    """Every emitted interval is non-empty and correctly ordered."""
    df = getattr(clean_batch, table).collect()

    assert (df['end'] > df['pos']).all()


# ---------------------------------------------------------------------------
# INFO/END recovery

def test_info_end_column_populated(clean_batch: agglovar.vcf.VcfBatch) -> None:
    """``vcf_info_END`` carries INFO/END even though pysam hides it from ``record.info``."""
    assert _by_id(clean_batch, 'inv')['inv']['vcf_info_END'] == 1900
    assert _by_id(clean_batch, 'dup')['dup_tandem']['vcf_info_END'] == 2195
    assert _by_id(clean_batch, 'cpx')['cpx']['vcf_info_END'] == 4100


def test_info_end_null_without_end(clean_batch: agglovar.vcf.VcfBatch) -> None:
    """Records declaring no END report a null rather than a fabricated span."""
    assert _by_id(clean_batch, 'snv')['snv']['vcf_info_END'] is None
    assert _by_id(clean_batch, 'insdel')['del_seq']['vcf_info_END'] is None
    assert _by_id(clean_batch, 'insdel')['ins_sym']['vcf_info_END'] is None


def test_info_end_drives_varlen_when_alone(tmp_path: Path) -> None:
    """END alone determines the length when no SVLEN or SEQ is present."""
    path = _write_vcf(
        tmp_path / 'end_only.vcf',
        ['chr1\t7000\tdel_end\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;END=7300\tGT\t0/1'],
    )
    row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'insdel')['del_end']

    assert (row['pos'], row['end'], row['varlen']) == (7000, 7300, 300)
    assert row['vcf_info_END'] == 7300


# ---------------------------------------------------------------------------
# Warnings

def test_clean_vcf_is_warning_free(clean_vcf: Path) -> None:
    """Well-formed records, including subtyped symbolic ALTs, read without warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        batch = next(agglovar.vcf.iter_vcf(clean_vcf))
        for table in agglovar.schema.STANDARD_FIELDS:
            getattr(batch, table).collect()

    assert [str(w.message) for w in caught] == []


def test_svtype_subtype_does_not_warn(tmp_path: Path) -> None:
    """``<DUP:TANDEM>`` against ``SVTYPE=DUP:TANDEM`` agrees; only base types are compared."""
    path = _write_vcf(
        tmp_path / 'subtype.vcf',
        ['chr1\t2000\tdup\tN\t<DUP:TANDEM>\t50\tPASS\tSVTYPE=DUP:TANDEM;SVLEN=196\tGT\t0/1'],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        batch = next(agglovar.vcf.iter_vcf(path))

    assert [str(w.message) for w in caught] == []
    assert _by_id(batch, 'dup')['dup']['vartype'] == 'DUP'


def test_svtype_real_mismatch_warns(tmp_path: Path) -> None:
    """A genuine ALT/SVTYPE disagreement still warns and keeps the ALT type."""
    path = _write_vcf(
        tmp_path / 'mismatch.vcf',
        ['chr1\t2000\tbad\tN\t<DEL>\t50\tPASS\tSVTYPE=INV;SVLEN=50\tGT\t0/1'],
    )
    with pytest.warns(UserWarning, match='disagrees with INFO/SVTYPE'):
        batch = next(agglovar.vcf.iter_vcf(path))

    assert _by_id(batch, 'insdel')['bad']['vartype'] == 'DEL'


def test_varlen_conflict_warns_beyond_one_base(tmp_path: Path) -> None:
    """A varlen disagreement larger than the 1 bp padding convention still warns."""
    path = _write_vcf(
        tmp_path / 'conflict.vcf',
        ['chr1\t6000\tbad\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;SVLEN=50;END=6100\tGT\t0/1'],
    )
    with pytest.warns(UserWarning, match='conflicting varlen sources'):
        batch = next(agglovar.vcf.iter_vcf(path))

    # SVLEN keeps priority over the END-derived length
    assert _by_id(batch, 'insdel')['bad']['varlen'] == 50


# ---------------------------------------------------------------------------
# Routing

def test_bnd_routed_to_ignored(clean_batch: agglovar.vcf.VcfBatch) -> None:
    """Breakend alleles are held out of the type tables with their reason recorded."""
    ignored = clean_batch.ignored.collect()

    assert ignored.height == 1
    assert ignored['vartype'].to_list() == ['BND']
    assert ignored['vcf_ignored'].to_list() == ['BND breakend allele']


def test_sample_table_covers_emitted_records(clean_batch: agglovar.vcf.VcfBatch) -> None:
    """Every record contributing a base-table row contributes one sample row per sample."""
    emitted = {
        rec
        for table in agglovar.schema.STANDARD_FIELDS
        for rec in getattr(clean_batch, table).collect()['vcf_rec'].to_list()
    }
    samples = clean_batch.sample_table.collect()

    assert samples['vcf_rec'].to_list() == sorted(emitted)
    assert samples['GT'].to_list() == ['0/1', '1/1', '0|1', '0/1', '0/1', '0/1', '0/1', '0/1']
