"""Tests for VCF reading, variant classification, and per-type table construction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

import polars as pl
import pysam
import pytest

import agglovar
from agglovar.expr.variant import id_expr_for_table
from agglovar.vcf._vcf_read import (
    _classify_symbolic_allele,
    _frame_from_cols,
)


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
# Placeholder-anchor alleles (REF or ALT written as "N" instead of the anchor base)

def test_placeholder_ref_ins_matches_conforming_record(tmp_path: Path) -> None:
    """``N`` REF against a sequence ALT lands where the same insertion written to spec does."""
    path = _write_vcf(
        tmp_path / 'placeholder_ins.vcf',
        [
            'chr1\t1200\tspec\tA\tATTTG\t50\tPASS\tSVTYPE=INS;SVLEN=4\tGT\t0/1',
            'chr1\t1200\tplaceholder\tN\tTTTG\t50\tPASS\tSVTYPE=INS;SVLEN=4\tGT\t0/1',
        ],
    )
    insdel = _by_id(next(agglovar.vcf.iter_vcf(path)), 'insdel')
    spec, placeholder = insdel['spec'], insdel['placeholder']

    assert (spec['pos'], spec['end'], spec['varlen'], spec['seq']) == (1200, 1201, 4, 'TTTG')
    assert (placeholder['pos'], placeholder['end']) == (spec['pos'], spec['end'])
    assert (placeholder['varlen'], placeholder['seq']) == (spec['varlen'], spec['seq'])


def test_placeholder_alt_del_is_not_recognised(tmp_path: Path) -> None:
    """The mirrored deletion shape stays SUB; no known caller writes it, so it is left alone."""
    path = _write_vcf(
        tmp_path / 'placeholder_del.vcf',
        ['chr1\t1100\tmirrored\tTTTG\tN\t50\tPASS\tSVTYPE=DEL;SVLEN=-4\tGT\t0/1'],
    )
    batch = next(agglovar.vcf.iter_vcf(path))

    assert _by_id(batch, 'insdel') == {}
    assert _by_id(batch, 'sub')['mirrored']['vartype'] == 'SUB'


def test_placeholder_sequence_overrides_svlen(tmp_path: Path) -> None:
    """The allele string measures the event; a disagreeing SVLEN is ignored, not averaged in.

    Severus reports SVLEN as a breakpoint estimate that differs from the assembled ALT in
    roughly a third of records, so the sequence has to win for merges to match on sequence.
    """
    path = _write_vcf(
        tmp_path / 'svlen_disagrees.vcf',
        ['chr1\t1200\tins\tN\tTTTGCA\t50\tPASS\tSVTYPE=INS;SVLEN=5\tGT\t0/1'],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'insdel')['ins']

    assert (row['varlen'], row['seq']) == (6, 'TTTGCA')
    assert [str(w.message) for w in caught] == []


def test_placeholder_needs_svtype(tmp_path: Path) -> None:
    """Without an SVTYPE the allele stays SUB, protecting a complex substitution at a real N."""
    path = _write_vcf(
        tmp_path / 'no_svtype.vcf',
        ['chr1\t1200\tcomplex\tN\tACGT\t50\tPASS\t.\tGT\t0/1'],
    )
    batch = next(agglovar.vcf.iter_vcf(path))

    assert _by_id(batch, 'insdel') == {}
    assert _by_id(batch, 'sub')['complex']['vartype'] == 'SUB'


def test_placeholder_discarding_sequence_warns(tmp_path: Path) -> None:
    """Forcing the declared type over a REF that carries real sequence is reported."""
    path = _write_vcf(
        tmp_path / 'discard.vcf',
        ['chr1\t1200\tforced\tACGT\tTTTGCA\t50\tPASS\tSVTYPE=INS;SVLEN=6\tGT\t0/1'],
    )
    with pytest.warns(UserWarning, match='discards a 4 bp REF allele'):
        row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'insdel')['forced']

    assert (row['vartype'], row['varlen'], row['seq']) == ('INS', 6, 'TTTGCA')


def test_placeholder_svtype_indexed_per_allele(tmp_path: Path) -> None:
    """A ``Number=A`` SVTYPE is read per allele rather than allele 1's value being reused."""
    header = _HEADER.replace(
        '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">',
        '##INFO=<ID=SVTYPE,Number=A,Type=String,Description="Type of structural variant">',
    )
    path = tmp_path / 'multiallelic.vcf'
    path.write_text(header + 'chr1\t1200\tma\tN\tTTTG,ACG\t50\tPASS\tSVTYPE=INS,INS\tGT\t1/2\n')
    rows = sorted(
        next(agglovar.vcf.iter_vcf(path)).insdel.collect().to_dicts(),
        key=lambda r: r['vcf_allele'],
    )

    assert [(r['vartype'], r['varlen'], r['seq']) for r in rows] == [
        ('INS', 4, 'TTTG'),
        ('INS', 3, 'ACG'),
    ]


# ---------------------------------------------------------------------------
# Mistyped anchor base on an insertion (VcfOpt.INS_ALT_ANCHORED)

# A spec-conforming insertion, the same insertion with an anchor base REF disagrees with, and
# the same allele read as a whole inserted sequence (what Severus means by the shape)
_INS_SPEC = 'chr1\t1200\tspec\tA\tATTTG\t50\tPASS\tSVTYPE=INS;SVLEN=4\tGT\t0/1'
_INS_BAD_ANCHOR = 'chr1\t1200\tbad_anchor\tA\tGTTTG\t50\tPASS\tSVTYPE=INS;SVLEN=4\tGT\t0/1'


def test_ins_alt_anchored_matches_the_conforming_record(tmp_path: Path) -> None:
    """Dropping the mistyped anchor base lands the insertion exactly where a good one lands."""
    path = _write_vcf(tmp_path / 'bad_anchor.vcf', [_INS_SPEC, _INS_BAD_ANCHOR])
    insdel = _by_id(
        next(agglovar.vcf.iter_vcf(path, vcf_opt='INS_ALT_ANCHORED')), 'insdel',
    )
    spec, fixed = insdel['spec'], insdel['bad_anchor']

    assert (fixed['pos'], fixed['end'], fixed['varlen']) == (spec['pos'], spec['end'], 4)
    assert fixed['seq'] == 'TTTG'


def test_ins_alt_anchored_off_reads_the_whole_alt(tmp_path: Path) -> None:
    """By default the allele is the inserted sequence entire, which is what Severus writes."""
    path = _write_vcf(tmp_path / 'whole_alt.vcf', [_INS_BAD_ANCHOR])
    row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'insdel')['bad_anchor']

    assert (row['varlen'], row['seq']) == (5, 'GTTTG')


def test_ins_alt_anchored_leaves_a_matching_anchor_alone(tmp_path: Path) -> None:
    """A conforming insertion is stripped by the prefix rule and never reaches the branch."""
    path = _write_vcf(tmp_path / 'anchored_spec.vcf', [_INS_SPEC])
    rows = {
        opt: _by_id(next(agglovar.vcf.iter_vcf(path, vcf_opt=opt)), 'insdel')['spec']
        for opt in ('', 'INS_ALT_ANCHORED')
    }

    assert rows[''] == rows['INS_ALT_ANCHORED']


def test_ins_alt_anchored_ignores_a_multi_base_ref(tmp_path: Path) -> None:
    """A REF longer than one base is not an anchor, so the whole ALT stays the insertion."""
    path = _write_vcf(
        tmp_path / 'long_ref.vcf',
        ['chr1\t1200\tlong_ref\tACGT\tTTTG\t50\tPASS\tSVTYPE=INS;SVLEN=4\tGT\t0/1'],
    )
    with pytest.warns(UserWarning, match='discards a 4 bp REF allele'):
        row = _by_id(
            next(agglovar.vcf.iter_vcf(path, vcf_opt='INS_ALT_ANCHORED')), 'insdel',
        )['long_ref']

    assert (row['varlen'], row['seq']) == (4, 'TTTG')


# ---------------------------------------------------------------------------
# INFO/SVTYPE against the type REF and ALT describe

# (svtype, ref, alt, expected vartype, or None where the pair contradicts the declared type)
_SVTYPE_CASES = [
    ('DEL', 'ATTTG', 'A',     'DEL'),   # exact
    ('DEL', 'TTTG',  'N',     'SUB'),   # deletion against a placeholder ALT
    ('DEL', 'AAC',   'AAG',   'SNV'),   # sawfish: SVLEN=2 INSLEN=2, minimised to one base
    ('DEL', 'A',     'ATTTG', None),    # declares a deletion, spells an insertion
    ('INS', 'A',     'ATTTG', 'INS'),   # exact
    ('INS', 'ATTTG', 'A',     None),    # declares an insertion, spells a deletion
    ('DUP', 'A',     'ATTTG', 'INS'),   # a tandem duplication resolved as inserted sequence
    ('INV', 'A',     'G',     None),    # nothing about one base is an inversion
    ('CPX', 'A',     'G',     None),
    ('BND', 'A',     'G',     'SNV'),   # not a type this reader models, so nothing to contradict
    ('DEL+INS', 'A', 'G',     'SNV'),   # compound SVision type, likewise unmodelled
]


@pytest.mark.parametrize(('svtype', 'ref', 'alt', 'expected'), _SVTYPE_CASES)
def test_svtype_against_sequence(
    tmp_path: Path, svtype: str, ref: str, alt: str, expected: Optional[str],
) -> None:
    """A coarser SVTYPE is accepted; one naming a different event stops the read."""
    path = _write_vcf(
        tmp_path / 'svtype_seq.vcf',
        [f'chr1\t1200\tv\t{ref}\t{alt}\t50\tPASS\tSVTYPE={svtype}\tGT\t0/1'],
    )
    if expected is None:
        with pytest.raises(agglovar.vcf.VcfSvtypeMismatchError, match='conflicts with'):
            next(agglovar.vcf.iter_vcf(path))
        return

    batch = next(agglovar.vcf.iter_vcf(path))
    table = {'INS': 'insdel', 'DEL': 'insdel', 'SNV': 'snv', 'SUB': 'sub'}[expected]

    assert _by_id(batch, table)['v']['vartype'] == expected


def test_svtype_mismatch_is_a_value_error() -> None:
    """The error subclasses ValueError so existing handlers still catch it."""
    assert issubclass(agglovar.vcf.VcfSvtypeMismatchError, ValueError)


def test_svtype_mismatch_ignore_keeps_reading(tmp_path: Path) -> None:
    """The opt-in downgrades the conflict to a warning and an ignored row."""
    path = _write_vcf(
        tmp_path / 'svtype_ignore.vcf',
        [
            'chr1\t1200\tconflict\tA\tATTTG\t50\tPASS\tSVTYPE=DEL\tGT\t0/1',
            'chr1\t1300\tfine\tA\tATTTG\t50\tPASS\tSVTYPE=INS\tGT\t0/1',
        ],
    )
    with pytest.warns(UserWarning, match='conflicts with'):
        batch = next(agglovar.vcf.iter_vcf(path, vcf_opt='SVTYPE_MISMATCH_IGNORE'))

    assert _by_id(batch, 'insdel').keys() == {'fine'}

    ignored = _by_id(batch, 'ignored')['conflict']
    assert ignored['vartype'] == 'INS'
    assert 'conflicts with' in ignored['vcf_ignored']


def test_svtype_mismatch_does_not_apply_to_symbolic_alts(tmp_path: Path) -> None:
    """A symbolic ALT states its own type and keeps its existing warn-and-prefer policy."""
    path = _write_vcf(
        tmp_path / 'svtype_symbolic.vcf',
        ['chr1\t1200\tsym\tN\t<DUP>\t50\tPASS\tSVTYPE=INV;END=1300\tGT\t0/1'],
    )
    with pytest.warns(UserWarning, match='disagrees with'):
        batch = next(agglovar.vcf.iter_vcf(path))

    assert _by_id(batch, 'dup')['sym']['vartype'] == 'DUP'


# ---------------------------------------------------------------------------
# Sequence-resolved inversions (svim, svim-asm: REF is the inverted span, no anchor base)

# 12 bp spans; the ALT below each REF is its exact reverse complement
_INV_REF = 'ACGTTGCAAGCC'
_INV_ALT = 'GGCTTGCAACGT'

# Ends in its own complement, so the reverse complement matches REF at both edges and the
# common-prefix and common-suffix strips eat two real bases from each end
_INV_REF_PALINDROMIC = 'ATTGCAATCGAT'
_INV_ALT_PALINDROMIC = 'ATCGATTGCAAT'


def _inv_batch(tmp_path: Path, records: list[str], name: str) -> agglovar.vcf.VcfBatch:
    """Read a VCF of *records* and return its single batch."""
    return next(agglovar.vcf.iter_vcf(_write_vcf(tmp_path / name, records)))


def test_sequence_inv_spans_the_ref_allele(tmp_path: Path) -> None:
    """A sequence-resolved inversion covers the whole REF allele, with no anchor base."""
    batch = _inv_batch(
        tmp_path,
        [
            f'chr1\t1000\tinv_seq\t{_INV_REF}\t{_INV_ALT}'
            f'\t50\tPASS\tSVTYPE=INV;END=1011\tGT\t0/1'
        ],
        'inv_seq.vcf',
    )
    row = _by_id(batch, 'inv')['inv_seq']

    assert row['vartype'] == 'INV'
    assert (row['pos'], row['end']) == (999, 1011)   # 0-based half-open, POS through INFO/END
    assert row['varlen'] == len(_INV_REF)
    assert row['seq'] == _INV_REF                    # REF holds the inverted bases


def test_sequence_inv_ignores_a_coincidental_strip(tmp_path: Path) -> None:
    """Bases the strips remove by chance do not move the inversion or shorten it.

    A reverse-complemented ALT matches REF at both edges whenever the span ends in its own
    complement, which is unrelated to the anchor base the strips exist to remove.
    """
    batch = _inv_batch(
        tmp_path,
        [
            f'chr1\t1000\tpalindromic\t{_INV_REF_PALINDROMIC}\t{_INV_ALT_PALINDROMIC}'
            f'\t50\tPASS\tSVTYPE=INV;END=1011\tGT\t0/1'
        ],
        'inv_palindromic.vcf',
    )
    row = _by_id(batch, 'inv')['palindromic']

    assert (row['pos'], row['end']) == (999, 1011)
    assert row['varlen'] == len(_INV_REF_PALINDROMIC)


def test_sequence_inv_needs_svtype(tmp_path: Path) -> None:
    """Without INFO/SVTYPE the same alleles stay a substitution rather than an inversion."""
    batch = _inv_batch(
        tmp_path,
        [
            f'chr1\t1000\tno_svtype\t{_INV_REF}\t{_INV_ALT}'
            f'\t50\tPASS\tEND=1011\tGT\t0/1'
        ],
        'inv_no_svtype.vcf',
    )

    assert _by_id(batch, 'inv') == {}
    assert _by_id(batch, 'sub')['no_svtype']['vartype'] == 'SUB'


def test_sequence_inv_accepts_an_svtype_subtype(tmp_path: Path) -> None:
    """A subtype on INFO/SVTYPE ("INV:PERFECT") still reads as an inversion."""
    batch = _inv_batch(
        tmp_path,
        [
            f'chr1\t1000\tsubtype\t{_INV_REF}\t{_INV_ALT}'
            f'\t50\tPASS\tSVTYPE=INV:PERFECT;END=1011\tGT\t0/1'
        ],
        'inv_subtype.vcf',
    )

    assert _by_id(batch, 'inv')['subtype']['varlen'] == len(_INV_REF)


def test_sequence_inv_prefers_info_end_and_warns(tmp_path: Path) -> None:
    """An INFO/END past the REF span wins and reports the disagreement."""
    path = _write_vcf(
        tmp_path / 'inv_end_conflict.vcf',
        [
            f'chr1\t1000\tlong_end\t{_INV_REF}\t{_INV_ALT}'
            f'\t50\tPASS\tSVTYPE=INV;END=1020\tGT\t0/1'
        ],
    )
    with pytest.warns(UserWarning, match='INV INFO/END'):
        row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'inv')['long_end']

    assert (row['pos'], row['end']) == (999, 1020)
    assert row['varlen'] == 21
    assert row['seq'] is None    # REF no longer spells the span


@pytest.mark.parametrize('alt', ['<INV>', '<INV:PERFECT>', '<INV:>'])
def test_symbolic_inv_routes_to_the_inv_table(tmp_path: Path, alt: str) -> None:
    """A symbolic inversion reads as an inversion whatever subtype follows the type."""
    batch = _inv_batch(
        tmp_path,
        [f'chr1\t1000\tsym\tA\t{alt}\t50\tPASS\tSVTYPE=INV;END=1100\tGT\t0/1'],
        'inv_symbolic.vcf',
    )
    row = _by_id(batch, 'inv')['sym']

    assert row['vartype'] == 'INV'
    assert (row['pos'], row['end'], row['varlen']) == (1000, 1100, 100)


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
# Varlen sources for a symbolic ALT (SVLEN and END measure the reference, SEQ the query)

def _warnings_from(path: Path) -> list[str]:
    """Read *path* to exhaustion and return every warning message it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in agglovar.vcf.iter_vcf(path):
            pass

    return [str(w.message) for w in caught]


def _read_with_warnings(path: Path) -> tuple[agglovar.vcf.VcfBatch, list[str]]:
    """Read the first batch of *path*, returning it alongside the warnings the read raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        batch = next(agglovar.vcf.iter_vcf(path))

    return batch, [str(w.message) for w in caught]


def test_symbolic_seq_is_not_weighed_against_the_reference_span(tmp_path: Path) -> None:
    """A SEQ that disagrees with SVLEN and END is not a conflict.

    SEQ is the query-side allele, so an inversion carrying internal indels legitimately differs
    in length from the reference span it replaces without either field being wrong.
    """
    path = _write_vcf(
        tmp_path / 'inv_seq.vcf',
        ['chr1\t1400\tinv\tN\t<INV>\t50\tPASS\tSVTYPE=INV;SVLEN=500;END=1900;SEQ=ACGTACGTAC'
         '\tGT\t0/1'],
    )

    assert _warnings_from(path) == []

    row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'inv')['inv']
    assert (row['pos'], row['end'], row['varlen']) == (1400, 1900, 500)


@pytest.mark.parametrize('vartype', ['DEL', 'DUP', 'CNV', 'INV'])
def test_svlen_end_conflict_warns_for_a_reserved_type(tmp_path: Path, vartype: str) -> None:
    """The specification defines SVLEN as a reference count for these types, so they must agree."""
    path = _write_vcf(
        tmp_path / f'{vartype.lower()}_conflict.vcf',
        [f'chr1\t6000\tbad\tN\t<{vartype}>\t50\tPASS\tSVTYPE={vartype};SVLEN=50;END=6100'
         f'\tGT\t0/1'],
    )

    assert any('conflicting varlen sources' in m for m in _warnings_from(path))


def test_svlen_end_conflict_is_silent_for_a_non_reserved_type(tmp_path: Path) -> None:
    """``<CPX>`` is not a reserved ALT, so SVLEN is undefined for it and cannot conflict.

    A caller is free to put a query-side length in SVLEN for a type the specification does not
    describe, which makes it orthogonal to the reference breakpoint span END measures.
    """
    path = _write_vcf(
        tmp_path / 'cpx_conflict.vcf',
        ['chr1\t4000\tcpx\tN\t<CPX>\t50\tPASS\tSVTYPE=CPX;SVLEN=100;END=4500\tGT\t0/1'],
    )

    assert _warnings_from(path) == []


def test_non_reserved_end_comes_from_info_end(tmp_path: Path) -> None:
    """END sets the reference end for a non-reserved type while SVLEN keeps varlen."""
    path = _write_vcf(
        tmp_path / 'cpx_end.vcf',
        ['chr1\t4000\tcpx\tN\t<CPX>\t50\tPASS\tSVTYPE=CPX;SVLEN=100;END=4500\tGT\t0/1'],
    )
    row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'cpx')['cpx']

    assert (row['pos'], row['end'], row['varlen']) == (4000, 4500, 100)


def test_non_reserved_without_end_claims_one_base(tmp_path: Path) -> None:
    """Without END nothing states where a non-reserved event ends, so it claims only its anchor.

    Stretching the span by SVLEN would write a query-side length into a reference coordinate,
    which reads as a confident interval that no field in the record supports.
    """
    path = _write_vcf(
        tmp_path / 'cpx_no_end.vcf',
        ['chr1\t4000\tcpx\tN\t<CPX>\t50\tPASS\tSVTYPE=CPX;SVLEN=100\tGT\t0/1'],
    )
    row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'cpx')['cpx']

    assert (row['pos'], row['end'], row['varlen']) == (4000, 4001, 100)


def test_seq_still_supplies_varlen_as_a_last_resort(tmp_path: Path) -> None:
    """SEQ keeps its fallback role for a record declaring neither SVLEN nor END."""
    path = _write_vcf(
        tmp_path / 'del_seq_only.vcf',
        ['chr1\t6000\tdel_seq\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;SEQ=ACGTACGTAC\tGT\t0/1'],
    )
    row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'insdel')['del_seq']

    assert (row['pos'], row['end'], row['varlen']) == (6000, 6010, 10)


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


def test_svtype_mismatch_prefer_svtype(tmp_path: Path) -> None:
    """``SYM_ALT_MISMATCH_PREFER_SVTYPE`` takes the SVTYPE type without warning."""
    path = _write_vcf(
        tmp_path / 'prefer_svtype.vcf',
        ['chr1\t2000\tbad\tN\t<DEL>\t50\tPASS\tSVTYPE=INV;SVLEN=50\tGT\t0/1'],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        batch = next(agglovar.vcf.iter_vcf(path, vcf_opt='SYM_ALT_MISMATCH_PREFER_SVTYPE'))
        vartype = _by_id(batch, 'inv')['bad']['vartype']

    assert [str(w.message) for w in caught] == []
    assert vartype == 'INV'


def test_svtype_mismatch_prefer_alt_is_silent(tmp_path: Path) -> None:
    """``alt_prefer_svtype=False`` keeps the ALT type and suppresses the mismatch warning."""
    path = _write_vcf(
        tmp_path / 'prefer_alt.vcf',
        ['chr1\t2000\tbad\tN\t<DEL>\t50\tPASS\tSVTYPE=INV;SVLEN=50\tGT\t0/1'],
    )
    with pysam.VariantFile(str(path)) as vcf_file:
        record = next(iter(vcf_file))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        vartype = _classify_symbolic_allele(
            record.start, '<DEL>', {'SVTYPE': 'INV', 'SVLEN': 50}, alt_prefer_svtype=False,
        )[0]

    assert [str(w.message) for w in caught] == []
    assert vartype == 'DEL'


def test_svtype_agreement_ignores_prefer_svtype(tmp_path: Path) -> None:
    """With no disagreement the flag changes nothing; subtypes still collapse to the base type."""
    path = _write_vcf(
        tmp_path / 'agree.vcf',
        ['chr1\t2000\tdup\tN\t<DUP:TANDEM>\t50\tPASS\tSVTYPE=DUP:TANDEM;SVLEN=196\tGT\t0/1'],
    )
    batch = next(agglovar.vcf.iter_vcf(path, vcf_opt='SYM_ALT_MISMATCH_PREFER_SVTYPE'))

    assert _by_id(batch, 'dup')['dup']['vartype'] == 'DUP'


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


# ---------------------------------------------------------------------------
# Per-ALT (Number=A) fields

_PER_ALT_HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=VAF,Number=A,Type=Float,Description="Variant allele frequency">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Read depth for each allele">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
"""


def _write_per_alt_vcf(path: Path, records: list[str]) -> Path:
    """Write a VCF declaring ``Number=A`` INFO and FORMAT fields, returning *path*."""
    path.write_text(_PER_ALT_HEADER + ''.join(f'{rec}\n' for rec in records))
    return path


def test_format_number_a_reads_as_list(tmp_path: Path) -> None:
    """A ``Number=A`` FORMAT field is a list column even on a single-ALT record.

    Sample rows carry no allele index, so a per-ALT FORMAT value cannot collapse to a scalar
    the way its INFO counterpart does in the per-allele base table.
    """
    path = _write_per_alt_vcf(
        tmp_path / 'one_alt.vcf',
        ['chr1\t1000\tsnv\tA\tG\t50\tPASS\tAF=0.4\tGT:VAF:AD\t0/1:0.4:5,5'],
    )
    samples = next(agglovar.vcf.iter_vcf(path)).sample_table.collect()

    assert samples.schema['VAF'] == pl.List(pl.Float64)
    assert samples['VAF'].to_list() == [pytest.approx([0.4])]


def test_format_number_a_holds_one_value_per_alt(tmp_path: Path) -> None:
    """A ``Number=A`` FORMAT list is ordered by ALT and indexes with ``vcf_allele``."""
    path = _write_per_alt_vcf(
        tmp_path / 'multi_alt.vcf',
        ['chr1\t1000\tsnv\tA\tG,T\t50\tPASS\tAF=0.3,0.7\tGT:VAF:AD\t1/2:0.3,0.7:2,4,6'],
    )
    batch = next(agglovar.vcf.iter_vcf(path))
    per_allele = (
        batch.snv.collect()
        .join(batch.sample_table.collect(), on='vcf_rec')
        .with_columns(vaf=pl.col('VAF').list.get(pl.col('vcf_allele') - 1))
        .sort('vcf_allele')
    )

    assert per_allele['vcf_alt'].to_list() == ['G', 'T']
    assert per_allele['vaf'].to_list() == pytest.approx([0.3, 0.7])


def test_info_number_a_splits_across_alleles(tmp_path: Path) -> None:
    """A ``Number=A`` INFO field stays scalar, one value per per-allele base-table row."""
    path = _write_per_alt_vcf(
        tmp_path / 'info_multi_alt.vcf',
        ['chr1\t1000\tsnv\tA\tG,T\t50\tPASS\tAF=0.3,0.7\tGT:VAF:AD\t1/2:0.3,0.7:2,4,6'],
    )
    snv = next(agglovar.vcf.iter_vcf(path)).snv.collect().sort('vcf_allele')

    assert snv.schema['vcf_info_AF'] == pl.Float64
    assert snv['vcf_info_AF'].to_list() == pytest.approx([0.3, 0.7])


# ---------------------------------------------------------------------------
# Reserved characters in INFO and FORMAT values

_ESCAPE_HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##INFO=<ID=S1,Number=1,Type=String,Description="Scalar string">
##INFO=<ID=SA,Number=A,Type=String,Description="Per-allele string">
##INFO=<ID=SD,Number=.,Type=String,Description="Unbounded string">
##INFO=<ID=I1,Number=1,Type=Integer,Description="Scalar integer">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=FS,Number=1,Type=String,Description="Scalar string">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
"""

# Every character §1.2 reserves, encoded, and the text it must decode back to
_ALL_ESCAPED = 'a%3Ab%3Bc%3Dd%25e%2Cf%0Dg%0Ah%09i'
_ALL_DECODED = 'a:b;c=d%e,f\rg\nh\ti'


def _write_escape_vcf(path: Path, records: list[str]) -> Path:
    """Write a VCF declaring string INFO and FORMAT fields of each ``Number``, returning *path*."""
    path.write_text(_ESCAPE_HEADER + ''.join(f'{rec}\n' for rec in records))
    return path


def test_percent_escapes_decode(tmp_path: Path) -> None:
    """Every escape the specification defines is decoded on read."""
    path = _write_escape_vcf(
        tmp_path / 'escaped.vcf',
        [f'chr1\t1000\tesc\tA\tG\t50\tPASS\tS1={_ALL_ESCAPED}\tGT\t0/1'],
    )
    row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'snv')['esc']

    assert row['vcf_info_S1'] == _ALL_DECODED


def test_percent_escape_is_decoded_once(tmp_path: Path) -> None:
    """An encoded escape survives the round trip: ``%253B`` is the text ``%3B``, not ``;``."""
    path = _write_escape_vcf(
        tmp_path / 'double.vcf',
        ['chr1\t1000\tesc\tA\tG\t50\tPASS\tS1=%253B\tGT\t0/1'],
    )
    row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'snv')['esc']

    assert row['vcf_info_S1'] == '%3B'


def test_unencoded_percent_is_left_alone(tmp_path: Path) -> None:
    """A literal ``%`` that starts no defined escape is not a general percent-encoding."""
    path = _write_escape_vcf(
        tmp_path / 'literal.vcf',
        ['chr1\t1000\tesc\tA\tG\t50\tPASS\tS1=5%AF%3A100%\tGT\t0/1'],
    )
    row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'snv')['esc']

    assert row['vcf_info_S1'] == '5%AF:100%'


def test_unescaped_comma_in_a_scalar_string_is_rejoined(tmp_path: Path) -> None:
    """Splitting happens before Number is consulted, so rejoining restores the field text."""
    path = _write_escape_vcf(
        tmp_path / 'split.vcf',
        ['chr1\t1000\tsplit\tA\tG\t50\tPASS\tS1=DEL[1,234]:INV[5,678]\tGT\t0/1'],
    )
    batch, messages = _read_with_warnings(path)
    row = _by_id(batch, 'snv')['split']

    assert row['vcf_info_S1'] == 'DEL[1,234]:INV[5,678]'
    assert any('INFO/S1 declares Number=1' in m and 'rejoining them' in m for m in messages)


def test_unescaped_comma_in_a_scalar_number_only_warns(tmp_path: Path) -> None:
    """A numeric Number=1 field with several values is a header conflict, not a split value."""
    path = _write_escape_vcf(
        tmp_path / 'split_int.vcf',
        ['chr1\t1000\tsplit\tA\tG\t50\tPASS\tI1=3,7\tGT\t0/1'],
    )
    batch, messages = _read_with_warnings(path)
    row = _by_id(batch, 'snv')['split']

    assert row['vcf_info_I1'] == 3
    assert any('INFO/I1 declares Number=1' in m and 'discarding the rest' in m for m in messages)


def test_number_a_value_count_mismatch_warns(tmp_path: Path) -> None:
    """A Number=A field holding the wrong number of values indexes the wrong allele."""
    path = _write_escape_vcf(
        tmp_path / 'split_a.vcf',
        ['chr1\t1000\tsplit\tA\tG\t50\tPASS\tSA=DEL[1,234]\tGT\t0/1'],
    )

    assert any(
        'INFO/SA declares Number=A' in m and '2 values for 1 ALT alleles' in m
        for m in _warnings_from(path)
    )


def test_unbounded_string_is_left_split(tmp_path: Path) -> None:
    """Number=. is a list, so a split value is the declared shape and draws no warning."""
    path = _write_escape_vcf(
        tmp_path / 'split_d.vcf',
        ['chr1\t1000\tsplit\tA\tG\t50\tPASS\tSD=DEL[1,234]\tGT\t0/1'],
    )
    row = _by_id(next(agglovar.vcf.iter_vcf(path)), 'snv')['split']

    assert row['vcf_info_SD'] == ['DEL[1', '234]']
    assert _warnings_from(path) == []


def test_a_split_field_is_reported_once_per_file(tmp_path: Path) -> None:
    """A writer that escapes nothing breaks every record, so one message identifies the field."""
    path = _write_escape_vcf(
        tmp_path / 'split_many.vcf',
        [
            'chr1\t1000\ta\tA\tG\t50\tPASS\tS1=x,y\tGT\t0/1',
            'chr1\t2000\tb\tA\tG\t50\tPASS\tS1=p,q\tGT\t0/1',
            'chr1\t3000\tc\tA\tG\t50\tPASS\tS1=m,n\tGT\t0/1',
        ],
    )

    assert sum('INFO/S1' in m for m in _warnings_from(path)) == 1


def test_format_values_are_decoded_and_rejoined(tmp_path: Path) -> None:
    """FORMAT values are split and escaped the same way INFO values are."""
    path = _write_escape_vcf(
        tmp_path / 'fmt.vcf',
        [
            'chr1\t1000\tesc\tA\tG\t50\tPASS\t.\tGT:FS\t0/1:a%2Cb',
            'chr1\t2000\tsplit\tA\tG\t50\tPASS\t.\tGT:FS\t0/1:DEL[1,234]',
        ],
    )
    batch, messages = _read_with_warnings(path)
    samples = batch.sample_table.collect().sort('vcf_rec')

    assert samples['FS'].to_list() == ['a,b', 'DEL[1,234]']
    assert any('FORMAT/FS declares Number=1' in m for m in messages)


# ---------------------------------------------------------------------------
# Multiallelic symbolic records

_MULTI_SYM_HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##INFO=<ID=SVTYPE,Number=A,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Length of structural variant">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
"""


def test_multiallelic_symbolic_reads_per_allele_info(tmp_path: Path) -> None:
    """Each symbolic allele takes its own Number=A SVTYPE and SVLEN, not allele 1's.

    No callset in the test set carries a multiallelic symbolic record, so this is the only
    exercise of the per-allele read on that path.
    """
    path = tmp_path / 'multi_sym.vcf'
    path.write_text(
        _MULTI_SYM_HEADER
        + 'chr1\t1000\tmulti\tN\t<DEL>,<INS>\t50\tPASS\tSVTYPE=DEL,INS;SVLEN=-100,50\tGT\t1/2\n'
    )
    batch = next(agglovar.vcf.iter_vcf(path))
    rows = batch.insdel.collect().sort('vcf_allele').to_dicts()

    assert [r['vartype'] for r in rows] == ['DEL', 'INS']
    assert [r['varlen'] for r in rows] == [100, 50]
    assert [(r['pos'], r['end']) for r in rows] == [(1000, 1100), (1000, 1001)]
    assert _warnings_from(path) == []


# ---------------------------------------------------------------------------
# Header type checks

_OFF_SPEC_HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##ALT=<ID=DEL,Description="Deletion">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Difference in length between REF and ALT">
"""


def _write_vcf_info(path: Path, info_lines: list[str], records: list[str]) -> Path:
    """Write a sample-free VCF with extra INFO declarations *info_lines*, returning *path*."""
    path.write_text(
        _OFF_SPEC_HEADER
        + ''.join(f'{line}\n' for line in info_lines)
        + '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
        + ''.join(f'{rec}\n' for rec in records)
    )
    return path


def test_info_end_off_spec_number_reads(tmp_path: Path) -> None:
    """END declared ``Number=.`` reads as a scalar Int64 rather than failing the column build."""
    path = _write_vcf_info(
        tmp_path / 'end_dot.vcf',
        ['##INFO=<ID=END,Number=.,Type=Integer,Description="End position">'],
        ['chr1\t1000\tdel_dot\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;SVLEN=-50;END=1050'],
    )
    with pytest.warns(UserWarning, match='INFO/END declares'):
        batch = next(agglovar.vcf.iter_vcf(path))

    assert batch.insdel.collect_schema()['vcf_info_END'] == pl.Int64
    assert _by_id(batch, 'insdel')['del_dot']['vcf_info_END'] == 1050


def test_info_number_g_reads_as_null(tmp_path: Path) -> None:
    """Number=G INFO fields are unreadable through pysam; the column is null, not an error."""
    path = _write_vcf_info(
        tmp_path / 'number_g.vcf',
        ['##INFO=<ID=GL,Number=G,Type=Float,Description="Genotype likelihoods">'],
        ['chr1\t1000\tdel_g\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;SVLEN=-50;GL=1.0,2.0,3.0'],
    )
    with pytest.warns(UserWarning, match='Number=G'):
        batch = next(agglovar.vcf.iter_vcf(path))

    assert _by_id(batch, 'insdel')['del_g']['vcf_info_GL'] is None


def test_flag_with_non_zero_number_warns(tmp_path: Path) -> None:
    """A Flag declared with a Number other than 0 warns and still reads as Boolean."""
    path = _write_vcf_info(
        tmp_path / 'flag_dot.vcf',
        ['##INFO=<ID=FL,Number=.,Type=Flag,Description="Off-spec flag">'],
        ['chr1\t1000\tdel_fl\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;SVLEN=-50;FL'],
    )
    with pytest.warns(UserWarning, match='Type=Flag with Number'):
        batch = next(agglovar.vcf.iter_vcf(path))

    assert _by_id(batch, 'insdel')['del_fl']['vcf_info_FL'] is True


def test_number_zero_non_flag_warns(tmp_path: Path) -> None:
    """``Number=0`` outside a Flag carries no readable value, which is reported up front."""
    path = _write_vcf_info(
        tmp_path / 'number_zero.vcf',
        ['##INFO=<ID=X0,Number=0,Type=Integer,Description="Off-spec zero">'],
        ['chr1\t1000\tdel_x0\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;SVLEN=-50;X0'],
    )
    with pytest.warns(UserWarning, match='Number=0 requires Type=Flag'):
        batch = next(agglovar.vcf.iter_vcf(path))

    assert _by_id(batch, 'insdel')['del_x0']['vcf_info_X0'] is None


# ---------------------------------------------------------------------------
# Column type diagnostics

def test_frame_from_cols_reports_column_and_record() -> None:
    """A value conflicting with its column type names the column, INFO field, and record."""
    with pytest.raises(TypeError, match=r"'vcf_info_SVLEN'.*INFO/SVLEN.*vcf_rec=11"):
        _frame_from_cols(
            {'vcf_rec': [10, 11], 'vcf_info_SVLEN': [5, '50bp']},
            {'vcf_rec': pl.Int64, 'vcf_info_SVLEN': pl.Int64},
            'base',
        )


def test_frame_from_cols_reports_row_without_vcf_rec() -> None:
    """Accumulators carrying no ``vcf_rec`` fall back to the row offset within the batch."""
    with pytest.raises(TypeError, match=r"'varlen'.*batch row 1"):
        _frame_from_cols({'varlen': [1, 'x']}, {'varlen': pl.Int64}, 'base')


def test_frame_from_cols_passes_matching_columns() -> None:
    """Well-typed accumulators build without the diagnostic path altering the result."""
    df = _frame_from_cols(
        {'vcf_rec': [1, 2], 'chrom': ['chr1', 'chr2']},
        {'vcf_rec': pl.Int64, 'chrom': pl.String},
        'base',
    )

    assert df.to_dicts() == [
        {'vcf_rec': 1, 'chrom': 'chr1'},
        {'vcf_rec': 2, 'chrom': 'chr2'},
    ]


# ---------------------------------------------------------------------------
# Per-sample callsets

_MULTI_SAMPLE_HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
##FORMAT=<ID=VAF,Number=A,Type=Float,Description="Variant allele frequency">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Read depth for each allele">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
"""

_MULTI_SAMPLE_RECORDS = [
    'chr1\t1000\tsnv_multi\tA\tG,T\t50\tPASS\t.\tGT:DP:VAF:AD\t1/2:12:0.3,0.7:2,4,6\t0/1:6:0.1,0.9:1,2,3',
    'chr1\t1100\tdel_seq\tAT\tA\t50\tPASS\t.\tGT:DP:VAF:AD\t1/1:9:0.9:0,9\t0/0:8:0.0:8,0',
]


@pytest.fixture
def multi_sample_batch(tmp_path: Path) -> agglovar.vcf.VcfBatch:
    """Read a two-sample VCF with per-ALT FORMAT fields and return its single batch."""
    path = tmp_path / 'multi_sample.vcf'
    path.write_text(
        _MULTI_SAMPLE_HEADER + ''.join(f'{rec}\n' for rec in _MULTI_SAMPLE_RECORDS)
    )
    return next(agglovar.vcf.iter_vcf(path))


def test_get_sample_callset_stays_lazy(multi_sample_batch: agglovar.vcf.VcfBatch) -> None:
    """The callset is returned unevaluated for the caller to filter or project further."""
    assert isinstance(multi_sample_batch.get_sample_callset('S1', 'snv'), pl.LazyFrame)


def test_get_sample_callset_keeps_every_variant_row(
    multi_sample_batch: agglovar.vcf.VcfBatch,
) -> None:
    """One row per variant of the requested type, all carrying the requested sample."""
    callset = multi_sample_batch.get_sample_callset('S1', 'snv').collect()

    assert callset['vcf_alt'].to_list() == ['G', 'T']
    assert callset['vcf_sample'].to_list() == ['S1', 'S1']


def test_get_sample_callset_column_order(multi_sample_batch: agglovar.vcf.VcfBatch) -> None:
    """Variant columns lead and the sample columns follow in sample-table order."""
    columns = multi_sample_batch.get_sample_callset('S1', 'snv').collect_schema().names()
    var_cols = multi_sample_batch.snv.collect_schema().names()

    assert columns[:len(var_cols)] == var_cols
    assert columns[len(var_cols):] == ['vcf_sample', 'GT', 'DP', 'VAF', 'AD']


def test_get_sample_callset_number_a_collapses_to_allele(
    multi_sample_batch: agglovar.vcf.VcfBatch,
) -> None:
    """A ``Number=A`` FORMAT list becomes the scalar for that row's ``vcf_allele``."""
    callset = multi_sample_batch.get_sample_callset('S1', 'snv').collect().sort('vcf_allele')

    assert callset.schema['VAF'] == pl.Float64
    assert callset['VAF'].to_list() == pytest.approx([0.3, 0.7])


def test_get_sample_callset_other_numbers_unchanged(
    multi_sample_batch: agglovar.vcf.VcfBatch,
) -> None:
    """Only ``Number=A`` collapses; ``Number=R`` stays a list and ``Number=1`` stays scalar."""
    callset = multi_sample_batch.get_sample_callset('S1', 'snv').collect()

    assert callset.schema['AD'] == pl.List(pl.Int64)
    assert callset['AD'].to_list() == [[2, 4, 6], [2, 4, 6]]
    assert callset['DP'].to_list() == [12, 12]


def test_get_sample_callset_selects_the_named_sample(
    multi_sample_batch: agglovar.vcf.VcfBatch,
) -> None:
    """Each sample gets its own FORMAT values over the same variant rows."""
    s1 = multi_sample_batch.get_sample_callset('S1', 'snv').collect().sort('vcf_allele')
    s2 = multi_sample_batch.get_sample_callset('S2', 'snv').collect().sort('vcf_allele')

    assert s1['id'].to_list() == s2['id'].to_list()
    assert s1['GT'].to_list() == ['1/2', '1/2']
    assert s2['GT'].to_list() == ['0/1', '0/1']
    assert s2['VAF'].to_list() == pytest.approx([0.1, 0.9])


def test_get_sample_callset_restricted_to_vartype(
    multi_sample_batch: agglovar.vcf.VcfBatch,
) -> None:
    """Sample rows for records with no variant of the requested type do not appear."""
    insdel = multi_sample_batch.get_sample_callset('S1', 'insdel').collect()

    assert insdel['vartype'].to_list() == ['DEL']
    assert multi_sample_batch.get_sample_callset('S1', 'cpx').collect().height == 0


def test_get_sample_callset_rejects_unknown_vartype(
    multi_sample_batch: agglovar.vcf.VcfBatch,
) -> None:
    """An unknown variant type is rejected rather than raising on a missing attribute."""
    with pytest.raises(ValueError, match='unknown variant type'):
        multi_sample_batch.get_sample_callset('S1', 'indel')


def test_get_sample_callset_rejects_unknown_sample(
    multi_sample_batch: agglovar.vcf.VcfBatch,
) -> None:
    """A sample not declared in the header is rejected rather than returning an empty table."""
    with pytest.raises(ValueError, match='unknown sample'):
        multi_sample_batch.get_sample_callset('S3', 'snv')
