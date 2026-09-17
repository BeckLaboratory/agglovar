"""Tests for the "filter" column contract shared by the VCF reader and its consumers.

An empty list means the variant passed; a non-empty list holds the codes that failed.  A null
appears only when a source declared no filters and the reader was asked to keep that distinction
(:attr:`agglovar.vcf.VcfOpt.FILTER_RETAIN_NULL`), and every consumer treats it as passing.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import agglovar
from agglovar.expr.variant import sort_cols
from agglovar.pairwise.overlap._stage import PairwiseOverlapStage
from agglovar.util.var import id_version_expr


_HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=HET1,Description="Heterozygous in haplotype 1">
##FILTER=<ID=GAP1,Description="Assembly gap in haplotype 1">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
"""

_RECORDS = [
    'chr1\t1000\tmissing\tA\tG\t50\t.\t.\tGT\t0/1',
    'chr1\t1100\tpassing\tA\tG\t50\tPASS\t.\tGT\t0/1',
    'chr1\t1200\tone_code\tA\tG\t50\tHET1\t.\tGT\t0/1',
    'chr1\t1300\ttwo_codes\tA\tG\t50\tHET1;GAP1\t.\tGT\t0/1',
]


def _filters(path: Path, **kwargs) -> dict[str, list | None]:
    """Read *path* and map each record's VCF ID to its "filter" column value."""
    batch = next(agglovar.vcf.iter_vcf(path, **kwargs))
    return {
        row['vcf_id']: row['filter']
        for row in batch.snv.collect().to_dicts()
    }


@pytest.fixture
def filter_vcf(tmp_path: Path) -> Path:
    """Build a VCF covering missing, PASS, single-code, and multi-code FILTER values."""
    path = tmp_path / 'filters.vcf'
    path.write_text(_HEADER + ''.join(f'{rec}\n' for rec in _RECORDS))
    return path


# ---------------------------------------------------------------------------
# Reading

def test_missing_filter_reads_as_empty_list(filter_vcf: Path) -> None:
    """A missing FILTER reads as passing, like PASS, rather than as a null."""
    assert _filters(filter_vcf) == {
        'missing':   [],
        'passing':   [],
        'one_code':  ['HET1'],
        'two_codes': ['HET1', 'GAP1'],
    }


def test_filter_retain_null_keeps_the_vcf_distinction(filter_vcf: Path) -> None:
    """``FILTER_RETAIN_NULL`` separates "filters not applied" from "passed"."""
    assert _filters(filter_vcf, vcf_opt='FILTER_RETAIN_NULL') == {
        'missing':   None,
        'passing':   [],
        'one_code':  ['HET1'],
        'two_codes': ['HET1', 'GAP1'],
    }


def test_pass_is_never_reported_as_a_failed_code(filter_vcf: Path) -> None:
    """PASS is dropped from the codes rather than carried as one of them."""
    for value in _filters(filter_vcf).values():
        assert 'PASS' not in (value or [])


# ---------------------------------------------------------------------------
# Consumers

@pytest.fixture
def filter_frame() -> pl.DataFrame:
    """Build a frame holding each of the three filter states."""
    return pl.DataFrame(
        {
            'chrom':  ['chr1'] * 3,
            'pos':    [100, 100, 100],
            'end':    [101, 101, 101],
            'id':     ['v1', 'v1', 'v1'],
            'filter': [['HET1'], None, []],
        },
        schema={
            'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64,
            'id': pl.String, 'filter': pl.List(pl.String),
        },
    )


def test_sort_cols_sorts_null_with_passing(filter_frame: pl.DataFrame) -> None:
    """A null filter sorts alongside the empty list, not ahead of every other row."""
    assert filter_frame.sort(*sort_cols)['filter'].to_list() == [None, [], ['HET1']]


def test_id_version_expr_versions_null_filters(filter_frame: pl.DataFrame) -> None:
    """Duplicate IDs are versioned even when a filter is null, rather than colliding."""
    versioned = filter_frame.with_columns(id_version_expr())['id'].to_list()

    assert sorted(versioned) == ['v1', 'v1.1', 'v1.2']


def test_id_version_expr_ranks_passing_variants_first(filter_frame: pl.DataFrame) -> None:
    """Variants that passed take the lower versions; the one that failed is versioned last."""
    versioned = filter_frame.with_columns(id_version_expr())

    by_filter = {
        str(row['filter']): row['id']
        for row in versioned.to_dicts()
    }
    assert by_filter["['HET1']"] == 'v1.2'
    assert sorted((by_filter['None'], by_filter['[]'])) == ['v1', 'v1.1']


# ---------------------------------------------------------------------------
# match_filter_pass

@pytest.mark.parametrize(
    ('filter_a', 'filter_b', 'expected'),
    [
        ([],        [],        True),    # both passed
        (['HET1'],  ['GAP1'],  True),    # both failed, codes are not compared
        ([],        ['HET1'],  False),   # one passed, one failed
        (['HET1'],  [],        False),
        (None,      [],        True),    # null counts as passing
        (None,      None,      True),
        (None,      ['HET1'],  False),
    ],
)
def test_match_filter_pass(
    filter_a: list | None,
    filter_b: list | None,
    expected: bool,
) -> None:
    """Variants match when both passed or both failed, never when they disagree."""
    stage = PairwiseOverlapStage(match_filter_pass=True)
    predicate, = stage.join_predicates

    df = pl.DataFrame(
        {'filter_a': [filter_a], 'filter_b': [filter_b]},
        schema={'filter_a': pl.List(pl.String), 'filter_b': pl.List(pl.String)},
    )

    assert df.select(predicate.alias('match'))['match'][0] is expected
