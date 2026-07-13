"""
Tests for segment reciprocal overlap (segment RO) of complex variants.

Segment RO is checked against a brute-force implementation of the measure (:func:`oracle_seg_ro`),
which enumerates every atomic range and counts segment depth directly. The implementation under test
computes the same quantity with a sweep, so the two agree only if the sweep is right.
"""

import random
from typing import Any

import polars as pl
import pytest

from agglovar.pairwise.overlap import (
    PairwiseOverlap,
    PairwiseOverlapStage,
)

SEG_DTYPE = pl.List(
    pl.Struct({
        'chrom': pl.String,
        'pos': pl.Int64,
        'end': pl.Int64,
        'qry_id': pl.String,
        'qry_pos': pl.Int64,
        'qry_end': pl.Int64,
        'is_rev': pl.Boolean,
    })
)
"""Segment column type. Matches the "seg" column PAV3 emits for complex variants."""


#
# Oracle
#

def oracle_overlap(seg_a: list[dict], seg_b: list[dict]) -> int:
    """Sum "width * min(depth_a, depth_b)" over atomic ranges, per chromosome and orientation.

    Brute-force: enumerate every segment boundary, then count covering segments in each range.

    :param seg_a: Segments of variant A.
    :param seg_b: Segments of variant B.

    :returns: Overlapping reference bases.
    """
    overlap = 0

    for key in {(seg['chrom'], seg['is_rev']) for seg in seg_a + seg_b}:
        key_a = [seg for seg in seg_a if (seg['chrom'], seg['is_rev']) == key]
        key_b = [seg for seg in seg_b if (seg['chrom'], seg['is_rev']) == key]

        bound_list = sorted(
            {seg['pos'] for seg in key_a + key_b} | {seg['end'] for seg in key_a + key_b}
        )

        for pos, end in zip(bound_list, bound_list[1:]):
            depth_a = sum(1 for seg in key_a if seg['pos'] <= pos and seg['end'] >= end)
            depth_b = sum(1 for seg in key_b if seg['pos'] <= pos and seg['end'] >= end)

            overlap += (end - pos) * min(depth_a, depth_b)

    return overlap


def oracle_unaligned(var: dict[str, Any]) -> int:
    """Get query bases covered by no segment.

    :param var: Variant record.

    :returns: Unaligned query bases.
    """
    return abs(var['qry_end'] - var['qry_pos']) - sum(
        abs(seg['qry_end'] - seg['qry_pos']) for seg in var['seg']
    )


def oracle_total_len(var: dict[str, Any]) -> int:
    """Get total segment length, aligned (reference bases) plus unaligned (query bases).

    :param var: Variant record.

    :returns: Total length.
    """
    return sum(seg['end'] - seg['pos'] for seg in var['seg']) + oracle_unaligned(var)


def oracle_seg_ro(var_a: dict[str, Any], var_b: dict[str, Any]) -> float:
    """Compute segment RO by brute force.

    :param var_a: Variant A.
    :param var_b: Variant B.

    :returns: Segment RO.
    """
    total_len = max(oracle_total_len(var_a), oracle_total_len(var_b))

    if total_len <= 0:
        return 1.0

    return (
        oracle_overlap(var_a['seg'], var_b['seg'])
        + min(oracle_unaligned(var_a), oracle_unaligned(var_b))
    ) / total_len


#
# Resources
#

def random_variant(seed: int) -> dict[str, Any]:
    """Build a random complex variant.

    Segment counts include 0 (no aligned segments), segments may repeat a locus (depth > 1), span
    two chromosomes, take either orientation, and have differing reference and query lengths (an
    internal indel).

    :param seed: Random seed.

    :returns: Variant record.
    """
    random.seed(seed)

    seg_list = []
    qry_pos = 0

    for _ in range(random.randint(0, 5)):
        pos = random.randint(0, 300)
        ref_len = random.randint(1, 60)
        qry_len = max(1, ref_len + random.randint(-8, 8))

        seg_list.append({
            'chrom': random.choice(['chr1', 'chr2']),
            'pos': pos,
            'end': pos + ref_len,
            'qry_id': 'qry1',
            'qry_pos': qry_pos,
            'qry_end': qry_pos + qry_len,
            'is_rev': random.random() < 0.35,
        })

        qry_pos += qry_len

    return {
        'seg': seg_list,
        'qry_pos': 0,
        'qry_end': qry_pos + random.randint(0, 50),
    }


def variant_frame(var_list: list[dict[str, Any]]) -> pl.LazyFrame:
    """Build a variant table from segment records.

    Reference coordinates are constant so that every pair joins and segment RO alone is tested.

    :param var_list: Variant records.

    :returns: Variant table.
    """
    return pl.DataFrame(
        [
            {
                'chrom': 'chr1',
                'pos': 0,
                'end': 100,
                'id': f'var{index}',
                'vartype': 'CPX',
                'varlen': 100,
                'seq': 'A' * 100,
                'seg': var['seg'],
                'qry_pos': var['qry_pos'],
                'qry_end': var['qry_end'],
            }
            for index, var in enumerate(var_list)
        ],
        schema_overrides={'seg': SEG_DTYPE},
    ).lazy()


def seg_ro_join(var_a: list[dict[str, Any]], var_b: list[dict[str, Any]]) -> pl.DataFrame:
    """Join two sets of complex variants emitting segment RO for every pair.

    :param var_a: Variants in table A.
    :param var_b: Variants in table B.

    :returns: Join table with "index_a", "index_b", and "seg_ro".
    """
    pairwise_join = PairwiseOverlap(
        [PairwiseOverlapStage(seg_ro_min=0.0)],
        join_cols=['seg_ro'],
    )

    return (
        pairwise_join
        .join(variant_frame(var_a), variant_frame(var_b))
        .select('index_a', 'index_b', 'seg_ro')
        .collect()
    )


@pytest.fixture(scope='module')
def var_list() -> list[dict[str, Any]]:
    """Get a set of random complex variants."""
    return [random_variant(seed) for seed in range(60)]


#
# Tests
#

def test_worked_example() -> None:
    """Check segment overlap against a hand-computed example.

    A: [100, 200) [300, 400) [800, 850) [1010, 1020)
    B: [150, 250) [300, 350) [1000, 1200)

    Overlaps: [150, 200) = 50, [300, 350) = 50, [1010, 1020) = 10.
    """
    def seg(pos, end, qry_pos):
        return {
            'chrom': 'chr1', 'pos': pos, 'end': end, 'qry_id': 'qry1',
            'qry_pos': qry_pos, 'qry_end': qry_pos + (end - pos), 'is_rev': False,
        }

    var_a = {'seg': [seg(100, 200, 0), seg(300, 400, 100), seg(800, 850, 200), seg(1010, 1020, 250)],
             'qry_pos': 0, 'qry_end': 260}
    var_b = {'seg': [seg(150, 250, 0), seg(300, 350, 100), seg(1000, 1200, 150)],
             'qry_pos': 0, 'qry_end': 350}

    assert oracle_overlap(var_a['seg'], var_b['seg']) == 110

    df_join = seg_ro_join([var_a], [var_b])

    # No unaligned bases on either side, so seg_ro is 110 / max(total_len).
    assert df_join.height == 1
    assert df_join['seg_ro'][0] == pytest.approx(
        110 / max(oracle_total_len(var_a), oracle_total_len(var_b)), abs=1e-6
    )


def test_matches_oracle(var_list: list[dict[str, Any]]) -> None:
    """Check segment RO against a brute-force implementation for every pair."""
    var_a = var_list[:30]
    var_b = var_list[30:]

    df_join = seg_ro_join(var_a, var_b)

    assert df_join.height == len(var_a) * len(var_b), 'Expected a full cross-join'

    for row in df_join.iter_rows(named=True):
        assert row['seg_ro'] == pytest.approx(
            oracle_seg_ro(var_a[row['index_a']], var_b[row['index_b']]), abs=1e-6
        ), f'Segment RO mismatch: index_a={row["index_a"]}, index_b={row["index_b"]}'


def test_self_is_one(var_list: list[dict[str, Any]]) -> None:
    """Check that a variant compared against itself has a segment RO of exactly 1.0."""
    df_join = seg_ro_join(var_list, var_list).filter(
        pl.col('index_a') == pl.col('index_b')
    )

    assert df_join.height == len(var_list)
    assert (df_join['seg_ro'] == 1.0).all()


def test_symmetric(var_list: list[dict[str, Any]]) -> None:
    """Check that segment RO is symmetric: seg_ro(A, B) == seg_ro(B, A)."""
    var_a = var_list[:30]
    var_b = var_list[30:]

    df_fwd = seg_ro_join(var_a, var_b)
    df_rev = seg_ro_join(var_b, var_a)

    df_cmp = df_fwd.join(
        df_rev, left_on=['index_a', 'index_b'], right_on=['index_b', 'index_a'], how='inner'
    )

    assert df_cmp.height == df_fwd.height
    assert (df_cmp['seg_ro'] - df_cmp['seg_ro_right']).abs().max() == pytest.approx(0.0, abs=1e-6)


def test_bounds(var_list: list[dict[str, Any]]) -> None:
    """Check that segment RO is bounded by [0, 1]."""
    df_join = seg_ro_join(var_list[:30], var_list[30:])

    assert df_join['seg_ro'].min() >= 0.0
    assert df_join['seg_ro'].max() <= 1.0


def test_chrom_separates_segments() -> None:
    """Check that segments on different chromosomes do not overlap."""
    def seg(chrom):
        return {
            'chrom': chrom, 'pos': 100, 'end': 200, 'qry_id': 'qry1',
            'qry_pos': 0, 'qry_end': 100, 'is_rev': False,
        }

    var_a = {'seg': [seg('chr1')], 'qry_pos': 0, 'qry_end': 100}
    var_b = {'seg': [seg('chr2')], 'qry_pos': 0, 'qry_end': 100}

    assert seg_ro_join([var_a], [var_b])['seg_ro'][0] == 0.0
    assert seg_ro_join([var_a], [var_a])['seg_ro'][0] == 1.0


def test_orientation_separates_segments() -> None:
    """Check that segments in opposite orientations do not overlap."""
    def seg(is_rev):
        return {
            'chrom': 'chr1', 'pos': 100, 'end': 200, 'qry_id': 'qry1',
            'qry_pos': 0, 'qry_end': 100, 'is_rev': is_rev,
        }

    var_fwd = {'seg': [seg(False)], 'qry_pos': 0, 'qry_end': 100}
    var_rev = {'seg': [seg(True)], 'qry_pos': 0, 'qry_end': 100}

    assert seg_ro_join([var_fwd], [var_rev])['seg_ro'][0] == 0.0
    assert seg_ro_join([var_fwd], [var_fwd])['seg_ro'][0] == 1.0


def test_no_double_count() -> None:
    """Check that one segment overlapping two segments is not counted twice.

    A has a single segment [100, 200).  B has two segments that each cover it, [100, 200) twice, so
    B has depth 2 across the range.  The minimum depth is 1, so the overlap is 100 bases, not 200.
    """
    def seg(pos, end, qry_pos):
        return {
            'chrom': 'chr1', 'pos': pos, 'end': end, 'qry_id': 'qry1',
            'qry_pos': qry_pos, 'qry_end': qry_pos + (end - pos), 'is_rev': False,
        }

    var_a = {'seg': [seg(100, 200, 0)], 'qry_pos': 0, 'qry_end': 100}
    var_b = {'seg': [seg(100, 200, 0), seg(100, 200, 100)], 'qry_pos': 0, 'qry_end': 200}

    assert oracle_overlap(var_a['seg'], var_b['seg']) == 100

    # total_len: A = 100, B = 200 (the duplicated segment counts twice).  seg_ro = 100 / 200.
    assert seg_ro_join([var_a], [var_b])['seg_ro'][0] == pytest.approx(0.5, abs=1e-6)


def test_no_segments() -> None:
    """Check variants with no aligned segments.

    All query bases are unaligned, so segment RO is the ratio of the shorter to the longer.
    """
    var_a = {'seg': [], 'qry_pos': 0, 'qry_end': 100}
    var_b = {'seg': [], 'qry_pos': 0, 'qry_end': 200}

    assert seg_ro_join([var_a], [var_b])['seg_ro'][0] == pytest.approx(0.5, abs=1e-6)
    assert seg_ro_join([var_a], [var_a])['seg_ro'][0] == 1.0
