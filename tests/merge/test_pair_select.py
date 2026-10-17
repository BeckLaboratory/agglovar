"""Tests for selecting one partner per variant in cumulative merges.

When a variant in a new source matches more than one variant in the cumulative table, the merge
must keep the highest-weight pair. Ties are broken by the lower cumulative index, then the lower
source index, so that the result does not depend on row order or thread count.
"""

import polars as pl

from agglovar.merge.cumulative import MergeCumulative
from agglovar.pairwise.overlap import PairwiseOverlap, PairwiseOverlapStage

# Number of independent competing groups. Many groups make an order-dependent selection fail with
# near certainty.
N_GROUPS = 2000

# Spacing between groups, far larger than any variant, so groups never interact.
GROUP_SPACE = 10_000

_FRAME_SCHEMA = {'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64, 'id': pl.String}


def _merge(df_a, df_b):
    """Merge two sources with a 50% reciprocal-overlap join."""
    join = PairwiseOverlap(stages=[PairwiseOverlapStage(ro_min=0.5)])
    return MergeCumulative(pairwise_join=join)([(df_a, 'a'), (df_b, 'b')]).collect()


def _partner_of_b(df_merged):
    """Map each source "b" variant ID to the source "a" variant ID it merged with."""
    return dict(
        df_merged
        .select(pl.col('mg_src'))
        .filter(pl.col('mg_src').list.len() == 2)
        .select(
            pl.col('mg_src').list.eval(
                pl.element().filter(pl.element().struct.field('src_name') == 'a').struct.field('var_id')
            ).list.first().alias('id_a'),
            pl.col('mg_src').list.eval(
                pl.element().filter(pl.element().struct.field('src_name') == 'b').struct.field('var_id')
            ).list.first().alias('id_b'),
        )
        .select('id_b', 'id_a')
        .iter_rows()
    )


def test_best_weight_partner_wins():
    """Each "b" variant merges with its higher-overlap "a" candidate, never the lower one.

    Per group, a_hi is [0, 100) and a_lo is [40, 140). b is [5, 105), so RO(b, a_hi) is 0.95 and
    RO(b, a_lo) is 0.65. Both pass the 0.5 threshold, a_lo has no other candidate, and only the
    second (per "b") selection decides which pair is kept.
    """
    rows_a, rows_b = [], []

    for g in range(N_GROUPS):
        base = g * GROUP_SPACE
        rows_a.append(('chr1', base, base + 100, f'a_hi-{g}'))
        rows_a.append(('chr1', base + 40, base + 140, f'a_lo-{g}'))
        rows_b.append(('chr1', base + 5, base + 105, f'b-{g}'))

    df_a = pl.DataFrame(rows_a, schema=_FRAME_SCHEMA, orient='row')
    df_b = pl.DataFrame(rows_b, schema=_FRAME_SCHEMA, orient='row')

    partner = _partner_of_b(_merge(df_a, df_b))

    assert len(partner) == N_GROUPS
    wrong = sorted(id_b for id_b, id_a in partner.items() if not id_a.startswith('a_hi-'))
    assert not wrong, f'{len(wrong)} variants merged with a lower-weight partner, e.g. {wrong[:5]}'


def test_tied_weight_lower_index_wins():
    """On equal weight, "b" merges with the "a" variant that comes first in the cumulative table.

    Per group, a_first is [0, 100) and a_second is [10, 110). b is [5, 105), which overlaps each by
    95 bp, so the two pairs tie.
    """
    rows_a, rows_b = [], []

    for g in range(N_GROUPS):
        base = g * GROUP_SPACE
        rows_a.append(('chr1', base, base + 100, f'a_first-{g}'))
        rows_a.append(('chr1', base + 10, base + 110, f'a_second-{g}'))
        rows_b.append(('chr1', base + 5, base + 105, f'b-{g}'))

    df_a = pl.DataFrame(rows_a, schema=_FRAME_SCHEMA, orient='row')
    df_b = pl.DataFrame(rows_b, schema=_FRAME_SCHEMA, orient='row')

    partner = _partner_of_b(_merge(df_a, df_b))

    assert len(partner) == N_GROUPS
    wrong = sorted(id_b for id_b, id_a in partner.items() if not id_a.startswith('a_first-'))
    assert not wrong, f'{len(wrong)} tied variants merged with the later partner, e.g. {wrong[:5]}'
