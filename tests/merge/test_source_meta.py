"""Tests for source and metadata tracking in cumulative merges.

These tests pin the contract for the ``mg_src`` column produced by a merge. Each merged record
carries an ``mg_src`` list with one struct per contributing source. Each struct has fields:

    * ``src_index``: Integer source index (0 for the first source, increments by 1).
    * ``src_name``: Source name (string).
    * ``src_meta``: Opaque metadata string carried through the merge, or null when no metadata
      was given. Strings pass through unchanged; dicts are JSON-encoded (an explicit empty dict
      becomes "{}").
    * ``var_index``: Integer row index of the variant within its source table.
    * ``var_id``: Variant ID (string).

Two callsets are merged, one with two sources and one with five sources. Each callset is defined
three ways: a bare list of LazyFrames, a list of (frame, name) tuples, and a list of
(frame, name, metadata) tuples. Two callsets times three definition styles gives six test cases.
The expected source/metadata for every merged variant is known up front and checked against the
result.
"""

import json

import polars as pl
import pytest

from agglovar.merge.cumulative import MergeCumulative
from agglovar.pairwise.overlap import PairwiseOverlap, PairwiseOverlapStage

# Expected struct fields on each ``mg_src`` entry.
MG_SRC_FIELDS = {'src_index', 'src_name', 'src_meta', 'var_index', 'var_id'}

# Definition styles for a callset.
STYLES = ['frames', 'named', 'meta']

# Frame schema used to build each source table.
_FRAME_SCHEMA = {'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64, 'id': pl.String}


# Each scenario describes a callset as an ordered list of sources. Every source has a name, an
# optional metadata value (used only by the "meta" style), and an ordered list of variants. The
# variant order fixes the per-source ``var_index``. The "clusters" list records which
# (source_index, var_index) pairs the merge is expected to combine into a single record, derived by
# hand from the variant positions and a 50% reciprocal-overlap threshold.
SCENARIOS = {
    'two_source': {
        'sources': [
            {
                'name': 'HG001',
                'meta': {'hap': '1'},
                'variants': [
                    {'id': 'HG001-v0', 'chrom': 'chr1', 'pos': 1000, 'end': 1100},
                    {'id': 'HG001-v1', 'chrom': 'chr1', 'pos': 2000, 'end': 2100},
                ],
            },
            {
                'name': 'HG002',
                'meta': {'hap': '2'},
                'variants': [
                    {'id': 'HG002-v0', 'chrom': 'chr1', 'pos': 1010, 'end': 1100},  # RO 0.90 with HG001-v0
                    {'id': 'HG002-v1', 'chrom': 'chr1', 'pos': 5000, 'end': 5100},  # No overlap
                ],
            },
        ],
        'clusters': [
            [(0, 0), (1, 0)],  # HG001-v0 + HG002-v0
            [(0, 1)],          # HG001-v1
            [(1, 1)],          # HG002-v1
        ],
    },
    'five_source': {
        'sources': [
            {
                'name': 'S1',
                'meta': {'hap': '1', 'tech': 'hifi'},
                'variants': [
                    {'id': 'S1-v0', 'chrom': 'chr1', 'pos': 1000, 'end': 1100},  # Cluster A lead
                ],
            },
            {
                'name': 'S2',
                'meta': {'hap': '2'},
                'variants': [
                    {'id': 'S2-v0', 'chrom': 'chr1', 'pos': 2000, 'end': 2100},  # Cluster B lead
                    {'id': 'S2-v1', 'chrom': 'chr1', 'pos': 3000, 'end': 3100},  # Singleton
                ],
            },
            {
                'name': 'S3',
                'meta': 'h1',  # Plain string metadata, passed through unchanged
                'variants': [
                    {'id': 'S3-v0', 'chrom': 'chr1', 'pos': 1005, 'end': 1105},  # RO 0.95 with S1-v0 (A)
                ],
            },
            {
                'name': 'S4',
                'meta': {'hap': '2', 'tech': 'ont'},
                'variants': [
                    {'id': 'S4-v0', 'chrom': 'chr1', 'pos': 2008, 'end': 2108},  # RO 0.92 with S2-v0 (B)
                ],
            },
            {
                'name': 'S5',
                'meta': {},  # Empty dict -> "{}"
                'variants': [
                    {'id': 'S5-v0', 'chrom': 'chr1', 'pos': 1010, 'end': 1110},  # RO 0.90 with S1-v0 (A)
                    {'id': 'S5-v1', 'chrom': 'chr1', 'pos': 4000, 'end': 4100},  # Singleton
                ],
            },
        ],
        'clusters': [
            [(0, 0), (2, 0), (4, 0)],  # Cluster A: S1-v0 + S3-v0 + S5-v0
            [(1, 0), (3, 0)],          # Cluster B: S2-v0 + S4-v0
            [(1, 1)],                  # S2-v1
            [(4, 1)],                  # S5-v1
        ],
    },
}


def _merge(callsets):
    """Merge callsets with a 50% reciprocal-overlap join (no sequence match)."""
    join = PairwiseOverlap(stages=[PairwiseOverlapStage(ro_min=0.5)])
    return MergeCumulative(pairwise_join=join)(callsets).collect()


def _build_frame(variants, lazy):
    """Build a source variant frame preserving variant order (fixes ``var_index``)."""
    df = pl.DataFrame(
        {
            'chrom': [v['chrom'] for v in variants],
            'pos': [v['pos'] for v in variants],
            'end': [v['end'] for v in variants],
            'id': [v['id'] for v in variants],
        },
        schema=_FRAME_SCHEMA,
    )
    return df.lazy() if lazy else df


def _make_callsets(scenario, style):
    """Build the callset argument for one scenario and definition style."""
    callsets = []

    for source in scenario['sources']:
        frame = _build_frame(source['variants'], lazy=(style == 'frames'))

        if style == 'frames':
            callsets.append(frame)
        elif style == 'named':
            callsets.append((frame, source['name']))
        elif style == 'meta':
            callsets.append((frame, source['name'], source['meta']))
        else:
            raise ValueError(f'Unknown style: {style}')

    return callsets


def _expected_src_name(scenario, style, src_index):
    """Expected source name: auto-generated for bare frames, otherwise the given name."""
    if style == 'frames':
        return f'source_{src_index + 1}'
    return scenario['sources'][src_index]['name']


def _expected_src_meta(scenario, style, src_index):
    """Expected metadata string: null unless an explicit metadata value was supplied."""
    if style != 'meta':
        return None

    meta = scenario['sources'][src_index]['meta']

    if isinstance(meta, str):
        return meta

    return json.dumps(meta)


def _expected_mg_src(scenario, style, cluster):
    """Expected ``mg_src`` entries for one merged cluster, sorted by source index."""
    rows = []

    for src_index, var_index in cluster:
        variant = scenario['sources'][src_index]['variants'][var_index]

        rows.append({
            'src_index': src_index,
            'src_name': _expected_src_name(scenario, style, src_index),
            'src_meta': _expected_src_meta(scenario, style, src_index),
            'var_index': var_index,
            'var_id': variant['id'],
        })

    return sorted(rows, key=lambda row: row['src_index'])


@pytest.mark.parametrize('scenario_name', sorted(SCENARIOS.keys()))
@pytest.mark.parametrize('style', STYLES)
def test_merge_tracks_source_and_metadata(scenario_name, style):
    """A merge tracks each variant's source index, name, metadata, and variant identity."""
    scenario = SCENARIOS[scenario_name]

    df = _merge(_make_callsets(scenario, style))

    # One merged record per expected cluster.
    assert df.height == len(scenario['clusters'])

    # mg_src struct must carry the new field set.
    mg_src_dtype = df.schema['mg_src']
    assert isinstance(mg_src_dtype, pl.List)
    assert {field.name for field in mg_src_dtype.inner.fields} == MG_SRC_FIELDS

    # Map each expected cluster by the set of variant IDs it should contain.
    expected_by_var_ids = {}

    for cluster in scenario['clusters']:
        var_ids = frozenset(
            scenario['sources'][src_index]['variants'][var_index]['id']
            for src_index, var_index in cluster
        )
        expected_by_var_ids[var_ids] = _expected_mg_src(scenario, style, cluster)

    # Every merged record matches its expected source/metadata entries.
    seen = set()

    for mg_src in df['mg_src'].to_list():
        var_ids = frozenset(entry['var_id'] for entry in mg_src)

        assert var_ids in expected_by_var_ids, f'Unexpected merged record: {sorted(var_ids)}'

        actual = sorted(mg_src, key=lambda entry: entry['src_index'])
        assert actual == expected_by_var_ids[var_ids]

        seen.add(var_ids)

    assert seen == set(expected_by_var_ids)


@pytest.mark.parametrize('scenario_name', sorted(SCENARIOS.keys()))
@pytest.mark.parametrize('style', STYLES)
def test_mg_src_lead_indexes_lead_entry(scenario_name, style):
    """``mg_src_lead`` indexes the lead entry within each record's ``mg_src`` list.

    The default ``LeadStrategy.LEFT`` picks the source with the earliest start position. This is
    checked against the actual ``mg_src`` list order (not an assumed sort), so it stays valid even
    if the list ordering changes.
    """
    scenario = SCENARIOS[scenario_name]

    df = _merge(_make_callsets(scenario, style))

    assert df.schema['mg_src_lead'].is_integer()

    for mg_src, lead in zip(df['mg_src'].to_list(), df['mg_src_lead'].to_list()):
        assert 0 <= lead < len(mg_src)

        # Lead is the entry with the smallest source start position (first on a tie), computed
        # over the list as actually returned.
        positions = [
            scenario['sources'][entry['src_index']]['variants'][entry['var_index']]['pos']
            for entry in mg_src
        ]
        assert lead == positions.index(min(positions))
