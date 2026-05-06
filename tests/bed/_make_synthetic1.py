"""Deterministic generator for the ``synthetic1`` BED test dataset.

The dataset is laid out by chromosome, where each chromosome is a labelled
case probing a specific behaviour of the bed module (touching boundaries,
nested intervals, multi-B coverage, B-only chromosomes, large-vs-small
mixes, chunking pressure, etc.). Records are hand-written; nothing is
random.

Run this module to (re)generate the parquet files at
``local/test_data/bed/synthetic1/{a,b}.parquet``::

    uv run python -m tests.bed._make_synthetic1

The harness in ``tests/bed/conftest.py`` will also call ``write_parquet``
automatically the first time the dataset is requested.

Schema written: ``chrom`` (String), ``pos`` (Int64), ``end`` (Int64), ``id``
(String). Only the fields used by ``agglovar.bed`` are included; the
``agglovar.schema.VARIANT`` types are still honoured.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import polars as pl

import agglovar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / 'local' / 'test_data' / 'bed' / 'synthetic1'

SCHEMA = {
    'chrom': agglovar.schema.VARIANT['chrom'],
    'pos': agglovar.schema.VARIANT['pos'],
    'end': agglovar.schema.VARIANT['end'],
    'id': agglovar.schema.VARIANT['id'],
}


@dataclass(frozen=True)
class Case:
    """A labelled chromosome of records exercising one or more behaviours.

    :ivar name: Human-readable case identifier (also used as the chromosome
        name for the records).
    :ivar description: What this case is meant to exercise.
    :ivar a: Iterable of ``(pos, end)`` pairs for table A.
    :ivar b: Iterable of ``(pos, end)`` pairs for table B.
    """

    name: str
    description: str
    a: tuple[tuple[int, int], ...]
    b: tuple[tuple[int, int], ...]


CASES: tuple[Case, ...] = (
    # --- size range: 1 bp to 1 Mbp ----------------------------------------
    Case(
        name='chr_size_range',
        description=(
            'Sizes 1 bp, 5 bp, 50 bp, 500 bp, 5 kbp, 50 kbp, 500 kbp, '
            '1 Mbp. Mix of exact match, partial overlap, full containment, '
            'no overlap, and a touching boundary (A.pos == B.end).'
        ),
        a=(
            (100, 101),               # 1 bp
            (200, 205),               # 5 bp
            (1_000, 1_050),           # 50 bp, A fully inside B
            (10_000, 10_500),         # 500 bp, partial overlap
            (100_000, 105_000),       # 5 kbp, partial overlap
            (1_000_000, 1_050_000),   # 50 kbp, no overlap with any B
            (5_000_000, 5_500_000),   # 500 kbp, partial overlap
            (10_000_000, 11_000_000), # 1 Mbp, partial overlap (large)
            (15_000_000, 15_000_001), # 1 bp at the right boundary
        ),
        b=(
            (100, 101),               # exact match
            (203, 207),               # 5 bp partial overlap with A's 5 bp
            (1_000, 1_100),           # contains A's 50 bp
            (10_250, 10_750),         # offset overlap with A's 500 bp
            (99_000, 100_500),        # partial overlap with A's 5 kbp
            (1_100_000, 1_200_000),   # gap from A's 50 kbp record
            (5_250_000, 5_750_000),   # partial overlap with A's 500 kbp
            (10_500_000, 12_000_000), # partial overlap with A's 1 Mbp
            (14_999_999, 15_000_000), # touches A.pos=15_000_000 (B.end == A.pos)
        ),
    ),

    # --- sparse non-overlapping --------------------------------------------
    Case(
        name='chr_sparse',
        description='A and B records do not overlap and have measurable gaps.',
        a=(
            (100, 200),
            (1_000, 2_000),
            (10_000, 20_000),
        ),
        b=(
            (300, 400),               # 100 bp gap from A.end
            (2_500, 3_000),           # 500 bp gap
            (50_000, 60_000),         # 30 kbp gap
        ),
    ),

    # --- A-only / B-only chromosomes ---------------------------------------
    Case(
        name='chr_a_only',
        description='A has records, B has none on this chromosome.',
        a=(
            (100, 200),
            (300, 400),
        ),
        b=(),
    ),
    Case(
        name='chr_b_only',
        description='B has records, A has none on this chromosome.',
        a=(),
        b=(
            (100, 200),
            (300, 400),
        ),
    ),

    # --- touching boundary (BED half-open) ---------------------------------
    Case(
        name='chr_touch',
        description=(
            'BED half-open touching cases: B.pos == A.end and B.end == A.pos. '
            'At distance=0, touching pairs are included with output distance=0.'
        ),
        a=(
            (100, 200),
            (300, 400),
            (500, 600),
        ),
        b=(
            (200, 300),               # touches both (100,200) and (300,400)
            (400, 500),               # touches both (300,400) and (500,600)
            (600, 700),               # touches (500,600)
        ),
    ),

    # --- multi-B coverage (no overlap among B) -----------------------------
    Case(
        name='chr_multi_b_disjoint',
        description=(
            'One A interval covered by multiple non-overlapping B intervals. '
            'Expected as_proportion: (300+200+200)/1000 = 0.7'
        ),
        a=(
            (1_000, 2_000),           # 1 kbp
        ),
        b=(
            (1_000, 1_300),           # 30 % of A
            (1_500, 1_700),           # 20 % of A (interior)
            (1_800, 2_200),           # 20 % of A (extends past)
        ),
    ),

    # --- multi-B coverage with B intervals overlapping each other ----------
    Case(
        name='chr_multi_b_overlap',
        description=(
            'B intervals overlap each other; as_proportion must internally '
            'merge B before computing coverage so overlap is not double '
            'counted. Expected: (1800-1000) + (2000-1900) = 900/1000 = 0.9.'
        ),
        a=(
            (1_000, 2_000),
        ),
        b=(
            (1_000, 1_500),
            (1_300, 1_800),           # overlaps (1_000, 1_500)
            (1_900, 2_100),           # disjoint from the merged (1_000, 1_800)
        ),
    ),

    # --- containment, both directions -------------------------------------
    Case(
        name='chr_contains_a',
        description='A contains all B intervals. as_proportion < 1.',
        a=(
            (1_000, 2_000),
        ),
        b=(
            (1_200, 1_300),           # 100 bp inside A
            (1_500, 1_600),           # 100 bp inside A
        ),
    ),
    Case(
        name='chr_contains_b',
        description='B contains A. as_proportion = 1.0.',
        a=(
            (1_500, 1_600),
        ),
        b=(
            (1_000, 2_000),
        ),
    ),

    # --- duplicates --------------------------------------------------------
    Case(
        name='chr_duplicates',
        description='Identical intervals duplicated in both tables.',
        a=(
            (1_000, 2_000),
            (1_000, 2_000),
        ),
        b=(
            (1_000, 2_000),
            (1_000, 2_000),
        ),
    ),

    # --- many-to-many cross --------------------------------------------------
    Case(
        name='chr_many_to_many',
        description='Three A x two B all overlapping; six join records.',
        a=(
            (1_000, 1_100),
            (1_000, 1_100),
            (1_000, 1_100),
        ),
        b=(
            (1_050, 1_150),
            (1_050, 1_150),
        ),
    ),

    # --- chunking pressure --------------------------------------------------
    Case(
        name='chr_chunk_dense',
        description=(
            '50 small (1 bp) records on A bunched in [1000, 1050) plus one '
            'far away. Used with chunk_size in {1,2,5,...} to force the '
            'chunking loop to iterate without changing the underlying join.'
        ),
        a=tuple(
            (1_000 + i, 1_000 + i + 1)
            for i in range(50)
        ) + (
            (1_000_000, 1_000_001),
        ),
        b=(
            (1_000, 1_010),           # overlaps A[0..9]
            (1_025, 1_030),           # overlaps A[25..29]
            (1_045, 1_055),           # overlaps A[45..49]
            (1_000_000, 1_000_001),   # exact match for A's far record
        ),
    ),

    # --- zero-length intervals ----------------------------------------------
    Case(
        name='chr_zero_len',
        description=(
            'Zero-length records (pos == end). BED allows these as insertion '
            'points. Pin behaviour explicitly: pairwise_join filter is '
            'B.pos <= A.end and B.end >= A.pos so a (1000,1000) on each side '
            'matches; as_proportion divides by len = end - pos = 0 (NaN/inf).'
        ),
        a=(
            (1_000, 1_000),
            (2_000, 2_000),
            (3_000, 3_100),
        ),
        b=(
            (1_000, 1_000),
            (1_500, 1_500),
            (3_050, 3_050),
        ),
    ),

    # --- large interval over many small ones --------------------------------
    Case(
        name='chr_large_over_small',
        description=(
            'One large A interval [0, 1_000_000) over 20 small B records. '
            'Stresses interval-tree query with a single very wide A.'
        ),
        a=(
            (0, 1_000_000),
        ),
        b=tuple(
            (50_000 * (i + 1), 50_000 * (i + 1) + 100)
            for i in range(19)
        ) + (
            (1_500_000, 1_500_100),   # outside A; should not match
        ),
    ),

    # --- entire 1 Mbp records overlapping each other ------------------------
    Case(
        name='chr_huge_overlap',
        description='Two 1 Mbp records partially overlapping.',
        a=(
            (0, 1_000_000),
        ),
        b=(
            (500_000, 1_500_000),     # 500 kbp overlap
        ),
    ),

    # --- gap = exactly 1 bp (for distance parameter tests) ------------------
    Case(
        name='chr_gap_one',
        description=(
            'Records separated by exactly 1 bp gap. Hits at distance>=1, '
            'not at distance=0. Touching pair with B.pos == A.end is also '
            'present and hits at distance=0 (under the polars filter).'
        ),
        a=(
            (100, 200),
            (1_000, 1_100),
            (5_000, 5_100),
        ),
        b=(
            (201, 300),               # gap = 1 (after A.end=200)
            (900, 999),               # gap = 1 (before A.pos=1000)
            (5_100, 5_200),           # touches A.end=5100
        ),
    ),

    # --- non-canonical chromosome name --------------------------------------
    Case(
        name='chr_unplaced_random_19_long_name_test_lexsort',
        description='Long chromosome name with underscores; verifies string handling and sort.',
        a=(
            (100, 200),
        ),
        b=(
            (150, 250),
        ),
    ),

    # --- short common name --------------------------------------------------
    Case(
        name='chrM',
        description='Mitochondrial-style short name; 1 bp records.',
        a=(
            (1, 2),
        ),
        b=(
            (1, 2),
        ),
    ),
)


def _build_table(cases: Iterable[Case], side: str) -> pl.DataFrame:
    """Materialize one side of the dataset into a typed DataFrame."""
    if side not in ('a', 'b'):
        raise ValueError(f'side must be "a" or "b", got {side!r}')

    rows: list[tuple[str, int, int, str]] = []

    for case in cases:
        records = case.a if side == 'a' else case.b
        for i, (pos, end) in enumerate(records):
            rows.append((case.name, int(pos), int(end), f'{side}_{case.name}_{i:03d}'))

    return pl.DataFrame(
        rows,
        schema=SCHEMA,
        orient='row',
    )


def write_parquet(out_dir: Path = DATASET_DIR) -> dict[str, Path]:
    """Write A and B parquet files, plus a small metadata.json describing the cases.

    :param out_dir: Destination directory. Created if missing.

    :return: Mapping of ``{'a': path_a, 'b': path_b, 'meta': path_meta}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df_a = _build_table(CASES, 'a')
    df_b = _build_table(CASES, 'b')

    path_a = out_dir / 'a.parquet'
    path_b = out_dir / 'b.parquet'
    path_meta = out_dir / 'metadata.json'

    df_a.write_parquet(path_a)
    df_b.write_parquet(path_b)

    metadata = {
        'name': 'synthetic1',
        'description': (
            'Hand-laid deterministic BED records for unit tests. Each '
            'chromosome is one case; see "cases" for what each exercises.'
        ),
        'schema': {col: str(dtype) for col, dtype in SCHEMA.items()},
        'n_records': {
            'a': df_a.height,
            'b': df_b.height,
        },
        'cases': [
            {
                'name': case.name,
                'description': case.description,
                'a_count': len(case.a),
                'b_count': len(case.b),
            }
            for case in CASES
        ],
    }

    path_meta.write_text(json.dumps(metadata, indent=2) + '\n')

    return {'a': path_a, 'b': path_b, 'meta': path_meta}


def main() -> None:
    """CLI entry point: regenerate the dataset."""
    paths = write_parquet()
    print(f'Wrote {paths["a"]}')
    print(f'Wrote {paths["b"]}')
    print(f'Wrote {paths["meta"]}')


if __name__ == '__main__':
    main()
