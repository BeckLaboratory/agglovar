"""
Get nearest variant by SVLEN overlap. Used for merging and comparing callsets.
"""

import polars as pl

from .. import seqmatch

COL_MAP_BED = {
    '#CHROM': 'chrom',
    'POS': 'pos',
    'END': 'end',
    'SVLEN': 'svlen',
    'ID': 'id',
    'REF': 'ref',
    'ALT': 'alt',
    'SEQ': 'seq',
    'MERGE_SAMPLES': 'merge_samples'
}

DEFAULT_PRIORITY_MATCH =[
    ('ro', 0.2),
    ('size_ro', 0.2),
    ('offset_prop', 0.1),
    ('match_prop', 0.5)
]

DEFAULT_PRIORITY_NOMATCH =[
    ('ro', 0.4),
    ('size_ro', 0.4),
    ('offset_prop', 0.2),
]

DEFAULT_PRIORITY = {
    'match': DEFAULT_PRIORITY_MATCH,
    'nomatch': DEFAULT_PRIORITY_NOMATCH,
    'default': DEFAULT_PRIORITY_MATCH
}

KNOWN_COL_SET = {
    'chrom', 'pos', 'end', 'id', 'svtype', 'svlen', 'ref', 'alt', 'seq'
}

def intersect(
    df_a: pl.DataFrame|pl.LazyFrame,
    df_b: pl.DataFrame|pl.LazyFrame,
    ro_min: float=None,
    size_ro_min: float=None,
    offset_max: int=None,
    offset_prop_max: float=None,
    match_prop_min: float=None,
    match_ref: bool=False,
    match_alt: bool=False,
    col_map: dict[str,str]=None,
    match_score_model: seqmatch.MatchScoreModel=None,
    force_end_ro: bool=False
) -> pl.LazyFrame:
    """
    Find all pairs of variants in two sources that meet a set of criteria.

    The columns expected in `df_a` and `df_b` are:
    * chrom: Variant chromosome.
    * pos: Variant position.
    * end: End position. If missing, is set to pos + svlen.
    * svlen: Variant length.
    * id: Variant ID. If missing or empty, will be generated using string "VarIdx" with the record index appended.
    * ref: Optional, required if `match_ref`.
    * alt: Optional, required if `match_alt`.
    * seq: Optional, required if `match_prop_min`.

    Columns in returned DataFrame:
    * index_a: Row index in `df_a`.
    * index_b: Row index in `df_b`.
    * id_a: Record id in `df_a`.
    * id_b: Record id in `df_b`.
    * ro: Reciprocal overlap if variants intersect.
    * size_ro: Size reciprocal overlap (Max RO if variants were shifted to maximally intersect).
    * offset_dist: Minimum of start position distance and end position distance.
    * offset_prop: Offset / variant length.
    * match_prop: Proportion of an alignment/match score between two sequences (from `df_a` and `df_b`) divided by the
        maximum match score if sequences were identical. If `match_prop_min` is None, this column contains only null
        values.

    :param df_a: Source dataframe.
    :param df_b: Target dataframe.
    :param ro_min: Minimum reciprocal overlap for allowed matches. If 0.0, then any overlap matches.
    :param size_ro_min: Reciprocal length proportion of allowed matches. If `match_prop_min` is set and the
        value of this parameter is `None` or is less than `match_prop_min`, then it is set to `match_prop_min` since
        this value represents the lower-bound of allowed match proportions.
    :param offset_max: Maximum offset allowed (minimum of start or end postion distance).
    :param offset_prop_max: Maximum size-offset (offset / svlen) allowed.
    :param match_prop_min: Minimum matched base proportion in alignment or None to not match.
    :param match_ref: "REF" column must match between two variants.
    :param match_alt: "ALT" column must match between two variants.
    :param col_map: Map column names from input dataframes by this dictionary. May be string "bed" to use default
        BED column names.
    :param match_score_model: Configured model for scoring similarity between pairs of sequences. If `None` and
        `match_prop_min` is set, then a default aligner will be used.
    :param force_end_ro: By default, reciprocal overlap is calculated with the end position set to the start
        position plus the variant length. For all variants except insertions, this will typically match the end value
        in the source DataFrame. If `True`, the end position in the DataFrame is also used for reciprocal overlap
        without changes. Typically, this option should not be used and will break reciprocal overlap for insertions.

    :return: A DataFrame describing overlap values.
    """

    #
    # Base arguments
    #

    # Check input types
    if isinstance(df_a, pl.DataFrame):
        df_a = df_a.lazy()
    elif not isinstance(df_a, pl.LazyFrame):
        raise TypeError(f'Variant source: Expected DataFrame or LazyFrame, got {type(df_a)}')

    if isinstance(df_b, pl.DataFrame):
        df_b = df_b.lazy()
    elif not isinstance(df_b, pl.LazyFrame):
        raise TypeError(f'Variant target: Expected DataFrame or LazyFrame, got {type(df_b)}')

    # Check match
    match_seq = match_prop_min is not None

    if match_seq:
        if match_score_model is None:
            match_score_model = seqmatch.MatchScoreModel()

        try:
            match_prop_min = float(match_prop_min)
        except ValueError:
            raise ValueError(f'Alignment proportion (match_prop_min) must be a number: {match_prop_min} (type {type(match_prop_min)})')

        if match_prop_min <= 0.0 or match_prop_min > 1.0:
            raise ValueError(f'Alignment proportion (match_prop_min) must be between 0.0 (exclusive) and 1.0 (inclusive): {match_prop_min}')


    #
    # Prepare and check DataFrames
    #

    # Column name mapping
    if isinstance(col_map, str) and col_map.strip().lower() == 'bed':
        col_map = COL_MAP_BED

    if col_map is not None:
        df_a = df_a.rename(lambda col: col_map.get(col, col))
        df_b = df_b.rename(lambda col: col_map.get(col, col))

    # Check for expected columns
    columns_a = [col for col in df_a.collect_schema().names() if col in KNOWN_COL_SET]
    columns_b = [col for col in df_b.collect_schema().names() if col in KNOWN_COL_SET]

    expected_cols = ['chrom', 'pos', 'end', 'svlen']

    if match_ref:
        expected_cols.append('ref')
    if match_alt:
        expected_cols.append('alt')
    if match_seq:
        expected_cols.append('seq')

    missing_cols_a = [col for col in expected_cols if col not in columns_a]
    missing_cols_b = [col for col in expected_cols if col not in columns_b]

    if missing_cols_a or missing_cols_b:
        if missing_cols_a == missing_cols_b:
            raise ValueError(f'DataFrames "A" and "B" are missing expected column(s): {", ".join(missing_cols_a)}')
        else:
            raise ValueError(f'DataFrame "A" missing expected column(s): {", ".join(missing_cols_a)}; DataFrame "B" missing expected column(s): {", ".join(missing_cols_b)}')

    # Subset to known columns only (i.e. drop other columns that might confilct, such as an arbitrary "index" column).
    df_a = df_a.select(columns_a)
    df_b = df_b.select(columns_b)

    # Set index
    df_a = df_a.with_row_index('index')
    df_b = df_b.with_row_index('index')

    # Set ID
    if 'id' not in columns_a:
        df_a = df_a.with_columns(
            pl.lit(None).alias('id').cast(pl.String)
        )

    if 'id' not in columns_b:
        df_b = df_b.with_columns(
            pl.lit(None).alias('id').cast(pl.String)
        )

    df_a = df_a.with_columns(
        pl.col('id').fill_null(
            pl.concat_str(pl.lit('VarIdx'), pl.col('index'))
        )
    )

    df_b = df_b.with_columns(
        pl.col('id').fill_null(
            pl.concat_str(pl.lit('VarIdx'), pl.col('index'))
        )
    )

    # ABS SVLEN (some sources may have negative SVLEN for DEL)
    df_a = df_a.with_columns(
        pl.col('svlen').abs()
    )

    df_b = df_b.with_columns(
        pl.col('svlen').abs()
    )

    # Prepare REF & ALT
    if match_ref:
        df_a.with_columns(
            pl.col('ref').str.to_uppercase().str.strip_chars()
        )

        df_b.with_columns(
            pl.col('ref').str.to_uppercase().str.strip_chars()
        )

    if match_alt:
        df_a.with_columns(
            pl.col('alt').str.to_uppercase().str.strip_chars()
        )

        df_b.with_columns(
            pl.col('alt').str.to_uppercase().str.strip_chars()
        )

    # Prepare SEQ
    if match_seq:
        df_a = df_a.with_columns(
            pl.col('seq').str.to_uppercase().str.strip_chars()
        )

        df_b = df_b.with_columns(
            pl.col('seq').str.to_uppercase().str.strip_chars()
        )

    # Get END for RO
    if not force_end_ro:
        df_a = df_a.with_columns(
            (pl.col('pos') + pl.col('svlen')).alias('end_ro')
        )

        df_b = df_b.with_columns(
            (pl.col('pos') + pl.col('svlen')).alias('end_ro')
        )
    else:
        df_a = df_a.with_columns(
            pl.col('end').alias('end_ro')
        )

        df_b = df_b.with_columns(
            pl.col('end').alias('end_ro')
        )

    # Append suffixes to all columns
    df_a = df_a.with_columns(pl.all().name.suffix('_a'))
    df_b = df_b.with_columns(pl.all().name.suffix('_b'))


    #
    # Reusable Expressions
    #

    expr_overlap_ro = (
        (
            pl.min_horizontal([pl.col('end_ro_a'), pl.col('end_ro_b')]) - pl.max_horizontal([pl.col('pos_a'), pl.col('pos_b')])
        ) / (
            pl.max_horizontal([pl.col('svlen_a'), pl.col('svlen_right')])
        )
    ).clip(0.0, 1.0)

    expr_szro = (
        pl.min_horizontal([pl.col('svlen_a'), pl.col('svlen_b')])
    ) / (
        pl.max_horizontal([pl.col('svlen_a'), pl.col('svlen_b')])
    )

    expr_offset_dist = pl.min_horizontal(*[
        (pl.col('pos_a').cast(pl.Int64) - pl.col('pos_b').cast(pl.Int64)).abs(),
        (pl.col('end_a').cast(pl.Int64) - pl.col('end_b').cast(pl.Int64)).abs()
    ]).cast(pl.UInt32)

    expr_offset_prop = (
        expr_offset_dist / pl.max_horizontal([pl.col('svlen_a'), pl.col('svlen_b')])
    )

    expr_match_prop = pl.struct(
        pl.col('seq_a'), pl.col('seq_b')
    ).map_elements(
        lambda s: match_score_model.match_prop(s['seq_a'], s['seq_b']),
        return_dtype=pl.Float32
    )


    #
    # Join rules
    #

    join_predicates = [
        pl.col.chrom == pl.col.chrom_right
    ]

    join_filters = list()

    # Check ro_min
    if ro_min is not None and ro_min > 0.0:
        try:
            ro_min = float(ro_min)
        except ValueError:
            raise ValueError(f'Reciprocal-overlap parameter (ro_min) is not a floating point number: {ro_min}')

        if ro_min < 0.0 or ro_min > 1.0:
            raise ValueError(f'Reciprocal-overlap parameter (ro_min) must be between 0.0 and 1.1 (inclusive): {ro_min}')

        join_predicates.extend([
            pl.col.pos < pl.col.end_ro_right,
            pl.col.pos_right < pl.col.end_ro,
            expr_overlap_ro >= ro_min
        ])

        join_filters.append(
            expr_overlap_ro >= ro_min
        )

    # Check szro_min
    if size_ro_min is not None:
        try:
            size_ro_min = float(size_ro_min)
        except ValueError:
            raise RuntimeError(f'Size-reciprocal-overlap parameter (size_ro_min) is not a floating point number: {size_ro_min}')

        if size_ro_min <= 0.0 or size_ro_min > 1.0:
            raise RuntimeError(f'Size-reciprocal-overlap parameter (size_ro_min) must be between 0 (exclusive) and 1 (inclusive): {size_ro_min}')

        if match_seq and match_prop_min > size_ro_min:
            size_ro_min = match_prop_min

        join_predicates.append(
            expr_szro >= size_ro_min
        )

    # Check offset_max
    if offset_max is not None:
        try:
            offset_max = int(offset_max)
        except ValueError:
            raise RuntimeError(f'Offset-max parameter (offset_max) is not an integer: {offset_max}')
        except OverflowError:
            raise RuntimeError(f'Offset-max parameter (offset_max) exceeds the max size (32 bits): {offset_max}')

        if offset_max < 0:
            raise RuntimeError(f'Offset-max parameter (offset_max) must not be negative: {offset_max}')

        join_predicates.append(
            expr_offset_dist <= offset_max
        )

    # Check offsz_max
    if offset_prop_max is not None:
        try:
            offset_prop_max = float(offset_prop_max)
        except ValueError:
            raise RuntimeError(f'Size-offset-max parameter (offset_prop_max) is not a floating point number: {offset_prop_max}')

        if offset_prop_max < float(0.0):
            raise RuntimeError(f'Size-offset-max parameter (offset_prop_max) must not be negative: {offset_prop_max}')

        join_predicates.append(
            expr_offset_prop <= offset_prop_max
        )

    # Match ref
    if match_ref:
        join_predicates.append(
            pl.col('ref') == pl.col('ref_right')
        )

    # Match alt
    if match_alt:
        join_predicates.append(
            pl.col('alt') == pl.col('alt_right')
        )


    #
    # Execute join and collect
    #

    col_list = [
        pl.col('index_a'),
        pl.col('index_b'),
        pl.col('id_a'),
        pl.col('id_b'),
        expr_overlap_ro.alias('ro').cast(pl.Float32),
        expr_offset_dist.alias('offset_dist').cast(pl.Int32),
        expr_offset_prop.alias('offset_prop').cast(pl.Float32),
        expr_szro.alias('size_ro').cast(pl.Float32)
    ] + (
        [pl.col('seq_a'), pl.col('seq_b')]
            if match_seq else []
    )

    # Join
    chrom_list = df_a.select('chrom').unique().collect().to_series().sort().to_list()

    df_join = pl.concat(
        [
            (
                (
                    df_a
                    .filter(pl.col('chrom') == chrom)
                )
                .join_where(
                    (
                        df_b
                        .filter(pl.col('chrom') == chrom)
                    ),
                    *join_predicates
                )
                .filter(
                    *join_filters
                )
                .select(
                    *col_list
                )
            )
            for chrom in chrom_list
        ]
    )

    # df_join = df_a.join_where(
    #     df_b,
    #     *join_predicates
    # ).filter(
    #     *join_filters
    # ).select(
    #     *col_list
    # )

    # Add match properties
    if match_seq:
        df_join = df_join.with_columns(
            expr_match_prop.alias('match_prop').cast(pl.Float32)
        )

        if match_prop_min is not None:
            df_join = df_join.filter(
                pl.col('match_prop') >= match_prop_min
            )

    else:
        df_join = df_join.with_columns(
            pl.lit(None).cast(pl.Float32).alias('match_prop')
        )

    df_join = df_join.drop(
        pl.col('^seq.*$'),
        strict=False
    )

    return df_join

def join_weight(
    df_join: pl.DataFrame|pl.LazyFrame,
    priority: list[tuple[str,float]]=None,
    offset_prop_max: float=2.0
) -> pl.DataFrame|pl.LazyFrame:
    """
    Add a "join_weight" column to an intersect DataaFrame computed by summing weighted values across columns of the
    input DataFrame.

    :param df_join: Intersect DataFrame or LazyFrame.
    :param priority: A list of (column, weight) tuples.
    :param offset_prop_max: Maximum value for `offset_prop`.

    :return: A DataFrame or LazyFrame (same type as `df_join`) with a "join_weight" column.
    """

    # Fill priority presets
    if priority is None:
        priority = DEFAULT_PRIORITY['default']
    elif isinstance(priority, str):
        priority = DEFAULT_PRIORITY.get(priority.strip().lower(), None)

        if priority is None:
            raise ValueError(f'Unknown priority preset: "{priority}"')

    # Set lazy
    if isinstance(df_join, pl.DataFrame):
        df_join = df_join.lazy()
        do_collect = True
    else:
        do_collect = False

    # Compute weights
    df_join = df_join.with_columns(
        pl.lit(0.0).alias('join_weight').cast(pl.Float32)
    )

    for col, weight in priority:
        if col not in {'ro', 'size_ro', 'offset_prop', 'match_prop'}:
            raise ValueError(f'Priority column must be one of ro, size_ro, offset_prop, or match_prop: {col}')

        if col != 'offset_prop':
            df_join = df_join.with_columns(
                (
                    pl.col('join_weight') +
                    pl.col(col).cast(pl.Float32) * float(weight)
                ).alias('join_weight')
            )

        else:
            df_join = df_join.with_columns(
                (
                    pl.col('join_weight') +
                    (
                        (1 - pl.col(col).clip(0.0, offset_prop_max) / offset_prop_max).cast(pl.Float32)
                    ) * float(weight)
                ).alias('join_weight')
            )

    # Collect and return
    if do_collect:
        df_join = df_join.collect()

    return df_join
