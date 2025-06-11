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
    ('overlap_ro', 0.2),
    ('overlap_size_ro', 0.2),
    ('offset_prop', 0.1),
    ('match_prop', 0.5)
]

DEFAULT_PRIORITY_NOMATCH =[
    ('overlap_ro', 0.4),
    ('overlap_size_ro', 0.4),
    ('offset_prop', 0.2),
]

DEFAULT_PRIORITY = {
    'match': DEFAULT_PRIORITY_MATCH,
    'nomatch': DEFAULT_PRIORITY_NOMATCH,
    'default': DEFAULT_PRIORITY_MATCH
}

def intersect_pair(
    df_a: pl.DataFrame|pl.LazyFrame,
    df_b: pl.DataFrame|pl.LazyFrame,
    overlap_ro_min: float=None,
    offset_dist_max: int=None,
    overlap_size_ro_min: float=None,
    offset_prop_max: float=None,
    match_prop_min: float=None,
    match_ref: bool=False,
    match_alt: bool=False,
    col_map: dict[str,str]=None,
    match_score_model: seqmatch.MatchScoreModel=None,
    force_end_ro: bool=False,
) -> pl.DataFrame:
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
    * overlap_ro: Reciprocal overlap if variants intersect.
    * overlap_size_ro: Size reciprocal overlap (Max RO if variants were shifted to maximally intersect).
    * offset_dist: Minimum of start position distance and end position distance.
    * offset_prop: Offset / variant length.
    * match_prop_min: Proportion of an alignment/match score between two sequences (from `df_a` and `df_b`) divided by the
        maximum match score if sequences were identical.

    :param df_a: Source dataframe.
    :param df_b: Target dataframe.
    :param overlap_ro_min: Minimum reciprocal overlap for allowed matches.
    :param offset_dist_max: Maximum offset allowed (minimum of start or end postion distance).
    :param overlap_size_ro_min: Reciprocal length proportion of allowed matches.
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

    if isinstance(match_prop_min, bool):
        match_seq = match_prop_min
        match_prop_min = None
    else:
        match_seq = match_prop_min is not None

    # Check input types
    if isinstance(df_a, pl.DataFrame):
        df_a = df_a.lazy()
    elif not isinstance(df_a, pl.LazyFrame):
        raise TypeError(f'Variant source: Expected DataFrame or LazyFrame, got {type(df_a)}')

    if isinstance(df_b, pl.DataFrame):
        df_b = df_b.lazy()
    elif not isinstance(df_b, pl.LazyFrame):
        raise TypeError(f'Variant target: Expected DataFrame or LazyFrame, got {type(df_b)}')

    if match_score_model is None:
        match_score_model = seqmatch.MatchScoreModel()


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
    columns_a = df_a.collect_schema().names()
    columns_b = df_b.collect_schema().names()

    missing_cols = [col for col in ('chrom', 'pos', 'end', 'svlen') if col not in columns_a]

    if missing_cols:
        raise ValueError(f'DataFrame "A" missing expected column(s): {", ".join(missing_cols)}')

    missing_cols = [col for col in ('chrom', 'pos', 'end', 'svlen') if col not in columns_b]

    if missing_cols:
        raise ValueError(f'DataFrame "B" missing expected column(s): {", ".join(missing_cols)}')

    # Set index
    df_a = df_a.with_row_index('_index')
    df_b = df_b.with_row_index('_index')

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
        pl.col.id.fill_null(
            pl.concat_str(pl.lit('VarIdx'), pl.col('_index'))
        )
    )

    df_b = df_b.with_columns(
        pl.col.id.fill_null(
            pl.concat_str(pl.lit('VarIdx'), pl.col('_index'))
        )
    )

    # ABS SVLEN (some sources may have negative SVLEN for DEL)
    df_a = df_a.with_columns(
        pl.col.svlen.abs()
    )

    df_b = df_b.with_columns(
        pl.col.svlen.abs()
    )

    # Prepare REF
    if match_ref:
        if 'ref' not in columns_a:
            raise ValueError('Source table is missing REF column (required when matching reference base)')

        if 'ref' not in columns_b:
            raise ValueError('Target table is missing REF column (required when matching reference base)')

        df_a.with_columns(
            pl.col.ref.str.to_uppercase().str.strip_chars()
        )

        df_b.with_columns(
            pl.col.ref.str.to_uppercase().str.strip_chars()
        )

    # Prepare ALT
    if match_alt:
        if 'alt' not in df_a.columns:
            raise ValueError('Source table is missing ALT column (required when matching reference base)')

        if 'alt' not in df_a.columns:
            raise ValueError('Target table is missing ALT column (required when matching reference base)')

        df_a.with_columns(
            pl.col.alt.str.to_uppercase().str.strip_chars()
        )

        df_b.with_columns(
            pl.col.alt.str.to_uppercase().str.strip_chars()
        )

    # Prepare SEQ
    if match_seq:

        if 'seq' not in columns_a:
            raise ValueError('DataFrame "A" is missing "seq" column (required when matching variant sequences)')

        if 'seq' not in columns_b:
            raise ValueError('DataFrame "B" is missing "seq" column (required when matching variant sequences)')

        if match_prop_min is not None:
            match_prop_min = float(match_prop_min)

            if match_prop_min <= 0.0 or match_prop_min > 1.0:
                raise ValueError(f'Alignment proportion (match_prop_min) must be between 0.0 (exclusive) and 1.0 (inclusive): {match_prop_min}')

        df_a = df_a.with_columns(
            pl.col.seq.fill_null('N').str.to_uppercase().str.strip_chars()
        )

        df_b = df_b.with_columns(
            pl.col.seq.fill_null('N').str.to_uppercase().str.strip_chars()
        )

    # Get END for RO
    if not force_end_ro:
        df_a = df_a.with_columns(
            (pl.col.pos + pl.col.svlen).alias('end_ro')
        )

        df_b = df_b.with_columns(
            (pl.col.pos + pl.col.svlen).alias('end_ro')
        )
    else:
        df_a = df_a.with_columns(
            pl.col.end.alias('end_ro')
        )

        df_b = df_b.with_columns(
            pl.col.end.alias('end_ro')
        )


    #
    # Reusable Expressions
    #

    expr_overlap_ro = (
        pl.min_horizontal([pl.col.end_ro, pl.col.end_ro_right]) - pl.max_horizontal([pl.col.pos, pl.col.pos_right])
    ) / (
        pl.max_horizontal([pl.col.svlen, pl.col.svlen_right])
    )

    expr_szro = (
        pl.min_horizontal([pl.col.svlen, pl.col.svlen_right])
    ) / (
        pl.max_horizontal([pl.col.svlen, pl.col.svlen_right])
    )

    expr_offset_dist = pl.min_horizontal(*[
        (pl.col.pos.cast(pl.Int64) - pl.col.pos_right.cast(pl.Int64)).abs(),
        (pl.col.end.cast(pl.Int64) - pl.col.end_right.cast(pl.Int64)).abs()
    ]).cast(pl.UInt32)

    expr_offset_prop = (
        expr_offset_dist / pl.max_horizontal([pl.col.svlen, pl.col.svlen_right])
    )

    expr_match_prop = pl.struct(
        pl.col.seq, pl.col.seq_right
    ).map_elements(
        lambda s: match_score_model.match_prop(s['seq'], s['seq_right']),
        return_dtype=pl.Float32
    )



    #
    # Join rules
    #

    join_predicates = [
        pl.col.chrom == pl.col.chrom
    ]

    join_filters = list()

    # Check ro_min
    if overlap_ro_min is not None:
        try:
            overlap_ro_min = float(overlap_ro_min)
        except ValueError:
            raise ValueError(f'Reciprocal-overlap parameter (overlap_ro) is not a floating point number: {overlap_ro_min}')

        if overlap_ro_min <= 0.0 or overlap_ro_min > 1.0:
            raise ValueError(f'Reciprocal-overlap parameter (overlap_ro_min) must be between 0 (exclusive) and 1 (inclusive): {overlap_ro_min}')

        join_predicates.extend([
            pl.col.pos < pl.col.end_ro_right,
            pl.col.pos_right < pl.col.end_ro,
            expr_overlap_ro >= overlap_ro_min
        ])

        join_filters.append(
            expr_overlap_ro >= overlap_ro_min
        )

    # Check szro_min
    if overlap_size_ro_min is not None:
        try:
            overlap_size_ro_min = float(overlap_size_ro_min)
        except ValueError:
            raise RuntimeError(f'Size-reciprocal-overlap parameter (overlap_size_ro_min) is not a floating point number: {overlap_size_ro_min}')

        if overlap_size_ro_min <= 0.0 or overlap_size_ro_min > 1.0:
            raise RuntimeError(f'Size-reciprocal-overlap parameter (overlap_size_ro_min) must be between 0 (exclusive) and 1 (inclusive): {overlap_size_ro_min}')

        join_predicates.append(
            expr_szro >= overlap_size_ro_min
        )

    # Check offset_max
    if offset_dist_max is not None:
        try:
            offset_dist_max = int(offset_dist_max)
        except ValueError:
            raise RuntimeError(f'Offset-max parameter (offset_dist_max) is not an integer: {offset_dist_max}')
        except OverflowError:
            raise RuntimeError(f'Offset-max parameter (offset_dist_max) exceeds the max size (32 bits): {offset_dist_max}')

        if offset_dist_max < 0:
            raise RuntimeError(f'Offset-max parameter (offset_dist_max) must not be negative: {offset_dist_max}')

        join_predicates.append(
            expr_offset_dist <= offset_dist_max
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
            pl.col.ref == pl.col.ref_right
        )

    # Match alt
    if match_alt:
        join_predicates.append(
            pl.col.alt == pl.col.alt_right
        )


    #
    # Execute join and collect
    #

    col_list = [
        pl.col('_index').alias('index_a'),
        pl.col('_index_right').alias('index_b'),
        pl.col.id.alias('id_a'),
        pl.col.id_right.alias('id_b'),
        expr_overlap_ro.alias('overlap_ro').cast(pl.Float32),
        expr_offset_dist.alias('offset_dist').cast(pl.UInt32),
        expr_offset_prop.alias('offset_prop').cast(pl.Float32),
        expr_szro.alias('overlap_size_ro').cast(pl.Float32)
    ] + (
        [pl.col.seq, pl.col.seq_right]
            if match_seq else []
    )

    # Join
    df_intersect = df_a.join_where(
        df_b,
        *join_predicates
    ).filter(
        *join_filters
    ).select(
        *col_list
    ).collect()

    # Add match properties
    if match_seq:
        df_intersect = df_intersect.with_columns(
            expr_match_prop.alias('match_prop').cast(pl.Float32)
        )

        if match_prop_min is not None:
            df_intersect = df_intersect.filter(
                pl.col.match_prop >= match_prop_min
            )

    else:
        df_intersect = df_intersect.with_columns(
            pl.lit(None).cast(pl.Float32).alias('match_prop')
        )

    df_intersect = df_intersect.drop(
        pl.col('^seq.*$'),
        strict=False
    )

    return df_intersect

def weight_intersect(
    df_intersect: pl.DataFrame|pl.LazyFrame,
    priority: list[tuple[str,float]]=None,
    offset_prop_max: float=2.0
) -> pl.DataFrame|pl.LazyFrame:
    """
    Add a "weight" column to an intersect DataaFrame computed by summing weigted values across columns of the input
    DataFrame.

    :param df_intersect: Intersect DataFrame or LazyFrame.
    :param priority: A list of (column, weight) tuples.
    :param offset_prop_max: Maximum value for `offset_prop`.

    :return: A DataFrame or LazyFrame (same type as `df_intersect`) with a "weight" column.
    """

    # Fill priority presets
    if priority is None:
        priority = DEFAULT_PRIORITY['default']
    elif isinstance(priority, str):
        priority = DEFAULT_PRIORITY.get(priority.strip().lower(), None)

        if priority is None:
            raise ValueError(f'Unknown priority preset: "{priority}"')

    # Set lazy
    if isinstance(df_intersect, pl.DataFrame):
        df_intersect = df_intersect.lazy()
        do_collect = True
    else:
        do_collect = False

    # Compute weights
    df_intersect = df_intersect.with_columns(
        pl.lit(0.0).alias('weight').cast(pl.Float32)
    )

    for col, weight in priority:
        if col not in {'overlap_ro', 'overlap_size_ro', 'offset_prop', 'match_prop'}:
            raise ValueError(f'Priority column must be one of overlap_ro, overlap_size_ro, offset_prop, or match_prop: {col}')

        if col != 'offset_prop':
            df_intersect = df_intersect.with_columns(
                (
                    pl.col.weight +
                    pl.col(col).cast(pl.Float32) * float(weight)
                ).alias('weight')
            )

        else:
            df_intersect = df_intersect.with_columns(
                (
                    pl.col.weight +
                    (
                        (1 - pl.col(col).clip(0.0, offset_prop_max) / offset_prop_max).cast(pl.Float32)
                    ) * float(weight)
                ).alias('weight')
            )

    # Collect and return
    if do_collect:
        df_intersect = df_intersect.collect()

    return df_intersect
