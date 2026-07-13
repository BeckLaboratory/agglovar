"""Pairwise overlap runner object."""

__all__ = [
    'PairwiseOverlap',
]

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
import functools
import logging
import multiprocessing
import operator
import os
from pathlib import Path
import time
from typing import (
    Any,
    Optional,
)
from warnings import warn

import polars as pl

from ...meta.decorators import immutable
from ...meta.descriptors import (
    CheckedBool,
    CheckedObject,
    BoundedInt,
)
from ... import schema
from ...seqmatch import MatchScoreModel
from ...util.lazy import materialize_pair

from ..base import PairwiseJoin
from ..weights import (
    WeightStrategy,
    DEFAULT_WEIGHT_STRATEGY,
)

from ._const import (
    AUTOGEN_COLS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_JOIN_COLS,
    INVARIANT_JOIN_COLS,
    JOIN_COL_EXPR,
    MATCH_CHUNK_SIZE,
    RESERVED_COLS,
)
from ._stage import PairwiseOverlapStage

logger = logging.getLogger(__name__)


# Match-pool worker globals. Each spawned worker builds one scoring model via _match_worker_init
# (the model is picklable and passed once as an initializer argument), then scores chunks of pairs.
_MATCH_WORKER_MODEL: Optional[MatchScoreModel] = None


def _match_worker_init(match_score_model: MatchScoreModel) -> None:
    """Process-pool initializer: store the scoring model once per worker process."""
    global _MATCH_WORKER_MODEL
    _MATCH_WORKER_MODEL = match_score_model


def _match_worker_chunk(pairs: list[tuple[str, str]]) -> list[float]:
    """Score a chunk of ``(seq_a, seq_b)`` pairs in a worker process."""
    model = _MATCH_WORKER_MODEL
    return [model.match_prop(seq_a, seq_b) for seq_a, seq_b in pairs]


def _seg_len_exprs() -> tuple[pl.Expr, pl.Expr]:
    """Build the per-variant segment length expressions used by :meth:`PairwiseOverlap._seg_ro`.

    Expressions apply to an unsuffixed table with "seg", "qry_pos", and "qry_end" columns.

    Only aligned segments appear in "seg", so unaligned bases are the query bases that no segment
    covers. Aligned segments are measured in reference bases, matching the segment overlap they are
    compared against, which is what makes segment RO exactly 1.0 for a variant against itself.

    :returns: Tuple of ("_seg_unaligned", "_seg_total_len") expressions.
    """
    seg_ref_len = (
        pl.col('seg').list.eval(
            pl.element().struct.field('end') - pl.element().struct.field('pos')
        )
        .list.sum()
        .fill_null(0)
    )

    seg_qry_len = (
        pl.col('seg').list.eval(
            (
                pl.element().struct.field('qry_end')
                - pl.element().struct.field('qry_pos')
            ).abs()
        )
        .list.sum()
        .fill_null(0)
    )

    seg_unaligned = (
        (pl.col('qry_end') - pl.col('qry_pos')).abs() - seg_qry_len
    ).clip(0)

    return (
        seg_unaligned.alias('_seg_unaligned'),
        (seg_ref_len + seg_unaligned).alias('_seg_total_len'),
    )


@immutable
class PairwiseOverlap(PairwiseJoin):
    """Pairwise overlap class.

    Join by overlapping variants by position.

    :ivar match_score_model: (Advanced) Configured model for scoring similarity between pairs of
        sequences. If `None` and `match_prop_min` is set, then a default aligner will be used.
    :ivar force_end_ro: (Advanced) By default, reciprocal overlap is calculated with the end
        position set to the start position plus the variant length. For all variants except
        insertions, this will typically match the end value in the source DataFrame. If `True`, the
        end position in the DataFrame is also used for reciprocal overlap without changes.
        Typically, this option should not be used and will break reciprocal overlap for insertions.
    :ivar chunk_size: (Advanced) Chunk df_a into partitions of this size, and for each chunk,
        subset df_b to include only variants that may overlap with variants in the chunk. If
        None, each chromosome is a single chunk, which will lead to a combinatorial explosion
        unless offset_max is greater than 0.
    """

    # Advanced Configuration Attributes
    match_score_model: MatchScoreModel = CheckedObject(default=MatchScoreModel())
    force_end_ro: bool = CheckedBool()
    chunk_size: Optional[int] = BoundedInt(0, default=DEFAULT_CHUNK_SIZE)
    n_threads: Optional[int] = BoundedInt(1)

    # Join control
    stages: tuple[PairwiseOverlapStage, ...]
    join_cols: tuple[pl.Expr, ...]
    join_col_exprs: tuple[pl.Expr, ...]
    equi_join_exprs: tuple[pl.Expr, ...]

    # Special join columns
    compute_seg_ro: bool = CheckedBool(default=False)
    compute_weight: bool = CheckedBool(default=False)

    expected_cols: frozenset[str]

    def __init__(
            self,
            stages: Iterable[PairwiseOverlapStage],
            join_cols: Optional[Iterable[str | pl.Expr]] = None,
            drop_default_join_cols: bool = False,
            match_score_model: Optional[MatchScoreModel] = None,
            weight_strategy: WeightStrategy = DEFAULT_WEIGHT_STRATEGY,
            force_end_ro: bool = False,
            chunk_size: Optional[int] = None,
            n_threads: Optional[int] = None,
    ) -> None:
        """Create a pairwise overlap join with the given stages and parameters."""
        super().__init__(weight_strategy)

        # Set join stages
        self.stages = tuple(stages)

        if not self.stages:
            raise ValueError('No defined overlap stages')

        # Set parameters
        if match_score_model is not None:
            self.match_score_model = match_score_model

        self.force_end_ro = force_end_ro

        # Chunking control
        self.chunk_size = chunk_size
        self.n_threads = n_threads

        # Serial (pool=None) match_prop expression, kept only for schema derivation and the
        # match_prop_expr property. The expression actually executed by a join is built per-call by
        # _match_prop_expr(pool), binding scoring to a caller-provided pool; no pool is retained on
        # this (immutable) object. See make_match_pool / _match_prop_batch.
        self.expr_match_prop = self._match_prop_expr(None)

        # Set join columns
        self._set_join_cols(
            join_cols=join_cols,
            drop_default_join_cols=drop_default_join_cols,
        )

        # Set expected columns
        self.expected_cols = self._get_expected_cols()

        # Create a tuple of common equi-join expressions. Speeds up join operations.
        equi_join_list = []

        if all(stage.offset_max == 0 for stage in self.stages):
            equi_join_list.append(pl.col('pos_a') == pl.col('pos_b'))
            equi_join_list.append(pl.col('end_a') == pl.col('end_b'))

        if all(stage.match_ref for stage in self.stages):
            equi_join_list.append(pl.col('ref_a') == pl.col('ref_b'))

        if all(stage.match_alt for stage in self.stages):
            equi_join_list.append(pl.col('alt_a') == pl.col('alt_b'))

        self.equi_join_exprs = tuple(equi_join_list)

    @property
    def required_cols(self) -> set[str]:
        """The minimum set of columns that must be present in input tables.

        This is set based on parameters needed to perform the join. For example, if sequence
        matching is required, then "seq" will be in this list, and if "seq" does not exist
        in both df_a and df_b, then an error is raised.
        """
        return set(self.expected_cols - AUTOGEN_COLS - RESERVED_COLS)

    @property
    def reserved_cols(self) -> set[str]:
        """Columns reserved for internal use during the join."""
        return set(RESERVED_COLS)

    @property
    def has_match(self) -> bool:
        """True if any stage performs sequence matching."""
        return any(stage.has_match for stage in self.stages)

    @property
    def has_seg_ro(self) -> bool:
        """True if any stage computes segment reciprocal overlap."""
        return any(stage.has_seg_ro for stage in self.stages)

    @property
    def match_prop_expr(self):
        """Polars expression that computes the per-pair match proportion."""
        return self.expr_match_prop

    @property
    def has_equi_join(self) -> bool:
        """True if all stages have an equivalent equi-join expression."""
        return bool(self.equi_join_exprs)

    @property
    def is_equi_offset(self) -> bool:
        """True if all stages have an equivalent equi-join expression for position."""
        return all(stage.offset_max == 0 for stage in self.stages)

    @property
    def weight_expr(self):
        """Polars expression for the configured weight strategy."""
        return self.weight_strategy.expr

    def _match_prop_expr(self, match_pool: Optional[ProcessPoolExecutor]) -> pl.Expr:
        """Build the ``match_prop`` column expression, binding scoring to ``match_pool``.

        The expression receives the whole ``(seq_a, seq_b)`` batch per join so scoring can be
        dispatched across a process pool. The pool is captured in the callback closure (thread-safe
        regardless of which thread Polars runs the UDF on) rather than stored on this object.

        :param match_pool: Process pool to score across, or ``None`` to score serially.
        """
        if not self.has_match:
            return pl.lit(None).cast(pl.Float32)

        return (
            pl.struct(pl.col('seq_a'), pl.col('seq_b'))
            .map_batches(
                functools.partial(self._match_prop_batch, pool=match_pool),
                return_dtype=pl.Float64,
            )
            .cast(pl.Float32)
        )

    def _score_serial(self, seq_a: list[str], seq_b: list[str]) -> pl.Series:
        """Score ``match_prop`` for the given sequence pairs serially in this process."""
        model = self.match_score_model
        return pl.Series([model.match_prop(a, b) for a, b in zip(seq_a, seq_b)], dtype=pl.Float64)

    def _match_prop_batch(
            self,
            batch: pl.Series,
            pool: Optional[ProcessPoolExecutor],
    ) -> pl.Series:
        """Score ``match_prop`` for a batch of ``(seq_a, seq_b)`` struct rows.

        When ``pool`` is not ``None`` and the batch has at least ``2 * MATCH_CHUNK_SIZE`` rows, the
        pairs are scored across worker processes in fixed-size chunks; otherwise they are scored
        serially in this process. Below the threshold the pool's per-batch overhead outweighs the
        parallel gain (benchmarked), so the guard reverts to serial.

        :param batch: Struct series with ``seq_a`` and ``seq_b`` fields.
        :param pool: Process pool to score across, or ``None`` to score serially.
        """
        seq_a = batch.struct.field('seq_a').to_list()
        seq_b = batch.struct.field('seq_b').to_list()
        n = len(seq_a)

        if pool is None or n < 2 * MATCH_CHUNK_SIZE:
            return self._score_serial(seq_a, seq_b)

        pairs = list(zip(seq_a, seq_b))
        chunks = [pairs[i:i + MATCH_CHUNK_SIZE] for i in range(0, n, MATCH_CHUNK_SIZE)]

        try:
            values: list[float] = []
            for chunk_values in pool.map(_match_worker_chunk, chunks):
                values.extend(chunk_values)
        except BrokenProcessPool:
            # A worker died (e.g. OOM on a pathological sequence). Fall back to serial so one bad
            # batch does not abort the whole join. A permanently broken pool re-raises immediately on
            # each subsequent batch, so every later batch simply takes this same serial path.
            logger.warning('Match pool broke; scoring this batch serially')
            return self._score_serial(seq_a, seq_b)

        return pl.Series(values, dtype=pl.Float64)

    @contextmanager
    def make_match_pool(
            self,
            match_threads: Optional[int] = None,
    ) -> Iterator[Optional[ProcessPoolExecutor]]:
        """Create a match-scoring process pool for the duration of the block, or yield ``None``.

        The caller passes the yielded pool to :meth:`join`/:meth:`join_iter` via ``match_pool``; the
        pool is not retained on this object. It uses the ``spawn`` start method (forking deadlocks
        with Polars' thread pool) and seeds one scoring model per worker. Worker count is capped to
        the CPUs available to the process. A warm-up probe forces the workers to start inside this
        call, so a spawn-restricted or resource-limited environment is detected here and degraded to
        serial scoring (with a logged warning) instead of failing mid-join.

        :param match_threads: Requested worker count, or ``None`` to score serially.

        :raises ValueError: If ``match_threads`` is not ``None`` and is less than 1.

        :yields: A live process pool, or ``None`` when scoring serially.
        """
        if match_threads is not None and match_threads < 1:
            raise ValueError(f'match_threads must be None or a positive integer: {match_threads}')

        if match_threads is None or not self.has_match:
            yield None
            return

        try:
            available = len(os.sched_getaffinity(0))
        except AttributeError:  # sched_getaffinity is Linux-only
            available = os.cpu_count() or 1

        workers = max(1, min(match_threads, available))

        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_match_worker_init,
            initargs=(self.match_score_model,),
        )

        try:
            # Warm-up probe: force the workers to spawn now so a restricted environment (no spawn,
            # PID/semaphore limits, blocked /dev/shm) fails here rather than deep inside a join.
            list(pool.map(_match_worker_chunk, [[('A', 'A')]] * workers))
        except (OSError, BrokenProcessPool) as e:
            logger.warning('Match pool unavailable (%s); scoring match_prop serially', e)
            pool.shutdown()
            yield None
            return

        try:
            yield pool
        finally:
            pool.shutdown()

    def join_iter(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame,
            retain_index: bool = False,
            temp_dir: bool | str | Path = False,
            match_pool: Optional[ProcessPoolExecutor] = None,
    ) -> Iterator[pl.LazyFrame]:
        """Find all pairs of variants in two sources that meet a set of criteria.

        :param df_a: Source dataframe.
        :param df_b: Target dataframe.
        :param retain_index: If True, do not drop an existing "_index" column in callset tables
            if they exist.
        :param temp_dir: How to materialise the prepared tables before the chunked loop.
            ``False`` (default) collects both into memory; ``True`` writes them to the
            system temp directory as parquet files; a ``str``/``Path`` writes them to
            that directory. Temp files are always removed on exit.
        :param match_pool: Process pool for parallel ``match_prop`` scoring (from
            :meth:`make_match_pool`), or ``None`` to score serially.

        :yields: A LazyFrame for each chunk.
        """
        # Prepare tables
        df_a, df_b = self._prepare_tables(df_a, df_b, warn_on_reserved=True, retain_index=retain_index)

        if self.chunk_size == 0 or (self.is_equi_offset and self.chunk_size is None):
            return self._join_iter_notchunked(df_a, df_b, match_pool=match_pool)

        # Materialise both tables once before the chunked loop. with_row_index in _prepare_tables
        # blocks predicate pushdown into the parquet source, so every downstream collect would
        # otherwise trigger a full-dataset scan.
        return self._join_iter_chunked_materialised(df_a, df_b, temp_dir, match_pool=match_pool)

    def _join_iter_chunked(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame,
            match_pool: Optional[ProcessPoolExecutor] = None,
    ) -> Iterator[pl.LazyFrame]:
        """Find all pairs of variants in two sources that meet a set of criteria.

        :param df_a: Source dataframe after `_prepare_tables()`.
        :param df_b: Target dataframe after `_prepare_tables()`.
        :param match_pool: Process pool for parallel ``match_prop`` scoring, or ``None`` for serial.

        :yields: A LazyFrame for each chunk.
        """
        logger.debug('Starting join (chunked)...')

        # Per-chunk timing/size logging is gated on DEBUG so an INFO-level run pays nothing: the
        # per-stage df_b row counts and the position-range are only computed when the log record
        # would actually be emitted.
        debug = logger.isEnabledFor(logging.DEBUG)

        join_empty = True

        # Per-stage chunk-range dicts are static (derived from each stage's own two-sided bounds),
        # so build them once. Each stage filters df_b with its own bounds inside _join_pairwise.
        stage_ranges = [self._stage_chunk_range_dict(stage) for stage in self.stages]
        chunk_size = self.chunk_size if self.chunk_size is not None else DEFAULT_CHUNK_SIZE

        for chrom, last_index_a in (
            df_a
            .group_by('chrom_a')
            .agg(pl.len().alias('last_index'))
            .sort('chrom_a')
        ).collect().rows():
            start_index_a = 0

            df_a_chrom = (
                df_a.filter(pl.col('chrom_a') == chrom)
                .with_row_index('_index_chrom_a')
                .collect()
                .lazy()
            )

            # Collect this chromosome's df_b once into memory so the per-stage relative filters are
            # lazy views over an in-memory frame rather than re-scanning the (possibly temp-parquet)
            # df_b once per stage per chunk. Symmetric with the df_a_chrom collect above.
            df_b_chrom = (
                df_b.filter(pl.col('chrom_b') == chrom)
                .collect()
                .lazy()
            )

            chrom_t0 = time.perf_counter()
            chrom_chunks = 0
            chrom_pairs = 0
            chrom_join_s = 0.0
            chrom_b_max = 0

            while start_index_a < last_index_a:
                end_index_a = start_index_a + chunk_size

                df_a_chunk_df = df_a_chrom.filter(
                    pl.col('_index_chrom_a') >= start_index_a,
                    pl.col('_index_chrom_a') < end_index_a,
                ).collect()
                df_a_chunk = df_a_chunk_df.lazy()

                diag = {'n_b': []} if debug else None

                if debug:
                    t0 = time.perf_counter()
                    df_join = self._join_pairwise(
                        df_a_chunk, df_b_chrom, match_pool=match_pool,
                        stage_ranges=stage_ranges, diag=diag,
                    ).collect()
                    dt = time.perf_counter() - t0

                    # n_b is now per stage: with two-sided bounds the offset stage collapses to a
                    # small local window instead of scaling with the chunk's max position. pos_a is
                    # the chunk's position span (relevant to whether chunks are position-local).
                    logger.debug(
                        'Join chunk: chrom %s [%d:%d] pos_a=%s-%s n_a=%d n_b=%s pairs=%d in %.3fs'
                        % (chrom, start_index_a, end_index_a,
                           df_a_chunk_df['pos_a'].min(), df_a_chunk_df['pos_a'].max(),
                           df_a_chunk_df.height, diag['n_b'], df_join.height, dt)
                    )

                    chrom_chunks += 1
                    chrom_pairs += df_join.height
                    chrom_join_s += dt
                    chunk_b_max = max(diag['n_b'], default=0)
                    if chunk_b_max > chrom_b_max:
                        chrom_b_max = chunk_b_max

                    yield df_join.lazy()
                else:
                    yield (
                        self._join_pairwise(
                            df_a_chunk, df_b_chrom, match_pool=match_pool, stage_ranges=stage_ranges,
                        )
                        .collect()
                        .lazy()
                    )

                join_empty = False
                start_index_a = end_index_a

            if debug:
                logger.debug(
                    'Join chrom done: chrom %s n_a=%d chunks=%d n_b_max=%d pairs=%d '
                    'join=%.3fs wall=%.3fs'
                    % (chrom, last_index_a, chrom_chunks, chrom_b_max, chrom_pairs,
                       chrom_join_s, time.perf_counter() - chrom_t0)
                )

        if join_empty:
            # Yield an empty frame with the correct schema so pl.concat never sees an empty list.
            # Collect so the yielded LazyFrame does not reference df_a/df_b — they may be
            # scan_parquet nodes over temp files that _join_iter_chunked_disk will unlink
            # once this generator exits. stage_ranges is left as None (no relative filter needed
            # for an empty frame).
            yield self._join_pairwise(
                df_a.head(0), df_b.head(0), match_pool=match_pool,
            ).collect().lazy()

    def _join_iter_chunked_materialised(
            self,
            df_a: pl.LazyFrame,
            df_b: pl.LazyFrame,
            temp_dir: bool | str | Path,
            match_pool: Optional[ProcessPoolExecutor] = None,
    ) -> Iterator[pl.LazyFrame]:
        """Materialise the prepared tables once and run the chunked join.

        :param df_a: Source dataframe after `_prepare_tables()`.
        :param df_b: Target dataframe after `_prepare_tables()`.
        :param temp_dir: Materialisation policy. See :meth:`join_iter`.
        :param match_pool: Process pool for parallel ``match_prop`` scoring, or ``None`` for serial.

        :yields: A LazyFrame for each chunk.
        """
        with materialize_pair(
            df_a, df_b, temp_dir, prefix='pairwise_overlap_prep_',
        ) as (df_a_mat, df_b_mat):
            yield from self._join_iter_chunked(df_a_mat, df_b_mat, match_pool=match_pool)

    def _join_iter_notchunked(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame,
            match_pool: Optional[ProcessPoolExecutor] = None,
    ) -> Iterator[pl.LazyFrame]:
        """Find all pairs of variants in two sources that meet a set of criteria.

        :param df_a: Source dataframe after `_prepare_tables()`.
        :param df_b: Target dataframe after `_prepare_tables()`.
        :param match_pool: Process pool for parallel ``match_prop`` scoring, or ``None`` for serial.

        :yields: A LazyFrame for each chunk.
        """
        logger.debug('Starting join (not chunked)...')

        join_empty = True  # Detects if no joins were written

        chrom_list = sorted(
            set(df_a.select('chrom_a').unique().collect().to_series().to_list())
            | set(df_b.select('chrom_b').unique().collect().to_series().to_list())
        )

        for chrom in chrom_list:
            yield self._join_pairwise(
                df_a.filter(pl.col('chrom_a') == chrom),
                df_b.filter(pl.col('chrom_b') == chrom),
                match_pool=match_pool,
            )

            join_empty = False

        if join_empty:
            # If no join tables were yielded, yield an empty one. This creates an empty join table
            # with the correct structure and prevents pl.concat from failing on an empty list.
            yield self._join_pairwise(df_a.head(0), df_b.head(0), match_pool=match_pool)

    def _join_pairwise(
            self,
            df_a: pl.LazyFrame,
            df_b: pl.LazyFrame,
            match_pool: Optional[ProcessPoolExecutor] = None,
            stage_ranges: Optional[list[dict[tuple[str, str], list[pl.Expr]]]] = None,
            diag: Optional[dict] = None,
    ) -> pl.LazyFrame:
        """Non-equi-join (pos and end not the same).

        Assumes both tables are filtered to the same chromosome.

        :param df_a: Source chunk table.
        :param df_b: Target table (chromosome-sliced when ``stage_ranges`` is given).
        :param match_pool: Process pool for parallel ``match_prop`` scoring, or ``None`` for serial.
            The ``match_prop`` column is built per call with the pool bound into its callback.
        :param stage_ranges: Per-stage chunk-range dicts, one per stage in ``self.stages``. When
            given, each stage filters ``df_b`` with its own two-sided bounds (see
            :meth:`_chunk_relative`) before the join. When ``None``, every stage joins ``df_b``
            unchanged — the behavior used by the non-chunked path and the empty-schema guard.
        :param diag: Optional DEBUG collector. If not ``None``, ``diag['n_b']`` is appended with the
            per-stage filtered df_b row count.
        """
        # match_prop is built per call so scoring binds to the caller's pool; join_col_exprs holds
        # the remaining (pool-independent) columns. seq_a/seq_b are still present here, before the
        # select below narrows the frame.
        select_exprs = list(self.join_col_exprs)
        if self._emit_match_prop:
            select_exprs.append(self._match_prop_expr(match_pool).alias('match_prop'))

        join_list = []

        for stage_idx, stage in enumerate(self.stages):
            if stage_ranges is not None:
                df_b_stage = self._chunk_relative(df_a, df_b, stage_ranges[stage_idx])
            else:
                df_b_stage = df_b

            if diag is not None:
                diag['n_b'].append(df_b_stage.select(pl.len()).collect().item())

            all_predicates = (*self.equi_join_exprs, *stage.join_predicates)

            if all_predicates:
                df_join = df_a.join_where(df_b_stage, *all_predicates)
            else:
                df_join = df_a.join(df_b_stage, how='cross')

            df_join = df_join.select(
                *select_exprs,
            )

            if self.compute_weight:
                df_join = df_join.with_columns(
                    self.weight_expr.alias('weight'),
                )

            if self.compute_seg_ro:
                df_join = self._seg_ro(df_join, df_a, df_b_stage)

            df_join = (
                df_join
                .filter(*stage.join_filters)
                .select(*self.join_cols)
                # .sort(['index_a', 'index_b'])
            )

            join_list.append(df_join)

        return (
            pl.concat(join_list)
            .group_by('index_a', 'index_b')
            .agg(
                pl.all().get(pl.col('weight').fill_null(0.0).arg_max())
            )
            # .sort('index_a', 'index_b')
        )

    @staticmethod
    def _stage_chunk_range_dict(
            stage: PairwiseOverlapStage,
    ) -> dict[tuple[str, str], list[pl.Expr]]:
        """Flatten one stage's ``chunk_range`` into the dict form ``_chunk_relative`` expects.

        Each stage carries its own two-sided position/length bounds as ``(col, limit, exprs)``
        tuples. These are collected here per stage — with no cross-stage intersection — so that
        every stage chunks df_b with exactly its own necessary-condition bounds. A stage with an
        empty ``chunk_range`` yields an empty dict, which leaves df_b unfiltered for that stage
        (an unconstrained stage must not be pruned).

        :param stage: Stage whose chunk range is flattened.

        :returns: Mapping of ``(col, limit)`` to the list of df_a expressions bounding that limit.
        """
        chunk_range: dict[tuple[str, str], list[pl.Expr]] = {}

        for col, limit, exprs in stage.chunk_range:
            chunk_range.setdefault((col, limit), []).extend(exprs)

        return chunk_range

    def _prepare_tables(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame,
            retain_index: bool = False,
            warn_on_reserved: bool = False,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """Prepare tables for join.

        Checks for expected columns and formats, adds missing columns as needed, and
        appends "_a" and "_b" suffixes to column names.

        :param df_a: Table A.
        :param df_b: Table B.
        :param retain_index: If True, do not drop an existing "_index" column in callset tables
            if they exist.
        :param warn_on_reserved: If True, generate a warning if reserved columns are found and
            drop them. If false, raise an error.

        :returns: Tuple of normalized tables (df_a, df_b).

        :raises ValueError: If missing or malform columns are detected.
        :raises TypeError: If input is not a DataFrame or LazyFrame.
        """
        # Check input types
        if isinstance(df_a, pl.DataFrame):
            df_a = df_a.lazy()
        elif not isinstance(df_a, pl.LazyFrame):
            raise TypeError(f'Variant source: Expected DataFrame or LazyFrame, got {type(df_a)}')

        if isinstance(df_b, pl.DataFrame):
            df_b = df_b.lazy()
        elif not isinstance(df_b, pl.LazyFrame):
            raise TypeError(f'Variant target: Expected DataFrame or LazyFrame, got {type(df_b)}')

        # Check for expected columns
        columns_a = set(df_a.collect_schema().names())
        columns_b = set(df_b.collect_schema().names())

        missing_cols_a = sorted(self.check_required_cols(columns_a))
        missing_cols_b = sorted(self.check_required_cols(columns_b))

        if missing_cols_a or missing_cols_b:
            if missing_cols_a == missing_cols_b:
                raise ValueError(f'DataFrames "A" and "B" are missing expected column(s): {", ".join(missing_cols_a)}')

            raise ValueError(
                f'DataFrame "A" missing expected column(s): {", ".join(missing_cols_a)}; '
                f'DataFrame "B" missing expected column(s): {", ".join(missing_cols_b)}'
            )

        # Drop reserved columns
        reserved_col_a = self.check_reserved_cols(columns_a)
        reserved_col_b = self.check_reserved_cols(columns_b)

        retain_index_a = '_index' in reserved_col_a and retain_index
        retain_index_b = '_index' in reserved_col_b and retain_index

        if retain_index:
            reserved_col_a -= {'_index', }
            reserved_col_b -= {'_index', }

        if reserved_col_a:
            reserved_col_a = sorted(reserved_col_a)
            err_str = f'Reserved columns in table "A": {", ".join(reserved_col_a)}'

            if warn_on_reserved:
                warn(f'{err_str}: Dropping column(s)')
            else:
                raise ValueError(err_str)

            df_a = df_a.drop(reserved_col_a, strict=False)

        if reserved_col_b:
            reserved_col_b = sorted(reserved_col_b)
            err_str = f'Reserved columns in table "B": {", ".join(reserved_col_b)}'

            if warn_on_reserved:
                warn(f'{err_str}: Dropping column(s)')
            else:
                raise ValueError(err_str)

            df_b = df_b.drop(reserved_col_b, strict=False)

        # Cast columns
        try:
            df_a = df_a.cast({col: schema.VARIANT[col] for col in columns_a if col in schema.VARIANT.keys()})
        except pl.exceptions.InvalidOperationError as e:
            raise ValueError(f'Unexpected columns types encountered in DataFrame "A": {e}') from e

        try:
            df_b = df_b.cast({col: schema.VARIANT[col] for col in columns_b if col in schema.VARIANT.keys()})
        except pl.exceptions.InvalidOperationError as e:
            raise ValueError(f'Unexpected columns types encountered in DataFrame "B": {e}') from e

        # Set and prepare varlen
        if 'varlen' not in columns_a:
            df_a = (
                df_a
                .with_columns(
                    (pl.col('end') - pl.col('pos'))
                    .cast(schema.VARIANT['varlen'])
                    .alias('varlen')
                )
            )

        if 'varlen' not in columns_b:
            df_b = (
                df_b
                .with_columns(
                    (pl.col('end') - pl.col('pos'))
                    .cast(schema.VARIANT['varlen'])
                    .alias('varlen')
                )
            )

        # Ensure positive values
        df_a = df_a.with_columns(
            pl.col('varlen')
            .cast(schema.VARIANT['varlen'])
            .abs()
        )

        df_b = df_b.with_columns(
            pl.col('varlen')
            .cast(schema.VARIANT['varlen'])
            .abs()
        )

        # Set index
        if not retain_index_a:
            df_a = (
                df_a
                .drop('_index', strict=False)
                .with_row_index('_index')
            )

        if not retain_index_b:
            df_b = (
                df_b
                .drop('_index', strict=False)
                .with_row_index('_index')
            )

        # Set ID
        if 'id' not in columns_a:
            df_a = df_a.with_columns(
                pl.lit(None).alias('id').cast(schema.VARIANT['id'])
            )

        if 'id' not in columns_b:
            df_b = df_b.with_columns(
                pl.lit(None).alias('id').cast(schema.VARIANT['id'])
            )

        # Prepare REF & ALT
        if 'ref' in self.required_cols:
            df_a = df_a.with_columns(
                pl.col('ref').str.to_uppercase().str.strip_chars()
            )

            df_b = df_b.with_columns(
                pl.col('ref').str.to_uppercase().str.strip_chars()
            )

        if 'alt' in self.required_cols:
            df_a = df_a.with_columns(
                pl.col('alt').str.to_uppercase().str.strip_chars()
            )

            df_b = df_b.with_columns(
                pl.col('alt').str.to_uppercase().str.strip_chars()
            )

        # Get END for RO
        if not self.force_end_ro:
            df_a = df_a.with_columns(
                (pl.col('pos') + pl.col('varlen')).alias('_end_ro')
            )

            df_b = df_b.with_columns(
                (pl.col('pos') + pl.col('varlen')).alias('_end_ro')
            )

        else:
            df_a = df_a.with_columns(
                pl.col('end').alias('_end_ro')
            )

            df_b = df_b.with_columns(
                pl.col('end').alias('_end_ro')
            )

        # Per-variant segment lengths for segment RO. These are constant per variant, so they are
        # computed once here rather than once per chunk inside _seg_ro.
        if self.has_seg_ro:
            df_a = df_a.with_columns(*_seg_len_exprs())
            df_b = df_b.with_columns(*_seg_len_exprs())

        # Append suffixes to all columns
        df_a = df_a.select(pl.all().name.suffix('_a'))
        df_b = df_b.select(pl.all().name.suffix('_b'))

        return df_a, df_b

    def _set_join_cols(
            self,
            join_cols: Optional[Iterable[str | pl.Expr]] = None,
            drop_default_join_cols: bool = False,
    ) -> None:
        """Set join columns.

        :param join_cols: Join columns to include (expressions or names).
        :param drop_default_join_cols: Drop default join columns if True.
        """
        join_expr_map: dict[str, Optional[pl.Expr]] = {}

        for col in (
                INVARIANT_JOIN_COLS
                + (DEFAULT_JOIN_COLS if not drop_default_join_cols else [])
                + (list(join_cols) if join_cols else [])
        ):
            self._append_join_cols(col, join_expr_map)

        join_col_list = list(
            pl.col(col) for col in join_expr_map.keys()
        )

        for stage in self.stages:  # Add join columns used by stages
            for join_filter in stage.join_filters:
                for col in join_filter.meta.root_names():
                    self._append_join_cols(col, join_expr_map, True)

        join_col_set = functools.reduce(
            operator.or_,
            (
                set(expr.meta.root_names())
                for stage in self.stages
                for expr in stage.join_filters
            ),
            set()
        )

        if 'seg_ro' in join_col_set and join_expr_map['seg_ro'] is None:
            self.compute_seg_ro = True

        if 'weight' in join_expr_map.keys() and join_expr_map['weight'] is None:
            self.compute_weight = True

        self._append_join_cols(join_col_set, join_expr_map, True)

        self.join_cols = tuple(join_col_list)
        self.join_col_exprs = tuple(expr for expr in join_expr_map.values() if expr is not None)

        # match_prop is emitted as an output column but built per join (pool-bound) in
        # _join_pairwise rather than precomputed into join_col_exprs above.
        self._emit_match_prop = 'match_prop' in join_expr_map

    def _append_join_cols(
            self,
            exprs: Iterable[pl.Expr | str] | pl.Expr | str,
            join_expr_map: MutableMapping[str, Optional[pl.Expr]],
            no_replace: bool = False,
    ) -> None:
        """Append expressions to the list of columns included in the join table.

        These columns will be appended to the standard join table columns. Each expression
        should name the column it creates using ".alias()" if necessary. These columns do
        not affect the join itself, just the columns that appear in the join table.

        For example, to retain the "pos" column from df_a and df_b, then append
        "pl.col('pos_a')" and "pl.col('pos_b')". If you wanted to set a flag for whether
        the variant in df_a comes before df_b, then a new columns could be added:
        "(pl.col('pos_a') <= pl.col('pos_b')).alias('left_a')"

        :param exprs: A join column or a list of join columns where each may be defined as a string
            (known column name or a field already in the join table) or a Polars expression.
        :param join_expr_map: A dictionary to which the expressions will be added.
        :param no_replace: Do not replace existing join column expressions if True.
        """
        if isinstance(exprs, (pl.Expr, str)):
            exprs = [exprs]

        for expr in exprs:
            if isinstance(expr, str):
                col_name = expr

                if col_name in JOIN_COL_EXPR:
                    col_expr = JOIN_COL_EXPR[col_name].alias(col_name)

                elif col_name == 'match_prop':
                    col_expr = None  # Built per join (pool-bound) in _join_pairwise, not precomputed

                elif col_name == 'weight':
                    col_expr = None  # Set later

                elif col_name == 'seg_ro':
                    col_expr = None  # Ignore, cannot be set as a column expression, handled separately

                else:
                    raise ValueError(
                        f'Join column "{col_name} is not a known column name in '
                        f'"{", ".join(JOIN_COL_EXPR.keys())}": '
                        f'Custom columns must be Polars expressions over columns ending with "_a" and "_b" '
                        f'(for source table) and an "alias" to the desired column name.'
                    )
            else:
                col_name = expr.meta.output_name()
                col_expr = expr

            if no_replace and col_name in join_expr_map:
                continue

            join_expr_map[col_name] = col_expr

    def _get_expected_cols(self) -> frozenset[str]:
        """Extract expected columns from all stages and join columns.

        Call after join columns are set.

        :return: A set of join column names.
        """
        col_set: set[str] = {'chrom', 'pos', 'end'}

        stage_i = 0

        for stage in self.stages:
            stage_i += 1

            expr_i = 0
            for expr in stage.join_predicates:
                expr_i += 1
                for col in expr.meta.root_names():
                    col_set.add(_check_expected_col(col, 'join_predicates', stage_i, expr_i))

            expr_i = 0
            for col, _bound, _exprs in stage.chunk_range:
                expr_i += 1
                col_set.add(_check_expected_col(col, 'chunk_range', stage_i, expr_i))

        if self.has_seg_ro:
            col_set.add(_check_expected_col('seg_a', 'has_seg_ro', None, None))
            col_set.add(_check_expected_col('seg_b', 'has_seg_ro', None, None))
            col_set.add(_check_expected_col('qry_end_a', 'has_seg_ro', None, None))
            col_set.add(_check_expected_col('qry_end_b', 'has_seg_ro', None, None))
            col_set.add(_check_expected_col('qry_pos_a', 'has_seg_ro', None, None))
            col_set.add(_check_expected_col('qry_pos_b', 'has_seg_ro', None, None))

        for col in self.expr_match_prop.meta.root_names():
            col_set.add(_check_expected_col(col, 'expr_match_prop', None, None))

        expr_i = 0
        for expr in self.join_col_exprs:
            expr_i += 1

            for col in expr.meta.root_names():
                col_set.add(_check_expected_col(col, 'join_cols', None, expr_i))

        return frozenset(col_set - RESERVED_COLS)

    def _chunk_relative(
            self,
            df_a: pl.LazyFrame,
            df_b: pl.LazyFrame,
            chunk_range: dict[tuple[str, str], list[pl.Expr]],
    ) -> pl.LazyFrame:
        """Chunk one DataFrame relative to another.

        Chunk df_b relative to df_a choosing records in df_b that could possibly be joined with some record in df_a.
        For example, this function may determine the minimum and maximum values of pos and end, and then subset df_b
        by those values. The actual subset values are determined by the `chunk_range` argument.

        `chunk_range` is a dictionary with keys formatted as ('column', 'limit') where "column" is a column name and
        "limit" is "min" or "max". Each value is a list of expressions to be applied to df_a, which will then determine
        the minimum or maximum value to be applied.

        For example, if ('pos', 'min') is a key in chunk_range, then chunk_range['pos', 'min'] is a list of expressions.
        In this example, assume it is a list with the single expression "pl.col('pos_a') - pl.col('varlen_a')"). For
        each record in df_a, the expression will compute the position minus the variant length producing a single value
        for each record. Since this is a minimum value, the minimum of these values (one per record in df_a) will
        be used to filter records in df_b by excluding any records with "pos_b" less than this minimum.

        The flexibility of this function is needed to support different limits. For example, when reciprocal overlap is
        used as a limit, the maximum value of pos_b is based on the maximum value of end_a (i.e. "chunk_range['pos',
        'max']" will contain "pl.col('end_a')" because if pos_b greater than any "end_a", then variants cannot overlap.

        `df_b` is expected to be already sliced to the relevant chromosome, so no chromosome filter is applied here.
        With an empty `chunk_range` (an unconstrained stage), df_b is returned unfiltered.

        :param df_a: Table chunk.
        :param df_b: Table to be chunked to records that may overlap with df_a.
        :param chunk_range: Per-stage bounds (see :meth:`_stage_chunk_range_dict`).

        :returns: df_b partitioned (LazyFrame).
        """
        filter_list: list[pl.Expr] = []

        for (col_name, limit), expr_list in chunk_range.items():
            if limit == 'min':
                scalar = (
                    df_a
                    .select(pl.min_horizontal(*expr_list))
                    .collect()
                    .to_series()
                    .min()
                )

                if scalar is not None:
                    filter_list.append(pl.col(col_name) >= scalar)

            elif limit == 'max':
                scalar = (
                    df_a
                    .select(pl.max_horizontal(*expr_list))
                    .collect()
                    .to_series()
                    .max()
                )

                if scalar is not None:
                    filter_list.append(pl.col(col_name) <= scalar)

            else:
                raise ValueError(f'Unknown limit: "{limit}"')

        if not filter_list:
            return df_b

        return df_b.filter(*filter_list)

    @staticmethod
    def _seg_events(
            df_pairs: pl.LazyFrame,
            df: pl.LazyFrame,
            side: str,
    ) -> pl.LazyFrame:
        """Expand one side's segments into depth events, two rows per segment.

        Each segment becomes a ``+1`` event at its start and a ``-1`` event at its end on its own
        side's depth counter, leaving the other side's counter untouched. Variants with no segments
        contribute no events (they are all unaligned bases).

        :param df_pairs: Candidate pairs with a ``_pair_id`` row index.
        :param df: Prepared (suffixed) table for this side.
        :param side: Either "a" or "b".

        :returns: Depth events for this side.
        """
        df_seg = (
            df_pairs
            .join(
                (
                    df
                    .select(f'_index_{side}', f'seg_{side}')
                    .explode(f'seg_{side}', empty_as_null=False)
                    .select(
                        f'_index_{side}',
                        pl.col(f'seg_{side}').struct.field('chrom').alias('_seg_chrom'),
                        pl.col(f'seg_{side}').struct.field('is_rev').alias('_seg_rev'),
                        pl.col(f'seg_{side}').struct.field('pos').alias('_seg_pos'),
                        pl.col(f'seg_{side}').struct.field('end').alias('_seg_end'),
                    )
                    # A null segment list explodes to a single null row (an empty one drops out).
                    .filter(pl.col('_seg_pos').is_not_null())
                ),
                left_on=f'index_{side}', right_on=f'_index_{side}', how='inner',
            )
        )

        depth_a, depth_b = (1, 0) if side == 'a' else (0, 1)

        return pl.concat([
            df_seg.select(
                '_pair_id', '_seg_chrom', '_seg_rev',
                pl.col('_seg_pos').alias('_seg_x'),
                pl.lit(depth_a, pl.Int32).alias('_seg_depth_a'),
                pl.lit(depth_b, pl.Int32).alias('_seg_depth_b'),
            ),
            df_seg.select(
                '_pair_id', '_seg_chrom', '_seg_rev',
                pl.col('_seg_end').alias('_seg_x'),
                pl.lit(-depth_a, pl.Int32).alias('_seg_depth_a'),
                pl.lit(-depth_b, pl.Int32).alias('_seg_depth_b'),
            ),
        ])

    def _seg_ro(
            self,
            df_join: pl.LazyFrame,
            df_a: pl.LazyFrame,
            df_b: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Compute segment reciprocal overlap (RO) for complex variants.

        Complex variants are composed of segments, some aligned to the reference and some not.
        Overlap is the size of the *multiset* intersection of the two variants' aligned segments:
        decompose both variants' segments into atomic ranges at every segment boundary, and sum
        ``width * min(depth_a, depth_b)`` over them, where depth is the number of segments covering
        a range. Taking the minimum depth is what stops a segment from being counted more than once
        when it overlaps several segments on the other side. Depth is tracked per chromosome and
        orientation, so segments never overlap across either.

        Unaligned bases (query bases no segment covers) have no reference locus to intersect, so
        ``min(unaligned_a, unaligned_b)`` of them are credited as overlapping. The RO is then::

            (overlap + min(unaligned_a, unaligned_b)) / max(total_len_a, total_len_b)

        This is symmetric, and it is 1.0 exactly when a variant is compared against itself.

        Rather than the O(seg_a * seg_b) cross product of segment pairs, the overlap is computed by
        a sweep costing O(seg_a + seg_b): each segment becomes a ``+1``/``-1`` event pair, and depth
        is the running sum of those events in coordinate order. Sorting the events puts each
        ``(pair, chrom, is_rev)`` group in one contiguous block whose events sum to zero, so a plain
        ``cum_sum`` (no window) yields the correct depth within every group and returns to zero at
        each group boundary. Widths follow the same way: the final row of a group has zero depth, so
        the meaningless width spanning into the next group is multiplied by zero. Coincident
        boundaries need no special handling either -- they simply span zero width.

        :param df_join: Join table without segment RO.
        :param df_a: Table A.
        :param df_b: Table B.

        :return: Join table with a "seg_ro" column.
        """
        # Materialize the join once. The pairs feed both sides' events and the table they are
        # joined back onto, so leaving it lazy re-runs the join (and any match scoring in it) for
        # each of those branches.
        df_join = df_join.collect().lazy()

        df_pairs = df_join.select('index_a', 'index_b').with_row_index('_pair_id')

        df_overlap = (
            pl.concat([
                self._seg_events(df_pairs, df_a, 'a'),
                self._seg_events(df_pairs, df_b, 'b'),
            ])
            .sort('_pair_id', '_seg_chrom', '_seg_rev', '_seg_x')
            .select(
                '_pair_id',
                (pl.col('_seg_x').shift(-1) - pl.col('_seg_x')).alias('_seg_width'),
                pl.min_horizontal(
                    pl.col('_seg_depth_a').cum_sum(),
                    pl.col('_seg_depth_b').cum_sum(),
                )
                .clip(0)
                .alias('_seg_depth'),
            )
            .group_by('_pair_id')
            .agg(
                (pl.col('_seg_width').fill_null(0) * pl.col('_seg_depth'))
                .sum()
                .alias('_seg_overlap')
            )
        )

        seg_ro_num = (
            pl.col('_seg_overlap').fill_null(0)
            + pl.min_horizontal('_seg_unaligned_a', '_seg_unaligned_b')
        )

        seg_ro_den = pl.max_horizontal('_seg_total_len_a', '_seg_total_len_b')

        df_seg_ro = (
            df_pairs
            .join(df_overlap, on='_pair_id', how='left')
            .join(
                df_a.select('_index_a', '_seg_unaligned_a', '_seg_total_len_a'),
                left_on='index_a', right_on='_index_a', how='left',
            )
            .join(
                df_b.select('_index_b', '_seg_unaligned_b', '_seg_total_len_b'),
                left_on='index_b', right_on='_index_b', how='left',
            )
            .select(
                'index_a',
                'index_b',
                (
                    pl.when(seg_ro_den > 0)
                    .then(seg_ro_num / seg_ro_den)
                    .otherwise(1.0)  # Two empty variants: nothing to disagree about
                )
                .clip(0.0, 1.0)  # Guard float error only, the ratio cannot exceed 1.0
                .cast(pl.Float32)
                .alias('seg_ro')
            )
        )

        return (
            df_join
            .join(df_seg_ro, on=['index_a', 'index_b'], how='left')
            .with_columns(
                pl.col('seg_ro').fill_null(1.0)
            )
        )

    def _seg_ro_old(
            self,
            df_join: pl.LazyFrame,
            df_a: pl.LazyFrame,
            df_b: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Compute segment RO.

        :param df_join: Join table without segment RO.
        :param df_a: Table A.
        :param df_b: Table B.

        :return: Join table with segment RO.
        """
        df_seg_ro = (
            df_join
            .select('index_a', 'index_b')
            .join(
                (
                    df_a
                    .with_columns(
                        # Query bases aligned in segments
                        pl.col('seg_a').list.eval(
                            (
                                pl.element().struct.field('qry_end')
                                - pl.element().struct.field('qry_pos')
                            ).abs()
                        )
                        .list.sum()
                        .alias('seg_qry_len_a')
                    )
                    .explode('seg_a')
                    .with_row_index('_seg_a_index')
                ),
                left_on='index_a', right_on='_index_a', how='inner'
            )
            .join(
                (
                    df_b
                    .with_columns(
                        # Query bases aligned in segments
                        pl.col('seg_b').list.eval(
                            (
                                pl.element().struct.field('qry_end')
                                - pl.element().struct.field('qry_pos')
                            ).abs()
                        )
                        .list.sum()
                        .alias('seg_qry_len_b')
                    )
                    .explode('seg_b')
                    .with_row_index('_seg_b_index')
                ),
                left_on='index_b', right_on='_index_b', how='inner'
            )
            .with_columns(
                (
                    # Overlapping bases (reference)
                    (
                        pl.min_horizontal(
                            pl.col('seg_a').struct.field('end'), pl.col('seg_b').struct.field('end')
                        )
                        - pl.max_horizontal(
                            pl.col('seg_a').struct.field('pos'), pl.col('seg_b').struct.field('pos')
                        )
                    )
                    / (  # Reference bp to query bp
                        pl.col('seg_a').struct.field('end')
                        - pl.col('seg_a').struct.field('pos')
                    ).abs()
                    * (
                        pl.col('seg_a').struct.field('qry_end')
                        - pl.col('seg_a').struct.field('qry_pos')
                    ).abs()
                )
                .fill_null(0.0)
                .clip(0.0)
                .cast(pl.Float32)
                .alias('seg_ro_len')
            )
            .group_by('_seg_a_index', '_seg_b_index')
            .agg(
                # Resolve cross-join among segments, choose the best pairs
                pl.all().get(pl.col('seg_ro_len').arg_max()),
            )
            .group_by('index_a', 'index_b')
            .agg(
                (
                    # Compute segment RO...
                    (
                        # Sum of overlapping segments in query bp (estimated, above)
                        pl.col('seg_ro_len').sum()

                        # Unaligned segment bp
                        + pl.min_horizontal(
                            (pl.col('qry_end_a').first() - pl.col('qry_pos_a').first()).abs()
                            - pl.col('seg_qry_len_a').first(),
                            (pl.col('qry_end_b').first() - pl.col('qry_pos_b').first()).abs()
                            - pl.col('seg_qry_len_b').first()
                        )
                    )

                    # Divide by max length for RO
                    / pl.max_horizontal(
                        (pl.col('qry_end_a').first() - pl.col('qry_pos_a').first()).abs(),
                        (pl.col('qry_end_b').first() - pl.col('qry_pos_b').first()).abs(),
                    )
                )
                .clip(0.0, 1.0)
                .cast(pl.Float32)
                .alias('seg_ro')
            )
        )

        return (
            df_join
            .join(
                df_seg_ro,
                on=['index_a', 'index_b'],
                how='left'
            )
            .with_columns(
                pl.col('seg_ro').fill_null(1.0)
            )
        )

    @classmethod
    def from_definition(
            cls,
            overlap_def: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    ):
        """Build a :class:`PairwiseOverlap` instance from a stage definition mapping or sequence of mappings."""
        keys: Optional[set[str]] = None

        # Try keys
        try:
            keys = set(overlap_def.keys())

            if 'stages' not in keys:
                raise ValueError('Missing "stages" key in overlap definition')
        except AttributeError:
            pass

        if keys is None:
            try:
                stage_list = list(overlap_def)
            except TypeError:
                raise ValueError(
                    f'Definition must must be a dict-like with keys or a list-like (or itera ble) '
                    f'of stage definitions: {type(overlap_def)}'
                )

            overlap_def: Mapping[str, Any] | Iterable[Mapping[str, Any]] = {
                'stages': stage_list,
            }

        # Parse arguments
        parsed_def: dict[str, Any] = {}

        for key, value in overlap_def.items():
            if key == 'stages':
                try:
                    parsed_def['stages'] = [
                        PairwiseOverlapStage(**stage_def)
                        if not isinstance(stage_def, PairwiseOverlapStage) else stage_def
                        for stage_def in value
                    ]
                except Exception as e:
                    raise ValueError(f'Error creating stages: {e}') from e

            elif key == 'join_cols':
                try:
                    parsed_def['join_cols'] = list(value)
                except Exception as e:
                    raise ValueError(f'Error creating join columns: {e}') from e

            elif key == 'drop_default_join_cols':
                try:
                    parsed_def['drop_default_join_cols'] = bool(value)
                except Exception as e:
                    raise ValueError(f'Error creating drop_default_join_cols: {e}') from e

            elif key == 'match_score_model':
                try:
                    parsed_def['match_score_model'] = (
                        MatchScoreModel(*value)
                        if not isinstance(value, MatchScoreModel) else value
                    )
                except Exception as e:
                    raise ValueError(f'Error creating match_score_model: {e}') from e

            elif key == 'weight_strategy':
                try:
                    parsed_def['weight_strategy'] = (
                        WeightStrategy(*value)
                        if not isinstance(value, WeightStrategy) else value
                    )
                except Exception as e:
                    raise ValueError(f'Error creating weight_strategy: {e}') from e

            elif key == 'force_end_ro':
                try:
                    parsed_def['force_end_ro'] = bool(value)
                except Exception as e:
                    raise ValueError(f'Error creating force_end_ro: {e}') from e

            elif key == 'chunk_size':
                try:
                    parsed_def['chunk_size'] = int(value)
                except Exception as e:
                    raise ValueError(f'Error creating chunk_size: {e}') from e

        return cls(**parsed_def)


def _check_expected_col(
        col: str,
        source: str,
        stage_i: Optional[int],
        expr_i: Optional[int],
):
    """Check a column name for expected suffixes when joining tables.

    :param col: Column name to check.
    :param source: Source of the column name.
    :param stage_i: Stage index.
    :param expr_i: Expression index.

    :returns: Column name without suffix.
    """
    if not (col.endswith('_a') or col.endswith('_b')):
        raise ValueError(
            f'Column {col} must end with "_a" or "_b" ('
            f'{source}, '
            f'stage {stage_i if stage_i is not None else "NA"}, '
            f'expression {expr_i if expr_i is not None else "NA"}'
            f')'
        )

    return col[:-2]
