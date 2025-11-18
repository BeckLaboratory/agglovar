"""Generate pairwise overlaps between two variant tables.

For a pair of variant tables (A and B), find all pairs of variants between A and B that meet a set
of overlap parameters. Each pair of variants appears at most once, and a distinct variant in
either table may have multiple matches.

The join is extremely flexible and customizable. A default set of parameters should be adequate
for almost all tasks, and each stage of the join algorithm can be customized for more complex
cases.


Input variant tables
====================

Variants are input as Polars tables (:class:`polars.DataFrame` or :class:`polars.LazyFrame`)
conforming to schema :const:`agglovar.schema.VARIANT`.

.. Warning::
    To avoid extremely high CPU and memory usage, input tables should be sorted by "chrom", "pos",
    then "end". Because joins are chunked by chromosome, the chromosome sort order will not affect
    performance if records for each chromosome are grouped and sorted. If numeric chromosome names
    are sorted numerically (not default or advised), the join is not impacted, although the join
    table will be ordered by chromosome lexicographically (i.e. "1" < "10" < "2"...).

.. Warning::
   Input tables are processed lazily, but must collect results periodically to avoid memory
   exhaustion. If lazy tables have been transformed since they were read into memory, those
   transformations will be repeated with each processed chunk. Consider collecting results
   if they fit into memory, and if not, write/sink to a temporary parquet file and join the
   re-scanned temporary files.

.. Note::
   For best performance, read sorted input lazily (i.e. "scan_parquet()") or join tables in memory
   if they are small enough (i.e. "collected" tables). Pushdown operations on parquet files will
   avoid re-reading while files, and only required columns will be extracted. When reading from
   non-parquet files, read tables into memory and join them. Joining unsorted tables or tables
   scanned from non-parquet files will work, but will incur a significant performance penalty on
   large tables.

.. Note::
   No assumptions are made about the order of columns, they are referenced strictly by name. While
   "chrom", "pos", and "end" are typically the leading columns, it is not required.

Input tables are ``df_a`` and ``df_b`` where "a" is the left (first) table and "b" is the right
(second) table. Tables are treated the same by built-in join rules, and reversing their order
**should** produce equivalent results. If standard join parameters or custom parameters
unexpectedly produce violate this condition, please report it using the
`GitHub issue page <https://github.com/audanp/agglovar/issues>`__.

Joins take advantage of lazy evaluation and Polars's query optimization. Large parquet tables can
be read with ``polars.scan_parquet()`` and joined efficiently.

.. Warning::
    Reserved columns names start with "_" and may be replaced by the join process. Current reserved
    column names are in :const:`RESERVED_COLS`, which may change with any update. If these are
    present in an input table, a warning is generated, and the column is replaced. To avoid
    unexpected behavior, or drop or rename columns with a leading "_" before joining (using a
    ``polars.LazyFrame`` will avoid duplicating the table).

.. Note::
   An input variant table can be checked for errors by calling
   :meth:`PairwiseOverlap.prepare_tables()` and catching :class:`ValueError`. To check a single
   table, call this method with ``df_a`` and ``df_b`` set to the same object.


Output join tables
==================

The resulting join table describes each pairwise overlaps, but does not contain whole variant
records. Columns referring to a distinct input table have suffix "_a" or "_b" (e.g. "index_a" is
the row index in ``df_a``).

These tables can be filtered or merged with other pairwise join tables, then joined with the
original variant tables by the row index ("index_a" and "index_b") or by variant IDs
("id_a" and "id_b") if the IDs are guaranteed to be unique in each callset.

Join columns:
    0. index_a: Row index in df_a.
    #. index_b: Row index in df_b
    #. id_a: Variant ID in df_a.
    #. id_b: Variant ID in df_b.
    #. ro: Reciprocal overlap if variants overlap (0.0 if no overlap).
    #. size_ro: Size reciprocal overlap (maximum RO if variants were shifted to maximally
       overlap).
    #. offset_dist: The maximum of the start position distance and end position distance.
    #. offset_prop: Offset / variant length. Variant length is the minimum length of the two
       variants in the record.
    #. match_prop: Alignment match score divided by the maximum alignment match score

Row indexes start at 1 and increment by 1 for each row in the input table regardless of if the row
is part of a join or not. This is consistent with the behavior of
:meth:`polars.DataFrame.with_row_index()` and can be used for joining with the original variant
table.

The ID columns may also be used for joining with the input variant tables if they are unique
in each callset, although this cannot be guaranteed by Agglovar. If the "id" column is not in
the input tables, these values are null.

Additional join columns may be defined, see :ref:`agglovar_join_pair_customizing` below for
details.

.. _agglovar_join_pair_parameters:

Join parameters
===============

A set of join parameters are defined in the :class:`PairwiseOverlap` class. These parameters
should cover most joins, although they can be replaced or augmented with custom expressions
(see :ref:`agglovar_join_pair_customizing` below for details).

.. Warning::
   Without join parameters, a cross-join is performed without warning (all combinations of variants
   in both tables). This may results in a very large join table and very poor performance.

Minimum reciprocal overlap (ro_min)
-----------------------------------

**ro_min**: The minimum reciprocal-overlap (RO) between variants.

RO is defined as the number of bases overlapping in reference coordinates divided by the maximum
length of the two variants. Acceptable values are in the range [0.0, 1.0] where "0.0" means any
match, and "1.0" forces the start position and variant size to match exactly.

While this can be computed using "pos" and "end" for most variant types, the end position is
defined as ``pos + varlen`` to support insertions. Setting ``force_end_ro`` to `True` in
:class:`PairwiseOverlap` will force the existing end position to be used, but it will break
insertion overlaps. Unless "pos + varlen != end" for some valid reason, ``force_end_ro`` should
never be used.

Size reciprocal overlap (size_ro_min)
-------------------------------------

**size_ro_min**: Minimum size-RO. Size-RO is defined as the minimum "varlen" divided by the maximum
"varlen" of two variants. Size-RO is similar to RO if one variant was shifted to maximally
overlap the other (i.e. shift to the same start position). It does not reqire that the variant
overlap in reference space.

.. Warning::
   Using Size-RO without criteria limiting the distance of variants, such as "ro" or "offset_max"
   will result in a large number of joins and may dramatically impact performance.

Maximum offset (offset_max)
---------------------------

**offset_max**: Maximum offset allowed (minimum of start or end position distance).

The offset between two variants is defined as the minimum of the start or end position distance
between the two variants.

.. code-block:: python

    offset_distance = max(
        abs(var_a.pos - var_b.pos),
        abs(var_a.end - var_b.end)
    )


Maximum offset proportion (offset_prop_max)
-------------------------------------------

**offset_prop_max**: Maximum offset proportion allowed.

The offset proportion is defined as the offset distance divided by the mimum length of the two
variants being compared.

Match reference base (match_ref)
--------------------------------

**match_ref**: Match reference base ("ref" column). Only defined for SNVs.

Setting `match_ref` forces the reference base for SNVs to match. If SNV matches allow for an offset
distance, setting this parameter is advised to avoid nonsensical matches (i.e. an A->C SNV
should not match a G->T SNV). If `offset_max` is 0, this parameter should not have an effect unless
the "ref" column is malformed.

Match alternate base (match_alt)
--------------------------------

**match_alt**: Match alternate base ("alt" column). Only defined for SNVs.

Setting `match_alt` forces alternate bases for SNVs to match and is advised for all SNV matching.


Minimum sequence match proportion (match_prop_min)
--------------------------------------------------

**match_prop_min**: Minimum sequence match proportion.

When variant overlaps use only size and position, it is difficult to control false-positive
and false-negative matches while permitting normal genetic variation. Setting `match_prop_min` will
force a match threshold based on variant sequence similarity. Using permissive parameters on size
and position to reduce false negatives with a strict match proportion to reduce false positives is
one powerful join strategy.

Match proportion uses an alignment scores between two variant sequences. A score model
(``match_score_model`` parameter of :class:`PairwiseOverlap`) defines how to score alignments
based on matches, mismatches, and gaps. The match proportion is defined as the alignment score
between two sequence divided by the maximum alignment score that could be obtained if every base
matched.

.. code-block:: python

    match_prop = alignment_score / (match_score * min(var_a.varlen, var_b.varlen))

Where `match_score` is the value added for each matching base in the alignment
(:meth:`ScoreModel.match`).

Using a match proportion avoids using direct alignment scores, which are not intuitive and are
significantly affected by sequence lengths how alignments are scored.

When variants are shifted in one sample relative to another, the variant sequence is "rotated"
making direct sequence comparisons difficult. However, Agglovar corrects for this sequence
representation. See
`Small polymorphisms are a source of ancestral bias in structural variant breakpoint placement
<https://genome.cshlp.org/content/34/1/7.long>`__  for more details about why this occurs.

Both Agglovar and `Truvari <https://github.com/ACEnglish/truvari>`__ have a sequence match criteria,
but they are different and both have merits. While Agglovar compares *only* variant sequences even
if they are shifted, Truvari includes the shift in the sequence match (i.e. the compared sequences
include the variant and flanking reference bases).

Reasonable join parameters
--------------------------

While there are no "optimal" join parameters, these are parameters we have found to work well.

SVs (INS, DEL, INV, DUP >= 50 bp):

==============  ======
Parameter       Value
==============  ======
ro_min          0.5
match_prop_min  0.8
==============  ======

indels (INS, DEL < 50 bp):

==============  ======
Parameter       Value
==============  ======
size_ro_min     0.8
offset_max      200
match_prop_min  0.8
==============  ======

SNVs:SVs (INS, DEL, INV, DUP >= 50 bp):

==============  ======
Parameter       Value
==============  ======
offset_max      0
match_ref       True
==============  ======


Join process
============

The initial join stage is performed using the following logic:

.. code-block:: python

    (
        df_a
        .join_where(
            df_b,
            *join_predicates
        )
        .select(*join_cols)
        .filter(*join_filters)
        .sort(['index_a', 'index_b'])
    )

This expression contains several key components:
    * join_predicates: A list of predicates to applied during the join.
    * join_cols: A list of columns to select from the joined table.
    * join_filters: A list of filters to apply to the join.

**Join predicates** (``join_predicates``)are expressions on ``df_a`` and ``df_b`` that determine
which rows are joined. At this stage, column names will have "_a" or "_b" suffixes to indicate
their origin.

**Join columns** (``join_cols``) is a list of expressions that generate the final join table.
These expressions may be a single column, such as ``id_a``, or an expression that generates a new
column, such as "ro" (derived from existing position and end columns).

**Join filters** (``join_filters``) are expressions that filter the joined table, and they operate
on columns created by **Join columns**.

More on the join strategy
-------------------------

With lazy evaluation and Polars query optimization, expressions may be used at multiple stages. For
example, if a minimum "ro" (reciprocal overlap) is set, then the expression that calculates
"ro" is used as a join predicate to limit the size of joins and again as a join column.

Alternatively, the expression generating "ro" could be used once as a join column, then the "ro"
result could be used as a filter at the filter stage, however, applying the filter at the predicate
stage may eliminate pairs of variants before they are processed through remaining stages.

This strategy also allows additional flexibility in the join process, for example, a filter can be
enforced at the predicate stage and never appear as a column in the final table.

Note that currently, there is not a "drop" stage following the filter stage, so filters that do not
use join columns must be applied as join predicates.


.. _agglovar_join_pair_customizing:

Customizing the join process
============================

Most join operations can be defined simply by setting parameters (see
:ref:`agglovar_join_pair_parameters`).

Join filters operate on the join columns and provide further filtering. These expressions are set
by the join parameters passed to the :class:`PairwiseOverlap` constructor. Additional expressions
can also be added. These may affect join predicates, columns, or filters.

This is a powerful feature, and it should be used with caution. Use join parameters whenever
possible, and augment them with additional expressions only when necessary. It is possible to
use both, and the join parameters will come first.

When a :class:`PairwiseOverlap` instance is created, additional expressions can be added to it
until either :meth:`PairwiseOverlap.lock` is called or a join is performed (which locks the
object from further modifications).

Three methods can insert expressions:

* :meth:`PairwiseOverlap.append_join_predicates`
* :meth:`PairwiseOverlap.append_join_cols`
* :meth:`PairwiseOverlap.append_join_filters`

Each of these takes a single expression or an iterable object of expressions. Note that the
output join table columns are affected by appended columns. This can be used to add additional
statistics to the output table or to retain columns from the original tables (consider joining
the output table with the original tables to retrieve columns in this case).

Expected columns (:const:`PairwiseOverlap.expected_cols`) is updated automatically with each new
expression and cannot be modified directly.

Join Class Inspection
=====================

A :class:`PairwiseOverlap` instance has properties for inspecting the join process. These
properties are set by join parameters and customized expressions.

Join inspection properties:
    * ``join_predicates``: A list of join predicates expressions.
    * ``join_cols``: A list of join column expressions.
    * ``join_filters``: A list of filter expressions.
    * ``expected_cols``: A list of column names expected in ``df_a`` and ``df_b``.

The values in :meth:`PairwiseOverlap.expected_cols` are automatically derived from expressions in
:meth:`PairwiseOverlap.join_predicates` and :meth:`PairwiseOverlap.join_cols`. These columns
must be either present in ``df_a`` and ``df_b``, or they must be auto-generated columns
(see :const:`AUTOGEN_COLS`).

.. note::
   These properties return copies of the instance's internal lists. Modifying them will not affect
   the joins. See :ref:`agglovar_join_pair_customizing` below for information on altering the join
   process for complex use cases.
"""

__all__ = [
    'RESERVED_COLS',
    'AUTOGEN_COLS',
    'DEFAULT_CHUNK_SIZE',
    'PairwiseOverlap',
    'chunk_index',
    'join_iter',
    'join',
]

from collections.abc import Iterable, Iterator
from typing import Optional
from warnings import warn

import polars as pl

from .. import schema
from ..meta.decorators import immutable
from ..meta.descriptors import BoundedFloat, CheckedBool, BoundedInt
from ..seqmatch import MatchScoreModel

from .base import PairwiseJoin
from .weights import WeightStrategy, DEFAULT_WEIGHT_STRATEGY

RESERVED_COLS: frozenset[str] = frozenset({
    '_index',
    '_end_ro',
})
"""Reserved columns are added automatically to input tables."""

AUTOGEN_COLS: frozenset[str] = frozenset({
    'varlen',
    'id',
})
"""Columns automatically generated by the join process for input tables if they are not present."""

MIN_CHUNK_SIZE: int = 100
"""Minimum chunk size."""

DEFAULT_CHUNK_SIZE: int = 5_000
"""
Default size to chunk tables before joining. A value of 10,000 works well to balance combinatorial
explosion without exploding the number of chunks to merge when overlapping large variant tables.
"""


@immutable
class PairwiseOverlap(PairwiseJoin):
    """Pairwise overlap class.

    Join by overlapping variants by position.

    :ivar ro_min: Minimum reciprocal overlap for allowed matches. If 0.0, then any overlap matches.
    :ivar size_ro_min: Reciprocal length proportion of allowed matches. If `match_prop_min` is set
        and the value of this parameter is `None` or is less than `match_prop_min`, then it is set
        to `match_prop_min` since this value represents the lower-bound of allowed match
        proportions.
    :ivar offset_max: Maximum offset allowed (minimum of start or end position distance).
    :ivar offset_prop_max: Maximum size-offset (offset / varlen) allowed.
    :ivar match_ref: "REF" column must match between two variants.
    :ivar match_alt: "ALT" column must match between two variants.
    :ivar match_prop_min: Minimum matched base proportion in alignment or None to not match.
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

    # Join parameters
    ro_min: Optional[float] = BoundedFloat(min_val=0.0, max_val=1.0)
    size_ro_min: Optional[float] = BoundedFloat(min_val=(0.0, False), max_val=1.0)
    offset_max: Optional[int] = BoundedInt(0)
    offset_prop_max: Optional[float] = BoundedFloat(0.0)
    seg_ro_min: Optional[float] = BoundedFloat(min_val=0.0, max_val=1.0)
    match_ref: bool = CheckedBool()
    match_alt: bool = CheckedBool()
    match_prop_min: Optional[float]

    # Advanced Configuration Attributes
    match_score_model: MatchScoreModel
    force_end_ro: bool = CheckedBool()
    chunk_size: int = BoundedInt(MIN_CHUNK_SIZE, default=DEFAULT_CHUNK_SIZE)

    # Table and Join Control
    _join_predicates: list[pl.Expr]
    _join_filters: list[pl.Expr]
    _join_cols: list[pl.Expr]

    _expected_cols: set[str]
    _chunk_range: dict[tuple[str, str], list[pl.Expr]]

    def __init__(
            self,
            *,
            ro_min: Optional[float] = None,
            size_ro_min: Optional[float] = None,
            offset_max: Optional[int] = None,
            offset_prop_max: Optional[float] = None,
            match_ref: bool = False,
            match_alt: bool = False,
            seg_ro_min: Optional[float] = None,
            match_prop_min: Optional[float] = None,
            match_score_model: Optional[MatchScoreModel] = None,
            force_end_ro: bool = False,
            weight_strategy: WeightStrategy = DEFAULT_WEIGHT_STRATEGY,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__(weight_strategy)

        # Set parameters
        self.ro_min = ro_min
        self.size_ro_min = size_ro_min
        self.offset_max = offset_max
        self.offset_prop_max = offset_prop_max
        self.match_ref = match_ref
        self.match_alt = match_alt
        self.seg_ro_min = seg_ro_min
        self.match_prop_min = match_prop_min
        self.match_score_model = match_score_model
        self.force_end_ro = force_end_ro
        self.chunk_size = chunk_size

        # Set internal attributes and containers
        self._join_predicates = []
        self._join_filters = []
        self._join_cols = []

        self._expected_cols = {'chrom', 'pos', 'end'}
        self._chunk_range = dict()

        # Reusable join-table expressions
        expr_overlap_ro = (
            (
                (
                    pl.min_horizontal('_end_ro_a', '_end_ro_b')
                    - pl.max_horizontal('pos_a', 'pos_b')
                ) / (
                    pl.max_horizontal('varlen_a', 'varlen_b')
                )
            )
            .clip(0.0, 1.0)
            .cast(pl.Float32)
        )

        expr_szro = (
            (
                (
                    pl.min_horizontal('varlen_a', 'varlen_b')
                ) / (
                    pl.max_horizontal('varlen_a', 'varlen_b')
                )
            )
            .cast(pl.Float32)
        )

        expr_offset_dist = (
            pl.max_horizontal(
                (pl.col('pos_a') - pl.col('pos_b')).abs(),
                (pl.col('end_a') - pl.col('end_b')).abs()
            )
            .cast(pl.Int32)
        )

        expr_offset_prop = (
            (
                expr_offset_dist / pl.min_horizontal('varlen_a', 'varlen_b')
            )
            .cast(pl.Float32)
        )

        expr_match_prop = (
            pl.struct('seq_a', 'seq_b')
            .map_elements(
                lambda s: self.match_score_model.match_prop(s['seq_a'], s['seq_b']),
                return_dtype=pl.Float32
            )
        ) if self.match_prop_min is not None else (
            pl.lit(None).cast(pl.Float32)
        )

        # Set ranges and expressions from parameters
        if self.ro_min is not None:
            self.append_join_predicates(
                expr_overlap_ro >= self.ro_min
            )

            # self.append_join_filters(
            #     pl.col('ro') >= ro_min
            # )

            self._append_chunk_range('pos', 'max', pl.col('_end_ro_a'))
            self._append_chunk_range('_end_ro', 'min', pl.col('pos_a'))

        if self.size_ro_min is not None:
            self.append_join_predicates(
                expr_szro >= self.size_ro_min
            )

            self._append_chunk_range('varlen', 'min', pl.col('varlen_a') * self.size_ro_min)
            self._append_chunk_range('varlen', 'max', pl.col('varlen_a') * (1 / self.size_ro_min))

        if self.offset_max is not None:
            if self.offset_max == 0:
                self.append_join_predicates([  # Very fast joins on equality
                    pl.col('pos_a') == pl.col('pos_b'),
                    pl.col('end_a') == pl.col('end_b'),
                ])
            else:
                self.append_join_predicates(
                    expr_offset_dist <= self.offset_max
                )

            self._append_chunk_range('pos', 'min', pl.col('pos_a') - self.offset_max)
            self._append_chunk_range('end', 'max', pl.col('end_a') + self.offset_max)

        if self.offset_prop_max is not None:
            self.append_join_predicates(
                expr_offset_prop <= self.offset_prop_max
            )

            self._append_chunk_range('pos', 'min', pl.col('pos_a') - pl.col('varlen_a') * self.offset_prop_max)
            self._append_chunk_range('end', 'max', pl.col('end_a') + pl.col('varlen_a') * self.offset_prop_max)

        if self.match_ref:
            self.append_join_predicates(
                pl.col('ref_a') == pl.col('ref_b')
            )

        if self.match_alt:
            self.append_join_predicates(
                pl.col('alt_a') == pl.col('alt_b')
            )

        if self.match_prop_min is not None and self.match_prop_min > 0.0:
            self.append_join_filters(
                pl.col('match_prop') >= self.match_prop_min
            )

        if self.seg_ro_min is not None:
            self._add_expected_cols((pl.col('seg_a'), pl.col('seg_b')))
            # self.append_join_predicates(
            #     pl.col('ro_seg') >= self.seg_ro_min
            # )

        #
        # Advanced Configuration Attributes
        #

        if self.match_score_model is None:
            self.match_score_model = MatchScoreModel()

        elif not isinstance(self.match_score_model, MatchScoreModel):
            raise ValueError(
                f'Alignment model (match_score_model) must be a seqmatch.MatchScoreModel: '
                f'{type(self.match_score_model)}'
            )

        # Set join columns
        self.append_join_cols([
            pl.col('_index_a').alias('index_a'),
            pl.col('_index_b').alias('index_b'),
            pl.col('id_a'),
            pl.col('id_b'),
            expr_overlap_ro.alias('ro'),
            expr_offset_dist.alias('offset_dist'),
            expr_offset_prop.alias('offset_prop'),
            expr_szro.alias('size_ro'),
            expr_match_prop.alias('match_prop')
        ])

    @property
    def join_predicates(self) -> list[pl.Expr]:
        """List of expressions to be applied during the table join.

        These expressions are arguments to pl.join_where() and operate on columns
        of df_a and df_b joined into one record where all columns from df_a have the suffix "_a"
        and all columns from df_b have the suffix "_b". For example, if match_alt is True, the expression
        "pl.col('alt_a') == pl.col('alt_b')" will be in this list.
        """
        return self._join_predicates.copy()

    @property
    def join_filters(self) -> list[pl.Expr]:
        """List of expressions to be applied after the join is performed.

        These expressions operate on the columns of the joined table (see join table columns in the class docstring).
        """
        return self._join_filters.copy()

    @property
    def join_cols(self) -> list[pl.Expr]:
        """List of columns to include in the join table."""
        return self._join_cols.copy()

    @property
    def required_cols(self) -> set[str]:
        """The minimum set of columns that must be present in input tables.

        This is set based on parameters needed to perform the join. For example, if sequence
        matching is required, then "seq" will be in this list, and if "seq" does not exist
        in both df_a and df_b, then an error is raised.
        """
        return set(self._expected_cols - AUTOGEN_COLS - RESERVED_COLS)

    @property
    def reserved_cols(self) -> set[str]:
        """A set of columns that are reserved for internal use and must not be present in input tables."""
        return set(RESERVED_COLS)

    @property
    def chunk_range(self) -> dict[tuple[str, str], list[pl.Expr]]:
        """Get expressions for chunking.

        A dict of keys to a list of expressions used to subset df_b to include only variants
        that may match variants in a df_a chunk.

        Keys are formatted as "field_limit" where "limit" is "min" or "max" (e.g. "pos_min"
        is the minimum value for "pos"). The list of expressions associated with a key are
        executed on a df_a chunk, and the minimum or maximum value from the list (one element
        per record in df_a) is used as the limit value for a field in df_b. For example, if
        "pos_min" is a key and [pl.col('pos_a')] is the value, then the expression takes the
        minimum value of pos_a across all records in df_a and uses it to filter df_b such that
        no variant in the chunked df_b table has "pos_b" less than this minimum value. If
        multiple expressions are given, then all expressions are executed and the minimum or
        maximum value for all is taken. This allows non-trivial chunking of df_b necessary to
        restrict combinatorial explosion for certain parameters. For example, if reciprocal
        overlap (ro_min) is set, the maximum position in df_b is determined by the minimum end
        position in df_a (i.e. "pos_max" will contain "pl.col('end_ro_a'))".
        """
        return self._chunk_range.copy()

    @property
    def has_match_prop(self) -> bool:
        """`True` if "match_prop" is computed for the join table."""
        return self.match_prop_min is not None

    def append_join_predicates(
            self,
            expr: Iterable[pl.Expr] | pl.Expr
    ) -> None:
        """Append expressions to a list of join predicates given as arguments to pl.join_where().

        This class will construct a list of join predicates from the constructor arguments,
        but additional join control may be added here.

        .. Warning::
            Adding predicates may alter the join results so that they are not reproducible
            based on join arguments. Use with caution.

        :param expr: An expression or list of expressions.
        """
        if self.is_locked:
            raise AttributeError('Pairwise join object is locked and cannot be modified.')

        if isinstance(expr, pl.Expr):
            expr = [expr]

        self._add_expected_cols(expr)
        self._join_predicates.extend(expr)

    def append_join_filters(
            self,
            expr: Iterable[pl.Expr] | pl.Expr
    ) -> None:
        """Append expressions to a list of join filters.

        These filters are applied to the join table immediately after the join through pl.filter().

        .. Warning::
            Adding predicates may alter the join results so that they are not reproducible
            based on join arguments. Use with caution.

        :param expr: An expression or list of expressions.
        """
        if self.is_locked:
            raise AttributeError('Pairwise join object is locked and cannot be modified.')

        if isinstance(expr, pl.Expr):
            expr = [expr]

        self._join_filters.extend(expr)

    def append_join_cols(
            self,
            expr: Iterable[pl.Expr] | pl.Expr
    ) -> None:
        """Append expressions to the list of columns included in the join table.

        These columns will be appended to the standard join table columns. Each expression
        should name the column it creates using ".alias()" if necessary. These columns do
        not affect the join itself, just the columns that appear in the join table.

        For example, to retain the "pos" column from df_a and df_b, then append
        "pl.col('pos_a')" and "pl.col('pos_b')". If you wanted to set a flag for whether
        the variant in df_a comes before df_b, then a new columns could be added:
        "(pl.col('pos_a') <= pl.col('pos_b')).alias('left_a')"

        :param expr: Expression or list of expressions.
        """
        if self.is_locked:
            raise AttributeError('Pairwise join object is locked and cannot be modified.')

        if isinstance(expr, pl.Expr):
            expr = [expr]

        self._add_expected_cols(expr)
        self._join_cols.extend(expr)

    def _add_expected_cols(
            self,
            expr: Iterable[pl.Expr] | pl.Expr
    ) -> None:
        """Inspect expressions and add each required column to the set of expected columns in the source dataframe.

        :param expr: List of expressions.
        """
        if isinstance(expr, pl.Expr):
            expr = [expr]

        for e in expr:
            for col_name in e.meta.root_names():
                if not col_name.endswith('_a') and not col_name.endswith('_b'):
                    raise ValueError(
                        f'Expected column name to end with "_a" or "_b": Found column "{col_name}" in expression "{e}"'
                    )

                col_name = col_name[:-2]

                if col_name not in RESERVED_COLS:
                    self._expected_cols.add(col_name)

    def _append_chunk_range(
            self,
            key: str,
            limit: str,
            expr: pl.Expr
    ) -> None:
        """Append a rule for chunking df_b based on a subset of df_a.

        :param key: Column name in df_b without "_b" suffix (e.g. "pos" for "pos_a").
        :param limit: If the limit is "min" or "max".
        :param expr: An expression applied to df_a to determine a minimum or maximum value for column "key" in df_b.
        """
        if limit not in {'min', 'max'}:
            raise ValueError(f'Limit must be "min" or "max": {limit}')

        if not (key := key.strip() if key else None):
            raise ValueError('Key must not be empty')

        if key not in RESERVED_COLS:
            self._expected_cols.add(key)

        self._add_expected_cols(expr)

        if (key, limit) not in self._chunk_range:
            self._chunk_range[key, limit] = []

        self._chunk_range[key, limit].append(expr)

    def join_iter(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame
    ) -> Iterator[pl.LazyFrame]:
        """Find all pairs of variants in two sources that meet a set of criteria.

        :param df_a: Source dataframe.
        :param df_b: Target dataframe.

        :yields: A LazyFrame for each chunk.
        """
        join_empty = True  # Detects if no joins were written

        # Prepare tables
        df_a, df_b = self.prepare_tables(df_a, df_b, warn_on_reserved=True)

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
            )

            while start_index_a < last_index_a:
                end_index_a = start_index_a + self.chunk_size

                df_a_chunk = df_a_chrom.filter(
                    pl.col('_index_chrom_a') >= start_index_a,
                    pl.col('_index_chrom_a') < end_index_a
                ).collect().lazy()

                df_b_chunk = (
                    self._chunk_relative(df_a_chunk, df_b, chrom)
                    .with_row_index('_index_chunk_b')
                ).collect().lazy()

                start_index_b = 0
                last_index_b = df_b_chunk.select(pl.col('_index_chunk_b').max() + 1).collect().item()

                if last_index_b is None:
                    start_index_a = end_index_a
                    continue

                while start_index_b < last_index_b:
                    end_index_b = start_index_b + self.chunk_size

                    df_join = (
                        df_a_chunk
                        .join_where(
                            df_b_chunk.filter(
                                pl.col('_index_chunk_b') >= start_index_b,
                                pl.col('_index_chunk_b') < end_index_b
                            ),
                            *(self._join_predicates if self._join_predicates else [pl.lit(True)])
                        )
                        .select(
                            *self._join_cols
                        )
                        .filter(
                            *(self._join_filters if self._join_filters else [pl.lit(True)])
                        )
                        .with_columns(
                            self.weight_strategy.expr.alias('weight')
                        )
                        .sort(
                            ['index_a', 'index_b']
                        )
                    )

                    if self.seg_ro_min is not None:
                        df_seg_ro = (
                            df_join
                            .select('index_a', 'index_b')
                            .join(
                                (
                                    df_a_chunk
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
                                    df_b_chunk
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

                        df_join = (
                            df_join
                            .join(
                                df_seg_ro,
                                on=['index_a', 'index_b'],
                                how='left'
                            )
                            .with_columns(
                                pl.col('seg_ro').fill_null(1.0)
                            )
                            .filter(
                                pl.col('seg_ro') >= self.seg_ro_min
                            )
                            .collect()
                            .lazy()
                        )

                    yield df_join.collect().lazy()

                    join_empty = False

                    start_index_b = end_index_b

                start_index_a = end_index_a

        if join_empty:
            # If no join tables were yielded, yield an empty one. This creates an empty join table
            # with the correct structure and prevents pl.concat from failing on an empty list.
            yield (
                df_a.head(0)
                .join_where(
                    df_b.head(0),
                    *(self._join_predicates if self._join_predicates else [pl.lit(True)])
                )
                .select(
                    *self._join_cols
                )
                .filter(
                    *(self._join_filters if self._join_filters else [pl.lit(True)])
                )
                .with_columns(
                    self.weight_strategy.expr.alias('weight')
                )
                .sort(
                    ['index_a', 'index_b']
                )
                .collect()
                .lazy()
            )

    # def join_iter(
    #         self,
    #         df_a: pl.DataFrame | pl.LazyFrame,
    #         df_b: pl.DataFrame | pl.LazyFrame
    # ) -> Iterator[pl.LazyFrame]:
    #     """Find all pairs of variants in two sources that meet a set of criteria.
    #
    #     :param df_a: Source dataframe.
    #     :param df_b: Target dataframe.
    #
    #     :yields: A LazyFrame for each chunk.
    #     """
    #     # Prepare tables
    #     df_a, df_b = self.prepare_tables(df_a, df_b, warn_on_reserved=True)
    #
    #     # Join per chromosome
    #     chrom_list = df_a.select('chrom_a').unique().collect().to_series().sort().to_list()
    #
    #     self.lock()
    #
    #     if len(chrom_list) == 0:
    #         # No chromosomes found, yield empty table (otherwise, concat across iterater will fail on an empty list)
    #         yield (
    #             df_a.head(0)
    #             .join_where(
    #                 df_b.head(0),
    #                 *(self._join_predicates if self._join_predicates else [pl.lit(True)])
    #             )
    #             .select(
    #                 *self._join_cols
    #             )
    #             .filter(
    #                 *(self._join_filters if self._join_filters else [pl.lit(True)])
    #             )
    #             .sort(
    #                 ['index_a', 'index_b']
    #             )
    #         )
    #
    #     for chrom in chrom_list:
    #         print(f'Join chrom: {chrom}')  # DEBUG
    #
    #         range_a_min, range_a_max = (
    #             df_a
    #             .filter(
    #                 pl.col('chrom_a') == chrom
    #             )
    #             .select(
    #                 pl.col('_index_a').min().alias('min'),
    #                 (pl.col('_index_a').max() + 1).alias('max')
    #             )
    #             .collect()
    #             .transpose()
    #             .to_series()
    #         )
    #
    #         chunk_size_chrom = self.chunk_size if self.chunk_size > 0 else range_a_max - range_a_min
    #
    #         for i_a in range(range_a_min, range_a_max, chunk_size_chrom):
    #             i_a_end = min(i_a + chunk_size_chrom, range_a_max)
    #
    #             df_a_chunk = chunk_index(df_a, i_a, i_a_end)
    #             df_b_chunk = self._chunk_relative(df_a_chunk, df_b, chrom)
    #
    #             df_a.chunk = df_a_chunk.collect()
    #
    #             print(f'\t* Chunk size: {i_a_end - i_a}x{df_b_chunk.select(pl.len()).collect().item()}')  # DEBUG
    #
    #             yield (
    #                 df_a_chunk
    #                 .join_where(
    #                     df_b_chunk,
    #                     *(self._join_predicates if self._join_predicates else [pl.lit(True)])
    #                 )
    #                 .select(
    #                     *self._join_cols
    #                 )
    #                 .filter(
    #                     *(self._join_filters if self._join_filters else [pl.lit(True)])
    #                 )
    #                 .with_columns(
    #                     self.weight_strategy.expr.alias('weight')
    #                 )
    #                 .sort(
    #                     ['index_a', 'index_b']
    #                 )
    #             )

    def prepare_tables(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame,
            warn_on_reserved: bool = False
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """Prepares tables for join.

        Checks for expected columns and formats, adds missing columns as needed, and
        appends "_a" and "_b" suffixes to column names.

        :param df_a: Table A.
        :param df_b: Table B.
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
        if reserved_cols := sorted(self.check_reserved_cols(columns_a)):
            err_str = f'Reserved columns in table "A": {", ".join(reserved_cols)}'

            if warn_on_reserved:
                warn(f'{err_str}: Dropping column(s)')
            else:
                raise ValueError(err_str)

            df_a = df_a.drop(reserved_cols, strict=False)

        if reserved_cols := sorted(self.check_reserved_cols(columns_b)):
            err_str = f'Reserved columns in table "B": {", ".join(reserved_cols)}'

            if warn_on_reserved:
                warn(f'{err_str}: Dropping column(s)')
            else:
                raise ValueError(err_str)

            df_b = df_b.drop(reserved_cols, strict=False)

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
        df_a = (
            df_a
            .with_row_index('_index')
        )

        df_b = (
            df_b
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
        if self.match_ref:
            df_a = df_a.with_columns(
                pl.col('ref').str.to_uppercase().str.strip_chars()
            )

            df_b = df_b.with_columns(
                pl.col('ref').str.to_uppercase().str.strip_chars()
            )

        if self.match_alt:
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

        # Append suffixes to all columns
        df_a = df_a.select(pl.all().name.suffix('_a'))
        df_b = df_b.select(pl.all().name.suffix('_b'))

        return df_a, df_b

    def _chunk_relative(
            self,
            df_a: pl.LazyFrame,
            df_b: pl.LazyFrame,
            chrom: str
    ) -> pl.LazyFrame:
        """Chunk one DataFrame relative to another.

        Chunk df_b relative to df_a choosing records in df_b that could possibly be joined with some record in df_a.
        For example, this function may determine the minimum and maximum values of pos and end, and then subset df_b
        by those values. The actual subset values are determined by the `chunk_range` attribute.

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

        :param df_a: Table chunk.
        :param df_b: Table to be chunked to records that may overlap with df_a.
        :param chrom: Chromosome name.

        :returns: df_b partitioned (LazyFrame).
        """
        filter_list = [
            pl.col('chrom_b') == chrom
        ]

        for (col_name, limit), expr_list in self._chunk_range.items():
            if limit == 'min':
                filter_list.append(
                    pl.col(col_name + '_b') >= (
                        df_a
                        .select(pl.min_horizontal(*expr_list))
                        .collect()
                        .to_series()
                        .min()
                    )
                )

            elif limit == 'max':
                filter_list.append(
                    pl.col(col_name + '_b') <= (
                        df_a
                        .select(pl.max_horizontal(*expr_list))
                        .collect()
                        .to_series()
                        .max()
                    )
                )

            else:
                raise ValueError(f'Unknown limit: "{limit}"')

        return df_b.filter(*filter_list)


def chunk_index(
        df_a: pl.LazyFrame,
        i_a: int,
        i_a_max: int
) -> pl.LazyFrame:
    """Chunk df_a by a range of indices.

    WARNING: This function assumes the index range contains a single chromosome,
    and this is not checked.

    :param df_a: Table to chunk.
    :param i_a: Minimum index.
    :param i_a_max: Maximum index.

    :returns: df_a chunked (LazyFrame).
    """
    if i_a < 0 or i_a_max <= i_a:
        raise ValueError(f'Invalid index range: [{i_a}, {i_a_max})')

    return (
        df_a
        .filter(pl.col('_index_a') >= i_a)
        .filter(pl.col('_index_a') < i_a_max)
    )


def join_iter(
    df_a: pl.DataFrame | pl.LazyFrame,
    df_b: pl.DataFrame | pl.LazyFrame,
    ro_min: Optional[float] = None,
    size_ro_min: Optional[float] = None,
    offset_max: Optional[int] = None,
    offset_prop_max: Optional[float] = None,
    match_ref: bool = False,
    match_alt: bool = False,
    match_prop_min: Optional[float] = None
) -> Iterator[pl.LazyFrame]:
    """A convenience wrapper for running joins.

    This function creates a :class:`PairwiseOverlap` object and calls
    :meth:`PairwiseOverlap.join_iter()`.

    :param df_a: Table A.
    :param df_b: Table B.
    :param ro_min: Minimum reciprocal overlap for allowed matches. If 0.0, then any overlap matches.
    :param size_ro_min: Reciprocal length proportion of allowed matches.
        this value represents the lower-bound of allowed match proportions.
    :param offset_max: Maximum offset allowed (minimum of start or end position distance).
    :param offset_prop_max: Maximum size-offset (offset / varlen) allowed.
    :param match_ref: "REF" column must match between two variants.
    :param match_alt: "ALT" column must match between two variants.
    :param match_prop_min: Minimum matched base proportion in alignment or None to not match.

    :yields: A LazyFrame for each chunk.
    """  # noqa: D402 (complains about "join()" in the dicstring).
    return PairwiseOverlap(
        ro_min=ro_min,
        size_ro_min=size_ro_min,
        offset_max=offset_max,
        offset_prop_max=offset_prop_max,
        match_ref=match_ref,
        match_alt=match_alt,
        match_prop_min=match_prop_min
    ).join_iter(df_a, df_b)


def join(
    df_a: pl.DataFrame | pl.LazyFrame,
    df_b: pl.DataFrame | pl.LazyFrame,
    ro_min: Optional[float] = None,
    size_ro_min: Optional[float] = None,
    offset_max: Optional[int] = None,
    offset_prop_max: Optional[float] = None,
    match_ref: bool = False,
    match_alt: bool = False,
    match_prop_min: Optional[float] = None
) -> pl.LazyFrame:
    """A convenience wrapper for running joins.

    This function creates a :class:`PairwiseOverlap` object and calls
    :meth:`PairwiseOverlap.join()`.

    :param df_a: Table A.
    :param df_b: Table B.
    :param ro_min: Minimum reciprocal overlap for allowed matches. If 0.0, then any overlap matches.
    :param size_ro_min: Reciprocal length proportion of allowed matches.
    :param offset_max: Maximum offset allowed (minimum of start or end position distance).
    :param offset_prop_max: Maximum size-offset (offset / varlen) allowed.
    :param match_ref: "REF" column must match between two variants.
    :param match_alt: "ALT" column must match between two variants.
    :param match_prop_min: Minimum matched base proportion in alignment or None to not match.

    :returns: A join table.
    """  # noqa: D402 (complains about "join()" in the dicstring).
    return PairwiseOverlap(
        ro_min=ro_min,
        size_ro_min=size_ro_min,
        offset_max=offset_max,
        offset_prop_max=offset_prop_max,
        match_ref=match_ref,
        match_alt=match_alt,
        match_prop_min=match_prop_min,
    ).join(df_a, df_b)
