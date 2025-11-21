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

from . import _const
from . import _overlap
from . import _stage

__all__ = (
    getattr(_const, '__all__', [])
    + getattr(_overlap, '__all__', [])
    + getattr(_stage, '__all__', [])
)

from ._const import *
from ._overlap import *
from ._stage import *
