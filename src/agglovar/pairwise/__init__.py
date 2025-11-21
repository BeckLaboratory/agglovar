"""Pairwise variant joining.

Pairwise joining intersects two variant tables, table "a" and table "b", and reports pairs of variants, one from each
table, meeting some defined match criteria. Pairwise joins are the basis of more complex merging strategies involving
two or more variant tables.

Pairwise joins only report intersects, they do not created merged tables from two callsets. Each record in the join
table describes the row index and the variant ID from each table along with statistics related to the matche, such
as reciprocal-overlap, distance, and sequence identity. Optionally, join tables may carry other columns derived from
each variant table using custom expressions.
"""

__all__ = [
    'base',
    'overlap',
    'weights',
]

from . import base
from . import overlap
from . import weights
