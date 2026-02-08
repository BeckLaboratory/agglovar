"""Base class for callset intersects."""

__all__ = [
    'CallsetDefType',
    'MergeBase',
]

from abc import ABC, abstractmethod
from collections.abc import Container, Iterable
from typing import (
    Any,
    Optional,
    TypeAlias,
)

import polars as pl

from ..meta.decorators import lockable
from ..util.str import collision_rename

CallsetDefType: TypeAlias = (
    pl.DataFrame
    | pl.LazyFrame
    | Iterable[
        pl.DataFrame | pl.LazyFrame | str
    ]
)
"""Alias for acceptable types."""

@lockable
class MergeBase(ABC):
    """Base class for callset intersects."""

    def __init__(self) -> None:
        """Initialize this base."""
        pass

    @abstractmethod
    def __call__(
            self,
            callsets: Iterable[CallsetDefType],
            retain_index: bool = False,
            pre_filter: Optional[Iterable[pl.Expr]] = None,
    ) -> pl.LazyFrame:
        """
        Intersect callsets.

        :param callsets: Callsets to intersect.
        :param retain_index: If `True`, do not drop an existing "_index" column if it exists.
        :param pre_filter: If set, filter each table with these expressions. Filter is applied
            last (after "_index" is set).

        :return: A merged callset table.
        """
        ...

    @staticmethod
    def get_intersect_tuples(
            callsets: Iterable[CallsetDefType],
            retain_index: bool = False,
            pre_filter: Optional[Iterable[pl.Expr]] = None,
    ) -> list[tuple[pl.LazyFrame, str, int]]:
        """
        Transform input arguments to a list of tuples with set fields.

        Each returned tuple has three fields:

            1. A lazy frame
            2. A name for the source
            3. An index for the source (0 for the first, increments by 1)

        If a source name is given, the given name is used. If it is not, then a default name is
        generated using the source index.

        Lazy frames are transformed to add an index ("_index" column) and to enusre a variant ID
        is present ("id" column) and that all non-null values are filled in.

        :param callsets: Callsets parameter. May be an iterable of DataFrames, LazyFrames, or
            tuples of (DataFrame, name).
        :param retain_index: If `True`, do not drop an existing "_index" column if it exists.
        :param pre_filter: If set, filter each table with these expressions. Filter is applied
            last (after "_index" is set).

        :return: A list of tuples where each tuple element represents one input source.
        """

        name_set = set()
        callset_table: pl.LazyFrame
        callset_name_pre: Any

        callset_tuple_list: list[tuple[pl.LazyFrame, str, int]] = []

        i = 0

        pre_filter = list(pre_filter if pre_filter is not None else [])

        if callsets is None:
            raise ValueError('Missing callsets')

        for callset in callsets:
            # Get callset table and name

            if isinstance(callset, (pl.LazyFrame, pl.DataFrame)):
                callset_table = callset
                callset_name = None
            else:
                try:
                    callset = tuple(callset)
                except TypeError:
                    raise ValueError(
                        f'Callset at index {i}: Callset must be a Polars table or an iterable '
                        f'of two elements (table, name): Received {type(callset)}'
                    )

                if len(callset) == 1:
                    callset_table = callset[0]
                    callset_name = None

                elif len(callset) == 2:
                    callset_table = callset[0]
                    callset_name = str(callset[1])

                else:
                    raise ValueError(f'Callset at index {i}: Input callsets must be a sequence of one or two elements: Received {len(callset)}')

            if not isinstance(callset_table, (pl.LazyFrame, pl.DataFrame)):
                raise ValueError(f'Callset at index {i}: First element Polars table: Received {type(callset_table)}')

            callset_name = _get_name(callset_name, i, name_set)

            # Set index if missing or forced
            if not (retain_index and '_index' in callset_table.collect_schema().names()):
                callset_table = (
                    callset_table
                    .drop('_index', strict=False)
                    .with_row_index('_index')
                )

            # Add missing IDs
            callset_table = (
                callset_table
                .rename({'_index': '_mg_src_index'})
                .with_columns(
                    pl.coalesce(
                        pl.col(r'^id$'),  # Existing ID if present
                        pl.concat_str(pl.lit('var'), pl.col('_mg_src_index'))  # Fill missing/null
                    ).cast(pl.String).alias('id')
                )
                .with_columns(
                    (
                        pl.when(pl.col('id').is_null())
                        .then(pl.concat_str(pl.lit('var'),  pl.col('_mg_src_index')))
                        .otherwise(pl.col('id'))
                    ).cast(pl.String).alias('id')
                )
                .filter(
                    *pre_filter
                )
            )

            # Finalize
            callset_tuple_list.append(tuple((callset_table, callset_name, i)))
            name_set.add(callset_name)

            i += 1

        return callset_tuple_list


def _get_name(
        name: Optional[Any],
        i: int,
        *args: Container[str]
) -> str:
    """Get a name for a source with a default set for None.

    Gets a name for the variant call input source and de-duplicates names.

    :param name: Input source name or None to choose a default name.
    :param i: Index of the input source (0 for the first source, etc).
    :param args: Containers with other names to avoid collisions with.

    :returns: A name for this input source.
    """

    if name is None:
        if i is None:
            i = 0

        name = f'source_{int(i + 1)}'

    name = collision_rename(str(name), '.', *args)

    return str(name)
