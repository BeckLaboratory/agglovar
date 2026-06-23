"""Base class for callset intersects.


"""

__all__ = [
    'CallsetDef',
    'MergeBase',
    'CallsetDefInputType',
]

from abc import ABC, abstractmethod
from collections.abc import Container, Iterable
import json
from pathlib import Path
from typing import (
    Any,
    NamedTuple,
    Optional,
    TypeAlias,
)

import polars as pl

from ..meta.decorators import lockable
from ..util.str import collision_rename


class CallsetDef(NamedTuple):
    """Callset definition.

    :ivar table: The callset table to be merged.
    :ivar name: The name of the callset within a collection of callsets.
    :ivar metadata: The metadata of the callset or None if no metadata is provided.
    """
    table: pl.LazyFrame
    name: str
    metadata: Optional[str]

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'CallsetDef(table={self.table!r}, name={self.name}, metadata={self.metadata})'

CallsetDefInputType: TypeAlias = (
    pl.DataFrame
    | pl.LazyFrame
    | tuple[
        pl.DataFrame | pl.LazyFrame,
    ]
    | tuple [
        pl.DataFrame | pl.LazyFrame,
        Optional[str],
    ]
    | tuple [
        pl.DataFrame | pl.LazyFrame,
        Optional[str],
        Optional[str | dict],
    ]
    | CallsetDef
)
"""Alias for acceptable input types."""


def _get_definition(
        callset_def: CallsetDefInputType,
        callset_index: int,
        name_set: set[str],
        retain_index: bool,
        pre_filter: Iterable[pl.Expr],
) -> CallsetDef:
    """Get a callset definition from an input type.

    :param callset_def: The callset definition to be processed.
    :param callset_index: The index of the callset.
    :param name_set: A set of names that have already been used.
    :param retain_index: If `True`, do not drop an existing "_index" column if it exists.
    :param pre_filter: If set, filter each table with these expressions. Filter is applied
        last (after "_index" is set).

    :return: A callset definition.
    """

    # Get callset table and name
    callset_name = None
    callset_meta = None

    if isinstance(callset_def, CallsetDef):
        callset_table = callset_def.table
        callset_name = callset_def.name
        callset_meta = callset_def.metadata

    elif isinstance(callset_def, pl.LazyFrame):
        callset_table = callset_def

    elif isinstance(callset_def, pl.DataFrame):
        callset_table = callset_def.lazy()

    elif isinstance(callset_def, tuple):
        if len(callset_def) < 1:
            raise ValueError(
                f'Callset must be a Polars table or an iterable '
                f'of callset definition elements (table, name, metadata): Received an empty iterable'
            )

        if isinstance(callset_def[0], pl.LazyFrame):
            callset_table = callset_def[0]
        elif isinstance(callset_def[0], pl.DataFrame):
            callset_table = callset_def[0].lazy()
        else:
            raise ValueError(
                f'First element of callset definition must be a Polars table: '
                f'Received {type(callset_def[0])}'
            )

        callset_name = str(callset_def[1]) if len(callset_def) > 1 else None

        callset_meta = callset_def[2] if len(callset_def) > 2 else None

        if callset_meta is not None:
            if isinstance(callset_meta, dict):
                callset_meta = json.dumps(callset_meta)
            else:
                callset_meta = str(callset_meta)

        if len(callset_def) > 3:
            raise ValueError(
                f'Input callsets must be a sequence of up to three items (table, name, metadata): '
                f'Received {len(callset_def)} items'
            )

    else:
        raise ValueError(
            f'Callset must be a Polars table or an iterable '
            f'of callset definition elements (table, name, metadata): Received {type(callset_def)}'
        )

    # Unique callset name
    if callset_name is None:
        callset_name = f'source_{callset_index + 1}'

    callset_name = _get_name(callset_name, name_set)

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
                pl.lit(None)
            )
            .cast(pl.String)
            .fill_null(
                pl.concat_str(pl.lit('var'),
                pl.col('_mg_src_index')),
            )
            .alias('id')
        )
        .filter(
            *pre_filter
        )
    )

    name_set.add(callset_name)

    return CallsetDef(
        table=callset_table,
        name=callset_name,
        metadata=callset_meta,
   )



@lockable
class MergeBase(ABC):
    """Base class for callset intersects."""

    @abstractmethod
    def __call__(
            self,
            callsets: Iterable[CallsetDefInputType],
            retain_index: bool = False,
            pre_filter: Optional[Iterable[pl.Expr]] = None,
            temp_dir: bool | str | Path = False,
    ) -> pl.LazyFrame:
        """
        Intersect callsets.

        :param callsets: Callsets to intersect.
        :param retain_index: If `True`, do not drop an existing "_index" column if it exists.
        :param pre_filter: If set, filter each table with these expressions. Filter is applied
            last (after "_index" is set).
        :param temp_dir: How the underlying pairwise intersect materialises prepared tables before
            its chunked loop. ``False`` (default) keeps tables in memory; ``True`` writes them to
            the system temp directory as parquet; a ``str``/``Path`` writes them to that directory.
            Temp files are always removed on exit.

        :return: A merged callset table.
        """
        ...

    @staticmethod
    def get_intersect_tuples(
            callsets: Iterable[CallsetDefInputType],
            retain_index: bool = False,
            pre_filter: Optional[Iterable[pl.Expr] | pl.Expr] = None,
    ) -> list[CallsetDef]:
        """
        Transform input callset definitions into a normalized list of :class:`CallsetDef`.

        Each returned :class:`CallsetDef` (a named tuple) has three fields:

            1. ``table``: A lazy frame.
            2. ``name``: A name for the source.
            3. ``metadata``: A metadata string for this source, or ``None`` if none was given.

        The source index is not stored on the definition; it is implicitly the position of the
        element in the returned list (0 for the first source, incrementing by 1).

        If a source name is given, the given name is used. If it is not, then a default name is
        generated from the source index (e.g. "source_1" for the first source). Names are
        de-duplicated against names already seen.

        Lazy frames are transformed to add a source index ("_mg_src_index" column) and to ensure a
        variant ID is present ("id" column), filling missing IDs from the index.

        :param callsets: Callsets parameter. May be an iterable of DataFrames, LazyFrames,
            ``(table, name)`` tuples, ``(table, name, metadata)`` tuples, or :class:`CallsetDef`
            objects.
        :param retain_index: If `True`, do not drop an existing "_index" column if it exists.
        :param pre_filter: If set, filter each table with these expressions. Filter is applied
            last (after "_index" is set).

        :return: A list of :class:`CallsetDef`, one per input source, in input order.
        """
        name_set = set()
        callset_table: pl.LazyFrame

        callset_tuple_list: list[CallsetDef] = []

        if isinstance(pre_filter, pl.Expr):
            pre_filter = [pre_filter]
        else:
            pre_filter = list(pre_filter if pre_filter is not None else [])

        if callsets is None:
            raise ValueError('Missing callsets')

        for callset_def in callsets:
            try:
                callset = _get_definition(
                    callset_def=callset_def,
                    callset_index=len(callset_tuple_list),
                    name_set=name_set,
                    retain_index=retain_index,
                    pre_filter=pre_filter,
                )
            except ValueError as e:
                raise ValueError(
                    f'Failed getting callset at index {len(callset_tuple_list)}: {e}'
                ) from e

            # Append
            callset_tuple_list.append(callset)

        return callset_tuple_list


def _get_name(
        name: Optional[Any],
        *args: Container[str]
) -> str:
    """Get a name for a source with a default set for None.

    Gets a name for the variant call input source and de-duplicates names.

    :param name: Input source name or None to choose a default name.
    :param args: Containers with other names to avoid collisions with.

    :returns: A name for this input source.
    """
    if name is None:
        name = f'source.1'

    name = collision_rename(str(name), '.', *args)

    return str(name)
