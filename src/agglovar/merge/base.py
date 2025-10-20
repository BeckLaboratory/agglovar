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
    | tuple[
        pl.DataFrame | pl.LazyFrame, Optional[Any]
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
            callsets: Iterable[CallsetDefType]
    ) -> pl.LazyFrame:
        """
        Intersect callsets.

        :param callsets: Callsets to intersect.

        :return: A merged callset table.
        """
        ...

    @staticmethod
    def get_intersect_tuples(
            callsets: Iterable[CallsetDefType]
    ) -> list[tuple[pl.LazyFrame, str, int]]:

        name_set = set()
        callset_table: pl.LazyFrame
        callset_name_pre: Any

        callset_tuple_list: list[tuple[pl.LazyFrame, str, int]] = []

        i = 0

        """Get intersect tuples."""
        for callset in callsets:
            if isinstance(callset, tuple):
                if not len(callset) == 2:
                    raise ValueError(
                        f'Callset at index {i} tuple must have exactly 2 elements: {callset}'
                    )

                callset_table = callset[0]
                callset_name = _get_name(callset[1], i, name_set)

            else:
                callset_table = callset
                callset_name = _get_name(None, i, name_set)


            if isinstance(callset_table, pl.DataFrame):
                callset_table = callset_table.lazy()
            elif not isinstance(callset_table, pl.LazyFrame):
                raise TypeError(
                    f'Callset at index {i} must be a DataFrame or LazyFrame, got {type(callset_table)}'
                )

            callset_tuple_list.append((callset_table, callset_name, i))
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
