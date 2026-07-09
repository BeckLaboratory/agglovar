"""Base class for pairwise intersect strategies.

Defines an interface and implements common functionality for pairwise intersect strategies.
"""

__all__ = [
    'PairwiseJoin',
]

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Iterator,
    Iterable,
)
from concurrent.futures import Executor
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Optional

import polars as pl

from ..meta.descriptors import CheckedObject

from .weights import (
    WeightStrategy,
    DEFAULT_WEIGHT_STRATEGY,
)

logger = logging.getLogger(__name__)


class PairwiseJoin(ABC):
    """Base class for pairwise intersection classes."""

    _weight_strategy: WeightStrategy = CheckedObject(default=DEFAULT_WEIGHT_STRATEGY)

    def __init__(
            self,
            weight_strategy: WeightStrategy = DEFAULT_WEIGHT_STRATEGY,
    ):
        """Create a join object.

        :param weight_strategy: Weight strategy to use for this join or default weight strategy if
            not overridden by some other mechanism (i.e. default for multi-joins).
        """
        if weight_strategy is not None:
            self._weight_strategy = weight_strategy

    @abstractmethod
    def join_iter(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame,
            retain_index: bool = False,
            temp_dir: bool | str | Path = False,
            match_pool: Optional[Executor] = None,
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
        :param match_pool: Optional worker pool for parallel sequence matching (from
            :meth:`make_match_pool`), or ``None`` to score serially. Ignored by strategies that do
            no sequence matching.

        :yields: A LazyFrame for each chunk.
        """
        raise NotImplementedError

    def join(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame,
            retain_index: bool = False,
            temp_dir: bool | str | Path = False,
            match_pool: Optional[Executor] = None,
    ) -> pl.LazyFrame:
        """Find all pairs of variants in two sources that meet a set of criteria.

        This is a convenience function that calls join_iter() and concatenates the results.

        :param df_a: Table A.
        :param df_b: Table B.
        :param retain_index: If True, do not drop an existing "_index" column in callset tables
            if they exist.
        :param temp_dir: See :meth:`join_iter`.
        :param match_pool: See :meth:`join_iter`.

        :returns: A join table.
        """
        return pl.concat(
            self.join_iter(
                df_a, df_b, retain_index=retain_index, temp_dir=temp_dir, match_pool=match_pool,
            )
        )

    @contextmanager
    def make_match_pool(self, match_threads: Optional[int] = None) -> Iterator[Optional[Executor]]:
        """Context manager yielding a worker pool for parallel sequence matching, or ``None``.

        The default is a no-op that always yields ``None`` (serial scoring): strategies that do no
        sequence matching (or do not parallelize it) ignore ``match_threads``. Subclasses that
        parallelize sequence matching override this to create and tear down a worker pool the caller
        passes to :meth:`join`/:meth:`join_iter` via ``match_pool``. See
        :meth:`agglovar.pairwise.overlap.PairwiseOverlap.make_match_pool`.

        :param match_threads: Requested worker count, or ``None`` to keep scoring serial.

        :yields: A worker pool, or ``None`` when scoring serially.
        """
        yield None

    @property
    @abstractmethod
    def required_cols(self) -> set[str]:
        """The minimum set of columns that must be present in input tables."""
        raise NotImplementedError

    def check_required_cols(
            self,
            df: pl.LazyFrame | pl.DataFrame | Iterable[str],
            raise_exception: bool = False,
    ) -> set[str]:
        """Check if a table has the expected columns.

        :param df: Table to check.
        :param raise_exception: If True, raise an exception if any expected columns are missing.

        :returns: A set of missing columns.

        :raises ValueError: If any expected columns are missing and `raise_exception` is True.
        """
        if isinstance(df, pl.LazyFrame):
            col_names = set(df.collect_schema().names())
        elif isinstance(df, pl.DataFrame):
            col_names = set(df.columns)
        else:
            col_names = set(df)

        missing_cols = self.required_cols - col_names

        if missing_cols and raise_exception:
            raise ValueError(f'Missing columns: "{", ".join(sorted(missing_cols))}"')

        return missing_cols

    @property
    def reserved_cols(self) -> set[str]:
        """A set of columns that are reserved for internal use and must not be present in input tables."""
        return set()

    @property
    def weight_strategy(self) -> WeightStrategy:
        """Weight strategy to use for this join."""
        return self._weight_strategy

    def check_reserved_cols(
            self,
            df: pl.LazyFrame | pl.DataFrame | Iterable[str],
            raise_exception: bool = False,
    ) -> set[str]:
        """Check if a table has reserved columns.

        :param df: Table to check.
        :param raise_exception: If True, raise an exception if any reserved columns are found.

        :returns: A set of reserved columns found in the table.

        :raises ValueError: If any reserved columns are found and `raise_exception` is True.
        """
        if isinstance(df, pl.LazyFrame):
            col_names = set(df.collect_schema().names())
        elif isinstance(df, pl.DataFrame):
            col_names = set(df.columns)
        else:
            col_names = set(df)

        reserved_cols = self.reserved_cols & col_names

        if reserved_cols and raise_exception:
            raise ValueError(f'Found reserved columns: "{", ".join(sorted(reserved_cols))}"')

        return reserved_cols
