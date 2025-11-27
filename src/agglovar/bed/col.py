"""Utilities for managing columns."""

__all__ = [
    'CoordColExpr',
    'CoordCol',
    'get_coord_cols',
    'standardize',
]

from collections.abc import Iterable, Container
from dataclasses import dataclass
import re
from typing import Optional, Self, Any

import polars as pl

@dataclass(frozen=True)
class CoordColExpr(Iterable[pl.Expr]):
    """Represents coordinate column expressions.

    :ivar chrom: Chromosome column expression.
    :ivar pos: Position column expression.
    :ivar end: End column expression.
    """
    chrom: pl.Expr
    pos: pl.Expr
    end: pl.Expr

    def col_names(self):
        """Get the resulting column name for each column expression.

        :return: Column names.
        """
        return CoordCol(self.chrom_name, self.pos_name, self.end_name)

    @property
    def chrom_name(self) -> str:
        """Output column name of the chromosome expression."""
        return self.chrom.meta.output_name()

    @property
    def pos_name(self) -> str:
        """Output column name of the pos expression."""
        return self.pos.meta.output_name()

    @property
    def end_name(self) -> str:
        """Output column name of the end expression."""
        return self.end.meta.output_name()

    def __iter__(self):
        return iter((self.chrom, self.pos, self.end))

    def __repr__(self):
        return f"CoordColExpr({self.chrom!r}, {self.pos!r}, {self.end!r})"

@dataclass(frozen=True, order=True)
class CoordCol(Iterable[str], Container[str]):
    """Represents coordinate columns to use.

    :ivar chrom: Chromosome column name.
    :ivar pos: Position column name.
    :ivar end: End column name.
    """
    chrom: str
    pos: str
    end: str

    def exprs(
            self,
            alias: Optional[Self | str | tuple[str, str, str]] = None,
            suffix: str = None
    ) -> CoordColExpr:
        """Get column expressions.

        :param alias: Alias to these columns.
        :param suffix: Suffix to add to the column names.

        :return: A tuple of column expressions.
        """
        if alias is None and suffix is None:
            return CoordColExpr(*(pl.col(col) for col in self))

        if alias is None:
            alias = self

        suffix = str(suffix).strip() if suffix is not None else ''

        return CoordColExpr(
            *(
                pl.col(col).alias(alias + suffix)
                    for col, alias in zip(self, alias)
            )
        )

    def __iter__(self):
        return iter((self.chrom, self.pos, self.end))

    def __contains__(self, o: Optional[Any]):
        return o in (self.chrom, self.pos, self.end)

    def __repr__(self):
        return f"CoordCol({self.chrom!r}, {self.pos!r}, {self.end!r})"

COL_CHROM: CoordCol = CoordCol('chrom', 'pos', 'end')
"""Standard chromosome columns."""

COL_QRY: CoordCol = CoordCol('qry_id', 'qry_pos', 'qry_end')
"""Standard query columns."""

def get_coord_cols(
        col_names: Optional[CoordCol | str | Iterable[str]] = None,
) -> CoordCol:
    """Get columns representing coordinates.

    Columns may have different names, such as "chrom", "pos", "end" for references or "qry_id", "qry_pos", and
    "qry_end" for queries. Returns a tuple of three expressions to be used in Polars selects, one for each of these
    three columns, to select the correct column from the input table and alias them to "chrom", "pos", and "end"
    with a set suffix.

    :param col_names: Column names. Can be an iterable of three strings, or a keyword in "ref" or "qry". None is
        equivalent to "ref".

    :return: An object with column names.
    """

    if isinstance(col_names, CoordCol):
        return col_names

    # Get column names
    if col_names is None or col_names == 'ref':
        return CoordCol('chrom', 'pos', 'end')

    elif col_names == 'qry':
        return CoordCol('qry_id', 'qry_pos', 'qry_end')

    else:
        col_names_tuple = tuple((
            col.strip() if col is not None else None
                for col in col_names
        ))

        if len(col_names_tuple) != 3:
            raise ValueError(f'Invalid cols: Expected 3 elements, found {len(col_names_tuple)}: {col_names:r}')

        if any(col is None or col == '' for col in col_names_tuple):
            raise ValueError(f'Columns is missing values: {col_names:r}')

        return CoordCol(*col_names_tuple)


def standardize(col: str) -> str:
    """Standardize column names.

    Standard column names are lower-case with alphanumeric, underscore, and dot characters only. Leading and trailing
    whitespace is stripped, and runs of spaces are replaced with a single underscore. All other characters are
    removed.

    :param col: Column name.

    :return: Standard column names.
    """
    new_col = col.strip().lower()
    new_col = re.sub(r'\s+', '_', new_col)
    new_col = re.sub(r'[^a-zA-Z0-9_.]', '', new_col)

    if not new_col:
        raise ValueError(f'Empty col after standardization: {col}')

    return new_col


def make_unique_col(
        col: str,
        *args
):
    """Make a column name unique by appending a number if it matches any value in some container.

    :param col: Column name.
    :param args: Containers to check for matches.

    :return: Unique column name.
    """

    new_col = col
    i = 0

    while any(new_col in arg for arg in args):
        i += 1
        new_col = f'{col}_{i}'


    return new_col
