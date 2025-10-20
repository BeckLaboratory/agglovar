"""Weights for pairwise joins.

Provides a mechanism for weighing and scoring pairwise joins. A weight assigned to each pair
of matched variants can help to prioritize the matches. For example, if variant with ID
"A1" matches "B1" with a weight of 0.5 and "B2" with a weight of 0.75, then the "B2" match
might be chosen over the "B1" match for variant "A1".

Terminology:

    * Weight column: One column used to compute weights, each with their own weight.
    * Weight element: Computes a weight for each row using by multiplying each column by its
        assigned weight and summing.
    * Weight strategy: One or more weight elements, each computing their own weight.

Each weight element extracts a set of columns, multiplies each by a specified weight, and sums the
results producing one weight per row of the join table.

For example, one weight strategy might have these column weights:

====== ======
Column Weight
====== ======
    ro 0.5
 match 0.5
====== ======

In this case, the weight for each row is computed as (0.5 * ro) + (0.5 * match). Both "ro" and
"match" have a maximum value of 1, so the total weighted sum is between 0.0 and 1.0.

The columns attributes are:

    * col: Column name.
    * weight: Column weight.
    * scale: Column scale.
    * missing: Missing value.

The column scale is used to rescale the column to a range between 0.0 and 1.0. It can take
several forms:

    * float: Maximum value. Minimun value is 0.0.
    * (float, float): Minimum and maximum values.
    * (float, float, bool): Minimum and maximum values, and a flag for inverting the range
        after scaling.

Values are first clipped to within the [min, max] range, then scaled
(i.e. (value - min) / (max - min)). If the range is inverted, then the scaled value is inverted
(i.e. 1 - scaled_value). Inverting values is only support if both min and max are defined. Finally,
the scaled value is multiplied by the column weight. For columns that are not scaled, no
assumptions are made about their range, they are left as-is and multiplied by the weight.

No assumptions are made about weights, they could be negative or greater than 1. A good design
should typically keep weights within [0.0, 1.0], although Agglovar allows designers to make
choices about weights the developers might not have anticipated. Use this power with care.

Agglovar allows flexible weight strategies where multiple weight elements are used. For example,
it might make sense to compute another set of weights on "size_ro" and "offset_prop" for smaller
variants.

Any row with a null value in a weight column causes the weighted sum to become null. By default,
this causes that weight element to be skipped. For example, assume that "match" might not have
been computed, so the column is null for all values. A weight strategy can use multiple elements,
one with a "match" weight and one without. In this case, the weight element computed on "match"
will be ignored and the weight will automatically be computed on other weight elements.

When multiple match elements are used in a strategy, the maximum computed non-null weight is used.
If all weights are null for a row, then it gets 0.0. This strategy can be changed to accept the
first non-null weight instead of the maximum. Null behavior can also be set per column and per
weight element by specifying a value for "missing", to fill in missing values. For example,
"missing=0.0" if null values in a column should not cause the whole weight element to be discarded.
The weight strategy also has a missing value set to 0.0 by default, so if all weights compute
null, the result is 0.0. Set missing=None for the strategy to produce nulls for rows where there
were no computable weights.

A weight strategy can be specified in a number of ways. Each level of the strategy has an object,
:class:`WeightColumn`, :class:`WeightElement`, and :class:`WeightStrategy`. Each one can take
an iterator (usually a tuple or a list) or a dictionary. Each of these is fed to the clas
constructor.

For example, a weight column could be specified as "('ro', 0.5)" or "{'col'='ro': 'weight'=0.5}".

A weight element could be specified as a list of columns, such as "(('ro', 0.2), ('match', 0.5))"
or as a dictionary where with constructor arguments, which would be required to set "missing", such
as "{'columns': (('ro', 0.2), ('match': 0.5)), 'missing': 0.0}"

An example for constructing using tuples:

.. code-block:: python

    WeightStrategy(
        *(
            (
                ('ro', 0.2),
                ('size_ro', 0.2),
                ('offset_prop', 0.1, (0.0, 2.0, True)),
                ('match_prop', 0.5),
            ), (
                ('ro', 0.4, None),
                ('size_ro', 0.4, None),
                ('offset_prop', 0.2, (0.0, 2.0, True)),
            ),
        )
    )

And an equivalent form using a mix of dicts and tuples, but with some "missing" values set for
illustration purposes:

.. code-block:: python

    WeightStrategy(
        *(
            {
                'columns': (
                    ('ro', 0.2),
                    ('size_ro', 0.2),
                    ('offset_prop', 0.1, (0.0, 2.0, True)),
                    ('match_prop', 0.5),
                ),
                'missing': 0.0,
            }, (
                {'col': 'ro', 'weight': 0.4},
                ('size_ro', 0.4, None),
                ('offset_prop', 0.2, (0.0, 2.0, True)),
            ),
        ),
        priority='MAX'
    )
"""

__all__ = [
    'ElementPriority',
    'WeightColumn',
    'WeightElement',
    'WeightStrategy',
    'DEFAULT_WEIGHT_STRATEGY',
]

from collections.abc import Container
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

import polars as pl

from ..meta.decorators import immutable
from ..meta.descriptors import CheckedBool, BoundedFloat, CheckedString

class ElementPriority(Enum):
    """Prioritize multiple matches.

    When multiple match elements are used (each with their own columns and weights), then this
    enum describes which weight should be chosen.

    When MAX is used, take the maximum weight of all the match elements.

    When FIRST is used, take the first non-null weight of all the match elements.

    Null values appear in the weight column if it is computed on null values in the columns
    or a whole column in the weight calculation is missing. By default, null is carried through
    to the weight element, but this can be changed by the columns or the weight element itself.
    """
    MAX = 'MAX'
    FIRST = 'FIRST'

@immutable
class WeightColumn:
    """An element representing the weight for one column.

    :ivar col: Column name
    :ivar weight: Weight of the column
    :ivar max_value: Maximum value of the column (if None, no max value is used).
    :ivar missing: Use this value if the column is missing or any columns contain a null value.
    """
    col: str = CheckedString(match=r'[^^*$]+')
    weight: float = BoundedFloat()
    min_value: Optional[float] = BoundedFloat()
    max_value: Optional[float] = BoundedFloat()
    invert: bool = CheckedBool()
    missing: Optional[float] = CheckedBool()

    def __init__(
            self,
            col: str,
            weight: float,
            scale: Optional[
                float
                | tuple[Optional[float], Optional[float]]
                | tuple[Optional[float], Optional[float], bool]
            ] = None,
            missing: Optional[float] = None,
    ):
        """Set weight fields."""
        if col is None or not (col := str(col).strip()):
            raise ValueError('Column name is required')

        if weight is None:
            raise ValueError('Weight missing')

        # Set scaling
        if scale is not None:
            if isinstance(scale, tuple):
                if len(scale) == 2:
                    self.min_value = scale[0]
                    self.max_value = scale[1]
                    self.invert = False
                elif len(scale) == 1:
                    self.min_value = 0.0
                    self.max_value = scale[0]
                    self.invert = False
                elif len(scale) == 3:
                    self.min_value = scale[0]
                    self.max_value = scale[1]
                    self.invert = bool(scale[2])
                else:
                    raise ValueError('Scaling must be a float or a tuple of length 1, 2, or 3')
            else:
                self.min_value = 0.0
                self.max_value = scale
                self.invert = False

            if (
                    any([self.invert, self.min_value is not None, self.max_value is not None])
                    and not all([self.min_value is not None, self.max_value is not None])
            ):
                raise ValueError(
                    f'Clip incompletely specified: min and max cannot be missing when either '
                    f'is set or inverted is true: '
                    f'min={self.min_value}, max={self.max_value}, invert={self.invert}'
                )

            if self.min_value >= self.max_value:
                raise ValueError(
                    f'Minimum clip value must be less than the maximum: '
                    f'{self.min_value:,g} >= {self.max_value:,g}'
                )

        self.col = col
        self.weight = float(weight)
        self.missing = float(missing) if missing is not None else None

        if self.max_value == 0.0:
            raise ValueError('Max value cannot be 0.0 (division by zero errors)')

    def __eq__(self, other: object) -> bool:
        """Determine if this weight column is equal to another."""
        if not isinstance(other, WeightColumn):
            return False

        return (
            self.col == other.col
            and self.weight == other.weight
            and self.max_value == other.max_value
            and self.missing == other.missing
        )

    @property
    def expr(self) -> pl.Expr:
        col = pl.coalesce(pl.col(f'^{self.col}$'), pl.lit(self.missing)).cast(pl.Float32)

        if self.max_value is not None:
            assert self.min_value is not None

            col = (
                (
                    col.clip(self.min_value, self.max_value)
                    - pl.lit(self.min_value)
                )
                / (self.max_value - self.min_value)
            )

        if self.invert:
            col = pl.lit(1.0) - col

        return col * pl.lit(self.weight)

    def __repr__(self) -> str:
        return f'WeightColumn({self.col!r}, {self.weight!r}, {self.max_value!r}, {self.missing!r})'


@immutable
class WeightElement(Container[WeightColumn]):
    """Represents one strategy for computing weights.

    :ivar columns: Columns and their weights.
    :ivar missing: Use this value if the weighted columns sum to null.
    """
    columns: tuple[WeightColumn, ...]
    missing: Optional[float] = None

    def __init__(
            self,
            *columns: WeightColumn | Mapping[str, Any] | Iterable[Any],
            missing: Optional[float] = None,
    ):
        """Initialize a weight strategy across one or more columns.

        May be constructed with a variety of approaches:

            - An iterable of :class:`WeightColumn` objects already constructed.
            - An iterable of mappings with keys corresponding to WeightColumn constructor
                parameters.
            - An iterable of iterables with corresponding to WeightColumn constructor parameters
                in order.

        :param columns: An iterable of column definitions.
        """
        if columns is None:
            raise ValueError('Missing "columns"')

        column_list = []

        for column in columns:
            if isinstance(column, WeightColumn):
                column_list.append(column)
            elif isinstance(column, Mapping):
                column_list.append(WeightColumn(**column))
            elif isinstance(column, Iterable):
                column_list.append(WeightColumn(*tuple(column)))
            else:
                raise ValueError(f'Unrecognized column type {type(column)!r}: {column!r}')

        if len(column_list) == 0:
            raise ValueError('No columns provided')

        self.columns = tuple(column_list)
        self.missing = float(missing) if missing is not None else None

    @property
    def expr(self) -> pl.Expr:
        """Get an expression for computing weights."""
        sum_expr = pl.sum_horizontal(
            *[column.expr for column in self.columns],
            ignore_nulls=False,
        )

        if self.missing is not None:
            sum_expr = sum_expr.fill_null(self.missing)

        return sum_expr

    @property
    def cols(self) -> tuple[str, ...]:
        """Get the columns used by this element."""
        return tuple(column.col for column in self.columns)

    @property
    def max_weight(self) -> float:
        """Get the maximum weight sum this element can generate."""
        return sum(column.weight if column.weight > 0.0 else 0.0 for column in self.columns)

    @property
    def min_weight(self) -> float:
        """Get the minimum weight sum this element can generate."""
        return sum(column.weight if column.weight < 0.0 else 0.0 for column in self.columns)

    def __contains__(self, item: object) -> bool:
        """Determines if a column is in this element."""
        return item in self.columns

    def __eq__(self, other: object) -> bool:
        """Determine if this weight element is equal to another."""
        if not isinstance(other, WeightElement):
            return False

        return self.columns == other.columns and self.missing == other.missing

    def __repr__(self) -> str:
        return f'WeightElement(columns={self.columns!r}, missing={self.missing!r})'

@immutable
class WeightStrategy(Container[WeightElement]):
    """Represents one strategy for computing weights."""
    elements: tuple[WeightElement, ...]
    missing: Optional[float]
    priority: ElementPriority

    def __init__(
            self,
            *elements: Iterable[WeightElement] | Mapping[str, Any] | Iterable[Any],
            missing: Optional[float] = 0.0,
            priority: Optional[ElementPriority | str] = ElementPriority.MAX,
    ):
        """Initialize a weight strategy across one or more columns."""
        if elements is None:
            raise ValueError('Missing "columns"')

        if priority is None:
            priority = ElementPriority.MAX
        else:
            if not isinstance(priority, ElementPriority):
                try:
                    priority = ElementPriority(str(priority).upper())
                except ValueError:
                    raise ValueError(f'Unrecognized element priority {priority!r}')

        element_list = []

        for element in elements:
            if isinstance(element, WeightElement):
                element_list.append(element)
            elif isinstance(element, Mapping):
                if 'columns' not in element:
                    raise ValueError('Missing "columns"')

                element_list.append(WeightElement(
                    *element['columns'],
                    **{k: v for k, v in element.items() if k != 'columns'}
                ))
            elif isinstance(element, Iterable):
                element_list.append(WeightElement(*tuple(element)))
            else:
                raise ValueError(f'Unrecognized element type {type(element)!r}: {element!r}')

        if len(element_list) == 0:
            raise ValueError('No elements provided')

        self.elements = tuple(element_list)
        self.missing = float(missing) if missing is not None else None
        self.priority = priority

    @property
    def expr(self) -> pl.Expr:
        """Get and expression for computing weights."""
        if self.priority is ElementPriority.MAX:
            weight_expr = pl.max_horizontal(
                *[element.expr for element in self.elements]
            )
        elif self.priority is ElementPriority.FIRST:
            weight_expr = pl.coalesnce(
                *[element.expr for element in self.elements]
            )
        else:
            raise ValueError(f'Unrecognized priority: {self.priority}')

        return weight_expr

    def __contains__(self, item: object) -> bool:
        """Determine if a weight element is in this strategy."""
        return item in self.elements

    def __eq__(self, other: object) -> bool:
        """Determine if this weight strategy is equal to another."""
        if not isinstance(other, WeightStrategy):
            return False

        return (
            self.elements == other.elements
            and self.missing == other.missing
        )

    def __repr__(self) -> str:
        """String representation."""
        return f'WeightStrategy(elements={self.elements!r}, missing={self.missing!r}, priority={self.priority!r})'


DEFAULT_WEIGHT_STRATEGY: WeightStrategy = WeightStrategy(
    *(
        (
            ('ro', 0.2),
            ('size_ro', 0.2),
            ('offset_prop', 0.1, (0.0, 2.0, True)),
            ('match_prop', 0.5),
        ), (
            ('ro', 0.4, None),
            ('size_ro', 0.4, None),
            ('offset_prop', 0.2, (0.0, 2.0, True)),
        ),
    )
)
"""Default strategy for computing weights."""