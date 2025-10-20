"""Intersect stage definitions.

Intersect strategies may include one or more steps, each performing a specific type of intersect task. Stages define
the type and parameters for each of these steps.
"""

from __future__ import annotations

__all__ = [
    'RESERVED_PARAM_NAMES',
    'TYPE_MATCHER',
    'IntersectStage',
    'IntersectStageRo',
    'IntersectStageDistance',
    'IntersectStageExact',
    'IntersectStageMatch',
    'IntersectParamSpec',
]

from abc import ABCMeta
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Optional, Mapping

from ...align import score
from ...seqmatch import MatchScoreModel

if TYPE_CHECKING:
    from .strategy import IntersectStrategy


RESERVED_PARAM_NAMES: frozenset[str] = frozenset({
    'spec_type',
    'arg_list',
    'intersect_strategy',
    'param_spec_list',
    'allow_matcher',
})
"""Reserved parameter names for each specification (ro, szro, etc).

These are fields used by the parameter object and no parameter arguments
are allowed to have the same names.
"""

TYPE_MATCHER: Mapping[str, set[str]] = MappingProxyType({
    'num': {'float', 'int'},
    'num_u': {'float', 'int', 'unlimited'},
    'int_u': {'int', 'unlimited'},
    'float_u': {'float', 'unlimited'},
    'int': {'int'},
    'float': {'float'},
    'list': {'list'},
})
"""Recognized values for the val_type field of specification objects.

Maps val_type strings to sets of types they may match. For example, "num" type
can accept a float or int value, but not unlimited. In many cases, float is
assumed, but an int is allowed to convert to a float so specifications don't
have to require float-like input (e.g. "2" is converted to "2.0").
"""


class IntersectStage(object, metaclass=ABCMeta):
    """A set of positional and/or named parameters describing an intersect stage.

    :ivar spec_type: String describing the specification type (e.g. "nr", etc).
    :ivar arg_list: List of actual arguments used to create this spec. Each element
        is a tuple of (type, value, name) extracted from the parameter
        specification string during parsing.
    :ivar intersect_strategy: Link to the IntersectStrategy object creating this step.
        This is used to query the intersect configuration for global parameters
        and a global matcher.
    :ivar param_spec_list: List of parameter specifications describing the parameters
        this specification can take, their types, ranges, and default values.
    :ivar match_stage: Match stage, if set.
    :ivar allow_match_stage: True if this specification allows a sequence match stage.
    """

    spec_type: str
    arg_list: list[tuple[str, Any, str]]
    intersect_strategy: IntersectStrategy
    param_spec_list: list[IntersectParamSpec]
    match_stage: Optional[IntersectStageMatch]
    allow_match_stage: bool

    def __init__(
            self,
            spec_type: str,
            arg_list: Optional[list[tuple[str, Any, str]]],
            intersect_strategy: IntersectStrategy,
            param_spec_list: Optional[list[IntersectParamSpec]],
            allow_match_stage: bool = True,
    ) -> None:
        """Create an intersect specification describing an intersect step.

        :param spec_type: Specification type name (e.g. "ro", "szro", etc).
        :param arg_list: List of arguments used to configure this parameter set. Set to an
            empty list if None.
        :param intersect_strategy: Link to the IntersectStrategy object creating this step.
        :param param_spec_list: List of parameter specifications. Set to an empty list if None.
        :param allow_match_stage: If True, allow matches to be set on this specification.
        """
        # Check arguments
        if intersect_strategy is None:
            raise ValueError('IntersectStrategy object is None')

        if spec_type is None or not isinstance(spec_type, str):
            raise ValueError(f'Spec type is not a string: {type(spec_type)}')

        spec_type = spec_type.lower().strip()

        if not spec_type or not spec_type.isalpha():
            raise ValueError(f'Spec type is not a valid string: {spec_type}')

        if arg_list is None:
            arg_list = list()

        if param_spec_list is None:
            param_spec_list = list()

        # Save fields
        self.spec_type = spec_type
        self.arg_list = arg_list
        self.intersect_strategy = intersect_strategy
        self.param_spec_list = param_spec_list
        self.allow_match_stage = allow_match_stage

        # Sequence match parameters
        self.match_stage = None

        # Other attributes
        self.col_ref = False
        self.col_alt = False
        self.read_seq = None
        self.vcf_temp = False

        # Check param spec and make a dictionary of indices, and set default values
        param_spec_index = dict()

        for index, param_spec in enumerate(param_spec_list):
            assert param_spec.name not in RESERVED_PARAM_NAMES, (
                'Reserved parameter name cannot be part of a parameter spec: %s' % param_spec.name
            )

            param_spec_index[param_spec.name] = index

            assert param_spec.name not in self.__dict__.keys(), (
                'Specification type %s: Param spec name clashed with a name already defined in the parameter set: %s' %
                (spec_type, param_spec.name)
            )

            self.__dict__[param_spec.name] = param_spec.default

        # States for parameter checking
        named_state = False    # Set to True when a named parameters is found. Only named parameters are allowed after
        matcher_state = False  # Set to True when a matcher is found. No matchers are allowed after.

        # Iterate parameters
        for index in range(len(arg_list)):

            param_tuple = arg_list[index]

            if param_tuple is None:
                continue

            # Check formatting
            # TODO: Consider using match/case
            if len(param_tuple) != 3:
                raise ValueError(
                    f'Specification type {self.spec_type}: '
                    f'Expected 3 parameters in param_tuple: Found {len(param_tuple)}'
                )

            # Check for matcher
            # TODO: Move this under "if param_tuple[0] == 'match':"?
            if matcher_state:
                raise ValueError(
                    f'Specification type {self.spec_type}: '
                    f'Parameter at position {index + 1} follows a matcher '
                    f'(matcher must be the last item in a parameter list)'
                )

            if param_tuple[0] == 'match':
                if not allow_match_stage:
                    raise ValueError(
                        f'Specification type {self.spec_type}: '
                        f'Found matcher at position {index + 1}: '
                        f'No matcher is allowed for this specification type'
                    )

                self.set_match_stage(IntersectStageMatch(param_tuple[1], self.intersect_strategy))

                matcher_state = True

                continue

            # Get spec
            if param_tuple[2] is None:
                if named_state:
                    raise ValueError(
                        f'Specification type {self.spec_type}: '
                        f'Unnamed value {param_tuple[0]} at position {index + 1} follows a named value: '
                        f'Positional parameters must come before named parameters'
                    )

                param_index = index

                if param_index >= len(param_spec_list):
                    raise ValueError(
                        f'Specification type {self.spec_type}: '
                        f'Positional parameter {index + 1} is out of range: '
                        f'Max {len(param_spec_list)}'
                    )

            else:
                param_index = param_spec_index.get(param_tuple[2], None)

                if param_index is None:
                    raise ValueError(
                        f'Specification type {self.spec_type}: Unknown named parameter: {param_tuple[2]}'
                    )

                assert param_index < len(param_spec_list), (
                    'Specification type %s: '
                    'Named parameter at position %d (%s) returns a parameter specification index '
                    'that is out of range: %d: Max %d' % (
                        self.spec_type, index + 1, param_tuple[2], param_index, len(param_spec_list)
                    )
                )

                named_state = True

            param_spec = param_spec_list[param_index]

            # Check and set
            try:
                self.__dict__[param_spec.name] = param_spec.check(param_tuple)

            except (ValueError, TypeError) as e:
                raise ValueError(f'Specification type {self.spec_type}: {e}')

    def set_match_stage(
            self,
            match_stage: Optional[IntersectStageMatch] = None
    ) -> None:
        """Set the match stage on this object.

        :param match_stage: Match stage or None to clear.
        """
        if match_stage is None:
            self.match_stage = None
            return

        if not self.allow_match_stage:
            raise ValueError(
                f'Specification type {self.spec_type}: Match stage is not allowed for this specification type'
            )

        self.match_stage = match_stage

    def get_matcher(self) -> Optional[MatchScoreModel]:
        """Get the matcher object for this intersect spec.

        If a matcher is set for this spec, it is returned. If not, a global
        matcher is returned if set. If neither, None is returned.

        :returns: Matcher object or None.
        """
        if self.match_stage is not None:
            return self.match_stage.matcher

        if self.intersect_strategy.default_match_stage is not None and self.allow_match_stage:
            return self.intersect_strategy.default_match_stage.matcher

        return None

    def __repr__(
            self,
            show_match_stage: bool = True
    ) -> str:
        """Get a string representation for this intersect specification.

        :param show_match_stage: If True, show the match stage parameters.

        :returns: String representation.
        """
        repr_str = f'IntersectStage({self.spec_type}:'

        arg_str = ','.join([
            '{}={}'.format(
                param_spec.name,
                self.__dict__[param_spec.name]
            ) for param_spec in self.param_spec_list
        ])

        if self.match_stage is not None and show_match_stage:
            if arg_str:
                arg_str += ','

            arg_str += self.match_stage.__repr__()

        repr_str += arg_str + ')'

        return repr_str


#
# Concrete classes for intersect strategies
#

class IntersectStageRo(IntersectStage):
    """Param set: Reciprocal overlap (RO).

    Classic RO definition. For insertions, the END position is POS + SVLEN for
    RO computation (does not affect merged calls, END is still POS + 1 for INS
    after intersects).
    """
    def __init__(
            self,
            arg_list: list[tuple[str, Any, str]],
            intersect_strategy: IntersectStrategy,
    ):
        """Initialize a reciprocal-overlap intersect stage.

        :param arg_list: Argument list
        :param intersect_strategy: Intersect strategy.
        """
        super().__init__(
            spec_type='ro',
            arg_list=arg_list,
            intersect_strategy=intersect_strategy,
            param_spec_list=[
                IntersectParamSpec(
                    name='ro', default='0.5', val_type='num',
                    min_val_tuple=(0.0, False), max_val_tuple=(1.0, True)
                ),
                IntersectParamSpec(
                    name='dist', default=None, val_type='int_u',
                    min_val_tuple=(0, True)
                ),
                IntersectParamSpec(
                    name='szdist', default=None, val_type='num_u',
                    min_val_tuple=(0.0, True)
                ),
                IntersectParamSpec(
                    name='szro', default=None, val_type='num',
                    min_val_tuple=(0.0, False), max_val_tuple=(1.0, True)
                )
            ],
            allow_match_stage=True
        )

        return


class IntersectStageDistance(IntersectStage):
    """Intersect by distance where distance is defined as the minimum of start or end position differences.

    Distance is calculated as: `min(abs(pos_l - pos_r), abs(end_l - end_r))`

    At least one of `dist` or `szdist` must be set.

    :param dist: Distance threshold.
    :param szdist: Distance threshold for SVs with size less than `szdist`.
    :param szro: RO threshold for SVs with size less than `szdist`.
    :param alt: Include ALT calls in intersect.
    :param ref: Include REF calls in intersect.
    """

    dist: int
    szdist: Optional[float]
    szro: Optional[float]
    alt: bool
    ref: bool

    def __init__(
            self,
            arg_list,
            intersect_strategy
    ):
        """Initialize the distance intersect stage.

        :param arg_list: Argument list
        :param intersect_strategy: Intersect strategy.
        """
        super().__init__(
            spec_type='distance',
            arg_list=arg_list,
            intersect_strategy=intersect_strategy,
            param_spec_list=[
                IntersectParamSpec(
                    name='dist', default=500, val_type='int',
                    min_val_tuple=(0, True)
                ),
                IntersectParamSpec(
                    name='szdist', default=None, val_type='num_u',
                    min_val_tuple=(0.0, True)
                ),
                IntersectParamSpec(
                    name='szro', default=None, val_type='num_u',
                    min_val_tuple=(0.0, False), max_val_tuple=(1.0, True)
                ),
                IntersectParamSpec(
                    name='alt', default=False, val_type='bool'
                ),
                IntersectParamSpec(
                    name='ref', default=False, val_type='bool'
                )
            ],
            allow_match_stage=True
        )

        if self.dist is None and self.szdist is None:
            raise ValueError(
                f'Specification type {self.spec_type}: '
                f'At least one of "dist" or "szdist" arguments must not be unlimited'
            )

        return


class IntersectStageExact(IntersectStage):
    """Match on exact POS and END.

    :param alt: Match on ALT.
    :param ref: Match on REF.
    """

    alt: bool
    ref: bool

    def __init__(
            self,
            arg_list: list[tuple[str, Any, str]],
            intersect_strategy: IntersectStrategy,
    ):
        """Initialize an exact-match intersect stage.

        :param arg_list: Argument list
        :param intersect_strategy: Intersect strategy.
        """
        super().__init__(
            spec_type='exact',
            arg_list=arg_list,
            intersect_strategy=intersect_strategy,
            param_spec_list=[
                IntersectParamSpec(
                    name='alt', default=False, val_type='bool'
                ),
                IntersectParamSpec(
                    name='ref', default=False, val_type='bool'
                )
            ],
            allow_match_stage=True
        )

        return

# class IntersectStageTruvari(IntersectStage):
#     """Match by calling Truvari."""
#
#     def __init__(
#             self,
#             arg_list: list[tuple[str, Any, str]],
#             intersect_strategy: IntersectStrategy,
#     ):
#         """Initialize a Truvari intersect stage.
#
#         :param arg_list: Argument list
#         :param intersect_strategy: Intersect strategy.
#         """
#         super().__init__(
#             spec_type='truvari',
#             arg_list=arg_list,
#             intersect_strategy=intersect_strategy,
#             param_spec_list=[
#                 IntersectParamSpec(
#                     name='refdist', default=500, val_type='int',
#                     min_val_tuple=(0, True)
#                 ),
#                 IntersectParamSpec(
#                     name='pctseq', default=0.7, val_type='num',
#                     min_val_tuple=(0.0, True)
#                 ),
#                 IntersectParamSpec(
#                     name='pctsize', default=0.7, val_type='num',
#                     min_val_tuple=(0.0, False)
#                 ),
#                 IntersectParamSpec(
#                     name='pctovl', default=0.0, val_type='num',
#                     min_val_tuple=(0.0, True)
#                 )
#             ],
#             allow_match_stage=False
#         )
#
#         self.vcf_temp = True
#
#         return


class IntersectStageMatch(IntersectStage):
    """Parameter set: Matcher specification objects.

    :ivar score: Minimum score proportion for intersects.
    :ivar match: Match score.
    :ivar mismatch: Mismatch score.
    :ivar gap: List of gap scores. Must contain an even number of values where each
        pair of values is a gap-open and gap-extend score. This is a flat list,
        not a list of tuples, although pairs of values are packaged as tuples
        when creating the match object.
    :ivar limit: Maximum number of matches.
    :ivar ksize: Size of k-mer to use.
    :ivar matcher: Matcher object configured with the above parameters.
    """

    score: float
    match: float
    mismatch: float
    gap: list[float]
    limit: int
    ksize: int
    matcher: Optional[MatchScoreModel]

    def __init__(
            self,
            arg_list: list[tuple[str, Any, str]],
            intersect_strategy: IntersectStrategy,
    ):
        """Initialize a match intersect stage.

        :param arg_list: Argument list
        :param intersect_strategy: Intersect strategy.
        """
        super().__init__(
            spec_type='match',
            arg_list=arg_list,
            intersect_strategy=intersect_strategy,
            param_spec_list=[
                IntersectParamSpec(
                    name='score', default=0.8, val_type='num',
                    min_val_tuple=(0.0, False), max_val_tuple=(1.0, True)
                ),
                IntersectParamSpec(
                    name='match', default=score.AFFINE_SCORE_MATCH, val_type='num'
                ),
                IntersectParamSpec(
                    name='mismatch', default=score.AFFINE_SCORE_MISMATCH, val_type='num'
                ),
                IntersectParamSpec(
                    name='gap', default=[val for vals in score.AFFINE_SCORE_GAP for val in vals], val_type='list',
                    list_type='num'
                ),
                IntersectParamSpec(
                    name='limit', default=2000, val_type='int_u',
                    min_val_tuple=(0, True)
                ),
                IntersectParamSpec(
                    name='ksize', default=9, val_type='int',
                    min_val_tuple=(1, True)
                ),
            ],
            allow_match_stage=False
        )

        # Check and set affine gap
        if self.gap is not None:
            if len(self.gap) == 0 or len(self.gap) % 2 != 0:
                raise ValueError(f'Invalid affine gap: Expected an even number of values: {self.gap}')

            gap_tuples = [(self.gap[i], self.gap[i + 1]) for i in range(0, len(self.gap), 2)]
        else:
            gap_tuples = None

        # Create aligner and set matcher fields
        self.matcher = MatchScoreModel(
            match=self.match,
            mismatch=self.mismatch,
            affine_gap=gap_tuples,
            map_limit=self.limit,
            jaccard_kmer=self.ksize
        )

        return


@dataclass(frozen=True)
class IntersectParamSpec(object):
    """Parameter specification for defining legal parameters, their types, and ranges.

    Objects of this type are used to define legal parameters, their types, and ranges
    of values. This class provides functions for checking and enforcing parameter values
    so that configuration objects can use them safely, and error checking code is largely
    consolidated here.

    If the value is a list, then each element inside the list is checked against the list
    type. If successful, then each element inside the list is checked against the minimum
    and maximum values, if set.

    :ivar name: Parameter name.
    :ivar val_type: Type of value (e.g. "num", "int", "list").
    :ivar default: Default value if not specified.
    :ivar min_val_tuple: Tuple of (minimum value, inclusive flag) or None.
    :ivar max_val_tuple: Tuple of (maximum value, inclusive flag) or None.
    :ivar unlimited_val: If "unlimited" is given, then set this value.
    :ivar allow_unlimited: Whether unlimited values are allowed.
    :ivar list_type: If not None, then this is a list of values and each element in
        the list must match this type. If None, then the value is not a list.
    """

    name: str
    default: Any
    val_type: str
    min_val_tuple: Optional[tuple[Any, bool]] = None
    max_val_tuple: Optional[tuple[Any, bool]] = None
    unlimited_val: Optional[Any] = None
    allow_unlimited: Optional[bool] = True
    list_type: Optional[str] = None

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        if self.min_val_tuple is not None:
            if (
                    not isinstance(self.min_val_tuple, tuple) or
                    len(self.min_val_tuple) != 2 or
                    not isinstance(self.min_val_tuple[1], bool)
            ):
                try:
                    tuple_len = len(self.min_val_tuple)
                except TypeError:
                    tuple_len = 'NA'
                raise RuntimeError(
                    f'min_val_tuple: Expected a tuple of two values: '
                    f'{self.min_val_tuple} (type "{type(self.min_val_tuple)}", length {tuple_len})'
                )

        if self.max_val_tuple is not None:
            if (
                    not isinstance(self.max_val_tuple, tuple) or
                    len(self.max_val_tuple) != 2 or
                    not isinstance(self.max_val_tuple[1], bool)
            ):
                try:
                    tuple_len = len(self.max_val_tuple)
                except TypeError:
                    tuple_len = 'NA'
                raise RuntimeError(
                    f'max_val_tuple: Expected a tuple of two values: '
                    f'{self.max_val_tuple} (type "{type(self.max_val_tuple)}", length {tuple_len})'
                )

        if self.val_type is None or not self.val_type.strip():
            raise RuntimeError('ParamSpec.__init__(): val_type may not be None or empty')

        if self.name is None or not self.name.strip():
            raise RuntimeError('ParamSpec.__init__(): name may not be None or empty')

    @property
    def has_min_val(self) -> bool:
        """Determine if a minimum value is set."""
        return self.min_val_tuple is not None

    @property
    def min_val(self) -> Any:
        """Get the minimum value (None if not defined)."""
        return self.min_val_tuple[0] if self.has_min_val else None

    @property
    def min_inclusive(self) -> bool:
        """Determine if minimum value is inclusive in the range, False if exclusive (None if not defined)."""
        return self.min_val_tuple[1] if self.has_min_val else None

    @property
    def has_max_val(self) -> bool:
        """Determine if a maximum value is set."""
        return self.max_val_tuple is not None

    @property
    def max_val(self) -> Any:
        """Get the maximum value (None if not defined)."""
        return self.max_val_tuple[0] if self.has_max_val else None

    @property
    def max_inclusive(self) -> bool:
        """Determine if maximum value is inclusive in the range, False if exclusive (None if not defined)."""
        return self.max_val_tuple[1] if self.has_max_val else None

    def check(
            self,
            param_tuple: tuple
    ) -> Any:
        """Take a parameter tuple in the form (type, value, name) and check it against the specification.

        :param param_tuple: Tuple containing (type, value, name) to check.

        :returns: The value of the object after checking (i.e. the value extracted from `param_tuple`).
        """
        # Check parameter tuple
        if param_tuple is None:
            raise ValueError('Cannot check paramters: None')

        if len(param_tuple) != 3:
            raise ValueError(f'Expected 3 parameters in param_tuple: Found {len(param_tuple)}')

        param_type, param_value, param_name = param_tuple

        # Check type
        if param_type not in TYPE_MATCHER[self.val_type]:
            assert param_type in TYPE_MATCHER.keys(), (
                'Unrecognized type for parameter "%s = %s": %s' % (param_name, param_value, param_type)
            )

            allowed_types = ', '.join(sorted(TYPE_MATCHER[self.val_type]))

            raise TypeError(f'Illegal type for "{param_name} = {param_value}": {param_type}: expected {allowed_types}')

        # Check list elements
        if param_type == 'list' and self.list_type is not None:
            match_types = TYPE_MATCHER.get(self.list_type, set())

            bad_type_list = [
                (i, param_element) for i, param_element in enumerate(param_value)
                if param_element[0] not in match_types
            ]

            if bad_type_list:
                found_types = ','.join(sorted(
                    {bt[0] for i, bt in bad_type_list}
                ))
                indexes = ','.join([str(i) for i, bt in bad_type_list])
                vals = ','.join([str(bt[1]) for i, bt in bad_type_list])

                raise TypeError(
                    f'Illegal type for "{param_name} = {vals}" for list elements at indexs {indexes}: '
                    f'Expected type "{self.list_type}": Found "{found_types}"'
                )

            param_value = [param_element[1] for param_element in param_value]

            # Check bounds
            for param_val_element in param_value:
                if self.min_val is not None:
                    if (
                            param_val_element < self.min_val or
                            (param_val_element == self.min_val and not self.min_inclusive)
                    ):
                        raise ValueError(
                            f'Illegal range for "{param_name} = {param_value}": '
                            f'Minimum allowed value is {self.min_val} '
                            f'({"inclusive" if self.min_inclusive else "exclusive"})'
                        )

                if self.max_val is not None:
                    if (
                            param_val_element > self.max_val or
                            (param_val_element == self.max_val and not self.max_inclusive)
                    ):
                        raise ValueError(
                            f'Illegal range for "{param_name} = {param_value}": '
                            f'Maximum allowed value is {self.max_val} '
                            f'({"inclusive" if self.max_inclusive else "exclusive"})'
                        )

        # Check name and value
        if param_type == 'unlimited':
            if not self.allow_unlimited:
                raise ValueError(
                    f'Illegal value for "{param_name} = unlimited": '
                    f'Unlimited values are not allowed for this parameter in this intersect specification'
                )

            return self.unlimited_val

        if param_value is None:
            raise ValueError(f'Illegal value for "{param_name} = {param_value}"')

        # Check bounds
        if self.has_min_val:
            if param_value < self.min_val or (param_value == self.min_val and not self.min_inclusive):
                raise ValueError(
                    f'Illegal range for "{param_name} = {param_value}": '
                    f'Minimum allowed value is {self.min_val} ({"inclusive" if self.min_inclusive else "exclusive"})'
                )

        if self.has_max_val:
            if param_value > self.max_val or (param_value == self.max_val and not self.max_inclusive):
                raise ValueError(
                    f'Illegal range for "{param_name} = {param_value}": '
                    f'Maximum allowed value is {self.max_val} ({"inclusive" if self.max_inclusive else "exclusive"})'
                )

        return param_value
