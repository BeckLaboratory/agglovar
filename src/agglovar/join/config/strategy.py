"""
The intersect strategy contains a full intersect specification including the strategy type and list of stages.
"""

#
# IntersectConfig (full configuration specification)
#

import re

from ... import seqmatch

from . import parser
from . import stage


class IntersectStrategy(object):
    strategy_type: str | None
    stage_list: list[stage.IntersectStage]
    default_match_stage: stage.IntersectStageMatch|None

    """
    A configured intersect strategy. Takes the AST output from the parser and generates a configuration object.
    
    :ivar strategy_type: Intersect strategy type. If None, then the configuration object is in an undefined or uninitialized
        state and must not be used until initialized.
    :ivar stage_list: List of intersect stages.
    :ivar default_match_stage: Default match stage object applied to all stages (if there are none).
    """

    def __init__(
            self,
            intersect_strategy: str|None
        ):
        """
        Parse intersect configuration parameters and create a configuration object.

        :param intersect_strategy: Configuration parameter string to initialize this object or None to create an empty
            configuration object.

        :raises ValueError: If an error occurs while parsing the configuration string.
        """

        self.strategy_type = None
        self.stage_list = list()
        self.default_match_stage = None

        # Set defaults (clear) and then initialize by the configuration string.
        if intersect_strategy is not None:
            self.set_config_string(intersect_strategy)

        return

    def clear(self) -> None:
        """
        Clear all merge stages and global options.
        """

        self.strategy_type = None
        self.stage_list = list()
        self.default_match_stage = None

    def set_strategy_type(
            self,
            strategy_type: str
    ) -> None:
        """
        Set the strategy type.

        :param strategy_type: A string strategy type to use, such as "nr" (for non-redundant strategy). Must not be
            missing or empty, must contain only alphanumeric characters and underscores, and may not start with a
            digit.

        :raises ValueError: If strategy type is missing or empty, if the strategy string is not alphanumeric, or if
            the strategy string starts with a digit.
        """

        if strategy_type is None or (strategy_type := strategy_type.strip().lower()) == '':
            raise ValueError('Strategy type is missing or empty')

        if not re.search(r'^\w(\w|\d)*$', strategy_type):
            raise ValueError(f'Strategy type must be alphanumeric and may not start with a digit: {strategy_type}')

        self.strategy_type = strategy_type

    def set_config_string(
            self,
            config_string: str,
            clear: bool=True
    ) -> None:
        """
        Configure this intersect strategy using a configuration string. The string is parsed according to an intersect
        definition grammar and used to configure this object.

        :param config_string: Configuration string. Must not be Null or empty.
        :param clear: If true, clear any existing configuration. If false and an existing configuration exists, then
            global options are overridden and new intersect stages are added to existing ones already in this
            configuration object.

        raises ValueError: If configuration fails for any reason. This may or may not leave the configuration object in
            an undefined state. If the "strategy" attribute is None after calling this method, then the configuration
            object is in an undefined state and must not be used; otherwise, a failure occurred before the object was
            altered and the object state is unchanged by this method call.
        """

        # Check arguments
        if config_string is None or (config_string := config_string.strip()) == '':
            raise ValueError('Cannot configure intersect strategy with an empty or missing configuration string.')

        # Parse configuration string
        intersect_config_parser = parser.IntersectConfigParser()

        try:
            parser_ast = intersect_config_parser.parse(config_string)
        except Exception as e:
            raise ValueError(f'Error parsing intersect configuration string "{config_string}": {e}')

        # Check arguments
        if parser_ast is None:
            raise ValueError('Unknown error parsing intersect configuration string: parser_ast is None')

        missing_keys = [val for val in ('strategy', 'spec_list') if val not in parser_ast.keys()]

        if missing_keys:
            raise ValueError('Incomplete configuration string: Missing {} top-level key(s): {}'.format(
                len(missing_keys),
                ', '.join(missing_keys)
            ))

        # Clear
        if clear:
            self.clear()

        # Add intersect specifications
        self.strategy_type = parser_ast['strategy']

        for index, stage_ast in enumerate(parser_ast['spec_list']):

            # Check keys
            missing_keys = [val for val in ('type', 'val_list') if val not in stage_ast.keys()]

            if missing_keys:
                raise ValueError('Incomplete intersect specification: Missing {} top-level key(s) at position {}: {}'.format(
                    len(missing_keys),
                    index + 1,
                    ', '.join(missing_keys)
                ))

            # Process specification type
            intersect_stage = None

            if self.strategy_type == 'nr':

                if stage_ast['type'] == 'ro':
                    intersect_stage = stage.IntersectStageRo(stage_ast['val_list'], self)

                elif stage_ast['type'] == 'distance':
                    intersect_stage = stage.IntersectStageDistance(stage_ast['val_list'], self)

                elif stage_ast['type'] == 'exact':
                    intersect_stage = stage.IntersectStageExact(stage_ast['val_list'], self)

                # elif stage_ast['type'] == 'truvari':
                #     intersect_stage = stage.IntersectStageTruvari(stage_ast['val_list'], self)

                elif stage_ast['type'] == 'match':
                    self.default_match_stage = stage.IntersectStageMatch(stage_ast['val_list'], self)

                else:
                    raise ValueError(f'IntersectStrategy {self.strategy_type}: Stage specification type at {index + 1} is unknown: {stage_ast["type"]}')

            else:
                raise ValueError(f'IntersectStrategy: Unrecognized strategy {self.strategy_type}')

            if intersect_stage is not None:
                self.stage_list.append(intersect_stage)

        # Set strategy - indicates the object is in a valid state
        self.set_strategy_type(parser_ast['strategy'])

    def has_vcf_temp(self) -> bool:
        """
        Determine if any merge stage uses a VCF temporary file.

        :return: True if any merge stage uses a VCF temporary file
        """
        return any([intersect_stage.vcf_temp for intersect_stage in self.stage_list])

    def is_any_match(self) -> bool:
        """
        Return True if any merge spec has a match stage.

        :return: True if any merge spec has a match_stage.
        """
        return any([intersect_stage.get_matcher() is not None for intersect_stage in self.stage_list])

    def __repr__(self, pretty=False):

        repr_str = 'IntersectStrategy(' + self.strategy_type

        # Strings for each specification
        spec_str = [
            intersect_stage.__repr__() for intersect_stage in self.stage_list
        ]

        # Double colon separates strategy from specifications
        if len(spec_str) > 0:
            if pretty:
                repr_str += '\n'
            else:
                repr_str += '::'

        # Add stages
        if pretty:
            repr_str += ''.join(['    {}\n'.format(val) for val in spec_str])
        else:
            repr_str += ':'.join(spec_str)

        # Add global default matcher
        if self.default_match_stage is not None:
            if pretty:
                repr_str += '\n' if len(spec_str) == 0 else ''
                repr_str += '    ' + self.default_match_stage.__repr__() + '\n'
            else:
                if len(spec_str) > 0:
                    repr_str += ':'

                repr_str += self.default_match_stage.__repr__()

        repr_str += ')'

        return repr_str
