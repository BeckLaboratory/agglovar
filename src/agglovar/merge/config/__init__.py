"""Intersect configuration strategies.

Intersect configuration objects describe a full intersect strategy, which may include one or more stages.

For full documentation on using the configuration API including definition string syntax, see
`README.md <https://github.com/BeckLaboratory/agglovar/blob/main/README.md>`__.

This module is organized into several submodules:

* strategy: Contains the top-level configuration object and the only object most code outside the configuration module
  is likely to use directly.
* parser: Contains the lexer and parser for parsing intersect configuration strings.
* stage: Contains objects defining intersect stages. These stages are part of the intersect strategy.
* param: Provides a framework for handling parameters and is used by stages to check and set parameters based on
  expected values, types, and ranges.

Intersect strategies may be configured directly or using a configuration string. The "parser" submodule translates the
configuration string to an AST structure that the strategy object can use to configure itself.

Most code outside the config module will create strategy objects directly with
``agglovar.join.config.strategy.IntersectStrategy()``.

A simple strategy string might look like "nr::ro(0.5)" to specify a non-redundant intersect strategy with a single stage
performing a 50% reciprocal-overlap. More complex strategies are possible, such as
"nr::ro(0.5):szro(0.5,200):match(0.8)".
"""

__all__ = [
    'parser',
    'strategy',
    'stage',
]

from . import parser
from . import strategy
from . import stage
