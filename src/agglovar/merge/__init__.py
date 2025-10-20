"""Create one merged callset derived from one or more variant tables.

Merging variants is the process of combining variants from one or more tables into one merged
callset. The merged table is a variant table containing key variant columns, such as genomic
location and an ID. Merged tables may include other columns derived from the input tables, such as
lists of fields or statistics derived from them.
"""

__all__ = [
    # 'config',
    # 'om',
    'base',
    'cumulative',
]

# from . import config
# from . import om
from . import base
from . import cumulative
