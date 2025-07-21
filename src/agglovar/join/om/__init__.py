"""
Intersect executors are objects that run the intersect process. Each intersect strategy has a specific runner that is
discoverable using a standard nameing convention. For example, the `nr` strategy is executed by
`agglovar.intersect.executor.IntersectExecutorNr`. New executors can be added without modifying utility functions
that load them.
"""

from .executor_base import *
from .executor_nr import *
