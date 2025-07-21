"""
Utility functions for intersect runners.
"""

from . import config
from . import executor

def load_executor(
        intersect_runner_config: config.IntersectConfig
) -> executor.IntersectExecutor:

    if intersect_runner_config is None:
        raise ValueError('intersect_runner_config: None')

    return executor.IntersectExecutorNr(intersect_runner_config)
