"""
Nonredundant (NR) intersect executor.
"""

import itertools

import polars as pl

from ... import seqmatch

from .executor_base import JoinExecutor

class JoinExecutorNrStage():
    def __init__(
            self,
            overlap_ro_min: float=None,
            offset_dist_max: int=None,
            overlap_size_ro_min: float=None,
            offset_prop_max: float=None,
            match_prop_min: float=None,
            match_ref: bool=False,
            match_alt: bool=False,
            col_map: dict[str,str]=None,
            match_score_model: seqmatch.MatchScoreModel=None,
            force_end_ro: bool=False
    ):

        self.overlap_ro_min = overlap_ro_min
        self.offset_dist_max = offset_dist_max
        self.overlap_size_ro_min = overlap_size_ro_min
        self.offset_prop_max = offset_prop_max
        self.match_prop_min = match_prop_min
        self.match_ref = match_ref
        self.match_alt = match_alt
        self.col_map = col_map
        self.match_score_model = match_score_model
        self.force_end_ro = force_end_ro

class IntersectExecutorNr(JoinExecutor):

    def __init__(
            self,
            col_map: dict[str,str]=None
    ):
        super().__init__()

        self.stage_list = list()

        pass

    def _join(
            self,
            df_list: list[pl.LazyFrame],
            source_names: list[str]
    ) -> pl.DataFrame:

        n_df = len(df_list)

        if self.stage_list is None or len(self.stage_list) == 0:
            raise RuntimeError(f'No join stages configured.')

        for indexe_pairs in itertools.combinations(range(n_df), 2):
            index_a = indexe_pairs[0]
            index_b = indexe_pairs[1]

            df_a = df_list[index_a]
            df_b = df_list[index_b]





