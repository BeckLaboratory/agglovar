"""
Nonredundant (NR) intersect executor.
"""

import itertools

import polars as pl

from ... import seqmatch

from .. import pair

from .executor_base import JoinExecutor

class JoinExecutorNrStage():
    def __init__(
            self,
            ro_min: float=None,
            size_ro_min: float=None,
            offset_max: int=None,

            offset_prop_max: float=None,
            match_prop_min: float=None,
            match_ref: bool=False,
            match_alt: bool=False,
            col_map: dict[str,str]=None,
            match_score_model: seqmatch.MatchScoreModel=None,
            force_end_ro: bool=False
    ):

        self.ro_min = ro_min
        self.size_ro_min = size_ro_min
        self.offset_max = offset_max
        self.offset_prop_max = offset_prop_max
        self.match_prop_min = match_prop_min
        self.match_ref = match_ref
        self.match_alt = match_alt
        self.col_map = col_map
        self.force_end_ro = force_end_ro

        if match_score_model is None:
            match_score_model = seqmatch.MatchScoreModel()

        self.match_score_model = match_score_model

class IntersectExecutorNr(JoinExecutor):

    def __init__(
            self,
            col_map: dict[str,str]=None
    ):
        super().__init__()

        self.stage_list = list()
        self.col_map = None

        pass

    def _join(
            self,
            df_list: list[pl.LazyFrame],
            source_names: list[str]
    ) -> pl.DataFrame:

        n_df = len(df_list)

        if self.stage_list is None or len(self.stage_list) == 0:
            raise RuntimeError(f'No join stages configured.')

        for index_pairs in itertools.combinations(range(n_df), 2):
            index_a = index_pairs[0]
            index_b = index_pairs[1]

            df_a = df_list[index_a]
            df_b = df_list[index_b]

            df_join = pair.intersect(
                df_a,
                df_b,
                ro_min=self.ro_min,
                size_ro_min=self.size_ro_min,
                offset_max=self.offset_max,
                offset_prop_max=self.offset_prop_max,
                match_prop_min=self.match_prop_min,
                match_ref=self.match_ref,
                match_alt=self.match_alt,
                col_map=self.col_map,
                match_score_model=self.match_score_model,
                force_end_ro=False
            ).collect()

