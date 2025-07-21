"""
Base class for intersect executors.
"""

import abc

import polars as pl

class JoinExecutor(metaclass=abc.ABCMeta):
    """
    Interface for intersect runners.
    """

    def join(
            self,
            df_list: list[pl.DataFrame|pl.LazyFrame],
            source_names: list[str]=None
    ) -> pl.DataFrame:
        """
        Run intersect for a list of callsets.

        During the intersect process, intermediate intersect results must be collected, so the whole process cannot
        run in streaming/lazy mode. However, the final step uses the resolved merge to extract and formats the merged
        callset, which is streamable and can require significant memory for large callsets with many fields.

        :param df_list: List of callsets. Must be a list or tuple of Polars DataFrames or LazyFrames.
        :param source_names: Names of each callset in `df_list`. If any names are missing, they will be set by the
            index in `df_list` (i.e. SourceIdx0, SourceIdx1, etc.). Missing names could occur if elements of this list
            are None, the list is shorter than `df_list`, or the list itself is None.

        :return: A Polars DataFrame (collect = True) or LazyFrame (collect = False).
        """

        # Check dataframe list
        if df_list is None:
            raise RuntimeError('Cannot run intersect with callset list: None')

        if not isinstance(df_list, (tuple,list)):
            raise RuntimeError(f'Expected callset list as a list or tuple: {type(df_list)}')

        df_list = list(df_list)

        if len(df_list) < 2:
            raise RuntimeError(f'Cannot run intersect with less than 2 callsets: {len(df_list)}')

        # Check input frames and get LazyFrames (leave existing list unaltered)
        df_list_new = list()

        for index, df in enumerate(df_list):
            if isinstance(df, pl.DataFrame):
                df_list_new.append(df.lazy())
            elif isinstance(df, pl.LazyFrame):
                df_list_new.append(df)
            else:
                raise RuntimeError(f'Expected callset list elements to be Polars DataFrames or LazyDataFrames: Found type "{type(df)}" at df_list[{index}]')

        df_list = df_list_new

        # Set source names
        if source_names is None:
            source_names = list()

        elif not isinstance(source_names, (tuple,list)):
            raise RuntimeError(f'Expected source name list as a list or tuple: {type(df_list)}')

        if len(source_names) > len(df_list):
            raise RuntimeError(f'Cannot set more source names than callsets: {len(source_names)} > {len(df_list)}')

        if len(source_names) < len(df_list):
            source_names = source_names + [None] * (len(df_list) - len(source_names))

        for index, name in enumerate(source_names):
            if name is None:
                source_names[index] = f'SourceIdx{index}'
            elif not isinstance(name, str):
                raise RuntimeError(f'Expected source names to be strings: Found type "{type(name)}" at source_names[{index}]')

        # Run intersect
        return self._join(
            df_list,
            source_names
        )

    @abc.abstractmethod
    def _join(
            self,
            df_list: list[pl.LazyFrame],
            source_names: list[str]
    ) -> pl.DataFrame:
        """
        Executor implementation for joining callsets. Parameters are already checked and normalized by `join()`.

        :param df_list: List of callsets.
        :param source_names: List of callset names.

        :return: A Polars DataFrame describing the join.
        """

        pass
