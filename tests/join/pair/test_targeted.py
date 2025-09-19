"""
Targeted tests for agglovar.join.pair
"""

from tests.join.resources import *

@pytest.mark.parametrize(PARAM_KEY_SPEC, PARAM_TUPLES, scope='class')
class TestJoin:
    """Fixture for join tests."""

    def test_check_schema(
        self,
        df_join: pl.DataFrame,
        df_exp: pl.DataFrame
    ) -> None:
        """Check join schema.

        :param df_join: Actual join.
        :param df_exp: Expected join.
        """

        # Expected columns
        assert list(df_join.columns) == list(df_exp.columns)

        # Types match
        assert dict(df_join.schema) == dict(df_exp.schema), 'Schema does not match'

    def test_join_records(self,
        df_join: pl.DataFrame,
        df_exp: pl.DataFrame
    ) -> None:
        """Check that the correct records were joined (does not check metadata).

        :param df_join: Actual join.
        :param df_exp: Expected join.
        """

        df_miss = subset_miss(df_join, df_exp)
        df_extra = subset_extra(df_join, df_exp)

        if df_miss.height:
            assert False, (
                f'Missing {df_miss.height} join records ({df_extra.height} extraneous): '
                + '; '.join([
                    f'"{str(row)}"'
                        for row in [
                            df_miss.row(row_index, named=True)
                                for row_index in range(min(df_miss.height, 2))
                        ]
                ])
                + ('...' if df_miss.height > 2 else '')
            )

        if df_extra.height:
            assert False, (
                f'Found {df_extra.height} extraneous join records: '
                + '; '.join([
                    f'"{str(row)}"'
                        for row in [
                            df_extra.row(row_index, named=True)
                                for row_index in range(min(df_extra.height, 2))
                        ]
                ])
                + ('...' if df_extra.height > 2 else '')
            )

    def test_join_meta(self,
        df_join: pl.DataFrame,
        df_exp: pl.DataFrame
    ) -> None:
        """Check join metadata.

        These fields are reported with the join table and should be accurate.

        :param df_join: Actual join.
        :param df_exp: Expected join.
        """

        join_cols = list(df_exp.columns)
        join_cols_no_index = [col for col in join_cols if col not in {'index_a', 'index_b'}]

        # Check expected join parameters: join both "join" and "exp" tables), then compare values
        # across records.
        df_mg = (
            df_join
            .join(
                df_exp,
                on=['index_a', 'index_b'],
                how='inner',
                nulls_equal=True,
                suffix='_exp'
            )
        )

        df_mg_join = (
            df_mg
            .select(list(df_exp.columns))
        )

        df_mg_exp = (
            df_mg
            .select(['index_a', 'index_b'] + [col + '_exp' for col in join_cols_no_index])
            .rename(lambda col: col[:-4] if col != 'index_a' and col != 'index_b' else col)
        )

        for index in range(df_mg.height):
            assert (
                (row_a := row_to_dict(df_mg_join, index))
                == row_to_dict(df_mg_exp, index, approx=True)
            ), (
                f'Join metadata mismatch in join table at index {index} '
                f'(join: {row_a["index_a"]} "{row_a["id_a"]}") <-> '
                f'(exp: {row_a["index_b"]} "{row_a["id_b"]}")'
            )
