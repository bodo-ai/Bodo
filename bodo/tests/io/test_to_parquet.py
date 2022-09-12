import pandas as pd
import pytest

import bodo
from bodo.tests.utils import _get_dist_arg
from bodo.utils.testing import datadir, ensure_clean2


def check_write(df):
    fp_bodo = (datadir / "bodo.pq").as_posix()
    fp_pandas = datadir / "pandas.pq"
    write = lambda df: df.to_parquet(fp_bodo)
    distributions = {
        "sequential": [lambda x, *args: x, [], dict()],
        "1d-distributed": [
            _get_dist_arg,
            [False],
            {"all_args_distributed_block": True},
        ],
        "1d-distributed-varlength": [
            _get_dist_arg,
            [False, True],
            {"all_args_distributed_varlength": True},
        ],
    }
    for dist_func, args, kwargs in distributions.values():
        with ensure_clean2(fp_bodo), ensure_clean2(fp_pandas):
            write_jit = bodo.jit(write, **kwargs)
            write_jit(dist_func(df, *args))
            df.to_parquet(fp_pandas)
            bodo.barrier()
            df_bodo = pd.read_parquet(fp_bodo)
            df_pandas = pd.read_parquet(fp_pandas)
        pd.testing.assert_frame_equal(df_bodo, df_pandas, check_column_type=False)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {
                "A": [
                    ["a", None, "cde"],
                    None,
                    ["random"],
                    ["pad"],
                    ["shrd", "lu"],
                    ["je", "op", "ardy"],
                    ["double", "je", "op", "ardy"],
                    None,
                ]
            }
        ),
    ],
)
def test_nested_string(df):
    check_write(df)
