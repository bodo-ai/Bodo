import pandas as pd

import bodo.pandas as bd
from bodo.pandas.series import dt_accessors
from bodo.tests.utils import _test_equal


def gen_dt_accessor_test(name):
    """
    Generates a test case for Series.dt.<name> accessor.
    """

    def test_func():
        date_m = pd.Series(pd.date_range("20130101 09:10:12", periods=10, freq="MS"))
        date_s = pd.Series(pd.date_range("20220101 09:10:12", periods=10, freq="s"))
        date_y = pd.Series(pd.date_range("19990303 09:10:12", periods=10, freq="YE"))

        df = pd.DataFrame({"A": date_m, "B": date_s, "C": date_y})
        bdf = bd.from_pandas(df)

        keys = ["A", "B", "C"]

        for key in keys:
            a = getattr(df, key)
            b = getattr(bdf, key)
            pd_accessor = getattr(a.dt, name)
            bodo_accessor = getattr(b.dt, name)
            assert bodo_accessor.is_lazy_plan()
            _test_equal(pd_accessor, bodo_accessor, check_pandas_types=False)

    return test_func


for accessor_pair in dt_accessors:
    for accessor_name in accessor_pair[0]:
        test = gen_dt_accessor_test(accessor_name)
        globals()[f"test_dt_{accessor_name}"] = test
