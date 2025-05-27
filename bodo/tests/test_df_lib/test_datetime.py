import pandas as pd

import bodo.pandas as bd
from bodo.tests.utils import _test_equal


def gen_dt_accessor_test(name):
    """
    Generates a parameterized test case for Series.str.<name> method.
    """

    def test_func():
        date_m = pd.Series(pd.date_range("20130101 09:10:12", periods=10, freq="m"))
        date_s = pd.Series(pd.date_range("20220101 09:10:12", periods=10, freq="s"))
        date_y = pd.Series(pd.date_range("19990303 09:10:12", periods=10, freq="y"))

        df = pd.DataFrame({"A": date_m, "B": date_s, "C": date_y})
        bdf = bd.from_pandas(df)

        keys = ["A", "B", "C"]

        for key in keys:
            pd_accessor = getattr(df.key.dt, name)
            bodo_accessor = getattr(bdf.key.dt, name)

        assert bodo_accessor.is_lazy_plan()
        _test_equal(pd_accessor, bodo_accessor)

    return test_func


gen_dt_accessor_test("year")
print("Success")
