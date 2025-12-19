import pandas as pd

from bodo.pandas.frame import BodoDataFrame
from bodo.tests.test_lazy.utils import pandas_managers  # noqa


def test_bodo_frame_pandas_manager(pandas_managers):
    """
    Test basic operations on a bodo frame using a pandas manager.
    """
    base_df = pd.DataFrame(
        {
            "a": pd.array(["a", "bc", "def", "ghij", "klmno"] * 8),
            "b": pd.array([1, 2, 3, 10, None] * 8),
        }
    )

    bodo_df = BodoDataFrame._from_mgr(base_df._mgr, [])
    assert bodo_df.shape == base_df.shape
    assert bodo_df.dtypes.equals(base_df.dtypes)

    assert bodo_df.head(5).equals(bodo_df.head(5))
