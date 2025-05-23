"""
Provides a Bodo implementation of the pandas groupby API.
"""

from __future__ import annotations

from typing import Literal

from bodo.pandas.utils import BodoLibNotImplementedException, check_args_fallback


class DataFrameGroupBy:
    """
    Similar to pandas DataFrameGroupBy. See Pandas code for reference:
    https://github.com/pandas-dev/pandas/blob/0691c5cf90477d3503834d983f69350f250a6ff7/pandas/core/groupby/generic.py#L1329
    """

    def __init__(self, obj, keys, selection=None):
        self._obj = obj
        self._keys = keys
        self._selection = selection

    def __getitem__(self, key) -> DataFrameGroupBy | SeriesGroupBy:
        """
        Return a DataFrameGroupBy or SeriesGroupBy for the selected data columns.
        """
        if isinstance(key, str):
            return SeriesGroupBy(self._obj, self._keys, key)
        else:
            raise BodoLibNotImplementedException(
                f"DataFrameGroupBy: Invalid key type: {type(key)}"
            )


class SeriesGroupBy:
    """
    Similar to pandas SeriesGroupBy.
    """

    def __init__(self, obj, keys, selection):
        self._obj = obj
        self._keys = keys
        self._selection = selection

    @check_args_fallback(supported="none")
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        """
        Compute the sum of each group.
        """
