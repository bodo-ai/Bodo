"""
Provides a Bodo implementation of the pandas groupby API.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import pandas as pd

from bodo.pandas.utils import (
    BodoLibFallbackWarning,
    BodoLibNotImplementedException,
    LazyPlan,
    check_args_fallback,
    make_col_ref_exprs,
    wrap_plan,
)


class DataFrameGroupBy:
    """
    Similar to pandas DataFrameGroupBy. See Pandas code for reference:
    https://github.com/pandas-dev/pandas/blob/0691c5cf90477d3503834d983f69350f250a6ff7/pandas/core/groupby/generic.py#L1329
    """

    def __init__(
        self, obj: pd.DataFrame, keys: list[str], selection: list[str] | None = None
    ):
        self._obj = obj
        self._keys = keys
        self._selection = selection

    def __getitem__(self, key) -> DataFrameGroupBy | SeriesGroupBy:
        """
        Return a DataFrameGroupBy or SeriesGroupBy for the selected data columns.
        """
        if isinstance(key, str):
            return SeriesGroupBy(self._obj, self._keys, [key])
        else:
            raise BodoLibNotImplementedException(
                f"DataFrameGroupBy: Invalid key type: {type(key)}"
            )

    @check_args_fallback(unsupported="none")
    def __getattribute__(self, name: str, /) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            msg = (
                f"DataFrameGroupBy.{name} is not "
                "implemented in Bodo dataframe library yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            warnings.warn(BodoLibFallbackWarning(msg))
            gb = pd.DataFrame(self._obj).groupby(self._keys)
            if self._selection is not None:
                gb = gb[self._selection]
            return object.__getattribute__(gb, name)


class SeriesGroupBy:
    """
    Similar to pandas SeriesGroupBy.
    """

    def __init__(self, obj: pd.DataFrame, keys: list[str], selection: list[str]):
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
        from bodo.pandas.base import _empty_like

        zero_size_df = _empty_like(self._obj)

        assert len(self._selection) == 1, (
            "SeriesGroupBy.sum() should only be called on a single column selection."
        )

        empty_data = zero_size_df.groupby(self._keys)[self._selection[0]].sum()

        key_indices = [self._obj.columns.get_loc(c) for c in self._keys]
        exprs = [
            LazyPlan(
                "AggregateExpression",
                zero_size_df[c],
                self._obj._plan,
                "sum",
                [self._obj.columns.get_loc(c)],
            )
            for c in self._selection
        ]

        agg_plan = LazyPlan(
            "LogicalAggregate",
            empty_data,
            self._obj._plan,
            key_indices,
            exprs,
        )

        # Add the data column then the keys since they become Index columns in output.
        # DuckDB generates keys first in output so we need to reverse the order.
        col_indices = [len(self._keys)]
        col_indices += list(range(len(self._keys)))

        exprs = make_col_ref_exprs(col_indices, agg_plan)
        proj_plan = LazyPlan(
            "LogicalProjection",
            empty_data,
            agg_plan,
            exprs,
        )

        return wrap_plan(proj_plan)

    @check_args_fallback(unsupported="none")
    def __getattribute__(self, name: str, /) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            msg = (
                f"SeriesGroupBy.{name} is not "
                "implemented in Bodo dataframe library yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            warnings.warn(BodoLibFallbackWarning(msg))
            gb = pd.DataFrame(self._obj).groupby(self._keys)[self._selection[0]]
            return object.__getattribute__(gb, name)
