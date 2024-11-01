from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.series import BodoSeries
from bodo.pandas.arrow.array import LazyArrowExtensionArray
from bodo.pandas.managers import LazyBlockManager, LazySingleBlockManager
from bodo.pandas.array_manager import LazyArrayManager, LazySingleArrayManager
from pandas._config.config import _get_option
import typing as pt


def get_lazy_manager_class() -> pt.Type[LazyArrayManager | LazyBlockManager]:
    """Get the lazy manager class based on the pandas option mode.data_manager, suitable for DataFrame."""
    data_manager = _get_option("mode.data_manager", silent=True)
    if data_manager == "block":
        return LazyBlockManager
    elif data_manager == "array":
        return LazyArrayManager
    raise Exception(f"Got unexpected value of pandas option mode.manager: {data_manager}")

def get_lazy_single_manager_class() -> pt.Type[LazySingleArrayManager | LazySingleBlockManager]:
    """Get the lazy manager class based on the pandas option mode.data_manager, suitable for Series."""
    data_manager = _get_option("mode.data_manager", silent=True)
    if data_manager == "block":
        return LazySingleBlockManager
    elif data_manager == "array":
        return LazySingleArrayManager
    raise Exception(f"Got unexpected value of pandas option mode.manager: {data_manager}")
