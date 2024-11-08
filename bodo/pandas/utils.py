from pandas._config.config import _get_option

from bodo.pandas.array_manager import LazyArrayManager, LazySingleArrayManager
from bodo.pandas.managers import LazyBlockManager, LazySingleBlockManager


def get_lazy_manager_class() -> type[LazyArrayManager | LazyBlockManager]:
    """Get the lazy manager class based on the pandas option mode.data_manager, suitable for DataFrame."""
    data_manager = _get_option("mode.data_manager", silent=True)
    if data_manager == "block":
        return LazyBlockManager
    elif data_manager == "array":
        return LazyArrayManager
    raise Exception(
        f"Got unexpected value of pandas option mode.manager: {data_manager}"
    )


def get_lazy_single_manager_class() -> (
    type[LazySingleArrayManager | LazySingleBlockManager]
):
    """Get the lazy manager class based on the pandas option mode.data_manager, suitable for Series."""
    data_manager = _get_option("mode.data_manager", silent=True)
    if data_manager == "block":
        return LazySingleBlockManager
    elif data_manager == "array":
        return LazySingleArrayManager
    raise Exception(
        f"Got unexpected value of pandas option mode.manager: {data_manager}"
    )
