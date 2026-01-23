import pandas as pd
import pytest
from pandas.core.internals.managers import BlockManager, SingleBlockManager

try:
    from pandas.core.internals.array_manager import ArrayManager, SingleArrayManager
except ModuleNotFoundError:
    # Pandas >= 3 does not have an array_manager module (uses BlockManager/SinglBlockManager).
    class ArrayManager:
        pass

    class SingleArrayManager:
        pass


from bodo.pandas.array_manager import LazyArrayManager, LazySingleArrayManager
from bodo.pandas.managers import LazyBlockManager, LazySingleBlockManager

pandas_version = tuple(map(int, pd.__version__.split(".")[:2]))


@pytest.fixture(params=["array", "block"] if pandas_version < (3, 0) else ["block"])
def pandas_managers(request):
    if pandas_version < (3, 0):
        from pandas._config.config import _get_option, _set_option

        prev_option = _get_option("mode.data_manager", silent=True)
        _set_option("mode.data_manager", request.param, silent=True)
        assert _get_option("mode.data_manager", silent=True) == request.param

    yield (
        (LazyBlockManager, BlockManager)
        if request.param == "block"
        else (LazyArrayManager, ArrayManager)
    )

    if pandas_version < (3, 0):
        _set_option("mode.data_manager", prev_option, silent=True)


@pytest.fixture(params=["array", "block"] if pandas_version < (3, 0) else ["block"])
def single_pandas_managers(request):
    if pandas_version < (3, 0):
        from pandas._config.config import _get_option, _set_option

        prev_option = _get_option("mode.data_manager", silent=True)
        _set_option("mode.data_manager", request.param, silent=True)
        assert _get_option("mode.data_manager", silent=True) == request.param

    yield (
        (LazySingleBlockManager, SingleBlockManager)
        if request.param == "block"
        else (LazySingleArrayManager, SingleArrayManager)
    )

    if pandas_version < (3, 0):
        _set_option("mode.data_manager", prev_option, silent=True)
