import pytest
from pandas.core.internals.array_manager import ArrayManager, SingleArrayManager
from pandas.core.internals.managers import BlockManager, SingleBlockManager

from bodo.pandas.array_manager import LazyArrayManager, LazySingleArrayManager
from bodo.pandas.managers import LazyBlockManager, LazySingleBlockManager


@pytest.fixture(params=["array", "block"])
def pandas_managers(request):
    from pandas._config.config import _get_option, _set_option

    prev_option = _get_option("mode.data_manager", silent=True)
    _set_option("mode.data_manager", request.param, silent=True)
    assert _get_option("mode.data_manager", silent=True) == request.param
    yield (
        (LazyBlockManager, BlockManager)
        if request.param == "block"
        else (LazyArrayManager, ArrayManager)
    )
    _set_option("mode.data_manager", prev_option, silent=True)


@pytest.fixture(params=["array", "block"])
def single_pandas_managers(request):
    from pandas._config.config import _get_option, _set_option

    prev_option = _get_option("mode.data_manager", silent=True)
    _set_option("mode.data_manager", request.param, silent=True)
    assert _get_option("mode.data_manager", silent=True) == request.param
    yield (
        (LazySingleBlockManager, SingleBlockManager)
        if request.param == "block"
        else (LazySingleArrayManager, SingleArrayManager)
    )
    _set_option("mode.data_manager", prev_option, silent=True)
