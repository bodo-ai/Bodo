import pandas as pd
import pyarrow as pa

import bodosql
from bodo.tests.caching_tests.caching_tests_common import fn_distribution  # noqa
from bodo.tests.utils import InputDist, check_caching


def test_nested_arr_cache(fn_distribution, is_cached, memory_leak_check):
    """
    Test caching workaround for nested array handling.
    See https://bodo.atlassian.net/browse/BSE-3359
    """

    check_cache = is_cached == "y"

    def impl(arr):
        A = bodosql.kernels.array_construct((arr,), (False,))
        return len(A)

    arr = pd.arrays.ArrowExtensionArray(
        pa.array([[1, 3, None]], pa.large_list(pa.int64()))
    )
    check_caching(impl, (arr,), check_cache, InputDist.REP, py_output=1)
