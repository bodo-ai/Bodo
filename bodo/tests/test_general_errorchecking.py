import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


def test_undefined_variable():
    message = "name 'undefined_variable' is not defined"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: pd.read_csv(undefined_variable))()


@pytest.mark.slow
def test_fn_return_type_error():
    def test_impl(n):
        if n > 10:
            return "hello world"
        else:
            return False

    message = r"Unable to unify the following function return types.*"
    with pytest.raises(BodoError, match=message):
        bodo.jit(test_impl)(3)


@pytest.mark.slow
def test_bcast_scalar_type_error():
    def test_impl():
        return bodo.libs.distributed_api.bcast_scalar(b"I'm a bytes val")

    message = r"bcast_scalar requires an argument of type Integer, Float, datetime64ns, timedelta64ns, string, None, or Bool. Found type.*"
    with pytest.raises(BodoError, match=message):
        bodo.jit(test_impl)()
