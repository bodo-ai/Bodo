import pytest

import bodo
from bodo.libs.bodosql_array_kernels import (
    create_javascript_udf,
)
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.utils.typing import MetaType


@pytest.mark.skipif(
    bodo.libs.bodosql_javascript_udf_array_kernels.javascript_udf_enabled,
    reason="JavaScript UDFs enabled",
)
def test_javascript_udf_error_if_not_in_build():
    body = MetaType("return 2 + 1")
    args = MetaType(())
    ret_type = IntegerArrayType(bodo.int64)

    @bodo.jit
    def f():
        create_javascript_udf(body, args, ret_type)

    with pytest.raises(
        Exception,
        match="JavaScript UDF support is only available on Bodo Platform. https://docs.bodo.ai/latest/quick_start/quick_start_platform/",
    ):
        f()
