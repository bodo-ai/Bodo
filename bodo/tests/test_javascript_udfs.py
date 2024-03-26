import pytest

import bodo
from bodo.libs.bodosql_array_kernels import (
    create_javascript_udf,
    delete_javascript_udf,
    execute_javascript_udf,
)
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.tests.conftest import (
    pytest_mark_javascript,
)
from bodo.tests.utils import check_func
from bodo.utils.typing import MetaType

pytestmark = pytest_mark_javascript


def test_javascript_udf_no_args_return_int(memory_leak_check):
    body = MetaType("return 2 + 1")
    args = MetaType(tuple())
    ret_type = IntegerArrayType(bodo.int64)

    def f():
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, tuple())
        delete_javascript_udf(f)
        return out_arr

    expected_output = 3
    check_func(f, tuple(), py_output=expected_output)


def test_javascript_interleaved_execution(memory_leak_check):
    body_a = MetaType("return 2 + 1")
    body_b = MetaType("return 2 + 2")
    args = MetaType(tuple())
    ret_type = IntegerArrayType(bodo.int64)

    def f():
        a = create_javascript_udf(body_a, args, ret_type)
        b = create_javascript_udf(body_b, args, ret_type)
        out_1 = execute_javascript_udf(a, tuple())
        out_2 = execute_javascript_udf(b, tuple())
        out_3 = execute_javascript_udf(a, tuple())
        out_4 = execute_javascript_udf(b, tuple())
        delete_javascript_udf(a)
        delete_javascript_udf(b)
        return out_1, out_2, out_3, out_4

    expected_output = (3, 4, 3, 4)
    check_func(f, tuple(), py_output=expected_output)


@pytest.mark.skip(reason="Returning strings isn't implemented yet")
def test_javascript_unicode_in_body(memory_leak_check):
    body = MetaType("return 'hëllo'")
    args = MetaType(tuple())
    ret_type = IntegerArrayType(bodo.int64)

    def f():
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, tuple())
        delete_javascript_udf(f)
        return out_arr

    expected_output = "hëllo"
    check_func(f, tuple(), py_output=expected_output)
