import io

import numpy as np
import pytest
from numba.core import types  # noqa TID253
from numba.core.typeconv.castgraph import Conversion  # noqa TID253

import bodo
from bodo.tests.user_logging_utils import (
    check_logger_no_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func

# TODO[BSE-5071]: Re-enable native typer when its coverage improved
pytestmark = [pytest.mark.compiler, pytest.mark.skip]


def check_native_type_inferrer(impl, args):
    """Test native type inference for input function and arguments"""

    old_use_native_type_inference = bodo.bodo_use_native_type_inference
    try:
        bodo.bodo_use_native_type_inference = True

        check_func(impl, args, only_seq=True)
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 2):
            bodo_func = bodo.jit(distributed=False)(impl)
            bodo_func(*args)
            check_logger_no_msg(
                stream, "Native type inference failed, falling back to Numba."
            )

    finally:
        bodo.bodo_use_native_type_inference = old_use_native_type_inference


def test_basic_func(memory_leak_check):
    """
    Make sure a trivial function can go through native type inference end to end
    """

    def impl(a):
        return 3

    check_native_type_inferrer(impl, (3,))


def test_simple_call(memory_leak_check):
    """
    Make sure a simple call can go through native type inference end to end
    """

    def impl2(a):
        return bool(a)

    check_native_type_inferrer(impl2, (3,))


def test_control_flow(memory_leak_check):
    """
    Make sure simple control flow with multiple return types works in native type
    inference
    """

    def impl3(a, b):
        c = 1
        if a:
            c = b
        else:
            c = 2
        return c

    check_native_type_inferrer(impl3, (1, 2))


def test_type_unification(memory_leak_check):
    """
    Make sure basic integer type unification works
    """

    def impl(a, b, d):
        c = 1
        if a:
            c = b
        else:
            c = d
        return c

    check_native_type_inferrer(impl, (1, np.int32(2), 4))


def test_conversion_rules(memory_leak_check):
    """
    Make sure conversion rules are initialized correctly in native typer
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.transforms.type_inference.native_typer import check_compatible_types

    conv = check_compatible_types(types.boolean, types.int8)
    assert conv == Conversion.safe, (
        "check_compatible_types(types.boolean, types.int8) failed"
    )

    conv = check_compatible_types(types.int8, types.bool_)
    assert conv == Conversion.unsafe, (
        "check_compatible_types(types.int8, types.bool_) failed"
    )


def test_getattr(memory_leak_check):
    """
    Make sure basic gettar works
    """

    def impl(a):
        return np.int32(a)

    check_native_type_inferrer(impl, (1,))


def test_call_shortcut(memory_leak_check):
    """
    Make sure native call resolve shortcut works
    """

    def impl1():
        return bodo.get_rank()

    check_native_type_inferrer(impl1, ())

    def impl2():
        return bodo.libs.distributed_api.get_rank()

    check_native_type_inferrer(impl2, ())
