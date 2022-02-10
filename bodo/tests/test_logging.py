"""
Test logging module.
"""

import io
import logging
import re

import pytest

import bodo
from bodo.utils.typing import BodoError


def _test_logging_impl(impl, l):
    f = io.StringIO()
    logging.basicConfig(stream=f, force=True, level=logging.INFO)
    bodo.jit(impl)(l)
    g = io.StringIO()
    logging.basicConfig(stream=g, force=True, level=logging.INFO)
    impl(l)
    assert f.getvalue() == g.getvalue()


def test_logging_logger_info():
    def test_impl(l):
        l.warning("info1")
        l.warning("info2")

    l = logging.getLogger()
    o = logging.getLogger("Other")
    _test_logging_impl(test_impl, l)
    _test_logging_impl(test_impl, o)


# TODO: [BE-2121] fix debug
# def test_logging_rootlogger_debug():
#     def test_impl(l):
#         l.debug("debug1")
#         l.debug("debug2")

#     l = logging.getLogger()

#     f = io.StringIO()
#     logging.basicConfig(stream=f, force=True, level=logging.DEBUG)
#     bodo.jit(test_impl)(l)
#     g = io.StringIO()
#     logging.basicConfig(stream=g, force=True, level=logging.DEBUG)
#     test_impl(l)
#     assert f.getvalue() == g.getvalue()

# TODO: deprecation warning
# def test_logging_logger_warn():
#     def test_impl(l):
#         l.warn("warning1")
#         l.warn("warning2")

#     l = logging.getLogger()
#     _test_logging_impl(test_impl, l)


def test_logging_logger_warning():
    def test_impl(l):
        l.warning("warning1")
        l.warning("warning2")

    l = logging.getLogger()
    o = logging.getLogger("Other")
    _test_logging_impl(test_impl, l)
    _test_logging_impl(test_impl, o)


def test_logging_logger_error():
    def test_impl(l):
        l.error("error1")
        l.error("error2")

    l = logging.getLogger()
    o = logging.getLogger("Other")
    _test_logging_impl(test_impl, l)
    _test_logging_impl(test_impl, o)


def test_logging_logger_exception():
    def test_impl(l):
        l.exception("except1")
        l.exception("except2")

    l = logging.getLogger()
    o = logging.getLogger("Other")
    _test_logging_impl(test_impl, l)
    _test_logging_impl(test_impl, o)


def test_logging_logger_critical():
    def test_impl(l):
        l.critical("critical1")
        l.critical("critical2")

    l = logging.getLogger()
    o = logging.getLogger("Other")
    _test_logging_impl(test_impl, l)
    _test_logging_impl(test_impl, o)


def test_logging_logger_log():
    def test_impl(l):
        l.log(logging.INFO, "info")
        l.log(logging.WARNING, "warning")
        l.log(logging.ERROR, "error")
        l.log(logging.CRITICAL, "critical")

    l = logging.getLogger()
    o = logging.getLogger("Other")
    _test_logging_impl(test_impl, l)
    _test_logging_impl(test_impl, o)


def test_logging_logger_setLevel():
    def test_impl(l):
        l.setLevel(logging.WARNING)
        l.info("info1")
        l.setLevel(logging.INFO)
        l.info("info2")

    l = logging.getLogger()
    o = logging.getLogger("Other")
    _test_logging_impl(test_impl, l)
    _test_logging_impl(test_impl, o)


def test_logging_logger_attrs():
    level_impl = lambda l: l.level
    name_impl = lambda l: l.name
    parent_impl = lambda l: l.parent
    root_impl = lambda l: l.root
    prop_impl = lambda l: l.propagate
    disabled_impl = lambda l: l.disabled

    l = logging.getLogger()
    o = logging.getLogger("Other")
    for impl in (
        level_impl,
        name_impl,
        parent_impl,
        prop_impl,
        root_impl,
        disabled_impl,
    ):
        assert bodo.jit(impl)(l) == impl(l) and bodo.jit(impl)(o) == impl(o)


def test_logging_logger_name():
    def test_impl(l):
        return l.name

    l = logging.getLogger()
    o = logging.getLogger("Other")
    assert bodo.jit(test_impl)(l) == test_impl(l) and bodo.jit(test_impl)(
        o
    ) == test_impl(o)


def test_logging_logger_lowering():
    logger = logging.getLogger()
    other = logging.getLogger("Other")

    def test_impl1():
        logger.info("info1")
        logger.info("info2")

    def test_impl2():
        other.info("info1")
        other.info("info2")

    f = io.StringIO()
    logging.basicConfig(stream=f, force=True, level=logging.INFO)
    bodo.jit(test_impl1)()
    g = io.StringIO()
    logging.basicConfig(stream=g, force=True, level=logging.INFO)
    test_impl1()
    assert f.getvalue() == g.getvalue()

    x = io.StringIO()
    logging.basicConfig(stream=x, force=True, level=logging.INFO)
    bodo.jit(test_impl2)()
    y = io.StringIO()
    logging.basicConfig(stream=y, force=True, level=logging.INFO)
    test_impl2()
    assert x.getvalue() == y.getvalue()


def test_logging_logger_unsupported():
    @bodo.jit
    def test_unsupp_attr(l):
        return l.handlers

    @bodo.jit
    def test_unsupp_method(l):
        l.hasHandlers()

    l = logging.getLogger()
    with pytest.raises(
        BodoError, match=re.escape("logging.Logger.handlers not supported yet")
    ):
        test_unsupp_attr(l)
    with pytest.raises(
        BodoError, match=re.escape("logging.Logger.hasHandlers not supported yet")
    ):
        test_unsupp_method(l)
