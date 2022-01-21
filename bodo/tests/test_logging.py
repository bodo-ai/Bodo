"""
Test logging module.
"""

import io
import logging
import re

import pytest

import bodo
from bodo.utils.typing import BodoError


def test_logging_rootlogger_info():
    def test_impl(l):
        l.info("info1")
        l.info("info2")

    l = logging.getLogger()

    f = io.StringIO()
    logging.basicConfig(stream=f, force=True, level=logging.INFO)
    bodo.jit(test_impl)(l)
    g = io.StringIO()
    logging.basicConfig(stream=g, force=True, level=logging.INFO)
    test_impl(l)
    assert f.getvalue() == g.getvalue()


def test_logging_rootlogger_lowering():
    logger = logging.getLogger()

    def test_impl():
        logger.info("info1")
        logger.info("info2")

    f = io.StringIO()
    logging.basicConfig(stream=f, force=True, level=logging.INFO)
    bodo.jit(test_impl)()
    g = io.StringIO()
    logging.basicConfig(stream=g, force=True, level=logging.INFO)
    test_impl()
    assert f.getvalue() == g.getvalue()


def test_logging_rootlogger_unsupported():
    @bodo.jit
    def test_unsupp_attr(l):
        l.propagate

    @bodo.jit
    def test_unsupp_method(l):
        l.warning("warning")

    l = logging.getLogger()
    with pytest.raises(
        BodoError, match=re.escape("logging.RootLogger.propagate not supported yet")
    ):
        test_unsupp_attr(l)
    with pytest.raises(BodoError, match="logging.Rootlogger.warning not supported yet"):
        test_unsupp_method(l)
