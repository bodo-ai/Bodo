# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Utility functions to help write tests that depend on the
    bodo logging level.
"""
import logging
from contextlib import contextmanager

import bodo
from bodo.tests.utils import reduce_sum


@contextmanager
def set_logging_stream(logger, verbose_level):
    try:
        passed = 1
        bodo.set_verbose_level(verbose_level)
        bodo.set_bodo_verbose_logger(logger)
        yield
    except Exception as e:
        # Print the error message
        print(e, flush=True)
        passed = 0
    finally:
        bodo.user_logging.restore_default_bodo_verbose_level()
        bodo.user_logging.restore_default_bodo_verbose_logger()
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size(), "Error while testing logging stream"


def create_string_io_logger(stream):
    """
    Creates a IO logger that records
    the verbose info with the given
    stream.
    """
    logger = logging.getLogger("Testing Logger")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)

    formater = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formater)
    logger.addHandler(handler)

    return logger


def check_logger_msg(stream, msg):
    """
    Checks that a specific msg in found inside logger.
    This simply checks if the logger contains the exact
    msg string and doesn't not check a regex.

    We only check the logger on rank 0 because we only
    write on rank 0.
    """
    if bodo.get_rank() == 0:
        assert msg in stream.getvalue()


def check_logger_no_msg(stream, msg):
    """
    Checks that a specific msg is not found inside logger.
    This simply checks if the logger contains the exact
    msg string and doesn't not check a regex.

    We only check the logger on rank 0 because we only
    write on rank 0.
    """
    if bodo.get_rank() == 0:
        assert msg not in stream.getvalue()
