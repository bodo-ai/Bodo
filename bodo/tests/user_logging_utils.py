"""
Utility functions to help write tests that depend on the
bodo logging level.
"""

import logging
from contextlib import contextmanager

import bodo


@contextmanager
def set_logging_stream(logger, verbose_level):
    from bodo.tests.utils_jit import reduce_sum

    err = None
    try:
        passed = 1
        # TODO(aneesh) this is slightly hacky - remove when tracing level is 1
        # by default
        original_tracing_level = bodo.tracing_level
        bodo.tracing_level = 1
        bodo.set_verbose_level(verbose_level)
        bodo.set_bodo_verbose_logger(logger)
        yield
    except Exception as e:
        err = e
        passed = 0
    finally:
        bodo.tracing_level = original_tracing_level
        bodo.user_logging.restore_default_bodo_verbose_level()
        bodo.user_logging.restore_default_bodo_verbose_logger()
        n_passed = reduce_sum(passed)
        if n_passed != bodo.get_size():
            if err is not None:
                raise err
            else:
                raise AssertionError(
                    "Error while testing logging stream. See other rank."
                )


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


def check_logger_msg(stream, msg, check_case=True):
    """
    Checks that a specific msg in found inside logger.
    This simply checks if the logger contains the exact
    msg string and doesn't not check a regex.

    We only check the logger on rank 0 because we only
    write on rank 0.
    """
    # TODO[BSE-4152]: support checking logs in spawn testing
    if bodo.tests.utils.test_spawn_mode_enabled:
        return
    if bodo.get_rank() == 0:
        if check_case:
            assert msg in stream.getvalue(), (
                f"Cannot find message in logging stream: '{msg}'"
            )
        else:
            assert msg.lower() in stream.getvalue().lower(), (
                f"Cannot find message in logging stream: '{msg}'"
            )


def check_logger_no_msg(stream, msg):
    """
    Checks that a specific msg is not found inside logger.
    This simply checks if the logger contains the exact
    msg string and doesn't not check a regex.

    We only check the logger on rank 0 because we only
    write on rank 0.
    """
    # TODO[BSE-4152]: support checking logs in spawn testing
    if bodo.tests.utils.test_spawn_mode_enabled:
        return
    if bodo.get_rank() == 0:
        assert msg not in stream.getvalue(), (
            f"Found find message in logging stream that should have been absent: '{msg}'"
        )
