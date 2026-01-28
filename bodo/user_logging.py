"""
Contains all of the helper functions and state information
for user logging. This includes helpers for setting debug
targets with the verbose flag.
"""

import logging
import sys

# Create the default logger
_default_logger = logging.getLogger("Bodo Default Logger")
_default_logger.setLevel(logging.DEBUG)


# Create the default handler
default_handler = logging.StreamHandler(sys.stdout)
default_handler.setLevel(logging.DEBUG)

# create formatter
default_formater = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

default_handler.setFormatter(default_formater)
_default_logger.addHandler(default_handler)

_bodo_logger = _default_logger

_default_verbose_level = 0
_bodo_verbose_level = _default_verbose_level


def restore_default_bodo_verbose_level():
    """
    Restore the verbose level back to the default level. This
    is primarily intended for internal usage and testing.
    """
    set_verbose_level(_default_verbose_level)


def get_verbose_level():
    """
    Returns the current verbose level in Bodo.
    """
    return _bodo_verbose_level


def set_verbose_level(level):
    """
    User facing function to set the verbose level in Bodo.
    Level should be an integer >= 0. Larger values are intended
    to output more detailed information.
    """
    global _bodo_verbose_level
    if not isinstance(level, int) or level < 0:
        raise TypeError("set_verbose_level(): requires an integer level >= 0")

    _bodo_verbose_level = level

    # If BodoSQL is imported, update its logging level as well.
    # NOTE: Avoiding extra BodoSQL import if not imported already to reduce overheads.
    # BodoSQL picks up the logging level during import.
    # TODO: move to bodosql if possible
    if "bodosql" in sys.modules:
        import bodosql.py4j_gateway

        bodosql.py4j_gateway.configure_java_logging(level)


def restore_default_bodo_verbose_logger():
    """
    Restore the logger back to the default logger. This
    is primarily intended for internal usage and testing.
    """
    set_bodo_verbose_logger(_default_logger)


def get_current_bodo_verbose_logger():
    """
    Returns the current bodo_logger.
    """
    return _bodo_logger


def set_bodo_verbose_logger(logger):
    """
    User facing function to set the logger used when
    setting the verbose flag in JIT. This logger should be
    a fully initialized logging.Logger instance.

    This code is intended to be called from regular Python,
    ideally when you initialize your JIT module. All verbose
    messages are written with the INFO message.

    Note: Bodo only logs verbose info with rank 0, so
    you only need to set the target on rank 0.
    """
    global _bodo_logger
    if not isinstance(logger, logging.Logger):
        raise TypeError(
            "set_bodo_verbose_logger(): requires providing an initialized  logging.Logger type"
        )
    _bodo_logger = logger


def log_message(header, msg, *args, **kws):
    """
    Logs a message to the Bodo logger using logger.info.
    Bodo only information on rank 0 to avoid issues with
    file collisions.
    """
    # NOTE: avoiding bodo.get_rank() which uses the compiler to allow using the logger
    # inside the compiler
    from mpi4py import MPI

    # Avoid putting a mutable value as a default argument.
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger = get_current_bodo_verbose_logger()
        # Surround messages in some formatting
        equal_str = "\n" + ("=" * 80)
        final_msg = "\n".join([equal_str, header.center(80, "-"), msg, equal_str])
        logger.info(final_msg, *args, **kws)
