"""Utilities for Spawn Mode"""

import logging
import typing as pt
from enum import Enum
from time import sleep

import bodo.user_logging
from bodo.mpi4py import MPI


class CommandType(str, Enum):
    """
    Enum of the different types of commands that the spawner
    can send to the workers.
    """

    EXEC_FUNCTION = "exec"
    EXIT = "exit"
    BROADCAST = "broadcast"
    SCATTER = "scatter"
    GATHER = "gather"
    DELETE_RESULT = "delete_result"
    REGISTER_TYPE = "register_type"
    SET_CONFIG = "set_config"


def poll_for_barrier(comm: MPI.Comm, poll_freq: float | None = 0.1):
    """
    Barrier that doesn't busy-wait, but instead polls on a defined interval.
    The poll_freq kwarg controls the rate of polling. When set to None it will
    busy-wait.
    """
    # Start a non-blocking barrier operation
    req = comm.Ibarrier()
    if not poll_freq:
        # If polling is disabled, just wait for the barrier synchronously
        req.Wait()
    else:
        # Check if the barrier has completed and sleep if not.
        # TODO Add exponential backoff (e.g. start with 0.01 and go up
        # to 0.1). This could provide a faster response in many cases.
        while not req.Test():
            sleep(poll_freq)


def debug_msg(logger: logging.Logger, msg: str):
    """Send debug message to logger if Bodo verbose level 2 is enabled"""
    if bodo.user_logging.get_verbose_level() >= 2:
        logger.debug(msg)


class ArgMetadata(str, Enum):
    """Argument metadata to inform workers about other arguments to receive separately.
    E.g. broadcast or scatter a dataframe from spawner to workers.
    Used for DataFrame/Series/Index/array arguments.
    """

    BROADCAST = "broadcast"
    SCATTER = "scatter"
    LAZY = "lazy"


def set_global_config(config_name: str, config_value: pt.Any):
    """Set global configuration value by name (for internal testing use only)
    (e.g. "bodo.hiframes.boxing._use_dict_str_type")
    """
    # Get module and attribute sections of config_name
    # (e.g. "bodo.hiframes.boxing._use_dict_str_type" -> "bodo.hiframes.boxing"
    # and "_use_dict_str_type")
    c_split = config_name.split(".")
    attr = c_split[-1]
    mod_name = ".".join(c_split[:-1])
    locs = {}
    exec(f"import {mod_name}; mod = {mod_name}", globals(), locs)
    mod = locs["mod"]
    setattr(mod, attr, config_value)
