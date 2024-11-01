# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""Utilities for Spawn Mode"""

import logging
from dataclasses import dataclass
from enum import Enum
from time import sleep

import pandas as pd
from pandas.core.arrays import ArrowExtensionArray

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


@dataclass
class DistributedReturnMetadata:
    result_id: str
    head: pd.DataFrame | pd.Series | ArrowExtensionArray
    nrows: int


def debug_msg(logger: logging.Logger, msg: str):
    """Send debug message to logger if Bodo verbose level 2 is enabled"""
    if bodo.user_logging.get_verbose_level() >= 2:
        logger.debug(msg)
