# Copyright (C) 2019 Bodo Inc.
# Turn on tracing for all tests in this file.
import os
import time
from tempfile import TemporaryDirectory

import pytest
from mpi4py import MPI

import bodo
from bodo.tests.tracing_utils import TracingContextManager
from bodo.utils import tracing
from bodo.utils.tracing import TRACING_MEM_WARN
from bodo.utils.typing import BodoWarning

# Enable tracing for all test in this file. This should be fine because
# runtest.py ensure our tests run 1 file at a time so we will avoid any
# unnecessary tracing for other tests.
#
# Regardless this should be the only place in the test suite that calls
# tracing.start(), so we shouldn't have any issues with other tests.
os.environ["BODO_TRACE_DEV"] = "1"


def test_tracing():
    """Test tracing utility"""

    rank = bodo.get_rank()

    # Test normal operation of tracing with synced and non-synced events
    def impl1():
        with TemporaryDirectory() as tempdir:
            tracing.start()

            if rank == 0:
                ev1 = tracing.Event("event1", is_parallel=False, sync=False)
                ev1.finalize()
            ev2 = tracing.Event("event2", sync=False)
            ev2.finalize()
            tracing.dump(f"{tempdir}/bodo_trace.json")

    impl1()

    # Test that tracing does not hang due to different number of `_bodo_aggr` events
    def impl2():
        with TemporaryDirectory() as tempdir:
            tracing.start()

            if rank == 0:
                ev1 = tracing.Event("event1", sync=False)
                ev1.finalize()
            ev2 = tracing.Event("event2", sync=False)
            ev2.finalize()
            tracing.dump(f"{tempdir}/bodo_trace.json")

    if bodo.get_size() == 1:
        impl2()
    else:
        err_msg = (
            "Bodo tracing programming error: "
            "Cannot perform tracing dump because there are a different "
            "number of aggregated tracing events on each rank."
        )
        with pytest.raises(RuntimeError, match=err_msg):
            impl2()

    tracing.reset()
    tracing.stop()


# Recwarn is a built-in PyTest fixture that captures all warnings and lets us
# check them within the test. We can't use pytest.warns because it will raise
# an Exception when no warning is raised on ranks != 0
def test_tracing_warning(recwarn):
    """Test if Memory Warning is Raised on Rank 0"""

    def impl1():
        with TemporaryDirectory() as tempdir:
            tracing.start()
            ev = tracing.Event("event", sync=False)
            ev.finalize()
            tracing.dump(f"{tempdir}/bodo_trace.json")

    impl1()

    tracing.reset()
    tracing.stop()

    comm = MPI.COMM_WORLD

    if comm.Get_rank() == 0:
        rank_no_warns = True
        rank_0_warn = any(
            isinstance(warning.message, BodoWarning)
            and warning.message.args[0] == TRACING_MEM_WARN
            for warning in recwarn
        )
    else:
        rank_0_warn = False
        rank_no_warns = all(
            not isinstance(warning.message, BodoWarning)
            or warning.message.args[0] != TRACING_MEM_WARN
            for warning in recwarn
        )

    rank_0_warn = comm.bcast(rank_0_warn)
    assert rank_0_warn, "Memory warning was not raised on rank 0"

    rank_no_warns = comm.allreduce(rank_no_warns, MPI.BAND)
    assert rank_no_warns, "Memory warning was raised on ranks != 0"


def test_resumable_event():
    """Test that a resumable event and non-resumable event come up with similar numbers"""

    # Test normal operation of tracing with synced and non-synced events
    def impl1():
        tracing_info = TracingContextManager()
        with tracing_info:
            resumable_event = tracing.ResumableEvent("resumable")
            event1 = tracing.Event("simple")

            with resumable_event.iteration():
                time.sleep(0.1)

            event1.finalize()

            time.sleep(0.1)

            event2 = tracing.Event("simple2")

            with resumable_event.iteration():
                time.sleep(0.1)

            event2.finalize()
            resumable_event.finalize()

        simple1 = tracing_info.get_event("simple", 0)
        simple2 = tracing_info.get_event("simple2", 0)
        resumable = tracing_info.get_event("resumable", 0)

        assert resumable["args"][
            "resumable"
        ], "No resumable attribute on resumable event"
        assert (
            resumable["args"]["iteration_count"] == 2
        ), "Resumable event iteration count is wrong"

        # the first and second simple event occur around the two iterations of resumable, thus they should take longer
        assert (
            simple1["dur"] + simple2["dur"] >= resumable["tdur"]
        ), "Resumable event duration is incorrect"
        # However, the simple events occur in between the construction and finalization of resumable, so the total duration should be longer
        assert (
            simple1["dur"] + simple2["dur"] <= resumable["dur"]
        ), "Resumable event duration is incorrect"

    impl1()
    tracing.reset()
    tracing.stop()
