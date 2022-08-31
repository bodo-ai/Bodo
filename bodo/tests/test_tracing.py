# Copyright (C) 2019 Bodo Inc.
# Turn on tracing for all tests in this file.
import os
from tempfile import TemporaryDirectory

import pytest

import bodo
from bodo.utils import tracing

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
                ev1.finalize(sync=False)
            ev2 = tracing.Event("event2", sync=False)
            ev2.finalize(sync=False)
            tracing.dump(f"{tempdir}/bodo_trace.json")

    impl1()

    # Test that tracing does not hang due to different number of `_bodo_aggr` events
    def impl2():
        with TemporaryDirectory() as tempdir:
            tracing.start()

            if rank == 0:
                ev1 = tracing.Event("event1", sync=False)
                ev1.finalize(sync=False)
            ev2 = tracing.Event("event2", sync=False)
            ev2.finalize(sync=False)
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
