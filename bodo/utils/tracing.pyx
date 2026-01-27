import atexit
import os
import time
import warnings

import numpy as np

from bodo import BodoWarning

from mpi4py.libmpi cimport (
    MPI_COMM_WORLD,
    MPI_Barrier,
    MPI_Comm_rank,
    MPI_Comm_size,
)

# We don't want to expose these variables outside this module (cdef
# variables can't be accessed directly from Python). Also, variables with
# basic C types allow for code with less overhead compared to using Python objects
cdef bint TRACING = 0  # 1 if started tracing, else 0
cdef list traceEvents = []  # list of tracing events
cdef str trace_filename = "bodo_trace.json" if BODO_DEV_BUILD else "bodo_trace.dat"
cdef object time_start = time.time()  # time at which tracing starts

TRACING_MEM_WARN = "Tracing is still experimental and has been known to have memory related issues. In our testing we have at times seen 1-2 GB be used per rank to support tracing. If you are running out of memory or you are attempted to document the memory footprint for a workload, please disable tracing."

# Events are stored in Google's trace event format:
# https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit?pli=1
# as a dict, and will be converted to JSON when dumping


# BODO_DEV_BUILD is a compile-time constant passed from setup.py that tells
# if we are in development mode (Bodo built with `python setup.py develop`)


cdef inline tracing_supported():
    if BODO_DEV_BUILD:
        return True
    else:
        key1 = b"{\xd5'*\xc1N\x90\xf1\xf9\xbfy\xc7\xf4\xc0"
        key2 = b"9\x9ace\x9e\x1a\xc2\xb0\xba\xfa&\x83\xb1\x96"
        # bitwise_xor(key1, key2) == b"BODO_TRACE_DEV"
        # BODO_TRACE_DEV is an undocumented environment variable that we don't
        # provide to everyone. This is a way of not revealing the name in the
        # the compiled binary
        key = bytes(a ^ b for (a, b) in zip(key1, key2)).decode()
        return os.environ.get(key, None) == "1"


def reset(trace_fname=None):
    """ Set time_start to current time, clear any stored events
        trace_fname is the file name to store tracing information on call to dump()
    """
    cdef int rank
    if tracing_supported():
        global traceEvents, trace_filename, time_start, TRACING
        traceEvents = []
        if trace_fname is not None:
            trace_filename = trace_fname
        MPI_Barrier(MPI_COMM_WORLD)
        time_start = time.time()
        TRACING = 1
        MPI_Comm_rank(MPI_COMM_WORLD, &rank)
        if rank == 0:
            traceEvents.append(
                {
                    "name": "START",
                    "ph": "i",  # "Instant" event
                    "ts": get_timestamp(),
                    "pid": rank,
                }
            )


def start(trace_fname=None):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        warnings.warn(TRACING_MEM_WARN, BodoWarning)

    reset(trace_fname)


def stop():
    global TRACING
    TRACING = 0


def is_tracing():
    return TRACING


cdef convert_np_scalars_to_python(event):
    if "args" in event:
        for k, val in list(event["args"].items()):
            if isinstance(val, np.generic):
                # convert to Python type to avoid JSON serialization errors
                event["args"][k] = val.item()


def dump(fname=None, clear_traces=True):
    """Dump current traces to JSON file"""
    cdef int rank, num_ranks, num_nodes
    if tracing_supported():
        global traceEvents
        from bodo.libs.distributed_api import get_num_nodes
        num_nodes = get_num_nodes()
        MPI_Comm_rank(MPI_COMM_WORLD, &rank)
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks)
        if num_ranks > 1:
            aggregate_events()
        else:
            for e in traceEvents:
                convert_np_scalars_to_python(e)
        if rank == 0 and len(traceEvents) > 0:
            if fname is None:
                fname = trace_filename
            traceEvents.append(
                {
                    "name": "END",
                    "ph": "i",  # "Instant" event
                    "ts": get_timestamp(),
                    "pid": rank,
                }
            )
            trace_obj = {"num_ranks": num_ranks}
            trace_obj["num_nodes"] = num_nodes
            import json

            import bodo
            trace_obj["bodo_version"] = bodo.__version__
            for var in os.environ:
                if "MPI" in var or var.startswith("FI_") or var.startswith("BODO"):
                    trace_obj[var] = os.environ[var]
            trace_obj["traceEvents"] = traceEvents
            if BODO_DEV_BUILD:
                with open(fname, "w") as f:
                    json.dump(trace_obj, f)
            else:
                # write obscured traces by compressing with zlib without zlib
                # header and writing to a binary file (the `file` command will
                # only identify as "data")
                # To decompress, use buildscripts/decompress_traces.py
                with open(fname, "wb") as f:
                    import zlib

                    # wbits=-15 won't add a header, so output will just be a
                    # raw binary stream
                    c = zlib.compressobj(9, wbits=-15)
                    out_data = c.compress(json.dumps(trace_obj).encode())
                    out_data += c.flush()
                    f.write(out_data)
        if clear_traces:
            traceEvents = []


# cdef functions aren't visible from Python, and have less overhead when called
# from inside this module (from C)
cdef inline object get_timestamp():
    """Get current timestamp (from time_start)"""
    return (time.time() - time_start) * 1e6  # convert to us


# On exit, call dump if tracing is set
if is_tracing():
    atexit.register(dump)


cdef aggregate_helper(values, arg_name, out):
    index_min = int(np.argmin(values))
    index_max = int(np.argmax(values))
    out["args"][arg_name + "_min"] = values[index_min].item()
    out["args"][arg_name + "_avg"] = values.mean().item()
    out["args"][arg_name + "_max"] = values[index_max].item()
    out["args"][arg_name + "_min_rank"] = index_min
    out["args"][arg_name + "_max_rank"] = index_max


cdef generic_aggregate_func(object traces_all):
    """Aggregate a single event from traces collected from all ranks"""
    result = traces_all[0]
    if "args" in result:
        args = list(result["args"].keys())
        for arg in args:
            if arg.startswith("g_") or isinstance(result["args"][arg], (str, list)) or arg == "resumable":
                # attributes called g_xxx are global and don't need aggregation
                # and we don't aggregate string or list values
                # "resumable" identifies the type of event and doesn't need aggregation
                continue
            # We use .get and default to 0 since it's possible that some ranks don't have
            # some attributes.
            values = np.array([t["args"].get(arg, 0) for t in traces_all])
            try:
                aggregate_helper(values, arg, result)
                del result["args"][arg]
            except:
                if "tracing_errors" not in result["args"]:
                    result["args"]["tracing_errors"] = []
                result["args"]["tracing_errors"].append(f"aggregate error {arg}")
        convert_np_scalars_to_python(result)
    return result


cdef aggregate_events():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    cdef int rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank)
    start_ts = get_timestamp()
    global traceEvents

    # The event aggregation code below hangs if there are a different number
    # of events with `_bodo_aggr == 1` on each rank.
    # Detect that error condition beforehand and raise a programming error.
    from bodo.libs.distributed_api import Reduce_Type, dist_reduce

    err = None  # Forward declaration
    aggr_events = [e["name"] for e in traceEvents if "_bodo_aggr" in e]
    num_aggr_events = len(aggr_events)
    # max(x) is equivalent to -min(-x).
    # This allows using just one MPI operation to compute both the max and min.
    min_max_nevents = dist_reduce(
        np.array([-num_aggr_events, num_aggr_events]),
        np.int32(Reduce_Type.Min.value),
    )
    if (-min_max_nevents[0] != min_max_nevents[1]):
        from bodo.libs.distributed_api import allgatherv

        all_events = allgatherv(np.array(aggr_events, dtype=object))
        missing_events = set(all_events).difference(set(aggr_events))
        missing_events = sorted(list(missing_events))
        if len(missing_events) == 0:
            missing_err = f"No events are missing from rank {rank}"
        else:
            missing_err = f"Events {missing_events} are missing from rank {rank}"
        missing_errs = allgatherv(np.array([missing_err], dtype=object))

        err = RuntimeError(
            "Bodo tracing programming error: "
            "Cannot perform tracing dump because there are a different "
            "number of aggregated tracing events on each rank.\n"
            + "\n".join(missing_errs)
        )

    # If any rank raises an exception, re-raise that error on all non-failing
    # ranks to prevent deadlock on future MPI collective ops.
    # We use allreduce with MPI.MAXLOC to communicate the rank of the lowest
    # failing process, then broadcast the error backtrace across all ranks.
    err_on_this_rank = int(err is not None)
    err_on_any_rank, failing_rank = comm.allreduce(
        (err_on_this_rank, comm.Get_rank()), op=MPI.MAXLOC
    )
    if err_on_any_rank:
        if comm.Get_rank() == failing_rank:
            lowest_err = err
        else:
            lowest_err = None
        lowest_err = comm.bcast(lowest_err, root=failing_rank)

        # Each rank that already has an error will re-raise their own error, and
        # any rank that doesn't have an error will re-raise the lowest rank's error.
        if err_on_this_rank:
            raise err
        else:
            raise lowest_err

    # End of error-handling code, begin event aggregation
    for i, e in enumerate(traceEvents):
        if not "_bodo_aggr" in e:
            continue
        e.pop("_bodo_aggr")
        if rank == 0:
            events_all = comm.gather(e)
            new_e = generic_aggregate_func(events_all)
            durations = np.array([t["dur"] for t in events_all])
            new_e["dur"] = durations.max().item()
            aggregate_helper(durations, "dur", new_e)
            traceEvents[i] = new_e
        else:
            comm.gather(e)
    if rank == 0 and len(traceEvents) > 0:
        traceEvents.append({
            "name": "dump_traces",
            "pid": rank,
            "ts": start_ts,
            "ph": "X",  # "Complete" event
            "dur": get_timestamp() - start_ts
        })
    if rank != 0:
        traceEvents = []


ITERATION_STACK = []


cdef class EventBase:

    cdef bint is_parallel  # True if refers to parallel event, False if replicated
    cdef bint sync # Should we sync the event
    cdef dict trace  # dictionary with event data

    def __cinit__(self, name not None, bint is_parallel=1, bint sync=1, bint is_batchable=1): # Base
        """ Start event named 'name'
            is_parallel=True indicates a parallel event, otherwise replicated
            If sync=True do a barrier before creating the event
        """
        if TRACING == 0:
            return  # nop
        if sync and is_parallel:
            # wait for all processes before starting event
            MPI_Barrier(MPI_COMM_WORLD)
        start_ts = get_timestamp()
        cdef int rank
        MPI_Comm_rank(MPI_COMM_WORLD, &rank)
        self.trace = {
            "name": name,
            "pid": rank,
            "ts": start_ts,
            "ph": "X",  # "Complete" event
            "args": {},
        }
        self.is_parallel = is_parallel
        self.sync = sync

    def add_attribute(self, str name not None, value):
        """Add attribute 'name' with value 'value' to event"""
        if TRACING == 0:
            return  # nop
        # For replicated events we don't need to add attributes on
        # ranks != 0, but the common case is parallel events so we
        # add attributes always to avoid overhead of checking
        self.trace["args"][name] = value

    def finalize(self, bint aggregate=True):
        """ Finalize event
            If aggregate=True, aggregate the info of the event across all ranks
        """
        if TRACING == 0:
            return  # nop
        self.trace["dur"] = get_timestamp() - self.trace["ts"]
        if self.sync and self.is_parallel:
            # wait for all processes to finish event
            MPI_Barrier(MPI_COMM_WORLD)
        if aggregate and self.is_parallel:
            self.trace["_bodo_aggr"] = 1


cdef class Event(EventBase):

    cdef object parent_event

    def __cinit__(self, name not None, bint is_parallel=1, bint sync=1, bint is_batchable=1):  # Event
        """Create an event and determine if we're going to batch this event within an existing iteration."""
        super().__init__(name, is_parallel, sync)
        if TRACING == 0:
            return  # nop
        if is_batchable and len(ITERATION_STACK) > 0:
            self.parent_event = ITERATION_STACK[-1].get_or_create_child_event(name, is_parallel)
            self.parent_event.start_iteration()
        else:
            self.parent_event = None

    def finalize(self, bint aggregate=True):
        super().finalize(aggregate)

        if self.parent_event is not None:
            self.parent_event.end_iteration()
        else:  # No parent, behave as normal
            traceEvents.append(self.trace)
