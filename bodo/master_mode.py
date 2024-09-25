import gc
import inspect
import sys
import types as pytypes

import bodo

master_mode_on = False
MASTER_RANK = 0


class MasterModeDispatcher(object):
    """Encapsulates a numba dispatcher but is not a dispatcher itself
    currently"""

    def __init__(self, dispatcher):  # pragma: no cover
        self.dispatcher = dispatcher

    def __call__(self, *args, **kwargs):  # pragma: no cover
        assert bodo.get_rank() == MASTER_RANK
        return master_wrapper(self.dispatcher, *args, **kwargs)

    def __getstate__(self):  # pragma: no cover
        assert bodo.get_rank() == MASTER_RANK
        return self.dispatcher.py_func

    def __setstate__(self, state):  # pragma: no cover
        assert bodo.get_rank() != MASTER_RANK
        pyfunc = state
        # get the decorator
        # TODO this needs to be made more robust
        decorator_src_line = inspect.getsourcelines(pyfunc)[0][0]
        assert decorator_src_line.startswith(
            "@bodo.jit"
        ) or decorator_src_line.startswith("@jit")
        decorator = eval(decorator_src_line[1:])
        self.dispatcher = decorator(pyfunc)


def worker_loop():  # pragma: no cover
    # this is where all the workers (processes != MASTER) enter a loop to
    # wait for and execute commands from master
    assert bodo.get_rank() != MASTER_RANK
    comm = MPI.COMM_WORLD
    while True:
        command = comm.bcast(None, root=MASTER_RANK)
        if command[0] == "exec":
            # unpickle the python function (this is not the dispatcher)
            pyfunc = pickle.loads(command[1])

            for objname, obj in list(pyfunc.__globals__.items()):
                if isinstance(obj, MasterModeDispatcher):
                    pyfunc.__globals__[objname] = obj.dispatcher

            if pyfunc.__module__ not in sys.modules:
                sys.modules[pyfunc.__module__] = pytypes.ModuleType(pyfunc.__module__)

            # get the decorator
            # TODO this needs to be made more robust
            decorator_src_line = inspect.getsourcelines(pyfunc)[0][0]
            assert decorator_src_line.startswith(
                "@bodo.jit"
            ) or decorator_src_line.startswith("@jit")
            decorator = eval(decorator_src_line[1:])

            # apply decorator to get the dispatcher
            func = decorator(pyfunc)

            # receive the arguments and keyword arguments with which to
            # call the bodo function on this worker
            pos_arg_distribution = command[2]
            kwargs_distribution = command[3]

            real_args = []
            for send_mode in pos_arg_distribution:
                if send_mode == "scatter":
                    real_args.append(bodo.scatterv(None))
                elif send_mode == "bcast":
                    real_args.append(comm.bcast(None, root=MASTER_RANK))

            real_kwargs = {}
            for argname, send_mode in kwargs_distribution.items():
                if send_mode == "scatter":
                    real_kwargs[argname] = bodo.scatterv(None)
                elif send_mode == "bcast":
                    real_kwargs[argname] = comm.bcast(None, root=MASTER_RANK)

            # call bodo function
            retval = func(*real_args, **real_kwargs)

            # send result to MASTER
            # TODO: find the right signature if several
            if (
                retval is not None
                and func.overloads[func.signatures[0]].metadata["is_return_distributed"]
            ):
                bodo.gatherv(retval)
            # delete and garbage collect everything now (otherwise objects
            # won't be deleted until the next command replaces them)
            del (
                command,
                pyfunc,
                func,
                decorator,
                pos_arg_distribution,
                kwargs_distribution,
                real_args,
                real_kwargs,
                retval,
            )
            gc.collect()
        elif command[0] == "exit":
            exit()
    assert False


def master_wrapper(func, *args, **kwargs):  # pragma: no cover
    """wrapper for bodo functions in master mode (only on MASTER)
    func is a numba dispatcher"""

    # NOTE: this code cannot be inside the bodo compiled function because a
    # bodo function might call other bodo functions and only the top-level one
    # should execute this wrapper

    comm = MPI.COMM_WORLD

    # first determine which of the received arguments need to be
    # scattered or broadcast to workers
    if {
        "all_args_distributed",
        "all_args_distributed_block",
        "all_args_distributed_varlength",
    } & set(func.targetoptions.keys()):
        # scatter all args
        pos_arg_distribution = ["scatter" for _ in range(len(args))]
        kwargs_distribution = {argname: "scatter" for argname in kwargs.keys()}
    else:
        argnames = func.py_func.__code__.co_varnames
        options = func.targetoptions

        def get_distribution(argname):
            if argname in options.get("distributed", []) or argname in options.get(
                "distributed_block", []
            ):
                return "scatter"
            else:
                return "bcast"

        pos_arg_distribution = [
            get_distribution(argname) for argname in argnames[: len(args)]
        ]
        kwargs_distribution = {
            argname: get_distribution(argname) for argname in kwargs.keys()
        }

    # pickle the Python function (not the numba-generated dispatcher)
    # with cloudpickle. Note that cloudpickle will automatically
    # include globals that are used by the function in the serialized
    # data. Some of these globals could be bodo functions (encapsulated inside
    # MasterModeDispatcher). MasterModeDispatcher has custom (un)pickle code
    # which is executed by cloudpickle
    pickled_func = pickle.dumps(func.py_func)
    # broadcast the execute command to workers
    comm.bcast(["exec", pickled_func, pos_arg_distribution, kwargs_distribution])

    # build list of real arguments and keyword arguments with which
    # to call the bodo function (some arguments could be distributed
    # and I only need my chunk)
    real_args = []
    for arg, send_mode in zip(args, pos_arg_distribution):
        if send_mode == "scatter":
            real_args.append(bodo.scatterv(arg))
        elif send_mode == "bcast":
            comm.bcast(arg)
            real_args.append(arg)

    real_kwargs = {}
    for argname, arg in kwargs.items():
        send_mode = kwargs_distribution[argname]
        if send_mode == "scatter":
            real_kwargs[argname] = bodo.scatterv(arg)
        elif send_mode == "bcast":
            comm.bcast(arg)
            real_kwargs[argname] = arg

    # TODO: this needs to be revisited, tested more and made more robust.
    # For one, it needs to happen recursively: bodo functions might call
    # bodo functions which might call other bodo functions
    restore_globals = []
    for objname, obj in list(func.py_func.__globals__.items()):
        if isinstance(obj, MasterModeDispatcher):
            restore_globals.append(
                (func.py_func.__globals__, objname, func.py_func.__globals__[objname])
            )
            func.py_func.__globals__[objname] = obj.dispatcher

    # call bodo function
    retval = func(*real_args, **real_kwargs)

    for glbs, objname, obj in restore_globals:
        glbs[objname] = obj

    # collect result on MASTER
    # TODO: find the right signature if several
    if (
        retval is not None
        and func.overloads[func.signatures[0]].metadata["is_return_distributed"]
    ):
        retval = bodo.gatherv(retval)
    return retval


def init_master_mode():  # pragma: no cover
    if bodo.get_size() == 1:
        return

    global master_mode_on
    assert (
        master_mode_on is False
    ), "init_master_mode can only be called once on each process"
    master_mode_on = True

    # Python 3.8+ required for cloudpickle_fast
    assert sys.version_info[:2] >= (3, 8), "Python 3.8+ required for master mode"

    # cannot import jit at module top level since jit does not exist yet
    from bodo import jit

    globals()["jit"] = jit
    # we only import cloudpickle and mpi4py if master mode is needed
    import cloudpickle

    from bodo.mpi4py import MPI

    globals()["pickle"] = cloudpickle
    globals()["MPI"] = MPI

    def master_exit():
        """this is called at exit on MASTER to tell workers to exit. this
        function is meant to be registered with atexit"""
        MPI.COMM_WORLD.bcast(["exit"])

    if bodo.get_rank() == MASTER_RANK:
        import atexit

        atexit.register(master_exit)
    else:
        worker_loop()
