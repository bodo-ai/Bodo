# Copyright (C) 2019 Bodo Inc. All rights reserved.
import os
import shutil
from contextlib import contextmanager
import bodo


@bodo.jit
def get_rank():  # pragma: no cover
    return bodo.libs.distributed_api.get_rank()


@bodo.jit
def barrier():  # pragma: no cover
    return bodo.libs.distributed_api.barrier()


@contextmanager
def ensure_clean(filename):
    """deletes filename if exists after test is done.
    """
    try:
        yield
    finally:
        try:
            # wait for all ranks to complete
            barrier()
            # delete on rank 0
            if (
                get_rank() == 0
                and os.path.exists(filename)
                and os.path.isfile(filename)
            ):
                os.remove(filename)
        except Exception as e:
            print("Exception on removing file: {error}".format(error=e))


@contextmanager
def ensure_clean_dir(dirname):
    """deletes filename if exists after test is done.
    """
    try:
        yield
    finally:
        try:
            # wait for all ranks to complete
            barrier()
            # delete on rank 0
            if (
                get_rank() == 0 
                and os.path.exists(dirname) 
                and os.path.isdir(dirname)
            ):
                shutil.rmtree(dirname)
        except Exception as e:
            print("Exception on removing directory: {error}".format(error=e))
