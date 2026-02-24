import os
import shutil
import warnings
import weakref
from tempfile import gettempdir
from uuid import uuid4

from mpi4py import MPI


class LazyTemporaryDirectory:
    """
    Based on tempfile.TemporaryDirectory, except the directory
    is not created immediately, and the randomness of the
    name of directory is synchronized between all ranks.
    The former is useful for cases such as creating the
    temporary directory for core-site.xml for ADLS (Snowflake write
    or regular use) since it may not get used, in which case we don't
    want the overhead of filesystem operations.
    The latter is useful for managing temporary directories in a
    parallel setting.
    Original TemporaryDirectory:
    https://github.com/python/cpython/blob/9c7b4bd1646f2170247f88cf59936740d9c4c004/Lib/tempfile.py#L805
    """

    def __init__(self, ignore_cleanup_errors=False, is_parallel=True):
        """
        Unlike tempfile.TemporaryDirectory
        (https://github.com/python/cpython/blob/9c7b4bd1646f2170247f88cf59936740d9c4c004/Lib/tempfile.py#L817),
        this only creates the directory name, and not the directory itself.
        That functionality is now in the `initialize` function.
        The name is common on all ranks if `is_parallel=True`.

        Args:
            ignore_cleanup_errors (bool, optional):
                See tempfile.TemporaryDirectory. Defaults to False.
            is_parallel (bool, optional):
                Whether or not this directory will be used in parallel by all ranks.
                Defaults to True.
        """
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self.initialized = False
        self.is_parallel = is_parallel
        self.active_rank = False
        if self.is_parallel:
            # Create the random temp directory name
            # on rank 0 and broadcast it to all other
            # ranks
            comm = MPI.COMM_WORLD
            random_id = None
            if comm.Get_rank() == 0:
                random_id = str(uuid4())
            random_id = comm.bcast(random_id)
        else:
            random_id = str(uuid4())
        self.name = os.path.join(gettempdir(), random_id)

    def initialize(self):
        """
        Create the temporary directory, and set up the finalizer
        for its deletion during garbage collection.
        If `self.is_parallel=True`, this is only done
        on one rank on every node to minimize filesystem access
        conflicts.
        """
        if not self.initialized:
            if self.is_parallel:
                import bodo

                # Only one rank on each node needs to be involved for the
                # filesystem operations.
                self.active_rank = bodo.get_rank() in bodo.get_nodes_first_ranks()
            else:
                # If not parallel, each rank creates its own directory
                # and is responsible for filesystem operations.
                self.active_rank = True

            dir_creation_e = None
            if self.active_rank:
                # Create the directory.
                try:
                    os.mkdir(self.name, 0o700)
                except Exception as e:
                    dir_creation_e = e

            if self.is_parallel:
                # Synchronize and raise error in parallel
                comm = MPI.COMM_WORLD
                # This will be false on non active ranks
                local_err = isinstance(dir_creation_e, Exception)
                err = comm.allreduce(local_err, op=MPI.LOR)
                if err:
                    if local_err:
                        raise dir_creation_e
                    else:
                        raise Exception(
                            "Error during temporary directory creation. See exception on other ranks."
                        )
            elif isinstance(dir_creation_e, Exception):
                raise dir_creation_e

            if self.active_rank:
                # Set up a finalizer to delete the directory and its contents
                # during garbage collection of this object.
                # Read more about finalizers here:
                # https://docs.python.org/3/library/weakref.html#weakref.finalize
                self._finalizer = weakref.finalize(
                    self,
                    self._cleanup,
                    self.name,
                    warn_message=f"Implicitly cleaning up {self!r}",
                    ignore_errors=self._ignore_cleanup_errors,
                )
            else:
                # Attach a dummy finalizer to the object
                self._finalizer = weakref.finalize(self, lambda: None)
            self.initialized = True

    # Same as tempfile.TemporaryDirectory
    # (https://github.com/python/cpython/blob/9c7b4bd1646f2170247f88cf59936740d9c4c004/Lib/tempfile.py#L827)
    @classmethod
    def _rmtree(cls, name, ignore_errors=False):
        def onerror(func, path, exc_info):
            if issubclass(exc_info[0], PermissionError):

                def resetperms(path):
                    try:
                        os.chflags(path, 0)
                    except AttributeError:
                        pass
                    os.chmod(path, 0o700)

                try:
                    if path != name:
                        resetperms(os.path.dirname(path))
                    resetperms(path)

                    try:
                        os.unlink(path)
                    # PermissionError is raised on FreeBSD for directories
                    except (IsADirectoryError, PermissionError):
                        cls._rmtree(path, ignore_errors=ignore_errors)
                except FileNotFoundError:
                    pass
            elif issubclass(exc_info[0], FileNotFoundError):
                pass
            else:
                if not ignore_errors:
                    raise

        shutil.rmtree(name, onerror=onerror)

    # Same as tempfile.TemporaryDirectory
    # (https://github.com/python/cpython/blob/9c7b4bd1646f2170247f88cf59936740d9c4c004/Lib/tempfile.py#L858)
    @classmethod
    def _cleanup(cls, name, warn_message, ignore_errors=False):
        cls._rmtree(name, ignore_errors=ignore_errors)
        warnings.warn(warn_message, ResourceWarning)

    # Same as tempfile.TemporaryDirectory
    # (https://github.com/python/cpython/blob/9c7b4bd1646f2170247f88cf59936740d9c4c004/Lib/tempfile.py#L862)
    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name!r}>"

    # Same as tempfile.TemporaryDirectory
    # (https://github.com/python/cpython/blob/9c7b4bd1646f2170247f88cf59936740d9c4c004/Lib/tempfile.py#L865)
    def __enter__(self):
        self.initialize()
        return self.name

    # Same as tempfile.TemporaryDirectory
    # (https://github.com/python/cpython/blob/9c7b4bd1646f2170247f88cf59936740d9c4c004/Lib/tempfile.py#L868)
    def __exit__(self, exc, value, tb):
        self.cleanup()

    def cleanup(self):
        """
        Similar to tempfile.TemporaryDirectory
        (https://github.com/python/cpython/blob/9c7b4bd1646f2170247f88cf59936740d9c4c004/Lib/tempfile.py#L871),
        except the cleanup is done only if
        the object was initialized (i.e. self.initialized=True). In case of parallel
        mode, the cleanup is only done on the first rank on every node (i.e.
        if self.active_rank=True), to reduce filesystem operation conflicts.
        """
        if (
            self.initialized
            and self.active_rank
            and (self._finalizer.detach() or os.path.exists(self.name))
        ):
            self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)

    # TODO: uncomment after deprecating Python 3.8
    # New in version 3.9. https://docs.python.org/3/library/types.html#types.GenericAlias
    # __class_getitem__ = classmethod(types.GenericAlias)
