"""
Here we define a subclass of PyFileSystem (which is Cython class defined in Arrow)
so as to be able to expose some parts of it to C++ using the `public` Cython
keyword. We need this to be able to get the shared_ptr<FileSystem> from
PyFileSystem
"""
cimport pyarrow.lib
from libcpp.memory cimport shared_ptr
from pyarrow._fs cimport PyFileSystem, FileSystem
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow_fs cimport CFileSystem


cdef public class PyFileSystemBodo(PyFileSystem) [object c_PyFileSystemBodo, type c_PyFileSystemBodo_t]:
    def __init__(self, handler):
        PyFileSystem.__init__(self, handler)


cdef public shared_ptr[CFileSystem] get_cpp_fs(PyFileSystemBodo fs):
    return fs.unwrap()

def get_pyarrow_fs_from_ptr(long fs_ptr_long):
    cdef CFileSystem *fs_ptr = <CFileSystem*>fs_ptr_long
    return FileSystem.wrap(shared_ptr[CFileSystem](fs_ptr))
