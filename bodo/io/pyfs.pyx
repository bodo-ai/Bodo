"""
Here we define a subclass of PyFileSystem (which is Cython class defined in Arrow)
so as to be able to expose some parts of it to C++ using the `public` Cython
keyword. We need this to be able to get the shared_ptr<FileSystem> from
PyFileSystem
"""
cimport pyarrow.lib
from pyarrow._fs cimport PyFileSystem
from pyarrow.includes.libarrow_fs cimport CFileSystem
from pyarrow.includes.common cimport *

cdef public class PyFileSystemBodo(PyFileSystem) [object c_PyFileSystemBodo, type c_PyFileSystemBodo_t]:

    def __init__(self, handler):
        PyFileSystem.__init__(self, handler)


cdef public shared_ptr[CFileSystem] get_cpp_fs(PyFileSystemBodo fs):
    return fs.unwrap()
