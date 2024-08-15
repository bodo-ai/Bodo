from pyarrow.includes.libarrow cimport *
from pyarrow.includes.libarrow_dataset cimport *
from pyarrow.includes.libarrow_fs cimport *

# Expressions
cdef public object pyarrow_wrap_expression(const CExpression& cexpr)
cdef public CExpression pyarrow_unwrap_expression(object expression)

# Dataset
cdef public object pyarrow_wrap_dataset(const shared_ptr[CDataset]& cdataset)
cdef public shared_ptr[CDataset] pyarrow_unwrap_dataset(object dataset)

# Fragment
cdef public object pyarrow_wrap_fragment(const shared_ptr[CFragment]& cfrag)
cdef public shared_ptr[CFragment] pyarrow_unwrap_fragment(object frag)


# FileSystem
cdef public object pyarrow_wrap_filesystem(const shared_ptr[CFileSystem]& cfilesystem)
cdef public shared_ptr[CFileSystem] pyarrow_unwrap_filesystem(object filesystem)
