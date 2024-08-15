cimport pyarrow.lib
from libcpp.memory cimport shared_ptr
from pyarrow._dataset cimport Dataset, Fragment
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow_dataset cimport CDataset, CFragment

from pyarrow._compute cimport Expression
from pyarrow.includes.libarrow cimport CExpression

from pyarrow._fs cimport FileSystem
from pyarrow.includes.libarrow_fs cimport CFileSystem

cdef api bint pyarrow_is_expression(object expression):
    return isinstance(expression, Expression)

cdef api CExpression pyarrow_unwrap_expression(object expression):
    cdef Expression e
    if pyarrow_is_expression(expression):
        e = <Expression>(expression)
        return e.expr

    return CMakeScalarExpression(
        <shared_ptr[CScalar]> make_shared[CBooleanScalar](True)
    )


cdef api object pyarrow_wrap_expression(
        const CExpression& cexpr):
    cdef Expression expr = Expression.__new__(Expression)
    expr.init(cexpr)
    return expr

cdef api bint pyarrow_is_dataset(object dataset):
    return isinstance(dataset, Dataset)

cdef api shared_ptr[CDataset] pyarrow_unwrap_dataset(object dataset):
    cdef Dataset d
    if pyarrow_is_dataset(dataset):
        d = <Dataset>(dataset)
        return d.unwrap()

    return shared_ptr[CDataset]()

cdef api object pyarrow_wrap_dataset(const shared_ptr[CDataset]& cdataset):
    cdef Dataset dataset = Dataset.__new__(Dataset)
    dataset.init(cdataset)
    return dataset


cdef api bint pyarrow_is_fragment(object frag):
    return isinstance(frag, Fragment)

cdef api shared_ptr[CFragment] pyarrow_unwrap_fragment(object frag):
    cdef Fragment f
    if pyarrow_is_fragment(frag):
        f = <Fragment>(frag)
        return f.unwrap()

    return shared_ptr[CFragment]()

cdef api object pyarrow_wrap_fragment(const shared_ptr[CFragment]& cfrag):
    cdef Fragment frag = Fragment.__new__(Fragment)
    frag.init(cfrag)
    return frag 


cdef api bint pyarrow_is_filesystem(object filesystem):
    return isinstance(filesystem, FileSystem)

cdef api shared_ptr[CFileSystem] pyarrow_unwrap_filesystem(object filesystem):
    cdef FileSystem f
    if pyarrow_is_filesystem(filesystem):
        f = <FileSystem>(filesystem)
        return f.unwrap()

    return shared_ptr[CFileSystem]()


cdef api object pyarrow_wrap_filesystem(const shared_ptr[CFileSystem]& cfilesystem):
    cdef FileSystem filesystem = FileSystem.__new__(FileSystem)
    filesystem.init(cfilesystem)
    return filesystem 
