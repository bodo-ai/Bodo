# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.vector cimport vector
from libcpp.string cimport string as c_string
from libcpp cimport bool as c_bool
from libcpp.memory cimport shared_ptr
from libc.stdint cimport int64_t
cimport pyarrow.lib
from pyarrow._fs cimport FileSystem
from pyarrow.includes.libarrow_fs cimport CFileSystem

from mpi4py import MPI
import bodo
import bodo.memory_cpp
import bodo.io.fs_io
from bodo.io.fs_io import parse_fpath, getfs, get_all_csv_json_data_files, get_compression_from_file_name


cdef extern from "_csv_json_reader.h" nogil:
    object csv_file_chunk_reader(
        const char *fname, c_bool is_parallel, int64_t *skiprows, int64_t nrows,
        c_bool header, const char *compression, const char *bucket_region,
        object storage_options, int64_t chunksize,
        c_bool is_skiprows_list, int64_t skiprows_list_len,
        c_bool pd_low_memory)

    c_bool update_csv_reader(object reader)

    void initialize_csv_reader(object reader)

    object json_file_chunk_reader(const char *fname, c_bool lines,
                                            c_bool is_parallel, int64_t nrows,
                                            const char *compression,
                                            const char *bucket_region,
                                            object storage_options);

    void init_stream_reader_type()
    void init_buffer_pool_ptr(int64_t ptr)


# Initialize Python type stream_reader_type defined in _csv_json_reader.cpp.
# Has to be called only once before using the stream_reader_type.
init_stream_reader_type()

# Initialize the buffer pool pointer to be the one from the main module and
# make sure we have a single buffer pool. Necessary since csv_json_reader is a separate module
# from bodo.memory_cpp.
init_buffer_pool_ptr(bodo.memory_cpp.default_buffer_pool_ptr())


cdef public void get_read_path_info(
        const char* file_path,
        c_string compression_pyarg,
        c_bool is_anon,
        # Outputs (workaround since Cython doesn't support C++ tuples)
        c_bool& is_remote_fs,
        c_string& compression,
        vector[c_string]& file_names,
        vector[int64_t]& file_sizes,
        shared_ptr[CFileSystem]& c_fs
    ):


    path, parsed_url, protocol = parse_fpath(file_path)

    is_remote_fs = (protocol != "")

    storage_options = {}
    if is_anon:
        storage_options["anon"] = True

    fs = getfs(path, protocol, storage_options=storage_options)

    # Get file names and sizes on rank 0 and propagate to all ranks
    metadata_or_err = None
    comm = MPI.COMM_WORLD
    if bodo.get_rank() == 0:
        try:
            err_msg = "Invalid data path path: " + path
            all_data_files = get_all_csv_json_data_files(fs, path, protocol, parsed_url, err_msg)
            f_sizes = [fs.get_file_info(p).size for p in all_data_files]
            metadata_or_err = (all_data_files, f_sizes)
        except Exception as e:
            metadata_or_err = e

    metadata_or_err = comm.bcast(metadata_or_err)
    if isinstance(metadata_or_err, Exception):
        raise metadata_or_err

    all_data_files, f_sizes = metadata_or_err

    c = <object>compression_pyarg
    uncompressed = "uncompressed"
    if c == "infer":
        inferred_compression = get_compression_from_file_name(all_data_files[0])
        compression = uncompressed if inferred_compression is None else inferred_compression
    else:
        compression = compression_pyarg

    for p in all_data_files:
        file_names.push_back(p)

    for s in f_sizes:
        file_sizes.push_back(s)

    c_fs = (<FileSystem>fs).unwrap()


def get_function_address(func_name):
    """
    Get the address of the function with the given name defined in
    _csv_json_reader.cpp and exported by the csv_json_reader module.
    """
    if func_name == "csv_file_chunk_reader":
        return <size_t>csv_file_chunk_reader
    elif func_name == "json_file_chunk_reader":
        return <size_t>json_file_chunk_reader
    elif func_name == "update_csv_reader":
        return <size_t>update_csv_reader
    elif func_name == "initialize_csv_reader":
        return <size_t>initialize_csv_reader

    raise ValueError("Unknown function name: " + func_name)


def get_pyarrow_fs_from_ptr(long fs_ptr_long):
    cdef CFileSystem *fs_ptr = <CFileSystem*>fs_ptr_long
    return FileSystem.wrap(shared_ptr[CFileSystem](fs_ptr))
