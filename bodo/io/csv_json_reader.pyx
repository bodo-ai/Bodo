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

import bodo
import bodo.io.fs_io
from bodo.io.fs_io import parse_fpath, getfs, get_all_csv_json_data_files, get_compression_from_file_name


cdef public void get_read_path_info(
        const char* file_path,
        c_string compression_pyarg,
        c_bool is_anon,
        # Outputs (workaround since Cython doesn't support C++ tuples)
        c_bool& is_remote_fs,
        c_string& compression,
        vector[c_string]& file_names,
        vector[int64_t]& file_sizes,
        shared_ptr[CFileSystem] c_fs
    ):


    path, parsed_url, protocol = parse_fpath(file_path)

    is_remote_fs = (protocol != "")

    storage_options = {}
    if is_anon:
        storage_options["anon"] = True

    fs = getfs(path, protocol, storage_options=storage_options)

    err_msg = "Invalid data path path: " + path
    all_csv_files = get_all_csv_json_data_files(fs, path, protocol, parsed_url, err_msg)

    c = <object>compression_pyarg
    if c == "infer":
        compression = get_compression_from_file_name(all_csv_files[0])
    else:
        compression = compression_pyarg

    for p in all_csv_files:
        file_names.push_back(p)
    
    for p in all_csv_files:
        file_sizes.push_back(fs.get_file_info(p).size)

    c_fs = (<FileSystem>fs).unwrap()
