// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <iostream>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/hdfs.h>
#include <arrow/io/hdfs.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include "../libs/_bodo_common.h"
#include "_bodo_file_reader.h"

#undef CHECK
#define CHECK(expr, msg)                                          \
    if (!(expr)) {                                                \
        std::cerr << "Error in arrow hdfs: " << msg << std::endl; \
    }

#define CHECK_ARROW(expr, msg)                                     \
    if (!(expr.ok())) {                                            \
        std::cerr << "Error in arrow hdfs: " << msg << " " << expr \
                  << std::endl;                                    \
    }

#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg);           \
    lhs = std::move(res).ValueOrDie();

// a global singleton instance of HadoopFileSystem that is
// initialized the first time it is needed and reused afterwards
// if the config does not change
static std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs;
// a global singleton instance of HdfsConnectionConfig
// used to check if a different HadoopFileSystem
// needs to be initialized
arrow::io::HdfsConnectionConfig hdfs_config;
bool is_hdfs_initialized = false;

std::shared_ptr<::arrow::fs::HadoopFileSystem> get_hdfs_fs(
    const std::string &uri_string) {
    arrow::fs::HdfsOptions options;
    arrow::Result<arrow::fs::HdfsOptions> result;
    arrow::Status status;

    // check if libhdfs exists
    if (!is_hdfs_initialized) {
        status = ::arrow::io::HaveLibHdfs();
        if (!status.ok()) {
            Bodo_PyErr_SetString(PyExc_RuntimeError, "libhdfs not found");
            return NULL;  // TODO: use Bodo_PyErr_SetString instead
        }
    }
    // parse hdfs options from file uri
    // options.connection_config includes:
    //   host, port, user, driver, kerb_ticket
    result = arrow::fs::HdfsOptions::FromUri(uri_string);
    CHECK_ARROW_AND_ASSIGN(result, "HdfsOptions::FromUri", options);
    // handle hdfs_config
    if (!is_hdfs_initialized) {
        hdfs_config = options.connection_config;
    } else {
        // check host, port, user to determine
        // if new connection needs to be made
        if ((hdfs_config.host != options.connection_config.host) ||
            (hdfs_config.port != options.connection_config.port) ||
            (hdfs_config.user != options.connection_config.user)) {
            hdfs_fs =
                nullptr;  // destructor calls Close, which calls Disconnect
            hdfs_config = options.connection_config;
        } else {
            return hdfs_fs;
        }
    }
    if (uri_string.starts_with("abfs://") || uri_string.starts_with("abfss://"))
        // need to pass whole URI as host to libhdfs
        options.ConfigureEndPoint(uri_string, 0);
    // connect to hdfs
    options.ConfigureReplication(0);
    options.ConfigureBufferSize(0);
    options.ConfigureBlockSize(0);
    hdfs_fs = arrow::fs::HadoopFileSystem::Make(
                  options, bodo::default_buffer_io_context())
                  .ValueOrDie();

    is_hdfs_initialized = true;
    return hdfs_fs;
}

static int disconnect_hdfs() {
    if (is_hdfs_initialized) {
        hdfs_fs = nullptr;  // destructor calls Close, which calls Disconnect
        is_hdfs_initialized = false;
    }
    return 0;
}

std::pair<std::string, int64_t> hdfs_extract_file_name_size(
    const arrow::fs::FileInfo &file_stat) {
    return make_pair(file_stat.path(), file_stat.size());
}

bool hdfs_sort_by_name(const std::pair<std::string, int64_t> &a,
                       const std::pair<std::string, int64_t> &b) {
    return (a.first < b.first);
}

// read hdfs files using Arrow 0.17
class HdfsFileReader : public SingleFileReader {
   public:
    std::shared_ptr<arrow::fs::HadoopFileSystem> fs;
    std::shared_ptr<arrow::io::RandomAccessFile> hdfs_file;
    arrow::Status status;
    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> result;

    HdfsFileReader(const char *_fname, const char *f_type, bool csv_header,
                   bool json_lines)
        : SingleFileReader(_fname, f_type, csv_header, json_lines) {
        // fname is in format of hdfs://host:port/dir/file]
        std::string fname(_fname);
        // path is in format of dir/file
        std::string path;

        fs = get_hdfs_fs(fname);
        arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempfs =
            ::arrow::fs::FileSystemFromUri(
                fname, bodo::default_buffer_io_context(), &path);
        // open file
        result = fs->OpenInputFile(path);
        CHECK_ARROW_AND_ASSIGN(result, "HdfsFileSystem::OpenInputFile",
                               hdfs_file)
    }
    bool seek(int64_t pos) {
        status = hdfs_file->Seek(pos + this->csv_header_bytes);
        return status.ok();
    }
    bool ok() { return status.ok(); }
    bool read_to_buff(char *s, int64_t size) {
        arrow::Result<int64_t> result = hdfs_file->Read(size, s);
        if (!(result.status().ok())) {
            return false;
        }
        int64_t bytes_read = std::move(result).ValueOrDie();
        return bytes_read == size;
    }
    uint64_t getSize() {
        int64_t size = -1;
        arrow::Result<int64_t> result = hdfs_file->GetSize();
        CHECK_ARROW_AND_ASSIGN(result, "HdfsReadableFile::GetSize", size);
        return (uint64_t)size;
    }
};

class HdfsDirectoryFileReader : public DirectoryFileReader {
   public:
    std::shared_ptr<arrow::fs::HadoopFileSystem> fs;
    // sorted names of each csv file inside the directory
    name_size_vec file_names_sizes;
    // FileSelector used to get Directory information
    arrow::fs::FileSelector dir_selector;
    // FileInfo used to determine types, names, sizes
    std::vector<arrow::fs::FileInfo> file_stats;
    arrow::Status status;

    HdfsDirectoryFileReader(const char *_dirname, const char *f_type,
                            bool csv_header, bool json_lines)
        : DirectoryFileReader(_dirname, f_type, csv_header, json_lines) {
        // path is in format of path/dirname
        std::string path;

        fs = get_hdfs_fs(this->dirname);

        // initialize dir_selector
        dir_selector.base_dir = this->dirname;

        arrow::Result<std::vector<arrow::fs::FileInfo>> result =
            fs->GetFileInfo(dir_selector);
        CHECK_ARROW_AND_ASSIGN(result, "fs->GetFileInfo", file_stats)

        // extract file names and file sizes from file_stats
        // then sort by file names
        // assuming the directory contains files only, i.e. no subdirectory
        std::transform(this->file_stats.begin(), this->file_stats.end(),
                       std::back_inserter(this->file_names_sizes),
                       hdfs_extract_file_name_size);
        std::sort(this->file_names_sizes.begin(), this->file_names_sizes.end(),
                  hdfs_sort_by_name);

        this->findDirSizeFileSizesFileNames(_dirname, file_names_sizes);
    };

    void initFileReader(const char *fname) {
        this->f_reader = new HdfsFileReader(fname, this->f_type_to_string(),
                                            this->csv_header, this->json_lines);
        this->f_reader->csv_header_bytes = this->csv_header_bytes;
    };
};

void hdfs_get_fs(const std::string &uri_string,
                 std::shared_ptr<::arrow::fs::HadoopFileSystem> *fs) {
    *fs = get_hdfs_fs(uri_string);
}

FileReader *init_hdfs_reader(const char *fname, const char *suffix,
                             bool csv_header, bool json_lines) {
    std::string path;
    arrow::io::HdfsPathInfo path_info;
    std::string f_name(fname);
    std::shared_ptr<::arrow::fs::HadoopFileSystem> fs = get_hdfs_fs(f_name);
    arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempfs =
        ::arrow::fs::FileSystemFromUri(fname, &path);
    arrow::fs::FileInfo file_stat;

    arrow::util::Uri uri;
    (void)uri.Parse(fname);
    arrow::Result<arrow::fs::FileInfo> result = fs->GetFileInfo(uri.path());
    CHECK_ARROW_AND_ASSIGN(result, "fs->GetFileInfo", file_stat)

    if (file_stat.IsDirectory()) {
        return new HdfsDirectoryFileReader(fname, suffix, csv_header,
                                           json_lines);
    } else if (file_stat.IsFile()) {
        return new HdfsFileReader(fname, suffix, csv_header, json_lines);
    } else {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Error in arrow hdfs: invalid path");
        return NULL;
    }
}

void hdfs_open_file(const char *fname,
                    std::shared_ptr<::arrow::io::RandomAccessFile> *file) {
    std::string path;
    std::string f_name(fname);
    std::shared_ptr<::arrow::fs::HadoopFileSystem> fs = get_hdfs_fs(f_name);
    arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempfs =
        ::arrow::fs::FileSystemFromUri(f_name, &path);
    arrow::Result<std::shared_ptr<::arrow::io::RandomAccessFile>> result;
    if (f_name.starts_with("abfs://") || f_name.starts_with("abfss://"))
        // need to pass whole URI to libhdfs
        result = fs->OpenInputFile(f_name);
    else
        result = fs->OpenInputFile(path);
    CHECK_ARROW_AND_ASSIGN(result, "fs->OpenInputFile", *file)
}

// #undef CHECK
// #undef CHECK_ARROW
// #undef CHECK_ARROW_AND_ASSIGN

/**
 * @brief Wrapper around disconnect_hdfs() to be called from Python (avoids
 Numba JIT overhead and makes compiler debugging easier by eliminating extra
 compilation)
 *
 */
static PyObject *disconnect_hdfs_py_wrapper(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError,
                        "disconnect_hdfs() does not take arguments");
        return NULL;
    }
    PyObject *ret_obj = PyLong_FromLong(disconnect_hdfs());
    return ret_obj;
}

static PyMethodDef ext_methods[] = {
#define declmethod(func) {#func, (PyCFunction)func, METH_VARARGS, NULL}
    declmethod(disconnect_hdfs_py_wrapper),
    {NULL},
#undef declmethod
};

PyMODINIT_FUNC PyInit_hdfs_reader(void) {
    PyObject *m;
    MOD_DEF(m, "hdfs_reader", "No docs", ext_methods);
    if (m == NULL)
        return NULL;

    SetAttrStringFromVoidPtr(m, init_hdfs_reader);
    SetAttrStringFromVoidPtr(m, hdfs_get_fs);
    SetAttrStringFromVoidPtr(m, disconnect_hdfs);

    return m;
}
