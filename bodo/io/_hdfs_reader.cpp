// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <iostream>

#include "../libs/_bodo_common.h"
#include "_bodo_csv_file_reader.h"
#include "arrow/filesystem/filesystem.h"
#include "arrow/filesystem/hdfs.h"
#include "arrow/io/interfaces.h"
#include "arrow/result.h"
#include "arrow/status.h"

#define CHECK_ARROW(expr, msg)                                     \
    if (!(expr.ok())) {                                            \
        std::cerr << "Error in arrow hdfs: " << msg << " " << expr \
                  << std::endl;                                    \
    }

#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg);           \
    lhs = std::move(res).ValueOrDie();

// a global singleton instance of HadoopFileSystem that is
// initialized the first time it is needed and reused afterwards
// if the config does not change
std::shared_ptr<::arrow::io::HadoopFileSystem> hdfs_fs;
// a global singleton instance of HdfsConnectionConfig
// used to check if a different HadoopFileSystem
// needs to be initialized
arrow::io::HdfsConnectionConfig hdfs_config;
bool is_fs_initialized = false;

std::shared_ptr<::arrow::io::HadoopFileSystem> get_hdfs_fs(
    const std::string &uri_string) {
    arrow::fs::HdfsOptions options;
    arrow::Result<arrow::fs::HdfsOptions> result;
    arrow::Status status;

    // check if libhdfs exists
    if (!is_fs_initialized) {
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
    if (!is_fs_initialized) {
        hdfs_config = options.connection_config;
    } else {
        // check host, port, user to determine
        // if new connection needs to be made
        if ((hdfs_config.host != options.connection_config.host) ||
            (hdfs_config.port != options.connection_config.port) ||
            (hdfs_config.user != options.connection_config.user)) {
            status = hdfs_fs->Disconnect();
            CHECK_ARROW(status, "Hdfs:Disconnect");
            hdfs_config = options.connection_config;
        } else {
            return hdfs_fs;
        }
    }
    // connect to hdfs
    status = ::arrow::io::HadoopFileSystem::Connect(&hdfs_config, &hdfs_fs);
    CHECK_ARROW(status, "hdfs::Connect");

    is_fs_initialized = true;
    return hdfs_fs;
}

static int disconnect_hdfs() {
    if (is_fs_initialized) {
        CHECK_ARROW(hdfs_fs->Disconnect(), "Disconnect hdfs");
        is_fs_initialized = false;
    }
    return 0;
}

// read hdfs files using Arrow 0.16
class HdfsFileReader : public FileReader {
   public:
    std::shared_ptr<arrow::io::HadoopFileSystem> fs;
    std::shared_ptr<::arrow::io::HdfsReadableFile> hdfs_file;
    arrow::Status status;
    HdfsFileReader(const char *_fname) : FileReader(_fname) {
        // fname is in format of hdfs://host:port/dir/file]
        std::string fname(_fname);
        // path is in format of dir/file
        std::string path;

        fs = get_hdfs_fs(fname);
        arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempfs =
            ::arrow::fs::FileSystemFromUri(fname, &path);
        // open file
        status = fs->OpenReadable(path, &hdfs_file);
        CHECK_ARROW(status, "fs->OpenInputFile");
    }
    bool seek(int64_t pos) {
        status = hdfs_file->Seek(pos);
        return status.ok();
    }
    bool ok() { return status.ok(); }
    bool read(char *s, int64_t size) {
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

extern "C" {

void hdfs_get_fs(const std::string &uri_string,
                 std::shared_ptr<::arrow::io::HadoopFileSystem> *fs) {
    *fs = get_hdfs_fs(uri_string);
}

FileReader *init_hdfs_reader(const char *fname) {
    return new HdfsFileReader(fname);
}

void hdfs_open_file(const char *fname,
                    std::shared_ptr<::arrow::io::HdfsReadableFile> *file) {
    std::string path;
    arrow::Status status;
    std::string f_name(fname);
    std::shared_ptr<::arrow::io::HadoopFileSystem> fs = get_hdfs_fs(f_name);
    arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempfs =
        ::arrow::fs::FileSystemFromUri(f_name, &path);
    status = fs->OpenReadable(path, file);
    CHECK_ARROW(status, "fs->OpenInputFile")
}

PyMODINIT_FUNC PyInit_hdfs_reader(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "hdfs_reader", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    PyObject_SetAttrString(m, "init_hdfs_reader",
                           PyLong_FromVoidPtr((void *)(&init_hdfs_reader)));
    PyObject_SetAttrString(m, "hdfs_get_fs",
                           PyLong_FromVoidPtr((void *)(&hdfs_get_fs)));
    PyObject_SetAttrString(m, "hdfs_open_file",
                           PyLong_FromVoidPtr((void *)(&hdfs_open_file)));
    PyObject_SetAttrString(m, "disconnect_hdfs",
                           PyLong_FromVoidPtr((void *)(&disconnect_hdfs)));
    return m;
}

}  // extern "C"
