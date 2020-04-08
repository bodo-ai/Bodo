// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <iostream>

#include "../libs/_bodo_common.h"
#include "_bodo_file_reader.h"
#include "arrow/filesystem/filesystem.h"
#include "arrow/filesystem/hdfs.h"
#include "arrow/io/hdfs.h"
#include "arrow/io/interfaces.h"
#include "arrow/result.h"
#include "arrow/status.h"

#define CHECK(expr, msg)                                          \
    if (!(expr)) {                                                \
        std::cerr << "Error in arrow hdfs: " << msg << std::endl; \
    }

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
            CHECK_ARROW(status, "Hdfs:Disconnect ");
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

std::pair<std::string, int64_t> extract_file_name_size(
    const arrow::io::HdfsPathInfo &path_info) {
    return make_pair(path_info.name, path_info.size);
}

bool sort_by_name(const std::pair<std::string, int64_t> &a,
                  const std::pair<std::string, int64_t> &b) {
    return (a.first < b.first);
}

// read hdfs files using Arrow 0.16
class HdfsFileReader : public SingleFileReader {
   public:
    std::shared_ptr<arrow::io::HadoopFileSystem> fs;
    std::shared_ptr<::arrow::io::HdfsReadableFile> hdfs_file;
    arrow::Status status;
    HdfsFileReader(const char *_fname) : SingleFileReader(_fname) {
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

class HdfsDirectoryFileReader : public DirectoryFileReader {
   public:
    std::shared_ptr<arrow::io::HadoopFileSystem> fs;
    // sorted names of each csv file inside the directory
    name_size_vec file_names_sizes;
    // PathInfo used to determine types, names, sizes
    std::vector<arrow::io::HdfsPathInfo> path_infos;
    arrow::Status status;

    HdfsDirectoryFileReader(const char *_dirname)
        : DirectoryFileReader(_dirname) {
        // path is in format of path/dirname
        std::string path;

        fs = get_hdfs_fs(this->dirname);

        arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempfs =
            ::arrow::fs::FileSystemFromUri(this->dirname, &path);
        status = fs->ListDirectory(path, &this->path_infos);
        CHECK_ARROW(status, "fs->ListDirectory");

        // extract file names and file sizes from path_infos
        // then sort by file names
        // assuming the directory contains files only, i.e. no subdirectory
        std::transform(this->path_infos.begin(), this->path_infos.end(),
                       std::back_inserter(this->file_names_sizes),
                       extract_file_name_size);
        std::sort(this->file_names_sizes.begin(), this->file_names_sizes.end(),
                  sort_by_name);

        this->file_names =
            this->findDirSizeFileSizesFileNames(file_names_sizes);
    };

    void initFileReader(const char *fname) {
        this->f_reader = new HdfsFileReader(fname);
    };
};

extern "C" {

void hdfs_get_fs(const std::string &uri_string,
                 std::shared_ptr<::arrow::io::HadoopFileSystem> *fs) {
    *fs = get_hdfs_fs(uri_string);
}

FileReader *init_hdfs_reader(const char *fname) {
    std::string path;
    arrow::io::HdfsPathInfo path_info;
    std::string f_name(fname);
    std::shared_ptr<::arrow::io::HadoopFileSystem> fs = get_hdfs_fs(f_name);
    arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempfs =
        ::arrow::fs::FileSystemFromUri(fname, &path);
    arrow::Status status = fs->GetPathInfo(path, &path_info);
    CHECK_ARROW(status, "fs->GetPathInfo");
    if (path_info.kind == arrow::io::ObjectType::DIRECTORY) {
        return new HdfsDirectoryFileReader(fname);
    }

    return new HdfsFileReader(fname);
}

void hdfs_open_file(const char *fname,
                    std::shared_ptr<::arrow::io::HdfsReadableFile> *file) {
    std::string path;
    std::string f_name(fname);
    std::shared_ptr<::arrow::io::HadoopFileSystem> fs = get_hdfs_fs(f_name);
    arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempfs =
        ::arrow::fs::FileSystemFromUri(f_name, &path);
    arrow::Status status = fs->OpenReadable(path, file);
    CHECK_ARROW(status, "fs->OpenInputFile")
}

// #undef CHECK
// #undef CHECK_ARROW
// #undef CHECK_ARROW_AND_ASSIGN

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
