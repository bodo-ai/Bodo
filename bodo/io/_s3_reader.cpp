// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <iostream>

#include "../libs/_bodo_common.h"
#include "_bodo_file_reader.h"
#include "arrow/filesystem/filesystem.h"
#include "arrow/filesystem/s3fs.h"
#include "arrow/io/interfaces.h"
#include "arrow/result.h"
#include "arrow/status.h"

#define CHECK(expr, msg)                                        \
    if (!(expr)) {                                              \
        std::cerr << "Error in arrow s3: " << msg << std::endl; \
    }

#define CHECK_ARROW(expr, msg)                                                 \
    if (!(expr.ok())) {                                                        \
        std::cerr << "Error in arrow s3: " << msg << " " << expr << std::endl; \
    }

#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();

// a global singleton instance of S3FileSystem that is
// initialized the first time it is needed and reused afterwards
std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
bool is_fs_initialized = false;

std::shared_ptr<arrow::fs::S3FileSystem> get_s3_fs() {
    if (!is_fs_initialized) {
        arrow::Status status;
        // initialize S3 APIs
        arrow::fs::S3GlobalOptions g_options;
        // g_options.log_level = arrow::fs::S3LogLevel::Trace;
        status = arrow::fs::InitializeS3(g_options);
        CHECK_ARROW(status, "InitializeS3");

        // get S3FileSystem
        arrow::fs::S3Options options = arrow::fs::S3Options::Defaults();
        char *default_region = std::getenv("AWS_DEFAULT_REGION");
        // TODO: handle regions properly
        if (default_region)
            options.region = std::string(default_region);
        else
            std::cerr << "Warning: AWS_DEFAULT_REGION environment variable not "
                         "found. Region defaults to 'us-east-1' currently."
                      << std::endl;
        char *custom_endpoint = std::getenv("AWS_S3_ENDPOINT");
        if (custom_endpoint)
            options.endpoint_override = std::string(custom_endpoint);

        arrow::Result<std::shared_ptr<arrow::fs::S3FileSystem>> result;
        result = arrow::fs::S3FileSystem::Make(options);
        CHECK_ARROW_AND_ASSIGN(result, "S3FileSystem::Make", s3_fs)
        is_fs_initialized = true;
    }
    return s3_fs;
}

static int finalize_s3() {
    if (is_fs_initialized) {
        CHECK_ARROW(arrow::fs::FinalizeS3(), "Finalize S3");
        is_fs_initialized = false;
    }
    return 0;
}

std::pair<std::string, int64_t> extract_file_name_size(
    const arrow::fs::FileInfo &file_stat) {
    return make_pair(file_stat.path(), file_stat.size());
}

bool sort_by_name(const std::pair<std::string, int64_t> &a,
                  const std::pair<std::string, int64_t> &b) {
    return (a.first < b.first);
}

// read S3 files using Arrow 0.15
class S3FileReader : public SingleFileReader {
   public:
    std::shared_ptr<arrow::io::RandomAccessFile> s3_file;
    std::shared_ptr<arrow::fs::S3FileSystem> fs;
    arrow::Status status;
    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> result;
    S3FileReader(const char *_fname, const char *f_type)
        : SingleFileReader(_fname, f_type) {
        fs = get_s3_fs();
        // open file
        result = fs->OpenInputFile(std::string(_fname));
        CHECK_ARROW_AND_ASSIGN(result, "S3FileSystem::OpenInputFile", s3_file)
        this->assign_f_type(_fname);
    }
    bool seek(int64_t pos) {
        status = s3_file->Seek(pos);
        return status.ok();
    }
    bool ok() { return status.ok(); }
    bool read_to_buff(char *s, int64_t size) {
        if (size == 0) {  // hack for minio, read_csv size 0
            return 1;
        }
        int64_t bytes_read = 0;
        arrow::Result<int64_t> res = s3_file->Read(size, s);
        CHECK_ARROW_AND_ASSIGN(res, "S3 file GetSize() ", bytes_read);
        return status.ok() && (bytes_read == size);
    }
    uint64_t getSize() {
        int64_t size = -1;
        arrow::Result<int64_t> res = s3_file->GetSize();
        CHECK_ARROW_AND_ASSIGN(res, "S3 file GetSize() ", size);
        return (uint64_t)size;
    }
};

class S3DirectoryFileReader : public DirectoryFileReader {
   public:
    std::shared_ptr<arrow::fs::S3FileSystem> fs;
    // sorted names of each csv file inside the directory
    name_size_vec file_names_sizes;
    // FileSelector used to get Directory information
    arrow::fs::FileSelector dir_selector;
    // FileInfo used to determine types, names, sizes
    std::vector<arrow::fs::FileInfo> file_stats;
    arrow::Status status;

    S3DirectoryFileReader(const char *_dirname, const char *f_type)
        : DirectoryFileReader(_dirname, f_type) {
        // dirname is in format of s3://host:port/path/dir]
        // initialize dir_selector
        dir_selector.base_dir = this->dirname;

        fs = get_s3_fs();

        arrow::Result<std::vector<arrow::fs::FileInfo>> result =
            fs->GetFileInfo(dir_selector);
        CHECK_ARROW_AND_ASSIGN(result, "fs->GetFileInfo", file_stats)

        // extract file names and file sizes from file_stats
        // then sort by file names
        // assuming the directory contains files only, i.e. no subdirectory
        std::transform(this->file_stats.begin(), this->file_stats.end(),
                       std::back_inserter(this->file_names_sizes),
                       extract_file_name_size);
        std::sort(this->file_names_sizes.begin(), this->file_names_sizes.end(),
                  sort_by_name);

        this->file_names =
            this->findDirSizeFileSizesFileNames(_dirname, file_names_sizes);
    };

    void initFileReader(const char *fname) {
        this->f_reader = new S3FileReader(fname, this->f_type_to_stirng());
        this->f_reader->json_lines = this->json_lines;
    };
};

extern "C" {

void s3_get_fs(std::shared_ptr<arrow::fs::S3FileSystem> *fs) {
    *fs = get_s3_fs();
}

FileReader *init_s3_reader(const char *fname, const char *suffix) {
    arrow::fs::FileInfo file_stat;
    std::shared_ptr<arrow::fs::S3FileSystem> fs = get_s3_fs();
    arrow::Result<arrow::fs::FileInfo> result =
        fs->GetFileInfo(std::string(fname));
    CHECK_ARROW_AND_ASSIGN(result, "fs->GetFileInfo", file_stat)
    if (file_stat.IsDirectory()) {
        return new S3DirectoryFileReader(fname, suffix);
    } else if (file_stat.IsFile()) {
        return new S3FileReader(fname, suffix);
    } else {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Error in arrow s3: invalid path");
        return NULL;
    }
}

void s3_open_file(const char *fname,
                  std::shared_ptr<::arrow::io::RandomAccessFile> *file) {
    std::shared_ptr<arrow::fs::S3FileSystem> fs = get_s3_fs();
    arrow::Result<std::shared_ptr<::arrow::io::RandomAccessFile>> result;
    result = fs->OpenInputFile(std::string(fname));
    CHECK_ARROW_AND_ASSIGN(result, "fs->OpenInputFile", *file)
}

PyMODINIT_FUNC PyInit_s3_reader(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "s3_reader", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    PyObject_SetAttrString(m, "init_s3_reader",
                           PyLong_FromVoidPtr((void *)(&init_s3_reader)));
    PyObject_SetAttrString(m, "s3_open_file",
                           PyLong_FromVoidPtr((void *)(&s3_open_file)));
    PyObject_SetAttrString(m, "s3_get_fs",
                           PyLong_FromVoidPtr((void *)(&s3_get_fs)));
    PyObject_SetAttrString(m, "finalize_s3",
                           PyLong_FromVoidPtr((void *)(&finalize_s3)));

    return m;
}

#undef CHECK
#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN

}  // extern "C"
