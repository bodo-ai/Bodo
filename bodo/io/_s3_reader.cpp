// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <iostream>

#include "_file_reader.h"
#include "arrow/io/interfaces.h"
#include "arrow/filesystem/s3fs.h"


#define CHECK_ARROW(expr, msg) if(!(expr.ok())){std::cerr << "Error in arrow s3 csv_read: " << msg << " " << expr <<std::endl;}


// a global singleton instance of S3FileSystem that is
// initialized the first time it is needed and reuse afterwards
std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
bool is_fs_initialized = false;


std::shared_ptr<arrow::fs::S3FileSystem> get_s3_fs()
{
    if (!is_fs_initialized) {
        arrow::Status status;
        // initialize S3 APIs
        arrow::fs::S3GlobalOptions g_options;
        // g_options.log_level = arrow::fs::S3LogLevel::Trace;
        status = arrow::fs::InitializeS3(g_options);
        CHECK_ARROW(status, "InitializeS3");

        // get S3FileSystem
        arrow::fs::S3Options options = arrow::fs::S3Options::Defaults();
        char* default_region = std::getenv("AWS_DEFAULT_REGION");
        // TODO: handle regions properly
        if (default_region)
            options.region = std::string(default_region);
        else
            std::cerr << "Warning: AWS_DEFAULT_REGION environment variable not found. Region defaults to 'us-east-1' currently." <<std::endl;
        status = arrow::fs::S3FileSystem::Make(options, &s3_fs);
        CHECK_ARROW(status, "S3FileSystem::Make");
        is_fs_initialized = true;
    }
    return s3_fs;
}


// read S3 files using Arrow 0.15
class S3FileReader : public FileReader
{
public:
    std::shared_ptr<arrow::io::RandomAccessFile> s3_file;
    std::shared_ptr<arrow::fs::S3FileSystem> fs;
    arrow::Status status;
    S3FileReader(const char *_fname) : FileReader(_fname) {
        fs = get_s3_fs();
        // open file
        status = fs->OpenInputFile(std::string(_fname), &s3_file);
        CHECK_ARROW(status, "S3FileSystem::OpenInputFile");

    }
    bool seek(int64_t pos) {
        status = s3_file->Seek(pos);
        return status.ok();
    }
    bool ok() {
        return status.ok();
    }
    bool read(char *s, int64_t size) {
        int64_t bytes_read;
        status = s3_file->Read(size, &bytes_read, s);
        return status.ok() && (bytes_read == size);
    }
    uint64_t getSize()
    {
        int64_t size = -1;
        status = s3_file->GetSize(&size);
        CHECK_ARROW(status, "S3 file GetSize()");
        return (uint64_t)size;
    }

    ~S3FileReader() {
        CHECK_ARROW(arrow::fs::FinalizeS3(), "Finalize34");
    }
};


extern "C" {

FileReader *init_s3_reader(const char *fname)
{
    return new S3FileReader(fname);
}


void s3_open_file(const char *fname, std::shared_ptr<::arrow::io::RandomAccessFile> *file)
{
    std::shared_ptr<arrow::fs::S3FileSystem> fs = get_s3_fs();
    arrow::Status status = fs->OpenInputFile(std::string(fname), file);
    CHECK_ARROW(status, "fs->OpenInputFile");
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

    return m;
}

}  // extern "C"
