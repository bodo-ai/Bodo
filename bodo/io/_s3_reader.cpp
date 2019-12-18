// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <iostream>

#include "_file_reader.h"
#include "arrow/io/interfaces.h"
#include "arrow/filesystem/s3fs.h"


#define CHECK_ARROW(expr, msg) if(!(expr.ok())){std::cerr << "Error in arrow s3 csv_read: " << msg << " " << expr <<std::endl;}

// read S3 files using Arrow 0.15
class S3FileReader : public FileReader
{
public:
    std::shared_ptr<arrow::io::RandomAccessFile> s3_file;
    std::shared_ptr<arrow::fs::S3FileSystem> fs;
    arrow::Status status;
    S3FileReader(const char *_fname) : FileReader(_fname) {
        // initialize S3 APIs
        arrow::fs::S3GlobalOptions g_options;
        // g_options.log_level = arrow::fs::S3LogLevel::Trace;
        status = arrow::fs::InitializeS3(g_options);
        CHECK_ARROW(status, "InitializeS3");

        // get S3FileSystem
        arrow::fs::S3Options options = arrow::fs::S3Options::Defaults();
        char* default_region = std::getenv("AWS_DEFAULT_REGION");
        // TODO: issue warning when region is not specified?
        if (default_region)
            options.region = std::string(default_region);
        status = arrow::fs::S3FileSystem::Make(options, &fs);
        CHECK_ARROW(status, "S3FileSystem::Make");

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


PyMODINIT_FUNC PyInit_s3_reader(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "s3_reader", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    PyObject_SetAttrString(m, "init_s3_reader",
                           PyLong_FromVoidPtr((void *)(&init_s3_reader)));

    return m;
}

}  // extern "C"
