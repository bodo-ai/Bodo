// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <iostream>
#include <mpi.h>

#include "../libs/_bodo_common.h"
#include "../libs/_distributed.h"
#include "_bodo_file_reader.h"
#include "arrow/filesystem/filesystem.h"
#include "arrow/filesystem/s3fs.h"
#include "arrow/io/interfaces.h"
#include "arrow/result.h"
#include "arrow/status.h"


// helper macro for CHECK_ARROW to add a message regarding AWS creds or S3 
// bucket region being an issue.
#define AWS_CREDS_OR_S3_REGION_WARNING(str_var, s3_fs_region)                \
    str_var = "This might be due to an issue with your "                     \
        "AWS credentials, or with the region of the S3 bucket.\n";           \
    if (s3_fs_region.length() > 0) {                                         \
        str_var += "Region currently being used: " + s3_fs_region + "\n";    \
    }                                                                        \
    str_var += "Please verify your credentials and provide the correct "     \
        "region using the AWS_DEFAULT_REGION environment variable "          \
        "and try again.";

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it.
#define CHECK_ARROW(expr, msg, s3_fs_region)                              \
    if (!(expr.ok())) {                                                   \
        std::string err_msg =                                             \
            std::string("Error in arrow s3: ") + msg                      \
            + " " + expr.ToString() + ".\n";                              \
        std::string aws_warning = "";                                     \
        AWS_CREDS_OR_S3_REGION_WARNING(aws_warning, s3_fs_region);        \
        err_msg += aws_warning;                                           \
        throw std::runtime_error(err_msg);                                \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it. If it is ok, get value using ValueOrDie
// and assign it to lhs using std::move
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs, s3_fs_region) \
    CHECK_ARROW(res.status(), msg, s3_fs_region)            \
    lhs = std::move(res).ValueOrDie();

/* Helper to broadcast std::string variable from rank 0 to all others.
   Get the size of the string and broadcast it to the rest of the ranks,
   resize the strings on all ranks to this size, and then broadcast the
   string characters themselves
*/
static void mpi_bcast_std_string(std::string &str) {
    int str_size = str.size();
    MPI_Bcast(&str_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    str.resize(str_size);
    MPI_Bcast(const_cast<char *>(str.data()), str_size, 
              MPI_CHAR, 0, MPI_COMM_WORLD);
}

// a global singleton instance of S3FileSystem that is
// initialized the first time it is needed and reused afterwards
std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
bool is_fs_initialized = false;

std::shared_ptr<arrow::fs::S3FileSystem> get_s3_fs(std::string bucket_region) {
    
    // if already initialized but region for this bucket is different
    // than the current region, then re-initialize with the right region
    if (is_fs_initialized && bucket_region != "") {
        arrow::fs::S3Options options = s3_fs->options();
        if (bucket_region != options.region) {
            options.region = bucket_region;
            arrow::Result<std::shared_ptr<arrow::fs::S3FileSystem>> result;
            result = arrow::fs::S3FileSystem::Make(options);
            CHECK_ARROW_AND_ASSIGN(result, "S3FileSystem::Make", s3_fs, std::string(""))
        }
    } else if (!is_fs_initialized) {
        arrow::Status status;
        // initialize S3 APIs
        arrow::fs::S3GlobalOptions g_options;
        g_options.log_level = arrow::fs::S3LogLevel::Off;
        status = arrow::fs::InitializeS3(g_options);
        CHECK_ARROW(status, "InitializeS3", std::string(""));

        // get S3FileSystem
        arrow::fs::S3Options options = arrow::fs::S3Options::Defaults();
        char *default_region = std::getenv("AWS_DEFAULT_REGION");
        
        // Set region if one is provided, else a default region if one is provided, 
        // else let Arrow use its default region (us-east-1)
        if (bucket_region != "") {
            options.region = std::string(bucket_region);
        } else if (default_region) {
            // Arrow actually seems to look for AWS_DEFAULT_REGION
            // variable and use other heuristics (based on version) 
            // to determine region if one isn't provided, but doesn't
            // hurt to set it explicitly if env var is provided
            options.region = std::string(default_region);
        }
        char *custom_endpoint = std::getenv("AWS_S3_ENDPOINT");
        if (custom_endpoint)
            options.endpoint_override = std::string(custom_endpoint);

        arrow::Result<std::shared_ptr<arrow::fs::S3FileSystem>> result;
        result = arrow::fs::S3FileSystem::Make(options);
        CHECK_ARROW_AND_ASSIGN(result, "S3FileSystem::Make", s3_fs, std::string(""))
        is_fs_initialized = true;
    }
    return s3_fs;
}

/*
Get region of bucket corresponding to the filepath.
Input is expected to be of the form 
s3://<BUCKET_NAME>/<(optional)PATH_TO_FILE>
Returns an empty string when region cannot be determined.
This can happen when a custom endpoint is being used or
due to permissions issues.
*/
std::string get_region_from_s3_path(const char* filepath) {
    std::string region = "";
    try {
        // Create URI object from the filepath
        arrow::internal::Uri uri;
        (void)uri.Parse(filepath);
        // Load S3Options from URI
        arrow::fs::S3Options options;
        arrow::Result<arrow::fs::S3Options> result;
        result = arrow::fs::S3Options::FromUri(uri);
        CHECK_ARROW_AND_ASSIGN(result, "S3Options::FromUri", options, std::string(""));
        // Get region from the S3Options object
        region = options.region;
    } catch (const std::exception &e) {
        // In case of an error, catch and ignore it.
        // Errors can occur when using a custom endpoint for instance.
        // They can potentially occur when AWS credentials don't have
        // the right permissions as well, in which case we'd want to
        // use the AWS_DEFAULT_REGION if one is provided
    }
    return region;
    
}

static int finalize_s3() {
    try {
    if (is_fs_initialized) {
        CHECK_ARROW(arrow::fs::FinalizeS3(), "Finalize S3", std::string(""));
        is_fs_initialized = false;
    }
    return 0;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
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
    S3FileReader(const char *_fname, const char *f_type, bool csv_header,
                 bool json_lines)
        : SingleFileReader(_fname, f_type, csv_header, json_lines) {
        fs = get_s3_fs("");
        // open file
        result = fs->OpenInputFile(std::string(_fname));
        CHECK_ARROW_AND_ASSIGN(result, "S3FileSystem::OpenInputFile", s3_file, fs->region())
        this->assign_f_type(_fname);
    }
    bool seek(int64_t pos) {
        status = s3_file->Seek(pos + this->csv_header_bytes);
        return status.ok();
    }
    bool ok() { return status.ok(); }
    bool read_to_buff(char *s, int64_t size) {
        if (size == 0) {  // hack for minio, read_csv size 0
            return 1;
        }
        int64_t bytes_read = 0;
        arrow::Result<int64_t> res = s3_file->Read(size, s);
        CHECK_ARROW_AND_ASSIGN(res, "S3 file GetSize() ", bytes_read, fs->region());
        return status.ok() && (bytes_read == size);
    }
    uint64_t getSize() {
        int64_t size = -1;
        arrow::Result<int64_t> res = s3_file->GetSize();
        CHECK_ARROW_AND_ASSIGN(res, "S3 file GetSize() ", size, fs->region());
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

    S3DirectoryFileReader(const char *_dirname, const char *f_type,
                          bool csv_header, bool json_lines)
        : DirectoryFileReader(_dirname, f_type, csv_header, json_lines) {
        // dirname is in format of s3://host:port/path/dir]
        // initialize dir_selector
        dir_selector.base_dir = this->dirname;

        fs = get_s3_fs("");

        arrow::Result<std::vector<arrow::fs::FileInfo>> result =
            fs->GetFileInfo(dir_selector);
        CHECK_ARROW_AND_ASSIGN(result, "fs->GetFileInfo", file_stats, fs->region())

        // extract file names and file sizes from file_stats
        // then sort by file names
        // assuming the directory contains files only, i.e. no
        // subdirectory
        std::transform(this->file_stats.begin(), this->file_stats.end(),
                       std::back_inserter(this->file_names_sizes),
                       extract_file_name_size);
        std::sort(this->file_names_sizes.begin(), this->file_names_sizes.end(),
                  sort_by_name);

        this->findDirSizeFileSizesFileNames(_dirname, file_names_sizes);
    };

    void initFileReader(const char *fname) {
        this->f_reader = new S3FileReader(fname, this->f_type_to_string(),
                                          this->csv_header, this->json_lines);
        this->f_reader->csv_header_bytes = this->csv_header_bytes;
    };
};

extern "C" {

void s3_get_fs(std::shared_ptr<arrow::fs::S3FileSystem> *fs,
               std::string bucket_region) {
    try {
    *fs = get_s3_fs(bucket_region);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

FileReader *init_s3_reader(const char *fname, const char *suffix,
                           bool csv_header, bool json_lines) {
    try {
    arrow::fs::FileInfo file_stat;
    std::shared_ptr<arrow::fs::S3FileSystem> fs = get_s3_fs("");
    arrow::Result<arrow::fs::FileInfo> result =
        fs->GetFileInfo(std::string(fname));
    CHECK_ARROW_AND_ASSIGN(result, "fs->GetFileInfo", file_stat, fs->region())
    if (file_stat.IsDirectory()) {
        return new S3DirectoryFileReader(fname, suffix, csv_header, json_lines);
    } else if (file_stat.IsFile()) {
        return new S3FileReader(fname, suffix, csv_header, json_lines);
    } else {
        throw std::runtime_error(
            "_s3_reader.cpp::init_s3_reader: Error in arrow s3: invalid path");
    }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    } 
}

void s3_open_file(const char *fname,
                  std::shared_ptr<::arrow::io::RandomAccessFile> *file,
                  const char *bucket_region) {
    std::shared_ptr<arrow::fs::S3FileSystem> fs = get_s3_fs(bucket_region);
    arrow::Result<std::shared_ptr<::arrow::io::RandomAccessFile>> result;
    result = fs->OpenInputFile(std::string(fname));
    CHECK_ARROW_AND_ASSIGN(result, "fs->OpenInputFile", *file, fs->region())
}

char* get_region_from_s3_path_py_entrypt(const char *filepath, int64_t *len) {
    std::string region_str = "";
    // Get the bucket region (on one rank)
    if (dist_get_rank() == 0)
        region_str = get_region_from_s3_path(filepath);
    // Broadcast to all other ranks
    mpi_bcast_std_string(region_str);
    // Get length of string and convert to char*
    *len = region_str.length() + 1;
    char *region = new char[*len];
    strcpy(region, region_str.c_str());
    return region;
}

void del_region_str(char *region_str) { delete region_str; }

PyMODINIT_FUNC PyInit_s3_reader(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "s3_reader", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    // Only ever called from C++
    PyObject_SetAttrString(m, "init_s3_reader",
                           PyLong_FromVoidPtr((void *)(&init_s3_reader)));
    // Only ever called from C++
    PyObject_SetAttrString(m, "s3_open_file",
                           PyLong_FromVoidPtr((void *)(&s3_open_file)));
    // Only ever called from C++
    PyObject_SetAttrString(m, "s3_get_fs",
                           PyLong_FromVoidPtr((void *)(&s3_get_fs)));
    PyObject_SetAttrString(m, "finalize_s3",
                           PyLong_FromVoidPtr((void *)(&finalize_s3)));
    PyObject_SetAttrString(m, "get_region_from_s3_path",
                           PyLong_FromVoidPtr((void *)(&get_region_from_s3_path_py_entrypt)));
    PyObject_SetAttrString(m, "del_region_str",
                           PyLong_FromVoidPtr((void *)(&del_region_str)));

    return m;
}

#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN

}  // extern "C"
