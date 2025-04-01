#include <Python.h>

#include <algorithm>
#include <cassert>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <fmt/format.h>

#include "../libs/_bodo_common.h"
#include "_bodo_file_reader.h"
#include "_s3_reader.h"

// helper macro for CHECK_ARROW to add a message regarding AWS creds or S3
// bucket region being an issue.
#define AWS_CREDS_OR_S3_REGION_WARNING(str_var, s3_fs_region)             \
    str_var =                                                             \
        "This might be due to an issue with your "                        \
        "AWS credentials, or with the region of the S3 bucket.\n";        \
    if (s3_fs_region.length() > 0) {                                      \
        str_var += "Region currently being used: " + s3_fs_region + "\n"; \
    }                                                                     \
    str_var +=                                                            \
        "Please verify your credentials and provide the correct "         \
        "region using the AWS_DEFAULT_REGION environment variable "       \
        "and try again.";

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it.
#define CHECK_ARROW(expr, msg, s3_fs_region)                                   \
    if (!(expr.ok())) {                                                        \
        std::string err_msg = std::string("Error in arrow s3: ") + msg + " " + \
                              expr.ToString() + ".\n";                         \
        std::string aws_warning = "";                                          \
        AWS_CREDS_OR_S3_REGION_WARNING(aws_warning, s3_fs_region);             \
        err_msg += aws_warning;                                                \
        throw std::runtime_error(err_msg);                                     \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it. If it is ok, get value using ValueOrDie
// and assign it to lhs using std::move
#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs, s3_fs_region) \
    CHECK_ARROW(res.status(), msg, s3_fs_region)            \
    lhs = std::move(res).ValueOrDie();

// Return a arrow::fs::S3ProxyOptions struct based
// on appropriate environment variables (if they are set), else
// return the default options (i.e. do not use a proxy)
arrow::fs::S3ProxyOptions get_s3_proxy_options_from_env_vars() {
    arrow::Result<arrow::fs::S3ProxyOptions> options_res;
    arrow::fs::S3ProxyOptions options;

    // There seems to be no industry-standard precedence for these
    // environment variables.
    // The precedence order below should be consistent with
    // get_proxy_uri_from_env_vars in fs_io.py
    // to avoid differences in compile-time and runtime
    // behavior.
    char *http_proxy = std::getenv("http_proxy");
    char *https_proxy = std::getenv("https_proxy");
    char *HTTP_PROXY = std::getenv("HTTP_PROXY");
    char *HTTPS_PROXY = std::getenv("HTTPS_PROXY");

    if (http_proxy) {
        options_res = arrow::fs::S3ProxyOptions::FromUri(http_proxy);
    } else if (https_proxy) {
        options_res = arrow::fs::S3ProxyOptions::FromUri(https_proxy);
    } else if (HTTP_PROXY) {
        options_res = arrow::fs::S3ProxyOptions::FromUri(HTTP_PROXY);
    } else if (HTTPS_PROXY) {
        options_res = arrow::fs::S3ProxyOptions::FromUri(HTTPS_PROXY);
    } else {
        // If no environment variable found, then return the default
        // S3ProxyOptions struct (the CHECK_ARROW_AND_ASSIGN macro
        // would fail on an unset options_res).
        return options;
    }

    // Safely set options, else throw an error.
    CHECK_ARROW_AND_ASSIGN(options_res, "S3ProxyOptions::FromUri", options,
                           std::string(""));
    return options;
}

void _sleep_exponential_backoff(unsigned int n, unsigned int max = 10) {
#ifdef _WIN32
    // convert to miliseconds before calling Sleep
    DWORD sleep_ms = (DWORD)((1 << n) * 1000);
    DWORD max_ms = (DWORD)(max * 1000);
    Sleep(std::min<DWORD>(sleep_ms, max_ms));
#else
    sleep(std::min<unsigned int>((1 << n), max));
#endif
}
// a global singleton instance of S3FileSystem that is
// initialized the first time it is needed and reused afterwards
static std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
bool is_fs_anonymous_mode = false;  // only valid when is_fs_initialized is true

/**
 * @brief Ensure that Arrow S3 APIs are initialized
 * If they are not initialized, initialize them
 */
void ensure_arrow_s3_initialized() {
    if (!arrow::fs::IsS3Initialized()) {
        arrow::fs::S3GlobalOptions g_options;
        g_options.log_level = arrow::fs::S3LogLevel::Off;
        arrow::Status status = arrow::fs::InitializeS3(g_options);
        CHECK_ARROW(status, "InitializeS3", std::string(""));
        if (!status.ok()) {
            throw std::runtime_error("Failed to initialize Arrow S3 APIs: " +
                                     status.ToString());
        }
    }
}

/**
 * @brief Initialize an S3FileSystem instance, some options are set through
 * environment variables
 * @param bucket_region The region of the S3 bucket, auto-detected if empty
 * @param anonymous Whether to use anonymous mode
 * @param aws_credentials_provider An optional pointer to an
 * AWSCredentialsProvider
 * @return An S3FileSystem instance
 */
arrow::Result<std::shared_ptr<arrow::fs::S3FileSystem>> init_s3fs_instance(
    std::string bucket_region, bool anonymous,
    std::unique_ptr<Aws::Auth::AWSCredentialsProvider>
        aws_credentials_provider) {
    // get S3FileSystem
    arrow::fs::S3Options options;
    if (!anonymous) {
        options = arrow::fs::S3Options::Defaults();
    } else {
        options = arrow::fs::S3Options::Anonymous();
    }
    char *default_region = std::getenv("AWS_DEFAULT_REGION");

    // Set region if one is provided, else a default region if one is
    // provided, else let Arrow use its default region (us-east-1)
    if (bucket_region != "") {
        options.region = std::string(bucket_region);
    } else if (default_region) {
        // Arrow actually seems to look for AWS_DEFAULT_REGION
        // variable and use other heuristics (based on AWS SDK version)
        // to determine region if one isn't provided, but doesn't
        // hurt to set it explicitly if env var is provided
        options.region = std::string(default_region);
    }
    char *custom_endpoint = std::getenv("AWS_S3_ENDPOINT");
    if (custom_endpoint) {
        options.endpoint_override = std::string(custom_endpoint);
    }
    if (aws_credentials_provider.get() != nullptr) {
        // Give ownership to S3FileSystem so the credentials_provider is
        // deallocated when the filesystem is destroyed
        options.credentials_provider = std::move(aws_credentials_provider);
    }

    // Set proxy options if the appropriate environment variables are set
    options.proxy_options = get_s3_proxy_options_from_env_vars();

    return arrow::fs::S3FileSystem::Make(options,
                                         bodo::default_buffer_io_context());
}

std::shared_ptr<arrow::fs::S3FileSystem> get_s3_fs(std::string bucket_region,
                                                   bool anonymous) {
    bool reinit_s3fs_instance;
    bool is_initialized = arrow::fs::IsS3Initialized();
    ensure_arrow_s3_initialized();

    if (is_initialized && s3_fs == nullptr) {
        // always init in this case
        reinit_s3fs_instance = true;

    } else if (!is_initialized) {
        // always init in this case
        reinit_s3fs_instance = true;

    } else {
        // If already initialized but region for this bucket is different
        // than the current region, then re-initialize with the right region.
        // Similarly if anonymous mode has changed, then re-initialize.
        reinit_s3fs_instance = ((bucket_region != "") &&
                                (s3_fs->options().region != bucket_region)) ||
                               (is_fs_anonymous_mode != anonymous);
    }

    if (reinit_s3fs_instance) {
        auto result = init_s3fs_instance(bucket_region, anonymous, nullptr);
        is_fs_anonymous_mode = anonymous;
        CHECK_ARROW_AND_ASSIGN(result, "S3FileSystem::Make", s3_fs,
                               std::string(""))
    }
    return s3_fs;
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
        fs = get_s3_fs("", false);
        // open file
        result = fs->OpenInputFile(std::string(_fname));
        CHECK_ARROW_AND_ASSIGN(result, "S3FileSystem::OpenInputFile", s3_file,
                               fs->region())
        this->assign_f_type(_fname);
    }
    bool seek(int64_t pos) override {
        status = s3_file->Seek(pos + this->csv_header_bytes);
        return status.ok();
    }
    bool ok() override { return status.ok(); }
    bool read_to_buff(char *s, int64_t size) override {
        if (size == 0) {  // hack for minio, read_csv size 0
            return true;
        }
        int64_t bytes_read = 0;
        arrow::Result<int64_t> res = s3_file->Read(size, s);
        CHECK_ARROW_AND_ASSIGN(res, "S3 file GetSize() ", bytes_read,
                               fs->region());
        return status.ok() && (bytes_read == size);
    }
    uint64_t getSize() override {
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

        fs = get_s3_fs("", false);

        arrow::Result<std::vector<arrow::fs::FileInfo>> result =
            fs->GetFileInfo(dir_selector);
        CHECK_ARROW_AND_ASSIGN(result, "fs->GetFileInfo", file_stats,
                               fs->region())

        // extract file names and file sizes from file_stats
        // then sort by file names
        // assuming the directory contains files only, i.e. no
        // subdirectory
        std::ranges::transform(this->file_stats,
                               std::back_inserter(this->file_names_sizes),
                               extract_file_name_size);
        std::ranges::sort(this->file_names_sizes, sort_by_name);

        this->findDirSizeFileSizesFileNames(_dirname, file_names_sizes);
    };

    void initFileReader(const char *fname) override {
        this->f_reader = new S3FileReader(fname, this->f_type_to_string(),
                                          this->csv_header, this->json_lines);
        this->f_reader->csv_header_bytes = this->csv_header_bytes;
    };
};

void s3_get_fs(std::shared_ptr<arrow::fs::S3FileSystem> *fs,
               std::string bucket_region, bool anon = false) {
    try {
        *fs = get_s3_fs(bucket_region, anon);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

FileReader *init_s3_reader(const char *fname, const char *suffix,
                           bool csv_header, bool json_lines) {
    arrow::fs::FileInfo file_stat;
    std::shared_ptr<arrow::fs::S3FileSystem> fs = get_s3_fs("", false);
    arrow::Result<arrow::fs::FileInfo> result =
        fs->GetFileInfo(std::string(fname));
    CHECK_ARROW_AND_ASSIGN(result, "fs->GetFileInfo", file_stat, fs->region())
    if (file_stat.IsDirectory()) {
        return new S3DirectoryFileReader(fname, suffix, csv_header, json_lines);
    } else if (file_stat.IsFile()) {
        return new S3FileReader(fname, suffix, csv_header, json_lines);
    } else {
        throw std::runtime_error(
            "_s3_reader.cpp::init_s3_reader: Error in arrow s3: invalid "
            "path");
    }
}

void s3_open_file(const char *fname,
                  std::shared_ptr<::arrow::io::RandomAccessFile> *file,
                  const char *bucket_region, bool anonymous) {
    std::shared_ptr<arrow::fs::S3FileSystem> fs =
        get_s3_fs(bucket_region, anonymous);
    arrow::Result<std::shared_ptr<::arrow::io::RandomAccessFile>> result;
    result = fs->OpenInputFile(std::string(fname));
    CHECK_ARROW_AND_ASSIGN(result, "fs->OpenInputFile", *file, fs->region())
}

PyMODINIT_FUNC PyInit_s3_reader(void) {
    PyObject *m;
    MOD_DEF(m, "s3_reader", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    // Both only ever called from C++
    SetAttrStringFromVoidPtr(m, init_s3_reader);
    SetAttrStringFromVoidPtr(m, s3_get_fs);

    return m;
}

#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
