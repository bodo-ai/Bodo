// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/utils/DateTime.h>
#include <fmt/format.h>
#include <boost/json/parser.hpp>
#include <boost/json/value_to.hpp>
#include <cassert>
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
    sleep(std::min<unsigned int>((1 << n), max));
}
// a global singleton instance of S3FileSystem that is
// initialized the first time it is needed and reused afterwards
static std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
bool is_fs_anonymous_mode = false;  // only valid when is_fs_initialized is true

std::pair<const std::string, const std::string>
IcebergRestAwsCredentialsProvider::get_warehouse_config() {
    const std::string uri = fmt::format("{}/v1/config?warehouse={}",
                                        this->catalog_uri, this->warehouse);
    curl_easy_setopt(hnd, CURLOPT_URL, uri.c_str());

    const std::string auth_header =
        fmt::format("Authorization: Bearer {}", this->bearer_token);

    struct curl_slist *slist1 = nullptr;
    slist1 = curl_slist_append(slist1, auth_header.c_str());
    curl_easy_setopt(hnd, CURLOPT_HTTPHEADER, slist1);

    for (unsigned int i = 0; i < this->n_retries; i++) {
        CURLcode res = curl_easy_perform(hnd);
        if (res != CURLE_OK) {
            if (i == this->n_retries - 1) {
                throw std::runtime_error(
                    fmt::format("Failed to get warehouse config: {}",
                                curl_easy_strerror(res)));
            }
            _sleep_exponential_backoff(i);
        } else {
            break;
        }
    }

    // Parse the response
    try {
        boost::json::parser parser;
        parser.write(curl_buffer);
        auto parsed_json = parser.release();
        auto config = parsed_json.as_object();
        auto overrides = config["overrides"].as_object();
        const std::string prefix =
            std::move(overrides["prefix"].as_string().c_str());
        const std::string warehouse_token =
            overrides.find("token") != overrides.end()
                ? std::move(overrides["token"].as_string().c_str())
                : this->bearer_token;
        curl_slist_free_all(slist1);
        curl_buffer.clear();
        return {prefix, warehouse_token};
    } catch (const std::exception &e) {
        throw std::runtime_error(
            fmt::format("Failed to parse warehouse config: {}", e.what()));
    }
}

std::tuple<const std::string, const std::string, const std::string,
           const std::string>
IcebergRestAwsCredentialsProvider::get_aws_credentials_from_rest_catalog(
    const std::string_view prefix, const std::string_view warehouse_token) {
    const std::string uri =
        fmt::format("{}/v1/{}/namespaces/{}/tables/{}", this->catalog_uri,
                    prefix, this->schema, this->table);
    curl_easy_setopt(hnd, CURLOPT_URL, uri.c_str());

    const std::string auth_header =
        fmt::format("Authorization: Bearer {}", warehouse_token);
    const std::string access_delegation_header =
        "X-Iceberg-Access-Delegation: vended_credentials";

    struct curl_slist *slist1 = nullptr;
    slist1 = curl_slist_append(slist1, auth_header.c_str());
    slist1 = curl_slist_append(slist1, access_delegation_header.c_str());
    curl_easy_setopt(hnd, CURLOPT_HTTPHEADER, slist1);
    for (unsigned int i = 0; i < this->n_retries; i++) {
        CURLcode res = curl_easy_perform(hnd);
        if (res != CURLE_OK) {
            if (i == this->n_retries - 1) {
                throw std::runtime_error(
                    fmt::format("Failed to get table credentials: {}",
                                curl_easy_strerror(res)));
            }
            _sleep_exponential_backoff(i);
        } else {
            break;
        }
    }

    try {
        boost::json::parser parser;
        parser.write(curl_buffer);
        auto parsed_json = parser.release();
        auto table_config = parsed_json.as_object()["config"].as_object();
        const std::string access_key =
            std::move(table_config["s3.access-key-id"].as_string().c_str());
        const std::string secret_key =
            std::move(table_config["s3.secret-access-key"].as_string().c_str());
        const std::string session_token =
            std::move(table_config["s3.session-token"].as_string().c_str());
        const std::string region =
            std::move(table_config["s3.region"].as_string().c_str());

        curl_slist_free_all(slist1);
        curl_buffer.clear();
        return {access_key, secret_key, session_token, region};
    } catch (const std::exception &e) {
        throw std::runtime_error(
            fmt::format("Failed to parse table credentials: {}", e.what()));
    }
}
Aws::Auth::AWSCredentials
IcebergRestAwsCredentialsProvider::GetAWSCredentials() {
    if (this->debug) {
        std::cerr << "[DEBUG] Getting AWS Credentials" << std::endl;
    }
    if (this->credentials.IsExpiredOrEmpty()) {
        this->Reload();
    }
    return this->credentials;
}
void IcebergRestAwsCredentialsProvider::Reload() {
    if (this->debug) {
        std::cerr << "[DEBUG] Reloading AWS Credentials" << std::endl;
    }
    auto [prefix, warehouse_token] = this->get_warehouse_config();
    auto [access_key, secret_key, session_token, region] =
        this->get_aws_credentials_from_rest_catalog(prefix, warehouse_token);
    this->region = region;
    // We don't know the expiration time of the credentials, so we set it to
    // 15 minutes by default which is the minimum from AWS
    this->credentials = Aws::Auth::AWSCredentials(
        access_key, secret_key, session_token,
        Aws::Utils::DateTime(std::chrono::system_clock::now() +
                             std::chrono::minutes(this->credential_timeout)));
    if (this->debug) {
        std::cerr << "[DEBUG] New AWS Credentials expire at"
                  << this->credentials.GetExpiration().ToLocalTimeString(
                         Aws::Utils::DateFormat::ISO_8601)
                  << std::endl;
    }
}

std::string IcebergRestAwsCredentialsProvider::getToken(
    const std::string_view base_url, const std::string_view credential) {
    // Generated using --libcurl
    CURL *hnd;
    struct curl_slist *slist1;
    slist1 = NULL;
    slist1 = curl_slist_append(
        slist1, "content-type: application/x-www-form-urlencoded");
    hnd = curl_easy_init();
    curl_easy_setopt(hnd, CURLOPT_BUFFERSIZE, 102400L);
    curl_easy_setopt(hnd, CURLOPT_URL,
                     fmt::format("{}/v1/oauth/tokens", base_url).c_str());
    curl_easy_setopt(hnd, CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(hnd, CURLOPT_HTTPHEADER, slist1);
    curl_easy_setopt(hnd, CURLOPT_USERAGENT, "curl/7.88.1");
    curl_easy_setopt(hnd, CURLOPT_MAXREDIRS, 50L);
    curl_easy_setopt(hnd, CURLOPT_HTTP_VERSION, (long)CURL_HTTP_VERSION_2TLS);
    curl_easy_setopt(hnd, CURLOPT_CUSTOMREQUEST, "POST");
    curl_easy_setopt(hnd, CURLOPT_FTP_SKIP_PASV_IP, 1L);
    curl_easy_setopt(hnd, CURLOPT_TCP_KEEPALIVE, 1L);

    // Set the POST data
    size_t split_idx = credential.find(':');
    std::string_view client_id = credential.substr(0, split_idx);
    std::string_view client_secret =
        credential.substr(split_idx + 1, credential.size() - (split_idx + 1));
    std::string post_data = fmt::format(
        "grant_type=client_credentials&client_id={}&client_secret={}",
        client_id, client_secret);
    curl_easy_setopt(hnd, CURLOPT_POSTFIELDS, post_data.data());
    curl_easy_setopt(hnd, CURLOPT_POSTFIELDSIZE_LARGE,
                     (curl_off_t)post_data.size());

    // Set a callback function to store the response from the Iceberg REST
    // API
    std::string curl_buffer;
    curl_easy_setopt(hnd, CURLOPT_WRITEFUNCTION,
                     IcebergRestAwsCredentialsProvider::CurlWriteCallback);
    curl_easy_setopt(hnd, CURLOPT_WRITEDATA, &curl_buffer);

    for (unsigned int i = 0; i < IcebergRestAwsCredentialsProvider::n_retries;
         i++) {
        CURLcode res = curl_easy_perform(hnd);
        if (res != CURLE_OK) {
            if (i == IcebergRestAwsCredentialsProvider::n_retries - 1) {
                throw std::runtime_error(fmt::format(
                    "Failed to get OAuth2 token: {}", curl_easy_strerror(res)));
            }
            _sleep_exponential_backoff(i);
        } else {
            break;
        }
    }

    curl_easy_cleanup(hnd);
    hnd = NULL;
    curl_slist_free_all(slist1);
    slist1 = NULL;

    try {
        boost::json::parser parser;
        parser.write(curl_buffer);
        curl_buffer.clear();
        auto parsed_json = parser.release();
        const std::string access_token = std::move(
            parsed_json.as_object()["access_token"].as_string().c_str());
        return access_token;
    } catch (const std::exception &e) {
        throw std::runtime_error(
            fmt::format("Failed to parse OAuth2 Token: {}", e.what()));
    }
}

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
    if (custom_endpoint)
        options.endpoint_override = std::string(custom_endpoint);
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

/**
 * @brief Create an S3FileSystem instance and return a pointer to it to python
 * @param bucket_region The region of the S3 bucket
 * @param anonymous Whether to use anonymous mode
 * @param aws_credentials_provider_opt Optional pointer to an
 * AWSCredentialsProvider, takes ownership
 */
void *create_s3_fs_instance_py_entry(
    const char *bucket_region, bool anonymous,
    numba_optional<Aws::Auth::AWSCredentialsProvider>
        aws_credentials_provider_opt) {
    try {
        ensure_arrow_s3_initialized();
        const Aws::SDKOptions options;
        Aws::InitAPI(options);
        std::unique_ptr<Aws::Auth::AWSCredentialsProvider>
            aws_credentials_provider;
        if (aws_credentials_provider_opt.has_value) {
            aws_credentials_provider =
                std::unique_ptr<Aws::Auth::AWSCredentialsProvider>(
                    static_cast<Aws::Auth::AWSCredentialsProvider *>(
                        aws_credentials_provider_opt.value));
        } else {
            aws_credentials_provider = std::make_unique<
                Aws::Auth::DefaultAWSCredentialsProviderChain>();
        }

        auto result = init_s3fs_instance(bucket_region, anonymous,
                                         std::move(aws_credentials_provider));
        std::shared_ptr<arrow::fs::S3FileSystem> s3_fs_instance;
        CHECK_ARROW_AND_ASSIGN(result, "S3FileSystem::Make", s3_fs_instance,
                               std::string(""));
        Aws::ShutdownAPI(options);
        return new arrow::fs::S3FileSystem(*s3_fs_instance.get());

    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
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
        CHECK_ARROW_AND_ASSIGN(res, "S3 file GetSize() ", bytes_read,
                               fs->region());
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

        fs = get_s3_fs("", false);

        arrow::Result<std::vector<arrow::fs::FileInfo>> result =
            fs->GetFileInfo(dir_selector);
        CHECK_ARROW_AND_ASSIGN(result, "fs->GetFileInfo", file_stats,
                               fs->region())

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

/**
 * @brief Create an AWSCredentialsProvider object
 * if any of the input arguments is empty or null, return a
 * default AWSCredentialsProvider object
 * otherwise, return an IcebergRestAwsCredentialsProvider object
 * @param catalog_uri URI of the Iceberg catalog
 * @param bearer_token Bearer token to authenticate with the Iceberg catalog
 * @param warehouse Warehouse name
 * @param schema Schema name
 * @param table Table name
 * @return an AWSCredentialsProvider
 */
void *create_iceberg_aws_credentials_provider_py_entry(const char *catalog_uri,
                                                       const char *bearer_token,
                                                       const char *warehouse,
                                                       const char *schema,
                                                       const char *table) {
    const Aws::SDKOptions options;
    try {
        [[maybe_unused]] bool all_args_not_null =
            catalog_uri && bearer_token && warehouse && schema && table;
        [[maybe_unused]] bool all_args_not_empty =
            catalog_uri[0] && bearer_token[0] && warehouse[0] && schema[0] &&
            table[0];
        assert(all_args_not_null && all_args_not_empty);

        // Init the AWS SDK with default options
        Aws::InitAPI(options);
        Aws::Auth::AWSCredentialsProvider *provider;
        {
            provider = new IcebergRestAwsCredentialsProvider(
                catalog_uri, bearer_token, warehouse, schema, table);
        }
        Aws::ShutdownAPI(options);
        return provider;
    } catch (const std::exception &e) {
        Aws::ShutdownAPI(options);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Get the region from an IcebergRestAwsCredentialsProvider
 * @returns std::string* containing the region
 */
std::string *get_region_from_creds_provider_py_entry(void *_provider) {
    try {
        auto provider =
            static_cast<IcebergRestAwsCredentialsProvider *>(_provider);
        auto region = provider->GetRegion();
        return new std::string(region);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

void destroy_iceberg_aws_credentials_provider_py_entry(void *provider) {
    try {
        if (provider != nullptr) {
            delete static_cast<Aws::Auth::AWSCredentialsProvider *>(provider);
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

unsigned int get_default_credential_timeout() {
    const char *default_credential_timeout_env_var =
        std::getenv("DEFAULT_ICEBERG_REST_AWS_CREDENTIALS_PROVIDER_TIMEOUT");
    if (default_credential_timeout_env_var != nullptr) {
        return std::stoi(default_credential_timeout_env_var);
    }
    return 15;
}

bool get_debug_credentials_provider() {
    const char *debug_credential_provider_env_var =
        std::getenv("DEBUG_ICEBERG_REST_AWS_CREDENTIALS_PROVIDER");
    if (debug_credential_provider_env_var != nullptr) {
        return std::strcmp(debug_credential_provider_env_var, "1") == 0;
    }
    return false;
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
    MOD_DEF(m, "s3_reader", "No docs", NULL);
    if (m == NULL)
        return NULL;

    // Both only ever called from C++
    SetAttrStringFromVoidPtr(m, init_s3_reader);
    SetAttrStringFromVoidPtr(m, s3_get_fs);

    SetAttrStringFromVoidPtr(m,
                             create_iceberg_aws_credentials_provider_py_entry);
    SetAttrStringFromVoidPtr(m,
                             destroy_iceberg_aws_credentials_provider_py_entry);
    SetAttrStringFromVoidPtr(m, create_s3_fs_instance_py_entry);
    SetAttrStringFromVoidPtr(m, get_region_from_creds_provider_py_entry);

    return m;
}

#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
