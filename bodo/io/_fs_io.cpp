#include <Python.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

#include <arrow/filesystem/azurefs.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/gcsfs.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/util/uri.h>
#include <mpi.h>

#include "../libs/_distributed.h"
#include "_fs_io.h"
#include "arrow_compat.h"

// Helper to ensure that the pyarrow wrappers have been imported.
// We use a static variable to make sure we only do the import once.
static bool imported_pyarrow_wrappers = false;
static void ensure_pa_wrappers_imported() {
#define CHECK(expr, msg)                                        \
    if (expr) {                                                 \
        throw std::runtime_error(std::string("fs_io: ") + msg); \
    }
    if (imported_pyarrow_wrappers) {
        return;
    }
    CHECK(arrow::py::import_pyarrow_wrappers(),
          "importing pyarrow_wrappers failed!");
    imported_pyarrow_wrappers = true;

#undef CHECK
}

// if expr is not true, form an err msg and raise a
// runtime_error with it
#define CHECK(expr, msg, file_type)                                  \
    if (!(expr)) {                                                   \
        std::string err_msg =                                        \
            std::string("Error in ") + file_type + " write: " + msg; \
        throw std::runtime_error(err_msg);                           \
    }

// if status of arrow::Result is not ok, print err msg and return
#define CHECK_ARROW(expr, msg, file_type)                                  \
    if (!(expr.ok())) {                                                    \
        std::string err_msg = std::string("Error in arrow ") + file_type + \
                              " write: " + msg + " " + expr.ToString();    \
        std::cerr << err_msg << std::endl;                                 \
        return;                                                            \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it. If it is ok, get value using ValueOrDie
// and assign it to lhs using std::move
#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs, file_type)                   \
    if (!(res.status().ok())) {                                            \
        std::string err_msg = std::string("Error in arrow ") + file_type + \
                              " write: " + msg + " " +                     \
                              res.status().ToString();                     \
        throw std::runtime_error(err_msg);                                 \
    }                                                                      \
    lhs = std::move(res).ValueOrDie();

// Same as fs_io.py
#define GCS_RETRY_LIMIT_SECONDS 3

std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;

std::string gen_pieces_file_name(int myrank, int num_ranks,
                                 const std::string &prefix,
                                 const std::string &suffix) {
    std::string part_number = std::to_string(myrank);
    std::string max_part_number = std::to_string(num_ranks - 1);
    int n_digits = max_part_number.length() +
                   1;  // number of digits I want the part numbers to have
    std::string new_part_number =
        std::string(n_digits - part_number.length(), '0') + part_number;
    std::stringstream ss;
    ss << prefix << new_part_number << suffix;  // this is the actual file name
    return ss.str();
}

void extract_fs_dir_path(const char *_path_name, bool is_parallel,
                         const std::string &prefix, const std::string &suffix,
                         int myrank, int num_ranks, Bodo_Fs::FsEnum *fs_option,
                         std::string *dirname, std::string *fname,
                         const std::string *orig_path, std::string *path_name) {
    *path_name = std::string(_path_name);

    if (strncmp(_path_name, "s3://", 5) == 0) {
        *fs_option = Bodo_Fs::s3;
        *path_name = std::string(_path_name + 5);  // remove s3://
    } else if ((strncmp(_path_name, "abfs://", 7) == 0 ||
                strncmp(_path_name, "abfss://", 8) == 0)) {
        *fs_option = Bodo_Fs::abfs;
#ifndef _WIN32
        auto parsed_opt =
            arrow::fs::AzureOptions::FromUri(_path_name, path_name);
        CHECK_ARROW(parsed_opt.status(), "FromUri failed for Azure File System",
                    "abfs");  // Check if parsing the URI was successful
#else
        throw std::runtime_error(
            "extract_fs_dir_path: Azure File System not supported on Windows.");
#endif
    } else if (strncmp(_path_name, "hdfs://", 7) == 0) {
        *fs_option = Bodo_Fs::hdfs;
        arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempRes =
            ::arrow::fs::FileSystemFromUri(*orig_path, path_name);
        if (!(tempRes.status().ok())) {
            std::cerr << "Error in arrow hdfs: FileSystemFromUri" << std::endl;
        }
    } else if ((strncmp(_path_name, "gcs://", 6) == 0) ||
               (strncmp(_path_name, "gs://", 5) == 0)) {
        *fs_option = Bodo_Fs::gcs;
        // remove gcs:// or gs://
        int chars_to_remove = (strncmp(_path_name, "gcs://", 6) == 0) ? 6 : 5;
        *path_name = std::string(_path_name + chars_to_remove);
    } else {  // posix
        *fs_option = Bodo_Fs::posix;
        *path_name = *orig_path;
    }

    if (is_parallel) {
        // construct file name for this process' piece
        *fname = gen_pieces_file_name(myrank, num_ranks, prefix, suffix);
        *dirname = *path_name;
    } else {
        // path_name is a file
        *fname = *path_name;
    }
}

std::pair<std::shared_ptr<arrow::fs::FileSystem>, std::string>
get_reader_file_system(std::string file_path, std::string s3_bucket_region,
                       bool s3fs_anon) {
    bool is_hdfs = file_path.starts_with("hdfs://") ||
                   file_path.starts_with("abfs://") ||
                   file_path.starts_with("abfss://");
    bool is_s3 = file_path.starts_with("s3://");
    bool is_gcs =
        file_path.starts_with("gcs://") || file_path.starts_with("gs://");
    std::shared_ptr<arrow::fs::FileSystem> fs;
    if (is_s3 || is_hdfs) {
        arrow::util::Uri uri;
        (void)uri.Parse(file_path);
        PyObject *fs_mod = nullptr;
        PyObject *func_obj = nullptr;
        if (is_s3) {
            import_fs_module(Bodo_Fs::s3, "", fs_mod);
            get_get_fs_pyobject(Bodo_Fs::s3, "", fs_mod, func_obj);
            s3_get_fs_t s3_get_fs =
                (s3_get_fs_t)PyNumber_AsSsize_t(func_obj, nullptr);
            std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
            s3_get_fs(&s3_fs, s3_bucket_region, s3fs_anon);
            fs = s3_fs;
            // remove s3:// prefix from file_path
            file_path = file_path.substr(strlen("s3://"));
        } else if (is_hdfs) {
            import_fs_module(Bodo_Fs::hdfs, "", fs_mod);
            get_get_fs_pyobject(Bodo_Fs::hdfs, "", fs_mod, func_obj);
            hdfs_get_fs_t hdfs_get_fs =
                (hdfs_get_fs_t)PyNumber_AsSsize_t(func_obj, nullptr);
            std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs;
            hdfs_get_fs(file_path, &hdfs_fs);
            fs = hdfs_fs;
            // remove hdfs://host:port prefix from file_path
            file_path = uri.path();
        }
        Py_DECREF(fs_mod);
        Py_DECREF(func_obj);
    } else if (is_gcs) {
        arrow::fs::GcsOptions options = arrow::fs::GcsOptions::Defaults();
        // Arrow seems to hang for a long time if retry isn't set
        options.retry_limit_seconds = GCS_RETRY_LIMIT_SECONDS;
        arrow::Result<std::shared_ptr<arrow::fs::GcsFileSystem>> result =
            arrow::fs::GcsFileSystem::Make(options,
                                           bodo::default_buffer_io_context());

        CHECK_ARROW_AND_ASSIGN(result, "GcsFileSystem::Make", fs,
                               std::string("GCS"));
        file_path = fs->PathFromUri(file_path).ValueOrDie();

        // Try with anonymous=True if not authenticated since Arrow doesn't try
        // automatically
        arrow::Status status = fs->GetFileInfo(file_path).status();
        if (!status.ok() && status.IsIOError() &&
            (status.message().find("Could not create a OAuth2 access token to "
                                   "authenticate the request") !=
             std::string::npos)) {
            arrow::fs::GcsOptions options = arrow::fs::GcsOptions::Anonymous();
            options.retry_limit_seconds = GCS_RETRY_LIMIT_SECONDS;
            arrow::Result<std::shared_ptr<arrow::fs::GcsFileSystem>> result =
                arrow::fs::GcsFileSystem::Make(
                    options, bodo::default_buffer_io_context());

            CHECK_ARROW_AND_ASSIGN(result, "GcsFileSystem::Make", fs,
                                   std::string("GCS"));
        }
    } else {
        fs = std::make_shared<arrow::fs::LocalFileSystem>();
    }
    return std::pair(fs, file_path);
}

void import_fs_module(Bodo_Fs::FsEnum fs_option, const std::string &file_type,
                      PyObject *&f_mod) {
    PyObject *ext_mod = PyImport_ImportModule("bodo.ext");
    if (fs_option == Bodo_Fs::s3) {
        f_mod = PyObject_GetAttrString(ext_mod, "s3_reader");
        CHECK(f_mod, "importing bodo.ext.s3_reader module failed", file_type);
    } else if (fs_option == Bodo_Fs::gcs) {
        f_mod = PyObject_GetAttrString(ext_mod, "gcs_reader");
        CHECK(f_mod, "importing bodo.ext.gcs_reader module failed", file_type);
    } else if (fs_option == Bodo_Fs::hdfs) {
        f_mod = PyObject_GetAttrString(ext_mod, "hdfs_reader");
        CHECK(f_mod, "importing bodo.ext.hdfs_reader module failed", file_type);
    }
    Py_DECREF(ext_mod);
}

void get_fs_reader_pyobject(Bodo_Fs::FsEnum fs_option,
                            const std::string &file_type, PyObject *f_mod,
                            PyObject *&func_obj) {
    if (fs_option == Bodo_Fs::s3) {
        func_obj = PyObject_GetAttrString(f_mod, "init_s3_reader");
        CHECK(func_obj, "getting s3_reader func_obj failed", file_type);
    } else if (fs_option == Bodo_Fs::gcs) {
        func_obj = PyObject_GetAttrString(f_mod, "init_gcs_reader");
        CHECK(func_obj, "getting gcs_reader func_obj failed", file_type);
    } else if (fs_option == Bodo_Fs::hdfs) {
        func_obj = PyObject_GetAttrString(f_mod, "init_hdfs_reader");
        CHECK(func_obj, "getting hdfs_reader func_obj failed", file_type);
    }
}

void get_get_fs_pyobject(Bodo_Fs::FsEnum fs_option,
                         const std::string &file_type, PyObject *f_mod,
                         PyObject *&func_obj) {
    if (fs_option == Bodo_Fs::s3) {
        func_obj = PyObject_GetAttrString(f_mod, "s3_get_fs");
        CHECK(func_obj, "getting s3_get_fs func_obj failed", file_type);
    } else if (fs_option == Bodo_Fs::gcs) {
        func_obj = PyObject_GetAttrString(f_mod, "gcs_get_fs");
        CHECK(func_obj, "getting gcs_get_fs func_obj failed", file_type);
    } else if (fs_option == Bodo_Fs::hdfs) {
        func_obj = PyObject_GetAttrString(f_mod, "hdfs_get_fs");
        CHECK(func_obj, "getting hdfs_get_fs func_obj failed", file_type);
    }
}

void open_file_outstream(
    Bodo_Fs::FsEnum fs_option, const std::string &file_type,
    const std::string &fname, std::shared_ptr<arrow::fs::S3FileSystem> s3_fs,
    std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs,
    std::shared_ptr<::arrow::io::OutputStream> *out_stream) {
    if (fs_option == Bodo_Fs::posix) {
        arrow::Result<std::shared_ptr<arrow::io::OutputStream>> result =
            arrow::io::FileOutputStream::Open(fname);
        CHECK_ARROW_AND_ASSIGN(result, "FileOutputStream::Open", *out_stream,
                               file_type)
    } else if (fs_option == Bodo_Fs::s3) {
        arrow::Result<std::shared_ptr<arrow::io::OutputStream>> result =
            s3_fs->OpenOutputStream(fname);
        CHECK_ARROW_AND_ASSIGN(result, "S3FileSystem::OpenOutputStream",
                               *out_stream, file_type)
    } else if (fs_option == Bodo_Fs::hdfs) {
        arrow::Result<std::shared_ptr<arrow::io::OutputStream>> result =
            hdfs_fs->OpenOutputStream(fname);
        CHECK_ARROW_AND_ASSIGN(result, "HdfsFileSystem::OpenOutputStream",
                               *out_stream, file_type)
    }
}

void posix_open_file_outstream(
    const std::string &file_type, const std::string &fname,
    std::shared_ptr<::arrow::io::OutputStream> *out_stream) {
    arrow::Result<std::shared_ptr<arrow::io::OutputStream>> result =
        arrow::io::FileOutputStream::Open(fname);
    CHECK_ARROW_AND_ASSIGN(result, "FileOutputStream::Open", *out_stream,
                           file_type)
}

void open_file_outstream_gcs(
    Bodo_Fs::FsEnum fs_option, const std::string &file_type,
    const std::string &fname, std::shared_ptr<arrow::py::fs::PyFileSystem> fs,
    std::shared_ptr<arrow::io::OutputStream> *out_stream) {
    arrow::Result<std::shared_ptr<arrow::io::OutputStream>> result =
        fs->OpenOutputStream(fname);
    CHECK_ARROW_AND_ASSIGN(result, "PyFileSystem::OpenOutputStream",
                           *out_stream, file_type);
}

void open_file_appendstream(
    const std::string &file_type, const std::string &fname,
    std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs,
    std::shared_ptr<::arrow::io::OutputStream> *out_stream) {
    *out_stream = hdfs_fs->OpenAppendStream(fname).ValueOrDie();
}

void create_dir_posix(int myrank, std::string &dirname, std::string &path_name,
                      bool recreate_if_present) {
    // create output directory
    if (myrank == 0) {
        if (std::filesystem::exists(dirname)) {
            if (recreate_if_present) {
                std::filesystem::remove_all(dirname);
                std::filesystem::create_directories(dirname);
            } else if (!std::filesystem::is_directory(dirname)) {
                std::cerr << "Bodo parquet write ERROR: a process reports "
                             "that path "
                          << path_name << " exists and is not a directory"
                          << std::endl;
            }
        } else {
            std::filesystem::create_directories(dirname);
        }
    }
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD),
              "create_dir_posix: MPI error on MPI_Barrier:");
}

void create_dir_hdfs(int myrank, std::string &dirname, std::string &orig_path,
                     const std::string &file_type, bool recreate_if_present) {
    PyObject *f_mod = nullptr;
    // get hdfs_get_fs function
    PyObject *hdfs_func_obj = nullptr;
    import_fs_module(Bodo_Fs::hdfs, file_type, f_mod);
    get_get_fs_pyobject(Bodo_Fs::hdfs, file_type, f_mod, hdfs_func_obj);
    hdfs_get_fs_t hdfs_get_fs =
        (hdfs_get_fs_t)PyNumber_AsSsize_t(hdfs_func_obj, nullptr);
    arrow::Status status;
    std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs;
    hdfs_get_fs(orig_path, &hdfs_fs);
    if (myrank == 0) {
        if (recreate_if_present) {
            arrow::Result<arrow::fs::FileInfo> result;
            result = hdfs_fs->GetFileInfo(dirname);
            if (!result.ok()) {
                throw std::runtime_error("GetFileInfo failed.");
            }
            arrow::fs::FileInfo info;
            info = result.ValueOrDie();
            if (info.type() != arrow::fs::FileType::NotFound) {
                status = hdfs_fs->DeleteDir(dirname);
                CHECK_ARROW(status, "Hdfs::DeleteDir", file_type);
            }
        }
        status = hdfs_fs->CreateDir(dirname);
        CHECK_ARROW(status, "Hdfs::MakeDirectory", file_type);
    }
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD),
              "create_dir_hdfs: MPI error on MPI_Barrier:");
    Py_DECREF(f_mod);
    Py_DECREF(hdfs_func_obj);
}

void create_dir_parallel(Bodo_Fs::FsEnum fs_option, int myrank,
                         std::string &dirname, std::string &path_name,
                         std::string &orig_path, const std::string &file_type,
                         bool recreate_if_present) {
    if (fs_option == Bodo_Fs::posix) {
        create_dir_posix(myrank, dirname, path_name, recreate_if_present);
    } else if (fs_option == Bodo_Fs::hdfs) {
        create_dir_hdfs(myrank, dirname, orig_path, file_type,
                        recreate_if_present);
    }
}

void open_outstream(Bodo_Fs::FsEnum fs_option, bool is_parallel,
                    const std::string &file_type, std::string &dirname,
                    std::string &fname, std::string &orig_path,
                    std::shared_ptr<::arrow::io::OutputStream> *out_stream,
                    const std::string &bucket_region) {
    PyObject *f_mod = nullptr;
    switch (fs_option) {
        case Bodo_Fs::posix: {
            if (file_type == "csv") {
                // csv does not need to open_outstream for posix
                // in fact, this function, open_outstream,
                // should never be called in such case
                return;
            }
            if (is_parallel) {
                // assumes that the directory has already been created
                std::filesystem::path out_path(dirname);
                out_path /= fname;  // append file name to output path
                posix_open_file_outstream(file_type, out_path.string(),
                                          out_stream);
            } else {
                posix_open_file_outstream(file_type, fname, out_stream);
            }
            return;
        } break;
        case Bodo_Fs::s3: {
            // get s3_get_fs function
            PyObject *s3_func_obj = nullptr;
            import_fs_module(fs_option, file_type, f_mod);
            get_get_fs_pyobject(fs_option, file_type, f_mod, s3_func_obj);
            s3_get_fs_t s3_get_fs =
                (s3_get_fs_t)PyNumber_AsSsize_t(s3_func_obj, nullptr);

            s3_get_fs(&s3_fs, bucket_region, false);
            if (is_parallel) {
                std::filesystem::path out_path(dirname);
                out_path /= fname;  // append file name to output path
                // Using generic_string() to avoid "\" generated on Windows for
                // remote object storage
                std::string out_path_str = out_path.generic_string();
                open_file_outstream(fs_option, file_type, out_path_str, s3_fs,
                                    nullptr, out_stream);
            } else {
                open_file_outstream(fs_option, file_type, fname, s3_fs, nullptr,
                                    out_stream);
            }

            Py_DECREF(f_mod);
            Py_DECREF(s3_func_obj);
            return;
        } break;
        case Bodo_Fs::hdfs: {
            // get hdfs_get_fs function
            PyObject *hdfs_func_obj = nullptr;
            import_fs_module(fs_option, file_type, f_mod);
            get_get_fs_pyobject(fs_option, file_type, f_mod, hdfs_func_obj);
            hdfs_get_fs_t hdfs_get_fs =
                (hdfs_get_fs_t)PyNumber_AsSsize_t(hdfs_func_obj, nullptr);

            std::shared_ptr<::arrow::io::HdfsOutputStream> hdfs_out_stream;
            arrow::Status status;
            // TODO: Do I need to make this Buffer Pool aware?
            std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs;
            hdfs_get_fs(orig_path, &hdfs_fs);
            if (is_parallel) {
                // assumes that the directory has already been created
                std::filesystem::path out_path(dirname);
                out_path /= fname;
                open_file_outstream(fs_option, file_type, out_path.string(),
                                    nullptr, hdfs_fs, out_stream);
            } else {
                open_file_outstream(fs_option, file_type, fname, nullptr,
                                    hdfs_fs, out_stream);
            }

            Py_DECREF(f_mod);
            Py_DECREF(hdfs_func_obj);
            return;
        } break;
        case Bodo_Fs::abfs: {
#ifndef _WIN32
            ensure_pa_wrappers_imported();

            std::string path =
                is_parallel ? (std::filesystem::path(dirname) / fname).string()
                            : fname;
            arrow::fs::AzureOptions opts;
            arrow::Result<arrow::fs::AzureOptions> opts_res =
                arrow::fs::AzureOptions::FromUri(path, &path);
            CHECK_ARROW_AND_ASSIGN(opts_res, "AzureOptions::FromUri", opts,
                                   file_type);

            PyObject *fs_io_mod = PyImport_ImportModule("bodo.io.fs_io");
            PyObject *abfs_get_fs =
                PyObject_GetAttrString(fs_io_mod, "abfs_get_fs");
            PyObject *storage_options = PyDict_New();
            if (opts.account_name != "") {
                PyObject *storage_account_py_str =
                    PyUnicode_FromString(opts.account_name.c_str());
                PyDict_SetItemString(storage_options, "account_name",
                                     storage_account_py_str);
                Py_DECREF(storage_account_py_str);
            }
            PyObject *fs_pyobject =
                PyObject_CallFunction(abfs_get_fs, "O", storage_options);
            std::shared_ptr<::arrow::fs::FileSystem> fs;
            CHECK_ARROW_AND_ASSIGN(arrow::py::unwrap_filesystem(fs_pyobject),
                                   "AzureFileSystem::unwrap_filesystem", fs,
                                   file_type);

            Py_DECREF(fs_pyobject);
            Py_DECREF(storage_options);
            Py_DECREF(abfs_get_fs);
            Py_DECREF(fs_io_mod);
            arrow::Result<std::shared_ptr<arrow::io::OutputStream>> result =
                fs->OpenOutputStream(path);
            CHECK_ARROW_AND_ASSIGN(result, "AzureFileSystem::OpenOutputStream",
                                   *out_stream, file_type)
#else
            // Using AzureFileSystem leads to a compilation error on Windows.
            // https://github.com/apache/arrow/issues/41990
            throw std::runtime_error(
                "open_outstream: AzureFileSystem not supported on Windows.");
#endif
        } break;
        case Bodo_Fs::gcs: {
            PyObject *gcs_func_obj = nullptr;
            import_fs_module(fs_option, file_type, f_mod);
            get_get_fs_pyobject(fs_option, file_type, f_mod, gcs_func_obj);
            gcs_get_fs_t gcs_get_fs =
                (gcs_get_fs_t)PyNumber_AsSsize_t(gcs_func_obj, nullptr);
            std::shared_ptr<::arrow::py::fs::PyFileSystem> fs;
            gcs_get_fs(&fs);
            if (is_parallel) {
                std::filesystem::path out_path(dirname);
                out_path /= fname;
                // Using generic_string() to avoid "\" generated on Windows for
                // remote object storage
                std::string out_path_str = out_path.generic_string();
                open_file_outstream_gcs(fs_option, file_type, out_path_str, fs,
                                        out_stream);
            } else {
                open_file_outstream_gcs(fs_option, file_type, fname, fs,
                                        out_stream);
            }

            Py_DECREF(f_mod);
            Py_DECREF(gcs_func_obj);
        } break;
        default: {
            throw std::runtime_error(
                "open output stream: unrecognized filesystem");
        }
    }
}

void parallel_in_order_write(
    Bodo_Fs::FsEnum fs_option, const std::string &file_type,
    const std::string &fname, char *buff, int64_t count, int64_t elem_size,
    std::shared_ptr<arrow::fs::S3FileSystem> s3_fs,
    std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs) {
    int myrank, num_ranks;
    std::shared_ptr<::arrow::io::OutputStream> out_stream;

    int token_tag = 0;
    int token = -1;  // placeholder token used for communication
    int64_t buff_size = count * elem_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    CHECK_MPI(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
              "parallel_in_order_write: MPI error on MPI_Comm_set_errhandler:");

    if (fs_option == Bodo_Fs::s3) {
        int size_tag = 1;
        int data_tag = 2;
        int complete_tag = 3;
        int complete = 0;       // set to 1 after sent/received the entire buff
                                // by a 0/non-zero rank
        int send_size = 0;      // size of buff sending, used by non-zero ranks
        int64_t sent_size = 0;  // size of buff sent, used by non-zero ranks
        int recv_buff_size = 0;

        // all but rank 0 receive message first
        // then send buff to rank 0
        if (myrank != 0) {
            // first receives signal from rank 0
            CHECK_MPI(MPI_Recv(&token, 1, MPI_INT, 0, token_tag, MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE),
                      "parallel_in_order_write: MPI error on MPI_Recv:");
            do {
                // Always send complete in case our rank has no data.
                complete = sent_size >= buff_size;
                CHECK_MPI(MPI_Send(&complete, 1, MPI_INT, 0, complete_tag,
                                   MPI_COMM_WORLD),
                          "parallel_in_order_write: MPI error on MPI_Send:");
                if (!complete) {
                    // send buff in chunks no bigger than INT_MAX
                    if (buff_size - sent_size > INT_MAX) {
                        send_size = INT_MAX;
                    } else {
                        send_size = buff_size - sent_size;
                    }
                    // send chunk size
                    CHECK_MPI(
                        MPI_Send(&send_size, 1, MPI_INT, 0, size_tag,
                                 MPI_COMM_WORLD),
                        "parallel_in_order_write: MPI error on MPI_Send:");
                    // send chunk data;
                    CHECK_MPI(
                        MPI_Send(buff + sent_size, send_size, MPI_CHAR, 0,
                                 data_tag, MPI_COMM_WORLD),
                        "parallel_in_order_write: MPI error on MPI_Send:");
                    sent_size += send_size;
                }
            } while (!complete);
        } else {
            // 0 rank open outstream first
            open_file_outstream(Bodo_Fs::s3, "", fname, s3_fs, nullptr,
                                &out_stream);
            // 0 rank use vector `recv_buffer` to store buff
            std::vector<char> recv_buffer;
            // receives buff from other ranks in order in chunks
            // write to outstream
            // once entire buff is received, sends signal to the next rank
            for (int rank = 0; rank < num_ranks; rank++) {
                if (rank == 0) {
                    // 0 rank write its own buff to outstream
                    CHECK_ARROW(out_stream->Write(buff, buff_size),
                                "arrow::io::S3OutputStream::Write", file_type);
                } else {
                    do {
                        // receive whether the entire buff is received
                        CHECK_MPI(
                            MPI_Recv(&complete, 1, MPI_INT, rank, complete_tag,
                                     MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                            "parallel_in_order_write: MPI error on MPI_Recv:");
                        if (!complete) {
                            // first receive size of incoming data
                            CHECK_MPI(MPI_Recv(&recv_buff_size, 1, MPI_INT,
                                               rank, size_tag, MPI_COMM_WORLD,
                                               MPI_STATUS_IGNORE),
                                      "parallel_in_order_write: MPI error on "
                                      "MPI_Recv:");
                            // resize recv_buffer to fit incoming data
                            recv_buffer.resize(recv_buff_size);
                            // receive buffer data

                            CHECK_MPI(
                                MPI_Recv(&recv_buffer[0], recv_buff_size,
                                         MPI_CHAR, rank, data_tag,
                                         MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                                "parallel_in_order_write: MPI error on "
                                "MPI_Recv:");
                            CHECK_ARROW(out_stream->Write(recv_buffer.data(),
                                                          recv_buff_size),
                                        "arrow::io::S3OutputStream::Write",
                                        file_type);
                        }
                    } while (!complete);
                }
                if (rank != num_ranks - 1) {
                    CHECK_MPI(
                        MPI_Send(&token, 1, MPI_INT, rank + 1, token_tag,
                                 MPI_COMM_WORLD),
                        "parallel_in_order_write: MPI error on MPI_Send:");
                }
            }
            CHECK_ARROW(out_stream->Close(), "arrow::io::S3OutputStream::Close",
                        file_type);
        }
    } else if (fs_option == Bodo_Fs::hdfs) {
        // all but the first rank receive message first
        // then open append stream
        if (myrank != 0) {
            CHECK_MPI(MPI_Recv(&token, 1, MPI_INT, myrank - 1, token_tag,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                      "parallel_in_order_write: MPI error on MPI_Recv:");
            open_file_appendstream(file_type, fname, hdfs_fs, &out_stream);
        } else {  // 0 rank open outstream instead
            open_file_outstream(Bodo_Fs::hdfs, file_type, fname, nullptr,
                                hdfs_fs, &out_stream);
        }

        // all ranks write & close stream
        CHECK_ARROW(out_stream->Write(buff, buff_size),
                    "arrow::io::HdfsOutputStream::Write", file_type);
        CHECK_ARROW(out_stream->Close(), "arrow::io::HdfsOutputStream::Close",
                    file_type);

        // all but the last rank send message
        if (myrank != num_ranks - 1) {
            CHECK_MPI(MPI_Send(&token, 1, MPI_INT, myrank + 1, token_tag,
                               MPI_COMM_WORLD),
                      "parallel_in_order_write: MPI error on MPI_Send:");
        }
    }
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD),
              "parallel_in_order_write: MPI error on MPI_Barrier:");
}

std::shared_ptr<::arrow::fs::FileSystem> get_fs_for_path(const char *_path_name,
                                                         bool is_parallel) {
    ensure_pa_wrappers_imported();
    std::shared_ptr<arrow::fs::FileSystem> fs;
    // scheme = bodo.io.fs_io(_path_name)
    // fs = bodo.io.fs_io.getfs(_path_name, scheme, None, is_parallel,
    // force_hdfs)
    PyObject *fs_io_mod = PyImport_ImportModule("bodo.io.fs_io");
    PyObject *scheme =
        PyObject_CallMethod(fs_io_mod, "get_uri_scheme", "s", _path_name);
    PyObject *fs_obj =
        PyObject_CallMethod(fs_io_mod, "getfs", "sOOO", _path_name, scheme,
                            Py_None, is_parallel ? Py_True : Py_False);
    CHECK_ARROW_AND_ASSIGN(arrow::py::unwrap_filesystem(fs_obj),
                           "arrow::py::unwrap_filesystem", fs, "");
    Py_DECREF(fs_io_mod);
    Py_DECREF(scheme);
    Py_DECREF(fs_obj);
    return fs;
}
