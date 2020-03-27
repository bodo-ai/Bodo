// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <boost/filesystem/operations.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <arrow/io/api.h>

#include "mpi.h"
#include "_writer.h"
#include "arrow/filesystem/filesystem.h"
#include "arrow/filesystem/s3fs.h"
#include "arrow/result.h"

typedef void (*s3_get_fs_t)(std::shared_ptr<::arrow::fs::S3FileSystem> *);
typedef void (*hdfs_get_fs_t)(const std::string &,
                              std::shared_ptr<::arrow::io::HadoopFileSystem> *);

#define CHECK(expr, msg, file_type)                               \
    if (!(expr)) {                                                \
        std::cerr << "Error in " << file_type << " write " << msg \
                  << std::endl;                                   \
        return;                                                   \
    }

#define CHECK_ARROW(expr, msg, file_type)                                      \
    if (!(expr.ok())) {                                                        \
        std::cerr << "Error in arrow " << file_type << " write " << msg << " " \
                  << expr << std::endl;                                        \
        return;                                                                \
    }

#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs, file_type) \
    CHECK_ARROW(res.status(), msg, file_type)            \
    lhs = std::move(res).ValueOrDie();

std::string gen_pieces_file_name(int myrank, int num_ranks,
                                 const std::string &suffix) {
    std::string part_number = std::to_string(myrank);
    std::string max_part_number = std::to_string(num_ranks - 1);
    int n_digits = max_part_number.length() +
                   1;  // number of digits I want the part numbers to have
    std::string new_part_number =
        std::string(n_digits - part_number.length(), '0') + part_number;
    std::stringstream ss;
    ss << "part-" << new_part_number << suffix;  // this is the actual file name
    return ss.str();
}

void extract_fs_dir_path(const char *_path_name, bool is_parallel,
                         const std::string &suffix, int myrank, int num_ranks,
                         Bodo_Fs::FsEnum *fs_option, std::string *dirname,
                         std::string *fname, std::string *orig_path,
                         std::string *path_name) {
    *path_name = std::string(_path_name);

    if (strncmp(_path_name, "s3://", 5) == 0) {
        *fs_option = Bodo_Fs::s3;
        *path_name = std::string(_path_name + 5);  // remove s3://
    } else if (strncmp(_path_name, "hdfs://", 7) == 0) {
        *fs_option = Bodo_Fs::hdfs;
        arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempRes =
            ::arrow::fs::FileSystemFromUri(*orig_path, path_name);
        if (!(tempRes.status().ok())) {
            std::cerr << "Error in arrow hdfs: FileSystemFromUri" << std::endl;
        }
    } else { //posix
        *fs_option = Bodo_Fs::posix;
        *path_name = *orig_path;
    }

    if (is_parallel) {
        // construct file name for this process' piece
        *fname = gen_pieces_file_name(myrank, num_ranks, suffix);
        *dirname = *path_name;
    } else {
        // path_name is a file
        *fname = *path_name;
    }
}

void open_outstream(Bodo_Fs::FsEnum fs_option, bool is_parallel, int myrank,
                    const std::string &file_type,
                    std::string &dirname, std::string &fname,
                    std::string &orig_path, std::string &path_name,
                    std::shared_ptr<::arrow::io::OutputStream> *out_stream) {
    if (fs_option == Bodo_Fs::posix) {
        if (file_type == "csv") {
            // csv does not need to open_outstream for posix
            // in fact, this function, open_outstream, 
            // should never be called in such case
            return;
        }
        if (is_parallel) {
            // create output directory
            int error = 0;
            if (boost::filesystem::exists(dirname)) {
                if (!boost::filesystem::is_directory(dirname)) error = 1;
            } else {
                // for the parallel case, 'dirname' is the directory where the
                // different parts of the distributed table are stored (each as
                // a file)
                boost::filesystem::create_directory(dirname);
            }
            MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_INT, MPI_LOR,
                          MPI_COMM_WORLD);
            if (error) {
                if (myrank == 0)
                    std::cerr << "Bodo parquet write ERROR: a process reports "
                                 "that path "
                              << path_name << " exists and is not a directory"
                              << std::endl;
                return;
            }
            boost::filesystem::path out_path(dirname);
            out_path /= fname;  // append file name to output path
            CHECK_ARROW(arrow::io::FileOutputStream::Open(out_path.string(),
                                                          out_stream),
                        "error opening file for parquet output", file_type);
        } else {
            CHECK_ARROW(arrow::io::FileOutputStream::Open(fname, out_stream),
                        "error opening file for parquet output", file_type);
        }
        return;
    } else if (fs_option == Bodo_Fs::s3) {
        // get s3_get_fs function
        PyObject *s3_mod = PyImport_ImportModule("bodo.io.s3_reader");
        CHECK(s3_mod, "importing bodo.io.s3_reader module failed", file_type);
        PyObject *s3_func_obj = PyObject_GetAttrString(s3_mod, "s3_get_fs");
        CHECK(s3_func_obj, "getting s3_get_fs func_obj failed", file_type);
        s3_get_fs_t s3_get_fs =
            (s3_get_fs_t)PyNumber_AsSsize_t(s3_func_obj, NULL);

        std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
        s3_get_fs(&s3_fs);
        if (is_parallel) {
            arrow::Result<std::shared_ptr<arrow::io::OutputStream>> result =
                s3_fs->OpenOutputStream(dirname + "/" + fname);
            CHECK_ARROW_AND_ASSIGN(result, "S3FileSystem::OpenOutputStream",
                                   *out_stream, file_type)
        } else {
            arrow::Result<std::shared_ptr<arrow::io::OutputStream>> result =
                s3_fs->OpenOutputStream(fname);
            CHECK_ARROW_AND_ASSIGN(result, "S3FileSystem::OpenOutputStream",
                                   *out_stream, file_type)
        }

        Py_DECREF(s3_mod);
        Py_DECREF(s3_func_obj);
        return;
    } else {  // fs == Bodo_Fs::hdfs
        // get hdfs_get_fs function
        PyObject *hdfs_mod = PyImport_ImportModule("bodo.io.hdfs_reader");
        CHECK(hdfs_mod, "importing bodo.io.hdfs_reader module failed",
              file_type);
        PyObject *hdfs_func_obj =
            PyObject_GetAttrString(hdfs_mod, "hdfs_get_fs");
        CHECK(hdfs_func_obj, "getting hdfs_get_fs func_obj failed", file_type);
        hdfs_get_fs_t hdfs_get_fs =
            (hdfs_get_fs_t)PyNumber_AsSsize_t(hdfs_func_obj, NULL);

        std::shared_ptr<::arrow::io::HadoopFileSystem> hdfs_fs;
        std::shared_ptr<::arrow::io::HdfsOutputStream> hdfs_out_stream;
        arrow::Status status;
        hdfs_get_fs(orig_path, &hdfs_fs);
        if (is_parallel) {
            if (myrank == 0) {
                status = hdfs_fs->MakeDirectory(dirname);
                CHECK_ARROW(status, "Hdfs::MakeDirectory", file_type);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            status = hdfs_fs->OpenWritable(dirname + "/" + fname, false,
                                           &hdfs_out_stream);
        } else {
            status = hdfs_fs->OpenWritable(fname, false, &hdfs_out_stream);
        }
        CHECK_ARROW(status, "Hdfs::OpenWritable", file_type);
        *out_stream = hdfs_out_stream;

        Py_DECREF(hdfs_mod);
        Py_DECREF(hdfs_func_obj);
        return;
    }
}
