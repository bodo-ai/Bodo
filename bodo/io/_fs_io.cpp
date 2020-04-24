// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <arrow/io/api.h>
#include <boost/filesystem/operations.hpp>
#include <iostream>
#include <sstream>
#include <string>

#include "../libs/_bodo_common.h"
#include "_fs_io.h"
#include "arrow/filesystem/filesystem.h"
#include "arrow/filesystem/s3fs.h"
#include "arrow/result.h"
#include "mpi.h"

// decimal_mpi_type declared in _distributed.h as extern has
// to be defined in each extension module that includes it
MPI_Datatype decimal_mpi_type = MPI_DATATYPE_NULL;

#define CHECK(expr, msg, file_type)                                \
    if (!(expr)) {                                                 \
        std::cerr << "Error in " << file_type << " write: " << msg \
                  << std::endl;                                    \
        return;                                                    \
    }

#define CHECK_ARROW(expr, msg, file_type)                                \
    if (!(expr.ok())) {                                                  \
        std::cerr << "Error in arrow " << file_type << " write: " << msg \
                  << " " << expr << std::endl;                           \
        return;                                                          \
    }

#define CHECK_MPI(ierr, err_class, err_string, err_len, file_name)         \
    if (ierr != 0) {                                                       \
        MPI_Error_class(ierr, &err_class);                                 \
        MPI_Error_string(ierr, err_string, err_len);                       \
        printf("Error %s\n", err_string);                                  \
        fflush(stdout);                                                    \
        Bodo_PyErr_SetString(                                              \
            PyExc_RuntimeError,                                            \
            ("File write error: " + std::to_string(err_class) + file_name) \
                .c_str());                                                 \
    }

#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs, file_type) \
    CHECK_ARROW(res.status(), msg, file_type)            \
    lhs = std::move(res).ValueOrDie();

std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
std::shared_ptr<::arrow::io::HadoopFileSystem> hdfs_fs;

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
    } else {  // posix
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

void import_fs_module(Bodo_Fs::FsEnum fs_option, const std::string &file_type,
                      PyObject *&f_mod) {
    if (fs_option == Bodo_Fs::s3) {
        f_mod = PyImport_ImportModule("bodo.io.s3_reader");
        CHECK(f_mod, "importing bodo.io.s3_reader module failed", file_type);
    } else if (fs_option == Bodo_Fs::hdfs) {
        f_mod = PyImport_ImportModule("bodo.io.hdfs_reader");
        CHECK(f_mod, "importing bodo.io.hdfs_readerder module failed",
              file_type);
    }
}

void get_fs_reader_pyobject(Bodo_Fs::FsEnum fs_option,
                            const std::string &file_type, PyObject *f_mod,
                            PyObject *&func_obj) {
    if (fs_option == Bodo_Fs::s3) {
        func_obj = PyObject_GetAttrString(f_mod, "init_s3_reader");
        CHECK(func_obj, "getting s3_reader func_obj failed", file_type);
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
    } else if (fs_option == Bodo_Fs::hdfs) {
        func_obj = PyObject_GetAttrString(f_mod, "hdfs_get_fs");
        CHECK(func_obj, "getting hdfs_get_fs func_obj failed", file_type);
    }
}

void open_file_outstream(
    Bodo_Fs::FsEnum fs_option, const std::string &file_type,
    const std::string &fname, std::shared_ptr<arrow::fs::S3FileSystem> s3_fs,
    std::shared_ptr<::arrow::io::HadoopFileSystem> hdfs_fs,
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
        std::shared_ptr<::arrow::io::HdfsOutputStream> hdfs_out_stream;
        CHECK_ARROW(hdfs_fs->OpenWritable(fname, false, &hdfs_out_stream),
                    "Hdfs::OpenWritable", file_type);
        *out_stream = hdfs_out_stream;
        ;
    }
}

void open_file_appendstream(
    const std::string &file_type, const std::string &fname,
    std::shared_ptr<::arrow::io::HadoopFileSystem> hdfs_fs,
    std::shared_ptr<::arrow::io::OutputStream> *out_stream) {
    std::shared_ptr<::arrow::io::HdfsOutputStream> hdfs_out_stream;
    CHECK_ARROW(hdfs_fs->OpenWritable(fname, true, &hdfs_out_stream),
                "Hdfs::OpenWritable", file_type);
    *out_stream = hdfs_out_stream;
    ;
}

void open_outstream(Bodo_Fs::FsEnum fs_option, bool is_parallel, int myrank,
                    const std::string &file_type, std::string &dirname,
                    std::string &fname, std::string &orig_path,
                    std::string &path_name,
                    std::shared_ptr<::arrow::io::OutputStream> *out_stream) {
    PyObject *f_mod = nullptr;
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
            open_file_outstream(fs_option, file_type, out_path.string(), NULL,
                                NULL, out_stream);
        } else {
            open_file_outstream(fs_option, file_type, fname, NULL, NULL,
                                out_stream);
        }
        return;
    } else if (fs_option == Bodo_Fs::s3) {
        // get s3_get_fs function
        PyObject *s3_func_obj = nullptr;
        import_fs_module(fs_option, file_type, f_mod);
        get_get_fs_pyobject(fs_option, file_type, f_mod, s3_func_obj);
        s3_get_fs_t s3_get_fs =
            (s3_get_fs_t)PyNumber_AsSsize_t(s3_func_obj, NULL);

        s3_get_fs(&s3_fs);
        if (is_parallel) {
            open_file_outstream(fs_option, file_type, dirname + "/" + fname,
                                s3_fs, NULL, out_stream);
        } else {
            open_file_outstream(fs_option, file_type, fname, s3_fs, NULL,
                                out_stream);
        }

        Py_DECREF(f_mod);
        Py_DECREF(s3_func_obj);
        return;
    } else {  // fs == Bodo_Fs::hdfs
        // get hdfs_get_fs function
        PyObject *hdfs_func_obj = nullptr;
        import_fs_module(fs_option, file_type, f_mod);
        get_get_fs_pyobject(fs_option, file_type, f_mod, hdfs_func_obj);
        hdfs_get_fs_t hdfs_get_fs =
            (hdfs_get_fs_t)PyNumber_AsSsize_t(hdfs_func_obj, NULL);

        std::shared_ptr<::arrow::io::HdfsOutputStream> hdfs_out_stream;
        arrow::Status status;
        hdfs_get_fs(orig_path, &hdfs_fs);
        if (is_parallel) {
            if (myrank == 0) {
                status = hdfs_fs->MakeDirectory(dirname);
                CHECK_ARROW(status, "Hdfs::MakeDirectory", file_type);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            open_file_outstream(fs_option, file_type, dirname + "/" + fname,
                                NULL, hdfs_fs, out_stream);
        } else {
            open_file_outstream(fs_option, file_type, fname, NULL, hdfs_fs,
                                out_stream);
        }

        Py_DECREF(f_mod);
        Py_DECREF(hdfs_func_obj);
        return;
    }
}

void parallel_in_order_write(
    Bodo_Fs::FsEnum fs_option, const std::string &file_type,
    const std::string &fname, char *buff, int64_t count, int64_t elem_size,
    std::shared_ptr<arrow::fs::S3FileSystem> s3_fs,
    std::shared_ptr<::arrow::io::HadoopFileSystem> hdfs_fs) {
    int myrank, num_ranks, ierr, err_len, err_class;
    std::shared_ptr<::arrow::io::OutputStream> out_stream;

    int token_tag = 0;
    int token = -1;  // placeholder token used for communication
    int64_t buff_size = count * elem_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    char err_string[MPI_MAX_ERROR_STRING];
    err_string[MPI_MAX_ERROR_STRING - 1] = '\0';
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

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
            MPI_Recv(&token, 1, MPI_INT, 0, token_tag, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            // send buff in chunks no bigger than INT_MAX
            while (sent_size < buff_size) {
                if (buff_size - sent_size > INT_MAX)
                    send_size = INT_MAX;
                else
                    send_size = buff_size - sent_size;
                // send chunk size
                ierr = MPI_Send(&send_size, 1, MPI_INT, 0, size_tag,
                                MPI_COMM_WORLD);
                CHECK_MPI(ierr, err_class, err_string, &err_len, fname);
                // send chunk data;
                ierr = MPI_Send(buff + sent_size, send_size, MPI_CHAR, 0,
                                data_tag, MPI_COMM_WORLD);
                CHECK_MPI(ierr, err_class, err_string, &err_len, fname);
                sent_size += send_size;
                if (sent_size == buff_size) complete = 1;
                // signal rank 0 whether the entire buff is sent
                ierr = MPI_Send(&complete, 1, MPI_INT, 0, complete_tag,
                                MPI_COMM_WORLD);
                CHECK_MPI(ierr, err_class, err_string, &err_len, fname);
            }
        } else {
            // 0 rank open outstream first
            open_file_outstream(Bodo_Fs::s3, "", fname, s3_fs, NULL,
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
                    while (complete == 0) {
                        // first receive size of incoming data
                        ierr = MPI_Recv(&recv_buff_size, 1, MPI_INT, rank,
                                        size_tag, MPI_COMM_WORLD,
                                        MPI_STATUS_IGNORE);
                        CHECK_MPI(ierr, err_class, err_string, &err_len, fname);
                        // resize recv_buffer to fit incoming data
                        recv_buffer.resize(recv_buff_size);
                        // receive buffer data
                        ierr = MPI_Recv(&recv_buffer[0], recv_buff_size,
                                        MPI_CHAR, rank, data_tag,
                                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        CHECK_MPI(ierr, err_class, err_string, &err_len, fname);
                        CHECK_ARROW(out_stream->Write(recv_buffer.data(),
                                                      recv_buff_size),
                                    "arrow::io::S3OutputStream::Write",
                                    file_type);
                        // receive whether the entire buff is received
                        ierr =
                            MPI_Recv(&complete, 1, MPI_INT, rank, complete_tag,
                                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        CHECK_MPI(ierr, err_class, err_string, &err_len, fname);
                    }
                }
                if (rank != num_ranks - 1) {
                    ierr = MPI_Send(&token, 1, MPI_INT, rank + 1, token_tag,
                                    MPI_COMM_WORLD);
                    CHECK_MPI(ierr, err_class, err_string, &err_len, fname);
                }
                complete = 0;
            }
            CHECK_ARROW(out_stream->Close(), "arrow::io::S3OutputStream::Close",
                        file_type);
        }
    } else if (fs_option == Bodo_Fs::hdfs) {
        // all but the first rank receive message first
        // then open append stream
        if (myrank != 0) {
            ierr = MPI_Recv(&token, 1, MPI_INT, myrank - 1, token_tag,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_MPI(ierr, err_class, err_string, &err_len, fname);
            open_file_appendstream(file_type, fname, hdfs_fs, &out_stream);
        } else {  // 0 rank open outstream instead
            open_file_outstream(Bodo_Fs::hdfs, file_type, fname, NULL, hdfs_fs,
                                &out_stream);
        }

        // all ranks write & close stream
        CHECK_ARROW(out_stream->Write(buff, buff_size),
                    "arrow::io::HdfsOutputStream::Write", file_type);
        CHECK_ARROW(out_stream->Close(), "arrow::io::HdfsOutputStream::Close",
                    file_type);

        // all but the last rank send message
        if (myrank != num_ranks - 1) {
            ierr = MPI_Send(&token, 1, MPI_INT, myrank + 1, token_tag,
                            MPI_COMM_WORLD);
            CHECK_MPI(ierr, err_class, err_string, &err_len, fname);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
