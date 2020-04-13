// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <boost/filesystem.hpp>
#include <climits>
#include <cstdio>
#include <iostream>
#include <string>
#include "../libs/_bodo_common.h"
#include "_bodo_file_reader.h"
#include "_fs_io.h"
#include "arrow/filesystem/s3fs.h"
#include "arrow/io/hdfs.h"
#include "arrow/result.h"
#include "mpi.h"

// decimal_mpi_type declared in _distributed.h as extern has
// to be defined in each extension module that includes it
MPI_Datatype decimal_mpi_type = MPI_DATATYPE_NULL;

FileReader* f_reader = nullptr;
;                           // File reader used by S3 / Hdfs
PyObject* f_mod = nullptr;  // imported python module:
                            // bodo.io.s3_reader, or bodo.io.hdfs_reader

#define CHECK_ARROW(expr, msg)                                      \
    if (!(expr.ok())) {                                             \
        std::cerr << "Error in arrow write: " << msg << " " << expr \
                  << std::endl;                                     \
    }

extern "C" {

uint64_t get_file_size(char* file_name);
void file_read(char* file_name, void* buff, int64_t size);
void file_write(char* file_name, void* buff, int64_t size);
void file_read_parallel(char* file_name, char* buff, int64_t start,
                        int64_t count);
void file_write_parallel(char* file_name, char* buff, int64_t start,
                         int64_t count, int64_t elem_size);

#define ROOT 0
#define LARGE_DTYPE_SIZE 1024

PyMODINIT_FUNC PyInit_hio(void) {
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "hio", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    // numpy read
    PyObject_SetAttrString(m, "get_file_size",
                           PyLong_FromVoidPtr((void*)(&get_file_size)));
    PyObject_SetAttrString(m, "file_read",
                           PyLong_FromVoidPtr((void*)(&file_read)));
    PyObject_SetAttrString(m, "file_write",
                           PyLong_FromVoidPtr((void*)(&file_write)));
    PyObject_SetAttrString(m, "file_read_parallel",
                           PyLong_FromVoidPtr((void*)(&file_read_parallel)));
    PyObject_SetAttrString(m, "file_write_parallel",
                           PyLong_FromVoidPtr((void*)(&file_write_parallel)));

    return m;
}

uint64_t get_file_size(char* file_name) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    uint64_t f_size = 0;
    PyObject* func_obj = nullptr;

    if (strncmp("s3://", file_name, 5) == 0) {
        // load s3_reader module, init s3_reader, then get size
        import_fs_module(Bodo_Fs::s3, "", f_mod);
        get_fs_reader_pyobject(Bodo_Fs::s3, "", f_mod, func_obj);

        s3_reader_init_t func =
            (s3_reader_init_t)PyNumber_AsSsize_t(func_obj, NULL);

        f_reader = func(file_name + 5);
        f_size = f_reader->getSize();

        Py_DECREF(func_obj);
    } else if (strncmp("hdfs://", file_name, 7) == 0) {
        // load hdfs_reader module, init hdfs_reader, then get size
        import_fs_module(Bodo_Fs::hdfs, "", f_mod);
        get_fs_reader_pyobject(Bodo_Fs::hdfs, "", f_mod, func_obj);

        hdfs_reader_init_t func =
            (hdfs_reader_init_t)PyNumber_AsSsize_t(func_obj, NULL);

        f_reader = func(file_name);
        f_size = f_reader->getSize();

        Py_DECREF(func_obj);
    } else {
        // posix
        if (rank == ROOT) {
            boost::filesystem::path f_path(file_name);
            // TODO: throw FileNotFoundError
            if (!boost::filesystem::exists(f_path)) {
                std::cerr << "No such file or directory: " << file_name << '\n';
                Bodo_PyErr_SetString(
                    PyExc_RuntimeError,
                    ("No such file or directory" + std::string(file_name))
                        .c_str());
                return 0;
            }
            f_size = (uint64_t)boost::filesystem::file_size(f_path);
        }
    }
    MPI_Bcast(&f_size, 1, MPI_UNSIGNED_LONG_LONG, ROOT, MPI_COMM_WORLD);
    return f_size;
}

void file_read(char* file_name, void* buff, int64_t size) {
    if (strncmp("s3://", file_name, 5) == 0 ||
        strncmp("hdfs://", file_name, 7) == 0) {
        f_reader->seek(0);
        f_reader->read((char*)buff, size);
        delete f_reader;
        f_reader = nullptr;
    } else {
        // posix
        FILE* fp = fopen(file_name, "rb");
        if (fp == NULL) return;
        size_t ret_code = fread(buff, 1, (size_t)size, fp);
        if (ret_code != (size_t)size) {
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                ("File read error: " + std::string(file_name)).c_str());
        }
        fclose(fp);
    }
    return;
}

void file_write(char* file_name, void* buff, int64_t size) {
    std::shared_ptr<::arrow::io::OutputStream> out_stream;
    PyObject* func_obj = nullptr;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (strncmp("s3://", file_name, 5) == 0) {
        std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
        std::string fname(file_name + 5);

        import_fs_module(Bodo_Fs::s3, "", f_mod);
        get_get_fs_pyobject(Bodo_Fs::s3, "", f_mod, func_obj);

        s3_get_fs_t s3_get_fs = (s3_get_fs_t)PyNumber_AsSsize_t(func_obj, NULL);
        s3_get_fs(&s3_fs);

        open_file_outstream(Bodo_Fs::s3, "", file_name + 5, s3_fs, NULL,
                            &out_stream);

        CHECK_ARROW(out_stream->Write(buff, size),
                    "arrow::io::OutputStream::Write");
        CHECK_ARROW(out_stream->Close(), "arrow::io::S3OutputStream::Close");

        Py_DECREF(f_mod);
        Py_DECREF(func_obj);
    } else if (strncmp("hdfs://", file_name, 7) == 0) {
        std::shared_ptr<::arrow::io::HadoopFileSystem> hdfs_fs;
        std::string orig_path(file_name);
        std::string fname;  // excluding hdfs:// prefix
        arrow::Status status;

        import_fs_module(Bodo_Fs::hdfs, "", f_mod);
        get_get_fs_pyobject(Bodo_Fs::hdfs, "", f_mod, func_obj);

        hdfs_get_fs_t hdfs_get_fs =
            (hdfs_get_fs_t)PyNumber_AsSsize_t(func_obj, NULL);
        hdfs_get_fs(file_name, &hdfs_fs);

        arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempRes =
            ::arrow::fs::FileSystemFromUri(orig_path, &fname);

        open_file_outstream(Bodo_Fs::hdfs, "", fname, NULL, hdfs_fs, &out_stream);

        CHECK_ARROW(out_stream->Write(buff, size),
                    "arrow::io::HdfsOutputStream::Write");
        CHECK_ARROW(out_stream->Close(), "arrow::io::HdfsOutputStream::Close");

        Py_DECREF(f_mod);
        Py_DECREF(func_obj);
    } else {
        // posix
        FILE* fp = fopen(file_name, "wb");
        if (fp == NULL) return;
        size_t ret_code = fwrite(buff, 1, (size_t)size, fp);
        if (ret_code != (size_t)size) {
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                ("File write error: " + std::string(file_name)).c_str());
        }
        fclose(fp);
    }
    return;
}

void file_read_parallel(char* file_name, char* buff, int64_t start,
                        int64_t count) {
    // printf("MPI READ %lld %lld\n", start, count);
    if (strncmp("s3://", file_name, 5) == 0 ||
        strncmp("hdfs://", file_name, 7) == 0) {
        // seek to start position, then read
        f_reader->seek(start);
        f_reader->read((char*)buff, count);
        delete f_reader;
    } else {
        // posix
        char err_string[MPI_MAX_ERROR_STRING];
        err_string[MPI_MAX_ERROR_STRING - 1] = '\0';
        MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        int err_len, err_class;

        MPI_File fh;
        int ierr = MPI_File_open(MPI_COMM_WORLD, (const char*)file_name,
                                 MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        if (ierr != 0)
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                ("File open error: " + std::string(file_name)).c_str());
        // work around MPI count limit by using a large dtype
        if (count >= (int64_t)INT_MAX) {
            MPI_Datatype large_dtype;
            MPI_Type_contiguous(LARGE_DTYPE_SIZE, MPI_CHAR, &large_dtype);
            MPI_Type_commit(&large_dtype);
            int read_size = (int)(count / LARGE_DTYPE_SIZE);

            ierr = MPI_File_read_at_all(fh, (MPI_Offset)start, buff, read_size,
                                        large_dtype, MPI_STATUS_IGNORE);
            if (ierr != 0) {
                MPI_Error_class(ierr, &err_class);
                MPI_Error_string(ierr, err_string, &err_len);
                printf("Error %s\n", err_string);
                fflush(stdout);
                Bodo_PyErr_SetString(PyExc_RuntimeError,
                                     ("File large read error: " +
                                      std::to_string(err_class) + file_name)
                                         .c_str());
            }
            MPI_Type_free(&large_dtype);
            int64_t left_over = count % LARGE_DTYPE_SIZE;
            int64_t read_byte_size = count - left_over;
            // printf("VAL leftover %lld read %lld\n", left_over,
            // read_byte_size);
            start += read_byte_size;
            buff += read_byte_size;
            count = left_over;
        }
        // printf("MPI leftover READ %lld %lld\n", start, count);

        ierr = MPI_File_read_at_all(fh, (MPI_Offset)start, buff, (int)count,
                                    MPI_CHAR, MPI_STATUS_IGNORE);

        // if (ierr!=0) std::cerr << "File read error: " << file_name << '\n';
        if (ierr != 0) {
            MPI_Error_class(ierr, &err_class);
            MPI_Error_string(ierr, err_string, &err_len);
            printf("Error %s\n", err_string);
            fflush(stdout);
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                ("File read error: " + std::to_string(err_class) + file_name)
                    .c_str());
        }
        MPI_File_close(&fh);
    }
    return;
}

void file_write_parallel(char* file_name, char* buff, int64_t start,
                         int64_t count, int64_t elem_size) {
    // std::cout << "file_write_parallel: " << file_name << "\n";
    // printf(" MPI WRITE %lld %lld %lld\n", start, count, elem_size);
    PyObject* func_obj = nullptr;

    if (strncmp("s3://", file_name, 5) == 0) {
        std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
        std::string fname(file_name + 5);
        // load s3_reader module, s3_get_fs, then write
        import_fs_module(Bodo_Fs::s3, "", f_mod);
        get_get_fs_pyobject(Bodo_Fs::s3, "", f_mod, func_obj);
        s3_get_fs_t s3_get_fs = (s3_get_fs_t)PyNumber_AsSsize_t(func_obj, NULL);
        s3_get_fs(&s3_fs);
        parallel_in_order_write(Bodo_Fs::s3, "", fname, buff, count, elem_size,
                                s3_fs, NULL);
        Py_DECREF(f_mod);
        Py_DECREF(func_obj);
    } else if (strncmp("hdfs://", file_name, 7) == 0) {
        std::shared_ptr<::arrow::io::HadoopFileSystem> hdfs_fs;
        std::string orig_path(file_name);
        std::string fname;  // excluding hdfs:// prefix
        arrow::Status status;
        // load hdfs_reader module, hdfs_get_fs, then write
        import_fs_module(Bodo_Fs::hdfs, "", f_mod);
        get_get_fs_pyobject(Bodo_Fs::hdfs, "", f_mod, func_obj);
        hdfs_get_fs_t hdfs_get_fs =
            (hdfs_get_fs_t)PyNumber_AsSsize_t(func_obj, NULL);
        hdfs_get_fs(file_name, &hdfs_fs);
        arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempRes =
            ::arrow::fs::FileSystemFromUri(orig_path, &fname);
        parallel_in_order_write(Bodo_Fs::hdfs, "", fname, buff, count,
                                elem_size, NULL, hdfs_fs);
        Py_DECREF(f_mod);
        Py_DECREF(func_obj);
    } else {
        // posix
        char err_string[MPI_MAX_ERROR_STRING];
        err_string[MPI_MAX_ERROR_STRING - 1] = '\0';
        int err_len, err_class;
        MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

        MPI_File fh;
        int ierr = MPI_File_open(MPI_COMM_WORLD, (const char*)file_name,
                                 MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                 MPI_INFO_NULL, &fh);
        if (ierr != 0)
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                ("File open error: " + std::string(file_name)).c_str());

        // work around MPI count limit by using a large dtype
        if (count >= (int64_t)INT_MAX) {
            MPI_Datatype large_dtype;
            int64_t eProd = LARGE_DTYPE_SIZE * elem_size;
            MPI_Type_contiguous(eProd, MPI_CHAR, &large_dtype);
            MPI_Type_commit(&large_dtype);
            int read_size = (int)(count / LARGE_DTYPE_SIZE);

            ierr = MPI_File_write_at_all(fh, (MPI_Offset)(start * elem_size),
                                         buff, read_size, large_dtype,
                                         MPI_STATUS_IGNORE);
            if (ierr != 0) {
                MPI_Error_class(ierr, &err_class);
                MPI_Error_string(ierr, err_string, &err_len);
                printf("Error %s\n", err_string);
                fflush(stdout);
                Bodo_PyErr_SetString(PyExc_RuntimeError,
                                     ("File large write error: " +
                                      std::to_string(err_class) + file_name)
                                         .c_str());
            }
            MPI_Type_free(&large_dtype);
            int64_t left_over = count % LARGE_DTYPE_SIZE;
            int64_t read_byte_size = count - left_over;
            // printf("VAL leftover %lld read %lld\n", left_over,
            // read_byte_size);
            start += read_byte_size;
            buff += read_byte_size * elem_size;
            count = left_over;
        }

        MPI_Datatype elem_dtype;
        MPI_Type_contiguous(elem_size, MPI_CHAR, &elem_dtype);
        MPI_Type_commit(&elem_dtype);

        ierr = MPI_File_write_at_all(fh, (MPI_Offset)(start * elem_size), buff,
                                     (int)count, elem_dtype, MPI_STATUS_IGNORE);

        MPI_Type_free(&elem_dtype);
        // if (ierr!=0) std::cerr << "File write error: " << file_name << '\n';
        if (ierr != 0) {
            MPI_Error_class(ierr, &err_class);
            MPI_Error_string(ierr, err_string, &err_len);
            printf("Error %s\n", err_string);
            fflush(stdout);
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                ("File write error: " + std::to_string(err_class) + file_name)
                    .c_str());
        }

        MPI_File_close(&fh);
    }
    return;
}

}  // extern "C"
