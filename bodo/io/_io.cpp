#include <Python.h>
#include <climits>
#include <cstdio>
#include <filesystem>
#include <string>

#include <arrow/filesystem/s3fs.h>
#include <arrow/io/hdfs.h>
#include <arrow/result.h>
#include <mpi.h>

#include "../libs/_bodo_common.h"
#include "../libs/_distributed.h"
#include "_bodo_file_reader.h"
#include "_fs_io.h"

FileReader* f_reader = nullptr;  // File reader used by S3 / Hdfs
PyObject* f_mod = nullptr;       // imported python module:
                                 // bodo.io.s3_reader, or bodo.io.hdfs_reader

#define CHECK_ARROW(expr, msg)                                                 \
    if (!(expr.ok())) {                                                        \
        std::string err_msg = std::string("_io.cpp: Error in arrow write: ") + \
                              msg + " " + expr.ToString();                     \
        throw std::runtime_error(err_msg);                                     \
    }

extern "C" {

uint64_t get_file_size(const char* file_name);
void file_read(const char* file_name, void* buff, int64_t size, int64_t offset);
void file_write(const char* file_name, void* buff, int64_t size);
void file_write_py_entrypt(const char* file_name, void* buff, int64_t size);
void file_read_parallel(const char* file_name, char* buff, int64_t start,
                        int64_t count);
void file_write_parallel(const char* file_name, char* buff, int64_t start,
                         int64_t count, int64_t elem_size);
void file_write_parallel_py_entrypt(const char* file_name, char* buff,
                                    int64_t start, int64_t count,
                                    int64_t elem_size);

#define ROOT 0
#define LARGE_DTYPE_SIZE 1024

PyMODINIT_FUNC PyInit_hio(void) {
    PyObject* m;
    MOD_DEF(m, "hio", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    // numpy read
    SetAttrStringFromVoidPtr(m, get_file_size);
    SetAttrStringFromVoidPtr(m, file_read);
    SetAttrStringFromVoidPtr(m, file_write_py_entrypt);
    SetAttrStringFromVoidPtr(m, file_read_parallel);
    SetAttrStringFromVoidPtr(m, file_write_parallel_py_entrypt);

    return m;
}

uint64_t get_file_size(const char* file_name) {
    try {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        uint64_t f_size = 0;
        PyObject* func_obj = nullptr;

        if (strncmp("s3://", file_name, 5) == 0) {
            // load s3_reader module, init s3_reader, then get size
            import_fs_module(Bodo_Fs::s3, "", f_mod);
            get_fs_reader_pyobject(Bodo_Fs::s3, "", f_mod, func_obj);

            s3_reader_init_t func =
                (s3_reader_init_t)PyNumber_AsSsize_t(func_obj, nullptr);

            f_reader = func(file_name + 5, "", false, true);
            f_size = f_reader->getSize();

            Py_DECREF(f_mod);
            Py_DECREF(func_obj);
        } else if (strncmp("hdfs://", file_name, 7) == 0) {
            // load hdfs_reader module, init hdfs_reader, then get size
            import_fs_module(Bodo_Fs::hdfs, "", f_mod);
            get_fs_reader_pyobject(Bodo_Fs::hdfs, "", f_mod, func_obj);

            hdfs_reader_init_t func =
                (hdfs_reader_init_t)PyNumber_AsSsize_t(func_obj, nullptr);

            f_reader = func(file_name, "", false, true);
            f_size = f_reader->getSize();

            Py_DECREF(f_mod);
            Py_DECREF(func_obj);
        } else {
            // posix
            bool throw_error = false;
            if (rank == ROOT) {
                std::filesystem::path f_path(file_name);
                // TODO: throw FileNotFoundError
                if (!std::filesystem::exists(f_path)) {
                    throw_error = true;
                }
                if (!throw_error) {
                    f_size = (uint64_t)std::filesystem::file_size(f_path);
                }
            }
            // Synchronize throw_error
            CHECK_MPI(MPI_Bcast(&throw_error, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD),
                      "get_file_size: MPI error on MPI_Bcast:");
            if (throw_error) {
                throw std::runtime_error(
                    "_io.cpp::get_file_size: No such file or directory " +
                    std::string(file_name));
            }
        }
        CHECK_MPI(
            MPI_Bcast(&f_size, 1, MPI_UNSIGNED_LONG_LONG, ROOT, MPI_COMM_WORLD),
            "get_file_size: MPI error on MPI_Bcast:");
        return f_size;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    }
}

void file_read(const char* file_name, void* buff, int64_t size,
               int64_t offset) {
    try {
        if (strncmp("s3://", file_name, 5) == 0 ||
            strncmp("hdfs://", file_name, 7) == 0) {
            // Assumes that the default offset when not given
            // will be 0.
            f_reader->seek(offset);
            f_reader->read((char*)buff, size);
            delete f_reader;
            f_reader = nullptr;
        } else {
            // posix
            FILE* fp = fopen(file_name, "rb");
            if (fp == nullptr) {
                return;
            }
            int64_t seek_res = fseek(fp, offset, SEEK_SET);
            if (seek_res != 0) {
                return;
            }
            size_t ret_code = fread(buff, 1, (size_t)size, fp);
            fclose(fp);
            if (ret_code != (size_t)size) {
                throw std::runtime_error(
                    "_io.cpp::file_read: File read error: " +
                    std::string(file_name));
            }
        }
        return;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

void file_write(const char* file_name, void* buff, int64_t size) {
    std::shared_ptr<::arrow::io::OutputStream> out_stream;
    PyObject* func_obj = nullptr;

    if (strncmp("s3://", file_name, 5) == 0) {
        std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
        std::string fname(file_name + 5);

        import_fs_module(Bodo_Fs::s3, "", f_mod);
        get_get_fs_pyobject(Bodo_Fs::s3, "", f_mod, func_obj);

        s3_get_fs_t s3_get_fs =
            (s3_get_fs_t)PyNumber_AsSsize_t(func_obj, nullptr);
        s3_get_fs(&s3_fs, "", false);

        open_file_outstream(Bodo_Fs::s3, "", file_name + 5, s3_fs, nullptr,
                            &out_stream);

        CHECK_ARROW(out_stream->Write(buff, size),
                    "arrow::io::OutputStream::Write");
        CHECK_ARROW(out_stream->Close(), "arrow::io::S3OutputStream::Close");

        Py_DECREF(f_mod);
        Py_DECREF(func_obj);
    } else if (strncmp("hdfs://", file_name, 7) == 0) {
        std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs;
        std::string orig_path(file_name);
        std::string fname;  // excluding hdfs:// prefix
        arrow::Status status;

        import_fs_module(Bodo_Fs::hdfs, "", f_mod);
        get_get_fs_pyobject(Bodo_Fs::hdfs, "", f_mod, func_obj);

        hdfs_get_fs_t hdfs_get_fs =
            (hdfs_get_fs_t)PyNumber_AsSsize_t(func_obj, nullptr);
        hdfs_get_fs(file_name, &hdfs_fs);

        arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempRes =
            ::arrow::fs::FileSystemFromUri(orig_path, &fname);

        open_file_outstream(Bodo_Fs::hdfs, "", fname, nullptr, hdfs_fs,
                            &out_stream);

        CHECK_ARROW(out_stream->Write(buff, size),
                    "arrow::io::HdfsOutputStream::Write");
        CHECK_ARROW(out_stream->Close(), "arrow::io::HdfsOutputStream::Close");

        Py_DECREF(f_mod);
        Py_DECREF(func_obj);
        // TODO gcs
    } else {
        // posix
        FILE* fp = fopen(file_name, "wb");
        if (fp == nullptr) {
            return;
        }
        size_t ret_code = fwrite(buff, 1, (size_t)size, fp);
        fclose(fp);
        if (ret_code != (size_t)size) {
            throw std::runtime_error("_io.cpp::file_write: File write error: " +
                                     std::string(file_name));
        }
    }
}

void file_write_py_entrypt(const char* file_name, void* buff, int64_t size) {
    try {
        file_write(file_name, buff, size);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

void file_read_parallel(const char* file_name, char* buff, int64_t start,
                        int64_t count) {
    try {
        // printf("MPI READ %lld %lld\n", start, count);
        if (strncmp("s3://", file_name, 5) == 0 ||
            strncmp("hdfs://", file_name, 7) == 0) {
            // seek to start position, then read
            f_reader->seek(start);
            f_reader->read((char*)buff, count);
            delete f_reader;
        } else {
            // posix
            CHECK_MPI(
                MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
                "_io.cpp::file_read_parallel: MPI error on "
                "MPI_Comm_set_errhandler:");

            MPI_File fh;
            CHECK_MPI(
                MPI_File_open(MPI_COMM_WORLD, (const char*)file_name,
                              MPI_MODE_RDONLY, MPI_INFO_NULL, &fh),
                "_io.cpp::file_read_parallel: MPI error on MPI_File_open:");
            // work around MPI count limit by using a large dtype
            if (count >= (int64_t)INT_MAX) {
                MPI_Datatype large_dtype;
                CHECK_MPI(MPI_Type_contiguous(LARGE_DTYPE_SIZE, MPI_CHAR,
                                              &large_dtype),
                          "_io.cpp::file_read_parallel: MPI error on "
                          "MPI_Type_contiguous:");
                CHECK_MPI(MPI_Type_commit(&large_dtype),
                          "_io.cpp::file_read_parallel: MPI error on "
                          "MPI_Type_commit:");
                int read_size = (int)(count / LARGE_DTYPE_SIZE);

                CHECK_MPI(
                    MPI_File_read_at_all(fh, (MPI_Offset)start, buff, read_size,
                                         large_dtype, MPI_STATUS_IGNORE),
                    "_io.cpp::file_read_parallel: MPI error on "
                    "MPI_File_read_at_all:");
                CHECK_MPI(
                    MPI_Type_free(&large_dtype),
                    "_io.cpp::file_read_parallel: MPI error on MPI_Type_free:");
                int64_t left_over = count % LARGE_DTYPE_SIZE;
                int64_t read_byte_size = count - left_over;
                // printf("VAL leftover %lld read %lld\n", left_over,
                // read_byte_size);
                start += read_byte_size;
                buff += read_byte_size;
                count = left_over;
            }
            // printf("MPI leftover READ %lld %lld\n", start, count);

            CHECK_MPI(
                MPI_File_read_at_all(fh, (MPI_Offset)start, buff, (int)count,
                                     MPI_CHAR, MPI_STATUS_IGNORE),
                "_io.cpp::file_read_parallel: MPI error on "
                "MPI_File_read_at_all:");
            CHECK_MPI(
                MPI_File_close(&fh),
                "_io.cpp::file_read_parallel: MPI error on MPI_File_close:");
        }
        return;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

void file_write_parallel(const char* file_name, char* buff, int64_t start,
                         int64_t count, int64_t elem_size) {
    // std::cout << "file_write_parallel: " << file_name << "\n";
    // printf(" MPI WRITE %lld %lld %lld\n", start, count, elem_size);
    PyObject* func_obj = nullptr;

    if (strncmp("s3://", file_name, 5) == 0) {
        std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
        std::string fname(file_name + 5);

        // load bodo module, s3_get_fs, then write
        import_fs_module(Bodo_Fs::s3, "", f_mod);
        get_get_fs_pyobject(Bodo_Fs::s3, "", f_mod, func_obj);
        s3_get_fs_t s3_get_fs =
            (s3_get_fs_t)PyNumber_AsSsize_t(func_obj, nullptr);
        s3_get_fs(&s3_fs, "", false);
        parallel_in_order_write(Bodo_Fs::s3, "", fname, buff, count, elem_size,
                                s3_fs, nullptr);
        Py_DECREF(f_mod);
        Py_DECREF(func_obj);
    } else if (strncmp("hdfs://", file_name, 7) == 0) {
        std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs;
        std::string orig_path(file_name);
        std::string fname;  // excluding hdfs:// prefix
        arrow::Status status;

        // load bodo module, hdfs_get_fs, then write
        import_fs_module(Bodo_Fs::hdfs, "", f_mod);
        get_get_fs_pyobject(Bodo_Fs::hdfs, "", f_mod, func_obj);
        hdfs_get_fs_t hdfs_get_fs =
            (hdfs_get_fs_t)PyNumber_AsSsize_t(func_obj, nullptr);
        hdfs_get_fs(file_name, &hdfs_fs);
        arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> tempRes =
            ::arrow::fs::FileSystemFromUri(orig_path, &fname);
        parallel_in_order_write(Bodo_Fs::hdfs, "", fname, buff, count,
                                elem_size, nullptr, hdfs_fs);
        Py_DECREF(f_mod);
        Py_DECREF(func_obj);
        // TODO gcs
    } else {
        // posix
        char err_string[MPI_MAX_ERROR_STRING];
        err_string[MPI_MAX_ERROR_STRING - 1] = '\0';
        int err_len, err_class;
        CHECK_MPI(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
                  "_io.cpp::file_write_parallel: MPI error on "
                  "MPI_Comm_set_errhandler:");

        int ierr;
        bool throw_error = false;
        if (dist_get_rank() == 0) {
            ierr = MPI_File_delete((const char*)file_name, MPI_INFO_NULL);
            if (ierr != 0) {
                MPI_Error_class(ierr, &err_class);
                if (err_class != MPI_ERR_NO_SUCH_FILE) {
                    throw_error = true;
                }
            }
        }
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD),
                  "_io.cpp::file_write_parallel: MPI error on MPI_Barrier:");
        // Synchronize throw_error
        CHECK_MPI(MPI_Bcast(&throw_error, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD),
                  "file_write_parallel: MPI error on MPI_Bcast:");
        if (throw_error) {
            throw std::runtime_error(
                "_io.cpp::file_write_parallel: File delete error: " +
                std::string(file_name));
        }

        MPI_File fh;
        CHECK_MPI(MPI_File_open(MPI_COMM_WORLD, (const char*)file_name,
                                MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                MPI_INFO_NULL, &fh),
                  "_io.cpp::file_write_parallel: MPI error on MPI_File_open:");

        // work around MPI count limit by using a large dtype
        if (count >= (int64_t)INT_MAX) {
            MPI_Datatype large_dtype;
            int64_t eProd = LARGE_DTYPE_SIZE * elem_size;
            CHECK_MPI(MPI_Type_contiguous(eProd, MPI_CHAR, &large_dtype),
                      "_io.cpp::file_write_parallel: MPI error on "
                      "MPI_Type_contiguous:");
            CHECK_MPI(
                MPI_Type_commit(&large_dtype),
                "_io.cpp::file_write_parallel: MPI error on MPI_Type_commit:");
            int read_size = (int)(count / LARGE_DTYPE_SIZE);

            CHECK_MPI(MPI_File_write_at_all(fh, (MPI_Offset)(start * elem_size),
                                            buff, read_size, large_dtype,
                                            MPI_STATUS_IGNORE),
                      "_io.cpp::file_write_parallel: MPI error on "
                      "MPI_File_write_at_all:");
            CHECK_MPI(
                MPI_Type_free(&large_dtype),
                "_io.cpp::file_write_parallel: MPI error on MPI_Type_free:");
            int64_t left_over = count % LARGE_DTYPE_SIZE;
            int64_t read_byte_size = count - left_over;
            // printf("VAL leftover %lld read %lld\n", left_over,
            // read_byte_size);
            start += read_byte_size;
            buff += read_byte_size * elem_size;
            count = left_over;
        }

        MPI_Datatype elem_dtype;
        CHECK_MPI(
            MPI_Type_contiguous(elem_size, MPI_CHAR, &elem_dtype),
            "_io.cpp::file_write_parallel: MPI error on MPI_Type_contiguous:");
        CHECK_MPI(
            MPI_Type_commit(&elem_dtype),
            "_io.cpp::file_write_parallel: MPI error on MPI_Type_commit:");

        ierr = MPI_File_write_at_all(fh, (MPI_Offset)(start * elem_size), buff,
                                     (int)count, elem_dtype, MPI_STATUS_IGNORE);

        CHECK_MPI(MPI_Type_free(&elem_dtype),
                  "_io.cpp::file_write_parallel: MPI error on MPI_Type_free:");
        // if (ierr!=0) std::cerr << "File write error: " << file_name <<
        // '\n';
        if (ierr != 0) {
            MPI_Error_class(ierr, &err_class);
            MPI_Error_string(ierr, err_string, &err_len);
            printf("Error %s\n", err_string);
            fflush(stdout);
            throw std::runtime_error(
                "_io.cpp::file_write_parallel: File write error: " +
                std::to_string(err_class) + file_name);
        }

        CHECK_MPI(MPI_File_close(&fh),
                  "_io.cpp::file_write_parallel: MPI error on MPI_File_close:");
    }
}

void file_write_parallel_py_entrypt(const char* file_name, char* buff,
                                    int64_t start, int64_t count,
                                    int64_t elem_size) {
    try {
        file_write_parallel(file_name, buff, start, count, elem_size);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

}  // extern "C"
