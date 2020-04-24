// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <iostream>

#include <arrow/io/api.h>
#include "_csv_json_reader.h"
#include "_fs_io.h"
#include "_io.h"
#include "mpi.h"

extern "C" {

#define CHECK_ARROW(expr, msg)                                         \
    if (!(expr.ok())) {                                                \
        std::cerr << "Error in arrow csv write " << msg << " " << expr \
                  << std::endl;                                        \
        return;                                                        \
    }

/*
 * Write output of pandas to_csv string to a csv file
 * steps:
 *  1. extract file system to write to, and path & filename information
 *  2. posix: write directly with file_write/file_write_parallel, return
 *     s3 & hdfs: open out stream to write to
 *  3. write to the output stream
 *  4. close the output stream
 * @param _path_name: file/directory name to write to
 * @param buff: buffer to be written
 * @param start: offset to write to(for MPI use only)
 * @param count: number of bytes to write
 * @param is_parallel: true if df, from pandas to_csv(df), is distributed
 */
void csv_write(char *_path_name, char *buff, int64_t start, int64_t count,
               bool is_parallel) {
    int myrank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    std::string orig_path(_path_name);  // original path passed to this function
    std::string
        path_name;  // original path passed to this function (excluding prefix)
    std::string dirname;  // path and directory name to store the parquet files
                          // (only if is_parallel=true)
    std::string fname;    // name of parquet file to write (excludes path)
    std::shared_ptr<::arrow::io::OutputStream> out_stream;
    Bodo_Fs::FsEnum fs_option;
    arrow::Status status;

    extract_fs_dir_path(_path_name, is_parallel, ".csv", myrank, num_ranks,
                        &fs_option, &dirname, &fname, &orig_path, &path_name);

    // handling posix with mpi/fwrite
    if (fs_option == Bodo_Fs::posix) {
        if (is_parallel) {
            file_write_parallel(_path_name, buff, start, count, 1);
        } else {
            file_write(_path_name, buff, count);
        }
        return;
    }

    // handling s3 and hdfs with arrow
    open_outstream(fs_option, is_parallel, myrank, "csv", dirname, fname,
                   orig_path, path_name, &out_stream);

    status = out_stream->Write(buff, count);
    CHECK_ARROW(status, "arrow::io::OutputStream::Write");

    status = out_stream->Close();
    CHECK_ARROW(status, "arrow::io::OutputStream::Close");
}

PyMODINIT_FUNC PyInit_csv_cpp(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "csv_cpp", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    PyObject_SetAttrString(m, "csv_write",
                           PyLong_FromVoidPtr((void *)(&csv_write)));

    PyInit_csv(m);
    return m;
}

}  // extern "C"
