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
 * Write output of pandas to_csv/to_json string to a csv/json file
 * steps:
 *  1. extract file system to write to, and path & filename information
 *  2. posix: write directly with file_write/file_write_parallel, return
 *            with json distributed write: if is_records_lines=True,
 *            append '\n' to the end of buff, because string from pd.to_json
 *            does not end with '\n'. If is_records_lines=True,
 *            open out stream to write to.
 *     s3 & hdfs: open out stream to write to
 *  3. write to the output stream
 *  4. close the output stream
 * @param _path_name: file/directory name to write to
 * @param buff: buffer to be written
 * @param start: offset to write to(for MPI use only)
 * @param count: number of bytes to write
 * @param is_parallel: true if df, from pandas to_csv(df), is distributed
 * @param suffix: ".csv"/".json"
 * @param is_records_lines: true if df.to_json(orient="records", lines=True), or df.to_csv()
 *                          false for all other "orient" and "lines" combinations
 */
void write_buff(char *_path_name, char *buff, int64_t start, int64_t count,
                bool is_parallel, const std::string &suffix,
                bool is_records_lines) {
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

    extract_fs_dir_path(_path_name, is_parallel, suffix, myrank, num_ranks,
                        &fs_option, &dirname, &fname, &orig_path, &path_name);
    // handling posix with mpi/fwrite
    // csv is always written into a single file
    // json is only written to a single file
    // when pd.to_json(orient='records', lines=True)
    if (fs_option == Bodo_Fs::posix && is_records_lines) {
        if (is_parallel) {
            if (suffix == ".json") {
                std::string buffer(buff);
                buffer.append("\n");
                file_write_parallel(_path_name, &buffer[0], start + myrank,
                                    count + 1, 1);
            } else {
                file_write_parallel(_path_name, buff, start, count, 1);
            }
        } else {
            file_write(_path_name, buff, count);
        }
        return;
    }

    // handling s3 and hdfs with arrow
    // & handling posix json directory outputs with boost
    open_outstream(fs_option, is_parallel, myrank, suffix.substr(1), dirname,
                   fname, orig_path, path_name, &out_stream);

    status = out_stream->Write(buff, count);
    CHECK_ARROW(status, "arrow::io::OutputStream::Write");

    // writing an extra '\n' to the end of json files inside of directory
    // because pandas.to_json(orient='records', lines=True) does not
    // end the file with '\n' which causes incorrect format when we read the
    // directory. (btw: spark outputs ends with '\n' always)
    if (suffix == ".json" && is_records_lines && buff[count - 1] != '\n') {
        status = out_stream->Write("\n", 1);
        CHECK_ARROW(status, "arrow::io::OutputStream::Write");
    }

    status = out_stream->Close();
    CHECK_ARROW(status, "arrow::io::OutputStream::Close");
}

/*
 * Write output of pandas to_csv string to a csv file
 * @param _path_name: file/directory name to write to
 * @param buff: buffer to be written
 * @param start: offset to write to(for MPI use only)
 * @param count: number of bytes to write
 * @param is_parallel: true if df, from pandas to_csv(df), is distributed
 */
void csv_write(char *_path_name, char *buff, int64_t start, int64_t count,
               bool is_parallel) {
    write_buff(_path_name, buff, start, count, is_parallel, ".csv", true);
}

/*
 * Write output of pandas to_json string to a json file
 * @param _path_name: file/directory name to write to
 * @param buff: buffer to be written
 * @param start: offset to write to(for MPI use only)
 * @param count: number of bytes to write
 * @param is_parallel: true if df, from pandas to_json(df), is distributed
 * @param is_record_lines: true if pd.to_json(oriend = 'records', lines=True)
 */
void json_write(char *_path_name, char *buff, int64_t start, int64_t count,
                bool is_parallel, bool is_records_lines) {
    write_buff(_path_name, buff, start, count, is_parallel, ".json",
               is_records_lines);
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

PyMODINIT_FUNC PyInit_json_cpp(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "json_cpp", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    PyObject_SetAttrString(m, "json_write",
                           PyLong_FromVoidPtr((void *)(&json_write)));

    PyInit_json(m);
    return m;
}

}  // extern "C"
