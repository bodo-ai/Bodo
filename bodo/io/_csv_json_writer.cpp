#include <Python.h>

#include <arrow/io/api.h>
#include <mpi.h>

#include "_fs_io.h"
#include "_io.h"

extern "C" {

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it
#define CHECK_ARROW(expr, msg)                                             \
    if (!(expr.ok())) {                                                    \
        std::string err_msg = std::string("Error in arrow write ") + msg + \
                              " " + expr.ToString();                       \
        throw std::runtime_error(err_msg);                                 \
    }

#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();

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
 * @param prefix: Prefix of file names during distributed write
 * @param suffix: ".csv"/".json"
 * @param is_records_lines: true if df.to_json(orient="records", lines=True), or
 * df.to_csv() false for all other "orient" and "lines" combinations
 */
void write_buff(char *_path_name, char *buff, int64_t start, int64_t count,
                bool is_parallel, const std::string &prefix,
                const std::string &suffix, bool is_records_lines,
                char *bucket_region) {
    try {
        int myrank, num_ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
        std::string orig_path(
            _path_name);        // original path passed to this function
        std::string path_name;  // original path passed to this function
                                // (excluding prefix)
        std::string dirname;    // path and directory name to store the parquet
                                // files (only if is_parallel=true)
        std::string fname;      // name of parquet file to write (excludes path)
        std::shared_ptr<::arrow::io::OutputStream> out_stream;
        Bodo_Fs::FsEnum fs_option;
        std::shared_ptr<arrow::fs::FileSystem> fs;
        arrow::Status status;
        extract_fs_dir_path(_path_name, is_parallel, prefix, suffix, myrank,
                            num_ranks, &fs_option, &dirname, &fname, &orig_path,
                            &path_name);
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

        // We need to create a directory when writing a distributed
        // table to a posix or hadoop filesystem.
        if (is_parallel) {
            create_dir_parallel(fs_option, myrank, dirname, path_name,
                                orig_path, suffix.substr(1), true);
        }

        fs = get_fs_for_path(_path_name, is_parallel);

        std::filesystem::path out_path(dirname);
        out_path /= fname;  // append file name to output path
        // Avoid "\" generated on Windows for remote object storage
        std::string out_path_str = fs->type_name() == "local"
                                       ? out_path.string()
                                       : out_path.generic_string();
        arrow::Result<std::shared_ptr<arrow::io::OutputStream>> result =
            fs->OpenOutputStream(out_path_str);
        CHECK_ARROW_AND_ASSIGN(result, "FileOutputStream::Open", out_stream);

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
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/*
 * Write output of pandas to_csv string to a csv file
 * @param _path_name: file/directory name to write to
 * @param buff: buffer to be written
 * @param start: offset to write to(for MPI use only)
 * @param count: number of bytes to write
 * @param is_parallel: true if df, from pandas to_csv(df), is distributed
 * @param prefix: prefix of files written in distributed case
 */
void csv_write(char *_path_name, char *buff, int64_t start, int64_t count,
               bool is_parallel, char *bucket_region, char *prefix) {
    write_buff(_path_name, buff, start, count, is_parallel, prefix, ".csv",
               true, bucket_region);
}

/*
 * Write output of pandas to_json string to a json file
 * @param _path_name: file/directory name to write to
 * @param buff: buffer to be written
 * @param start: offset to write to(for MPI use only)
 * @param count: number of bytes to write
 * @param is_parallel: true if df, from pandas to_json(df), is distributed
 * @param is_record_lines: true if pd.to_json(oriend = 'records', lines=True)
 * @param prefix: prefix of files written in distributed case
 */
void json_write(char *_path_name, char *buff, int64_t start, int64_t count,
                bool is_parallel, bool is_records_lines, char *bucket_region,
                char *prefix) {
    write_buff(_path_name, buff, start, count, is_parallel, prefix, ".json",
               is_records_lines, bucket_region);
}

/*
 * Check if the df.to_csv() output is a directory: output is a directory
 * if writing to s3/hdfs with more than 1 rank
 * @param _path_name: file/directory name to write to
 * @return if the output is a directory
 */
int8_t csv_output_is_dir(char *_path_name) {
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks > 1) {
        if (strncmp("s3://", _path_name, 5) == 0 ||
            strncmp("hdfs://", _path_name, 7) == 0) {
            return 1;
        }
    }
    return 0;
}

PyMODINIT_FUNC PyInit_csv_cpp(void) {
    PyObject *m;
    MOD_DEF(m, "csv_cpp", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    SetAttrStringFromVoidPtr(m, csv_write);
    SetAttrStringFromVoidPtr(m, csv_output_is_dir);

    return m;
}

PyMODINIT_FUNC PyInit_json_cpp(void) {
    PyObject *m;
    MOD_DEF(m, "json_cpp", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    SetAttrStringFromVoidPtr(m, json_write);

    return m;
}

}  // extern "C"
