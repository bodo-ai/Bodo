// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef _FS_IO_H_INCLUDED
#define _FS_IO_H_INCLUDED
#include <Python.h>
#include <arrow/io/api.h>
#include <string>
#include "_bodo_file_reader.h"
#include "arrow/filesystem/hdfs.h"
#include "arrow/filesystem/s3fs.h"

struct Bodo_Fs {
    enum FsEnum { posix = 0, s3 = 1, hdfs = 2, gcs = 3 };
};

// TODO gcs reader
typedef FileReader *(*s3_reader_init_t)(const char *, const char *, bool, bool);
typedef FileReader *(*hdfs_reader_init_t)(const char *, const char *, bool,
                                          bool);
typedef void (*s3_get_fs_t)(std::shared_ptr<::arrow::fs::S3FileSystem> *,
                            std::string, bool anon);
typedef void (*gcs_get_fs_t)(std::shared_ptr<::arrow::py::fs::PyFileSystem> *);
typedef void (*hdfs_get_fs_t)(const std::string &,
                              std::shared_ptr<::arrow::fs::HadoopFileSystem> *);

/*
 * Generate file names for files in directory: this process writes a file
 * named "prefix-000X.suffix", where X is myrank, into the directory
 * specified by 'path_name', and prefix / suffix is specified by arguments
 * @param myrank: current MPI rank
 * @param num_ranks: total MPI ranks
 * @param prefix: prefix of files in distributed case. This can be specified in
 * `to_parquet`, `to_csv`, and `to_json` using `_bodo_file_prefix`
 * @param suffix: suffix of file(s), '.csv', '.json' or '.parquet'
 */
std::string gen_pieces_file_name(int myrank, int num_ranks,
                                 const std::string &prefix,
                                 const std::string &suffix);

/*
 * Extract file system, and path information
 * @param _path_name: path of output file or directory
 * @param is_parallel: true if the table is part of a distributed table
 * @param prefix: prefix of files in distributed case. This can be specified in
 * `to_parquet`, `to_csv`, and `to_json` using `_bodo_file_prefix`
 * with a default value of 'part-'
 * @param suffix: suffix of file(s), '.csv', '.json', or '.parquet'
 * @param myrank: current MPI rank
 * @param num_ranks: total MPI ranks
 * @param fs_option: file system to write to
 * @param dirname: path and directory name to store the files
          (only if is_parallel=true)
 * @param fname: name of file to write (exclude prefix, excludes path)
 * @param orig_path: name of file to write (include prefix & path)
 * @param path_name: name of file to write (exclude prefix, include path)
 */
void extract_fs_dir_path(const char *_path_name, bool is_parallel,
                         const std::string &prefix, const std::string &suffix,
                         int myrank, int num_ranks, Bodo_Fs::FsEnum *fs,
                         std::string *dirname, std::string *fname,
                         std::string *orig_path, std::string *path_name);

/*
 * Import file system python module: bodo.io.s3_reader, bodo.io.hdfs_reader or
 * bodo.io.gcs_reader
 * @param fs_option: file system to write to
 * @param file_type: type of file, 'csv', 'json', 'parquet', or '' all others
 * @param f_mod: reference to the python module
 */
void import_fs_module(Bodo_Fs::FsEnum fs_option, const std::string &file_type,
                      PyObject *&f_mod);

/*
 * TODO init_gcs_reader
 * Get python attributes: init_s3_reader, or init_hdfs_reader
 * @param fs_option: file system to write to
 * @param file_type: type of file, 'csv', 'json', 'parquet', or '' all others
 * @param f_mod: reference to the python module
 * @param func_obj: reference to the python attribute
 */
void get_fs_reader_pyobject(Bodo_Fs::FsEnum fs_option,
                            const std::string &file_type, PyObject *f_mod,
                            PyObject *&func_obj);

/*
 * Get python attributes: s3_get_fs, hdfs_get_fs or gcs_get_fs
 * @param fs_option: file system to write to
 * @param file_type: type of file, 'csv', 'json', 'parquet', or '' all others
 * @param f_mod: reference to the python module
 * @param func_obj: reference to the python attribute
 */
void get_get_fs_pyobject(Bodo_Fs::FsEnum fs_option,
                         const std::string &file_type, PyObject *f_mod,
                         PyObject *&func_obj);

/*
 * TODO gcs
 * Open file output stream for posix & s3 & hdfs
 * @param fs_option: file system to write to
 * @param file_type: type of file, 'csv', 'json', 'parquet', or '' all others
 * @param fname: name of the file to open (exclude prefix, include path)
 * @param s3_fs: s3 file system, used when fs_option==Bodo_Fs::s3
 * @param hdfs_fs: hdfs file system, used when fs_option==Bodo_Fs::hdfs
 * @param out_stream: the OutputStream to open
 */
void open_file_outstream(
    Bodo_Fs::FsEnum fs_option, const std::string &file_type,
    const std::string &fname, std::shared_ptr<arrow::fs::S3FileSystem> s3_fs,
    std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs,
    std::shared_ptr<::arrow::io::OutputStream> *out_stream);

/*
 * TODO gcs
 * Open file append output stream for hdfs
 * @param file_type: type of file, 'csv', 'json', 'parquet', or '' all others
 * @param fname: name of the file to open (exclude prefix, include path)
 * @param hdfs_fs: hdfs file system
 * @param out_stream: the OutputStream to open
 */
void open_file_appendstream(
    const std::string &file_type, const std::string &fname,
    std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs,
    std::shared_ptr<::arrow::io::OutputStream> *out_stream);

/**
 * @brief Create a directory in a posix / hadoop filesystem.
 *
 * This function is assumed to be happening in parallel, i.e.
 * it should be called by all ranks
 * @param fs_option: file system to write to
 * @param myrank current MPI rank
 * @param dirname directory name to create
 * @param path_name path to dir to create (exclude prefix, include path)
 * @param orig_path: name of file to open (include prefix & path)
 * @param file_type: type of file, 'csv', 'json', or 'parquet'
 */
void create_dir_parallel(Bodo_Fs::FsEnum fs_option, int myrank,
                         std::string &dirname, std::string &path_name,
                         std::string &orig_path, const std::string &file_type);

/*
 * Open arrow::io::OutputStream for csv/json/parquet write
 * writing csv/json to posix:
 *  does not need to open outstream as we use MPI for writing
 * writing parquet to posix:
 *  use std::filesystem to create directory if necessary
 * @param fs_option: file system to write to
 * @param is_parallel: true if the table is part of a distributed table
 * @param file_type: type of file, 'csv', 'json', or 'parquet'
 * @param dirname: directory name to store the files
 * @param fname: name of file to open (exclude prefix, excludes path)
 * @param orig_path: name of file to open (include prefix & path)
 * @param out_stream: the OutputStream to open
 * @param bucket_region: Region of the S3 bucket in case writing to S3
 */
void open_outstream(Bodo_Fs::FsEnum fs_option, bool is_parallel,
                    const std::string &file_type, std::string &dirname,
                    std::string &fname, std::string &orig_path,
                    std::shared_ptr<::arrow::io::OutputStream> *out_stream,
                    const std::string &bucket_region);

/*
 * TODO gcs
 * Generic implementation for writing to s3 & hdfs,
 * where more than more processors write to the same file.
 * s3:
 *  rank 0 writes its own buff to outstream
 *  each non-zero ranks send buff to zero rank one at a time
 *  zero rank does all the writings, one at a time
 *  i.e. rank 1 sends buff1, rank 0 writes buff1
 *       then rank2 sends buff2, rank0 writes buff2 ...
 * hdfs:
 *  rank 0 creates/overwrites the file
 *  then rank 1 append to the file
 *  then rank 2 ...
 * @param fs_option: file system to write to
 * @param file_type: type of file, used in error messages
 * @param fname: name of the file to open (exclude prefix, include path)
 * @param buff: buffer containing data to write
 * @param count: number of elems in buff
 * @param elem_size: size of each elem in buff
 * @param s3_fs: s3 file system, used when fs_option==Bodo_Fs::s3
 * @param hdfs_fs: hdfs file system
 */
void parallel_in_order_write(
    Bodo_Fs::FsEnum fs_option, const std::string &file_type,
    const std::string &fname, char *buff, int64_t count, int64_t elem_size,
    std::shared_ptr<arrow::fs::S3FileSystem> s3_fs,
    std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs);

#endif  // _FS_IO_H_INCLUDED
