// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef _WRITER_H_INCLUDED
#define _WRITER_H_INCLUDED
#include <arrow/io/api.h>
#include <string>

struct Bodo_Fs {
    enum FsEnum { posix = 0, s3 = 1, hdfs = 2 };
};

/*
 * Generate file names for files in directory: this process writes a file 
 * named "part-000X.suffix", where X is myrank, into the directory
 * specified by 'path_name', and suffix is specified by suffix
 * @param myrank: current MPI rank
 * @param num_ranks: total MPI ranks
 * @param suffix: suffix of file(s), '.csv' or '.parquet'
 */
std::string gen_pieces_file_name(int myrank, int num_ranks,
                                 const std::string &suffix);

/*
 * Extract file system, and path information
 * @param _path_name: path of output file or directory
 * @param is_parallel: true if the table is part of a distributed table
 * @param suffix: suffix of file(s), '.csv' or '.parquet'
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
                         const std::string &suffix, int myrank, int num_ranks,
                         Bodo_Fs::FsEnum *fs, std::string *dirname,
                         std::string *fname, std::string *orig_path,
                         std::string *path_name);

/*
 * Open arrow::io::OutputStream for parquet write and csv write
 * writing csv to posix:
 *  does not need to open outstream as we use MPI for writing
 * writing parquet to posix: 
 *  use boost to create directory if necessary
 * @param fs_option: file system to write to
 * @param is_parallel: true if the table is part of a distributed table
 * @param myrank: current MPI rank
 * @param file_type: type of file, 'csv' or 'parquet'
 * @param dirname: directory name to store the files
 * @param fname: name of file to write (exclude prefix, excludes path)
 * @param orig_path: name of file to write (include prefix & path)
 * @param path_name: name of file to write (exclude prefix, include path)
 * @param out_stream: the OutputStream to open
 */
void open_outstream(Bodo_Fs::FsEnum fs_option, bool is_parallel, int myrank,
                    const std::string &file_type,
                    std::string &dirname, std::string &fname,
                    std::string &orig_path, std::string &path_name,
                    std::shared_ptr<::arrow::io::OutputStream> *out_stream);

#endif  // _WRITER_H_INCLUDED
