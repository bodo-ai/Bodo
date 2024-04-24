// Copyright (C) 2019 Bodo Inc. All rights reserved.
#pragma once

#include <Python.h>

#include "../libs/_bodo_common.h"

// CSV exports some stuff to the io module
extern "C" void PyInit_csv(PyObject *);

// JSON exports some stuff to the io module
extern "C" void PyInit_json(PyObject *);

/**
 * Split file into chunks and return a file-like object per rank. The returned
 *object
 * represents the data to be read on each process.
 *
 * @param[in]  f   the input file name
 * @param[in]  is_parallel   if parallel read of different chunks required
 * @param[in]  skiprows   number of rows to skip at the beginning
 * @param[in]  nrows   numebr of rows to read
 * @param[in]  header   whether csv file(s) contain header(s)
 * @param[in]  compression   compression scheme of file(s)
 * @param[in] bucket_region: s3 bucket server location (needed if data is in s3
 * path)
 * @param[in] storage_options: options to pass to S3FS.
 *            Example: anonymous connection `anon`.
 * @param[in] chunksize: number of rows per chunk
 * @param[in] is_skiprows_list: whether skiprows is a single number (i.e. skip
 *nrows from beginning) or list of specific row numbers. See file_chunk_reader
 *doxygen in _csv_json_readear.cpp
 * @param[in] skiprows_list_len: number of rows to skip
 * @param[in] pd_low_memory: flag whether user explicitly requested low_memory
 * mode.
 * @return     HPATIO file-like object to read the owned chunk through
 *pandas.read_csv
 **/
extern "C" PyObject *csv_file_chunk_reader(
    const char *fname, bool is_parallel, int64_t *skiprows, int64_t nrows,
    bool header, const char *compression, const char *bucket_region,
    PyObject *storage_options, int64_t chunksize = 0,
    bool is_skiprows_list = false, int64_t skiprows_list_len = 0,
    bool pd_low_memory = false);

/**
 * Split file into chunks and return a file-like object per rank. The returned
 *object
 * represents the data to be read on each process.
 *
 * @param[in]  fname   the input file name
 * @param[in]  lines   pd.read_json(lines) when lines = true,
                       each line is a single record, and we read the json object
 per line.
 * @param[in]  is_parallel   if parallel read of different chunks required
 * @param[in]  nrows   number of rows to read
 * @param[in]  compression   compression scheme of file(s)
 * @return     HPATIO file-like object to read the owned chunk through
 *pandas.read_json
 **/
extern "C" PyObject *json_file_chunk_reader(const char *fname, bool lines,
                                            bool is_parallel, int64_t nrows,
                                            const char *compression,
                                            const char *bucket_region,
                                            PyObject *storage_options);

/**
 * Update the reader being used in an iterator to
 * prepare for the next pd.read_csv call.
 * If there's more data to read, returns true and updates reader with new data.
 * Otherwise returns false to trigger a stop iteration
 *
 * @param[in, out]  reader  the CSVReader containing the file info.
 * @return   If there is more data data to read. false triggers
 * a stop iteration.
 */
extern "C" bool update_csv_reader(PyObject *reader);

/**
 * Mark a CSVReader as unread
 *
 * @param[in]  reader  the CSVReader containing the file info.
 */
extern "C" void initialize_csv_reader(PyObject *reader);

/**
 * Split string into chunks and return a file-like object per rank. The returned
 *object
 * represents the data to be read on each process.
 *
 * @param[in]  f   the input string
 * @param[in]  is_parallel   if parallel read of different chunks required
 * @return     HPATIO file-like object to read the owned chunk through
 *pandas.read_csv
 **/
// extern "C" PyObject* csv_string_chunk_reader(const std::string * str, bool
// is_parallel);
