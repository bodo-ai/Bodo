// Copyright (C) 2022 Bodo Inc. All rights reserved.

// Functions to write Bodo arrays to Iceberg table (parquet format)

#if _MSC_VER >= 1900
#undef timezone
#endif

#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>          // streaming operators etc.
#include "../libs/iceberg_transforms.h"
#include "parquet_write.h"

/**
 * @brief Generate a random file name for Iceberg Table write.
 * @return std::string (a random filename of form {rank:05}-rank-{uuid}.parquet)
 */
std::string generate_iceberg_file_name() {
    int rank = dist_get_rank();
    boost::uuids::uuid _uuid = boost::uuids::random_generator()();
    std::string uuid = boost::uuids::to_string(_uuid);
    int check;
    std::vector<char> fname;
    // 5+1+5+1+uuid+8
    fname.resize(20 + uuid.length());
    // The file name format is based on Spark (hence the double usage of rank in
    // the name)
    check =
        sprintf(fname.data(), "%05d-%d-%s.parquet", rank, rank, uuid.c_str());
    if (size_t(check + 1) > fname.size())
        throw std::runtime_error(
            "Fatal error: number of written char for iceberg file name is "
            "greater than fname size");
    return std::string(fname.data());
}

/**
 * Write the Bodo table (the chunk in this process) to a parquet file
 * as part of an Iceberg table
 * @param fpath full path of the parquet file to write
 * @param table table to write to parquet file
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param compression compression scheme to use
 * @param is_parallel true if the table is part of a distributed table
 * @param bucket_region in case of S3, this is the region the bucket is in
 * @param row_group_size Row group size in number of rows
 * @param iceberg_metadata Iceberg metadata string to be added to the parquet
 * schema's metadata with key 'iceberg.schema'
 * @param[out] record_count Number of records in this file
 * @param[out] file_size_in_bytes Size of the file in bytes
 */
void iceberg_pq_write_helper(const char *fpath, const table_info *table,
                             const array_info *col_names_arr,
                             const char *compression, bool is_parallel,
                             const char *bucket_region, int64_t row_group_size,
                             char *iceberg_metadata, int64_t *record_count,
                             int64_t *file_size_in_bytes) {
    std::unordered_map<std::string, std::string> md = {
        {"iceberg.schema", std::string(iceberg_metadata)}};

    // For Iceberg, all timestamp data needs to be written
    // as microseconds, so that's the type we
    // specify. `pq_write` will convert the nanoseconds to
    // microseconds during `bodo_array_to_arrow`.
    // See https://iceberg.apache.org/spec/#primitive-types,
    // https://iceberg.apache.org/spec/#parquet.
    // We've also made the decision to always
    // write the `timestamptz` type when writing
    // Iceberg data, similar to Spark.
    // The underlying already is in UTC already
    // for timezone aware types, and for timezone
    // naive, it won't matter.
    *file_size_in_bytes =
        pq_write(fpath, table, col_names_arr, nullptr, false, "", compression,
                 is_parallel, false, 0, 0, 0, "", bucket_region, row_group_size,
                 "", "UTC", arrow::TimeUnit::MICRO, md, std::string(fpath));
    *record_count = table->nrows();
}

/**
 * @brief Main function for Iceberg write which can handle partition-spec
 * and sort-orders.
 *
 * @param table_data_loc Location of the Iceberg warehouse (the data folder)
 * @param table Bodo table to write
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param partition_spec Python list of tuples containing description of the
 * partition fields (in the order that the partitions should be applied)
 * @param sort_order Python list of tuples containing description of the
 * sort fields (in the order that the sort should be performed)
 * @param compression compression scheme to use
 * @param is_parallel true if the table is part of a distributed table
 * @param bucket_region in case of S3, this is the region the bucket is in
 * @param row_group_size Row group size in number of rows
 * @param iceberg_metadata Iceberg metadata string to be added to the parquet
 * schema's metadata with key 'iceberg.schema'
 * @param[out] iceberg_files_info_py List of tuples for each of the files
 consisting of (file_name, record_count, file_size, *partition_values). Should
 be passed in as an empty list which will be filled during execution.
 */
void iceberg_pq_write(const char *table_data_loc, table_info *table,
                      const array_info *col_names_arr, PyObject *partition_spec,
                      PyObject *sort_order, const char *compression,
                      bool is_parallel, const char *bucket_region,
                      int64_t row_group_size, char *iceberg_metadata,
                      PyObject *iceberg_files_info_py) {
    tracing::Event ev("iceberg_pq_write", is_parallel);
    ev.add_attribute("table_data_loc", table_data_loc);
    ev.add_attribute("iceberg_metadata", iceberg_metadata);
    if (!PyList_Check(iceberg_files_info_py)) {
        throw std::runtime_error(
            "IcebergParquetWrite: iceberg_files_info_py is not a list");
    }
    if (!PyList_Check(sort_order)) {
        throw std::runtime_error(
            "IcebergParquetWrite: sort_order is not a list");
    }
    if (!PyList_Check(partition_spec)) {
        throw std::runtime_error(
            "IcebergParquetWrite: partition_spec is not a list");
    }

    table_info *working_table = table;

    // If sort order, then iterate over and create the transforms.
    // Sort Order should be a list of tuples.
    // Each tuple is of the form:
    // (int64_t column_idx, string transform_name,
    //  int64_t arg, int64_t asc, int64_t null_last).
    // column_idx is the position in table.
    // transform name is one of 'identity', 'bucket', 'truncate'
    // 'year', 'month', 'day', 'hour', 'void'.
    // arg is N (number of buckets) for bucket transform,
    // W (width) for truncate transform, and 0 otherwise.
    // asc when sort direction is ascending.
    // na_last when nulls should be last.
    if (PyList_Size(sort_order) > 0) {
        int64_t sort_order_len = PyList_Size(sort_order);
        tracing::Event ev_sort("iceberg_pq_write_sort", is_parallel);
        ev_sort.add_attribute("sort_order_len", sort_order_len);
        // Vector to collect the transformed columns
        std::vector<array_info *> transform_cols;
        transform_cols.reserve(sort_order_len);
        // Vector of ints (booleans essentially) to be
        // eventually passed to sort_values_table
        std::vector<int64_t> vect_ascending;
        vect_ascending.reserve(sort_order_len);
        std::vector<int64_t> na_position;
        na_position.reserve(sort_order_len);
        // Vector to record whether the transformed columns
        // should be free-ed after the sort. We will free in
        // case of non-identity transform, and in some cases
        // of identity transform (in particular in case
        // of DATETIME and DATE, see iceberg_identity_transform
        // for more details).
        std::vector<bool> free_transform_col;
        free_transform_col.reserve(sort_order_len);

        // Iterate over the python list describing the sort order
        // and create the transform columns.
        PyObject *sort_order_iter = PyObject_GetIter(sort_order);
        PyObject *sort_order_field_tuple;
        int i = 0;
        while ((sort_order_field_tuple = PyIter_Next(sort_order_iter))) {
            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *col_idx_py = PyTuple_GetItem(sort_order_field_tuple, 0);
            int64_t col_idx = PyLong_AsLongLong(col_idx_py);

            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *transform_name_py =
                PyTuple_GetItem(sort_order_field_tuple, 1);
            PyObject *transform_name_str =
                PyUnicode_AsUTF8String(transform_name_py);
            const char *transform_name_ = PyBytes_AS_STRING(transform_name_str);
            std::string transform_name(transform_name_);
            // PyBytes_AS_STRING returns the internal buffer without copy, so
            // decref should happen after the data is copied to std::string.
            // https:// docs.python.org/3/c-api/bytes.html#c.PyBytes_AsString
            Py_DECREF(transform_name_str);

            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *arg_py = PyTuple_GetItem(sort_order_field_tuple, 2);
            int64_t arg = PyLong_AsLongLong(arg_py);

            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *asc_py = PyTuple_GetItem(sort_order_field_tuple, 3);
            int64_t asc = PyLong_AsLongLong(asc_py);
            vect_ascending.push_back(asc);

            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *null_last_py = PyTuple_GetItem(sort_order_field_tuple, 4);
            int64_t null_last = PyLong_AsLongLong(null_last_py);
            na_position.push_back(null_last);

            array_info *transform_col;
            if (transform_name == "identity") {
                bool to_free;
                transform_col = iceberg_identity_transform(
                    working_table->columns[col_idx], &to_free, is_parallel);
                free_transform_col.push_back(to_free);
            } else {
                transform_col =
                    iceberg_transform(working_table->columns[col_idx],
                                      transform_name, arg, is_parallel);
                // Always free in case of non identity transform
                free_transform_col.push_back(true);
            }
            transform_cols.push_back(transform_col);

            Py_DECREF(sort_order_field_tuple);
            i++;
        }
        Py_DECREF(sort_order_iter);

        // Create a new table (a copy) with the transforms and then sort it.
        // The transformed columns go in front since they are the keys.
        std::vector<array_info *> new_cols;
        new_cols.reserve(transform_cols.size() + working_table->columns.size());
        new_cols.insert(new_cols.end(), transform_cols.begin(),
                        transform_cols.end());
        new_cols.insert(new_cols.end(), working_table->columns.begin(),
                        working_table->columns.end());
        table_info *new_table = new table_info(new_cols);

        // Convert all local dictionaries to global (and with sorted indices)
        // for all dict columns (not just transformed columns).
        // `sort_values_table` actually does this as well (call
        // convert_local_dictionary_to_global), but it then doesn't incref the
        // new arrays, so we need to convert them to global dictionaries here,
        // incref the new global dictionary approriately and then call
        // sort_values_table.
        if (is_parallel) {
            for (auto a : new_cols) {
                if ((a->arr_type == bodo_array_type::DICT) &&
                    !a->has_global_dictionary) {
                    convert_local_dictionary_to_global(a, true);
                }
            }
        }

        // sort_values_table decrefs all the arrays in the table,
        // but the original columns might still be in use, so
        // we need them to stick around.
        // We also incref the transform columns since some of them
        // might be identity columns. Note that we will incref
        // these identity columns twice (once as transformed col and again
        // as regular column), but that's ok since we also decref them twice
        // in sort_values_table.
        for (size_t i = 0; i < new_table->columns.size(); i++) {
            incref_array(new_table->columns[i]);
        }

        table_info *sorted_new_table = sort_values_table(
            new_table, transform_cols.size(), vect_ascending.data(),
            na_position.data(), /*dead_keys=*/nullptr, /*out_n_rows=*/nullptr,
            /*bounds=*/nullptr, is_parallel);

        // Remove the unused transform columns
        // TODO Optimize to not remove if they can be reused for partition spec
        sorted_new_table->columns.erase(
            sorted_new_table->columns.begin(),
            sorted_new_table->columns.begin() + transform_cols.size());

        // Decref and delete the transform columns (except the ones marked as
        // not to be free-ed, e.g. identity transform)
        for (size_t i = 0; i < transform_cols.size(); i++) {
            if (free_transform_col[i]) {
                decref_array(transform_cols[i]);
                delete transform_cols[i];
            }
        }
        transform_cols.clear();
        // Set working table for the subsequent steps
        working_table = sorted_new_table;
        // Delete unsorted table.
        delete new_table;
        ev_sort.finalize();
    }

    // If partition spec, iterate over and create the transforms
    // Partition Spec should be a list of tuples.
    // Each tuple is of the form:
    // (int64_t column_idx, string transform_name,
    //  int64_t arg, string partition_name).
    // column_idx is the position in table.
    // transform name is one of 'identity', 'bucket', 'truncate'
    // 'year', 'month', 'day', 'hour', 'void'.
    // arg is N (number of buckets) for bucket transform,
    // W (width) for truncate transform, and 0 otherwise.
    // partition_name is the name to set for folder (e.g.
    // /<partition_name>=<value>/)
    if (PyList_Size(partition_spec) > 0) {
        int64_t partition_spec_len = PyList_Size(partition_spec);
        tracing::Event ev_part("iceberg_pq_write_partition_spec", is_parallel);
        ev_part.add_attribute("partition_spec_len", partition_spec_len);
        std::vector<std::string> partition_names;
        partition_names.reserve(partition_spec_len);
        std::vector<int64_t> part_col_indices;
        part_col_indices.reserve(partition_spec_len);
        // Similar to the ones in sort-order handling
        std::vector<std::string> transform_names;
        transform_names.reserve(partition_spec_len);
        std::vector<array_info *> transform_cols;
        transform_cols.reserve(partition_spec_len);
        std::vector<bool> free_transform_col;
        free_transform_col.reserve(partition_spec_len);

        // Iterate over the partition fields and create the transform
        // columns
        PyObject *partition_spec_iter = PyObject_GetIter(partition_spec);
        PyObject *partition_spec_field_tuple;
        while (
            (partition_spec_field_tuple = PyIter_Next(partition_spec_iter))) {
            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *col_idx_py =
                PyTuple_GetItem(partition_spec_field_tuple, 0);
            int64_t col_idx = PyLong_AsLongLong(col_idx_py);

            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *transform_name_py =
                PyTuple_GetItem(partition_spec_field_tuple, 1);
            PyObject *transform_name_str =
                PyUnicode_AsUTF8String(transform_name_py);
            const char *transform_name_ = PyBytes_AS_STRING(transform_name_str);
            std::string transform_name(transform_name_);
            // PyBytes_AS_STRING returns the internal buffer without copy, so
            // decref should happen after the data is copied to std::string.
            // https:// docs.python.org/3/c-api/bytes.html#c.PyBytes_AsString
            Py_DECREF(transform_name_str);
            transform_names.push_back(transform_name);

            // PyTuple_GetItem returns borrowed reference, no
            // need to decref
            PyObject *arg_py = PyTuple_GetItem(partition_spec_field_tuple, 2);
            int64_t arg = PyLong_AsLongLong(arg_py);

            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *partition_name_py =
                PyTuple_GetItem(partition_spec_field_tuple, 3);
            PyObject *partition_name_str =
                PyUnicode_AsUTF8String(partition_name_py);
            const char *partition_name_ = PyBytes_AS_STRING(partition_name_str);
            std::string partition_name(partition_name_);
            // PyBytes_AS_STRING returns the internal buffer without copy, so
            // decref should happen after the data is copied to std::string.
            // https:// docs.python.org/3/c-api/bytes.html#c.PyBytes_AsString
            Py_DECREF(partition_name_str);
            partition_names.push_back(partition_name);

            array_info *transform_col;
            if (transform_name == "identity") {
                bool to_free;
                transform_col = iceberg_identity_transform(
                    working_table->columns[col_idx], &to_free, is_parallel);
                free_transform_col.push_back(to_free);
            } else {
                transform_col =
                    iceberg_transform(working_table->columns[col_idx],
                                      transform_name, arg, is_parallel);
                // Always free in case of non identity transform
                free_transform_col.push_back(true);
            }
            transform_cols.push_back(transform_col);
            part_col_indices.push_back(col_idx);

            Py_DECREF(partition_spec_field_tuple);
        }
        Py_DECREF(partition_spec_iter);

        // Convert all local dictionaries to global for dict columns.
        // to enable hashing.
        // This can only happen in the case of identity transform or
        // truncate transform on DICT arrays.
        // In case of identity transform, since transformed column is
        // just the original column, the original column will also
        // get a global dictionary.
        if (is_parallel) {
            for (auto a : transform_cols) {
                if ((a->arr_type == bodo_array_type::DICT) &&
                    !a->has_global_dictionary) {
                    convert_local_dictionary_to_global(a);
                }
            }
        }

        // Create a new table (a copy) with the transforms.
        // new_table will have transformed partition columns at the beginning
        // and the rest after (to use multi_col_key for hashing which assumes
        // that keys are at the beginning), and we will then drop the
        // transformed columns from it for writing
        std::vector<array_info *> new_cols;
        new_cols.reserve(transform_cols.size() + working_table->columns.size());
        new_cols.insert(new_cols.end(), transform_cols.begin(),
                        transform_cols.end());
        new_cols.insert(new_cols.end(), working_table->columns.begin(),
                        working_table->columns.end());
        table_info *new_table = new table_info(new_cols);

        // XXX Some of the code below is similar to that in
        // pq_write_partitioned, and should be refactored.

        // Create partition keys and "parts", populate partition_write_info
        // structs
        tracing::Event ev_part_gen("iceberg_pq_write_partition_spec_gen_parts",
                                   is_parallel);
        const uint32_t seed = SEED_HASH_PARTITION;
        uint32_t *hashes = hash_keys(transform_cols, seed, is_parallel);
        UNORD_MAP_CONTAINER<multi_col_key, partition_write_info,
                            multi_col_key_hash>
            key_to_partition;

        new_table->num_keys = transform_cols.size();
        for (uint64_t i = 0; i < new_table->nrows(); i++) {
            multi_col_key key(hashes[i], new_table, i);
            partition_write_info &p = key_to_partition[key];
            if (p.rows.size() == 0) {
                // This is the path after the table_loc that will
                // be returned to Python.
                std::string inner_path = "";

                // We store the iceberg-file-info as part of the
                // partition_write_info struct. It's a tuple of length
                // 3 (for file_name, record_count and file_size) + number of
                // partition fields (same as number of transform cols).
                // See function description string for more details.
                p.iceberg_file_info_py = PyTuple_New(3 + transform_cols.size());

                for (size_t j = 0; j < transform_cols.size(); j++) {
                    auto transformed_part_col = transform_cols[j];
                    // convert transformed partition value to string
                    std::string value_str = transform_val_to_str(
                        transform_names[j],
                        new_table->columns[part_col_indices[j] +
                                           transform_names.size()],
                        transformed_part_col, i);
                    inner_path += partition_names[j] + "=" + value_str + "/";
                    // Get python representation of the partition value
                    // and then add it to the icberg-file-info tuple.
                    PyObject *partition_val_py =
                        iceberg_transformed_val_to_py(transformed_part_col, i);
                    PyTuple_SET_ITEM(p.iceberg_file_info_py, 3 + j,
                                     partition_val_py);
                }
                // create a random file name
                inner_path += generate_iceberg_file_name();
                PyObject *file_name_py =
                    PyUnicode_FromString(inner_path.c_str());
                PyTuple_SET_ITEM(p.iceberg_file_info_py, 0, file_name_py);

                // Generate output file name
                // TODO: Make our path handling more consistent between C++ and
                // Java
                p.fpath = std::string(table_data_loc);
                if (p.fpath.back() != '/') p.fpath += "/";
                p.fpath += inner_path;
            }
            p.rows.push_back(i);
        }
        delete[] hashes;
        ev_part_gen.finalize();

        // Remove the unused transform columns
        // Note that we can remove even the identity transforms
        // since they were just copies (pointer copy)
        new_table->columns.erase(
            new_table->columns.begin(),
            new_table->columns.begin() + transform_cols.size());

        // Free the transform cols
        for (size_t i = 0; i < transform_cols.size(); i++) {
            if (free_transform_col[i]) {
                delete transform_cols[i];
            }
        }
        transform_cols.clear();

        tracing::Event ev_part_write(
            "iceberg_pq_write_partition_spec_write_parts", is_parallel);
        // Write the file for each partition key
        for (auto it = key_to_partition.begin(); it != key_to_partition.end();
             it++) {
            const partition_write_info &p = it->second;
            // RetrieveTable steals the reference but we still need them
            for (auto a : new_table->columns) incref_array(a);
            table_info *part_table =
                RetrieveTable(new_table, p.rows, new_table->ncols());
            // NOTE: we pass is_parallel=False because we already took care of
            // is_parallel here
            Bodo_Fs::FsEnum fs_type = filesystem_type(p.fpath.c_str());
            if (fs_type == Bodo_Fs::FsEnum::posix) {
                // s3 and hdfs create parent directories automatically when
                // writing partitioned columns
                std::filesystem::path path = p.fpath;
                std::filesystem::create_directories(path.parent_path());
            }

            // Write the file and then attach the relevant information
            // to the iceberg-file-info tuple for this file.
            // We don't need to check if record count is zero in this case.
            int64_t record_count, file_size_in_bytes;
            iceberg_pq_write_helper(p.fpath.c_str(), part_table, col_names_arr,
                                    compression, false, bucket_region,
                                    row_group_size, iceberg_metadata,
                                    &record_count, &file_size_in_bytes);
            PyObject *record_count_py = PyLong_FromLongLong(record_count);
            PyObject *file_size_in_bytes_py =
                PyLong_FromLongLong(file_size_in_bytes);
            PyTuple_SET_ITEM(p.iceberg_file_info_py, 1, record_count_py);
            PyTuple_SET_ITEM(p.iceberg_file_info_py, 2, file_size_in_bytes_py);
            // PyTuple_SET_ITEM steals a reference so decref isn't needed for
            // record_count_py and file_size_in_bytes_py
            PyList_Append(iceberg_files_info_py, p.iceberg_file_info_py);
            Py_DECREF(p.iceberg_file_info_py);
            delete_table_decref_arrays(part_table);
        }
        ev_part_write.finalize();
        delete new_table;
        ev_part.finalize();
    } else {
        tracing::Event ev_general("iceberg_pq_write_general", is_parallel);
        // If no partition spec, then write the working table (after sort if
        // there was a sort order else the table as is)
        std::string fname = generate_iceberg_file_name();
        int64_t record_count;
        int64_t file_size_in_bytes;
        std::string fpath(table_data_loc);
        if (fpath.back() != '/') fpath += "/";
        fpath += fname;
        Bodo_Fs::FsEnum fs_type = filesystem_type(fpath.c_str());
        if (fs_type == Bodo_Fs::FsEnum::posix) {
            // s3 and hdfs create parent directories automatically when
            // writing partitioned columns
            std::filesystem::path path = fpath;
            std::filesystem::create_directories(path.parent_path());
        }
        // XXX Should we pass is_parallel instead of always false?
        // We can't at the moment due to how pq_write works. Once we
        // refactor that a little, we should be able to handle this
        // more elegantly.
        iceberg_pq_write_helper(fpath.c_str(), working_table, col_names_arr,
                                compression, false, bucket_region,
                                row_group_size, iceberg_metadata, &record_count,
                                &file_size_in_bytes);

        if (record_count > 0) {
            // Only need to report the file back if we created it
            PyObject *file_name_py = PyUnicode_FromString(fname.c_str());
            PyObject *record_count_py = PyLong_FromLongLong(record_count);
            PyObject *file_size_in_bytes_py =
                PyLong_FromLongLong(file_size_in_bytes);
            PyObject *iceberg_file_info_py = PyTuple_New(3);
            PyTuple_SET_ITEM(iceberg_file_info_py, 0, file_name_py);
            PyTuple_SET_ITEM(iceberg_file_info_py, 1, record_count_py);
            PyTuple_SET_ITEM(iceberg_file_info_py, 2, file_size_in_bytes_py);
            // PyTuple_SET_ITEM steals a reference so decref isn't needed for
            // record_count_py and file_size_in_bytes_py
            PyList_Append(iceberg_files_info_py, iceberg_file_info_py);
            Py_DECREF(iceberg_file_info_py);
        }
        ev_general.finalize();
    }
}

/**
 * @brief Python entrypoint for the iceberg write function
 * with error handling.
 */
PyObject *iceberg_pq_write_py_entry(
    const char *table_data_loc, table_info *table,
    const array_info *col_names_arr, PyObject *partition_spec,
    PyObject *sort_order, const char *compression, bool is_parallel,
    const char *bucket_region, int64_t row_group_size, char *iceberg_metadata) {
    try {
        // Python list of tuples describing the data files written.
        // iceberg_pq_write will append to the list and then this will be
        // returned to the Python.
        PyObject *iceberg_files_info_py = PyList_New(0);
        iceberg_pq_write(table_data_loc, table, col_names_arr, partition_spec,
                         sort_order, compression, is_parallel, bucket_region,
                         row_group_size, iceberg_metadata,
                         iceberg_files_info_py);
        return iceberg_files_info_py;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
