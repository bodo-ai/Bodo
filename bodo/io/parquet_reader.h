// Copyright (C) 2021 Bodo Inc. All rights reserved.

// Implementation of ParquetReader (subclass of ArrowDataframeReader) with
// functionality that is specific to reading parquet datasets

#include "arrow_reader.h"
#include "parquet/api/reader.h"
#include "parquet/arrow/reader.h"

class ParquetReader : public ArrowDataframeReader {
   public:
    /**
     * Initialize ParquetReader.
     * See pq_read function below for description of arguments.
     */
    ParquetReader(PyObject* _path, bool _parallel, char* _bucket_region,
                  PyObject* _dnf_filters, PyObject* _expr_filters,
                  PyObject* _storage_options, int64_t _tot_rows_to_read,
                  int32_t* _selected_fields, int32_t num_selected_fields,
                  int32_t* is_nullable, bool _input_file_name_col)
        : ArrowDataframeReader(_parallel, _tot_rows_to_read, _selected_fields,
                               num_selected_fields, is_nullable),
          dnf_filters(_dnf_filters),
          expr_filters(_expr_filters),
          path(_path),
          storage_options(_storage_options),
          input_file_name_col(_input_file_name_col),
          bucket_region(_bucket_region) {
        if (storage_options == Py_None)
            throw std::runtime_error("ParquetReader: storage_options is None");

        // copy selected_fields to a Python list to pass to
        // parquet_pio.get_scanner_batches
        selected_fields_py = PyList_New(selected_fields.size());
        size_t i = 0;
        for (auto field_num : selected_fields) {
            PyList_SetItem(selected_fields_py, i++, PyLong_FromLong(field_num));
        }

        // Extract values from the storage_options dict
        // Check that it's a dictionary, else throw an error
        if (PyDict_Check(storage_options)) {
            // Get value of "anon". Returns NULL if it doesn't exist in the
            // dict. No need to decref s3fs_anon_py, PyDict_GetItemString
            // returns borrowed ref
            PyObject* s3fs_anon_py =
                PyDict_GetItemString(storage_options, "anon");
            if (s3fs_anon_py != NULL && s3fs_anon_py == Py_True) {
                this->s3fs_anon = true;
            }
        } else {
            throw std::runtime_error(
                "ParquetReader: storage_options is not a Python dictionary.");
        }
    }

    /**
     * Initialize the reader
     * See pq_read function below for description of arguments.
     */
    virtual void init(int32_t* _str_as_dict_cols, int32_t num_str_as_dict_cols,
                      int32_t* _part_cols_cat_dtype,
                      int32_t* _selected_part_cols,
                      int32_t num_partition_cols) {
        // initialize reader
        ArrowDataframeReader::init(
            {_str_as_dict_cols, _str_as_dict_cols + num_str_as_dict_cols});
        if (parallel) {
            // Get the average number of pieces per rank. This is used to
            // increase the number of threads of the Arrow batch reader
            // for ranks that have to read many more files than others.
            int num_ranks;
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
            uint64_t num_pieces = static_cast<uint64_t>(get_num_pieces());
            MPI_Allreduce(MPI_IN_PLACE, &num_pieces, 1, MPI_UINT64_T, MPI_SUM,
                          MPI_COMM_WORLD);
            avg_num_pieces = num_pieces / static_cast<double>(num_ranks);
        }
        // allocate output partition columns. These are categorical columns
        // where we only fill out the codes in C++ (see fill_partition_column
        // comments)
        for (auto i = 0; i < num_partition_cols; i++) {
            part_cols.push_back(alloc_array(
                count, -1, -1, bodo_array_type::NUMPY,
                Bodo_CTypes::CTypeEnum(_part_cols_cat_dtype[i]), 0, -1));
            selected_part_cols.push_back(_selected_part_cols[i]);
        }
        part_cols_offset.resize(num_partition_cols, 0);

        // TODO In case of S3, get the region here instead of having it
        // be passed into the constructor.
    }

    virtual ~ParquetReader() {}

    /// a piece is a single parquet file in the context of parquet
    virtual size_t get_num_pieces() const override { return file_paths.size(); }

    /// returns output partition columns
    std::vector<array_info*>& get_partition_cols() { return part_cols; }

    array_info* get_input_file_name_col() { return input_file_name_col_arr; }

   protected:
    virtual void add_piece(PyObject* piece, int64_t num_rows,
                           int64_t total_rows) override;

    virtual PyObject* get_dataset() override;

    virtual std::shared_ptr<arrow::Schema> get_schema(
        PyObject* dataset) override;

    virtual void read_all(TableBuilder& builder) override;

    PyObject* dnf_filters = nullptr;
    PyObject* expr_filters = nullptr;
    // selected columns in the parquet file (not fields). For example,
    // field "struct<A: int64, B: int64>" has two int64 columns in the
    // parquet file
    std::vector<int> selected_columns;
    // Prefix to add to each of the file paths before they are opened
    std::string prefix;

   private:
    PyObject* path;  // path passed to pd.read_parquet() call
    PyObject* storage_options;
    PyObject* selected_fields_py;
    bool input_file_name_col;
    // parquet dataset has partitions (regardless of whether we select them or
    // not)
    bool ds_has_partitions = false;

    std::vector<int64_t> pieces_nrows;
    double avg_num_pieces = 0;

    // Parquet files that this process has to read
    std::vector<std::string> file_paths;
    std::string bucket_region;  // s3 bucket region
    bool s3fs_anon = false;     // s3 anonymous mode

    // selected partition columns
    std::vector<int> selected_part_cols;
    // for each piece that this process reads, store the value of each partition
    // column (value is stored as the categorical code). Note that a given
    // piece/file has the same partition value for all of its rows
    std::vector<std::vector<int64_t>> part_vals;
    // output partition columns
    std::vector<array_info*> part_cols;
    // current fill offset of each partition column
    std::vector<int64_t> part_cols_offset;

    // output input_file_name column
    // indices for the dictionary-encoding
    array_info* input_file_name_col_indices_arr = nullptr;
    int64_t input_file_name_col_indices_offset = 0;
    // dictionary for the dictionary encoding
    array_info* input_file_name_col_dict_arr = nullptr;
    int64_t input_file_name_col_dict_arr_total_chars = 0;
    // output array_info for the dictionary-encoded
    // string array
    array_info* input_file_name_col_arr = nullptr;

    /**
     * Get values for all partition columns of a piece of
     * pyarrow.parquet.ParquetDataset and store in part_vals.
     * @param piece : ParquetDataset piece (a single parquet file)
     */
    void get_partition_info(PyObject* piece);
};
