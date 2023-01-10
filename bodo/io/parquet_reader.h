// Copyright (C) 2022 Bodo Inc. All rights reserved.

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
    ParquetReader(PyObject* _path, bool _parallel, PyObject* _dnf_filters,
                  PyObject* _expr_filters, PyObject* _storage_options,
                  PyObject* pyarrow_schema, int64_t _tot_rows_to_read,
                  std::set<int> _selected_fields, std::vector<bool> is_nullable,
                  bool _input_file_name_col, bool _use_hive = true)
        : ArrowDataframeReader(_parallel, pyarrow_schema, _tot_rows_to_read,
                               _selected_fields, is_nullable),
          dnf_filters(_dnf_filters),
          expr_filters(_expr_filters),
          path(_path),
          storage_options(_storage_options),
          input_file_name_col(_input_file_name_col),
          use_hive(_use_hive) {
        if (storage_options == Py_None)
            throw std::runtime_error("ParquetReader: storage_options is None");
    }

    /**
     * Initialize the reader
     * See pq_read function below for description of arguments.
     */
    void init_pq_reader(int32_t* _str_as_dict_cols,
                        int32_t num_str_as_dict_cols,
                        int32_t* _part_cols_cat_dtype,
                        int32_t* _selected_part_cols,
                        int32_t num_partition_cols) {
        // initialize reader
        ArrowDataframeReader::init_arrow_reader(
            {_str_as_dict_cols, _str_as_dict_cols + num_str_as_dict_cols},
            false);

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

    virtual void read_all(TableBuilder& builder) override;

    // Prefix to add to each of the file paths (only used for input_file_name)
    std::string prefix;

    PyObject* dnf_filters = nullptr;
    PyObject* expr_filters = nullptr;
    PyObject* filesystem = nullptr;
    // dataset partitioning info (regardless of whether we select partition
    // columns or not)
    PyObject* ds_partitioning = nullptr;

   private:
    PyObject* path;  // path passed to pd.read_parquet() call
    PyObject* storage_options;
    bool input_file_name_col;
    bool use_hive;

    std::vector<int64_t> pieces_nrows;
    double avg_num_pieces = 0;

    // Parquet files that this process has to read
    std::vector<std::string> file_paths;

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
