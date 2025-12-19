#pragma once
// Implementation of ParquetReader (subclass of ArrowReader) with
// functionality that is specific to reading parquet datasets

#include "../libs/_bodo_to_arrow.h"
#include "../libs/_distributed.h"
#include "../libs/_utils.h"
#include "arrow_reader.h"

class ParquetReader : public ArrowReader {
   public:
    /**
     * Initialize ParquetReader.
     * See pq_read_py_entry function below for description of arguments.
     */
    ParquetReader(PyObject* _path, bool _parallel, PyObject* _expr_filters,
                  PyObject* _storage_options, PyObject* _pyarrow_schema,
                  int64_t _tot_rows_to_read, std::vector<int> _selected_fields,
                  std::vector<bool> is_nullable, bool _input_file_name_col,
                  int64_t batch_size, bool _use_hive = true, int64_t op_id = -1)
        : ArrowReader(_parallel, _pyarrow_schema, _tot_rows_to_read,
                      _selected_fields, is_nullable, batch_size, op_id),
          empty_out_table(nullptr),
          expr_filters(_expr_filters),
          path(_path),
          storage_options(_storage_options),
          input_file_name_col(_input_file_name_col),
          use_hive(_use_hive) {
        if (storage_options == Py_None) {
            throw std::runtime_error("ParquetReader: storage_options is None");
        }
        Py_INCREF(storage_options);
        Py_INCREF(path);
        Py_INCREF(pyarrow_schema);
    }

    /**
     * Initialize the reader
     * See pq_read_py_entry function below for description of arguments.
     */
    void init_pq_reader(std::span<int32_t> str_as_dict_cols,
                        int32_t* _part_cols_cat_dtype,
                        int32_t* _selected_part_cols,
                        int32_t num_partition_cols) {
        // initialize reader
        ArrowReader::init_arrow_reader(str_as_dict_cols, false);
        this->dict_builders = std::vector<std::shared_ptr<DictionaryBuilder>>(
            schema->num_fields());
        if (parallel) {
            // Get the average number of pieces per rank. This is used to
            // increase the number of threads of the Arrow batch reader
            // for ranks that have to read many more files than others.
            int num_ranks;
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
            uint64_t num_pieces = static_cast<uint64_t>(get_num_pieces());
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &num_pieces, 1, MPI_UINT64_T,
                              MPI_SUM, MPI_COMM_WORLD),
                "ParquetReader::init_pq_reader: MPI error on MPI_Allreduce:");
            avg_num_pieces = num_pieces / static_cast<double>(num_ranks);
        }
        // allocate output partition columns. These are categorical columns
        // where we only fill out the codes in C++ (see fill_partition_column
        // comments)
        for (auto i = 0; i < num_partition_cols; i++) {
            part_cols.push_back(alloc_array_top_level(
                count, -1, -1, bodo_array_type::NUMPY,
                Bodo_CTypes::CTypeEnum(_part_cols_cat_dtype[i])));
            selected_part_cols.push_back(_selected_part_cols[i]);
            this->part_cols_cat_dtype.push_back(_part_cols_cat_dtype[i]);
        }
        part_cols_offset.resize(num_partition_cols, 0);

        this->init_pq_scanner();
        // Construct ChunkedTableBuilder for output in the streaming case.
        if (this->batch_size != -1) {
            this->dict_builders =
                std::vector<std::shared_ptr<DictionaryBuilder>>(
                    selected_fields.size() + num_partition_cols);
            for (size_t i = 0; i < selected_fields.size(); i++) {
                const std::shared_ptr<arrow::Field>& field =
                    schema->field(selected_fields[i]);
                this->dict_builders[i] = create_dict_builder_for_array(
                    arrow_type_to_bodo_data_type(field->type()), false);
            }
            for (int i = 0; i < num_partition_cols; i++) {
                auto partition_col = part_cols[i];
                this->dict_builders[selected_fields.size() + i] =
                    create_dict_builder_for_array(partition_col, false);
            }

            // Generate a mapping from schema index to selected fields for the
            // str_as_dict_cols.
            std::vector<int32_t> str_as_dict_cols_map(schema->num_fields(), -1);
            for (size_t i = 0; i < selected_fields.size(); i++) {
                str_as_dict_cols_map[selected_fields[i]] = i;
            }

            // TODO: Remove. This step is unnecessary if we can guarantee that
            // the arrow schema always specifies if the fields should be
            // dictionary.
            for (int str_as_dict_col : str_as_dict_cols) {
                int32_t index = str_as_dict_cols_map[str_as_dict_col];
                this->dict_builders[index] = create_dict_builder_for_array(
                    std::make_unique<bodo::DataType>(bodo_array_type::DICT,
                                                     Bodo_CTypes::STRING),
                    false);
            }
            auto empty_table = get_empty_out_table();
            this->out_batches = std::make_shared<ChunkedTableBuilder>(
                empty_table->schema(), this->dict_builders, (size_t)batch_size);
        }
    }

    virtual ~ParquetReader() {
        // Remove after reader is finished or on error
        Py_XDECREF(this->reader);
        Py_DECREF(this->storage_options);
        Py_DECREF(this->path);
        Py_DECREF(this->pyarrow_schema);
    }

    /// a piece is a single parquet file in the context of parquet
    size_t get_num_pieces() const override { return file_paths.size(); }

    /// returns output partition columns
    std::vector<std::shared_ptr<array_info>>& get_partition_cols() {
        return part_cols;
    }

    std::shared_ptr<array_info> get_input_file_name_col() {
        return input_file_name_col_arr;
    }

   protected:
    void add_piece(PyObject* piece, int64_t num_rows) override;

    PyObject* get_dataset() override;

    std::tuple<table_info*, bool, uint64_t> read_inner_row_level() override;

    std::tuple<table_info*, bool, uint64_t> read_inner_piece_level() override {
        throw std::runtime_error(
            "ParquetReader::read_inner_piece_level: Not supported!");
    }

    std::shared_ptr<table_info> get_empty_out_table() override;

    std::shared_ptr<table_info> empty_out_table;

    // Prefix to add to each of the file paths (only used for input_file_name)
    std::string prefix;

    PyObjectPtr expr_filters = nullptr;
    PyObjectPtr filesystem = nullptr;
    // dataset partitioning info (regardless of whether we select partition
    // columns or not)
    PyObjectPtr ds_partitioning = nullptr;

    // Parquet files that this process has to read
    std::vector<std::string> file_paths;

    PyObject* path;  // path passed to pd.read_parquet() call
    // We don't own storage options so store the raw pointer
    PyObject* storage_options;
    bool input_file_name_col;
    bool use_hive;

    std::vector<int64_t> pieces_nrows;
    double avg_num_pieces = 0;

    // Selected partition columns
    std::vector<int> selected_part_cols;
    // for each piece that this process reads, store the value of each partition
    // column (value is stored as the categorical code). Note that a given
    // piece/file has the same partition value for all of its rows
    std::vector<std::vector<int64_t>> part_vals;
    // output partition columns
    std::vector<std::shared_ptr<array_info>> part_cols;
    // current fill offset of each partition column
    std::vector<int64_t> part_cols_offset;

    // output input_file_name column
    // indices for the dictionary-encoding
    std::shared_ptr<array_info> input_file_name_col_indices_arr = nullptr;
    int64_t input_file_name_col_indices_offset = 0;
    // dictionary for the dictionary encoding
    std::shared_ptr<array_info> input_file_name_col_dict_arr = nullptr;
    int64_t input_file_name_col_dict_arr_total_chars = 0;
    // output array_info for the dictionary-encoded
    // string array
    std::shared_ptr<array_info> input_file_name_col_arr = nullptr;

    /**
     * Get values for all partition columns of a piece of
     * pyarrow.parquet.ParquetDataset and store in part_vals.
     * @param piece : ParquetDataset piece (a single parquet file)
     */
    void get_partition_info(PyObject* piece);

    /**
     * @brief Set up the Arrow Scanner to read the Parquet files
     * (pieces) associated for the current rank
     */
    virtual void init_pq_scanner();

    // Arrow Batched Reader to get next table iteratively
    PyObject* reader = nullptr;

    // -------------- Streaming Specific Parameters --------------

    // Number of remaining rows to skip outputting
    int64_t rows_to_skip = -1;

    // Index of the current piece (Parquet file) being read
    // Needed for constructing partition columns
    size_t cur_piece = 0;

    // Number of Rows Left in the Current Piece
    int64_t rows_left_cur_piece;

    // Index Dtype for Partition Columns
    // Streaming only needs this because we need to
    // reconstruct the arrays at read time
    std::vector<int32_t> part_cols_cat_dtype;
};
