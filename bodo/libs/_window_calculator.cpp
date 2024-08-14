// Copyright (C) 2024 Bodo Inc. All rights reserved.
#include "_window_calculator.h"
#include <iostream>
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_groupby_common.h"
#include "_groupby_do_apply_to_column.h"
#include "_groupby_ftypes.h"
#include "_memory.h"
#include "_shuffle.h"
#include "_table_builder_utils.h"

// See the following link for an explanation of the classes in this file:
// https://bodo.atlassian.net/wiki/spaces/B/pages/1832189971/Streaming+Window+Sort+Implementation+Design+Overhaul

/**
 * @brief Helper utility to fetch a specific set of columns from a chunk and
 * append them to a vector.
 *
 * @param[in] chunk: The table of input data whose columns are being fetched.
 * @param[in] col_indices: The vector of indices of columns from chunk to fetch.
 * @param[out] out: The vector to append the columns to.
 */
inline void fetch_columns(const std::shared_ptr<table_info> &chunk,
                          const std::vector<int32_t> &col_indices,
                          std::vector<std::shared_ptr<array_info>> &out) {
    for (int32_t idx : col_indices) {
        out.push_back(chunk->columns[idx]);
    }
}

template <bodo_array_type::arr_type_enum OrderByArrType>
    requires(valid_window_array_type<OrderByArrType>)
class BaseWindowCalculator {
   public:
    bodo::IBufferPool *const pool;
    std::shared_ptr<::arrow::MemoryManager> mm;

    // The schema of the input data
    std::shared_ptr<bodo::Schema> schema;

    // The vector of sorted data.
    std::vector<std::shared_ptr<table_info>> chunks;

    // The columns indices in chunks that are used for orderby values.
    std::vector<int32_t> order_col_indices;

    // The columns indices in chunks that are inputs to the function.
    std::vector<int32_t> input_col_indices;

    // The dictionary builders used for unification, mapping 1:1 to columns in
    // chunks.
    std::vector<std::shared_ptr<DictionaryBuilder>> builders;

    // Whether the current rank is empty.
    bool is_empty;

    /**
     * @brief Constructor for the generic window class calculator.
     * @param[in] _schema: the table schema of the input data.
     * @param[in] _chunks: the vector of table_info representing
     * the sorted table spread out across multiple chunks.
     * @param[in] _order_col_indices: the indices of columns
     * in _chunks that correspond to order columns.
     * @param[in] input_col_indices: the indices in _chunks corresponding
     * to inputs to the window function being calculated.
     * @param[in] _builders: the dictionary builders used for _chunks.
     * These should be used to ensure unification of columns.
     * @param[in] _is_empty: whether the current rank is empty.
     * @param[in] _pool: the buffer pool that should be used for allocations.
     * @param[in] _mm: the memory manager that should be used for allocations.
     */
    BaseWindowCalculator(
        std::shared_ptr<bodo::Schema> _schema,
        std::vector<std::shared_ptr<table_info>> _chunks,
        std::vector<int32_t> _order_col_indices,
        std::vector<int32_t> _input_col_indices,
        std::vector<std::shared_ptr<DictionaryBuilder>> _builders,
        bool _is_empty, bodo::IBufferPool *const _pool,
        std::shared_ptr<::arrow::MemoryManager> _mm)
        : pool(_pool),
          mm(_mm),
          schema(_schema),
          chunks(_chunks),
          order_col_indices(_order_col_indices),
          input_col_indices(_input_col_indices),
          builders(_builders),
          is_empty(_is_empty) {}

    virtual ~BaseWindowCalculator() = default;

    /**
     * @brief Function responsible for computing the local results of the window
     * function for an entire partition that starts on row partition_start_row
     * of chunk partition_start_chunk and ends on row partition_end_row of chunk
     * partition_end_chunk_idx (inclusive).
     *
     * @param[in] partition_start_chunk: the chunk index of the start of the
     * partition (inclusive).
     * @param[in] partition_start_row: the row index of the start of the
     * partition within partition_start_chunk (inclusive).
     * @param[in] partition_end_chunk_idx: the chunk index of the end of the
     * partition (inclusive).
     * @param[in] partition_end_row: the row index of the end of the partition
     * within partition_end_chunk_idx (inclusive).
     * @param[in] is_last: is this the last partition on the current rank?
     */
    virtual void ComputePartition(size_t partition_start_chunk,
                                  size_t partition_start_row,
                                  size_t partition_end_chunk_idx,
                                  size_t partition_end_row, bool is_last) {
        throw std::runtime_error("ComputePartition: requires implementation");
    }

    /**
     * Returns the number of columns that this calculator adds to the shared
     * state. Should be equal to the sizes of the vectors returned by
     * GetFirstRowInfo and GetLastRowInfo.
     */
    virtual size_t NumAuxillaryColumns() {
        throw std::runtime_error(
            "NumAuxillaryColumns: requires implementation");
    }

    /**
     * @brief Returns whether the calculator uses the shared state from
     * WindowCollectionComputer.
     */
    virtual bool UsesSharedCommunication() { return true; }

    /**
     * @brief Fetches the vector of single-row columns corresponding to data
     * passed along with the last row.
     *
     * @param[out] out_cols: the vector that the columns are appended to.
     */
    virtual void GetLastRowInfo(
        std::vector<std::shared_ptr<array_info>> &out_cols) {
        throw std::runtime_error("GetLastRowInfo: requires implementation");
    }

    /**
     * @brief Fetches the vector of single-row columns corresponding to data
     * passed along with the first row. Default implementation: re-use the
     * last-row logic.
     *
     * @param[out] out_cols: the vector that the columns are appended to.
     */
    virtual void GetFirstRowInfo(
        std::vector<std::shared_ptr<array_info>> &out_cols) {
        GetLastRowInfo(out_cols);
    }

    /**
     * @brief Updates the local results created by calling ComputePartition by
     * passing in the parallel communicated results across ranks and updating
     * accordingly.
     *
     * @param[in] first_boundary_info: the table of information shared across
     * ranks. The columns correspond to the partition values, followed by the
     * order values, followed by all of the GetFirstRowInfo columns from each
     * calculator.
     * @param[in] last_boundary_info: the table of information shared across
     * ranks. The columns correspond to the partition values, followed by the
     * order values, followed by all of the GetLastRowInfo columns from each
     * calculator.
     * @param[in] rank_idx: the row in the boundary info corresponding to the
     * current rank.
     * @param[in] func_offset: the column index in boundary_info where the
     * first/last row info for the current calculator begins.
     * @param[in] start_partition: the first rank belonging to the same
     * partition as the first partition of the current rank.
     * @param[in] start_order: the first rank belonging to the same partition
     * & orderby as the first row of the current rank.
     * @param[in] end_partition: the last rank belonging to the same partition
     * as the last partition of the current rank.
     * @param[in] end_order: the last rank belonging to the same partition
     * & orderby as the last row of the current rank.
     * @param[in] first_partition_chunk_end: the index of the last chunk
     * on the current rank belonging to the first partition.
     * @param[in] first_partition_row_end: the index of the last row
     * in chunks[first_partition_chunk_end] belonging to the first partition.
     * @param[in] last_partition_chunk_start: the index of the first chunk
     * on the current rank belonging to the last partition.
     * @param[in] last_partition_row_start: the index of the first row
     * in chunks[last_partition_chunk_start] belonging to the last partition.
     */
    virtual void UpdateBoundaryInfo(
        std::shared_ptr<table_info> &first_boundary_info,
        std::shared_ptr<table_info> &last_boundary_info, size_t rank_idx,
        size_t func_offset, size_t start_partition, size_t start_order,
        size_t end_partition, size_t end_order,
        size_t first_partition_chunk_end, size_t first_partition_row_end,
        size_t last_partition_chunk_start, size_t last_partition_row_start) {
        throw std::runtime_error("UpdateBoundaryInfo: requires implementation");
    }

    /**
     * @brief Returns a column corresponding to an output chunk.
     * @param[in] chunk_idx: the index of the chunk that the returned
     * output should line up with.
     */
    virtual std::shared_ptr<array_info> ProduceOutputBatch(size_t chunk_idx) {
        throw std::runtime_error("ProduceOutputBatch: requires implementation");
    }
};

template <bodo_array_type::arr_type_enum OrderByArrType>
    requires(valid_window_array_type<OrderByArrType>)
class RowNumberWindowCalculator : public BaseWindowCalculator<OrderByArrType> {
   public:
    RowNumberWindowCalculator(
        std::shared_ptr<bodo::Schema> _schema,
        std::vector<std::shared_ptr<table_info>> _chunks,
        std::vector<int32_t> _order_col_indices,
        std::vector<int32_t> _input_col_indices,
        std::vector<std::shared_ptr<DictionaryBuilder>> _builders,
        bool is_empty, bodo::IBufferPool *const _pool,
        std::shared_ptr<::arrow::MemoryManager> _mm)
        : BaseWindowCalculator<OrderByArrType>(
              _schema, _chunks, _order_col_indices, _input_col_indices,
              _builders, is_empty, _pool, _mm) {
        // For each input chunk, create a corresponding output chunk with the
        // same number of rows to store the row_number values.
        for (size_t chunk_idx = 0; chunk_idx < this->chunks.size();
             chunk_idx++) {
            size_t n_rows = this->chunks[chunk_idx]->nrows();
            std::shared_ptr<array_info> out_chunk =
                alloc_numpy(n_rows, Bodo_CTypes::UINT64, this->pool, this->mm);
            out_chunk->unpin();
            row_number_chunks.push_back(out_chunk);
        }
    }

    void ComputePartition(size_t partition_start_chunk,
                          size_t partition_start_row,
                          size_t partition_end_chunk, size_t partition_end_row,
                          bool is_last) final {
        // Skip if the input data was empty.
        if (this->is_empty) {
            return;
        }

        // Iterate across all of the chunks from partition_start_chunk to
        // partition_end_chunk, inclusive.
        size_t current_row_number = 0;
        for (size_t chunk_idx = partition_start_chunk;
             chunk_idx <= partition_end_chunk; chunk_idx++) {
            std::shared_ptr<array_info> row_number_chunk =
                row_number_chunks[chunk_idx];
            row_number_chunk->pin();
            size_t *row_number_data =
                row_number_chunk->data1<bodo_array_type::NUMPY, size_t>();

            // Iterate across all of the rows in the current chunk. If we are in
            // the first chunk, start at partition_start_row. If we are in the
            // last chunk, end on partition_end_row.
            size_t start_row =
                (chunk_idx == partition_start_chunk) ? partition_start_row : 0;
            size_t end_row = (chunk_idx == partition_end_chunk)
                                 ? partition_end_row
                                 : (row_number_chunk->length - 1);
            for (size_t row = start_row; row <= end_row; row++) {
                current_row_number++;
                row_number_data[row] = current_row_number;
            }
            row_number_chunk->unpin();
        }

        // If this is the last partition, store the current row number for
        // communication across ranks.
        if (is_last) {
            row_number = current_row_number;
        }
    }

    size_t NumAuxillaryColumns() final { return 1; }

    void GetLastRowInfo(
        std::vector<std::shared_ptr<array_info>> &out_cols) final {
        // Allocate a single column storing the last row_number value.
        std::shared_ptr<array_info> row_number_arr =
            alloc_numpy(1, Bodo_CTypes::UINT64);
        row_number_arr->data1<bodo_array_type::NUMPY, size_t>()[0] = row_number;
        if (this->is_empty) {
            // If the input data is empty, ensure the row_number arr is also
            // empty.
            row_number_arr =
                alloc_array_like(row_number_arr, true, this->pool, this->mm);
        }
        out_cols.push_back(row_number_arr);
    }

    /**
     * Arguments used by this implementation:
     * - last_boundary_info: used to send row_number values from previous ranks
     * - rank_idx: used to determine which row in last_boundary_info is the
     * current rank.
     * - func_offset: used to infer which column of last_boundary_info contains
     *   the ROW_NUMBER values from previous ranks.
     * - start_partition: used as the start of previous ranks to sum.
     * - first_partition_chunk_end: used to determine up to which chunk to
     * update with the ROW_NUMBER values from previous ranks.
     * - first_partition_row_end: used to determine up to which row in
     *   first_partition_chunk_end to update.
     */
    void UpdateBoundaryInfo(std::shared_ptr<table_info> &first_boundary_info,
                            std::shared_ptr<table_info> &last_boundary_info,
                            size_t rank_idx, size_t func_offset,
                            size_t start_partition, size_t start_order,
                            size_t end_partition, size_t end_order,
                            size_t first_partition_chunk_end,
                            size_t first_partition_row_end,
                            size_t last_partition_chunk_start,
                            size_t last_partition_row_start) final {
        // Compute the sum of the row_number values from previous ranks
        // in the same partition.
        size_t row_number_offset = 0;
        size_t *row_number_data = last_boundary_info->columns[func_offset]
                                      ->data1<bodo_array_type::NUMPY, size_t>();
        for (size_t rank = start_partition; rank < rank_idx; rank++) {
            row_number_offset += row_number_data[rank];
        }

        // If there is a value to add, do so to all rows in the prefix
        // of the current rank belonging to the first partition.
        if (row_number_offset > 0) {
            for (size_t chunk_idx = 0; chunk_idx <= first_partition_chunk_end;
                 chunk_idx++) {
                // Fetch the current chunk that is part of the partition.
                std::shared_ptr<array_info> row_number_chunk =
                    row_number_chunks[chunk_idx];
                row_number_chunk->pin();
                size_t *row_number_data =
                    row_number_chunk->data1<bodo_array_type::NUMPY, size_t>();
                // If this is the last chunk in the partition, go up to (but
                // including) first_partition_row_end. Otherwise, loop over all
                // rows in the chunk and update all of them.
                size_t rows_in_partition =
                    (chunk_idx == first_partition_chunk_end)
                        ? first_partition_row_end
                        : (row_number_chunk->length - 1);
                for (size_t row = 0; row <= rows_in_partition; row++) {
                    row_number_data[row] += row_number_offset;
                }
                row_number_chunk->unpin();
            }
        }
    }

    std::shared_ptr<array_info> ProduceOutputBatch(size_t chunk_idx) final {
        // Produce the requested section from row_number_chunks.
        return row_number_chunks[chunk_idx];
    }

   private:
    // The last row_number value from the current rank, which is to be
    // passed onto subsequent ranks.
    size_t row_number = 1;

    // The chunks of data used to store the row_number data on the
    // current rank, which becomes the output data.
    std::vector<std::shared_ptr<array_info>> row_number_chunks;
};

/**
 * @brief Copies over valuess from a singleton array into a range of rows
 * from a numeric array. Does not copy over nulls, and if the output
 * is nullable it flips all the null bits to true.
 * @param[out] arr: the array that the values are written into. Must be
 * a numeric array.
 * @param[in] value_arr: the singleton array containing the value to write into
 * arr. Must be a numeric array with the same dtype as arr.
 * @param[in] start: the begining of the rows off arr that are overwritten.
 * @param[in] end: the ending of the rows off arr that are overwritten.
 */
void fill_numeric_array_with_value(const std::shared_ptr<array_info> &arr,
                                   const std::shared_ptr<array_info> &value_arr,
                                   size_t start, size_t end) {
    assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
           arr->arr_type == bodo_array_type::NUMPY);
    assert(value_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
           value_arr->arr_type == bodo_array_type::NUMPY);
    assert(arr->dtype == value_arr->dtype);
#define FILL_DTYPE_CASE(dtype)                                                 \
    case dtype: {                                                              \
        using T = typename dtype_to_type<dtype>::type;                         \
        T val = value_arr->data1<bodo_array_type::NULLABLE_INT_BOOL, T>()[0];  \
        T *write_buffer = arr->data1<bodo_array_type::NULLABLE_INT_BOOL, T>(); \
        std::fill(write_buffer + start, write_buffer + end, val);              \
        break;                                                                 \
    }
    switch (arr->dtype) {
        FILL_DTYPE_CASE(Bodo_CTypes::INT8);
        FILL_DTYPE_CASE(Bodo_CTypes::INT16);
        FILL_DTYPE_CASE(Bodo_CTypes::INT32);
        FILL_DTYPE_CASE(Bodo_CTypes::INT64);
        FILL_DTYPE_CASE(Bodo_CTypes::UINT8);
        FILL_DTYPE_CASE(Bodo_CTypes::UINT16);
        FILL_DTYPE_CASE(Bodo_CTypes::UINT32);
        FILL_DTYPE_CASE(Bodo_CTypes::UINT64);
        FILL_DTYPE_CASE(Bodo_CTypes::INT128);
        FILL_DTYPE_CASE(Bodo_CTypes::DECIMAL);
        FILL_DTYPE_CASE(Bodo_CTypes::FLOAT32);
        FILL_DTYPE_CASE(Bodo_CTypes::FLOAT64);
        FILL_DTYPE_CASE(Bodo_CTypes::DATE);
        FILL_DTYPE_CASE(Bodo_CTypes::DATETIME);
        FILL_DTYPE_CASE(Bodo_CTypes::TIME);
        FILL_DTYPE_CASE(Bodo_CTypes::TIMEDELTA);
        default: {
            throw std::runtime_error(
                "fill_numeric_array_with_value: unsupported dtype " +
                (GetDtype_as_string(arr->dtype)));
        }
    }
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        for (size_t row = start; row < end; row++) {
            arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(row, true);
        }
    }
}

template <bodo_array_type::arr_type_enum OrderByArrType>
    requires(valid_window_array_type<OrderByArrType>)
class SimpleAggregationWindowCalculator
    : public BaseWindowCalculator<OrderByArrType> {
   public:
    SimpleAggregationWindowCalculator(
        std::shared_ptr<bodo::Schema> _schema,
        std::vector<std::shared_ptr<table_info>> _chunks,
        std::vector<int32_t> _order_col_indices,
        std::vector<int32_t> _input_col_indices,
        std::vector<std::shared_ptr<DictionaryBuilder>> _builders,
        bool _is_empty, Bodo_FTypes::FTypeEnum _ftype,
        bodo::IBufferPool *const _pool,
        std::shared_ptr<::arrow::MemoryManager> _mm)
        : BaseWindowCalculator<OrderByArrType>(
              _schema, _chunks, _order_col_indices, _input_col_indices,
              _builders, _is_empty, _pool, _mm),
          ftype(_ftype) {
        // Identify the index of the column from _chunks that is the target
        // of the aggregation function.
        if (this->input_col_indices.size() == 1) {
            agg_column = this->input_col_indices[0];
        } else {
            throw std::runtime_error(
                "SimpleAggregationWindowCalculator: expected 1 input column, "
                "received " +
                std::to_string(this->input_col_indices.size()));
        }

        // Identify the corresponding output dtype
        std::unique_ptr<bodo::DataType> dtype =
            this->schema->column_types[agg_column]->copy();
        std::tie(this->out_arr_type, this->out_dtype) =
            get_groupby_output_dtype(this->ftype, dtype->array_type,
                                     dtype->c_type);

        // For each input chunk, create a corresponding output chunk with the
        // same number of rows to store the aggregation values.
        for (size_t chunk_idx = 0; chunk_idx < this->chunks.size();
             chunk_idx++) {
            size_t n_rows = this->chunks[chunk_idx]->nrows();
            std::shared_ptr<array_info> out_chunk = alloc_array_top_level(
                n_rows, 0, 0, out_arr_type, out_dtype, -1, 0, 0, false, false,
                false, this->pool, this->mm);
            aggfunc_output_initialize(out_chunk, ftype, true);
            out_chunk->unpin();
            agg_chunks.push_back(out_chunk);
        }
    }

    void ComputePartition(size_t partition_start_chunk,
                          size_t partition_start_row,
                          size_t partition_end_chunk, size_t partition_end_row,
                          bool is_last) {
        if (this->is_empty) {
            return;
        }

        // Have a singleton array to store the aggregation value from the entire
        // partition.
        std::shared_ptr<array_info> curr_agg =
            alloc_array_top_level(1, 0, 0, this->out_arr_type, out_dtype, -1, 0,
                                  0, false, false, false, this->pool, this->mm);
        aggfunc_output_initialize(curr_agg, ftype, true);

        // Iterate across all of the chunks from partition_start_chunk to
        // partition_end_chunk, inclusive, and aggregate accordingly.
        for (size_t chunk_idx = partition_start_chunk;
             chunk_idx <= partition_end_chunk; chunk_idx++) {
            std::shared_ptr<array_info> in_chunk =
                this->chunks[chunk_idx]->columns[agg_column];
            std::shared_ptr<array_info> agg_chunk = agg_chunks[chunk_idx];
            in_chunk->pin();
            agg_chunk->pin();

            // Identify the range of rows to iterate across. If we are in
            // the first chunk, start at partition_start_row. If we are in the
            // last chunk, end on partition_end_row.
            size_t last_row = agg_chunk->length - 1;
            size_t start_row =
                (chunk_idx == partition_start_chunk) ? partition_start_row : 0;
            size_t end_row = (chunk_idx == partition_end_chunk)
                                 ? partition_end_row
                                 : last_row;
            std::shared_ptr<array_info> rows_to_aggregate;

            // If we are iterating across the entire chunk, we call the groupby
            // kernel for the aggregation on the entire chunk.
            if (start_row == 0 && end_row == last_row) {
                rows_to_aggregate = in_chunk;
            } else {
                std::vector<int64_t> idxs;
                for (size_t row = start_row; row <= end_row; row++) {
                    idxs.push_back(static_cast<int64_t>(row));
                }
                rows_to_aggregate = RetrieveArray_SingleColumn(in_chunk, idxs);
            }

            // Create a dummy group info mapping everything to row 0;
            grouping_info dummy_grp_info;
            dummy_grp_info.num_groups = 1;
            dummy_grp_info.row_to_group.resize(rows_to_aggregate->length, 0);
            std::vector<std::shared_ptr<array_info>> dummy_aux_cols;
            do_apply_to_column(rows_to_aggregate, curr_agg, dummy_aux_cols,
                               dummy_grp_info, ftype);

            in_chunk->unpin();
            agg_chunk->unpin();
        }

        // Iterate across the same chunks to update the corresponding rows to
        // use the aggregate value calculated in the previous loop. If the
        // aggregation value is null, we do nothing since the output should have
        // been pre-allocated as NULL if this was possible.
        if (curr_agg->arr_type == bodo_array_type::NUMPY ||
            curr_agg->get_null_bit(0)) {
            for (size_t chunk_idx = partition_start_chunk;
                 chunk_idx <= partition_end_chunk; chunk_idx++) {
                std::shared_ptr<array_info> agg_chunk = agg_chunks[chunk_idx];
                agg_chunk->pin();

                // Identify the range of rows to iterate across. If we are in
                // the first chunk, start at partition_start_row. If we are in
                // the last chunk, end on partition_end_row.
                size_t last_row = agg_chunk->length - 1;
                size_t start_row = (chunk_idx == partition_start_chunk)
                                       ? partition_start_row
                                       : 0;
                size_t end_row = (chunk_idx == partition_end_chunk)
                                     ? partition_end_row
                                     : last_row;

                fill_numeric_array_with_value(agg_chunk, curr_agg, start_row,
                                              end_row + 1);

                agg_chunk->unpin();
            }
        }

        // If this is the first partition, store the current aggregate value for
        // communication across ranks.
        std::vector<int64_t> idxs(1, 0);
        if (partition_start_chunk == 0 && partition_start_row == 0) {
            first_agg = RetrieveArray_SingleColumn(curr_agg, idxs);
        }
        // If this is the last partition, store the current aggregate value for
        // communication across ranks.
        if (is_last) {
            last_agg = RetrieveArray_SingleColumn(curr_agg, idxs);
        }
    }

    virtual size_t NumAuxillaryColumns() { return 1; }

    virtual void GetFirstRowInfo(
        std::vector<std::shared_ptr<array_info>> &out_cols) {
        if (this->is_empty) {
            // If empty, just create an empty array witht he correct dtype.
            out_cols.push_back(alloc_array_top_level(
                0, 0, 0, this->out_arr_type, out_dtype, -1, 0, 0, false, false,
                false, this->pool, this->mm));
            return;
        }
        out_cols.push_back(first_agg);
    }

    virtual void GetLastRowInfo(
        std::vector<std::shared_ptr<array_info>> &out_cols) {
        if (this->is_empty) {
            GetFirstRowInfo(out_cols);
        } else {
            out_cols.push_back(last_agg);
        }
    }

    /**
     * Arguments used by this implementation:
     * - first_boundary_info: used to send agg values from following ranks
     * - last_boundary_info: used to send agg values from previous ranks
     * - rank_idx: used to determine which rows in first/last boundary info are
     * the current rank.
     * - func_offset: used to infer which column of first/last boundary info
     * contains the agg values from previous/following ranks.
     * - start_partition: used as the start of previous ranks to aggregate.
     * - first_partition_chunk_end: used to determine up to which chunk to
     * update with the aggregation values from previous ranks.
     * - first_partition_row_end: used to determine up to which row in
     *   first_partition_chunk_end to update.
     * - end_partition: used as the end of following ranks to aggregate.
     * - last_partition_chunk_start: used to determine starting from which chunk
     * to update with the aggregation values from following ranks.
     * - last_partition_row_start: used to determine starting from which row in
     *   last_partition_chunk_start to update.
     */
    void UpdateBoundaryInfo(std::shared_ptr<table_info> &first_boundary_info,
                            std::shared_ptr<table_info> &last_boundary_info,
                            size_t rank_idx, size_t func_offset,
                            size_t start_partition, size_t start_order,
                            size_t end_partition, size_t end_order,
                            size_t first_partition_chunk_end,
                            size_t first_partition_row_end,
                            size_t last_partition_chunk_start,
                            size_t last_partition_row_start) {
        if (this->is_empty) {
            return;
        }

        std::shared_ptr<array_info> &first_aggs =
            first_boundary_info->columns[func_offset];
        std::shared_ptr<array_info> &last_aggs =
            last_boundary_info->columns[func_offset];

        // Fetch the aggregation values from all preceding ranks that end with
        // the same partition as the start of the current rank.
        std::vector<int64_t> last_part_rank_idxs;
        for (size_t rank = start_partition; rank < rank_idx; rank++) {
            last_part_rank_idxs.push_back(rank);
        }
        std::shared_ptr<array_info> preceding_aggs =
            RetrieveArray_SingleColumn(last_aggs, last_part_rank_idxs);

        // Fetch the aggregation values from all following ranks that start with
        // the same partition as the end of the current rank.
        std::vector<int64_t> first_part_rank_idxs;
        for (size_t rank = rank_idx + 1; rank <= end_partition; rank++) {
            first_part_rank_idxs.push_back(rank);
        }
        std::shared_ptr<array_info> following_aggs =
            RetrieveArray_SingleColumn(first_aggs, first_part_rank_idxs);

        // Create a dummy group info mapping everything to row 0
        grouping_info dummy_grp_info;
        dummy_grp_info.num_groups = 1;
        uint64_t n_rows = 1ULL;
        n_rows = std::max(preceding_aggs->length, n_rows);
        n_rows = std::max(following_aggs->length, n_rows);
        dummy_grp_info.row_to_group.resize(n_rows, 0);
        std::vector<std::shared_ptr<array_info>> dummy_aux_cols;

        // True if the current rank is all one partition
        if (last_partition_chunk_start == 0 && last_partition_row_start == 0) {
            std::shared_ptr<array_info> partition_agg = alloc_array_top_level(
                1, 0, 0, this->out_arr_type, out_dtype, -1, 0, 0, false, false,
                false, this->pool, this->mm);
            aggfunc_output_initialize(partition_agg, ftype, true);

            // Aggregate the values to get the entire aggregate of the current
            // partition.
            do_apply_to_column(preceding_aggs, partition_agg, dummy_aux_cols,
                               dummy_grp_info, ftype);
            do_apply_to_column(first_agg, partition_agg, dummy_aux_cols,
                               dummy_grp_info, ftype);
            do_apply_to_column(following_aggs, partition_agg, dummy_aux_cols,
                               dummy_grp_info, ftype);

            // If the result is non-null, override every row in the current
            // chunk with that value.
            if (partition_agg->arr_type == bodo_array_type::NUMPY ||
                partition_agg->get_null_bit(0)) {
                for (size_t chunk_idx = 0; chunk_idx < agg_chunks.size();
                     chunk_idx++) {
                    std::shared_ptr<array_info> agg_chunk =
                        agg_chunks[chunk_idx];
                    agg_chunk->pin();
                    fill_numeric_array_with_value(agg_chunk, partition_agg, 0,
                                                  agg_chunk->length);
                    agg_chunk->unpin();
                }
            }
        } else {
            // Otherwise, we need to deal with the first partition & last
            // partition seperately

            if (preceding_aggs->length > 0) {
                // Get the aggregate of the first partition on the current rank
                // plus all preceding ranks that end in the same partition.
                std::shared_ptr<array_info> first_part_agg =
                    alloc_array_top_level(1, 0, 0, this->out_arr_type,
                                          out_dtype, -1, 0, 0, false, false,
                                          false, this->pool, this->mm);
                aggfunc_output_initialize(first_part_agg, this->ftype, true);
                do_apply_to_column(preceding_aggs, first_part_agg,
                                   dummy_aux_cols, dummy_grp_info, ftype);
                do_apply_to_column(first_agg, first_part_agg, dummy_aux_cols,
                                   dummy_grp_info, ftype);

                // If the result is non-null, override every row in the first
                // partition with that value.
                if (first_part_agg->arr_type == bodo_array_type::NUMPY ||
                    first_part_agg->get_null_bit(0)) {
                    for (size_t chunk_idx = 0;
                         chunk_idx <= first_partition_chunk_end; chunk_idx++) {
                        std::shared_ptr<array_info> agg_chunk =
                            agg_chunks[chunk_idx];
                        agg_chunk->pin();
                        size_t last_row =
                            (chunk_idx == first_partition_chunk_end)
                                ? (first_partition_row_end + 1)
                                : (agg_chunk->length);
                        fill_numeric_array_with_value(agg_chunk, first_part_agg,
                                                      0, last_row);
                        agg_chunk->unpin();
                    }
                }
            }

            if (following_aggs->length > 0) {
                // Get the aggregate of the last partition on the current rank
                // plus all following ranks that start with the same partition.
                std::shared_ptr<array_info> last_part_agg =
                    alloc_array_top_level(1, 0, 0, this->out_arr_type,
                                          out_dtype, -1, 0, 0, false, false,
                                          false, this->pool, this->mm);
                aggfunc_output_initialize(last_part_agg, ftype, true);
                do_apply_to_column(last_agg, last_part_agg, dummy_aux_cols,
                                   dummy_grp_info, ftype);
                do_apply_to_column(following_aggs, last_part_agg,
                                   dummy_aux_cols, dummy_grp_info, ftype);

                // If the result is non-null, override every row in the last
                // partition with that value.
                if (last_part_agg->arr_type == bodo_array_type::NUMPY ||
                    last_part_agg->get_null_bit(0)) {
                    for (size_t chunk_idx = last_partition_chunk_start;
                         chunk_idx < agg_chunks.size(); chunk_idx++) {
                        std::shared_ptr<array_info> agg_chunk =
                            agg_chunks[chunk_idx];
                        agg_chunk->pin();
                        size_t first_row =
                            (chunk_idx == last_partition_chunk_start)
                                ? last_partition_row_start
                                : 0;
                        fill_numeric_array_with_value(agg_chunk, last_part_agg,
                                                      first_row,
                                                      agg_chunk->length);
                        agg_chunk->unpin();
                    }
                }
            }
        }
    }

    virtual std::shared_ptr<array_info> ProduceOutputBatch(size_t chunk_idx) {
        // Produce the requested section from agg_cunks.
        return agg_chunks[chunk_idx];
    }

   private:
    // Which type of aggregation is being done.
    Bodo_FTypes::FTypeEnum ftype;

    // The index of the column that is being aggregated
    int32_t agg_column;

    // The array/data type of the final array after being aggregated
    bodo_array_type::arr_type_enum out_arr_type;
    Bodo_CTypes::CTypeEnum out_dtype;

    // The first aggregation value from the current rank, which is to be
    // passed onto previous ranks.
    std::shared_ptr<array_info> first_agg;

    // The last aggregation value from the current rank, which is to be
    // passed onto subsequent ranks.
    std::shared_ptr<array_info> last_agg;

    // The chunks of data used to store the aggregated data on the
    // current rank, which becomes the output data.
    std::vector<std::shared_ptr<array_info>> agg_chunks;
};

/**
 * The orchestration class that handles a collection of window
 * functions being computed together.
 *
 * @tparam PartitionByArrType A single array type for the partition
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @tparam OrderByArrType A single array type for the order
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType>
    requires(valid_window_array_type<PartitionByArrType> &&
             valid_window_array_type<OrderByArrType>)
class WindowCollectionComputer {
   public:
    /**
     * @brief Create a new WindowCollectionComputer that orchestrates
     * the window function computation for 1 or more window functions.
     *
     * @param[in] _chunks: the vector of table_info representing
     * the sorted table spread out across multiple chunks.
     * @param[in] _partition_col_indices: the indices of columns
     * in _chunks that correspond to partition columns.
     * @param[in] _order_col_indices: the indices of columns
     * in _chunks that correspond to order columns.
     * @param[in] _keep_indices: the indices of columns in _chunks
     * that should be kept in the final output.
     * @param[in] input_col_indices: a vector of vectors where
     * each inner vector contains the indices of columns in
     * _chunks that are associated with that window function.
     * @param[in] window_funcs: the ftypes for the window functions
     * being calculated.
     * @param[in] _builders: the dictionary builders used for _chunks.
     * These should be used to ensure unification of columns.
     * @param[in] _is_parallel: is the computation happening in parallel.
     */
    WindowCollectionComputer(
        std::shared_ptr<bodo::Schema> _schema,
        std::vector<std::shared_ptr<table_info>> _chunks,
        std::vector<int32_t> _partition_col_indices,
        std::vector<int32_t> _order_col_indices,
        std::vector<int32_t> _keep_indices,
        const std::vector<std::vector<int32_t>> &input_col_indices,
        const std::vector<int32_t> &window_funcs,
        std::vector<std::shared_ptr<DictionaryBuilder>> _builders,
        bool _is_parallel, bodo::IBufferPool *const _pool,
        std::shared_ptr<::arrow::MemoryManager> _mm)
        : pool(_pool),
          mm(_mm),
          schema(_schema),
          chunks(_chunks),
          partition_col_indices(_partition_col_indices),
          order_col_indices(_order_col_indices),
          keep_indices(_keep_indices),
          builders(_builders),
          is_parallel(_is_parallel) {
        // Verify whether the current rank is all-empty.
        for (auto &it : this->chunks) {
            if (it->nrows() > 0) {
                this->is_empty = false;
                break;
            }
        }

        this->first_row = alloc_table(schema, pool, mm);
        this->last_row = alloc_table(schema, pool, mm);

        // For each window function, create the corresponding computer class.
        for (size_t func_idx = 0; func_idx < window_funcs.size(); func_idx++) {
            BaseWindowCalculator<OrderByArrType> *calculator = nullptr;
            switch (window_funcs[func_idx]) {
                case Bodo_FTypes::row_number: {
                    calculator = new RowNumberWindowCalculator<OrderByArrType>(
                        _schema, _chunks, _order_col_indices,
                        input_col_indices[func_idx], _builders, this->is_empty,
                        _pool, _mm);
                    break;
                }
                case Bodo_FTypes::sum: {
                    calculator =
                        new SimpleAggregationWindowCalculator<OrderByArrType>(
                            _schema, _chunks, _order_col_indices,
                            input_col_indices[func_idx], _builders,
                            this->is_empty,
                            (Bodo_FTypes::FTypeEnum)window_funcs[func_idx],
                            _pool, _mm);
                    break;
                }
                default: {
                    throw std::runtime_error(
                        "WindowCollectionComputer: unsupported window "
                        "function " +
                        get_name_for_Bodo_FTypes(window_funcs[func_idx]));
                }
            }
            calculators.push_back(
                std::shared_ptr<BaseWindowCalculator<OrderByArrType>>(
                    calculator));
        }
    }

    ~WindowCollectionComputer() = default;

    /**
     * @brief Invokes each calculator's ComputePartition when the boundary
     * of a partition has been found.
     *
     * @param[in] partition_start_chunk: the chunk index of the start of the
     * partition (inclusive).
     * @param[in] partition_start_row: the row index of the start of the
     * partition within partition_start_chunk (inclusive).
     * @param[in] partition_end_chunk_idx: the chunk index of the end of the
     * partition (inclusive).
     * @param[in] partition_end_row: the row index of the end of the partition
     * within partition_end_chunk_idx (inclusive).
     */
    void DispatchComputePartition(size_t partition_start_chunk,
                                  size_t partition_start_row,
                                  size_t partition_end_chunk,
                                  size_t partition_end_row) {
        bool is_last =
            this->is_empty ||
            ((partition_end_chunk == (chunks.size() - 1)) &&
             (partition_end_row == (chunks[partition_end_chunk]->nrows() - 1)));
        for (auto &it : calculators) {
            it->ComputePartition(partition_start_chunk, partition_start_row,
                                 partition_end_chunk, partition_end_row,
                                 is_last);
        }
        if (first_partition_chunk_end == -1) {
            first_partition_chunk_end = partition_end_chunk;
            first_partition_row_end = partition_end_row;
        }
        if (is_last) {
            last_partition_chunk_start = partition_start_chunk;
            last_partition_row_start = partition_start_row;
        }
    }

    /**
     * @brief Performs the local computations for the window functions
     * by having the collection computer iterate across rows of chunks
     * until it finds partition cutoffs, then having each computer
     * class within update their local state accordingly.
     */
    void ConsumeBatches() {
        // Special case: if there is no input data, just invoke the
        // partition computations with the corresponding indicators so
        // each calculator can set up its internal state.
        if (is_empty) {
            DispatchComputePartition(0, 0, 0, 0);
            return;
        }

        size_t n_chunks = chunks.size();

        // Special case: if there are no partition columns, invoke the partition
        // computations on the entire dataset.
        if (partition_col_indices.size() == 0) {
            size_t last_chunk_rows = chunks[n_chunks - 1]->nrows();
            DispatchComputePartition(0, 0, n_chunks - 1, last_chunk_rows - 1);
            return;
        }

        size_t partition_start_chunk = 0;
        size_t partition_start_row = 0;

        // Keep track of the current chunk and the chunk before it.
        std::shared_ptr<table_info> prev_chunk = nullptr;
        std::shared_ptr<table_info> curr_chunk = nullptr;

        // Loop over all of the chunks
        for (size_t chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++) {
            curr_chunk = chunks[chunk_idx];
            curr_chunk->pin();
            std::vector<std::shared_ptr<array_info>> curr_partition_cols;
            fetch_columns(curr_chunk, partition_col_indices,
                          curr_partition_cols);

            // If this is the first chunk, store its first row
            if (chunk_idx == 0) {
                std::vector<int64_t> idxs(1, 0);
                first_row = RetrieveTable(curr_chunk, idxs);
            }

            // Unless we are in the first chunk, check to see if we have
            // crossed into a new partition.
            if (chunk_idx > 0) {
                std::vector<std::shared_ptr<array_info>> prev_partition_cols;
                fetch_columns(prev_chunk, partition_col_indices,
                              prev_partition_cols);
                size_t last_row_of_prev = prev_chunk->nrows() - 1;
                bool new_partition =
                    distinct_from_other_row<PartitionByArrType>(
                        prev_partition_cols, last_row_of_prev,
                        curr_partition_cols, 0);
                prev_chunk->unpin();
                if (new_partition) {
                    curr_chunk->unpin();
                    DispatchComputePartition(partition_start_chunk,
                                             partition_start_row, chunk_idx - 1,
                                             last_row_of_prev);
                    partition_start_chunk = chunk_idx;
                    partition_start_row = 0;
                    curr_chunk->pin();
                }
            }
            // Update prev_chunk to the current chunk since we no longer need
            // to pay attention to the previous chunk.
            prev_chunk = curr_chunk;

            // Iterate across the remaining rows of the current chunk to
            // identify if the partition changes.
            size_t n_rows_in_current = curr_chunk->nrows();
            for (size_t row = 1; row < n_rows_in_current; row++) {
                curr_chunk->unpin();

                if (distinct_from_other_row<PartitionByArrType>(
                        curr_partition_cols, row - 1, curr_partition_cols,
                        row)) {
                    DispatchComputePartition(partition_start_chunk,
                                             partition_start_row, chunk_idx,
                                             row - 1);
                    partition_start_chunk = chunk_idx;
                    partition_start_row = row;
                }

                curr_chunk->pin();
            }
        }

        // Since we have reached the last chunk, store its last row.
        size_t final_row = curr_chunk->nrows() - 1;
        std::vector<int64_t> idxs(1, (int64_t)final_row);
        last_row = RetrieveTable(curr_chunk, idxs);
        curr_chunk->unpin();

        // Call the dispatcher one more time on
        // the remaining rows of the last chunk.
        DispatchComputePartition(partition_start_chunk, partition_start_row,
                                 n_chunks - 1,
                                 chunks[n_chunks - 1]->nrows() - 1);
    }

    /**
     * @brief Performs the parallel computations for the window functions
     * by having each window function communicate information to other
     * via a shared state, then having each rank receive the shared
     * information and having each computer class within update their
     * local state accordingly.
     */
    void CommunicateBoundary() {
        if (is_parallel) {
            int _myrank, _num_ranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &_myrank);
            MPI_Comm_size(MPI_COMM_WORLD, &_num_ranks);
            auto myrank = static_cast<size_t>(_myrank);
            size_t func_offset = first_row->columns.size();

            // For each calculator, append its communicaiton info to the
            // single-row tables for the first & last row of the rank.
            for (auto &it : calculators) {
                it->GetFirstRowInfo(first_row->columns);
                it->GetLastRowInfo(last_row->columns);
            }
            // Calculate the index of the row in the first/last tables
            // corresponding to the current rank.
            int64_t send_rows = static_cast<int64_t>(!is_empty);
            int64_t _rank_idx = 0;
            MPI_Exscan(&send_rows, &_rank_idx, 1, MPI_LONG_LONG_INT, MPI_SUM,
                       MPI_COMM_WORLD);
            auto rank_idx = static_cast<size_t>(_rank_idx);

            // Communicate the first/last row info across all ranks.
            std::shared_ptr<table_info> all_first_rows = gather_table(
                first_row, first_row->columns.size(), true, is_parallel);
            std::shared_ptr<table_info> all_last_rows = gather_table(
                last_row, last_row->columns.size(), true, is_parallel);

            // Skip the remaining steps if the current rank is empty.
            if (is_empty) {
                return;
            }

            std::vector<std::shared_ptr<array_info>> first_partition_cols;
            std::vector<std::shared_ptr<array_info>> first_order_cols;
            std::vector<std::shared_ptr<array_info>> last_partition_cols;
            std::vector<std::shared_ptr<array_info>> last_order_cols;

            fetch_columns(all_first_rows, partition_col_indices,
                          first_partition_cols);
            fetch_columns(all_first_rows, order_col_indices, first_order_cols);
            fetch_columns(all_last_rows, partition_col_indices,
                          last_partition_cols);
            fetch_columns(all_last_rows, order_col_indices, last_order_cols);

            // Calculate the ranks that have the same partition/order values as
            // the first/last partition on the current rank.
            size_t start_partition = rank_idx;
            size_t start_order = rank_idx;
            size_t end_partition = rank_idx;
            size_t end_order = rank_idx;

            // Work forward from the first rank to find the begining of the
            // ranks that end in the same partition as the current rank starts
            // with.
            for (size_t rank = 0; rank < rank_idx; rank++) {
                if (!distinct_from_other_row<PartitionByArrType>(
                        last_partition_cols, rank, first_partition_cols,
                        rank_idx)) {
                    start_partition = rank;
                    break;
                }
            }
            // Work forward from start_partition to find the begining of the
            // ranks that end in the same partition & orderby as the current
            // rank starts with.
            for (size_t rank = start_partition; rank < rank_idx; rank++) {
                if (!distinct_from_other_row<OrderByArrType>(
                        last_order_cols, rank, first_order_cols, rank_idx)) {
                    start_order = rank;
                    break;
                }
            }

            // Work backward from the last rank to find the end of the ranks
            // that start with the same partition that the current rank ends
            // with.
            size_t last_relevant_rank =
                std::max(all_first_rows->nrows(), (uint64_t)1);
            for (size_t rank = last_relevant_rank - 1; rank > rank_idx;
                 rank--) {
                if (!distinct_from_other_row<PartitionByArrType>(
                        first_partition_cols, rank, last_partition_cols,
                        rank_idx)) {
                    end_partition = rank;
                    break;
                }
            }
            // Work backward from end_partition to find the end of the ranks
            // that start with the same partition & orderby that the current
            // rank ends with.
            for (size_t rank = end_partition; rank > myrank; rank--) {
                if (!distinct_from_other_row<OrderByArrType>(
                        first_order_cols, rank, last_order_cols, myrank)) {
                    end_order = rank;
                    break;
                }
            }

            // For each calculator, invoke the logic to update its data
            // based on the communicated information.
            for (auto &it : calculators) {
                it->UpdateBoundaryInfo(
                    all_first_rows, all_last_rows, rank_idx, func_offset,
                    start_partition, start_order, end_partition, end_order,
                    static_cast<size_t>(first_partition_chunk_end),
                    static_cast<size_t>(first_partition_row_end),
                    static_cast<size_t>(last_partition_chunk_start),
                    static_cast<size_t>(last_partition_row_start));
                func_offset += it->NumAuxillaryColumns();
            }
        }
    }

    /**
     * @brief Collects the chunks of the output data starting with
     * all of the input columns that are to be kept, followed by
     * all of the window function output columns. Places each combined
     * output chunk into out_chunks.
     */
    void EmitBatches(std::vector<std::shared_ptr<table_info>> &out_chunks) {
        size_t n_chunks = chunks.size();
        for (size_t chunk_out_idx = 0; chunk_out_idx < n_chunks;
             chunk_out_idx++) {
            std::shared_ptr<table_info> chunk = chunks[chunk_out_idx];
            std::vector<std::shared_ptr<array_info>> out_cols;
            fetch_columns(chunk, keep_indices, out_cols);
            for (auto &it : calculators) {
                out_cols.push_back(it->ProduceOutputBatch(chunk_out_idx));
            }
            std::shared_ptr<table_info> out_chunk =
                std::make_shared<table_info>(table_info(out_cols));
            out_chunk->pin();
            out_chunks.push_back(out_chunk);
        }
    }

   private:
    bodo::IBufferPool *const pool;
    std::shared_ptr<::arrow::MemoryManager> mm;

    // The schema of the input data
    std::shared_ptr<bodo::Schema> schema;

    // The vector of sorted data
    std::vector<std::shared_ptr<table_info>> chunks;

    // The columns indices in chunks that are used for partition values.
    std::vector<int32_t> partition_col_indices;

    // The columns indices in chunks that are used for orderby values.
    std::vector<int32_t> order_col_indices;

    // The columns indices in chunks that are to be copied over into the output.
    std::vector<int32_t> keep_indices;

    // The dictionary builders used for unification, mapping 1:1 to columns in
    // chunks.
    std::vector<std::shared_ptr<DictionaryBuilder>> builders;

    // Whether the computation is being done in parallel.
    bool is_parallel;

    // Whether this rank has all-empty input data.
    bool is_empty = true;

    // The objects being used to compute each individual window function.
    std::vector<std::shared_ptr<BaseWindowCalculator<OrderByArrType>>>
        calculators;

    // The index of the last chunk in chunks belonging to the first partition.
    int64_t first_partition_chunk_end = -1;

    // The index of the last row in chunks[first_partition_chunk_end] belonging
    // to the first partition.
    int64_t first_partition_row_end = -1;

    // The index of the first chunk in chunks belonging to the last partition.
    int64_t last_partition_chunk_start = -1;

    // The index of the first row in chunks[last_partition_chunk_start]
    // belonging to the last partition.
    int64_t last_partition_row_start = -1;

    // The data corresponding to the first row on the current rank.
    std::shared_ptr<table_info> first_row = nullptr;

    // The data corresponding to the last row on the current rank.
    std::shared_ptr<table_info> last_row = nullptr;
};

void compute_window_functions_via_calculators(
    std::shared_ptr<bodo::Schema> schema,
    std::vector<std::shared_ptr<table_info>> in_chunks,
    std::vector<int32_t> partition_col_indices,
    std::vector<int32_t> order_col_indices, std::vector<int32_t> keep_indices,
    const std::vector<std::vector<int32_t>> &input_col_indices,
    const std::vector<int32_t> &window_funcs,
    std::vector<std::shared_ptr<DictionaryBuilder>> builders,
    bodo_array_type::arr_type_enum partition_arr_type,
    bodo_array_type::arr_type_enum order_arr_type,
    std::vector<std::shared_ptr<table_info>> &out_chunks, bool is_parallel,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
#define WINDOW_CALCULATE_PARTITION_ORDER_CASE(PartitionByArrType,             \
                                              OrderByArrType)                 \
    case OrderByArrType: {                                                    \
        auto computer = std::shared_ptr<                                      \
            WindowCollectionComputer<PartitionByArrType, OrderByArrType>>(    \
            new WindowCollectionComputer<PartitionByArrType, OrderByArrType>( \
                schema, in_chunks, partition_col_indices, order_col_indices,  \
                keep_indices, input_col_indices, window_funcs, builders,      \
                is_parallel, pool, mm));                                      \
        computer->ConsumeBatches();                                           \
        computer->CommunicateBoundary();                                      \
        computer->EmitBatches(out_chunks);                                    \
        break;                                                                \
    }

#define WINDOW_CALCULATE_PARTITION(PartitionByArrType)                       \
    case PartitionByArrType: {                                               \
        switch (order_arr_type) {                                            \
            WINDOW_CALCULATE_PARTITION_ORDER_CASE(PartitionByArrType,        \
                                                  bodo_array_type::NUMPY);   \
            WINDOW_CALCULATE_PARTITION_ORDER_CASE(                           \
                PartitionByArrType, bodo_array_type::NULLABLE_INT_BOOL);     \
            WINDOW_CALCULATE_PARTITION_ORDER_CASE(PartitionByArrType,        \
                                                  bodo_array_type::STRING);  \
            WINDOW_CALCULATE_PARTITION_ORDER_CASE(PartitionByArrType,        \
                                                  bodo_array_type::DICT);    \
            WINDOW_CALCULATE_PARTITION_ORDER_CASE(                           \
                PartitionByArrType, bodo_array_type::TIMESTAMPTZ);           \
            WINDOW_CALCULATE_PARTITION_ORDER_CASE(                           \
                PartitionByArrType, bodo_array_type::ARRAY_ITEM);            \
            WINDOW_CALCULATE_PARTITION_ORDER_CASE(PartitionByArrType,        \
                                                  bodo_array_type::STRUCT);  \
            WINDOW_CALCULATE_PARTITION_ORDER_CASE(PartitionByArrType,        \
                                                  bodo_array_type::MAP);     \
            WINDOW_CALCULATE_PARTITION_ORDER_CASE(PartitionByArrType,        \
                                                  bodo_array_type::UNKNOWN); \
            default: {                                                       \
                throw std::runtime_error(                                    \
                    "compute_window_functions_via_calculators: unsupported " \
                    "order array type " +                                    \
                    GetArrType_as_string(order_arr_type));                   \
            }                                                                \
        }                                                                    \
        break;                                                               \
    };

    switch (partition_arr_type) {
        WINDOW_CALCULATE_PARTITION(bodo_array_type::NUMPY);
        WINDOW_CALCULATE_PARTITION(bodo_array_type::NULLABLE_INT_BOOL);
        WINDOW_CALCULATE_PARTITION(bodo_array_type::STRING);
        WINDOW_CALCULATE_PARTITION(bodo_array_type::DICT);
        WINDOW_CALCULATE_PARTITION(bodo_array_type::TIMESTAMPTZ);
        WINDOW_CALCULATE_PARTITION(bodo_array_type::ARRAY_ITEM);
        WINDOW_CALCULATE_PARTITION(bodo_array_type::STRUCT);
        WINDOW_CALCULATE_PARTITION(bodo_array_type::MAP);
        WINDOW_CALCULATE_PARTITION(bodo_array_type::UNKNOWN);
        default: {
            throw std::runtime_error(
                "compute_window_functions_via_calculators: unsupported "
                "partition array type " +
                GetArrType_as_string(partition_arr_type));
        }
    }
}
