#pragma once
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_join.h"

/**
 * @brief allocate an empty table with provided column types
 *
 * @param arr_c_types vector of ints for column dtypes (in Bodo_CTypes format)
 * @param arr_array_types vector of ints for colmn array types (in
 * bodo_array_type format)
 * @return std::shared_ptr<table_info> allocated table
 */
std::shared_ptr<table_info> alloc_table(
    const std::vector<int8_t>& arr_c_types,
    const std::vector<int8_t>& arr_array_types);

/**
 * @brief Allocate an empty table with the same schema
 * (arr_types and dtypes) as 'table'.
 *
 * @param table Reference table
 * @return std::shared_ptr<table_info> Allocated table
 */
std::shared_ptr<table_info> alloc_table_like(
    const std::shared_ptr<table_info>& table);

class JoinPartition;

/**
 * @brief Wrapper around array_info to turn it into build buffer.
 * It allows appending elements while also providing random access, which is
 * necessary when used with a hash table. See
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1351974913/Implementation+Notes
 *
 */
struct ArrayBuildBuffer {
    // internal array with data values
    std::shared_ptr<array_info> data_array;
    // Current number of elements in the buffer
    int64_t size;
    // Total capacity for data elements (including current elements,
    // capacity>=size should always be true)
    int64_t capacity;

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @param in_arr input table with the new data element
     * @param row_ind index of data in input
     */
    void AppendRow(const std::shared_ptr<array_info>& in_arr, int64_t row_ind) {
        switch (in_arr->arr_type) {
            case bodo_array_type::NULLABLE_INT_BOOL: {
                if (in_arr->dtype == Bodo_CTypes::_BOOL) {
                    arrow::bit_util::SetBitTo(
                        (uint8_t*)data_array->data1(), size,
                        GetBit((uint8_t*)in_arr->data1(), row_ind));
                    size++;
                    data_array->length = size;
                    CHECK_ARROW_MEM(
                        data_array->buffers[0]->Resize(
                            arrow::bit_util::BytesForBits(size), false),
                        "Resize Failed!");
                    CHECK_ARROW_MEM(
                        data_array->buffers[1]->Resize(
                            arrow::bit_util::BytesForBits(size), false),
                        "Resize Failed!");
                } else {
                    uint64_t size_type = numpy_item_size[in_arr->dtype];
                    char* out_ptr = data_array->data1() + size_type * size;
                    char* in_ptr = in_arr->data1() + size_type * row_ind;
                    memcpy(out_ptr, in_ptr, size_type);
                    bool bit =
                        GetBit((uint8_t*)in_arr->null_bitmask(), row_ind);
                    SetBitTo((uint8_t*)data_array->null_bitmask(), size, bit);
                    size++;
                    data_array->length = size;
                    CHECK_ARROW_MEM(
                        data_array->buffers[0]->Resize(size * size_type, false),
                        "Resize Failed!");
                    CHECK_ARROW_MEM(
                        data_array->buffers[1]->Resize(
                            arrow::bit_util::BytesForBits(size), false),
                        "Resize Failed!");
                }
            } break;
            case bodo_array_type::STRING: {
                offset_t* curr_offsets = (offset_t*)data_array->data2();
                offset_t* in_offsets = (offset_t*)in_arr->data2();

                // append offset
                int64_t str_len = in_offsets[row_ind + 1] - in_offsets[row_ind];
                curr_offsets[size + 1] = curr_offsets[size] + str_len;

                // copy characters
                char* out_ptr = data_array->data1() + curr_offsets[size];
                char* in_ptr = in_arr->data1() + in_offsets[row_ind];
                memcpy(out_ptr, in_ptr, str_len);

                // set null bit
                bool bit = GetBit((uint8_t*)in_arr->null_bitmask(), row_ind);
                SetBitTo((uint8_t*)data_array->null_bitmask(), size, bit);

                // update size state
                size++;
                data_array->length = size;
                CHECK_ARROW_MEM(data_array->buffers[0]->Resize(
                                    data_array->n_sub_elems(), false),
                                "Resize Failed!");
                CHECK_ARROW_MEM(data_array->buffers[1]->Resize(
                                    (size + 1) * sizeof(offset_t), false),
                                "Resize Failed!");
                CHECK_ARROW_MEM(data_array->buffers[2]->Resize(
                                    arrow::bit_util::BytesForBits(size), false),
                                "Resize Failed!");
            } break;
            case bodo_array_type::NUMPY: {
                uint64_t size_type = numpy_item_size[in_arr->dtype];
                char* out_ptr = data_array->data1() + size_type * size;
                char* in_ptr = in_arr->data1() + size_type * row_ind;
                memcpy(out_ptr, in_ptr, size_type);
                size++;
                data_array->length = size;
                CHECK_ARROW_MEM(
                    data_array->buffers[0]->Resize(size * size_type, false),
                    "Resize Failed!");
            } break;
            default:
                throw std::runtime_error(
                    "invalid array type in AppendRow " +
                    GetArrType_as_string(in_arr->arr_type));
        }
    }

    /**
     * @brief Reserve enough space to potentially append all contents of input
     * array to buffer. NOTE: This requires reserving space for variable-sized
     * elements like strings and nested arrays.
     *
     * @param in_arr input array used for finding new buffer sizes to reserve
     */
    void ReserveArray(std::shared_ptr<array_info>& in_arr) {
        int64_t min_capacity = size + in_arr->length;
        switch (in_arr->arr_type) {
            case bodo_array_type::NULLABLE_INT_BOOL:
                if (min_capacity > capacity) {
                    int64_t new_capacity = std::max(min_capacity, capacity * 2);
                    if (in_arr->dtype == Bodo_CTypes::_BOOL) {
                        CHECK_ARROW_MEM(
                            data_array->buffers[0]->Reserve(
                                arrow::bit_util::BytesForBits(new_capacity)),
                            "Reserve failed!");
                        CHECK_ARROW_MEM(
                            data_array->buffers[1]->Reserve(
                                arrow::bit_util::BytesForBits(new_capacity)),
                            "Reserve failed!");
                    } else {
                        uint64_t size_type = numpy_item_size[in_arr->dtype];
                        CHECK_ARROW_MEM(data_array->buffers[0]->Reserve(
                                            new_capacity * size_type),
                                        "Reserve failed!");
                        CHECK_ARROW_MEM(
                            data_array->buffers[1]->Reserve(
                                arrow::bit_util::BytesForBits(new_capacity)),
                            "Reserve failed!");
                    }
                    capacity = new_capacity;
                }
                break;
            case bodo_array_type::STRING: {
                // update offset and null bitmap buffers
                if (min_capacity > capacity) {
                    int64_t new_capacity = std::max(min_capacity, capacity * 2);
                    CHECK_ARROW_MEM(data_array->buffers[1]->Reserve(
                                        (new_capacity + 1) * sizeof(offset_t)),
                                    "Reserve failed!");
                    CHECK_ARROW_MEM(
                        data_array->buffers[2]->Reserve(
                            arrow::bit_util::BytesForBits(new_capacity)),
                        "Reserve failed!");
                    capacity = new_capacity;
                }
                // update data buffer
                int64_t capacity_chars = data_array->buffers[0]->capacity();
                int64_t min_capacity_chars =
                    data_array->n_sub_elems() + in_arr->n_sub_elems();
                if (min_capacity_chars > capacity_chars) {
                    int64_t new_capacity_chars =
                        std::max(min_capacity_chars, capacity_chars * 2);
                    CHECK_ARROW_MEM(data_array->buffers[0]->Reserve(
                                        new_capacity_chars * sizeof(int8_t)),
                                    "Reserve failed!");
                }
            } break;
            case bodo_array_type::NUMPY: {
                uint64_t size_type = numpy_item_size[in_arr->dtype];
                if (min_capacity > capacity) {
                    int64_t new_capacity = std::max(min_capacity, capacity * 2);
                    CHECK_ARROW_MEM(data_array->buffers[0]->Reserve(
                                        new_capacity * size_type),
                                    "Reserve failed!");
                    capacity = new_capacity;
                }
            } break;
            default:
                throw std::runtime_error(
                    "invalid array type in ReserveArray " +
                    GetArrType_as_string(in_arr->arr_type));
        }
    }

    ArrayBuildBuffer(std::shared_ptr<array_info> _data_array)
        : data_array(_data_array), size(0), capacity(0) {}
};

/**
 * @brief Wrapper around table_info to turn it into build buffer.
 * It allows appending rows while also providing random access, which is
 * necessary when used with a hash table. See
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1351974913/Implementation+Notes
 *
 */
struct TableBuildBuffer {
    // internal data table with values
    std::shared_ptr<table_info> data_table;
    // buffer wrappers around arrays of data table
    std::vector<ArrayBuildBuffer> array_buffers;

    TableBuildBuffer(const std::vector<int8_t>& arr_c_types,
                     const std::vector<int8_t>& arr_array_types) {
        // allocate empty initial table with provided data types
        data_table = alloc_table(arr_c_types, arr_array_types);

        // initialize array buffer wrappers
        for (size_t i = 0; i < arr_c_types.size(); i++) {
            array_buffers.emplace_back(data_table->columns[i]);
        }
    }

    /**
     * @brief Append a row of data to the buffer, assuming
     * there is already enough space reserved (with ReserveTable).
     *
     * @param in_table input table with the new row
     * @param row_ind index of new row in input table
     */
    void AppendRow(const std::shared_ptr<table_info>& in_table,
                   int64_t row_ind) {
        for (size_t i = 0; i < in_table->ncols(); i++) {
            const std::shared_ptr<array_info>& in_arr = in_table->columns[i];
            array_buffers[i].AppendRow(in_arr, row_ind);
        }
    }

    /**
     * @brief Reserve enough space to potentially append all contents of input
     * table to buffer. NOTE: This requires reserving space for variable-sized
     * elements like strings and nested arrays.
     *
     * @param in_table input table used for finding new buffer sizes to reserve
     */
    void ReserveTable(const std::shared_ptr<table_info>& in_table) {
        for (size_t i = 0; i < in_table->ncols(); i++) {
            std::shared_ptr<array_info>& in_arr = in_table->columns[i];
            array_buffers[i].ReserveArray(in_arr);
        }
    }
};

struct HashHashJoinTable {
    /**
     * provides row hashes for join hash table (bodo::unordered_multimap)
     *
     * Input row number iRow can refer to either build or probe table.
     * If iRow >= 0 then it is in the build table at index iRow.
     * If iRow < 0 then it is in the probe table
     *    at index (-iRow - 1).
     *
     * @param iRow row number
     * @return hash of row iRow
     */
    uint32_t operator()(const int64_t iRow) const;
    JoinPartition* join_partition;
};

struct KeyEqualHashJoinTable {
    /**
     * provides row comparison for join hash table (bodo::unordered_multimap)
     *
     * Input row number iRow can refer to either build or probe table.
     * If iRow >= 0 then it is in the build table at index iRow.
     * If iRow < 0 then it is in the probe table
     *    at index (-iRow - 1).
     *
     * @param iRowA is the first row index for the comparison
     * @param iRowB is the second row index for the comparison
     * @return true if equal else false
     */
    bool operator()(const int64_t iRowA, const int64_t iRowB) const;
    JoinPartition* join_partition;
    const bool is_na_equal;
    const int64_t n_keys;
};

/**
 * @brief Holds the state of a single partition during
 * a join execution. This includes the build table buffer,
 * the hashtable (unordered_multimap), bitmap of the matches
 * in build records, etc.
 * 'top_bitmask' and 'num_top_bits' define the partition
 * itself, i.e. a record is in this partition if the top
 * 'num_top_bits' bits of its hash are 'top_bitmask'.
 *
 */
class JoinPartition {
   public:
    explicit JoinPartition(size_t num_top_bits_, uint32_t top_bitmask_,
                           const std::vector<int8_t>& build_arr_c_types,
                           const std::vector<int8_t>& build_arr_array_types,
                           const std::vector<int8_t>& probe_arr_c_types,
                           const std::vector<int8_t>& probe_arr_array_types,
                           const int64_t n_keys_, bool build_table_outer_,
                           bool probe_table_outer_)
        : build_table_buffer(build_arr_c_types, build_arr_array_types),
          build_table({}, HashHashJoinTable(this),
                      KeyEqualHashJoinTable(this, false, n_keys_)),
          num_top_bits(num_top_bits_),
          top_bitmask(top_bitmask_),
          build_table_outer(build_table_outer_) {}

    // Build state
    TableBuildBuffer build_table_buffer;  // Append only buffer.
    bodo::vector<uint32_t> build_table_join_hashes;

    bodo::unordered_multimap<int64_t, int64_t, HashHashJoinTable,
                             KeyEqualHashJoinTable>
        build_table;  // join hash table (key row number -> matching row
                      // numbers)

    // Probe state (for outer joins)
    bodo::vector<bool> build_table_matched;  // state for building output table

    // Temporary state during probe step. These will be
    // reset between iterations.
    std::shared_ptr<table_info> probe_table;
    // Join hashes corresponding to data in probe_table.
    std::shared_ptr<uint32_t[]> probe_table_hashes;

    /// @brief Get number of bits in the 'top_bitmask'.
    size_t get_num_top_bits() const { return this->num_top_bits; }

    /// @brief Get the 'top_bitmask'.
    uint32_t get_top_bitmask() const { return this->top_bitmask; }

    /// @brief Reserve space in build_table_buffer and build_table_join_hashes
    /// to add all rows from in_table.
    void ReserveBuildTable(const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Add a row from in_table  to this partition.
     * This includes populating the hash table.
     *
     * @param in_table Table from which we're adding the row.
     * @param row_ind Index of the row to add.
     * @param join_hash Join hash for the record.
     */
    void AppendBuildRow(const std::shared_ptr<table_info>& in_table,
                        int64_t row_ind, const uint32_t& join_hash);

    /**
     * @brief Add all rows from in_table to this partition.
     * This includes populating the hash table.
     *
     * @param in_table Table to insert.
     * @param join_hashes Join hashes for the table records.
     */
    void AppendBuildBatch(const std::shared_ptr<table_info>& in_table,
                          const std::shared_ptr<uint32_t[]>& join_hashes);

    /**
     * @brief Finalize the build step for this partition.
     * At this time, this just initializes the build_table_matched
     * bitmap in the build_table_outer case.
     *
     */
    void FinalizeBuild();

   private:
    const size_t num_top_bits = 0;
    const uint32_t top_bitmask = 0ULL;
    const bool build_table_outer = false;
    // Tracks the current size of the build table, i.e.
    // the number of rows from the build_table_buffer
    // that have been added to the hash table.
    int64_t curr_build_size = 0;
};

class JoinState {
   public:
    // Join properties
    const int64_t n_keys;
    cond_expr_fn_t cond_func;

    JoinState(int64_t n_keys_, cond_expr_fn_t _cond_func)
        : n_keys(n_keys_), cond_func(_cond_func) {}
};

class HashJoinState : public JoinState {
   public:
    // Join properties
    const bool build_table_outer;
    const bool probe_table_outer;

    // Partitioning information.
    // For now, we just have one partition
    // (the active one). In the future, there'll
    // be a vector of partitions.
    std::shared_ptr<JoinPartition> partition;

    // Shuffle state
    TableBuildBuffer build_shuffle_buffer;
    TableBuildBuffer probe_shuffle_buffer;

    // Dummy probe table. Useful for the build_table_outer case.
    std::shared_ptr<table_info> dummy_probe_table;

    HashJoinState(std::vector<int8_t> build_arr_c_types,
                  std::vector<int8_t> build_arr_array_types,
                  std::vector<int8_t> probe_arr_c_types,
                  std::vector<int8_t> probe_arr_array_types, int64_t n_keys_,
                  bool build_table_outer_, bool probe_table_outer_,
                  cond_expr_fn_t _cond_func)
        : JoinState(n_keys_, _cond_func),
          build_table_outer(build_table_outer_),
          probe_table_outer(probe_table_outer_),
          build_shuffle_buffer(build_arr_c_types, build_arr_array_types),
          probe_shuffle_buffer(probe_arr_c_types, probe_arr_array_types),
          dummy_probe_table(
              alloc_table(probe_arr_c_types, probe_arr_array_types)) {
        this->partition = std::make_shared<JoinPartition>(
            0, 0, build_arr_c_types, build_arr_array_types, probe_arr_c_types,
            probe_arr_array_types, n_keys_, build_table_outer_,
            probe_table_outer_);
    }
};

class NestedLoopJoinState : public JoinState {
   public:
    // Build state
    TableBuildBuffer build_table_buffer;  // Append only buffer.

    NestedLoopJoinState(const std::vector<int8_t>& build_arr_c_types,
                        const std::vector<int8_t>& build_arr_array_types,
                        cond_expr_fn_t _cond_func)
        : JoinState(
              0, _cond_func),  // NestedLoopJoin is only used when n_keys is 0
          build_table_buffer(build_arr_c_types, build_arr_array_types) {}
};

/**
 * @brief Python wrapper to consume build table batch in nested loop join
 *
 * @param join_state join state pointer
 * @param in_table build table batch
 * @param is_last is last batch
 */
void nested_loop_join_build_consume_batch_py_entry(
    NestedLoopJoinState* join_state, table_info* in_table, bool is_last,
    bool parallel);

/**
 * @brief consume probe table batch in streaming nested loop join and produce
 * output table batch Design doc:
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1373896721/Vectorized+Nested+Loop+Join+Design
 *
 * @param join_state join state pointer
 * @param in_table probe table batch
 * @param is_last is last batch
 * @param is_parallel parallel flag
 * @return std::shared_ptr<table_info> output table batch
 */
table_info* nested_loop_join_probe_consume_batch_py_entry(
    NestedLoopJoinState* join_state, table_info* in_table, bool is_last,
    bool* out_is_last, bool parallel);
