#pragma once
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_join.h"
#include "simd-block-fixed-fpp.h"

using BloomFilter = SimdBlockFilterFixed<::hashing::SimpleMixSplit>;

/**
 * @brief allocate an empty table with provided column types
 *
 * @param arr_c_types vector of ints for column dtypes (in Bodo_CTypes format)
 * @param arr_array_types vector of ints for colmun array types (in
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
struct ArrayBuildBuffer;

/**
 * @brief Similar to ArrayBuildBuffer, but for incrementally building a
 * dictionary (the string portion of a dictionary encoded array).
 * This is meant to be shared between multiple arrays.
 * e.g. In Hash Join, for DICT key columns, this dictionary is shared
 * between the build_table_buffer and probe_table_buffer of all
 * partitions, as well as the shuffle buffers for both tables.
 *
 */
struct DictionaryBuilder {
    // NOTE: dictionary state can be shared across buffers that require the same
    // dictionary hash table used for dictionary unification of new batches
    // (only for dictionary-encoded string arrays)
    std::shared_ptr<std::unordered_map<std::string, dict_indices_t, string_hash,
                                       std::equal_to<>>>
        dict_str_to_ind;
    // dictionary buffer to allow appending new dictionary values (only for
    // dictionary-encoded string arrays)
    std::shared_ptr<ArrayBuildBuffer> dict_buff;
    // hashes of dictionary elements (not actual array elements) to allow
    // consistent hashing (only for dictionary-encoded string arrays that are
    // key columns)
    std::shared_ptr<bodo::vector<uint32_t>> dict_hashes = nullptr;
    // Input is a key column of its table. This essentially decides
    // whether or not we maintain 'dict_hashes'.
    const bool is_key;

    /**
     * @brief Construct a new Dictionary Builder.
     *
     * @param dict String array to store the dictionary in. This
     * expects an empty array.
     * @param is_key_ Whether this is a key column. This essentially
     * decided whether we will compute and maintain the hashes for
     * the values in this dictionary.
     */
    DictionaryBuilder(std::shared_ptr<array_info> dict, bool is_key_)
        : is_key(is_key_) {
        this->dict_buff = std::make_shared<ArrayBuildBuffer>(dict);
        this->dict_hashes = std::make_shared<bodo::vector<uint32_t>>();
        this->dict_str_to_ind = std::make_shared<std::unordered_map<
            std::string, dict_indices_t, string_hash, std::equal_to<>>>();
    }

    /**
     * @brief Unify dictionary of input array with buffer by appending its new
     * dictionary values to buffer's dictionary and transposing input's indices.
     *
     * @param in_arr input array
     * @param is_key input is a key column of its table
     * @return std::shared_ptr<array_info> input array with its dictionary
     * replaced and indices transposed
     */
    std::shared_ptr<array_info> UnifyDictionaryArray(
        const std::shared_ptr<array_info>& in_arr);

    /**
     * @brief Get dictionary hashes
     *
     * @return std::shared_ptr<bodo::vector<uint32_t>> dictionary hashes or null
     * pointer
     */
    std::shared_ptr<bodo::vector<uint32_t>> GetDictionaryHashes();
};

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

    // Shared dictionary builder.
    // XXX Might not need this?
    std::shared_ptr<DictionaryBuilder> dict_builder = nullptr;
    // dictionary indices buffer for appending dictionary indices (only for
    // dictionary-encoded string arrays)
    std::shared_ptr<ArrayBuildBuffer> dict_indices;

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
                // Set null bit
                bool bit = GetBit((uint8_t*)in_arr->null_bitmask(), row_ind);
                SetBitTo((uint8_t*)data_array->null_bitmask(), size, bit);
                // set data value
                int64_t new_data_size = 0;
                if (in_arr->dtype == Bodo_CTypes::_BOOL) {
                    arrow::bit_util::SetBitTo(
                        (uint8_t*)data_array->data1(), size,
                        GetBit((uint8_t*)in_arr->data1(), row_ind));
                    new_data_size = arrow::bit_util::BytesForBits(size + 1);
                } else {
                    uint64_t size_type = numpy_item_size[in_arr->dtype];
                    char* out_ptr = data_array->data1() + size_type * size;
                    char* in_ptr = in_arr->data1() + size_type * row_ind;
                    memcpy(out_ptr, in_ptr, size_type);
                    new_data_size = size_type * (size + 1);
                }
                // Resize buffers
                size++;
                data_array->length = size;
                CHECK_ARROW_MEM(
                    data_array->buffers[0]->Resize(new_data_size, false),
                    "Resize Failed!");
                CHECK_ARROW_MEM(data_array->buffers[1]->Resize(
                                    arrow::bit_util::BytesForBits(size), false),
                                "Resize Failed!");
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
            case bodo_array_type::DICT: {
                if (this->data_array->child_arrays[0] !=
                    in_arr->child_arrays[0]) {
                    throw std::runtime_error(
                        "dictionary not unified in AppendRow");
                }
                this->dict_indices->AppendRow(in_arr->child_arrays[1], row_ind);
                size++;
                data_array->length = size;
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
    void ReserveArray(const std::shared_ptr<array_info>& in_arr) {
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
            case bodo_array_type::DICT: {
                if (this->data_array->child_arrays[0] !=
                    in_arr->child_arrays[0]) {
                    throw std::runtime_error(
                        "dictionary not unified in ReserveArray");
                }
                this->dict_indices->ReserveArray(in_arr->child_arrays[1]);
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

    /**
     * @brief Clear the buffers, i.e. set size to 0.
     *  Capacity is not changed and memory is not released
     */
    void Clear() {
        size = 0;
        data_array->length = 0;
        switch (data_array->arr_type) {
            case bodo_array_type::NULLABLE_INT_BOOL: {
                CHECK_ARROW_MEM(data_array->buffers[0]->Resize(0, false),
                                "Resize failed!");
                CHECK_ARROW_MEM(data_array->buffers[1]->Resize(0, false),
                                "Resize failed!");
            } break;
            case bodo_array_type::NUMPY: {
                CHECK_ARROW_MEM(data_array->buffers[0]->Resize(0, false),
                                "Resize failed!");

            } break;
            case bodo_array_type::STRING: {
                CHECK_ARROW_MEM(data_array->buffers[0]->Resize(0, false),
                                "Resize failed!");
                CHECK_ARROW_MEM(data_array->buffers[1]->Resize(0, false),
                                "Resize failed!");
                CHECK_ARROW_MEM(data_array->buffers[2]->Resize(0, false),
                                "Resize failed!");
            } break;
            case bodo_array_type::DICT: {
                this->dict_indices->Clear();
            } break;
            default: {
                throw std::runtime_error(
                    "invalid array type in Clear " +
                    GetArrType_as_string(data_array->arr_type));
            }
        }
    }
    /**
     * @brief Construct a new ArrayBuildBuffer for the provided data array.
     *
     * @param _data_array Data array that we will be appending to. This is
     * expected to be an empty array.
     * @param dict_builder If this is a dictionary encoded string array,
     * a DictBuilder must be provided that will be used as the dictionary.
     * The dictionary of the data_array (_data_array->child_arrays[0]) must
     * be the dictionary in dict_builder (_dict_builder->dict_buff->data_array).
     */
    ArrayBuildBuffer(std::shared_ptr<array_info> _data_array,
                     std::shared_ptr<DictionaryBuilder> _dict_builder = nullptr)
        : data_array(_data_array),
          size(0),
          capacity(0),
          dict_builder(_dict_builder) {
        if (_data_array->arr_type == bodo_array_type::DICT) {
            if (_dict_builder == nullptr) {
                throw std::runtime_error(
                    "ArrayBuildBuffer: dict_builder is nullptr for a "
                    "dict-encoded string array!");
            }
            if (_dict_builder->dict_buff->data_array.get() !=
                _data_array->child_arrays[0].get()) {
                throw std::runtime_error(
                    "ArrayBuildBuffer: specified dict_builder does not "
                    "match dictionary of _data_array!");
            }
            this->dict_indices = std::make_shared<ArrayBuildBuffer>(
                this->data_array->child_arrays[1]);
        }
    }
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

    // Only used for temporary objects. In particular,
    // in HashJoinState constructor, we cannot initialize
    // the shuffle buffers in the initialization list since
    // we need to build the dict_builders first. So we need
    // to provide this default constructor so that it
    // is initialized to an empty buffer by default and then
    // we can create and replace it with the actual TableBuildBuffer
    // later in the constructor.
    TableBuildBuffer() = default;

    TableBuildBuffer(
        const std::vector<int8_t>& arr_c_types,
        const std::vector<int8_t>& arr_array_types,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders) {
        // allocate empty initial table with provided data types
        data_table = alloc_table(arr_c_types, arr_array_types);

        // initialize array buffer wrappers
        for (size_t i = 0; i < arr_c_types.size(); i++) {
            if (arr_array_types[i] == bodo_array_type::DICT) {
                // Set the dictionary to the one from the dict builder:
                data_table->columns[i]->child_arrays[0] =
                    dict_builders[i]->dict_buff->data_array;
            }
            array_buffers.emplace_back(data_table->columns[i],
                                       dict_builders[i]);
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

    /**
     * @brief Clear the buffers, i.e. set size to 0.
     *  Capacity is not changed and memory is not released
     */
    void Clear() {
        for (size_t i = 0; i < array_buffers.size(); i++) {
            array_buffers[i].Clear();
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
    const uint64_t n_keys;
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
    explicit JoinPartition(
        size_t num_top_bits_, uint32_t top_bitmask_,
        const std::vector<int8_t>& build_arr_c_types,
        const std::vector<int8_t>& build_arr_array_types,
        const std::vector<int8_t>& probe_arr_c_types,
        const std::vector<int8_t>& probe_arr_array_types,
        const uint64_t n_keys_, bool build_table_outer_,
        bool probe_table_outer_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>&
            build_table_dict_builders,
        const std::vector<std::shared_ptr<DictionaryBuilder>>&
            probe_table_dict_builders)
        : build_table_buffer(build_arr_c_types, build_arr_array_types,
                             build_table_dict_builders),
          build_table({}, HashHashJoinTable(this),
                      KeyEqualHashJoinTable(this, n_keys_)),
          probe_table_buffer(probe_arr_c_types, probe_arr_array_types,
                             probe_table_dict_builders),
          num_top_bits(num_top_bits_),
          top_bitmask(top_bitmask_),
          build_table_outer(build_table_outer_),
          probe_table_outer(probe_table_outer_),
          n_keys(n_keys_) {}

    // Build state
    TableBuildBuffer build_table_buffer;  // Append only buffer.
    bodo::vector<uint32_t> build_table_join_hashes;

    bodo::unordered_multimap<int64_t, int64_t, HashHashJoinTable,
                             KeyEqualHashJoinTable>
        build_table;  // join hash table (key row number -> matching row
                      // numbers)

    // Probe state (for outer joins). Note we don't use
    // vector<bool> because we may need to do an allreduce
    // on the data directly and that can't be accessed for bool.
    bodo::vector<uint8_t>
        build_table_matched;  // state for building output table

    // Probe state (only used when this partition is inactive).
    // We don't need partitioning hashes since we should never
    // need to repartition.
    // XXX These will be converted to use chunked arrays.
    TableBuildBuffer probe_table_buffer;
    bodo::vector<uint32_t> probe_table_buffer_join_hashes;

    // Temporary state during probe step. These will be
    // reset between iterations.
    std::shared_ptr<table_info> probe_table;
    // Join hashes corresponding to data in probe_table.
    // We're using a raw pointer here so we can populate this field using the
    // vector (using .data()) or the uint32_t[] shared_ptr directly (using
    // .get())
    uint32_t* probe_table_hashes;

    /// @brief Get number of bits in the 'top_bitmask'.
    size_t get_num_top_bits() const { return this->num_top_bits; }

    /// @brief Get the 'top_bitmask'.
    uint32_t get_top_bitmask() const { return this->top_bitmask; }

    /// @brief Check if a row is part of this partition based on its
    /// partition hash.
    /// @param hash Partition hash for the row
    /// @return True if row is part of partition, False otherwise.
    inline bool is_in_partition(const uint32_t& hash);

    /// @brief Is the partition near full? This is used
    /// to determine whether this partition should be
    /// split into multiple partitions.
    inline bool is_near_full() const {
        // TODO Replace with proper implementation based
        // on buffer sizes, memory budget and Allocator statistics.
        return false;
    }

    /**
     * @brief Split the partition into 2^num_levels partitions.
     * This will produce a new set of partitions, each with their
     * new build_table_buffer and build_table_join_hashes.
     * The caller must explicitly rebuild the build_table on
     * the partition.
     *
     * @param num_levels Number of levels to split the partition. Only '1' is
     * supported at this point.
     * @return std::vector<std::shared_ptr<JoinPartition>>
     */
    std::vector<std::shared_ptr<JoinPartition>> SplitPartition(
        size_t num_levels = 1);

    /**
     * @brief Reserve space in build_table_buffer and build_table_join_hashes to
     * add all rows from in_table.
     *
     * @param in_table Table to reserve based on.
     */
    void ReserveBuildTable(const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Reserve space in probe_table_buffer and
     * probe_table_join_hashes to add all rows from in_table.
     *
     * @param in_table Table to reserve based on.
     */
    void ReserveProbeTable(const std::shared_ptr<table_info>& in_table);

    /// @brief Add rows from build_table_buffer into the
    /// hash table. This adds rows starting from curr_build_size.
    /// This is useful for rebuilding the hash table after
    /// repartitioning and for building hash tables of inactive
    /// partitions at the end of the build step (after we've
    /// seen all the data).
    void BuildHashTable();

    /**
     * @brief Add a row from in_table to this partition.
     * This includes populating the hash table.
     *
     * @tparam is_active Is this the active partition.
     * @param in_table Table from which we're adding the row.
     * @param row_ind Index of the row to add.
     * @param join_hash Join hash for the record.
     */
    template <bool is_active = false>
    void AppendBuildRow(const std::shared_ptr<table_info>& in_table,
                        int64_t row_ind, const uint32_t& join_hash);

    /**
     * @brief Add all rows from in_table to this partition.
     * This includes populating the hash table.
     *
     * @param in_table Table to insert.
     * @param join_hashes Join hashes for the table records.
     * @param partitioning_hashes Partitioning hashes for the table records.
     */
    void AppendBuildBatch(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& join_hashes,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes);

    /**
     * @brief Finalize the build step for this partition.
     * At this time, this just initializes the build_table_matched
     * bitmap in the build_table_outer case.
     *
     */
    void FinalizeBuild();

    /**
     * @brief Append a row into the probe table buffer.
     * Note that this is only used for inactive partitions
     * to buffer the inputs before we start processing them.
     *
     * @param in_table Table from which we're adding the row.
     * @param row_ind Index of the row to add.
     * @param join_hash Join hash for the record.
     */
    void AppendInactiveProbeRow(const std::shared_ptr<table_info>& in_table,
                                int64_t row_ind, const uint32_t& join_hash);

    /**
     * @brief Process the records in the probe table buffer
     * and produce the outputs for this partition.
     * Note that this is only used for inactive (non index 0)
     * partitions.
     *
     * @tparam build_table_outer
     * @tparam probe_table_outer
     * @tparam non_equi_condition
     * @param cond_func Condition function for the non-equi condition case,
     * nullptr otherwise.
     * @param build_kept_cols Which columns to generate in the output on the
     * build side.
     * @param probe_kept_cols Which columns to generate in the output on the
     * probe side.
     * @param build_needs_reduction Do the build misses need a reduction?
     * @return std::tuple<std::shared_ptr<table_info>,
     * std::shared_ptr<table_info>> (Build side table, Probe side table) of
     * output.
     */
    template <bool build_table_outer, bool probe_table_outer,
              bool non_equi_condition>
    std::tuple<std::shared_ptr<table_info>, std::shared_ptr<table_info>>
    FinalizeProbeForInactivePartition(
        cond_expr_fn_t cond_func, const std::vector<uint64_t>& build_kept_cols,
        const std::vector<uint64_t>& probe_kept_cols,
        const bool build_needs_reduction);

   private:
    const size_t num_top_bits = 0;
    const uint32_t top_bitmask = 0ULL;
    const bool build_table_outer = false;
    const bool probe_table_outer = false;
    const uint64_t n_keys;
    // Tracks the current size of the build table, i.e.
    // the number of rows from the build_table_buffer
    // that have been added to the hash table.
    int64_t curr_build_size = 0;
};

class JoinState {
   public:
    // Join properties
    const uint64_t n_keys;
    cond_expr_fn_t cond_func;
    const bool build_table_outer;
    const bool probe_table_outer;
    // Note: This isn't constant because we may change it
    // via broadcast decisions.
    bool build_parallel;
    const bool probe_parallel;
    // Has all of the input already been processed. This should be
    // updated after the last input to avoid repeating the outer
    // join output.
    bool build_input_finalized = false;
    bool probe_input_finalized = false;
    const int64_t output_batch_size;

    JoinState(uint64_t n_keys_, bool build_table_outer_,
              bool probe_table_outer_, cond_expr_fn_t cond_func_,
              bool build_parallel_, bool probe_parallel_,
              int64_t output_batch_size_)
        : n_keys(n_keys_),
          cond_func(cond_func_),
          build_table_outer(build_table_outer_),
          probe_table_outer(probe_table_outer_),
          build_parallel(build_parallel_),
          probe_parallel(probe_parallel_),
          output_batch_size(output_batch_size_) {}

    virtual ~JoinState() {}

    virtual void FinalizeBuild() { build_input_finalized = true; }

    void FinalizeProbe() { probe_input_finalized = true; }
};

class HashJoinState : public JoinState {
   public:
    // Partitioning information.
    std::vector<std::shared_ptr<JoinPartition>> partitions;

    const size_t max_partition_depth = 5;

    // Dictionary builders for the key columns. This is
    // always of length n_keys and is nullptr for non DICT keys.
    // These will be shared between the build_table_buffers and
    // probe_table_buffers of all partitions and the build_shuffle_buffer
    // and probe_shuffle_buffer.
    std::vector<std::shared_ptr<DictionaryBuilder>> key_dict_builders;

    // Dictionary builders for the non key DICT columns in the
    // build table. This is always of length
    // (#build_table_columns - n_keys) and has nullptr for non DICT
    // keys.
    std::vector<std::shared_ptr<DictionaryBuilder>>
        build_table_non_key_dict_builders;
    // Dictionary builders for the non key DICT columns in the
    // probe table. This is always of length
    // (#probe_table_columns - n_keys) and has nullptr for non DICT
    // keys.
    std::vector<std::shared_ptr<DictionaryBuilder>>
        probe_table_non_key_dict_builders;

    // Simple concatenation of key_dict_builders and
    // build_table_non_key_dict_builders. We maintain it
    // for convenience.
    std::vector<std::shared_ptr<DictionaryBuilder>> build_table_dict_builders;

    // Simple concatenation of key_dict_builders and
    // probe_table_non_key_dict_builders. We maintain it
    // for convenience.
    std::vector<std::shared_ptr<DictionaryBuilder>> probe_table_dict_builders;

    // Shuffle state
    TableBuildBuffer build_shuffle_buffer;
    TableBuildBuffer probe_shuffle_buffer;

    // Dummy probe table. Useful for the build_table_outer case.
    std::shared_ptr<table_info> dummy_probe_table;

    // Global bloom-filter. This is built during the build step
    // and used during the probe step.
    std::unique_ptr<BloomFilter> global_bloom_filter;

    // Current iteration of the build and probe steps
    uint64_t build_iter;
    uint64_t probe_iter;

    HashJoinState(std::vector<int8_t> build_arr_c_types,
                  std::vector<int8_t> build_arr_array_types,
                  std::vector<int8_t> probe_arr_c_types,
                  std::vector<int8_t> probe_arr_array_types, uint64_t n_keys_,
                  bool build_table_outer_, bool probe_table_outer_,
                  cond_expr_fn_t cond_func_, bool build_parallel_,
                  bool probe_parallel_, int64_t output_batch_size_,
                  size_t max_partition_depth_ = 5);

    /**
     * @brief Create a global bloom filter for this Hash Join
     * operation. This will return a nullptr in case bloom
     * filters are not supported on this architecture.
     *
     * @return std::unique_ptr<BloomFilter>
     */
    std::unique_ptr<BloomFilter> create_bloom_filter() {
        if (bloom_filter_supported()) {
            // Estimate the number of rows to specify based on
            // the target size in bytes in the env or 1MB
            // if not provided.
            int64_t target_bytes = 1000000;
            char* env_target_bytes =
                std::getenv("BODO_STREAM_JOIN_BLOOM_FILTER_TARGET_BYTES");
            if (env_target_bytes) {
                target_bytes = std::stoi(env_target_bytes);
            }
            int64_t num_entries = num_elements_for_bytes(target_bytes);
            return std::make_unique<BloomFilter>(num_entries);
        } else {
            return nullptr;
        }
    }

    /**
     * @brief Split the partition at index 'idx' into two partitions.
     *
     * @param idx Index of the partition (in this->partitions) to split.
     */
    void SplitPartition(size_t idx);

    /**
     * @brief Reserve space in build_table_buffer, build_table_join_hashes, etc.
     * of all partitions to add all rows from 'in_table'.
     * XXX This will likely change to only allocate required memory (or do away
     * with upfront reserve altogether -- at least for inactive partitions)
     *
     * @param in_table Reference table to reserve memory based on.
     */
    void ReserveBuildTable(const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Append a build row. It will figure out the correct
     * partition based on the partitioning hash. If the record
     * is in the "active" (i.e. index 0) partition, it will be
     * added to the hash table of that active partition
     * as well. If record belongs to an inactive partition, it
     * will be simply added to the build buffer of the partition.
     *
     * @param in_table Table to add the row from.
     * @param row_ind Index of the row to add.
     * @param join_hash Join hash of the record.
     * @param partitioning_hash Partitioning hash of the record.
     */
    void AppendBuildRow(const std::shared_ptr<table_info>& in_table,
                        int64_t row_ind, const uint32_t& join_hash,
                        const uint32_t& partitioning_hash);

    /**
     * @brief Append all rows from a table to their respective partitions.
     * This is just a utility wrapper around AppendBuildRow.
     * It is slightly optimized for the single partition case.
     *
     * @param in_table Table to add the rows from.
     * @param join_hashes Join hashes for the records.
     * @param partitioning_hashes Partitioning hashes for the records.
     */
    void AppendBuildBatch(
        const std::shared_ptr<table_info>& in_table,
        const std::shared_ptr<uint32_t[]>& join_hashes,
        const std::shared_ptr<uint32_t[]>& partitioning_hashes);

    /**
     * @brief Finalize build step for all partitions.
     * This will process the partitions one by one (only one is pinned in memory
     * at one time), build hash tables, split partitions as necessary, etc.
     *
     */
    void FinalizeBuild() override;

    /**
     * @brief Reserve enough space to accommodate in_table
     * in probe buffers of each of the inactive partitions.
     *
     * @param in_table Reference table to reserve space based on.
     */
    void ReserveProbeTableForInactivePartitions(
        const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Append probe row to the probe table buffer of the
     * appropriate inactive partition. This assumes that the row
     * is _not_ in the active (index 0) partition.
     *
     * @param in_table Table to add the record from.
     * @param row_ind Index of the row to append.
     * @param join_hash Join hash for the record.
     * @param partitioning_hash Partitioning hash for the record.
     */
    void AppendProbeRowToInactivePartition(
        const std::shared_ptr<table_info>& in_table, int64_t row_ind,
        const uint32_t& join_hash, const uint32_t& partitioning_hash);

    /**
     * @brief Finalize Probe step for all the inactive partitions.
     * This will output a concatenated table with outputs from
     * all the inactive partitions.
     *
     * @tparam build_table_outer
     * @tparam probe_table_outer
     * @tparam non_equi_condition
     * @param build_kept_cols Which columns to generate in the output on the
     * build side.
     * @param probe_kept_cols Which columns to generate in the output on the
     * probe side.
     * @return std::tuple<std::shared_ptr<table_info>,
     * std::shared_ptr<table_info>> Concatenated output from all inactive
     * partitions.
     */
    template <bool build_table_outer, bool probe_table_outer,
              bool non_equi_condition>
    std::tuple<std::shared_ptr<table_info>, std::shared_ptr<table_info>>
    FinalizeProbeForInactivePartitions(
        const std::vector<uint64_t>& build_kept_cols,
        const std::vector<uint64_t>& probe_kept_cols);

    /**
     * @brief Unify dictionaries of input table with build table
     * (build_table_buffer of all partitions and build_shuffle_buffer which all
     * share the same dictionaries) by appending its new dictionary values to
     * buffer's dictionaries and transposing input's indices.
     *
     * @param in_table input table
     * @param only_keys only unify key columns
     * @return std::shared_ptr<table_info> input table with dictionaries unified
     * with build table dictionaries.
     */
    std::shared_ptr<table_info> UnifyBuildTableDictionaryArrays(
        const std::shared_ptr<table_info>& in_table, bool only_keys = false);

    /**
     * @brief Unify dictionaries of input table with probe table
     * (probe_table_buffer of all partitions and probe_shuffle_buffer which all
     * share the same dictionaries) by appending its new dictionary values to
     * buffer's dictionaries and transposing input's indices.
     *
     * @param in_table input table
     * @param only_keys only unify key columns
     * @return std::shared_ptr<table_info> input table with dictionaries unified
     * with probe table dictionaries.
     */
    std::shared_ptr<table_info> UnifyProbeTableDictionaryArrays(
        const std::shared_ptr<table_info>& in_table, bool only_keys = false);

    /**
     * @brief Get dictionary hashes of dict-encoded string key columns (nullptr
     * for other key columns).
     * NOTE: output vector does not have values for data columns (length is
     * this->n_keys).
     *
     * @return
     * std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
     */
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
    GetDictionaryHashesForKeys();
};

class NestedLoopJoinState : public JoinState {
   public:
    // Build state
    TableBuildBuffer build_table_buffer;        // Append only buffer.
    bodo::vector<uint8_t> build_table_matched;  // state for building output
                                                // table (for outer joins)

    NestedLoopJoinState(const std::vector<int8_t>& build_arr_c_types,
                        const std::vector<int8_t>& build_arr_array_types,
                        bool build_table_outer_, bool probe_table_outer_,
                        cond_expr_fn_t cond_func_, bool build_parallel_,
                        bool probe_parallel_, int64_t output_batch_size_)
        : JoinState(0, build_table_outer_, probe_table_outer_, cond_func_,
                    build_parallel_, probe_parallel_,
                    output_batch_size_),  // NestedLoopJoin is only used when
                                          // n_keys is 0
          // For now, we pass nullptrs for dictionary builders, but this should
          // be modified once we support DICT columns in nested loop join
          // (https://bodo.atlassian.net/browse/BSE-478)
          build_table_buffer(build_arr_c_types, build_arr_array_types,
                             std::vector<std::shared_ptr<DictionaryBuilder>>(
                                 build_arr_c_types.size(), nullptr)) {}
};

/**
 * @brief Python wrapper to consume build table batch in nested loop join
 *
 * @param join_state join state pointer
 * @param in_table build table batch
 * @param is_last is last batch
 */
void nested_loop_join_build_consume_batch_py_entry(
    NestedLoopJoinState* join_state, table_info* in_table, bool is_last);

/**
 * @brief consume probe table batch in streaming nested loop join and produce
 * output table batch Design doc:
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1373896721/Vectorized+Nested+Loop+Join+Design
 *
 * @param join_state join state pointer
 * @param in_table probe table batch
 * @param kept_build_col_nums indices of kept columns in build table
 * @param num_kept_build_cols Length of kept_build_col_nums
 * @param kept_probe_col_nums indices of kept columns in probe table
 * @param num_kept_probe_cols Length of kept_probe_col_nums
 * @param[out] total_rows Store the number of rows in the output batch in case
 *        all columns are dead.
 * @param is_last is last batch
 * @return table_info* output table batch
 */
table_info* nested_loop_join_probe_consume_batch_py_entry(
    NestedLoopJoinState* join_state, table_info* in_table,
    uint64_t* kept_build_col_nums, int64_t num_kept_build_cols,
    uint64_t* kept_probe_col_nums, int64_t num_kept_probe_cols,
    int64_t* total_rows, bool is_last, bool* out_is_last);
