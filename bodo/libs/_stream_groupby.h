#pragma once
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_chunked_table_builder.h"
#include "_groupby.h"
#include "_groupby_col_set.h"
#include "_groupby_ftypes.h"
#include "_groupby_groups.h"
#include "_stream_join.h"

class GroupbyState;

template <bool is_local>
struct HashGroupbyTable {
    /**
     * provides row hashes for groupby hash table (bodo::unord_map_container)
     *
     * Input row number iRow can refer to either build or input table.
     * If iRow >= 0 then it is in the build table at index iRow.
     * If iRow < 0 then it is in the input table
     *    at index (-iRow - 1).
     *
     * @param iRow row number
     * @return hash of row iRow
     */
    uint32_t operator()(const int64_t iRow) const;
    GroupbyState* groupby_state;
};

template <bool is_local>
struct KeyEqualGroupbyTable {
    /**
     * provides row comparison for groupby hash table
     * (bodo::unord_map_container)
     *
     * Input row number iRow can refer to either build or input table.
     * If iRow >= 0 then it is in the build table at index iRow.
     * If iRow < 0 then it is in the input table
     *    at index (-iRow - 1).
     *
     * @param iRowA is the first row index for the comparison
     * @param iRowB is the second row index for the comparison
     * @return true if equal else false
     */
    bool operator()(const int64_t iRowA, const int64_t iRowB) const;
    GroupbyState* groupby_state;
    const uint64_t n_keys;
};

class GroupbyState {
   public:
    const uint64_t n_keys;
    bool parallel;
    const int64_t output_batch_size;

    std::vector<std::shared_ptr<BasicColSet>> col_sets;

    // Local build state
    // Current number of groups
    int64_t local_next_group = 0;
    // Map row number to group number
    bodo::unord_map_container<int64_t, int64_t, HashGroupbyTable<true>,
                              KeyEqualGroupbyTable<true>>
        local_build_table;
    // hashes of data in local_table_buffer
    bodo::vector<uint32_t> local_table_groupby_hashes;
    // Current running values (output of "combine" step)
    TableBuildBuffer local_table_buffer;

    // Shuffle build state
    int64_t shuffle_next_group = 0;
    bodo::unord_map_container<int64_t, int64_t, HashGroupbyTable<false>,
                              KeyEqualGroupbyTable<false>>
        shuffle_build_table;
    bodo::vector<uint32_t> shuffle_table_groupby_hashes;
    TableBuildBuffer shuffle_table_buffer;

    // temporary batch data
    std::shared_ptr<table_info> in_table = nullptr;
    std::shared_ptr<uint32_t[]> in_table_hashes = nullptr;

    // indices of input columns for each function
    // f_in_offsets contains the offsets into f_in_cols.
    // f_in_cols is a list of physical column indices.
    // For example:
    //
    // f_in_offsets = (0, 1, 5)
    // f_in_cols = (0, 7, 1, 3, 4, 0)
    // The first function uses the columns in f_in_cols[0:1]. IE physical index
    // 0 in the input table. The second function uses the column f_in_cols[1:5].
    // IE physical index 7, 1, 3, 4, 0 in the input table.
    const std::vector<int32_t> f_in_offsets;
    const std::vector<int32_t> f_in_cols;

    // indices of update and combine columns for each function
    std::vector<int32_t> f_running_value_offsets;

    // Current iteration of build steps
    uint64_t build_iter;
    // The number of iterations between syncs
    uint64_t sync_iter;

    // Accumulating all values before update is needed
    // when one of the groupby functions is
    // median/nunique/...
    // similar to shuffle_before_update in non-streaming groupby
    // see:
    // https://bodo.atlassian.net/wiki/spaces/B/pages/1346568245/Vectorized+Groupby+Design#Getting-All-Group-Data-before-Computation
    bool accumulate_before_update = false;
    bool req_extended_group_info = false;

    // Output buffer
    // This will be lazily initialized during the end of
    // the build step to simplify specifying the output column types.
    // TODO(njriasan): Move to initialization information.
    std::shared_ptr<ChunkedTableBuilder> output_buffer = nullptr;

    // Dictionary builders for the key columns. This is
    // always of length n_keys and is nullptr for non DICT keys.
    // These will be shared between the build_table_buffers and
    // probe_table_buffers of all partitions and the build_shuffle_buffer
    // and probe_shuffle_buffer.
    std::vector<std::shared_ptr<DictionaryBuilder>> key_dict_builders;

    // Simple concatenation of key_dict_builders and
    // non key dict builders
    std::vector<std::shared_ptr<DictionaryBuilder>> build_table_dict_builders;

    tracing::ResumableEvent groupby_event;

    GroupbyState(std::vector<int8_t> in_arr_c_types,
                 std::vector<int8_t> in_arr_array_types,
                 std::vector<int32_t> ftypes,
                 std::vector<int32_t> f_in_offsets_,
                 std::vector<int32_t> f_in_cols_, uint64_t n_keys_,
                 int64_t output_batch_size_, bool parallel_,
                 uint64_t sync_iter_)
        : n_keys(n_keys_),
          parallel(parallel_),
          output_batch_size(output_batch_size_),
          local_build_table({}, HashGroupbyTable<true>(this),
                            KeyEqualGroupbyTable<true>(this, n_keys)),
          shuffle_build_table({}, HashGroupbyTable<false>(this),
                              KeyEqualGroupbyTable<false>(this, n_keys)),
          sync_iter(sync_iter_),
          f_in_offsets(std::move(f_in_offsets_)),
          f_in_cols(std::move(f_in_cols_)),
          groupby_event("Groupby") {
        // Add key column types to runnig value buffer types (same type as
        // input)
        std::vector<int8_t> build_arr_array_types;
        std::vector<int8_t> build_arr_c_types;
        for (size_t i = 0; i < n_keys; i++) {
            build_arr_array_types.push_back(in_arr_array_types[i]);
            build_arr_c_types.push_back(in_arr_c_types[i]);
        }
        // Get offsets of update and combine columns for each function since
        // some functions have multiple update/combine columns
        f_running_value_offsets.push_back(n_keys);
        int32_t curr_running_value_offset = n_keys;

        for (size_t i = 0; i < ftypes.size(); i++) {
            int ftype = ftypes[i];
            // NOTE: adding all functions that need accumulating inputs for now
            // but they may not be supported in streaming groupby yet
            if (ftype == Bodo_FTypes::median || ftype == Bodo_FTypes::cumsum ||
                ftype == Bodo_FTypes::cumprod || ftype == Bodo_FTypes::cummin ||
                ftype == Bodo_FTypes::cummax || ftype == Bodo_FTypes::shift ||
                ftype == Bodo_FTypes::transform ||
                ftype == Bodo_FTypes::ngroup || ftype == Bodo_FTypes::window ||
                ftype == Bodo_FTypes::listagg ||
                ftype == Bodo_FTypes::nunique || ftype == Bodo_FTypes::head ||
                ftype == Bodo_FTypes::gen_udf) {
                accumulate_before_update = true;
            }
            if (ftype == Bodo_FTypes::median || ftype == Bodo_FTypes::cumsum ||
                ftype == Bodo_FTypes::cumprod || ftype == Bodo_FTypes::cummin ||
                ftype == Bodo_FTypes::cummax || ftype == Bodo_FTypes::shift ||
                ftype == Bodo_FTypes::transform ||
                ftype == Bodo_FTypes::ngroup || ftype == Bodo_FTypes::window ||
                ftype == Bodo_FTypes::listagg ||
                ftype == Bodo_FTypes::nunique) {
                req_extended_group_info = true;
            }
        }

        // TODO[BSE-578]: handle all necessary ColSet parameters for BodoSQL
        // groupby functions
        std::shared_ptr<array_info> index_col = nullptr;

        // Currently, all SQL aggregations that we support excluding count(*)
        // drop or ignore na values durring computation. Since count(*) maps to
        // size, and skip_na_data has no effect on that aggregation, we can
        // safely set skip_na_data to true for all SQL aggregations. There is an
        // issue to fix this behavior so that use_sql_rules trumps the value of
        // skip_na_data: https://bodo.atlassian.net/browse/BSE-841
        bool skip_na_data = true;
        bool use_sql_rules = true;
        bool do_combine = !accumulate_before_update;
        std::vector<bool> window_ascending_vect;
        std::vector<bool> window_na_position_vect;
        for (size_t i = 0; i < ftypes.size(); i++) {
            // set dummy input columns in ColSet since will be replaced by input
            // batches
            std::vector<std::shared_ptr<array_info>> input_cols;
            std::vector<bodo_array_type::arr_type_enum> in_arr_types;
            std::vector<Bodo_CTypes::CTypeEnum> in_dtypes;
            for (size_t logical_input_ind = (size_t)f_in_offsets[i];
                 logical_input_ind < (size_t)f_in_offsets[i + 1];
                 logical_input_ind++) {
                size_t physical_input_ind =
                    (size_t)f_in_cols[logical_input_ind];
                input_cols.push_back(nullptr);
                in_arr_types.push_back(
                    (bodo_array_type::arr_type_enum)
                        in_arr_array_types[physical_input_ind]);
                in_dtypes.push_back(
                    (Bodo_CTypes::CTypeEnum)in_arr_c_types[physical_input_ind]);
            }
            std::shared_ptr<BasicColSet> col_set = makeColSet(
                input_cols, index_col, ftypes[i], do_combine, skip_na_data, 0,
                {0}, 0, parallel, window_ascending_vect,
                window_na_position_vect, {nullptr}, 0, nullptr, nullptr, 0,
                nullptr, use_sql_rules);

            if (!accumulate_before_update) {
                // get update/combine type info to initialize build state
                std::tuple<std::vector<bodo_array_type::arr_type_enum>,
                           std::vector<Bodo_CTypes::CTypeEnum>>
                    running_values_arr_types =
                        col_set->getRunningValueColumnTypes(in_arr_types,
                                                            in_dtypes);

                for (auto t : std::get<0>(running_values_arr_types)) {
                    build_arr_array_types.push_back(t);
                }
                for (auto t : std::get<1>(running_values_arr_types)) {
                    build_arr_c_types.push_back(t);
                }

                curr_running_value_offset +=
                    std::get<0>(running_values_arr_types).size();
                f_running_value_offsets.push_back(curr_running_value_offset);
            }

            this->col_sets.push_back(col_set);
        }

        // build buffer types are same as input if just accumulating batches
        if (accumulate_before_update) {
            build_arr_array_types = in_arr_array_types;
            build_arr_c_types = in_arr_c_types;
        }

        this->key_dict_builders.resize(this->n_keys);

        // Create dictionary builders for key columns:
        for (uint64_t i = 0; i < this->n_keys; i++) {
            if (build_arr_array_types[i] == bodo_array_type::DICT) {
                std::shared_ptr<array_info> dict = alloc_array(
                    0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
                this->key_dict_builders[i] =
                    std::make_shared<DictionaryBuilder>(dict, true);
            } else {
                this->key_dict_builders[i] = nullptr;
            }
        }

        std::vector<std::shared_ptr<DictionaryBuilder>>
            build_table_non_key_dict_builders;
        // Create dictionary builders for non-key columns in build table:
        for (size_t i = this->n_keys; i < build_arr_array_types.size(); i++) {
            if (build_arr_array_types[i] == bodo_array_type::DICT) {
                std::shared_ptr<array_info> dict = alloc_array(
                    0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
                build_table_non_key_dict_builders.emplace_back(
                    std::make_shared<DictionaryBuilder>(dict, false));
            } else {
                build_table_non_key_dict_builders.emplace_back(nullptr);
            }
        }

        this->build_table_dict_builders.insert(
            this->build_table_dict_builders.end(),
            this->key_dict_builders.begin(), this->key_dict_builders.end());

        this->build_table_dict_builders.insert(
            this->build_table_dict_builders.end(),
            build_table_non_key_dict_builders.begin(),
            build_table_non_key_dict_builders.end());

        local_table_buffer =
            TableBuildBuffer(build_arr_c_types, build_arr_array_types,
                             this->build_table_dict_builders);
        shuffle_table_buffer =
            TableBuildBuffer(build_arr_c_types, build_arr_array_types,
                             this->build_table_dict_builders);
    }

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

    void InitOutputBuffer(const std::shared_ptr<table_info>& dummy_table) {
        auto [arr_c_types, arr_array_types] =
            get_dtypes_arr_types_from_table(dummy_table);
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders(
            dummy_table->columns.size(), nullptr);
        // Dictionary encoded arrays are only support for keys.
        for (size_t i = 0; i < this->n_keys; i++) {
            dict_builders[i] = this->build_table_dict_builders[i];
        }
        this->output_buffer = std::make_shared<ChunkedTableBuilder>(
            arr_c_types, arr_array_types, dict_builders,
            /*chunk_size*/ this->output_batch_size,
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES);
    }
};
