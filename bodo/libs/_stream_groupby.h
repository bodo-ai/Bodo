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
    const std::vector<int32_t> f_in_offsets;
    const std::vector<int32_t> f_in_cols;

    // indices of update and combine columns for each function
    std::vector<int32_t> f_update_offsets;
    std::vector<int32_t> f_combine_offsets;

    // Current iteration of build steps
    uint64_t build_iter;

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

    tracing::ResumableEvent groupby_event;

    GroupbyState(std::vector<int8_t> in_arr_c_types,
                 std::vector<int8_t> in_arr_array_types,
                 std::vector<int32_t> ftypes,
                 std::vector<int32_t> f_in_offsets_,
                 std::vector<int32_t> f_in_cols_, uint64_t n_keys_,
                 int64_t output_batch_size_, bool parallel_)
        : n_keys(n_keys_),
          output_batch_size(output_batch_size_),
          f_in_offsets(std::move(f_in_offsets_)),
          f_in_cols(std::move(f_in_cols_)),
          parallel(parallel_),
          local_build_table({}, HashGroupbyTable<true>(this),
                            KeyEqualGroupbyTable<true>(this, n_keys)),
          shuffle_build_table({}, HashGroupbyTable<false>(this),
                              KeyEqualGroupbyTable<false>(this, n_keys)),
          groupby_event("Groupby") {
        // TODO[BSE-566]: support dictionary arrays
        for (int8_t arr_type : in_arr_array_types) {
            if (arr_type == bodo_array_type::DICT) {
                throw std::runtime_error(
                    "dictionary-encoded input not supported yet in streaming "
                    "groupby");
            }
        }

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
        f_update_offsets.push_back(n_keys);
        int32_t curr_update_offset = n_keys;
        f_combine_offsets.push_back(n_keys);
        int32_t curr_combine_offset = n_keys;

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
        bool skipna = false;
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
            for (size_t input_ind = (size_t)f_in_offsets[i];
                 input_ind < (size_t)f_in_offsets[i + 1]; input_ind++) {
                input_cols.push_back(nullptr);
                in_arr_types.push_back((bodo_array_type::arr_type_enum)
                                           in_arr_array_types[input_ind]);
                in_dtypes.push_back(
                    (Bodo_CTypes::CTypeEnum)in_arr_c_types[input_ind]);
            }
            std::shared_ptr<BasicColSet> col_set = makeColSet(
                input_cols, index_col, ftypes[i], do_combine, skipna, 0, {0}, 0,
                parallel, window_ascending_vect, window_na_position_vect,
                {nullptr}, 0, nullptr, nullptr, 0, nullptr, use_sql_rules);

            if (!accumulate_before_update) {
                // get update/combine type info to initialize build state
                std::tuple<std::vector<bodo_array_type::arr_type_enum>,
                           std::vector<Bodo_CTypes::CTypeEnum>>
                    update_arr_types =
                        col_set->getUpdateColumnTypes(in_arr_types, in_dtypes);
                std::tuple<std::vector<bodo_array_type::arr_type_enum>,
                           std::vector<Bodo_CTypes::CTypeEnum>>
                    combine_arr_types = col_set->getCombineColumnTypes(
                        std::get<0>(update_arr_types),
                        std::get<1>(update_arr_types));
                for (auto t : std::get<0>(combine_arr_types)) {
                    build_arr_array_types.push_back(t);
                }
                for (auto t : std::get<1>(combine_arr_types)) {
                    build_arr_c_types.push_back(t);
                }
                curr_update_offset += std::get<0>(update_arr_types).size();
                f_update_offsets.push_back(curr_update_offset);
                curr_combine_offset += std::get<0>(combine_arr_types).size();
                f_combine_offsets.push_back(curr_combine_offset);
            }

            this->col_sets.push_back(col_set);
        }

        // build buffer types are same as input if just accumulating batches
        if (accumulate_before_update) {
            build_arr_array_types = in_arr_array_types;
            build_arr_c_types = in_arr_c_types;
        }

        // TODO[BSE-566]: support dictionary arrays
        local_table_buffer =
            TableBuildBuffer(build_arr_c_types, build_arr_array_types,
                             std::vector<std::shared_ptr<DictionaryBuilder>>(
                                 build_arr_c_types.size(), nullptr));
        shuffle_table_buffer =
            TableBuildBuffer(build_arr_c_types, build_arr_array_types,
                             std::vector<std::shared_ptr<DictionaryBuilder>>(
                                 build_arr_c_types.size(), nullptr));
    }

    void InitOutputBuffer(const std::shared_ptr<table_info>& dummy_table) {
        auto [arr_c_types, arr_array_types] =
            get_dtypes_arr_types_from_table(dummy_table);
        // TODO[BSE-566]: Support dict builders
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders(
            dummy_table->columns.size(), nullptr);
        this->output_buffer = std::make_shared<ChunkedTableBuilder>(
            arr_c_types, arr_array_types, dict_builders,
            /*chunk_size*/ this->output_batch_size,
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES);
    }
};
