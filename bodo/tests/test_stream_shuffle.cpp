#include <mpi.h>
#include "../libs/_stream_groupby.h"
#include "../libs/_stream_shuffle.h"
#include "./table_generator.hpp"
#include "./test.hpp"

// Mock class to test protected methods
class IncrementalShuffleStateTest : public IncrementalShuffleState {
   public:
    IncrementalShuffleStateTest(
        const std::vector<int8_t>& arr_c_types_,
        const std::vector<int8_t>& arr_array_types_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        const uint64_t n_keys_, const uint64_t& curr_iter_, int64_t& sync_freq_)
        : IncrementalShuffleState(arr_c_types_, arr_array_types_,
                                  dict_builders_, n_keys_, curr_iter_,
                                  sync_freq_, -1){};
    void ResetAfterShuffle() { IncrementalShuffleState::ResetAfterShuffle(); }
};

static bodo::tests::suite tests([] {
    bodo::tests::test("test_build_table_recreate", [] {
        int npes;
        MPI_Comm_size(MPI_COMM_WORLD, &npes);
        if (npes != 1) {
            std::cout
                << "this test depends on all data shuffling to the same rank"
                << std::endl;
            return;
        }

        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders =
            std::vector<std::shared_ptr<DictionaryBuilder>>(1);
        int64_t sync_freq = 10;
        IncrementalShuffleStateTest state = IncrementalShuffleStateTest(
            {0}, {0}, dict_builders, 1, 0, sync_freq);

        // Make it look like the shuffle buffer is too large
        // The utilization should already be low since we haven't added any data
        state.table_buffer->array_buffers[0]
            .data_array->buffers[0]
            ->getMeminfo()
            ->size = 100000000000000000L;

        // Keep a reference to the original table buffer
        auto* original_table_buffer = state.table_buffer.get();
        // This should recreate the table buffer
        state.ResetAfterShuffle();

        // Check that the table buffer is recreated
        bodo::tests::check(state.table_buffer->array_buffers[0]
                               .data_array->buffers[0]
                               ->getMeminfo()
                               ->size == 0);

        bodo::tests::check(state.table_buffer.get() != original_table_buffer);
    });
    bodo::tests::test("test_mrnf_shuffle_local_reduction", [] {
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders =
            std::vector<std::shared_ptr<DictionaryBuilder>>(2);
        int64_t sync_freq = 10;
        std::vector<int8_t> arr_arr_types = {
            bodo_array_type::NULLABLE_INT_BOOL,
            bodo_array_type::NULLABLE_INT_BOOL};
        std::vector<int8_t> arr_c_types = {Bodo_CTypes::INT32,
                                           Bodo_CTypes::UINT64};
        std::vector<std::shared_ptr<array_info>> local_input_cols = {
            alloc_nullable_array(10, (Bodo_CTypes::CTypeEnum)arr_c_types[1])};
        std::shared_ptr<BasicColSet> col_set =
            makeColSet(local_input_cols,                    // in_cols
                       nullptr,                             // index_col
                       Bodo_FTypes::min_row_number_filter,  // ftype
                       true,                                // do_combine
                       true,                                // skip_na_data
                       0,                                   // period
                       {0},                                 // transform_funcs
                       0,                                   // n_udf
                       false,                               // parallel
                       {true}, {true},  // window_na_position
                       {nullptr},       // window_args
                       0,               // n_input_cols
                       nullptr,         // udf_n_redvars
                       nullptr,         // udf_table
                       0,               // udf_table_idx
                       nullptr,         // nunique_table
                       true             // use_sql_rules
            );
        auto schema = bodo::Schema::Deserialize(arr_arr_types, arr_c_types);
        std::vector<std::shared_ptr<BasicColSet>> col_sets = {col_set};
        GroupbyIncrementalShuffleState state = GroupbyIncrementalShuffleState(
            std::make_shared<bodo::Schema>(*schema), dict_builders, {col_set},
            1, 1, 0, sync_freq, 0, false, true

        );
        // Table with only 1 unique key, should perform local reduction
        auto table = bodo::tests::cppToBodo({"A", "B"}, {true, true}, {},
                                            std::vector<int32_t>(100, 0),
                                            std::vector<uint64_t>(100, 1));
        std::vector<bool> append = std::vector<bool>(table->nrows(), true);
        state.AppendBatch(std::move(table), append);
        auto table_opt = state.ShuffleIfRequired(true);
        bodo::tests::check(table_opt.has_value());
        auto shuffled_table = table_opt.value();
        bodo::tests::check(shuffled_table->nrows() == 1);

        std::vector<int32_t> keys;
        for (size_t i = 0; i < 100; ++i) {
            keys.push_back(i);
        }
        // Table with all unique keys, shouldn't perform local reduction
        table = bodo::tests::cppToBodo({"A", "B"}, {true, true}, {}, keys,
                                       std::vector<uint64_t>(100, 1));
        state.AppendBatch(std::move(table), append);
        table_opt = state.ShuffleIfRequired(true);
        bodo::tests::check(table_opt.has_value());
        shuffled_table = table_opt.value();
        bodo::tests::check(shuffled_table->nrows() == 100);
    });
});
