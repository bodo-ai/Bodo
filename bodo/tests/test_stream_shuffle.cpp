#include "../libs/_stream_shuffle.h"
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
                                  sync_freq_){};
    void ResetAfterShuffle() { IncrementalShuffleState::ResetAfterShuffle(); }
};

static bodo::tests::suite tests([] {
    bodo::tests::test("test_build_table_recreate", [] {
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
});
