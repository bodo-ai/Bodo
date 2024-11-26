#include "../../libs/streaming/_groupby.h"
#include "../../libs/vendored/_murmurhash3.h"
#include "../test.hpp"

// Enable access to protected methods
class GroupbyIncrementalShuffleStateTest
    : public GroupbyIncrementalShuffleState {
   public:
    GroupbyIncrementalShuffleStateTest(
        const std::shared_ptr<bodo::Schema> shuffle_table_schema_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        const std::vector<std::shared_ptr<BasicColSet>>& col_sets_,
        const uint64_t mrnf_n_sort_cols_, const uint64_t n_keys_,
        const uint64_t& curr_iter_, int64_t& sync_freq_, int64_t op_id_,
        const bool nunique_only_, const AggregationType agg_type_,
        const std::vector<int32_t>& f_running_value_offsets_,
        const bool accumulate_before_update_)
        : GroupbyIncrementalShuffleState(
              shuffle_table_schema_, dict_builders_, col_sets_,
              mrnf_n_sort_cols_, n_keys_, curr_iter_, sync_freq_, op_id_,
              nunique_only_, agg_type_, f_running_value_offsets_,
              accumulate_before_update_) {}

    void AppendPreReductionTable(std::shared_ptr<table_info> table,
                                 std::vector<uint32_t> hashes) {
        this->pre_reduction_table_buffer->ReserveTable(table);
        this->pre_reduction_table_buffer->UnsafeAppendBatch(table);
        this->pre_reduction_hashes.insert(this->pre_reduction_hashes.end(),
                                          hashes.begin(), hashes.end());
    }

    /**
     * @brief Trigger the GroupbyIncrementalShuffleState reduction logic and
     * return if a reduction occurred.
     *
     * @return bool True if a reduction occurred.
     */
    bool TriggerReduction() {
        MetricBase::StatValue old_count = this->metrics.n_shuffle_reductions;
        this->ShouldShuffleAfterProcessing(true);
        return this->metrics.n_shuffle_reductions != old_count;
    }

    /**
     * @brief Force a reduction by setting the reduction threshold to a value
     * much greater than 1. Used to setup the hash table because a skipped
     * reduction may have incorrect assumptions.
     *
     * @return Should always return true, but returns if a reduction occurred to
     * allow asserting the reduction succeeded.
     */
    bool ForceReduction() {
        MetricBase::StatValue old_count = this->metrics.n_shuffle_reductions;
        double old_threshold = this->agg_reduction_threshold;
        this->agg_reduction_threshold = 5.0;
        this->ShouldShuffleAfterProcessing(true);
        this->agg_reduction_threshold = old_threshold;
        return this->metrics.n_shuffle_reductions != old_count;
    }
};

/**
 * @brief Create a GroupbyIncrementalShuffleState for a distinct operation
 * on a single integer column. This is the simplest operation we can generate
 * for testing reductions.
 *
 * @return std::unique_ptr<GroupbyIncrementalShuffleState> The new distinct
 * state.
 */
std::unique_ptr<GroupbyIncrementalShuffleStateTest> createDistinctState() {
    std::shared_ptr<bodo::Schema> schema = std::make_shared<bodo::Schema>();
    schema->append_column(bodo_array_type::NUMPY, Bodo_CTypes::INT64);
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    dict_builders.push_back(nullptr);
    std::vector<std::shared_ptr<BasicColSet>> empty_col_sets;
    uint64_t curr_iter = 0;
    int64_t sync_freq = 1000;
    std::vector<int32_t> empty_running_value_offsets;

    return std::make_unique<GroupbyIncrementalShuffleStateTest>(
        schema, dict_builders, empty_col_sets, 0, 1, curr_iter, sync_freq, -1,
        false, AggregationType::AGGREGATE, empty_running_value_offsets, false);
}

std::pair<std::unique_ptr<table_info>, std::vector<uint32_t>>
create_zeros_table(size_t n_rows) {
    std::unique_ptr<array_info> input_array =
        alloc_numpy(n_rows, Bodo_CTypes::INT64);
    std::vector<uint32_t> hashes;
    for (size_t i = 0; i < n_rows; i++) {
        getv<int64_t, bodo_array_type::NUMPY>(input_array, i) = 0;
        hashes.push_back(0);
    }
    std::vector<std::shared_ptr<array_info>> arrays = {std::move(input_array)};
    return std::make_pair(std::make_unique<table_info>(arrays, n_rows), hashes);
}

std::pair<std::unique_ptr<table_info>, std::vector<uint32_t>>
create_index_table(size_t n_rows) {
    std::unique_ptr<array_info> input_array =
        alloc_numpy(n_rows, Bodo_CTypes::INT64);
    std::vector<uint32_t> hashes;
    for (size_t i = 0; i < n_rows; i++) {
        getv<int64_t, bodo_array_type::NUMPY>(input_array, i) = i;
        // Note: You must use an actually hash function to get a decent
        // HLL estimate.
        uint32_t hash_value;
        hash_inner_32<size_t>(&i, 0, &hash_value);
        hashes.push_back(hash_value);
    }
    std::vector<std::shared_ptr<array_info>> arrays = {std::move(input_array)};
    return std::make_pair(std::make_unique<table_info>(arrays, n_rows), hashes);
}

static bodo::tests::suite tests([] {
    bodo::tests::test("test_distinct_shuffle_reduction", [] {
        // Test that having a table of matching data will take the reduction
        // path.
        size_t n_rows = 10000;
        std::unique_ptr<GroupbyIncrementalShuffleStateTest> shuffle_state =
            createDistinctState();
        auto [table, hashes] = create_zeros_table(n_rows);
        shuffle_state->AppendPreReductionTable(std::move(table),
                                               std::move(hashes));
        bodo::tests::check(shuffle_state->TriggerReduction());
    });
    bodo::tests::test("test_distinct_shuffle_no_reduction", [] {
        // Test that having a table of distinct data will not take the reduction
        // path.
        size_t n_rows = 10000;
        std::unique_ptr<GroupbyIncrementalShuffleStateTest> shuffle_state =
            createDistinctState();
        auto [table, hashes] = create_index_table(n_rows);
        shuffle_state->AppendPreReductionTable(std::move(table),
                                               std::move(hashes));
        bodo::tests::check(!shuffle_state->TriggerReduction());
    });
    bodo::tests::test("test_distinct_duplicate_reduction", [] {
        // Test that having a table of distinct data that matches the existing
        // hash table will take the reduction path.
        size_t n_rows = 10000;
        std::unique_ptr<GroupbyIncrementalShuffleStateTest> shuffle_state =
            createDistinctState();
        auto [table, hashes] = create_index_table(n_rows);
        shuffle_state->AppendPreReductionTable(std::move(table),
                                               std::move(hashes));
        bodo::tests::check(shuffle_state->ForceReduction());
        auto [new_table, new_hashes] = create_index_table(n_rows);
        shuffle_state->AppendPreReductionTable(std::move(new_table),
                                               std::move(new_hashes));
        // While the pre reduction table is unique it will match the hash table
        // exactly so it should trigger a reduction.
        bodo::tests::check(shuffle_state->TriggerReduction());
    });
});
