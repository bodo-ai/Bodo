#pragma once

#include "_bodo_common.h"
#include "_query_profile_collector.h"
#include "_table_builder.h"

#define DEFAULT_SHUFFLE_THRESHOLD 50 * 1024 * 1024  // 50MiB
#define MIN_SHUFFLE_THRESHOLD 50 * 1024 * 1024      // 50MiB
#define MAX_SHUFFLE_THRESHOLD 200 * 1024 * 1024     // 200MiB
#define DEFAULT_SHUFFLE_THRESHOLD_PER_MiB 12800     // 12.5KiB

// Factor in determining whether shuffle buffer is large enough to need cleared
constexpr float SHUFFLE_BUFFER_CUTOFF_MULTIPLIER = 3.0;

// Minimum utilization of shuffle buffer, used as a factor in determining when
// to clear
constexpr float SHUFFLE_BUFFER_MIN_UTILIZATION = 0.5;

// Streaming batch size. The default of 4096 should match the default of
// bodosql_streaming_batch_size defined in __init__.py
static char* __env_streaming_batch_size_str =
    std::getenv("BODO_STREAMING_BATCH_SIZE");
const int STREAMING_BATCH_SIZE = __env_streaming_batch_size_str != nullptr
                                     ? std::stoi(__env_streaming_batch_size_str)
                                     : 4096;

#ifndef DEFAULT_SYNC_ITERS
// Default number of iterations between syncs
// NOTE: should be the same as default_stream_loop_sync_iters in __init__.py
#define DEFAULT_SYNC_ITERS 1000
#endif

#ifndef DEFAULT_SYNC_UPDATE_FREQ
// Update sync freq every 10 syncs by default.
#define DEFAULT_SYNC_UPDATE_FREQ 10
#endif

/**
 * @brief Get the default shuffle threshold for all streaming operators. We
 * check the 'BODO_SHUFFLE_THRESHOLD' environment variable first. If it's not
 * set, we decide based on available memory per rank.
 *
 * @return int64_t Threshold (in bytes)
 */
int64_t get_shuffle_threshold();

/**
 * @brief Struct for Shuffle metrics.
 *
 */
class IncrementalShuffleMetrics {
   public:
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    // Time spent appending to the shuffle buffer.
    time_t append_time = 0;
    // Time spent in shuffling data.
    time_t shuffle_time = 0;
    // Total number of shuffles
    stat_t n_shuffles = 0;
    // Time spent hashing rows for shuffle.
    time_t hash_time = 0;
    // Time spent unifying the dictionaries globally before the shuffle.
    time_t dict_unification_time = 0;
    // Total number of rows appended to the shuffle buffer.
    stat_t total_appended_nrows = 0;
    // Total number of rows sent to other ranks across all shuffles.
    stat_t total_sent_nrows = 0;
    // Total number of rows received from other ranks across all shuffles.
    stat_t total_recv_nrows = 0;
    // Approximate number of bytes sent to other ranks across all shuffles.
    stat_t total_approx_sent_size_bytes = 0;
    // Total number of bytes received from other ranks across all shuffles.
    stat_t total_recv_size_bytes = 0;
    // Peak allocated size of the shuffle buffer.
    stat_t peak_capacity_bytes = 0;
    // Peak utilized size of the shuffle buffer.
    stat_t peak_utilization_bytes = 0;
    // Total number of times we reset the buffer since it grew too large.
    stat_t n_buffer_resets = 0;

    /**
     * @brief Helper function for exporting metrics during reporting steps in
     * Join/GroupBy.
     *
     * @param metrics Vector of metrics to append to.
     */
    void add_to_metrics(std::vector<MetricBase>& metrics);
};

/**
 * @brief Common hash-shuffle functionality for streaming operators such as
 * HashJoin and Groupby.
 *
 */
class IncrementalShuffleState {
   public:
    /// @brief Schema of the shuffle table
    const std::shared_ptr<bodo::Schema> schema;
    /// @brief Dictionary builder for the dictionary-encoded columns. Note that
    /// these are only for top-level dictionaries and not for dictionary-encoded
    /// fields within nested data types.
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    /// @brief Shuffle data buffer.
    std::unique_ptr<TableBuildBuffer> table_buffer;

    /**
     * @brief Constructor for new IncrementalShuffleState
     *
     * @param schema_ Schema of the shuffle table
     * @param dict_builders_ Dictionary builders for the top level columns.
     * @param n_keys_ Number of key columns (to shuffle based off of).
     * @param curr_iter_ Reference to the iteration counter from parent
     * operator. e.g. In Groupby, this is 'build_iter'. For HashJoin, this could
     * be either 'build_iter' or 'probe_iter' based on whether it's the
     * build_shuffle_state or probe_shuffle_state, respectively.
     * @param sync_freq_ Reference to the synchronization frequency variable of
     * the parent state. This will be modified by this state adaptively (if
     * enabled).
     */
    IncrementalShuffleState(
        std::shared_ptr<bodo::Schema> schema_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        const uint64_t n_keys_, const uint64_t& curr_iter_, int64_t& sync_freq_,
        int64_t parent_op_id_);
    /**
     * @brief Constructor for new IncrementalShuffleState
     *
     * @param arr_c_types_ Array types of the shuffle table.
     * @param arr_array_types_ CTypes of the shuffle table.
     * @param dict_builders_ Dictionary builders for the top level columns.
     * @param n_keys_ Number of key columns (to shuffle based off of).
     * @param curr_iter_ Reference to the iteration counter from parent
     * operator. e.g. In Groupby, this is 'build_iter'. For HashJoin, this could
     * be either 'build_iter' or 'probe_iter' based on whether it's the
     * build_shuffle_state or probe_shuffle_state, respectively.
     * @param sync_freq_ Reference to the synchronization frequency variable of
     * the parent state. This will be modified by this state adaptively (if
     * enabled).
     */
    IncrementalShuffleState(
        const std::vector<int8_t>& arr_c_types_,
        const std::vector<int8_t>& arr_array_types_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        const uint64_t n_keys_, const uint64_t& curr_iter_, int64_t& sync_freq_,
        int64_t parent_op_id_);

    /**
     * @brief Calculate initial synchronization frequency if syncing
     * adaptively.
     * This must be called in the first iteration. It estimates how many
     * iterations it will take to for the shuffle buffer size of any rank to be
     * larger than this->shuffle_threshold based on the size of the first input
     * batch. 'sync_freq' will be modified accordingly.
     *
     * @param sample_in_table_batch Input batch to use for estimating the
     * initial sync frequency.
     * @param is_parallel Parallel flag
     */
    void Initialize(const std::shared_ptr<table_info>& sample_in_table_batch,
                    bool is_parallel);

    /**
     * @brief Append a build batch to the shuffle table. Only the rows specified
     * in append_rows will be appended. Note that we will reserve enough space
     * to append the entire input batch.
     *
     * @param in_table Input batch to append rows from.
     * @param append_rows Bitmask specifying the rows to append.
     */
    void AppendBatch(const std::shared_ptr<table_info>& in_table,
                     const std::vector<bool>& append_rows);

    /**
     * @brief Shuffle the data in the shuffle-table if we're at a
     * synchronization iteration and the shuffle table is larger than 50MiB on
     * some rank, or if this is the last iteration. It will also update the
     * synchronization frequency if we're at an update-sync-freq iteration.
     * This must be called in every iteration, but only if shuffle is possible
     * (i.e. data is distributed and requires shuffling). If no shuffle was
     * done, we will not return anything. If we do shuffle, we will return the
     * output shuffle table. The DICT columns in the output will have
     * dictionaries unified with the dict-builders. We will also reset the
     * shuffle buffer after every shuffle.
     *
     * @param is_last Is the last iteration.
     * @return std::optional<std::shared_ptr<table_info>> Output shuffle table
     * if we did a shuffle.
     */
    std::optional<std::shared_ptr<table_info>> ShuffleIfRequired(
        const bool is_last);

    /**
     * @brief Finalize the shuffle state. This will free the shuffle buffer and
     * release memory. It will also release its references to the dictionary
     * builders.
     *
     */
    virtual void Finalize();

    /**
     * @brief Export shuffle metrics into the provided vector.
     *
     * @param[in, out] metrics Vector to append the metrics to.
     */
    virtual void ExportMetrics(std::vector<MetricBase>& metrics) {
        this->metrics.add_to_metrics(metrics);
    }

    /**
     * @brief Reset the metrics.
     *
     */
    virtual void ResetMetrics() { this->metrics = IncrementalShuffleMetrics(); }

   protected:
    /**
     * @brief Helper function for ShuffleIfRequired. In this base class,
     * this simply returns the shuffle-table and its hashes.
     * Child classes can modify this. e.g. Groupby may do a drop-duplicates on
     * the shuffle buffer in the nunique-only case.
     *
     * @return std::tuple<
     * std::shared_ptr<table_info>,
     * std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>,
     * std::shared_ptr<uint32_t[]>,
     * std::unique_ptr<uint8_t[]> shuffle_table, dict_hashes, shuffle_hashes
     * keep_row_bitmap. Only keep_row_bitmap is optional may be nullptr.
     */
    virtual std::tuple<
        /*shuffle_table*/ std::shared_ptr<table_info>,
        /*dict_hashes*/
        std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>,
        /*shuffle_hashes*/ std::shared_ptr<uint32_t[]>,
        /*keep_row_bitmap*/ std::unique_ptr<uint8_t[]>>
    GetShuffleTableAndHashes();

    /**
     * @brief Helper function for ShuffleIfRequired. This is called after every
     * shuffle. This resets the shuffle table buffer (which only resets the
     * size, without releasing any memory).
     *
     */
    virtual void ResetAfterShuffle();

    /**
     * @brief API to report the input and shuffle batch size at every iteration.
     * This is meant to be called either during AppendBatch or in
     * `GroupbyState::UpdateShuffleGroupsAndCombine`.
     *
     * @param batch_size Size of the input batch
     * @param shuffle_batch_size Size of the shuffle batch (out of the input
     * batch)
     */
    inline void UpdateAppendBatchSize(uint64_t batch_size,
                                      uint64_t shuffle_batch_size) {
        this->max_input_batch_size_since_prev_shuffle =
            std::max(this->max_input_batch_size_since_prev_shuffle, batch_size);
        this->max_shuffle_batch_size_since_prev_shuffle =
            std::max(this->max_shuffle_batch_size_since_prev_shuffle,
                     shuffle_batch_size);
    }

    /// @brief Number of shuffle keys.
    const uint64_t n_keys;

   private:
    /// @brief Reference to the iteration counter of the parent operator /
    /// operator-stage.
    const uint64_t& curr_iter;
    /// @brief Reference to the synchronization frequency variable of the parent
    /// operator. This will be modified as part of shuffle.
    /// This needs to be maintained in the operator because it's also used by
    /// the operator for its is_last synchronization.
    int64_t& sync_freq;
    /// @brief Counter of number of syncs since previous sync freq update.
    /// Set to -1 if not updating adaptively (user has specified sync freq).
    int64_t adaptive_sync_counter = -1;
    /// @brief The iteration number of last shuffle (used for adaptive sync
    /// estimation)
    uint64_t prev_shuffle_iter = 0;
    /// @brief Max input batch size seen since the previous shuffle.
    uint64_t max_input_batch_size_since_prev_shuffle = 0;
    /// @brief Max shuffle batch size seen since the previous shuffle.
    uint64_t max_shuffle_batch_size_since_prev_shuffle = 0;
    /// @brief Number of ranks.
    int n_pes;
    /// @brief Estimated size of a row based on just the dtypes.
    const size_t row_bytes_guesstimate;
    /// @brief Operator ID of the parent operator. Used for debug prints.
    const int64_t parent_op_id = -1;
    /// @brief Number of syncs after which we should re-evaluate the sync
    /// frequency.
    int64_t sync_update_freq = DEFAULT_SYNC_UPDATE_FREQ;
    /// @brief Threshold to use to decide when we should shuffle.
    const int64_t shuffle_threshold = DEFAULT_SHUFFLE_THRESHOLD;
    /// @brief Print information about the shuffle state during initialization,
    /// during every shuffle and after sync frequency is updated.
    bool debug_mode = false;
    /// @brief Metrics for query profile.
    IncrementalShuffleMetrics metrics;

    /**
     * @brief Helper function for ShuffleIfRequired. This determines whether we
     * should shuffle at this iteration based on the synchronization frequency,
     * size of the shuffle table (if it's larger than this->shuffle_threshold on
     * any rank) and whether it's the last iteration
     * ('is_last').
     * This also updates the synchronization frequency (based on current shuffle
     * buffer size and number of iterations from previous shuffle) if we're
     * using adaptive synchronization and this is an update iteration.
     *
     * @param is_last Whether this is the last iteration.
     * @return true Should shuffle in this iteration.
     * @return false Should not shuffle in this iteration.
     */
    bool check_if_shuffle_this_iter_and_update_sync_iter(const bool is_last);

    /**
     * @brief Helper function to get the dictionary hashes for key columns.
     *
     * @return
     * std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
     */
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
    get_dict_hashes_for_keys();

    /**
     * @brief Helper function to unify the dictionaries of the shuffle output
     * with the dictionaries in dictionary-builders.
     *
     * @param in_table Input table.
     * @return std::shared_ptr<table_info> Table with unified dictionaries.
     */
    std::shared_ptr<table_info> unify_table_dicts(
        const std::shared_ptr<table_info>& in_table);
};
