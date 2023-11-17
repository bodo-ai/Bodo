#pragma once

#include "_table_builder.h"

// Shuffle when streaming shuffle buffers are larger than 50MB
// TODO(ehsan): tune this parameter
static char* __env_threshold_str = std::getenv("BODO_SHUFFLE_THRESHOLD");
const int SHUFFLE_THRESHOLD = __env_threshold_str != nullptr
                                  ? std::stoi(__env_threshold_str)
                                  : 50 * 1024 * 1024;

// Factor in determining whether shuffle buffer is large enough to need cleared
const float SHUFFLE_BUFFER_CUTOFF_MULTIPLIER = 3.0;

// Minimum utilization of shuffle buffer, used as a factor in determining when
// to clear
const float SHUFFLE_BUFFER_MIN_UTILIZATION = 0.5;

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

#ifndef SYNC_UPDATE_FREQ
// Update sync freq every 10 syncs
#define SYNC_UPDATE_FREQ 10
#endif

/**
 * @brief Common hash-shuffle functionality for streaming operators such as
 * HashJoin and Groupby.
 *
 */
class IncrementalShuffleState {
   public:
    /// @brief Schema of the shuffle table
    const std::unique_ptr<bodo::Schema> schema;
    /// @brief Dictionary builder for the dictionary-encoded columns. Note that
    /// these are only for top-level dictionaries and not for dictionary-encoded
    /// fields within nested data types.
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    /// @brief Shuffle data buffer.
    std::unique_ptr<TableBuildBuffer> table_buffer;

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
        const uint64_t n_keys_, const uint64_t& curr_iter_,
        int64_t& sync_freq_);

    /**
     * @brief Calculate initial synchronization frequency if syncing
     * adaptively.
     * This must be called in the first iteration. It estimates how many
     * iterations it will take to for the shuffle buffer size of any rank to be
     * larger than SHUFFLE_THRESHOLD based on the size of the first input batch.
     * 'sync_freq' will be modified accordingly.
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

   protected:
    /**
     * @brief Helper function for ShuffleIfRequired. In this base class,
     * this simply be the shuffle-table and its hashes.
     * Child classes can modify this. e.g. Groupby may do a drop-duplicates on
     * the shuffle buffer in the nunique-only case.
     *
     * @return std::tuple<
     * std::shared_ptr<table_info>,
     * std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>,
     * std::shared_ptr<uint32_t[]>> shuffle_table, dict_hashes, shuffle_hashes.
     */
    virtual std::tuple<
        /*shuffle_table*/ std::shared_ptr<table_info>,
        /*dict_hashes*/
        std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>,
        /*shuffle_hashes*/ std::shared_ptr<uint32_t[]>>
    GetShuffleTableAndHashes();

    /**
     * @brief Helper function for ShuffleIfRequired. This is called after every
     * shuffle. This resets the shuffle table buffer (which only resets the
     * size, without releasing any memory).
     *
     */
    virtual void ResetAfterShuffle();

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
    /// @brief Number of ranks.
    int n_pes;

    /**
     * @brief Helper function for ShuffleIfRequired. This determines whether we
     * should shuffle at this iteration based on the synchronization frequency,
     * size of the shuffle table (if it's larger than SHUFFLE_THRESHOLD on any
     * rank) and whether it's the last iteration
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
