#include "_stream_shuffle.h"
#include <mpi.h>
#include <iostream>
#include <numeric>
#include "_array_hash.h"
#include "_memory_budget.h"
#include "_shuffle.h"

IncrementalShuffleState::IncrementalShuffleState(
    const std::vector<int8_t>& arr_c_types_,
    const std::vector<int8_t>& arr_array_types_,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
    const uint64_t n_keys_, const uint64_t& curr_iter_, int64_t& sync_freq_,
    int64_t parent_op_id_)
    : schema(bodo::Schema::Deserialize(arr_array_types_, arr_c_types_)),
      dict_builders(dict_builders_),
      table_buffer(std::make_unique<TableBuildBuffer>(this->schema,
                                                      this->dict_builders)),
      n_keys(n_keys_),
      curr_iter(curr_iter_),
      sync_freq(sync_freq_),
      // Update number of sync iterations adaptively based on batch byte size
      // if sync_freq == -1 (user hasn't specified number of syncs)
      adaptive_sync_counter(this->sync_freq == -1 ? 0 : -1),
      row_bytes_guesstimate(get_row_bytes(this->schema)),
      parent_op_id(parent_op_id_) {
    MPI_Comm_size(MPI_COMM_WORLD, &(this->n_pes));
    if (char* debug_env_ = std::getenv("BODO_DEBUG_STREAM_SHUFFLE")) {
        this->debug_mode = !std::strcmp(debug_env_, "1");
    }
}

void IncrementalShuffleState::Initialize(
    const std::shared_ptr<table_info>& sample_in_table_batch,
    bool is_parallel) {
    assert(this->curr_iter == 0);
    if (this->adaptive_sync_counter != -1 && is_parallel && this->n_pes > 1) {
        int64_t bytes_per_row_est;
        const uint64_t n_rows = sample_in_table_batch->nrows();
        uint64_t global_max_n_rows;
        MPI_Allreduce(&n_rows, &global_max_n_rows, 1, MPI_UINT64_T, MPI_MAX,
                      MPI_COMM_WORLD);
        if (global_max_n_rows == 0) {
            // Some operators, especially those deep in a pipeline, may not
            // receive any input in their first iteration. In those cases, we
            // need to make a guess based on just the dtypes.
            bytes_per_row_est = this->row_bytes_guesstimate;
            // Sync more frequently until we see some data.
            this->sync_update_freq = 1;
        } else {
            // If we do have some rows, get the average size of each row.
            // This should be much more accurate than the dtype based
            // guesstimate.
            if (n_rows == 0) {
                bytes_per_row_est = 0;
            } else {
                int64_t in_data_size =
                    table_local_memory_size(sample_in_table_batch, false);
                bytes_per_row_est =
                    std::ceil(static_cast<double>(in_data_size) /
                              static_cast<double>(n_rows));
            }
        }

        // Handle sparse inputs (common at beginning) by extrapolating our per
        // row bytes estimates to at least STREAMING_BATCH_SIZE rows. This is
        // intentionally conservative to prevent shuffle buffers from blowing up
        // in size. If the batch sizes are consistently small in the future, we
        // will self-correct during re-estimation (in
        // check_if_shuffle_this_iter_and_update_sync_iter).
        int64_t n_rows_per_batch_est =
            std::max(static_cast<int64_t>(n_rows),
                     static_cast<int64_t>(STREAMING_BATCH_SIZE));
        // Assume that (p-1)/p portion of batch is shuffled each iteration
        n_rows_per_batch_est =
            std::ceil((static_cast<double>(this->n_pes - 1) /
                       static_cast<double>(this->n_pes)) *
                      static_cast<double>(n_rows_per_batch_est));
        n_rows_per_batch_est = std::max<int64_t>(n_rows_per_batch_est, 1);
        int64_t batch_data_size_est = n_rows_per_batch_est * bytes_per_row_est;

        int64_t max_batch_data_size_est;
        MPI_Allreduce(&batch_data_size_est, &max_batch_data_size_est, 1,
                      MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);

        if (max_batch_data_size_est > 0) {
            // Calculate sync freq based on max batch size estimate across all
            // ranks:
            int64_t shuffle_iters = std::max<int64_t>(
                SHUFFLE_THRESHOLD / max_batch_data_size_est, 1);
            this->sync_freq =
                std::min<int64_t>(DEFAULT_SYNC_ITERS, shuffle_iters);
        } else {
            // Should never happen, but just in case.
            this->sync_freq = DEFAULT_SYNC_ITERS;
        }

    } else {
        // make sure sync_freq isn't -1 just in case
        this->sync_freq =
            this->sync_freq == -1 ? DEFAULT_SYNC_ITERS : this->sync_freq;
    }

    if (this->debug_mode) {
        std::cerr
            << "[DEBUG] IncrementalShuffleState::Initialize[PARENT_OP_ID: "
            << this->parent_op_id << "]: No. of rows in sample input batch: "
            << sample_in_table_batch->nrows()
            << ", No. of columns: " << sample_in_table_batch->ncols()
            << ", Estimated Sync Freq: " << this->sync_freq
            << ", Sync Update Freq: " << this->sync_update_freq
            << ", Shuffle Threshold: "
            << BytesToHumanReadableString(SHUFFLE_THRESHOLD) << std::endl;
    }
}

void IncrementalShuffleState::AppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::vector<bool>& append_rows) {
    this->table_buffer->ReserveTable(in_table);
    uint64_t append_rows_sum =
        std::accumulate(append_rows.begin(), append_rows.end(), (uint64_t)0);
    this->table_buffer->UnsafeAppendBatch(in_table, append_rows,
                                          append_rows_sum);
    // Update max_input_batch_size_since_prev_shuffle and
    // max_shuffle_batch_size_since_prev_shuffle
    this->UpdateAppendBatchSize(in_table->nrows(), append_rows_sum);
}

bool IncrementalShuffleState::check_if_shuffle_this_iter_and_update_sync_iter(
    const bool is_last) {
    if (is_last) {
        return true;

    }
    // Use iter + 1 to avoid a sync on the first iteration
    else if ((this->curr_iter + 1) % this->sync_freq == 0) {
        // shuffle now if shuffle buffer size of any rank is larger than
        // SHUFFLE_THRESHOLD
        const size_t n_rows = this->table_buffer->data_table->nrows();
        int64_t local_shuffle_buffer_size = table_local_memory_size(
            this->table_buffer->data_table, /*include_dict_size*/ false);
        int64_t reduced_shuffle_buffer_size;
        MPI_Allreduce(&local_shuffle_buffer_size, &reduced_shuffle_buffer_size,
                      1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
        bool shuffle_now = reduced_shuffle_buffer_size >= SHUFFLE_THRESHOLD;

        // Estimate how many iterations it will take until we need to shuffle
        // again and update sync frequency (every this->sync_update_freq syncs
        // and only if adaptive).
        if ((this->curr_iter > this->prev_shuffle_iter) &&
            (this->adaptive_sync_counter != -1) &&
            /*There's no shuffle when n_pes == 1, so we can skip re-estimation*/
            (this->n_pes > 1) &&
            ((this->adaptive_sync_counter + 1) % this->sync_update_freq == 0)) {
            int64_t old_sync_freq = this->sync_freq;

            // Get the max input batch size across all ranks since the last
            // shuffle.
            uint64_t global_max_input_batch_size_since_last_shuffle = 0;
            MPI_Allreduce(&(this->max_input_batch_size_since_prev_shuffle),
                          &global_max_input_batch_size_since_last_shuffle, 1,
                          MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

            int64_t batch_size_est;
            int64_t bytes_per_row_est;
            if (global_max_input_batch_size_since_last_shuffle == 0) {
                // If we haven't seen any data since the last shuffle,
                // make a guesstimate.
                batch_size_est =
                    std::ceil((static_cast<double>(this->n_pes - 1) /
                               static_cast<double>(this->n_pes)) *
                              static_cast<double>(STREAMING_BATCH_SIZE));
                bytes_per_row_est = this->row_bytes_guesstimate;
                // Since we haven't seen any data, re-estimate in the next sync
                // instead of waiting for DEFAULT_SYNC_UPDATE_FREQ syncs.
                this->sync_update_freq = 1;
            } else {
                // Estimate batch size based on the shuffle buffer.
                batch_size_est =
                    std::ceil(static_cast<double>(n_rows) /
                              (this->curr_iter - this->prev_shuffle_iter));

                int64_t sync_update_freq_local_rec = 0;
                if (batch_size_est == 0) {
                    // If we haven't seen any data, then we should recommend
                    // updating sync freq in the next sync.
                    sync_update_freq_local_rec = 1;
                } else if (
                    batch_size_est <
                    std::ceil(
                        0.25 *
                        this->max_shuffle_batch_size_since_prev_shuffle)) {
                    // If the average size is way too low compared to the max
                    // shuffle batch size, we have probably just begun seeing
                    // data recently, so we should remain conservative and sync
                    // more frequently.
                    batch_size_est =
                        this->max_shuffle_batch_size_since_prev_shuffle;
                    sync_update_freq_local_rec = 1;
                } else {
                    // If we have seen sufficient data, we should recommend
                    // using the default sync freq update cadence. The estimate
                    // will be based on the shuffle buffer.
                    sync_update_freq_local_rec =
                        static_cast<int64_t>(DEFAULT_SYNC_UPDATE_FREQ);
                }

                // Estimate row size based on the shuffle buffer. Even with
                // sparse buffers, this should be quite accurate.
                bytes_per_row_est =
                    n_rows > 0 ? (std::ceil(static_cast<double>(
                                                local_shuffle_buffer_size) /
                                            static_cast<double>(n_rows)))
                               : 0;

                // Get the most optimistic recommendation for sync_update_freq.
                int64_t global_sync_update_freq_local_rec = 0;
                MPI_Allreduce(&sync_update_freq_local_rec,
                              &global_sync_update_freq_local_rec, 1,
                              MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
                this->sync_update_freq = global_sync_update_freq_local_rec;
            }

            // Estimate the size of each shuffle batch based on stats from
            // the last (this->curr_iter - this->prev_shuffle_iter) iterations:
            int64_t local_shuffle_batch_size_est =
                batch_size_est * bytes_per_row_est;
            // Get the max estimate across all ranks.
            int64_t global_shuffle_batch_size_est = 0;
            MPI_Allreduce(&local_shuffle_batch_size_est,
                          &global_shuffle_batch_size_est, 1, MPI_INT64_T,
                          MPI_MAX, MPI_COMM_WORLD);

            if (global_shuffle_batch_size_est > 0) {
                // In the last (this->curr_iter - this->prev_shuffle_iter)
                // iterations, each shuffle batch had a size of approximately
                // global_shuffle_batch_size_est. Based on this, the new
                // shuffle frequency should be:
                this->sync_freq =
                    SHUFFLE_THRESHOLD / global_shuffle_batch_size_est;
                this->sync_freq = std::max<int64_t>(this->sync_freq, 1);
                this->sync_freq =
                    std::min<int64_t>(this->sync_freq, DEFAULT_SYNC_ITERS);
            } else {
                // Should never happen, but just in case.
                this->sync_freq = DEFAULT_SYNC_ITERS;
            }
            if (this->debug_mode && (old_sync_freq != this->sync_freq)) {
                size_t avg_shuffle_batch_size_since_prev_shuffle =
                    std::ceil(static_cast<double>(n_rows) /
                              (this->curr_iter - this->prev_shuffle_iter));
                std::cerr << "[DEBUG] "
                             "IncrementalShuffleState[PARENT_OP_ID: "
                          << this->parent_op_id << "][Iter: " << this->curr_iter
                          << "] Updating Sync Freq from " << old_sync_freq
                          << " to " << this->sync_freq
                          << ". Previous shuffle iter: "
                          << this->prev_shuffle_iter
                          << ", Max input batch size since prev shuffle: "
                          << this->max_input_batch_size_since_prev_shuffle
                          << ", Max shuffle batch size since prev shuffle: "
                          << this->max_shuffle_batch_size_since_prev_shuffle
                          << ", Avg shuffle batch size since prev shuffle: "
                          << avg_shuffle_batch_size_since_prev_shuffle
                          << ", Sync Update Freq: " << this->sync_update_freq
                          << ", is_last: " << is_last << std::endl;
            }
        }
        // Increment the sync counter
        this->adaptive_sync_counter = this->adaptive_sync_counter != -1
                                          ? this->adaptive_sync_counter + 1
                                          : this->adaptive_sync_counter;

        return shuffle_now;
    } else {
        return false;
    }
}

std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
IncrementalShuffleState::get_dict_hashes_for_keys() {
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = std::make_shared<
            bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>(
            this->n_keys);
    for (uint64_t i = 0; i < this->n_keys; i++) {
        if (this->dict_builders[i] == nullptr) {
            (*dict_hashes)[i] = nullptr;
        } else {
            (*dict_hashes)[i] = this->dict_builders[i]->GetDictionaryHashes();
        }
    }
    return dict_hashes;
}

std::shared_ptr<table_info> IncrementalShuffleState::unify_table_dicts(
    const std::shared_ptr<table_info>& in_table) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(in_table->ncols());
    for (size_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        std::shared_ptr<array_info> out_arr;
        if (in_arr->arr_type != bodo_array_type::DICT) {
            out_arr = in_arr;
        } else {
            out_arr = this->dict_builders[i]->UnifyDictionaryArray(in_arr);
        }
        out_arrs.emplace_back(out_arr);
    }
    return std::make_shared<table_info>(out_arrs);
}

void IncrementalShuffleState::ResetAfterShuffle() {
    // Reset the build shuffle buffer. This will also
    // reset the dictionaries to point to the shared dictionaries
    // and reset the dictionary related flags.
    // This is crucial for correctness.
    // If the build shuffle buffer is too large and utilization is below
    // SHUFFLE_BUFFER_MIN_UTILIZATION, it will be freed and reallocated.
    size_t capacity = this->table_buffer->EstimatedSize();
    if (capacity > (SHUFFLE_BUFFER_CUTOFF_MULTIPLIER * SHUFFLE_THRESHOLD) &&
        (capacity * SHUFFLE_BUFFER_MIN_UTILIZATION) >
            table_local_memory_size(this->table_buffer->data_table, false)) {
        this->table_buffer.reset(
            new TableBuildBuffer(this->schema, this->dict_builders));
    } else {
        this->table_buffer->Reset();
    }
    // Reset batch size counts
    this->max_shuffle_batch_size_since_prev_shuffle = 0;
    this->max_input_batch_size_since_prev_shuffle = 0;
}

std::tuple<
    std::shared_ptr<table_info>,
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>,
    std::shared_ptr<uint32_t[]>>
IncrementalShuffleState::GetShuffleTableAndHashes() {
    std::shared_ptr<table_info> shuffle_table = this->table_buffer->data_table;
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = this->get_dict_hashes_for_keys();

    // NOTE: shuffle hashes need to be consistent with partition hashes
    // above. Since we're using dict_hashes, global dictionaries are not
    // required.
    std::shared_ptr<uint32_t[]> shuffle_hashes = hash_keys_table(
        shuffle_table, this->n_keys, SEED_HASH_PARTITION, /*is_parallel*/ true,
        /*global_dict_needed*/ false, dict_hashes);

    return std::make_tuple(shuffle_table, dict_hashes, shuffle_hashes);
}

std::optional<std::shared_ptr<table_info>>
IncrementalShuffleState::ShuffleIfRequired(const bool is_last) {
    bool shuffle_now =
        this->check_if_shuffle_this_iter_and_update_sync_iter(is_last);
    if (!shuffle_now) {
        return {};
    }

    if (this->debug_mode) {
        size_t avg_shuffle_batch_size_since_prev_shuffle = std::ceil(
            static_cast<double>(this->table_buffer->data_table->nrows()) /
            (this->curr_iter - this->prev_shuffle_iter));
        std::cerr
            << "[DEBUG] "
               "IncrementalShuffleState::ShuffleIfRequired[PARENT_OP_ID: "
            << this->parent_op_id << "] Shuffling on iter " << this->curr_iter
            << ". Shuffle-Buffer-Capacity: "
            << BytesToHumanReadableString(this->table_buffer->EstimatedSize())
            << ", Shuffle-Buffer-Size: "
            << BytesToHumanReadableString(table_local_memory_size(
                   this->table_buffer->data_table, /*include_dict_size*/ false))
            << ", No. of rows: " << this->table_buffer->data_table->nrows()
            << ", Sync Freq: " << this->sync_freq
            << ", Prev Shuffle Iter: " << this->prev_shuffle_iter
            << ", Max input batch size since prev shuffle: "
            << this->max_input_batch_size_since_prev_shuffle
            << ", Max shuffle batch size since prev shuffle: "
            << this->max_shuffle_batch_size_since_prev_shuffle
            << ", Avg shuffle batch size since prev shuffle: "
            << avg_shuffle_batch_size_since_prev_shuffle
            << ", Sync Update Freq: " << this->sync_update_freq
            << ", is_last: " << is_last << std::endl;
    }

    auto [shuffle_table, dict_hashes, shuffle_hashes] =
        this->GetShuffleTableAndHashes();
    dict_hashes.reset();

    // make dictionaries global for shuffle
    for (size_t i = 0; i < shuffle_table->ncols(); i++) {
        std::shared_ptr<array_info> arr = shuffle_table->columns[i];
        if (arr->arr_type == bodo_array_type::DICT) {
            make_dictionary_global_and_unique(arr, /*is_parallel*/ true);
        }
    }

    // Shuffle the data
    mpi_comm_info comm_info_table(shuffle_table->columns, shuffle_hashes,
                                  /*is_parallel*/ true);
    std::shared_ptr<table_info> new_data =
        shuffle_table_kernel(std::move(shuffle_table), shuffle_hashes,
                             comm_info_table, /*is_parallel*/ true);
    shuffle_hashes.reset();

    this->ResetAfterShuffle();
    this->prev_shuffle_iter = this->curr_iter;

    // Unify dictionaries to allow consistent hashing and fast key
    // comparison using indices
    return this->unify_table_dicts(new_data);
}

void IncrementalShuffleState::Finalize() {
    this->dict_builders.clear();
    this->table_buffer.reset();
}
