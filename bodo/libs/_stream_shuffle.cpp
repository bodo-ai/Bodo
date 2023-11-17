#include "_stream_shuffle.h"
#include <mpi.h>
#include "_array_hash.h"
#include "_shuffle.h"

IncrementalShuffleState::IncrementalShuffleState(
    const std::vector<int8_t>& arr_c_types_,
    const std::vector<int8_t>& arr_array_types_,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
    const uint64_t n_keys_, const uint64_t& curr_iter_, int64_t& sync_freq_)
    : schema(bodo::Schema::Deserialize(arr_array_types_, arr_c_types_)),
      dict_builders(dict_builders_),
      table_buffer(std::make_unique<TableBuildBuffer>(this->schema,
                                                      this->dict_builders)),
      n_keys(n_keys_),
      curr_iter(curr_iter_),
      sync_freq(sync_freq_),
      // Update number of sync iterations adaptively based on batch byte size
      // if sync_freq == -1 (user hasn't specified number of syncs)
      adaptive_sync_counter(this->sync_freq == -1 ? 0 : -1) {
    MPI_Comm_size(MPI_COMM_WORLD, &(this->n_pes));
}

void IncrementalShuffleState::Initialize(
    const std::shared_ptr<table_info>& sample_in_table_batch,
    bool is_parallel) {
    assert(this->curr_iter == 0);
    if (this->adaptive_sync_counter != -1 && is_parallel && this->n_pes > 1) {
        // Get max batch size of ranks
        int64_t in_data_size =
            table_local_memory_size(sample_in_table_batch, false);
        int64_t max_in_data_size;
        MPI_Allreduce(&in_data_size, &max_in_data_size, 1, MPI_INT64_T, MPI_MAX,
                      MPI_COMM_WORLD);

        // Estimate shuffle buffer size assuming (p-1)/p portion of batch is
        // shuffled each iteration
        int64_t shuffle_size_iter =
            std::ceil(((this->n_pes - 1) / static_cast<double>(this->n_pes)) *
                      max_in_data_size);
        shuffle_size_iter = std::max<int64_t>(shuffle_size_iter, 1);
        int64_t shuffle_iters =
            std::max<int64_t>(SHUFFLE_THRESHOLD / shuffle_size_iter, 1);

        this->sync_freq = std::min<int64_t>(DEFAULT_SYNC_ITERS, shuffle_iters);
    } else {
        // make sure sync_freq isn't -1 just in case
        this->sync_freq =
            this->sync_freq == -1 ? DEFAULT_SYNC_ITERS : this->sync_freq;
    }
}

void IncrementalShuffleState::AppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::vector<bool>& append_rows) {
    this->table_buffer->ReserveTable(in_table);
    this->table_buffer->UnsafeAppendBatch(in_table, append_rows);
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
        int64_t local_shuffle_buffer_size = table_local_memory_size(
            this->table_buffer->data_table, /*include_dict_size*/ false);
        int64_t reduced_shuffle_buffer_size;
        MPI_Allreduce(&local_shuffle_buffer_size, &reduced_shuffle_buffer_size,
                      1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
        bool shuffle_now = reduced_shuffle_buffer_size >= SHUFFLE_THRESHOLD;

        // Estimate how many iterations it will take until we need to shuffle
        // again and update sync frequency (every SYNC_UPDATE_FREQ syncs and
        // only if adaptive).
        if (reduced_shuffle_buffer_size > 0 &&
            this->curr_iter > this->prev_shuffle_iter &&
            this->adaptive_sync_counter != -1 &&
            ((this->adaptive_sync_counter + 1) % SYNC_UPDATE_FREQ == 0)) {
            // It took (this->curr_iter - this->prev_shuffle_iter) iterations to
            // accumulate a buffer of size reduced_shuffle_buffer_size. Based on
            // this, the new shuffle frequency should be:
            this->sync_freq = (SHUFFLE_THRESHOLD *
                               (this->curr_iter - this->prev_shuffle_iter)) /
                              reduced_shuffle_buffer_size;
            this->sync_freq = std::max<uint64_t>(this->sync_freq, 1);
            this->sync_freq =
                std::min<uint64_t>(this->sync_freq, DEFAULT_SYNC_ITERS);
        }
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
