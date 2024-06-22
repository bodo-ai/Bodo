#include "_stream_shuffle.h"

#include <iostream>
#include <numeric>
#include <optional>

#include <arrow/util/bit_util.h>
#include <mpi.h>

#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_memory_budget.h"
#include "_query_profile_collector.h"
#include "_shuffle.h"
#include "_utils.h"

void IncrementalShuffleMetrics::add_to_metrics(
    std::vector<MetricBase>& metrics) {
    metrics.emplace_back(
        TimerMetric("shuffle_buffer_append_time", this->append_time));
    metrics.emplace_back(TimerMetric("shuffle_time", this->shuffle_time));
    metrics.emplace_back(StatMetric("n_shuffles", this->n_shuffles, true));
    metrics.emplace_back(TimerMetric("shuffle_hash_time", this->hash_time));
    metrics.emplace_back(TimerMetric("shuffle_dict_unification_time",
                                     this->dict_unification_time));
    metrics.emplace_back(
        StatMetric("shuffle_total_appended_nrows", this->total_appended_nrows));
    metrics.emplace_back(
        StatMetric("shuffle_total_sent_nrows", this->total_sent_nrows));
    metrics.emplace_back(
        StatMetric("shuffle_total_recv_nrows", this->total_recv_nrows));
    metrics.emplace_back(StatMetric("shuffle_total_approx_sent_size_bytes",
                                    this->total_approx_sent_size_bytes));
    metrics.emplace_back(StatMetric("shuffle_total_recv_size_bytes",
                                    this->total_recv_size_bytes));
    metrics.emplace_back(StatMetric("shuffle_buffer_peak_capacity_bytes",
                                    this->peak_capacity_bytes));
    metrics.emplace_back(StatMetric("shuffle_buffer_peak_utilization_bytes",
                                    this->peak_utilization_bytes));
    metrics.emplace_back(
        StatMetric("shuffle_n_buffer_resets", this->n_buffer_resets));
}

int64_t get_shuffle_threshold() {
    // Get shuffle threshold from an env var if provided.
    if (char* threshold_env_ = std::getenv("BODO_SHUFFLE_THRESHOLD")) {
        return std::stoi(threshold_env_);
    }
    // Get system memory size (rank)
    int64_t sys_mem_bytes = bodo::BufferPool::Default()->get_sys_memory_bytes();
    if (sys_mem_bytes == -1) {
        // Use default threshold if system memory size is not known.
        return DEFAULT_SHUFFLE_THRESHOLD;
    } else {
        int64_t sys_mem_mib = std::ceil(sys_mem_bytes / (1024.0 * 1024.0));
        // Else, return a value between MIN and MAX threshold based
        // on the available memory.
        return std::min(
            std::max(static_cast<int64_t>(MIN_SHUFFLE_THRESHOLD),
                     sys_mem_mib * DEFAULT_SHUFFLE_THRESHOLD_PER_MiB),
            static_cast<int64_t>(MAX_SHUFFLE_THRESHOLD));
    }
}

IncrementalShuffleState::IncrementalShuffleState(
    const std::vector<int8_t>& arr_c_types_,
    const std::vector<int8_t>& arr_array_types_,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
    const uint64_t n_keys_, const uint64_t& curr_iter_, int64_t& sync_freq_,
    int64_t parent_op_id_)
    : IncrementalShuffleState(
          bodo::Schema::Deserialize(arr_array_types_, arr_c_types_),
          dict_builders_, n_keys_, curr_iter_, sync_freq_, parent_op_id_){};

IncrementalShuffleState::IncrementalShuffleState(
    std::shared_ptr<bodo::Schema> schema_,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
    const uint64_t n_keys_, const uint64_t& curr_iter_, int64_t& sync_freq_,
    int64_t parent_op_id_)
    : schema(std::move(schema_)),
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
      parent_op_id(parent_op_id_),
      shuffle_threshold(get_shuffle_threshold()) {
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
                this->shuffle_threshold / max_batch_data_size_est, 1);
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
            << BytesToHumanReadableString(this->shuffle_threshold) << std::endl;
    }
}

void IncrementalShuffleState::AppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::vector<bool>& append_rows) {
    time_pt start = start_timer();
    this->table_buffer->ReserveTable(in_table);
    uint64_t append_rows_sum =
        std::accumulate(append_rows.begin(), append_rows.end(), (uint64_t)0);
    this->table_buffer->UnsafeAppendBatch(in_table, append_rows,
                                          append_rows_sum);
    this->metrics.append_time += end_timer(start);
    this->metrics.total_appended_nrows += append_rows_sum;
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
        // this->shuffle_threshold
        const size_t n_rows = this->table_buffer->data_table->nrows();
        int64_t local_shuffle_buffer_size = table_local_memory_size(
            this->table_buffer->data_table, /*include_dict_size*/ false);
        int64_t reduced_shuffle_buffer_size;
        MPI_Allreduce(&local_shuffle_buffer_size, &reduced_shuffle_buffer_size,
                      1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
        bool shuffle_now =
            reduced_shuffle_buffer_size >= this->shuffle_threshold;

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
                    this->shuffle_threshold / global_shuffle_batch_size_est;
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
        if (this->dict_builders[i] == nullptr) {
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
    this->metrics.peak_capacity_bytes = std::max(
        this->metrics.peak_capacity_bytes, static_cast<int64_t>(capacity));
    int64_t buffer_used_size =
        table_local_memory_size(this->table_buffer->data_table, false);
    this->metrics.peak_utilization_bytes =
        std::max(this->metrics.peak_utilization_bytes, buffer_used_size);
    if (capacity >
            (SHUFFLE_BUFFER_CUTOFF_MULTIPLIER * this->shuffle_threshold) &&
        (capacity * SHUFFLE_BUFFER_MIN_UTILIZATION) > buffer_used_size) {
        this->table_buffer.reset(
            new TableBuildBuffer(this->schema, this->dict_builders));
        this->metrics.n_buffer_resets++;
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
    std::shared_ptr<uint32_t[]>, std::unique_ptr<uint8_t[]>>
IncrementalShuffleState::GetShuffleTableAndHashes() {
    std::shared_ptr<table_info> shuffle_table = this->table_buffer->data_table;
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes = this->get_dict_hashes_for_keys();

    // NOTE: shuffle hashes need to be consistent with partition hashes
    // above. Since we're using dict_hashes, global dictionaries are not
    // required.
    time_pt start = start_timer();
    std::shared_ptr<uint32_t[]> shuffle_hashes = hash_keys_table(
        shuffle_table, this->n_keys, SEED_HASH_PARTITION, /*is_parallel*/ true,
        /*global_dict_needed*/ false, dict_hashes);
    this->metrics.hash_time += end_timer(start);

    return std::make_tuple(shuffle_table, dict_hashes, shuffle_hashes,
                           std::unique_ptr<uint8_t[]>(nullptr));
}

template <bodo_array_type::arr_type_enum arr_type>
inline void send_shuffle_null_bitmask(
    AsyncShuffleSendState& send_state, const MPI_Comm shuffle_comm,
    const mpi_comm_info& comm_info, const std::shared_ptr<array_info>& send_arr,
    std::vector<int>& curr_tags, size_t p) {
    MPI_Datatype mpi_type_null = MPI_UNSIGNED_CHAR;
    const void* buf =
        send_arr->null_bitmask<arr_type>() +
        comm_info.send_disp_null[p] * numpy_item_size[Bodo_CTypes::UINT8];

    MPI_Request send_req_null;
    // TODO: check err return
    MPI_Issend(buf, comm_info.send_count_null[p], mpi_type_null, p,
               curr_tags[p]++, shuffle_comm, &send_req_null);
    send_state.send_requests.push_back(send_req_null);
}

template <bodo_array_type::arr_type_enum arr_type>
void recv_null_bitmask(std::shared_ptr<array_info>& out_arr,
                       const MPI_Comm shuffle_comm, const int source,
                       int& curr_tag, AsyncShuffleRecvState& recv_state) {
    const MPI_Datatype mpi_type_null = MPI_UNSIGNED_CHAR;
    int recv_size_null = arrow::bit_util::BytesForBits(out_arr->length);
    MPI_Request recv_req_null;
    MPI_Irecv(out_arr->null_bitmask<arr_type>(), recv_size_null, mpi_type_null,
              source, curr_tag++, shuffle_comm, &recv_req_null);
    recv_state.recv_requests.push_back(recv_req_null);
}
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(arr_type == bodo_array_type::NUMPY)
void send_shuffle_data(AsyncShuffleSendState& send_state, MPI_Comm shuffle_comm,
                       const mpi_comm_info& comm_info,
                       const mpi_str_comm_info& str_comm_info,
                       const std::shared_ptr<array_info>& send_arr,
                       std::vector<int>& curr_tags) {
    const MPI_Datatype mpi_type = get_MPI_typ<dtype>();
    for (int p = 0; p < comm_info.n_pes; p++) {
        if (comm_info.send_count[p] == 0) {
            continue;
        }
        const void* buff = send_arr->data1<arr_type>() +
                           (numpy_item_size[dtype] * comm_info.send_disp[p]);
        MPI_Request send_req;
        // TODO: check err return
        MPI_Issend(buff, comm_info.send_count[p], mpi_type, p, curr_tags[p]++,
                   shuffle_comm, &send_req);
        send_state.send_requests.push_back(send_req);
    }
}

template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(arr_type == bodo_array_type::NUMPY)
std::shared_ptr<array_info> recv_shuffle_data(
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state) {
    const MPI_Datatype mpi_type = get_MPI_typ<dtype>();
    recv_state.initSimpleArrayLen(source, curr_tag, shuffle_comm, mpi_type);

    std::shared_ptr<array_info> out_arr = alloc_array_top_level<arr_type>(
        recv_state.simple_array_len, 0, 0, arr_type, dtype, -1, 0, 0);
    MPI_Request recv_req;
    MPI_Irecv(out_arr->data1<arr_type>(), recv_state.simple_array_len, mpi_type,
              source, curr_tag++, shuffle_comm, &recv_req);
    recv_state.recv_requests.push_back(recv_req);
    return out_arr;
}

template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
             dtype != Bodo_CTypes::_BOOL)
void send_shuffle_data(AsyncShuffleSendState& send_state,
                       const MPI_Comm shuffle_comm,
                       const mpi_comm_info& comm_info,
                       const mpi_str_comm_info& str_comm_info,
                       const std::shared_ptr<array_info>& send_arr,
                       std::vector<int>& curr_tags) {
    const MPI_Datatype mpi_type = get_MPI_typ<dtype>();
    for (int p = 0; p < comm_info.n_pes; p++) {
        if (comm_info.send_count[p] == 0) {
            continue;
        }

        MPI_Request send_req;
        const void* buff = send_arr->data1<arr_type>() +
                           (numpy_item_size[dtype] * comm_info.send_disp[p]);
        // TODO: check err return
        MPI_Issend(buff, comm_info.send_count[p], mpi_type, p, curr_tags[p]++,
                   shuffle_comm, &send_req);

        send_state.send_requests.push_back(send_req);
        send_shuffle_null_bitmask<arr_type>(send_state, shuffle_comm, comm_info,
                                            send_arr, curr_tags, p);
    }
}

template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
             dtype != Bodo_CTypes::_BOOL)
std::shared_ptr<array_info> recv_shuffle_data(
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state) {
    const MPI_Datatype mpi_type = get_MPI_typ<dtype>();
    recv_state.initSimpleArrayLen(source, curr_tag, shuffle_comm, mpi_type);

    std::shared_ptr<array_info> out_arr = alloc_array_top_level<arr_type>(
        recv_state.simple_array_len, 0, 0, arr_type, dtype, -1, 0, 0);

    MPI_Request recv_req;
    MPI_Irecv(out_arr->data1<arr_type>(), recv_state.simple_array_len, mpi_type,
              source, curr_tag++, shuffle_comm, &recv_req);
    recv_state.recv_requests.push_back(recv_req);

    recv_null_bitmask<arr_type>(out_arr, shuffle_comm, source, curr_tag,
                                recv_state);
    return out_arr;
}

template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
             dtype == Bodo_CTypes::_BOOL)
void send_shuffle_data(AsyncShuffleSendState& send_state,
                       const MPI_Comm shuffle_comm,
                       const mpi_comm_info& comm_info,
                       const mpi_str_comm_info& str_comm_info,
                       const std::shared_ptr<array_info>& send_arr,
                       std::vector<int>& curr_tags) {
    const MPI_Datatype mpi_type = get_MPI_typ<dtype>();
    for (int p = 0; p < comm_info.n_pes; p++) {
        if (comm_info.send_count[p] == 0) {
            continue;
        }

        // Since the data is stored as bits, we need to send the number
        // of bits used in the last byte so the receiver knows the
        // array's length
        // TODO: technically we only need to do this if
        // there hasn't been a simple array sent first
        send_state.bits_in_last_byte.push_back(comm_info.send_count[p] % 8);
        // TODO: check err return
        MPI_Request bits_in_last_byte_req;
        MPI_Issend(&send_state.bits_in_last_byte.back(), sizeof(uint8_t),
                   mpi_type, p, curr_tags[p]++, shuffle_comm,
                   &bits_in_last_byte_req);
        send_state.send_requests.push_back(bits_in_last_byte_req);

        // Send the data
        MPI_Request send_req;
        char* buff = send_arr->data1<arr_type>() + comm_info.send_disp_null[p];
        MPI_Issend(buff, comm_info.send_count_null[p], mpi_type, p,
                   curr_tags[p]++, shuffle_comm, &send_req);
        send_state.send_requests.push_back(send_req);

        send_shuffle_null_bitmask<arr_type>(send_state, shuffle_comm, comm_info,
                                            send_arr, curr_tags, p);
    }
}

template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
             dtype == Bodo_CTypes::_BOOL)
std::shared_ptr<array_info> recv_shuffle_data(
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state) {
    const MPI_Datatype mpi_type = get_MPI_typ<dtype>();

    // Add a new entry to the bits_in_last_byte vector
    recv_state.bits_in_last_byte.emplace_back();
    // Only adjust the resulting length if we don't already have a simple array
    std::get<0>(recv_state.bits_in_last_byte.back()) =
        recv_state.simple_array_len == -1;
    uint8_t& bits_in_last_byte =
        std::get<1>(recv_state.bits_in_last_byte.back());

    // Receive a message telling us how many bits of the last byte are valid
    MPI_Request bits_in_last_byte_req;
    MPI_Irecv(&bits_in_last_byte, sizeof(uint8_t), mpi_type, source, curr_tag++,
              shuffle_comm, &bits_in_last_byte_req);
    recv_state.recv_requests.push_back(bits_in_last_byte_req);

    int recv_size;
    size_t out_arr_len = -1;
    if (recv_state.simple_array_len == -1) {
        // If we don't have a simple array length, we need to get the incoming
        // message size we can't set simple_array_len here because the size of
        // the incoming message might not be the same as the size of the array
        // (some of the bits might not be valid)
        MPI_Status status;
        MPI_Probe(source, curr_tag, shuffle_comm, &status);
        MPI_Get_count(&status, mpi_type, &recv_size);

        out_arr_len = recv_size * 8;
    } else {
        // If we already have a simple array length, we can just use that to get
        // the size of the incoming message and set the array's length
        out_arr_len = recv_state.simple_array_len;
        recv_size = arrow::bit_util::BytesForBits(out_arr_len);
    }
    std::shared_ptr<array_info> out_arr = alloc_array_top_level<arr_type>(
        out_arr_len, 0, 0, arr_type, dtype, -1, 0, 0);

    MPI_Request recv_req;
    MPI_Irecv(out_arr->data1<arr_type>(), recv_size, mpi_type, source,
              curr_tag++, shuffle_comm, &recv_req);
    recv_state.recv_requests.push_back(recv_req);

    recv_null_bitmask<arr_type>(out_arr, shuffle_comm, source, curr_tag,
                                recv_state);
    return out_arr;
}

template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(arr_type == bodo_array_type::STRING)
void send_shuffle_data(AsyncShuffleSendState& send_state,
                       const MPI_Comm shuffle_comm,
                       const mpi_comm_info& comm_info,
                       const mpi_str_comm_info& str_comm_info,
                       const std::shared_ptr<array_info>& send_arr,
                       std::vector<int>& curr_tags) {
    const MPI_Datatype data_mpi_type = MPI_UNSIGNED_CHAR;
    // Fill_send_array converts offsets to send lengths of type uint32_t
    const MPI_Datatype len_mpi_type = MPI_UINT32_T;
    for (int p = 0; p < comm_info.n_pes; p++) {
        if (comm_info.send_count[p] == 0) {
            continue;
        }
        MPI_Request data_send_req;
        const void* data_buff =
            send_arr->data1<arr_type>() + str_comm_info.send_disp_sub[p];
        // TODO: check err return
        MPI_Issend(data_buff, str_comm_info.send_count_sub[p], data_mpi_type, p,
                   curr_tags[p]++, shuffle_comm, &data_send_req);
        send_state.send_requests.push_back(data_send_req);

        MPI_Request len_send_req;
        const void* len_buff = send_arr->data2<arr_type>() +
                               (sizeof(uint32_t) * comm_info.send_disp[p]);

        MPI_Issend(len_buff, comm_info.send_count[p], len_mpi_type, p,
                   curr_tags[p]++, shuffle_comm, &len_send_req);
        send_state.send_requests.push_back(len_send_req);

        send_shuffle_null_bitmask<arr_type>(send_state, shuffle_comm, comm_info,
                                            send_arr, curr_tags, p);
    }
}

template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(arr_type == bodo_array_type::STRING)
std::shared_ptr<array_info> recv_shuffle_data(
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state) {
    const MPI_Datatype data_mpi_type = MPI_UNSIGNED_CHAR;
    // Fill_send_array converts offsets to lengths of type uint32_t
    const MPI_Datatype len_mpi_type = MPI_UINT32_T;

    // Get the sizes of each incoming message
    MPI_Status data_status;
    int recv_size_sub;
    MPI_Probe(source, curr_tag, shuffle_comm, &data_status);
    MPI_Get_count(&data_status, data_mpi_type, &recv_size_sub);

    recv_state.initSimpleArrayLen(source, curr_tag + 1, shuffle_comm,
                                  data_mpi_type);

    std::shared_ptr<array_info> out_arr = alloc_array_top_level<arr_type>(
        recv_state.simple_array_len, recv_size_sub, 0, arr_type, dtype, -1, 0,
        0);

    MPI_Request data_req;
    MPI_Irecv(out_arr->data1<arr_type>(), recv_size_sub, data_mpi_type, source,
              curr_tag++, shuffle_comm, &data_req);
    recv_state.recv_requests.push_back(data_req);

    MPI_Request len_req;
    // Receive the lens, we know we can fit them in the offset buffer because
    // sizeof(offset_t) >= sizeof(uint32_t)
    MPI_Irecv(out_arr->data2<arr_type, offset_t>(), recv_state.simple_array_len,
              len_mpi_type, source, curr_tag++, shuffle_comm, &len_req);
    recv_state.recv_requests.push_back(len_req);

    recv_null_bitmask<arr_type>(out_arr, shuffle_comm, source, curr_tag,
                                recv_state);
    return out_arr;
}

template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(arr_type == bodo_array_type::DICT)
void send_shuffle_data(AsyncShuffleSendState& send_state,
                       const MPI_Comm shuffle_comm,
                       const mpi_comm_info& comm_info,
                       const mpi_str_comm_info& str_comm_info,
                       const std::shared_ptr<array_info>& send_arr,
                       std::vector<int>& curr_tags) {
    throw std::runtime_error("send_shuffle_data: DICT not implemented");
}

#define SEND_SHUFFLE_DATA(ARR_TYPE, DTYPE)                                 \
    send_shuffle_data<ARR_TYPE, DTYPE>(send_states.back(), shuffle_comm,   \
                                       comm_info, str_comm_info, send_arr, \
                                       curr_tags);

void shuffle_issend(std::shared_ptr<table_info> in_table,
                    const std::shared_ptr<uint32_t[]>& hashes,
                    std::vector<AsyncShuffleSendState>& send_states,
                    MPI_Comm shuffle_comm) {
    mpi_comm_info comm_info(in_table->columns, hashes,
                            /*is_parallel*/ true, /*filter*/ nullptr, nullptr,
                            /*keep_filter_misses*/ false, /*send_only*/ true);

    std::vector<int> curr_tags(comm_info.n_pes, 0);

    for (uint64_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        mpi_str_comm_info str_comm_info(in_arr, comm_info,
                                        /*send_only*/ true);
        std::shared_ptr<array_info> send_arr = alloc_array_top_level(
            comm_info.n_rows_send, str_comm_info.n_sub_send, 0,
            in_arr->arr_type, in_arr->dtype, -1, 2 * comm_info.n_pes,
            in_arr->num_categories);
        fill_send_array(send_arr, in_arr, comm_info, str_comm_info, true);
        send_states.emplace_back(send_arr);
        if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            switch (in_arr->dtype) {
                case Bodo_CTypes::INT8:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::INT8);
                    break;
                case Bodo_CTypes::INT16:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::INT16);
                    break;
                case Bodo_CTypes::INT32:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::INT32);
                    break;
                case Bodo_CTypes::INT64:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::INT64);
                    break;
                case Bodo_CTypes::UINT8:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::UINT8);
                    break;
                case Bodo_CTypes::UINT16:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::UINT16);
                    break;
                case Bodo_CTypes::UINT32:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::UINT32);
                    break;
                case Bodo_CTypes::UINT64:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::UINT64);
                    break;
                case Bodo_CTypes::FLOAT32:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::FLOAT32);
                    break;
                case Bodo_CTypes::FLOAT64:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::FLOAT64);
                    break;
                case Bodo_CTypes::_BOOL:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::_BOOL);
                    break;
                case Bodo_CTypes::DATETIME:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::DATETIME);
                    break;
                case Bodo_CTypes::TIMEDELTA:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::TIMEDELTA);
                    break;
                case Bodo_CTypes::TIME:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::TIME);
                    break;
                case Bodo_CTypes::DATE:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::DATE);
                    break;
                case Bodo_CTypes::DECIMAL:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::DECIMAL);
                    break;
                case Bodo_CTypes::INT128:
                    SEND_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                      Bodo_CTypes::INT128);
                    break;
                default:
                    throw std::runtime_error("Unsupported dtype " +
                                             GetDtype_as_string(in_arr->dtype) +
                                             "for shuffle "
                                             "send nullable int/bool");
                    break;
            }
        } else if (in_arr->arr_type == bodo_array_type::NUMPY) {
            switch (in_arr->dtype) {
                case Bodo_CTypes::INT8:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::INT8);
                    break;
                case Bodo_CTypes::INT16:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::INT16);
                    break;
                case Bodo_CTypes::INT32:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::INT32);
                    break;
                case Bodo_CTypes::INT64:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::INT64);
                    break;
                case Bodo_CTypes::UINT8:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::UINT8);
                    break;
                case Bodo_CTypes::UINT16:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::UINT16);
                    break;
                case Bodo_CTypes::UINT32:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::UINT32);
                    break;
                case Bodo_CTypes::UINT64:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::UINT64);
                    break;
                case Bodo_CTypes::FLOAT32:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::FLOAT32);
                    break;
                case Bodo_CTypes::FLOAT64:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::FLOAT64);
                    break;
                case Bodo_CTypes::_BOOL:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::_BOOL);
                    break;
                case Bodo_CTypes::DATETIME:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::DATETIME);
                    break;
                case Bodo_CTypes::TIMEDELTA:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::TIMEDELTA);
                    break;
                case Bodo_CTypes::TIME:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::TIME);
                    break;
                case Bodo_CTypes::DATE:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::DATE);
                    break;
                case Bodo_CTypes::DECIMAL:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::DECIMAL);
                    break;
                case Bodo_CTypes::INT128:
                    SEND_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                      Bodo_CTypes::INT128);
                    break;
                default:
                    throw std::runtime_error("Unsupported dtype " +
                                             GetDtype_as_string(in_arr->dtype) +
                                             "for shuffle "
                                             "send numpy");
                    break;
            }
        } else if (in_arr->arr_type == bodo_array_type::STRING) {
            if (in_arr->dtype == Bodo_CTypes::STRING) {
                SEND_SHUFFLE_DATA(bodo_array_type::STRING, Bodo_CTypes::STRING);
            } else {
                assert(in_arr->dtype == Bodo_CTypes::BINARY);
                SEND_SHUFFLE_DATA(bodo_array_type::STRING, Bodo_CTypes::BINARY);
            }
            //} else if (in_arr->arr_type == bodo_array_type::DICT) {
            //    assert(in_arr->dtype == Bodo_CTypes::STRING);
            //    SEND_SHUFFLE_DATA(bodo_array_type::DICT, Bodo_CTypes::STRING);
            //} else if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
            //    assert(in_arr->dtype == Bodo_CTypes::LIST);
            //    SEND_SHUFFLE_DATA(bodo_array_type::ARRAY_ITEM,
            //    Bodo_CTypes::LIST);
            //} else if (in_arr->arr_type == bodo_array_type::MAP) {
            //    assert(in_arr->dtype == Bodo_CTypes::MAP);
            //    SEND_SHUFFLE_DATA(bodo_array_type::MAP, Bodo_CTypes::MAP);
            //} else if (in_arr->arr_type == bodo_array_type::STRUCT) {
            //    assert(in_arr->dtype == Bodo_CTypes::STRUCT);
            //    SEND_SHUFFLE_DATA(bodo_array_type::STRUCT,
            //    Bodo_CTypes::STRUCT);
            //} else if (in_arr->arr_type == bodo_array_type::TIMESTAMPTZ) {
            //    assert(in_arr->dtype == Bodo_CTypes::TIMESTAMPTZ);
            //    SEND_SHUFFLE_DATA(bodo_array_type::TIMESTAMPTZ,
            //                      Bodo_CTypes::TIMESTAMPTZ);
        } else {
            throw std::runtime_error(
                "Unsupported array type for shuffle send: " +
                GetArrType_as_string(in_arr->arr_type));
        }
    }
}
#undef SEND_SHUFFLE_DATA

#define RECV_SHUFFLE_DATA(ARR_TYPE, DTYPE)                             \
    out_arr = recv_shuffle_data<ARR_TYPE, DTYPE>(shuffle_comm, source, \
                                                 curr_tag, recv_state);

void shuffle_irecv(std::shared_ptr<table_info> in_table, MPI_Comm shuffle_comm,
                   std::vector<AsyncShuffleRecvState>& recv_states) {
    // TODO: add output buffer size limit
    while (true) {
        int flag;
        MPI_Status status;
        // TODO: check err return
        MPI_Iprobe(MPI_ANY_SOURCE, 0, shuffle_comm, &flag, &status);
        if (!flag) {
            break;
        }

        int source = status.MPI_SOURCE;
        assert(in_table->ncols() > 0);

        // Post irecv for all data
        int curr_tag = 0;
        recv_states.emplace_back();
        AsyncShuffleRecvState& recv_state = recv_states.back();
        for (uint64_t i = 0; i < in_table->ncols(); i++) {
            std::shared_ptr<array_info>& in_arr = in_table->columns[i];
            std::shared_ptr<array_info> out_arr;
            if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                switch (in_arr->dtype) {
                    case Bodo_CTypes::INT8:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::INT8);
                        break;
                    case Bodo_CTypes::INT16:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::INT16);
                        break;
                    case Bodo_CTypes::INT32:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::INT32);
                        break;
                    case Bodo_CTypes::INT64:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::INT64);
                        break;
                    case Bodo_CTypes::UINT8:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::UINT8);
                        break;
                    case Bodo_CTypes::UINT16:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::UINT16);
                        break;
                    case Bodo_CTypes::UINT32:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::UINT32);
                        break;
                    case Bodo_CTypes::UINT64:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::UINT64);
                        break;
                    case Bodo_CTypes::FLOAT32:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::FLOAT32);
                        break;
                    case Bodo_CTypes::FLOAT64:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::FLOAT64);
                        break;
                    case Bodo_CTypes::_BOOL:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::_BOOL);
                        break;
                    case Bodo_CTypes::DATETIME:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::DATETIME);
                        break;
                    case Bodo_CTypes::TIMEDELTA:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::TIMEDELTA);
                        break;
                    case Bodo_CTypes::TIME:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::TIME);
                        break;
                    case Bodo_CTypes::DATE:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::DATE);
                        break;
                    case Bodo_CTypes::DECIMAL:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::DECIMAL);
                        break;
                    case Bodo_CTypes::INT128:
                        RECV_SHUFFLE_DATA(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::INT128);
                        break;
                    default:
                        throw std::runtime_error(
                            "Unsupported dtype " +
                            GetDtype_as_string(in_arr->dtype) +
                            "for shuffle "
                            "recv nullable int/bool");
                        break;
                }
            } else if (in_arr->arr_type == bodo_array_type::NUMPY) {
                switch (in_arr->dtype) {
                    case Bodo_CTypes::INT8:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::INT8);
                        break;
                    case Bodo_CTypes::INT16:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::INT16);
                        break;
                    case Bodo_CTypes::INT32:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::INT32);
                        break;
                    case Bodo_CTypes::INT64:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::INT64);
                        break;
                    case Bodo_CTypes::UINT8:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::UINT8);
                        break;
                    case Bodo_CTypes::UINT16:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::UINT16);
                        break;
                    case Bodo_CTypes::UINT32:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::UINT32);
                        break;
                    case Bodo_CTypes::UINT64:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::UINT64);
                        break;
                    case Bodo_CTypes::FLOAT32:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::FLOAT32);
                        break;
                    case Bodo_CTypes::FLOAT64:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::FLOAT64);
                        break;
                    case Bodo_CTypes::_BOOL:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::_BOOL);
                        break;
                    case Bodo_CTypes::DATETIME:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::DATETIME);
                        break;
                    case Bodo_CTypes::TIMEDELTA:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::TIMEDELTA);
                        break;
                    case Bodo_CTypes::TIME:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::TIME);
                        break;
                    case Bodo_CTypes::DATE:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::DATE);
                        break;
                    case Bodo_CTypes::DECIMAL:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::DECIMAL);
                        break;
                    case Bodo_CTypes::INT128:
                        RECV_SHUFFLE_DATA(bodo_array_type::NUMPY,
                                          Bodo_CTypes::INT128);
                        break;
                    default:
                        throw std::runtime_error(
                            "Unsupported dtype " +
                            GetDtype_as_string(in_arr->dtype) +
                            "for shuffle "
                            "recv numpy");
                        break;
                }
            } else if (in_arr->arr_type == bodo_array_type::STRING) {
                if (in_arr->dtype == Bodo_CTypes::STRING) {
                    RECV_SHUFFLE_DATA(bodo_array_type::STRING,
                                      Bodo_CTypes::STRING);
                } else {
                    assert(in_arr->dtype == Bodo_CTypes::BINARY);
                    RECV_SHUFFLE_DATA(bodo_array_type::STRING,
                                      Bodo_CTypes::BINARY);
                }
                //} else if (in_arr->arr_type == bodo_array_type::DICT) {
                //    assert(in_arr->dtype == Bodo_CTypes::STRING);
                //    RECV_SHUFFLE_DATA(bodo_array_type::DICT,
                //    Bodo_CTypes::STRING);
                //} else if (in_arr->arr_type ==
                // bodo_array_type::ARRAY_ITEM) {
                //    assert(in_arr->dtype == Bodo_CTypes::LIST);
                //    RECV_SHUFFLE_DATA(bodo_array_type::ARRAY_ITEM,
                //    Bodo_CTypes::LIST);
                //} else if (in_arr->arr_type == bodo_array_type::MAP) {
                //    assert(in_arr->dtype == Bodo_CTypes::MAP);
                //    RECV_SHUFFLE_DATA(bodo_array_type::MAP,
                //    Bodo_CTypes::MAP);
                //} else if (in_arr->arr_type == bodo_array_type::STRUCT) {
                //    assert(in_arr->dtype == Bodo_CTypes::STRUCT);
                //    RECV_SHUFFLE_DATA(bodo_array_type::STRUCT,
                //    Bodo_CTypes::STRUCT);
                //} else if (in_arr->arr_type ==
                // bodo_array_type::TIMESTAMPTZ) {
                //    assert(in_arr->dtype == Bodo_CTypes::TIMESTAMPTZ);
                //    RECV_SHUFFLE_DATA(bodo_array_type::TIMESTAMPTZ,
                //                      Bodo_CTypes::TIMESTAMPTZ);
            } else {
                throw std::runtime_error(
                    "Unsupported array type for shuffle recv: " +
                    GetArrType_as_string(in_arr->arr_type));
            }
            out_arr->precision = in_arr->precision;
            out_arr->scale = in_arr->scale;
            recv_state.out_arrs.push_back(out_arr);
        }
    }
}
#undef RECV_SHUFFLE_DATA

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

    auto [shuffle_table, dict_hashes, shuffle_hashes, keep_row_bitmask] =
        this->GetShuffleTableAndHashes();
    dict_hashes.reset();

    // Make dictionaries global for shuffle
    time_pt start = start_timer();
    for (size_t i = 0; i < shuffle_table->ncols(); i++) {
        std::shared_ptr<array_info> arr = shuffle_table->columns[i];
        if (arr->arr_type == bodo_array_type::DICT) {
            make_dictionary_global_and_unique(arr, /*is_parallel*/ true);
        }
    }
    this->metrics.dict_unification_time += end_timer(start);

    // Shuffle the data
    start = start_timer();
    mpi_comm_info comm_info_table(shuffle_table->columns, shuffle_hashes,
                                  /*is_parallel*/ true, /*filter*/ nullptr,
                                  keep_row_bitmask.get(),
                                  /*keep_filter_misses*/ false);
    this->metrics.total_sent_nrows += comm_info_table.n_rows_send;
    // TODO Make this exact by getting the size of the send_arr from the
    // intermediate arrays created during shuffle_table_kernel.
    int64_t shuffle_table_size = table_local_memory_size(shuffle_table, false);
    if ((shuffle_table_size > 0) && (comm_info_table.n_rows_send > 0) &&
        (shuffle_table->nrows() > 0)) {
        this->metrics.total_approx_sent_size_bytes +=
            ((((double)comm_info_table.n_rows_send) /
              ((double)shuffle_table->nrows())) *
             shuffle_table_size);
    }

    std::shared_ptr<table_info> new_data =
        shuffle_table_kernel(std::move(shuffle_table), shuffle_hashes,
                             comm_info_table, /*is_parallel*/ true);
    this->metrics.shuffle_time += end_timer(start);
    this->metrics.n_shuffles++;
    shuffle_hashes.reset();

    this->metrics.total_recv_nrows += comm_info_table.n_rows_recv;
    this->metrics.total_recv_size_bytes +=
        table_local_memory_size(new_data, false);

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

bool AsyncShuffleRecvState::recvDone(TableBuildBuffer& out_builder) {
    // This could be optimized by allocating the required size upfront and
    // having the recv step fill it directly instead of each rank having its own
    // array and inserting them all into a builder
    int flag;
    MPI_Testall(recv_requests.size(), recv_requests.data(), &flag,
                MPI_STATUSES_IGNORE);
    if (flag) {
        size_t nullable_bool_count = 0;
        for (auto& arr : out_arrs) {
            if (arr->arr_type == bodo_array_type::STRING) {
                // Now all the data in data2 is lengths not offsets, we need to
                // convert
                static_assert(sizeof(uint64_t) == sizeof(offset_t),
                              "uint64_t and offset_t must have the same size");
                std::vector<uint32_t> lens(arr->length);
                memcpy(lens.data(), arr->data2<bodo_array_type::STRING>(),
                       arr->length * sizeof(uint32_t));
                convert_len_arr_to_offset(
                    lens.data(),
                    arr->data2<bodo_array_type::STRING, offset_t>(),
                    arr->length);
            } else if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                       arr->dtype == Bodo_CTypes::_BOOL) {
                // Calculate the length of the data array in bits
                auto [adjust_length, used_bits_in_last_byte] =
                    this->bits_in_last_byte[nullable_bool_count];
                if (adjust_length && used_bits_in_last_byte != 0) {
                    arr->length = arr->length - 8 + used_bits_in_last_byte;
                }
                nullable_bool_count++;
            }
        }

        std::shared_ptr<table_info> out_table =
            std::make_shared<table_info>(out_arrs);
        out_builder.ReserveTable(out_table);
        out_builder.UnsafeAppendBatch(out_table);
    }
    return flag;
}

void AsyncShuffleRecvState::initSimpleArrayLen(
    const int source, const int tag, const MPI_Comm comm,
    const MPI_Datatype mpi_datatype) {
    if (this->simple_array_len != -1) {
        return;
    }

    MPI_Status status;
    MPI_Probe(source, tag, comm, &status);
    int count;
    MPI_Get_count(&status, mpi_datatype, &count);
    this->simple_array_len = count;
}
