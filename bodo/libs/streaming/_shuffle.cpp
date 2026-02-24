#include "_shuffle.h"

#include <iostream>
#include <memory>
#include <numeric>
#include <optional>

#include <arrow/util/bit_util.h>
#include <fmt/format.h>

#include "../_array_hash.h"
#include "../_bodo_common.h"
#include "../_dict_builder.h"
#include "../_mpi.h"
#include "../_query_profile_collector.h"
#include "../_shuffle.h"
#include "../_utils.h"

void IncrementalShuffleMetrics::add_to_metrics(
    std::vector<MetricBase>& metrics) {
    metrics.emplace_back(
        TimerMetric("shuffle_buffer_append_time", this->append_time));
    metrics.emplace_back(
        TimerMetric("shuffle_send_time", this->shuffle_send_time));
    metrics.emplace_back(
        TimerMetric("shuffle_recv_time", this->shuffle_recv_time));
    metrics.emplace_back(TimerMetric("shuffle_send_finalization_time",
                                     this->shuffle_send_finalization_time));
    metrics.emplace_back(TimerMetric("shuffle_recv_finalization_time",
                                     this->shuffle_recv_finalization_time));
    metrics.emplace_back(
        StatMetric("n_shuffle_send", this->n_shuffle_send, false));
    metrics.emplace_back(
        StatMetric("n_shuffle_recv", this->n_shuffle_recv, false));
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
    metrics.emplace_back(StatMetric("shuffle_approx_sent_size_bytes_dicts",
                                    this->approx_sent_size_bytes_dicts));
    metrics.emplace_back(StatMetric("shuffle_total_recv_size_bytes",
                                    this->total_recv_size_bytes));
    metrics.emplace_back(StatMetric("shuffle_approx_recv_size_bytes_dicts",
                                    this->approx_recv_size_bytes_dicts));
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
          dict_builders_, n_keys_, curr_iter_, sync_freq_, parent_op_id_) {};

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
      shuffle_threshold(get_shuffle_threshold()),
      curr_iter(curr_iter_),
      row_bytes_guesstimate(get_row_bytes(this->schema)),
      parent_op_id(parent_op_id_) {
    MPI_Comm_size(MPI_COMM_WORLD, &(this->n_pes));
    if (char* debug_env_ = std::getenv("BODO_DEBUG_STREAM_SHUFFLE")) {
        this->debug_mode = !std::strcmp(debug_env_, "1");
    }
}

void IncrementalShuffleState::Initialize(
    const std::shared_ptr<table_info>& sample_in_table_batch, bool is_parallel,
    MPI_Comm _shuffle_comm) {
    assert(this->curr_iter == 0);
    this->shuffle_comm = _shuffle_comm;
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
        this->table_buffer = std::make_unique<TableBuildBuffer>(
            this->schema, this->dict_builders);
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

AsyncShuffleSendState::AsyncShuffleSendState(int starting_msg_tag_)
    : starting_msg_tag(starting_msg_tag_) {
    if (this->starting_msg_tag <= SHUFFLE_METADATA_MSG_TAG) {
        throw std::runtime_error(
            fmt::format("[AsyncShuffleSendState] Starting tag ({}) must be "
                        "larger than SHUFFLE_METADATA_MSG_TAG ({})!",
                        this->starting_msg_tag, SHUFFLE_METADATA_MSG_TAG));
    }
}

#define SEND_SHUFFLE_DATA(ARR_TYPE, DTYPE)                                   \
    send_shuffle_data<ARR_TYPE, DTYPE>(                                      \
        shuffle_comm, comm_info, comm_info_iter, str_comm_info_iter, in_arr, \
        curr_tags, must_shuffle_to_rank);
void AsyncShuffleSendState::send_shuffle_data_unknown_type(
    const MPI_Comm shuffle_comm, const mpi_comm_info& comm_info,
    comm_info_iter_t& comm_info_iter, str_comm_info_iter_t& str_comm_info_iter,
    const std::shared_ptr<array_info>& in_arr, std::vector<int>& curr_tags,
    std::vector<bool>& must_shuffle_to_rank) {
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
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::INT8);
                break;
            case Bodo_CTypes::INT16:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::INT16);
                break;
            case Bodo_CTypes::INT32:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::INT32);
                break;
            case Bodo_CTypes::INT64:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::INT64);
                break;
            case Bodo_CTypes::UINT8:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::UINT8);
                break;
            case Bodo_CTypes::UINT16:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::UINT16);
                break;
            case Bodo_CTypes::UINT32:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::UINT32);
                break;
            case Bodo_CTypes::UINT64:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::UINT64);
                break;
            case Bodo_CTypes::FLOAT32:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT32);
                break;
            case Bodo_CTypes::FLOAT64:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64);
                break;
            case Bodo_CTypes::_BOOL:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::_BOOL);
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
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::TIME);
                break;
            case Bodo_CTypes::DATE:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::DATE);
                break;
            case Bodo_CTypes::DECIMAL:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::DECIMAL);
                break;
            case Bodo_CTypes::INT128:
                SEND_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::INT128);
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
    } else if (in_arr->arr_type == bodo_array_type::DICT) {
        assert(in_arr->dtype == Bodo_CTypes::STRING);
        SEND_SHUFFLE_DATA(bodo_array_type::DICT, Bodo_CTypes::STRING);
    } else if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
        assert(in_arr->dtype == Bodo_CTypes::LIST);
        SEND_SHUFFLE_DATA(bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST);
    } else if (in_arr->arr_type == bodo_array_type::MAP) {
        assert(in_arr->dtype == Bodo_CTypes::MAP);
        SEND_SHUFFLE_DATA(bodo_array_type::MAP, Bodo_CTypes::MAP);
    } else if (in_arr->arr_type == bodo_array_type::STRUCT) {
        assert(in_arr->dtype == Bodo_CTypes::STRUCT);
        SEND_SHUFFLE_DATA(bodo_array_type::STRUCT, Bodo_CTypes::STRUCT);
    } else if (in_arr->arr_type == bodo_array_type::TIMESTAMPTZ) {
        assert(in_arr->dtype == Bodo_CTypes::TIMESTAMPTZ);
        SEND_SHUFFLE_DATA(bodo_array_type::TIMESTAMPTZ,
                          Bodo_CTypes::TIMESTAMPTZ);
    } else {
        throw std::runtime_error("Unsupported array type for shuffle send: " +
                                 GetArrType_as_string(in_arr->arr_type));
    }
}
#undef SEND_SHUFFLE_DATA

#define RECV_SHUFFLE_DATA(ARR_TYPE, DTYPE)                \
    return recv_shuffle_data<ARR_TYPE, DTYPE, top_level>( \
        data_type, shuffle_comm, source, curr_tag, recv_state, lens_iter);
template <bool top_level>
std::unique_ptr<array_info> recv_shuffle_data_unknown_type(
    const std::unique_ptr<bodo::DataType>& data_type,
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state, len_iter_t& lens_iter) {
    if (data_type->array_type == bodo_array_type::NULLABLE_INT_BOOL) {
        switch (data_type->c_type) {
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
                    "Unsupported c_type " +
                    GetArrType_as_string(data_type->c_type) +
                    "for shuffle "
                    "recv nullable int/bool");
                break;
        }
    } else if (data_type->array_type == bodo_array_type::NUMPY) {
        switch (data_type->c_type) {
            case Bodo_CTypes::INT8:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::INT8);
                break;
            case Bodo_CTypes::INT16:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::INT16);
                break;
            case Bodo_CTypes::INT32:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::INT32);
                break;
            case Bodo_CTypes::INT64:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::INT64);
                break;
            case Bodo_CTypes::UINT8:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::UINT8);
                break;
            case Bodo_CTypes::UINT16:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::UINT16);
                break;
            case Bodo_CTypes::UINT32:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::UINT32);
                break;
            case Bodo_CTypes::UINT64:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::UINT64);
                break;
            case Bodo_CTypes::FLOAT32:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT32);
                break;
            case Bodo_CTypes::FLOAT64:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64);
                break;
            case Bodo_CTypes::_BOOL:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::_BOOL);
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
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::TIME);
                break;
            case Bodo_CTypes::DATE:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::DATE);
                break;
            case Bodo_CTypes::DECIMAL:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::DECIMAL);
                break;
            case Bodo_CTypes::INT128:
                RECV_SHUFFLE_DATA(bodo_array_type::NUMPY, Bodo_CTypes::INT128);
                break;
            default:
                throw std::runtime_error(
                    "Unsupported c_type " +
                    GetArrType_as_string(data_type->c_type) +
                    "for shuffle "
                    "recv numpy");
                break;
        }
    } else if (data_type->array_type == bodo_array_type::STRING) {
        if (data_type->c_type == Bodo_CTypes::STRING) {
            RECV_SHUFFLE_DATA(bodo_array_type::STRING, Bodo_CTypes::STRING);
        } else {
            assert(data_type->c_type == Bodo_CTypes::BINARY);
            RECV_SHUFFLE_DATA(bodo_array_type::STRING, Bodo_CTypes::BINARY);
        }
    } else if (data_type->array_type == bodo_array_type::DICT) {
        assert(data_type->c_type == Bodo_CTypes::STRING);
        RECV_SHUFFLE_DATA(bodo_array_type::DICT, Bodo_CTypes::STRING);
    } else if (data_type->array_type == bodo_array_type::ARRAY_ITEM) {
        assert(data_type->c_type == Bodo_CTypes::LIST);
        RECV_SHUFFLE_DATA(bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST);
    } else if (data_type->array_type == bodo_array_type::MAP) {
        assert(data_type->c_type == Bodo_CTypes::MAP);
        RECV_SHUFFLE_DATA(bodo_array_type::MAP, Bodo_CTypes::MAP);
    } else if (data_type->array_type == bodo_array_type::STRUCT) {
        assert(data_type->c_type == Bodo_CTypes::STRUCT);
        RECV_SHUFFLE_DATA(bodo_array_type::STRUCT, Bodo_CTypes::STRUCT);
    } else if (data_type->array_type == bodo_array_type::TIMESTAMPTZ) {
        assert(data_type->c_type == Bodo_CTypes::TIMESTAMPTZ);
        RECV_SHUFFLE_DATA(bodo_array_type::TIMESTAMPTZ,
                          Bodo_CTypes::TIMESTAMPTZ);
    } else {
        throw std::runtime_error("Unsupported array type for shuffle recv: " +
                                 GetArrType_as_string(data_type->array_type));
    }
}
#undef RECV_SHUFFLE_DATA

AsyncShuffleSendState shuffle_issend(std::shared_ptr<table_info> in_table,
                                     const std::shared_ptr<uint32_t[]>& hashes,
                                     const uint8_t* keep_row_bitmask,
                                     MPI_Comm shuffle_comm,
                                     int starting_msg_tag) {
    mpi_comm_info comm_info(in_table->columns, hashes,
                            /*is_parallel*/ true, /*filter*/ nullptr,
                            keep_row_bitmask,
                            /*keep_filter_misses*/ false, /*send_only*/ true);

    AsyncShuffleSendState send_state(starting_msg_tag);
    send_state.send(in_table, comm_info, shuffle_comm);

    return send_state;
}

void AsyncShuffleRecvState::TryRecvMetadataAndAllocArrs(
    MPI_Comm& shuffle_comm) {
    // Only post irecv if we haven't already
    if (!recv_requests.empty()) {
        return;
    }
    std::optional<std::vector<uint64_t>> recv_metadata_opt =
        GetRecvMetadata(shuffle_comm);
    // Only post irecv if we have received the lens
    if (!recv_metadata_opt.has_value()) {
        return;
    }
    std::vector<uint64_t>& metadata_vec = recv_metadata_opt.value();

    // In the metadata, the starting tag to use is the first element, followed
    // by the lengths.
    int curr_tag = metadata_vec.at(0);
    len_iter_t lens_iter = metadata_vec.cbegin() + 1;

    // Post irecv for all data
    for (auto& data_type : schema->column_types) {
        std::shared_ptr<array_info> out_arr =
            recv_shuffle_data_unknown_type<true>(
                data_type, shuffle_comm, source, curr_tag, *this, lens_iter);
        out_arr->precision = data_type->precision;
        out_arr->scale = data_type->scale;
        out_arrs.push_back(out_arr);
    }
}

void shuffle_irecv(std::shared_ptr<table_info> in_table, MPI_Comm shuffle_comm,
                   std::vector<AsyncShuffleRecvState>& recv_states,
                   size_t max_recv_states) {
    assert(in_table->ncols() > 0);
    while (recv_states.size() < max_recv_states) {
        int flag;
        MPI_Status status;

        // NOTE: We use Improbe instead of Iprobe intentionally. Iprobe can
        // return true for the same message even when an Irecv for the message
        // has been posted (until the receive has actually begun). This can
        // cause hangs since we could end up posting two Irecvs for the same
        // message. Therefore, for robustness, we use Improbe, which returns a
        // message handle directly and exactly once.
        // 'PostLensRecv' uses `Imrecv` which will start receive on the
        // message using the message handle returned by Improbe.
        // Reference:
        // https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node70.htm which
        // states that "Unlike MPI_IPROBE, no other probe or receive operation
        // may match the message returned by MPI_IMPROBE.".
        MPI_Message m;
        CHECK_MPI(MPI_Improbe(MPI_ANY_SOURCE, SHUFFLE_METADATA_MSG_TAG,
                              shuffle_comm, &flag, &m, &status),
                  "shuffle_irecv: MPI error on MPI_Improbe:")
        if (!flag) {
            break;
        }

        AsyncShuffleRecvState recv_state(status.MPI_SOURCE, in_table->schema());

        recv_state.PostMetadataRecv(status, m);
        recv_states.push_back(std::move(recv_state));
    }

    // TODO(aneesh) it seems like a weird API to have this happen in
    // shuffle_irecv instead of exclusively happening in recvDone
    for (auto& recv_state : recv_states) {
        recv_state.TryRecvMetadataAndAllocArrs(shuffle_comm);
    }
}

int get_max_allowed_tag_value() {
    int flag = 0;
    void* tag_ub;
    CHECK_MPI(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &flag),
              "[get_max_allowed_tag_value] MPI Error in MPI_Comm_get_attr:");
    if (flag) {
        // This value is typically in the 10s or 100s of millions.
        return *(int*)tag_ub;
    } else {
        // If we cannot get it from MPI, use the value guaranteed by the MPI
        // standard.
        return 32767;
    }
}

int get_next_available_tag(std::unordered_set<int>& inflight_tags) {
    // 10000 should be more than enough tags for a single send.
    // TODO Use std::max(10000, req_tags) based on the table schema for full
    // robustness.
    constexpr int TAG_OFFSET = 10000;
    // Even the MPI standard guaranteed value (32767) is sufficient for posting
    // 2 messages at once with our 10000 offset.
    const int MAX_TAG = get_max_allowed_tag_value();

    for (int tag = SHUFFLE_METADATA_MSG_TAG + 1; tag < (MAX_TAG - TAG_OFFSET);
         tag += TAG_OFFSET) {
        if (!inflight_tags.contains(tag)) {
            return tag;
        }
    }
    return -1;
}

std::optional<std::shared_ptr<table_info>>
IncrementalShuffleState::ShuffleIfRequired(const bool is_last) {
    // Reduce MPI call overheads by communicating only every 10 iterations
    if (!(is_last || ((this->curr_iter % 10) == 0))) {
        return std::nullopt;
    }

    // recv data first, but avoid receiving too much data at once
    if ((this->recv_states.size() == 0) || !this->BuffersFull()) {
        time_pt start = start_timer();
        size_t prev_recv_states_size = this->recv_states.size();
        shuffle_irecv(this->table_buffer->data_table, this->shuffle_comm,
                      this->recv_states);
        this->metrics.n_shuffle_recv +=
            this->recv_states.size() - prev_recv_states_size;
        this->metrics.shuffle_recv_time += end_timer(start);
    }

    TableBuildBuffer out_builder(this->schema, this->dict_builders);

    time_pt start = start_timer();
    // Check for finished recvs
    consume_completed_recvs(this->recv_states, this->shuffle_comm,
                            this->dict_builders, this->metrics, out_builder);
    this->metrics.shuffle_recv_finalization_time += end_timer(start);

    std::optional<std::shared_ptr<table_info>> new_data =
        out_builder.data_table->nrows() != 0
            ? std::make_optional(out_builder.data_table)
            : std::nullopt;
    this->metrics.total_recv_nrows +=
        new_data.has_value() ? new_data.value()->nrows() : 0;
    this->metrics.total_recv_size_bytes +=
        new_data.has_value() ? table_local_memory_size(new_data.value(), false)
                             : 0;

    start = start_timer();
    // Remove send state if recv done
    std::erase_if(this->send_states, [&](AsyncShuffleSendState& s) {
        bool done = s.sendDone();
        if (done) {
            inflight_tags.erase(s.get_starting_msg_tag());
        }
        return done;
    });
    this->metrics.shuffle_send_finalization_time += end_timer(start);

    bool shuffle_now = this->ShouldShuffleAfterProcessing(is_last);
    if (!shuffle_now) {
        return new_data;
    }

    if (this->debug_mode) {
        size_t avg_shuffle_batch_size_since_prev_shuffle = std::ceil(
            static_cast<double>(this->table_buffer->data_table->nrows()) /
            (this->curr_iter - this->prev_shuffle_iter));
        std::cerr << "[DEBUG] "
                     "IncrementalShuffleState::ShuffleIfRequired[PARENT_OP_ID: "
                  << this->parent_op_id << "] Shuffling on iter "
                  << this->curr_iter << ". Shuffle-Buffer-Capacity: "
                  << BytesToHumanReadableString(
                         this->table_buffer->EstimatedSize())
                  << ", Shuffle-Buffer-Size: "
                  << BytesToHumanReadableString(
                         table_local_memory_size(this->table_buffer->data_table,
                                                 /*include_dict_size*/ false))
                  << ", No. of rows: "
                  << this->table_buffer->data_table->nrows()
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

    // Shuffle the data
    start = start_timer();

    this->metrics.total_sent_nrows += shuffle_table->nrows();
    // TODO: Make this exact by getting the size of the send_arr from the
    // intermediate arrays created during shuffle_table_kernel.
    int64_t shuffle_table_size = table_local_memory_size(shuffle_table, false);
    if ((shuffle_table_size > 0) && (shuffle_table->nrows() > 0)) {
        this->metrics.total_approx_sent_size_bytes += shuffle_table_size;
        // We need to multiply by n_pes because the dictionary is sent to each
        // rank in full
        this->metrics.approx_sent_size_bytes_dicts +=
            table_local_dictionary_memory_size(shuffle_table) * this->n_pes;
    }

    int start_tag = get_next_available_tag(this->inflight_tags);
    if (start_tag == -1) {
        throw std::runtime_error(
            "[IncrementalShuffleState::ShuffleIfRequired] Unable to get "
            "available MPI tag for shuffle send. All tags are inflight.");
    }
    this->send_states.push_back(
        shuffle_issend(std::move(shuffle_table), shuffle_hashes,
                       keep_row_bitmask.get(), this->shuffle_comm, start_tag));
    this->inflight_tags.insert(start_tag);

    this->metrics.shuffle_send_time += end_timer(start);
    this->metrics.n_shuffle_send++;
    shuffle_hashes.reset();

    this->ResetAfterShuffle();
    this->prev_shuffle_iter = this->curr_iter;

    return new_data;
}

bool IncrementalShuffleState::BuffersFull() {
    size_t total_buff_size = 0;
    for (AsyncShuffleSendState& s : this->send_states) {
        total_buff_size += s.GetTotalBufferSize();
    }
    for (AsyncShuffleRecvState& s : this->recv_states) {
        total_buff_size += s.GetTotalBufferSize();
    }
    return total_buff_size >
           (SHUFFLE_BUFFER_CUTOFF_MULTIPLIER * this->shuffle_threshold);
}

void IncrementalShuffleState::Finalize() {
    this->dict_builders.clear();
    this->table_buffer.reset();
}

bool IncrementalShuffleState::SendRecvEmpty() {
    return (this->send_states.empty() && this->recv_states.empty());
}

std::shared_ptr<array_info> AsyncShuffleRecvState::finalize_receive_array(
    const std::shared_ptr<array_info>& arr,
    const std::shared_ptr<DictionaryBuilder>& dict_builder,
    std::vector<uint32_t>& data_lens_vec, IncrementalShuffleMetrics& metrics) {
    if (arr->arr_type == bodo_array_type::STRING) {
        data_lens_vec.resize(arr->length);
        // Now all the data in data2 is lengths not offsets, we
        // need to convert
        static_assert(sizeof(uint64_t) == sizeof(offset_t),
                      "uint64_t and offset_t must have the same size");
        memcpy(data_lens_vec.data(), arr->data2<bodo_array_type::STRING>(),
               arr->length * sizeof(uint32_t));
        convert_len_arr_to_offset(
            data_lens_vec.data(),
            arr->data2<bodo_array_type::STRING, offset_t>(), arr->length);
    } else if (arr->arr_type == bodo_array_type::DICT) {
        time_pt start = start_timer();
        // Unify the dictionary but don't use the cache
        // because the dictionary is not shared across ranks so we're unlikely
        // to get a cache hit and will just evict entries that may actually be
        // useful.
        metrics.approx_recv_size_bytes_dicts +=
            array_dictionary_memory_size(arr);
        auto out_arr =
            dict_builder->UnifyDictionaryArray(arr, /*use_cache=*/false);
        metrics.dict_unification_time += end_timer(start);
        return out_arr;
    } else if (arr->arr_type == bodo_array_type::ARRAY_ITEM) {
        // All of the data in data1 is lengths not offsets, we
        // need to convert
        data_lens_vec.resize(arr->length);
        memcpy(data_lens_vec.data(), arr->data1<bodo_array_type::ARRAY_ITEM>(),
               arr->length * sizeof(uint32_t));
        convert_len_arr_to_offset(
            data_lens_vec.data(),
            arr->data1<bodo_array_type::ARRAY_ITEM, offset_t>(), arr->length);
        arr->child_arrays[0] = this->finalize_receive_array(
            arr->child_arrays[0], dict_builder->child_dict_builders[0],
            data_lens_vec, metrics);
    } else if (arr->arr_type == bodo_array_type::STRUCT) {
        for (size_t i = 0; i < arr->child_arrays.size(); ++i) {
            arr->child_arrays[i] = this->finalize_receive_array(
                arr->child_arrays[i], dict_builder->child_dict_builders[i],
                data_lens_vec, metrics);
        }
        // We need to adjust the struct array's length in case the first child
        // array's length was adjusted due to nullable bools.
        if (arr->child_arrays.size() != 0) {
            arr->length = arr->child_arrays[0]->length;
        }

        // This should get optimized out when asserts are disabled
        for (size_t i = 1; i < arr->child_arrays.size(); ++i) {
            assert(arr->child_arrays[i]->length == arr->length);
        }
    } else if (arr->arr_type == bodo_array_type::MAP) {
        arr->child_arrays[0] = this->finalize_receive_array(
            arr->child_arrays[0], dict_builder->child_dict_builders[0],
            data_lens_vec, metrics);
    }
    return arr;
}

std::pair<bool, std::shared_ptr<table_info>> AsyncShuffleRecvState::recvDone(
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    MPI_Comm shuffle_comm, IncrementalShuffleMetrics& metrics) {
    if (recv_requests.empty()) {
        // Try receiving the length again and see if we can populate the data
        // requests.
        TryRecvMetadataAndAllocArrs(shuffle_comm);
        if (recv_requests.empty()) {
            return std::make_pair(false, nullptr);
        }
    }

    // This could be optimized by allocating the required size upfront and
    // having the recv step fill it directly instead of each rank having its own
    // array and inserting them all into a builder
    int flag;
    CHECK_MPI_TEST_ALL(
        recv_requests, flag,
        "[AsyncShuffleRecvState::recvDone] MPI Error on MPI_Testall: ");

    std::shared_ptr<table_info> out_table = nullptr;
    if (flag) {
        std::vector<uint32_t> lens;
        std::vector<std::shared_ptr<array_info>> out_table_arrs;
        for (size_t i = 0; i < this->out_arrs.size(); ++i) {
            const std::shared_ptr<array_info>& arr = this->out_arrs[i];
            const std::shared_ptr<DictionaryBuilder>& dict_builder =
                dict_builders[i];
            out_table_arrs.push_back(
                this->finalize_receive_array(arr, dict_builder, lens, metrics));
        }

        out_table = std::make_shared<table_info>(out_table_arrs);
    }
    return std::make_pair(flag, out_table);
}

std::shared_ptr<array_info> AsyncShuffleSendState::addArray(
    const std::shared_ptr<array_info>& in_arr, const mpi_comm_info& comm_info,
    const mpi_str_comm_info& str_comm_info) {
    std::shared_ptr<array_info> send_arr = alloc_array_top_level(
        comm_info.n_rows_send, str_comm_info.n_sub_send, 0, in_arr->arr_type,
        in_arr->dtype, -1, 2 * comm_info.n_pes, in_arr->num_categories);
    fill_send_array(send_arr, in_arr, comm_info, str_comm_info, true);
    this->send_arrs.push_back(send_arr);
    return send_arr;
}

void AsyncShuffleSendState::send(const std::shared_ptr<table_info>& in_table,
                                 mpi_comm_info& comm_info,
                                 MPI_Comm shuffle_comm) {
    comm_infos_t sub_comm_infos = this->compute_comm_infos(in_table, comm_info);
    this->send_metadata(in_table, comm_info, sub_comm_infos, shuffle_comm);
    this->send_arrays(in_table, comm_info, sub_comm_infos, shuffle_comm);
}

/**
 * @brief Helper function to compute the communication info for the given array
 * and its children and store it in the comm_infos vectors. Only intended to be
 * called by compute_comm_infos.
 * @param in_arr The array for which to compute the communication info.
 * @param parent The parent communication info.
 * @param comm_infos The vector of communication info to which to append the
 * comm_info
 * @param str_comm_infos The vector of string communication info to which to
 * append the str_comm_info
 */
void compute_comm_infos_helper(const std::shared_ptr<array_info>& in_arr,
                               mpi_comm_info& parent,
                               std::vector<mpi_comm_info>& comm_infos,
                               std::vector<mpi_str_comm_info>& str_comm_infos) {
    str_comm_infos.emplace_back(in_arr, parent, true);
    if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
        comm_infos.emplace_back(
            in_arr, parent, in_arr->child_arrays[0]->null_bitmask() != nullptr,
            true);
        compute_comm_infos_helper(in_arr->child_arrays[0], comm_infos.back(),
                                  comm_infos, str_comm_infos);
    } else if (in_arr->arr_type == bodo_array_type::STRUCT) {
        for (const auto& child : in_arr->child_arrays) {
            compute_comm_infos_helper(child, parent, comm_infos,
                                      str_comm_infos);
        }
    } else if (in_arr->arr_type == bodo_array_type::MAP) {
        compute_comm_infos_helper(in_arr->child_arrays[0], parent, comm_infos,
                                  str_comm_infos);
    } else if (in_arr->arr_type == bodo_array_type::DICT) {
        // Compute the comm info for the offsets
        compute_comm_infos_helper(in_arr->child_arrays[1], parent, comm_infos,
                                  str_comm_infos);
    }
}

AsyncShuffleSendState::comm_infos_t AsyncShuffleSendState::compute_comm_infos(
    const std::shared_ptr<table_info>& in_table, mpi_comm_info& comm_info) {
    std::vector<mpi_comm_info> comm_infos;
    std::vector<mpi_str_comm_info> str_comm_infos;
    for (const auto& arr : in_table->columns) {
        compute_comm_infos_helper(arr, comm_info, comm_infos, str_comm_infos);
    }
    return {comm_infos, str_comm_infos};
}

/**
 * @brief Helper function to compute the lengths of the arrays in the supplied
 * array_info and it's children and store them in the rank_to_lens vector. Only
 * intended to be called by send_metadata.
 * @param in_arr The array_info for which to compute the lengths
 * @param rank_to_lens length vector for each rank
 * @param rank The rank for which to compute the lengths
 * @param comm_info The communication info for the array
 * @param sub_comm_info_iter Iterator to the sub communication info for the
 * array
 * @param str_comm_info_iter Iterator to the string communication info for the
 * array
 */
void compute_lengths_helper(const std::shared_ptr<array_info>& in_arr,
                            std::vector<std::vector<uint64_t>>& rank_to_lens,
                            int rank, const mpi_comm_info& comm_info,
                            comm_info_iter_t& sub_comm_info_iter,
                            str_comm_info_iter_t& str_comm_info_iter) {
    rank_to_lens[rank].push_back(comm_info.send_count[rank]);
    if (in_arr->arr_type == bodo_array_type::STRING) {
        rank_to_lens[rank].push_back(str_comm_info_iter->send_count_sub[rank]);
    }
    // All arrays have a str_comm_info
    str_comm_info_iter++;

    if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
        const mpi_comm_info& sub_comm_info = *sub_comm_info_iter++;
        compute_lengths_helper(in_arr->child_arrays[0], rank_to_lens, rank,
                               sub_comm_info, sub_comm_info_iter,
                               str_comm_info_iter);
    } else if (in_arr->arr_type == bodo_array_type::STRUCT) {
        for (const auto& child : in_arr->child_arrays) {
            compute_lengths_helper(child, rank_to_lens, rank, comm_info,
                                   sub_comm_info_iter, str_comm_info_iter);
        }
    } else if (in_arr->arr_type == bodo_array_type::MAP) {
        compute_lengths_helper(in_arr->child_arrays[0], rank_to_lens, rank,
                               comm_info, sub_comm_info_iter,
                               str_comm_info_iter);
    } else if (in_arr->arr_type == bodo_array_type::DICT) {
        // Add total dict num elements and num characters
        rank_to_lens[rank].push_back(in_arr->child_arrays[0]->length);
        rank_to_lens[rank].push_back(in_arr->child_arrays[0]->n_sub_elems());
        // Update iterator since we are not calling the function recursively for
        // offset array here
        str_comm_info_iter++;
    }
}

void AsyncShuffleSendState::send_metadata(
    const std::shared_ptr<table_info>& in_table, mpi_comm_info& comm_info,
    comm_infos_t& sub_comm_infos, MPI_Comm shuffle_comm) {
    // Compute buffer lengths for each rank.
    std::vector<std::vector<uint64_t>> rank_to_lens(comm_info.n_pes);

    for (int rank = 0; rank < comm_info.n_pes; rank++) {
        comm_info_iter_t comm_info_iter = std::get<0>(sub_comm_infos).cbegin();
        str_comm_info_iter_t str_comm_info_iter =
            std::get<1>(sub_comm_infos).cbegin();
        for (const auto& in_arr : in_table->columns) {
            compute_lengths_helper(in_arr, rank_to_lens, rank, comm_info,
                                   comm_info_iter, str_comm_info_iter);
        }
    }

    // Construct and send the metadata message.
    this->rank_to_metadata_vec =
        std::vector<std::vector<uint64_t>>(comm_info.n_pes);
    for (int rank = 0; rank < comm_info.n_pes; rank++) {
        // Don't send messages to ranks with no data
        if (rank_to_lens[rank][0] == 0) {
            continue;
        }

        // The first element is the starting message tag. It will be followed by
        // the message lengths.
        this->rank_to_metadata_vec[rank] = {
            static_cast<uint64_t>(this->starting_msg_tag)};
        this->rank_to_metadata_vec[rank].insert(
            this->rank_to_metadata_vec[rank].end(), rank_to_lens[rank].begin(),
            rank_to_lens[rank].end());

        MPI_Request req;
        CHECK_MPI(
            MPI_Issend(this->rank_to_metadata_vec[rank].data(),
                       this->rank_to_metadata_vec[rank].size(), MPI_UINT64_T,
                       rank, SHUFFLE_METADATA_MSG_TAG, shuffle_comm, &req),
            "AsyncShuffleSendState::send_metadata: MPI error on MPI_Issend:");
        this->send_requests.push_back(req);
    }
}
void AsyncShuffleSendState::send_arrays(
    const std::shared_ptr<table_info>& in_table, mpi_comm_info& comm_info,
    AsyncShuffleSendState::comm_infos_t& comm_infos, MPI_Comm shuffle_comm) {
    std::vector<int> curr_tags(comm_info.n_pes, this->starting_msg_tag);
    comm_info_iter_t sub_comm_info_iter = std::get<0>(comm_infos).begin();
    str_comm_info_iter_t str_comm_info_iter = std::get<1>(comm_infos).begin();
    for (uint64_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        std::vector<bool> must_shuffle_to_rank(comm_info.n_pes, false);
        this->send_shuffle_data_unknown_type(
            shuffle_comm, comm_info, sub_comm_info_iter, str_comm_info_iter,
            in_arr, curr_tags, must_shuffle_to_rank);
    }
}

std::optional<std::vector<uint64_t>> AsyncShuffleRecvState::GetRecvMetadata(
    MPI_Comm shuffle_comm) {
    int flag;
    assert(this->metadata_request != MPI_REQUEST_NULL);
    CHECK_MPI(MPI_Test(&this->metadata_request, &flag, MPI_STATUS_IGNORE),
              "AsyncShuffleRecvState::GetRecvMetadata: MPI error on MPI_Test:");
    if (!flag) {
        return std::nullopt;
    }
    this->metadata_request = MPI_REQUEST_NULL;

    std::vector<uint64_t> ret_val = std::move(this->metadata_vec);
    this->metadata_vec.clear();
    return ret_val;
}

void AsyncShuffleRecvState::PostMetadataRecv(MPI_Status& status,
                                             MPI_Message& m) {
    assert(this->metadata_request == MPI_REQUEST_NULL);
    int md_size;
    CHECK_MPI(
        MPI_Get_count(&status, MPI_UINT64_T, &md_size),
        "AsyncShuffleRecvState::PostMetadataRecv: MPI error on MPI_Get_count:");
    this->metadata_vec.resize(md_size);

    CHECK_MPI(
        MPI_Imrecv(this->metadata_vec.data(), this->metadata_vec.size(),
                   MPI_UINT64_T, &m, &this->metadata_request),
        "AsyncShuffleRecvState::PostMetadataRecv: MPI error on MPI_Imrecv:");
}

void consume_completed_recvs(
    std::vector<AsyncShuffleRecvState>& recv_states, MPI_Comm shuffle_comm,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    IncrementalShuffleMetrics& metrics, TableBuildBuffer& out_builder) {
    std::erase_if(recv_states, [&](AsyncShuffleRecvState& s) {
        auto [done, table] = s.recvDone(dict_builders, shuffle_comm, metrics);
        if (done) {
            out_builder.ReserveTable(table);
            out_builder.UnsafeAppendBatch(table);
        }
        return done;
    });
}
