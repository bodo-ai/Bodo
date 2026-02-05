#include "operator.h"
#include <arrow/array/builder_base.h>

#ifdef USE_CUDF
#include "../../libs/gpu_utils.h"
#endif

int64_t PhysicalOperator::next_op_id = 1;

int get_streaming_batch_size() {
    char *env_str = std::getenv("BODO_STREAMING_BATCH_SIZE");
    return (env_str != nullptr) ? std::stoi(env_str) : 32768;
}

int get_gpu_streaming_batch_size() {
    char *env_str = std::getenv("BODO_GPU_STREAMING_BATCH_SIZE");
    return (env_str != nullptr) ? std::stoi(env_str) : 32768 * 10;
}

// Maximum Parquet file size for streaming Parquet write
int64_t get_parquet_chunk_size() {
    char *env_str = std::getenv("BODO_PARQUET_WRITE_CHUNK_SIZE");
    return (env_str != nullptr) ? std::stoll(env_str)
                                : 256e6;  // Default to 256 MiB
}

extern const bool G_USE_ASYNC = false;

#ifdef USE_CUDF

OperatorResult PhysicalSink::ConsumeBatch(GPU_DATA input_batch,
                                          OperatorResult prev_op_result) {
    auto cpu_batch = convertGPUToTable(input_batch);
    auto [cpu_batch_fragment, exchange_result] =
        gpu_to_cpu_exchange(cpu_batch, prev_op_result);
    return ConsumeBatch(cpu_batch_fragment, exchange_result);
}

std::pair<std::shared_ptr<table_info>, OperatorResult>
PhysicalProcessBatch::ProcessBatch(GPU_DATA input_batch,
                                   OperatorResult prev_op_result) {
    auto cpu_batch = convertGPUToTable(input_batch);
    auto [cpu_batch_fragment, exchange_result] =
        gpu_to_cpu_exchange(cpu_batch, prev_op_result);
    return ProcessBatch(cpu_batch_fragment, exchange_result);
}

OperatorResult PhysicalGPUSink::ConsumeBatch(GPU_DATA input_batch,
                                             OperatorResult prev_op_result) {
    std::shared_ptr<StreamAndEvent> se = make_stream_and_event(G_USE_ASYNC);
    // Wait until previous GPU pipeline processing is done.
    input_batch.stream_event->event.wait(se->stream);
    return ConsumeBatchGPU(input_batch, prev_op_result, se);
}

OperatorResult PhysicalGPUSink::ConsumeBatch(
    std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result) {
    auto [cpu_batch_fragment, exchange_result] =
        cpu_to_gpu_exchange(input_batch, prev_op_result);
    std::shared_ptr<StreamAndEvent> se = make_stream_and_event(G_USE_ASYNC);
    auto gpu_batch = convertTableToGPU(cpu_batch_fragment);
    return ConsumeBatchGPU(gpu_batch, exchange_result, se);
}

std::pair<GPU_DATA, OperatorResult> PhysicalGPUProcessBatch::ProcessBatch(
    GPU_DATA input_batch, OperatorResult prev_op_result) {
    std::shared_ptr<StreamAndEvent> se = make_stream_and_event(G_USE_ASYNC);
    // Wait until previous GPU pipeline processing is done.
    input_batch.stream_event->event.wait(se->stream);
    return ProcessBatchGPU(input_batch, prev_op_result, se);
}

std::pair<GPU_DATA, OperatorResult> PhysicalGPUProcessBatch::ProcessBatch(
    std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result) {
    std::shared_ptr<StreamAndEvent> se = make_stream_and_event(G_USE_ASYNC);
    auto [cpu_batch_fragment, exchange_result] =
        cpu_to_gpu_exchange(input_batch, prev_op_result);
    auto gpu_batch = convertTableToGPU(cpu_batch_fragment);
    return ProcessBatchGPU(gpu_batch, exchange_result, se);
}

std::shared_ptr<table_info> convertGPUToTable(GPU_DATA batch) {
    std::shared_ptr<arrow::Table> table = convertGPUToArrow(batch);

    return arrow_table_to_bodo(table, nullptr);
}

GPU_DATA convertTableToGPU(std::shared_ptr<table_info> batch) {
    std::shared_ptr<arrow::Table> arrow_table = bodo_table_to_arrow(batch);

    // Arrow tables can have fragmented columns (chunks). libcudf expects
    // contiguous memory. This merges all chunks into a single RecordBatch.
    std::shared_ptr<arrow::RecordBatch> arrow_batch;
    auto combined_table_result = arrow_table->CombineChunks();
    if (!combined_table_result.ok()) {
        throw std::runtime_error("Failed to combine Arrow chunks");
    }
    std::shared_ptr<arrow::Table> combined_table =
        combined_table_result.ValueOrDie();

    // Read the single batch
    arrow::TableBatchReader reader(*combined_table);
    arrow::Result<std::shared_ptr<arrow::RecordBatch>> batch_result =
        reader.Next();
    if (!batch_result.ok()) {
        throw std::runtime_error("Failed to extract batch from Arrow table");
    }
    arrow_batch = batch_result.ValueOrDie();

    // End of stream (Create RecordBatch of empty arrays for cudf)
    if (!arrow_batch) {
        std::vector<std::shared_ptr<arrow::Array>> empty_arrays;
        empty_arrays.reserve(combined_table->num_columns());
        for (auto &field : combined_table->schema()->fields()) {
            std::unique_ptr<arrow::ArrayBuilder> builder;
            if (!arrow::MakeBuilder(arrow::default_memory_pool(), field->type(),
                                    &builder)
                     .ok()) {
                throw std::runtime_error("Failed to create Arrow ArrayBuilder");
            }
            std::shared_ptr<arrow::Array> arr;
            if (!builder->Finish(&arr).ok()) {
                throw std::runtime_error("Failed to finish Arrow Array");
            }
            empty_arrays.push_back(arr);
        }
        arrow_batch =
            arrow::RecordBatch::Make(combined_table->schema(), 0, empty_arrays);
    }

    // Export to C Data Interface Structs
    // These are lightweight structs that hold pointers to the CPU data.
    struct ArrowSchema arrow_schema;
    struct ArrowArray arrow_array;

    // ExportRecordBatch fills these structs.
    arrow::Status status =
        arrow::ExportRecordBatch(*arrow_batch, &arrow_array, &arrow_schema);
    if (!status.ok()) {
        throw std::runtime_error("Failed to export to Arrow C interface");
    }

    // Wrap ArrowArray in ArrowDeviceArray for cudf
    struct ArrowDeviceArray device_array;

    // Copy the array content into the device wrapper
    // (ArrowDeviceArray contains an 'array' member which is the ArrowArray
    // struct)
    device_array.array = arrow_array;

    // Explicitly mark this as CPU data
    device_array.device_id = -1;
    device_array.device_type = ARROW_DEVICE_CPU;
    device_array.sync_event = nullptr;  // No CUDA event needed for CPU data

    // Move Data to GPU
    // from_arrow_host parses the structs, allocates GPU memory, and performs
    // the copy.
    std::unique_ptr<cudf::table> result =
        cudf::from_arrow_host(&arrow_schema, &device_array);

    // Clean up the C structs (Arrow requires manual release if not imported,
    // but Export gives us ownership, so we must release the release callbacks)
    if (device_array.array.release) {
        device_array.array.release(&device_array.array);
    }
    if (arrow_schema.release) {
        arrow_schema.release(&arrow_schema);
    }

    // Return the cudf::table (moving ownership)
    return GPU_DATA{std::move(result), arrow_batch->schema(),
                    make_stream_and_event(false)};
}

std::shared_ptr<arrow::Table> convertGPUToArrow(GPU_DATA batch) {
    cudf::table_view view = batch.table->view();
    // Setup Metadata (Arrow requires column names)
    // We must create a cudf::column_metadata hierarchy matching the table
    // structure.
    std::vector<cudf::column_metadata> meta;
    for (const auto &name : batch.schema->field_names()) {
        meta.emplace_back(name);
    }

    cudf::unique_schema_t unique_schema = cudf::to_arrow_schema(view, meta);

    // Move Data to Host (GPU -> CPU Copy happens here)
    // 'to_arrow_host' performs the D2H copy and populates an ArrowDeviceArray.
    cudf::unique_device_array_t unique_device_array = cudf::to_arrow_host(view);

    // We release the raw pointers from the unique_ptrs because Arrow's Import
    // function takes ownership of the C-structs and manages their lifecycle
    // internally.
    struct ArrowSchema *raw_schema = unique_schema.release();
    struct ArrowDeviceArray *raw_array = unique_device_array.release();

    // ImportDeviceRecordBatch is the zero-copy bridge from the C-structs to C++
    // objects. (The data is already on CPU from step 3, so this import is
    // cheap).
    arrow::Result<std::shared_ptr<arrow::RecordBatch>> maybe_batch =
        arrow::ImportDeviceRecordBatch(raw_array, raw_schema);

    if (!maybe_batch.ok()) {
        throw std::runtime_error("Failed to import Arrow RecordBatch: " +
                                 maybe_batch.status().ToString());
    }

    std::shared_ptr<arrow::RecordBatch> arrow_batch = maybe_batch.ValueOrDie();

    // Wrap in an Arrow Table
    arrow::Result<std::shared_ptr<arrow::Table>> maybe_table =
        arrow::Table::FromRecordBatches({arrow_batch});
    if (!maybe_table.ok()) {
        // Handle error (e.g., throw or log)
        throw std::runtime_error("Failed to create Arrow Table: " +
                                 maybe_table.status().ToString());
    }

    std::shared_ptr<arrow::Table> table = maybe_table.ValueOrDie();
    table = table->ReplaceSchemaMetadata(batch.schema->metadata());
    return table;
}
#else
OperatorResult PhysicalSink::ConsumeBatch(GPU_DATA input_batch,
                                          OperatorResult prev_op_result) {
    throw std::runtime_error("Should never be called in non-CUDF mode.");
}

std::pair<std::shared_ptr<table_info>, OperatorResult>
PhysicalProcessBatch::ProcessBatch(GPU_DATA input_batch,
                                   OperatorResult prev_op_result) {
    throw std::runtime_error("Should never be called in non-CUDF mode.");
}

OperatorResult PhysicalGPUSink::ConsumeBatch(
    std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result) {
    throw std::runtime_error("Should never be called in non-CUDF mode.");
}

OperatorResult PhysicalGPUSink::ConsumeBatch(GPU_DATA input_batch,
                                             OperatorResult prev_op_result) {
    throw std::runtime_error("Should never be called in non-CUDF mode.");
}

std::pair<GPU_DATA, OperatorResult> PhysicalGPUProcessBatch::ProcessBatch(
    GPU_DATA input_batch, OperatorResult prev_op_result) {
    throw std::runtime_error("Should never be called in non-CUDF mode.");
}

std::pair<GPU_DATA, OperatorResult> PhysicalGPUProcessBatch::ProcessBatch(
    std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result) {
    throw std::runtime_error("Should never be called in non-CUDF mode.");
}
#endif

/**
 * @brief State for sending contiguous chunks for data from source ranks to
 * destination ranks.
 */
class SrcDestIncrementalShuffleState : public IncrementalShuffleState {
   public:
    using dict_hashes_t = bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>;

    SrcDestIncrementalShuffleState(
        std::shared_ptr<bodo::Schema> schema_,
        const std::vector<std::shared_ptr<DictionaryBuilder>> &dict_builders_,
        const std::vector<int> &src_ranks_, const std::vector<int> &dest_ranks_,
        const uint64_t &curr_iter_, int64_t &sync_freq_, int64_t parent_op_id_)
        : IncrementalShuffleState(std::move(schema_), dict_builders_,
                                  static_cast<uint64_t>(0), curr_iter_,
                                  sync_freq_, parent_op_id_),
          src_ranks(src_ranks_),
          dest_ranks(dest_ranks_) {}

    /**
     * @brief Get the table to be shuffle and list of destination ranks to send
     to.
     *
     * @return std::tuple<std::shared_ptr<table_info>,
               std::shared_ptr<dict_hashes_t>,
               std::shared_ptr<uint32_t[]>,
               std::unique_ptr<uint8_t[]>>
               The table to be shuffled, empty dict hashes (not used here),
     destination ranks per row, and nullptr keep_row_bitmask (not used here).
     */
    std::tuple<std::shared_ptr<table_info>, std::shared_ptr<dict_hashes_t>,
               std::shared_ptr<uint32_t[]>, std::unique_ptr<uint8_t[]>>
    GetShuffleTableAndHashes() override {
        std::shared_ptr<table_info> shuffle_table =
            this->table_buffer->data_table;
        auto dict_hashes = std::make_shared<dict_hashes_t>();

        int rank;
        MPI_Comm_rank(this->shuffle_comm, &rank);
        std::shared_ptr<uint32_t[]> shuffle_hashes = nullptr;
        auto rank_it = std::ranges::find(src_ranks, rank);
        if (rank_it == src_ranks.end()) {
            throw std::runtime_error(
                "SrcDestIncrementalShuffleState::GetShuffleTableAndHashes: "
                "Current rank not in source ranks list");
        }
        int rank_idx = std::distance(src_ranks.begin(), rank_it);

        // Case 1: send many sources to fewer (or equal) destinations
        // Assign each source rank a single destination rank to send to.
        if (src_ranks.size() >= dest_ranks.size()) {
            uint32_t dest_rank =
                dest_ranks[(rank_idx * dest_ranks.size()) / src_ranks.size()];
            shuffle_hashes =
                std::make_shared<uint32_t[]>(shuffle_table->nrows(), dest_rank);
        }
        // Case 2: send fewer sources to many destinations
        // Distribute destination ranks evenly among source ranks.
        // Send contiguous blocks of rows to each destination rank.
        else {
            shuffle_hashes =
                std::make_shared<uint32_t[]>(shuffle_table->nrows());
            int dests_per_rank = dest_ranks.size() / src_ranks.size();
            int rem = dest_ranks.size() % src_ranks.size();
            int start_idx = rank_idx * dests_per_rank + std::min(rank_idx, rem);
            int local_n_dest = dests_per_rank + (rank_idx < rem ? 1 : 0);

            for (size_t i = 0; i < shuffle_table->nrows(); i++) {
                uint32_t dest_rank =
                    dest_ranks[start_idx +
                               ((i * local_n_dest) / shuffle_table->nrows())];
                shuffle_hashes[i] = dest_rank;
            }
        }

        return std::make_tuple(shuffle_table, dict_hashes, shuffle_hashes,
                               std::unique_ptr<uint8_t[]>(nullptr));
    }

   private:
    const std::vector<int> src_ranks;
    const std::vector<int> dest_ranks;
};

RankDataExchange::RankDataExchange(int64_t op_id_)
    : op_id(op_id_), is_last_state(std::make_shared<IsLastState>()) {
    // Create a communicator for all ranks on the node
    CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &this->shuffle_comm),
              "RankDataExchange::RankDataExchange:: MPI error on "
              "MPI_Comm_split_type:");

    // Get a list of all GPU ranks
    int n_pes;
    MPI_Comm_size(this->shuffle_comm, &n_pes);
    bool has_gpu = get_gpu_id().value() >= 0;
    int local = has_gpu ? 1 : 0;
    std::vector<int> has_gpu_all(n_pes);
    CHECK_MPI(
        MPI_Allgather(&local, 1, MPI_INT, has_gpu_all.data(), 1, MPI_INT,
                      this->shuffle_comm),
        "RankDataExchange::RankDataExchange: MPI error on MPI_Allgather:");
    for (size_t i = 0; i < has_gpu_all.size(); i++) {
        if (has_gpu_all[i] == 1) {
            this->gpu_ranks.push_back(i);
        }
        this->cpu_ranks.push_back(i);
    }
}

std::tuple<std::shared_ptr<table_info>, OperatorResult>
RankDataExchange::operator()(std::shared_ptr<table_info> input_batch,
                             OperatorResult prev_op_result) {
    if (!this->shuffle_state) {
        Initialize(input_batch.get());
    }

    // Shuffle data to destination ranks (either all ranks or GPU ranks)
    // and append result to output builder
    std::vector<bool> append_rows(input_batch->nrows(), true);
    this->shuffle_state->AppendBatch(input_batch, append_rows);
    auto result = this->shuffle_state->ShuffleIfRequired(true);
    // TODO: Start data transfer for CPU->GPU as soon as it's availible
    if (result.has_value()) {
        std::shared_ptr<table_info> shuffled_table = result.value();
        collected_rows->builder->UnifyDictionariesAndAppend(shuffled_table);
    }

    // Determine whether we need more input, have more output, or are finished
    // with the exchange.
    bool request_input =
        !(this->shuffle_state->BuffersFull() &&
          (collected_rows->builder->total_remaining >
           (2 * collected_rows->builder->active_chunk_capacity)));
    bool local_is_last = prev_op_result == OperatorResult::FINISHED &&
                         (this->shuffle_state->SendRecvEmpty());
    bool global_is_last = static_cast<bool>(sync_is_last_non_blocking(
        is_last_state.get(), static_cast<int32_t>(local_is_last)));
    auto [output_batch, _] = collected_rows->builder->PopChunk(
        /*force_return*/ global_is_last);

    bool finished =
        global_is_last && this->collected_rows->builder->total_remaining == 0;
    return std::make_tuple(
        output_batch, finished
                          ? OperatorResult::FINISHED
                          : (request_input ? OperatorResult::NEED_MORE_INPUT
                                           : OperatorResult::HAVE_MORE_OUTPUT));
}

void RankDataExchange::Initialize(table_info *input_batch) {
    std::unique_ptr<bodo::Schema> table_schema = input_batch->schema();
    collected_rows = std::make_unique<ChunkedTableBuilderState>(
        input_batch->schema(), GetOutBatchSize());

    InitializeShuffleState(input_batch, collected_rows->dict_builders);
}

RankDataExchange::~RankDataExchange() { MPI_Comm_free(&this->shuffle_comm); }

int64_t CPUtoGPUExchange::GetOutBatchSize() {
    return get_gpu_streaming_batch_size();
}

void CPUtoGPUExchange::InitializeShuffleState(
    table_info *input_batch,
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders) {
    uint64_t curr_iter = 0;
    int64_t sync_freq = 1;

    this->shuffle_state = std::make_unique<SrcDestIncrementalShuffleState>(
        input_batch->schema(), dict_builders, cpu_ranks, gpu_ranks, curr_iter,
        sync_freq, this->op_id);
    this->shuffle_state->Initialize(nullptr, true, this->shuffle_comm);
}

int64_t GPUtoCPUExchange::GetOutBatchSize() {
    return get_streaming_batch_size();
}

void GPUtoCPUExchange::InitializeShuffleState(
    table_info *input_batch,
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders) {
    uint64_t curr_iter = 0;
    int64_t sync_freq = 1;

    this->shuffle_state = std::make_unique<SrcDestIncrementalShuffleState>(
        input_batch->schema(), dict_builders, gpu_ranks, cpu_ranks, curr_iter,
        sync_freq, this->op_id);
    this->shuffle_state->Initialize(nullptr, true, this->shuffle_comm);
}
