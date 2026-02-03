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

// Maximum Parquet file size for streaming Parquet write
int64_t get_parquet_chunk_size() {
    char *env_str = std::getenv("BODO_PARQUET_WRITE_CHUNK_SIZE");
    return (env_str != nullptr) ? std::stoll(env_str)
                                : 256e6;  // Default to 256 MiB
}

#ifdef USE_CUDF
OperatorResult PhysicalSink::ConsumeBatch(GPU_DATA input_batch,
                                          OperatorResult prev_op_result) {
    auto cpu_batch = convertGPUToTable(input_batch);
    return ConsumeBatch(cpu_batch, prev_op_result);
}

std::pair<std::shared_ptr<table_info>, OperatorResult>
PhysicalProcessBatch::ProcessBatch(GPU_DATA input_batch,
                                   OperatorResult prev_op_result) {
    auto cpu_batch = convertGPUToTable(input_batch);
    return ProcessBatch(cpu_batch, prev_op_result);
}

OperatorResult PhysicalGPUSink::ConsumeBatch(
    std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result) {
    auto gpu_batch = convertTableToGPU(input_batch);
    return ConsumeBatch(gpu_batch, prev_op_result);
}

std::pair<GPU_DATA, OperatorResult> PhysicalGPUProcessBatch::ProcessBatch(
    std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result) {
    auto gpu_batch = convertTableToGPU(input_batch);
    return ProcessBatch(gpu_batch, prev_op_result);
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
    return GPU_DATA{std::move(result), arrow_batch->schema()};
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

std::pair<GPU_DATA, OperatorResult> PhysicalGPUProcessBatch::ProcessBatch(
    std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result) {
    throw std::runtime_error("Should never be called in non-CUDF mode.");
}
#endif

CPUtoGPUExchange::CPUtoGPUExchange(int64_t op_id_)
    : op_id(op_id_), is_last_state(std::make_shared<IsLastState>()) {
    // TODO: Get GPU Ranks to send to.
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    has_gpu = myrank == 0;
    gpu_ranks = {0};
    MPI_Comm_dup(MPI_COMM_WORLD, &this->shuffle_comm);
}

void CPUtoGPUExchange::Initialize(table_info *input_batch) {
    std::unique_ptr<bodo::Schema> table_schema = input_batch->schema();
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    for (const std::unique_ptr<bodo::DataType> &t :
         table_schema->column_types) {
        dict_builders.emplace_back(
            create_dict_builder_for_array(t->copy(), false));
    }
    this->collected_rows = std::make_unique<ChunkedTableBuilder>(
        input_batch->schema(), dict_builders, get_streaming_batch_size());

    uint64_t n_keys = 1;
    uint64_t curr_iter = 0;
    int64_t sync_freq = 1;

    // TODO: Free ShuffleState earlier ?
    this->shuffle_state = std::make_unique<IncrementalShuffleState>(
        input_batch->schema(), dict_builders, n_keys, curr_iter, sync_freq,
        this->op_id, gpu_ranks);
    this->shuffle_state->Initialize(nullptr, true, this->shuffle_comm);
}

std::tuple<std::shared_ptr<table_info>, bool>
CPUtoGPUExchange::CPURanksToGPURanks(std::shared_ptr<table_info> input_batch,
                                     OperatorResult prev_op_result) {
    if (!this->shuffle_state) {
        Initialize(input_batch.get());
    }
    bool local_is_last = prev_op_result == OperatorResult::FINISHED;

    std::vector<bool> append_rows(input_batch->nrows(), true);
    this->shuffle_state->AppendBatch(input_batch, append_rows);

    auto result = this->shuffle_state->ShuffleIfRequired(true);

    if (result.has_value()) {
        std::shared_ptr<table_info> shuffled_table = result.value();
        collected_rows->UnifyDictionariesAndAppend(shuffled_table);
    }

    local_is_last = local_is_last && (this->shuffle_state->SendRecvEmpty());

    bool global_is_last = static_cast<bool>(sync_is_last_non_blocking(
        is_last_state.get(), static_cast<int32_t>(local_is_last)));

    auto [output_batch, _] = collected_rows->PopChunk(
        /*force_return*/ global_is_last);

    bool finished =
        global_is_last && this->collected_rows->total_remaining == 0;

    return std::make_tuple(output_batch, finished);
}

CPUtoGPUExchange::~CPUtoGPUExchange() { MPI_Comm_free(&this->shuffle_comm); }
