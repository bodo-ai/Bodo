#include "operator.h"
#include <arrow/array/builder_base.h>
#include <arrow/util/endian.h>
#include <memory>
#include <string>

#ifdef USE_CUDF
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/cuda_stream_view.hpp>

#include "../../libs/gpu_utils.h"
#include "../libs/_table_builder_utils.h"
#endif

int64_t PhysicalOperator::next_op_id = 1;

int get_streaming_batch_size() {
    char *env_str = std::getenv("BODO_STREAMING_BATCH_SIZE");
    return (env_str != nullptr) ? std::stoi(env_str) : 32768;
}

int get_gpu_streaming_batch_size() {
    char *env_str = std::getenv("BODO_GPU_STREAMING_BATCH_SIZE");
    // TODO: tune batch size
    return (env_str != nullptr) ? std::stoi(env_str) : 32768 * 10;
}

// Maximum Parquet file size for streaming Parquet write
int64_t get_parquet_chunk_size() {
    char *env_str = std::getenv("BODO_PARQUET_WRITE_CHUNK_SIZE");
    return (env_str != nullptr) ? std::stoll(env_str)
                                : 256e6;  // Default to 256 MiB
}

#ifdef USE_CUDF

std::shared_ptr<cudf::table> make_empty_like(
    std::shared_ptr<cudf::table> input_table,
    std::shared_ptr<StreamAndEvent> se) {
    cudf::table_view tv = input_table->view();

    // slice produces a vector<table_view>
    auto sliced = cudf::slice(tv, {0, 0}, se->stream);
    cudf::table_view empty_view = sliced[0];

    // materialize into a real cudf::table
    return std::make_shared<cudf::table>(empty_view);
}

void GPUBatchGenerator::append_batch(GPU_DATA batch) {
    auto n = batch.table->num_rows();
    if (n == 0) {
        return;
    }
    collected_rows += n;
    batches.push_back(batch);
}

GPU_DATA GPUBatchGenerator::next(std::shared_ptr<StreamAndEvent> se,
                                 bool force_return) {
    if (collected_rows < out_batch_size && !force_return) {
        dummy_gpu_data->stream_event->event.wait(se->stream);
        return GPU_DATA(make_empty_like(dummy_gpu_data->table, se),
                        dummy_gpu_data->schema, se);
    }

    std::vector<GPU_DATA> gpu_tables;
    std::size_t rows_accum = 0;

    if (leftover_data) {
        std::size_t n = leftover_data->table->num_rows();
        if (n <= out_batch_size) {
            rows_accum += n;
            gpu_tables.push_back(*leftover_data);
            leftover_data.reset();
        } else {
            cudf::table_view tv = leftover_data->table->view();
            leftover_data->stream_event->event.wait(se->stream);
            auto batch =
                cudf::slice(tv, {0, (int)out_batch_size}, se->stream)[0];
            auto remain =
                cudf::slice(tv, {(int)out_batch_size, (int)n}, se->stream)[0];
            gpu_tables.emplace_back(std::make_shared<cudf::table>(batch),
                                    leftover_data->schema, se);
            leftover_data = std::make_unique<GPU_DATA>(
                std::make_shared<cudf::table>(remain), leftover_data->schema,
                se);
            rows_accum = out_batch_size;
        }
    }

    while (!batches.empty() && (rows_accum < out_batch_size)) {
        GPU_DATA &batch = batches.front();
        cudf::table_view tv = batch.table->view();
        std::shared_ptr<StreamAndEvent> &batch_se = batch.stream_event;
        std::size_t n = tv.num_rows();

        if (rows_accum + n <= out_batch_size) {
            rows_accum += n;
            gpu_tables.push_back(batch);
        } else {
            auto batch_part =
                cudf::slice(tv, {0, (int)(out_batch_size - rows_accum)},
                            batch_se->stream)[0];
            auto remain_part =
                cudf::slice(tv, {(int)(out_batch_size - rows_accum), (int)n},
                            batch_se->stream)[0];
            gpu_tables.emplace_back(std::make_shared<cudf::table>(batch_part),
                                    batch.schema, batch_se);
            leftover_data = std::make_unique<GPU_DATA>(
                std::make_shared<cudf::table>(remain_part), batch.schema,
                batch_se);
            rows_accum = out_batch_size;
        }
        batches.pop_front();
    }
    collected_rows -= rows_accum;

    if (rows_accum == 0) {
        dummy_gpu_data->stream_event->event.wait(se->stream);
        return GPU_DATA(make_empty_like(dummy_gpu_data->table, se),
                        dummy_gpu_data->schema, se);
    }

    if (gpu_tables.size() == 1) {
        gpu_tables[0].stream_event->event.wait(se->stream);
        return GPU_DATA(gpu_tables[0].table, dummy_gpu_data->schema, se);
    }

    std::vector<cudf::table_view> table_views;
    table_views.reserve(gpu_tables.size());
    for (auto &tptr : gpu_tables) {
        tptr.stream_event->event.wait(se->stream);
        table_views.emplace_back(tptr.table->view());
    }
    std::unique_ptr<cudf::table> batch =
        cudf::concatenate(table_views, se->stream);
    return GPU_DATA(std::move(batch), dummy_gpu_data->schema, se);
}

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

std::pair<GPU_DATA, OperatorResult> PhysicalGPUSource::ProduceBatch() {
    std::shared_ptr<StreamAndEvent> se = make_stream_and_event(G_USE_ASYNC);
    auto gpu_result = ProduceBatchGPU(se);
    se->event.record(se->stream);
    return gpu_result;
}

OperatorResult PhysicalGPUSink::ConsumeBatch(GPU_DATA input_batch,
                                             OperatorResult prev_op_result) {
    std::shared_ptr<StreamAndEvent> se = make_stream_and_event(G_USE_ASYNC);
    // Wait until previous GPU pipeline processing is done.
    input_batch.stream_event->event.wait(se->stream);
    auto gpu_result = ConsumeBatchGPU(input_batch, prev_op_result, se);
    se->event.record(se->stream);
    return gpu_result;
}

OperatorResult PhysicalGPUSink::ConsumeBatch(
    std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result) {
    std::shared_ptr<StreamAndEvent> se = make_stream_and_event(G_USE_ASYNC);
    auto [gpu_batch, exchange_result] =
        cpu_to_gpu_exchange(input_batch, se, prev_op_result);

    auto gpu_result = ConsumeBatchGPU(gpu_batch, exchange_result, se);
    se->event.record(se->stream);
    return gpu_result;
}

std::pair<GPU_DATA, OperatorResult> PhysicalGPUProcessBatch::ProcessBatch(
    GPU_DATA input_batch, OperatorResult prev_op_result) {
    std::shared_ptr<StreamAndEvent> se = make_stream_and_event(G_USE_ASYNC);
    // Wait until previous GPU pipeline processing is done.
    input_batch.stream_event->event.wait(se->stream);
    auto gpu_result = ProcessBatchGPU(input_batch, prev_op_result, se);
    se->event.record(se->stream);
    return gpu_result;
}

std::pair<GPU_DATA, OperatorResult> PhysicalGPUProcessBatch::ProcessBatch(
    std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result) {
    std::cout
        << "Calling PhysicalGPUProcessBatch::ProcessBatch with CPU batch of "
        << input_batch->nrows() << " rows" << std::endl;
    std::shared_ptr<StreamAndEvent> se = make_stream_and_event(G_USE_ASYNC);
    auto [gpu_batch, exchange_result] =
        cpu_to_gpu_exchange(input_batch, se, prev_op_result);
    std::cout << "exchange_result: " << toString(exchange_result) << std::endl;

    auto gpu_result = ProcessBatchGPU(gpu_batch, exchange_result, se);
    std::cout << "Process GPU result:" << toString(gpu_result.second)
              << std::endl;
    se->event.record(se->stream);
    return gpu_result;
}

std::shared_ptr<table_info> convertGPUToTable(GPU_DATA batch) {
    std::shared_ptr<arrow::Table> table = convertGPUToArrow(batch);

    return arrow_table_to_bodo(table, nullptr);
}

GPU_DATA convertTableToGPU(std::shared_ptr<table_info> batch,
                           std::shared_ptr<StreamAndEvent> se) {
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
        cudf::from_arrow_host(&arrow_schema, &device_array, se->stream);

    // Clean up the C structs (Arrow requires manual release if not imported,
    // but Export gives us ownership, so we must release the release callbacks)
    if (device_array.array.release) {
        device_array.array.release(&device_array.array);
    }
    if (arrow_schema.release) {
        arrow_schema.release(&arrow_schema);
    }

    // Return the cudf::table (moving ownership)
    return GPU_DATA{std::move(result), arrow_batch->schema(), se};
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
        auto rank_it = std::ranges::find(src_ranks, rank);
        if (rank_it == src_ranks.end()) {
            throw std::runtime_error(
                "SrcDestIncrementalShuffleState::GetShuffleTableAndHashes: "
                "Current rank not in source ranks list");
        }
        int rank_idx = std::distance(src_ranks.begin(), rank_it);

        std::shared_ptr<uint32_t[]> shuffle_hashes =
            std::make_shared<uint32_t[]>(shuffle_table->nrows());
        // Case 1: send many sources to fewer (or equal) destinations
        // Assign each source rank a single destination rank to send to.
        if (src_ranks.size() >= dest_ranks.size()) {
            uint32_t dest_rank =
                dest_ranks[(rank_idx * dest_ranks.size()) / src_ranks.size()];
            std::fill(shuffle_hashes.get(),
                      shuffle_hashes.get() + shuffle_table->nrows(), dest_rank);
        }
        // Case 2: send fewer sources to many destinations
        // Distribute destination ranks evenly among source ranks.
        // Send contiguous blocks of rows to each destination rank.
        else {
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

RankDataExchange::RankDataExchange(int64_t op_id_) : op_id(op_id_) {
    // Create a communicator for all ranks on the node
    CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &this->shuffle_comm),
              "RankDataExchange::RankDataExchange:: MPI error on "
              "MPI_Comm_split_type:");
    this->is_last_state = std::make_unique<IsLastState>(this->shuffle_comm);

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

RankDataExchange::~RankDataExchange() { MPI_Comm_free(&this->shuffle_comm); }

std::tuple<std::shared_ptr<table_info>, OperatorResult>
GPUtoCPUExchange::operator()(std::shared_ptr<table_info> input_batch,
                             OperatorResult prev_op_result) {
    if (!this->shuffle_state) {
        Initialize(input_batch.get());
    }

    // Shuffle data to destination ranks (either all ranks or GPU ranks)
    // and append result to output builder
    std::vector<bool> append_rows(input_batch->nrows(), true);
    this->shuffle_state->AppendBatch(input_batch, append_rows);
    auto result = this->shuffle_state->ShuffleIfRequired(true);
    if (result.has_value()) {
        std::shared_ptr<table_info> shuffled_table = result.value();
        ctb_state->builder->UnifyDictionariesAndAppend(shuffled_table);
    }

    // Determine whether we need more input, have more output, or are finished
    // with the exchange.
    bool request_input = !(this->shuffle_state->BuffersFull() &&
                           (ctb_state->builder->total_remaining >
                            (2 * ctb_state->builder->active_chunk_capacity)));
    bool local_is_last = prev_op_result == OperatorResult::FINISHED &&
                         (this->shuffle_state->SendRecvEmpty());
    bool global_is_last = static_cast<bool>(sync_is_last_non_blocking(
        is_last_state.get(), static_cast<int32_t>(local_is_last)));
    auto [output_batch, _] = ctb_state->builder->PopChunk(
        /*force_return*/ global_is_last);

    bool finished =
        global_is_last && this->ctb_state->builder->total_remaining == 0;

    if (finished) {
        this->shuffle_state->Finalize();
        this->ctb_state->builder->Finalize();
    }
    return std::make_tuple(
        output_batch, finished
                          ? OperatorResult::FINISHED
                          : (request_input ? OperatorResult::NEED_MORE_INPUT
                                           : OperatorResult::HAVE_MORE_OUTPUT));
}

void GPUtoCPUExchange::Initialize(table_info *input_batch) {
    std::unique_ptr<bodo::Schema> table_schema = input_batch->schema();
    ctb_state = std::make_unique<ChunkedTableBuilderState>(
        input_batch->schema(), get_streaming_batch_size());

    uint64_t curr_iter = 0;
    int64_t sync_freq = 1;
    this->shuffle_state = std::make_unique<SrcDestIncrementalShuffleState>(
        input_batch->schema(), ctb_state->dict_builders, gpu_ranks, cpu_ranks,
        curr_iter, sync_freq, this->op_id);
    this->shuffle_state->Initialize(nullptr, true, this->shuffle_comm);
}

std::tuple<GPU_DATA, OperatorResult> CPUtoGPUExchange::operator()(
    std::shared_ptr<table_info> input_batch, std::shared_ptr<StreamAndEvent> se,
    OperatorResult prev_op_result) {
    if (exchange_complete) {
        return std::make_tuple(convertTableToGPU(input_batch, se),
                               OperatorResult::FINISHED);
    }

    if (!this->shuffle_state) {
        Initialize(input_batch, se);
    }

    // Shuffle data to destination GPU ranks
    // and append result to output builder
    std::vector<bool> append_rows(input_batch->nrows(), true);
    this->shuffle_state->AppendBatch(input_batch, append_rows);
    auto result = this->shuffle_state->ShuffleIfRequired(true);

    if (result.has_value()) {
        std::cout << " Result has value, appending...  "
                  << result.value()->nrows() << " rows" << std::endl;
        gpu_batch_generator->append_batch(
            convertTableToGPU(result.value(), se));
        std::cout << " GPU batch generator collected rows: "
                  << gpu_batch_generator->collected_rows << std::endl;
    }
    // Determine whether we need more input, have more output, or are finished
    // with the exchange.
    bool request_input = !(this->shuffle_state->BuffersFull() &&
                           (gpu_batch_generator->collected_rows >
                            (2 * gpu_batch_generator->out_batch_size)));
    if (!request_input) {
        std::cout << " Requesting more input...  " << std::endl;
        std::cout << " Shuffle buffer full: "
                  << this->shuffle_state->BuffersFull() << std::endl;
        std::cout << " GPU batch generator collected rows: "
                  << gpu_batch_generator->collected_rows << std::endl;
    }

    bool local_is_last = prev_op_result == OperatorResult::FINISHED &&
                         (this->shuffle_state->SendRecvEmpty());
    std::cout << " SendRecv Empty?: " << this->shuffle_state->SendRecvEmpty()
              << std::endl;
    auto output_batch = gpu_batch_generator->next(se, local_is_last);
    std::cout << " GPU batch generator collected rows after next was called: "
              << gpu_batch_generator->collected_rows << std::endl;
    std::cout << output_batch.table->num_rows() << " rows in output batch"
              << std::endl;
    bool finished =
        static_cast<bool>(sync_is_last_non_blocking(
            is_last_state.get(), static_cast<int32_t>(local_is_last))) &&
        gpu_batch_generator->collected_rows == 0;

    if (finished) {
        exchange_complete = true;
        this->shuffle_state->Finalize();
    }
    OperatorResult res =
        finished ? OperatorResult::FINISHED
                 : (request_input ? OperatorResult::NEED_MORE_INPUT
                                  : OperatorResult::HAVE_MORE_OUTPUT);
    std::cout << "Exchange result: " << toString(res) << std::endl;

    return std::make_tuple(
        output_batch, finished
                          ? OperatorResult::FINISHED
                          : (request_input ? OperatorResult::NEED_MORE_INPUT
                                           : OperatorResult::HAVE_MORE_OUTPUT));
}

void CPUtoGPUExchange::Initialize(std::shared_ptr<table_info> input_batch,
                                  std::shared_ptr<StreamAndEvent> se) {
    // Initialize GPU batch generator
    auto dummy_gpu_data = convertTableToGPU(alloc_table_like(input_batch), se);
    gpu_batch_generator = std::make_unique<GPUBatchGenerator>(
        dummy_gpu_data, static_cast<size_t>(get_gpu_streaming_batch_size()));

    // Initialize Shuffle State
    uint64_t curr_iter = 0;
    int64_t sync_freq = 1;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    std::unique_ptr<bodo::Schema> table_schema = input_batch->schema();
    for (const std::unique_ptr<bodo::DataType> &t :
         table_schema->column_types) {
        dict_builders.emplace_back(
            create_dict_builder_for_array(t->copy(), false));
    }
    this->shuffle_state = std::make_unique<SrcDestIncrementalShuffleState>(
        input_batch->schema(), dict_builders, cpu_ranks, gpu_ranks, curr_iter,
        sync_freq, this->op_id);
    this->shuffle_state->Initialize(nullptr, true, this->shuffle_comm);
}
#else
std::pair<GPU_DATA, OperatorResult> PhysicalGPUSource::ProduceBatch() {
    throw std::runtime_error("Should never be called in non-CUDF mode.");
}

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
