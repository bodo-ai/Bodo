#include "_pipeline.h"
#include "physical/operator.h"

#include <arrow/c/bridge.h>
#include <arrow/table.h>
#include <cudf/copying.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table_view.hpp>
#include "physical/operator.h"
#include "physical/result_collector.h"

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_BEFORE_CONSUME(rank, sink, prev_op_result)   \
    do {                                                            \
        for (unsigned i = 0; i < idx; ++i)                          \
            std::cout << " ";                                       \
        std::cout << "Rank " << rank                                \
                  << " midPipelineExecute before ConsumeBatch "     \
                  << getNodeString(sink) << " "                     \
                  << static_cast<int>(prev_op_result) << std::endl; \
    } while (0)
#else
#define DEBUG_PIPELINE_BEFORE_CONSUME(rank, sink, prev_op_result) \
    do {                                                          \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_AFTER_CONSUME(rank, sink, consume_result)    \
    do {                                                            \
        for (unsigned i = 0; i < idx; ++i)                          \
            std::cout << " ";                                       \
        std::cout << "Rank " << rank                                \
                  << " midPipelineExecute after ConsumeBatch "      \
                  << getNodeString(sink) << " "                     \
                  << static_cast<int>(consume_result) << std::endl; \
    } while (0)
#else
#define DEBUG_PIPELINE_AFTER_CONSUME(rank, sink, consume_result) \
    do {                                                         \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_BEFORE_PRODUCE(rank, source)            \
    do {                                                       \
        std::cout << "Rank " << rank                           \
                  << " Pipeline::Execute before ProduceBatch " \
                  << getNodeString(source) << std::endl;       \
    } while (0)
#else
#define DEBUG_PIPELINE_BEFORE_PRODUCE(rank, source) \
    do {                                            \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_AFTER_PRODUCE(rank, source, produce_result)  \
    do {                                                            \
        std::cout << "Rank " << rank                                \
                  << " Pipeline::Execute after ProduceBatch "       \
                  << getNodeString(source) << " "                   \
                  << static_cast<int>(produce_result) << std::endl; \
    } while (0)
#else
#define DEBUG_PIPELINE_AFTER_PRODUCE(rank, source, produce_result) \
    do {                                                           \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_CONSUME_LOOP(rank)                             \
    do {                                                              \
        std::cout << "Rank " << rank                                  \
                  << " Looping in consume part of midPipelineExecute" \
                  << std::endl;                                       \
    } while (0)
#else
#define DEBUG_PIPELINE_CONSUME_LOOP(rank) \
    do {                                  \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_BEFORE_PROCESS(rank, op, prev_op_result)     \
    do {                                                            \
        for (unsigned i = 0; i < idx; ++i)                          \
            std::cout << " ";                                       \
        std::cout << "Rank " << rank                                \
                  << " midPipelineExecute before ProcessBatch "     \
                  << getNodeString(op) << " "                       \
                  << static_cast<int>(prev_op_result) << std::endl; \
    } while (0)
#else
#define DEBUG_PIPELINE_BEFORE_PROCESS(rank, op, prev_op_result) \
    do {                                                        \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_AFTER_PROCESS(rank, op, prev_op_result)      \
    do {                                                            \
        for (unsigned i = 0; i < idx; ++i)                          \
            std::cout << " ";                                       \
        std::cout << "Rank " << rank                                \
                  << " midPipelineExecute after ProcessBatch "      \
                  << getNodeString(op) << " "                       \
                  << static_cast<int>(prev_op_result) << std::endl; \
    } while (0)
#else
#define DEBUG_PIPELINE_AFTER_PROCESS(rank, op, prev_op_result) \
    do {                                                       \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_SOURCE_FINISHED(rank, source)                        \
    do {                                                                    \
        std::cout                                                           \
            << "Rank " << rank                                              \
            << " Pipeline::Execute calling with empty batch until finished" \
            << getNodeString(source) << std::endl;                          \
    } while (0)
#else
#define DEBUG_PIPELINE_SOURCE_FINISHED(rank, source) \
    do {                                             \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_FINALIZE(rank, op)                                      \
    do {                                                                       \
        std::cout << "Rank " << rank << " Pipeline::Execute calling Finalize " \
                  << getNodeString(op) << std::endl;                           \
    } while (0)
#else
#define DEBUG_PIPELINE_FINALIZE(rank, op) \
    do {                                  \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 2)
#define DEBUG_PIPELINE_IN_BATCH(rank, op, batch)                        \
    do {                                                                \
        for (unsigned i = 0; i < idx; ++i)                              \
            std::cout << " ";                                           \
        std::cout << "Rank " << rank << " midPipelineExecute in batch " \
                  << getNodeString(op) << " " << getBatchRows(batch)    \
                  << std::endl;                                         \
        DEBUG_PrintTable(std::cout, batch);                             \
    } while (0)
#elif defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_IN_BATCH(rank, op, batch)                        \
    do {                                                                \
        for (unsigned i = 0; i < idx; ++i)                              \
            std::cout << " ";                                           \
        std::cout << "Rank " << rank << " midPipelineExecute in batch " \
                  << getNodeString(op) << " " << getBatchRows(batch)    \
                  << std::endl;                                         \
    } while (0)
#else
#define DEBUG_PIPELINE_IN_BATCH(rank, op, batch) \
    do {                                         \
    } while (0)
#endif

std::shared_ptr<table_info> convertGPUToTable(GPU_DATA batch) {
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
    if (!batch_result.ok() || batch_result.ValueOrDie() == nullptr) {
        throw std::runtime_error("Failed to extract batch from Arrow table");
    }
    arrow_batch = batch_result.ValueOrDie();

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

std::shared_ptr<cudf::table> make_empty_like(
    std::shared_ptr<cudf::table> input_table) {
    cudf::table_view tv = input_table->view();

    // slice produces a vector<table_view>
    auto sliced = cudf::slice(tv, {0, 0});
    cudf::table_view empty_view = sliced[0];

    // materialize into a real cudf::table
    return std::make_shared<cudf::table>(empty_view);
}

/*
 * This has to be a recursive routine.  Each operator in the pipeline could
 * say that it HAVE_MORE_OUTPUT in which case we need to call it again for the
 * same input batch (or an empty input batch) once we've call the subsequent
 * nodes in the pipeline to process the first set of output.  Each operator
 * could theoretically require multiple (different) iterations in this manner.
 */
bool Pipeline::midPipelineExecute(
    unsigned idx, std::variant<std::shared_ptr<table_info>, GPU_DATA> batch,
    OperatorResult prev_op_result, int rank) {
    // Terminate the recursion when we have processed all the operators
    // and only have the sink to go which cannot HAVE_MORE_OUTPUT.
    if (idx >= between_ops.size()) {
        DEBUG_PIPELINE_IN_BATCH(rank, sink, batch);
        // Iterating here as in the normal section below so that if the sink
        // says HAVE_MORE_OUTPUT that we can iterate with an empty batch.
        while (true) {
            DEBUG_PIPELINE_BEFORE_CONSUME(rank, sink, prev_op_result);
            OperatorResult consume_result;
            std::visit(
                [&](auto &vop) {
                    std::visit(
                        [&](auto &vbatch) {
                            auto converted =
                                prepare_batch_for_operator(vop, vbatch);
                            consume_result =
                                vop->ConsumeBatch(converted, prev_op_result);
                        },
                        batch);
                },
                sink);
            DEBUG_PIPELINE_AFTER_CONSUME(rank, sink, consume_result);
            if (consume_result == OperatorResult::FINISHED) {
                return true;
            }
            if (consume_result == OperatorResult::NEED_MORE_INPUT) {
                return false;
            }
            size_t batch_nrows = getBatchRows(batch);
            if (batch_nrows != 0) {
                DEBUG_PIPELINE_CONSUME_LOOP(rank);
                std::visit(
                    [&](auto &x) {
                        using T = std::decay_t<decltype(x)>;
                        if constexpr (std::is_same_v<
                                          T, std::shared_ptr<table_info>>) {
                            batch = RetrieveTable(x, std::vector<int64_t>());
                        } else {
                            batch =
                                GPU_DATA(make_empty_like(x.table), x.schema);
                        }
                    },
                    batch);
            }
        }
    } else {
        // Get the current operator.
        PhysicalCpuGpuProcessBatch &op = between_ops[idx];
        DEBUG_PIPELINE_IN_BATCH(rank, op, batch);
        while (true) {
            DEBUG_PIPELINE_BEFORE_PROCESS(rank, op, prev_op_result);

            // Process this batch with this operator.
            std::variant<std::pair<std::shared_ptr<table_info>, OperatorResult>,
                         std::pair<GPU_DATA, OperatorResult>>
                result;
            std::visit(
                [&](auto &vop) {
                    std::visit(
                        [&](auto &vbatch) {
                            auto converted =
                                prepare_batch_for_operator(vop, vbatch);
                            auto pb_res =
                                vop->ProcessBatch(converted, prev_op_result);
                            result = pb_res;
                            prev_op_result = pb_res.second;
                        },
                        batch);
                },
                op);

            DEBUG_PIPELINE_AFTER_PROCESS(rank, op, prev_op_result);

            // Execute subsequent operators and if any of them said that
            // no more output is needed or the current operator knows no
            // more output is needed then return true to terminate the pipeline.
            bool mpe_res;
            std::visit(
                [&](auto &vres) {
                    mpe_res = midPipelineExecute(idx + 1, vres.first,
                                                 prev_op_result, rank);
                },
                result);
            if (mpe_res) {
                return true;
            }

            // op_result has to be NEED_MORE_INPUT or HAVE_MORE_OUTPUT since
            // FINISHED is checked above.  If this operator is done this part
            // of the pipeline is done and we aren't set to finish yet.
            if (prev_op_result == OperatorResult::NEED_MORE_INPUT) {
                return false;
            }

            // Must be the HAVE_MORE_OUTPUT case so iterate the while loop
            // to give this operator a chance to produce more output.
            // Currently, streaming operators assume input to be empty in this
            // case to match BodoSQL.
            size_t batch_nrows = getBatchRows(batch);
            if (batch_nrows != 0) {
                std::visit(
                    [&](auto &x) {
                        using T = std::decay_t<decltype(x)>;
                        if constexpr (std::is_same_v<
                                          T, std::shared_ptr<table_info>>) {
                            batch = RetrieveTable(x, std::vector<int64_t>());
                        } else {
                            batch =
                                GPU_DATA(make_empty_like(x.table), x.schema);
                        }
                    },
                    batch);
            }
        }
    }
}

uint64_t Pipeline::Execute() {
    // TODO: Do we need an explicit Init phase to measure initialization time
    // outside of the time spend in constructors?

    uint64_t batches_processed = 0;
    bool finished = false;
    std::variant<std::shared_ptr<table_info>, GPU_DATA> batch;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    while (!finished) {
        batches_processed++;

        DEBUG_PIPELINE_BEFORE_PRODUCE(rank, source);

        OperatorResult produce_result;
        OperatorResult source_result_or;
        std::variant<std::pair<std::shared_ptr<table_info>, OperatorResult>,
                     std::pair<GPU_DATA, OperatorResult>>
            result;
        // Execute the source to get the base batch
        std::visit(
            [&](auto &op) {
                auto pb_res = op->ProduceBatch();
                result = pb_res;
                source_result_or = pb_res.second;
                batch = std::variant<std::shared_ptr<table_info>, GPU_DATA>{
                    std::in_place_type<decltype(pb_res.first)>, pb_res.first};
                produce_result = pb_res.second == OperatorResult::FINISHED
                                     ? OperatorResult::FINISHED
                                     : OperatorResult::NEED_MORE_INPUT;
            },
            source);
        // Use NEED_MORE_INPUT for sources
        // just for compatibility with other operators' input expectations and
        // simplifying the pipeline code.

        DEBUG_PIPELINE_AFTER_PRODUCE(rank, source, produce_result);
        // Run the between_ops and sink of the pipeline allowing repetition
        // in the HAVE_MORE_OUTPUT case.
        finished = midPipelineExecute(0, batch, produce_result, rank);

        // If the next operator in the pipeline isn't finished even though we
        // told it that the input has been exhausted then create an empty batch
        // to pass to that operator until it isn't finished.  We do that looping
        // in the loop below and break out of this one so as not to record
        // additional batches processed and muddy this code with checks for this
        // state.
        if (!finished && source_result_or == OperatorResult::FINISHED) {
            std::visit(
                [&](auto &x) {
                    using T = std::decay_t<decltype(x)>;
                    if constexpr (std::is_same_v<T,
                                                 std::shared_ptr<table_info>>) {
                        batch = RetrieveTable(x, std::vector<int64_t>());
                    } else {
                        batch = GPU_DATA(make_empty_like(x.table), x.schema);
                    }
                },
                batch);
            break;
        }
    }

    // Iterate passing empty batch to the first op until it says it is done.
    while (!finished) {
        DEBUG_PIPELINE_SOURCE_FINISHED(rank, source);
        finished = midPipelineExecute(0, batch, OperatorResult::FINISHED, rank);
    }

    DEBUG_PIPELINE_FINALIZE(rank, source);
    // Finalize
    std::visit([&](auto &vop) { vop->FinalizeSource(); }, source);

    for (auto &op : between_ops) {
        DEBUG_PIPELINE_FINALIZE(rank, op);
        std::visit([&](auto &vop) { vop->FinalizeProcessBatch(); }, op);
    }
    DEBUG_PIPELINE_FINALIZE(rank, sink);
    std::visit([&](auto &vop) { vop->FinalizeSink(); }, sink);

    executed = true;
    return batches_processed;
}

std::variant<std::variant<std::shared_ptr<table_info>, PyObject *>,
             std::variant<GPU_DATA, PyObject *>>
Pipeline::GetResult() {
    std::variant<std::variant<std::shared_ptr<table_info>, PyObject *>,
                 std::variant<GPU_DATA, PyObject *>>
        res;
    std::visit([&](auto &vop) { res = vop->GetResult(); }, sink);
    return res;
}

std::shared_ptr<Pipeline> PipelineBuilder::Build(PhysicalCpuGpuSink sink) {
    auto pipeline = std::make_shared<Pipeline>();
    pipeline->source = source;
    pipeline->between_ops = std::move(between_ops);
    pipeline->sink = sink;
    pipeline->executed = false;
    pipeline->run_before = std::move(run_before);
    return pipeline;
}

std::shared_ptr<Pipeline> PipelineBuilder::BuildEnd(
    std::shared_ptr<bodo::Schema> in_schema,
    std::shared_ptr<bodo::Schema> out_schema) {
    auto sink =
        std::make_shared<PhysicalResultCollector>(in_schema, out_schema);
    return Build(sink);
}
