#include "_pipeline.h"
#include "physical/operator.h"

#ifdef USE_CUDF
#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>
#endif  // USE_CUDF
#include "physical/operator.h"
#include "physical/result_collector.h"

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#include <chrono>
using hrclock = std::chrono::high_resolution_clock;
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_BEFORE_CONSUME(rank, sink, prev_op_result, out, batch) \
    do {                                                                      \
        for (unsigned i = 0; i < idx; ++i)                                    \
            out << " ";                                                       \
        out << "Rank " << rank << " midPipelineExecute before ConsumeBatch "  \
            << getNodeString(sink) << " " << toString(prev_op_result)         \
            << " NumRows=>" << getBatchRows(batch) << std::endl;              \
    } while (0)
#else
#define DEBUG_PIPELINE_BEFORE_CONSUME(rank, sink, prev_op_result, out, batch) \
    do {                                                                      \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_AFTER_CONSUME(rank, sink, consume_result, out)       \
    do {                                                                    \
        for (unsigned i = 0; i < idx; ++i)                                  \
            out << " ";                                                     \
        out << "Rank " << rank << " midPipelineExecute after ConsumeBatch " \
            << getNodeString(sink) << " " << toString(consume_result)       \
            << std::endl;                                                   \
    } while (0)
#else
#define DEBUG_PIPELINE_AFTER_CONSUME(rank, sink, consume_result, out) \
    do {                                                              \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_BEFORE_PRODUCE(rank, source, out)                    \
    do {                                                                    \
        out << "Rank " << rank << " Pipeline::Execute before ProduceBatch " \
            << getNodeString(source) << std::endl;                          \
    } while (0)
#else
#define DEBUG_PIPELINE_BEFORE_PRODUCE(rank, source, out) \
    do {                                                 \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_AFTER_PRODUCE(rank, source, produce_result, out, batch) \
    do {                                                                       \
        out << "Rank " << rank << " Pipeline::Execute after ProduceBatch "     \
            << getNodeString(source) << " " << toString(produce_result)        \
            << " NumRows=>" << getBatchRows(batch) << std::endl;               \
        if (DEBUG_PIPELINE >= 2) {                                             \
            printBatchTypes(out, batch);                                       \
        }                                                                      \
    } while (0)
#else
#define DEBUG_PIPELINE_AFTER_PRODUCE(rank, source, produce_result, out, batch) \
    do {                                                                       \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_CONSUME_LOOP(rank, out)                                \
    do {                                                                      \
        out << "Rank " << rank                                                \
            << " Looping in consume part of midPipelineExecute" << std::endl; \
    } while (0)
#else
#define DEBUG_PIPELINE_CONSUME_LOOP(rank, out) \
    do {                                       \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_BEFORE_PROCESS(rank, op, prev_op_result, out, batch)  \
    auto before_process_start = hrclock::now();                              \
    do {                                                                     \
        for (unsigned i = 0; i < idx; ++i)                                   \
            out << " ";                                                      \
        out << "Rank " << rank << " midPipelineExecute before ProcessBatch " \
            << getNodeString(op) << " " << toString(prev_op_result)          \
            << " NumRows=>" << getBatchRows(batch) << std::endl;             \
        if (DEBUG_PIPELINE >= 2) {                                           \
            printBatchTypes(out, batch);                                     \
        }                                                                    \
    } while (0)
#else
#define DEBUG_PIPELINE_BEFORE_PROCESS(rank, op, prev_op_result, out, batch) \
    do {                                                                    \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_AFTER_PROCESS(rank, op, prev_op_result, out, batch)    \
    do {                                                                      \
        auto after_process_start = hrclock::now();                            \
        auto diff_ms = std::chrono::duration_cast<std::chrono::microseconds>( \
                           after_process_start - before_process_start)        \
                           .count();                                          \
        for (unsigned i = 0; i < idx; ++i)                                    \
            out << " ";                                                       \
        out << "Rank " << rank << " midPipelineExecute after ProcessBatch "   \
            << getNodeString(op) << " " << toString(prev_op_result) << " "    \
            << " NumRows=>" << getBatchRows(batch) << " " << diff_ms << "us"  \
            << std::endl;                                                     \
        if (DEBUG_PIPELINE >= 2) {                                            \
            printBatchTypes(out, batch);                                      \
        }                                                                     \
    } while (0)
#else
#define DEBUG_PIPELINE_AFTER_PROCESS(rank, op, prev_op_result, out, batch) \
    do {                                                                   \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_SOURCE_FINISHED(rank, source, out)                   \
    do {                                                                    \
        out << "Rank " << rank                                              \
            << " Pipeline::Execute calling with empty batch until finished" \
            << getNodeString(source) << std::endl;                          \
    } while (0)
#else
#define DEBUG_PIPELINE_SOURCE_FINISHED(rank, source, out) \
    do {                                                  \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_FINALIZE(rank, op, out)                           \
    do {                                                                 \
        out << "Rank " << rank << " Pipeline::Execute calling Finalize " \
            << getNodeString(op) << std::endl;                           \
    } while (0)
#else
#define DEBUG_PIPELINE_FINALIZE(rank, op, out) \
    do {                                       \
    } while (0)
#endif

#if defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 3)
#define DEBUG_PIPELINE_IN_BATCH(rank, op, batch, out)                        \
    do {                                                                     \
        for (unsigned i = 0; i < idx; ++i)                                   \
            out << " ";                                                      \
        out << "Rank " << rank << " midPipelineExecute in batch "            \
            << getNodeString(op) << " NumRows=>" << getBatchRows(batch)      \
            << std::endl;                                                    \
        std::visit(                                                          \
            [&](auto &x) {                                                   \
                using T = std::decay_t<decltype(x)>;                         \
                if constexpr (std::is_same_v<T,                              \
                                             std::shared_ptr<table_info>>) { \
                    DEBUG_PrintTable(out, x, true);                          \
                }                                                            \
            },                                                               \
            batch);                                                          \
    } while (0)
#elif defined(DEBUG_PIPELINE) && (DEBUG_PIPELINE >= 1)
#define DEBUG_PIPELINE_IN_BATCH(rank, op, batch, out)                   \
    do {                                                                \
        for (unsigned i = 0; i < idx; ++i)                              \
            out << " ";                                                 \
        out << "Rank " << rank << " midPipelineExecute in batch "       \
            << getNodeString(op) << " NumRows=>" << getBatchRows(batch) \
            << std::endl;                                               \
        if (DEBUG_PIPELINE >= 2) {                                      \
            printBatchTypes(out, batch);                                \
        }                                                               \
    } while (0)
#else
#define DEBUG_PIPELINE_IN_BATCH(rank, op, batch, out) \
    do {                                              \
    } while (0)
#endif

/*
 * This has to be a recursive routine.  Each operator in the pipeline could
 * say that it HAVE_MORE_OUTPUT in which case we need to call it again for the
 * same input batch (or an empty input batch) once we've call the subsequent
 * nodes in the pipeline to process the first set of output.  Each operator
 * could theoretically require multiple (different) iterations in this manner.
 */
bool Pipeline::midPipelineExecute(
    unsigned idx, std::variant<std::shared_ptr<table_info>, GPU_DATA> batch,
    OperatorResult prev_op_result, int rank, std::ostream &out) {
    // Terminate the recursion when we have processed all the operators
    // and only have the sink to go which cannot HAVE_MORE_OUTPUT.
    if (idx >= between_ops.size()) {
        DEBUG_PIPELINE_IN_BATCH(rank, sink, batch, out);
        // Iterating here as in the normal section below so that if the sink
        // says HAVE_MORE_OUTPUT that we can iterate with an empty batch.
        while (true) {
            DEBUG_PIPELINE_BEFORE_CONSUME(rank, sink, prev_op_result, out,
                                          batch);
            OperatorResult consume_result;
            std::visit(
                [&](auto &vop) {
                    std::visit(
                        [&](auto &vbatch) {
                            consume_result =
                                vop->ConsumeBatch(vbatch, prev_op_result);
                        },
                        batch);
                },
                sink);
            DEBUG_PIPELINE_AFTER_CONSUME(rank, sink, consume_result, out);
            if (consume_result == OperatorResult::FINISHED) {
                return true;
            }
            if (consume_result == OperatorResult::NEED_MORE_INPUT) {
                return false;
            }
            size_t batch_nrows = getBatchRows(batch);
            if (batch_nrows != 0) {
                DEBUG_PIPELINE_CONSUME_LOOP(rank, out);
                std::visit(
                    [&](auto &x) {
                        using T = std::decay_t<decltype(x)>;
                        if constexpr (std::is_same_v<
                                          T, std::shared_ptr<table_info>>) {
                            batch = RetrieveTable(x, std::vector<int64_t>());
#ifdef USE_CUDF
                        } else {
                            auto empty_se = make_stream_and_event(g_use_async);
                            x.stream_event->event.wait(empty_se->stream);
                            batch = GPU_DATA(make_empty_like(x.table, empty_se),
                                             x.schema, empty_se);
                            empty_se->event.record(empty_se->stream);
#endif
                        }
                    },
                    batch);
            }
        }
    } else {
        // Get the current operator.
        PhysicalCpuGpuProcessBatch &op = between_ops[idx];
        DEBUG_PIPELINE_IN_BATCH(rank, op, batch, out);
        while (true) {
            DEBUG_PIPELINE_BEFORE_PROCESS(rank, op, prev_op_result, out, batch);

            // Process this batch with this operator.
            std::variant<std::pair<std::shared_ptr<table_info>, OperatorResult>,
                         std::pair<GPU_DATA, OperatorResult>>
                result;
            std::visit(
                [&](auto &vop) {
                    std::visit(
                        [&](auto &vbatch) {
                            auto pb_res =
                                vop->ProcessBatch(vbatch, prev_op_result);
                            result = pb_res;
                            prev_op_result = pb_res.second;
                        },
                        batch);
                },
                op);

            DEBUG_PIPELINE_AFTER_PROCESS(rank, op, prev_op_result, out, result);

            // Execute subsequent operators and if any of them said that
            // no more output is needed or the current operator knows no
            // more output is needed then return true to terminate the pipeline.
            bool mpe_res;
            std::visit(
                [&](auto &vres) {
                    mpe_res = midPipelineExecute(idx + 1, vres.first,
                                                 prev_op_result, rank, out);
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
#ifdef USE_CUDF
                            auto empty_se = make_stream_and_event(g_use_async);
                            x.stream_event->event.wait(empty_se->stream);
                            batch = GPU_DATA(make_empty_like(x.table, empty_se),
                                             x.schema, empty_se);
                            empty_se->event.record(empty_se->stream);
#endif
                        }
                    },
                    batch);
            }
        }
    }
}

uint64_t Pipeline::Execute(int rank, std::ostream &out) {
    // TODO: Do we need an explicit Init phase to measure initialization time
    // outside of the time spend in constructors?

    uint64_t batches_processed = 0;
    bool finished = false;
    std::variant<std::shared_ptr<table_info>, GPU_DATA> batch;
    while (!finished) {
        batches_processed++;

        DEBUG_PIPELINE_BEFORE_PRODUCE(rank, source, out);

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

        DEBUG_PIPELINE_AFTER_PRODUCE(rank, source, produce_result, out, result);
        // Run the between_ops and sink of the pipeline allowing repetition
        // in the HAVE_MORE_OUTPUT case.
        finished = midPipelineExecute(0, batch, produce_result, rank, out);

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
#ifdef USE_CUDF
                        if (is_gpu_rank()) {
                            auto empty_se = make_stream_and_event(g_use_async);
                            x.stream_event->event.wait(empty_se->stream);
                            batch = GPU_DATA(make_empty_like(x.table, empty_se),
                                             x.schema, empty_se);
                            empty_se->event.record(empty_se->stream);
                        } else {
                            // Non-GPU ranks return nullptr so just pass it
                            // along
                            batch = x;
                        }
#endif
                    }
                },
                batch);
            break;
        }
    }

    // Iterate passing empty batch to the first op until it says it is done.
    while (!finished) {
        DEBUG_PIPELINE_SOURCE_FINISHED(rank, source, out);
        finished =
            midPipelineExecute(0, batch, OperatorResult::FINISHED, rank, out);
    }

    DEBUG_PIPELINE_FINALIZE(rank, source, out);
    // Finalize
    std::visit([&](auto &vop) { vop->FinalizeSource(); }, source);

    for (auto &op : between_ops) {
        DEBUG_PIPELINE_FINALIZE(rank, op, out);
        std::visit([&](auto &vop) { vop->FinalizeProcessBatch(); }, op);
    }
    DEBUG_PIPELINE_FINALIZE(rank, sink, out);
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
