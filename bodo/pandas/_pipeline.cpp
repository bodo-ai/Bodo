#include "_pipeline.h"

#include "physical/result_collector.h"

#ifdef DEBUG_PIPELINE
#define DEBUG_PIPELINE_BEFORE_CONSUME(rank, sink, prev_op_result)   \
    do {                                                            \
        for (unsigned i = 0; i < idx; ++i)                          \
            std::cout << " ";                                       \
        std::cout << "Rank " << rank                                \
                  << " midPipelineExecute before ConsumeBatch "     \
                  << sink->ToString() << " "                        \
                  << static_cast<int>(prev_op_result) << std::endl; \
    } while (0)
#else
#define DEBUG_PIPELINE_BEFORE_CONSUME(rank, sink, prev_op_result) \
    do {                                                          \
    } while (0)
#endif

#ifdef DEBUG_PIPELINE
#define DEBUG_PIPELINE_AFTER_CONSUME(rank, sink, consume_result)    \
    do {                                                            \
        for (unsigned i = 0; i < idx; ++i)                          \
            std::cout << " ";                                       \
        std::cout << "Rank " << rank                                \
                  << " midPipelineExecute after ConsumeBatch "      \
                  << sink->ToString() << " "                        \
                  << static_cast<int>(consume_result) << std::endl; \
    } while (0)
#else
#define DEBUG_PIPELINE_AFTER_CONSUME(rank, sink, consume_result) \
    do {                                                         \
    } while (0)
#endif

#ifdef DEBUG_PIPELINE
#define DEBUG_PIPELINE_BEFORE_PRODUCE(rank, source)            \
    do {                                                       \
        std::cout << "Rank " << rank                           \
                  << " Pipeline::Execute before ProduceBatch " \
                  << source->ToString() << std::endl;          \
    } while (0)
#else
#define DEBUG_PIPELINE_BEFORE_PRODUCE(rank, source) \
    do {                                            \
    } while (0)
#endif

#ifdef DEBUG_PIPELINE
#define DEBUG_PIPELINE_AFTER_PRODUCE(rank, source, produce_result)  \
    do {                                                            \
        std::cout << "Rank " << rank                                \
                  << " Pipeline::Execute after ProduceBatch "       \
                  << source->ToString() << " "                      \
                  << static_cast<int>(produce_result) << std::endl; \
    } while (0)
#else
#define DEBUG_PIPELINE_AFTER_PRODUCE(rank, source, produce_result) \
    do {                                                           \
    } while (0)
#endif

#ifdef DEBUG_PIPELINE
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

#ifdef DEBUG_PIPELINE
#define DEBUG_PIPELINE_BEFORE_PROCESS(rank, op, prev_op_result)                \
    do {                                                                       \
        for (unsigned i = 0; i < idx; ++i)                                     \
            std::cout << " ";                                                  \
        std::cout << "Rank " << rank                                           \
                  << " midPipelineExecute before ProcessBatch "                \
                  << op->ToString() << " " << static_cast<int>(prev_op_result) \
                  << std::endl;                                                \
    } while (0)
#else
#define DEBUG_PIPELINE_BEFORE_PROCESS(rank, op, prev_op_result) \
    do {                                                        \
    } while (0)
#endif

#ifdef DEBUG_PIPELINE
#define DEBUG_PIPELINE_AFTER_PROCESS(rank, op, prev_op_result)                 \
    do {                                                                       \
        for (unsigned i = 0; i < idx; ++i)                                     \
            std::cout << " ";                                                  \
        std::cout << "Rank " << rank                                           \
                  << " midPipelineExecute after ProcessBatch "                 \
                  << op->ToString() << " " << static_cast<int>(prev_op_result) \
                  << std::endl;                                                \
    } while (0)
#else
#define DEBUG_PIPELINE_AFTER_PROCESS(rank, op, prev_op_result) \
    do {                                                       \
    } while (0)
#endif

#ifdef DEBUG_PIPELINE
#define DEBUG_PIPELINE_SOURCE_FINISHED(rank, source)                        \
    do {                                                                    \
        std::cout                                                           \
            << "Rank " << rank                                              \
            << " Pipeline::Execute calling with empty batch until finished" \
            << source->ToString() << std::endl;                             \
    } while (0)
#else
#define DEBUG_PIPELINE_SOURCE_FINISHED(rank, source) \
    do {                                             \
    } while (0)
#endif

/*
 * This has to be a recursive routine.  Each operator in the pipeline could
 * say that it HAVE_MORE_OUTPUT in which case we need to call it again for the
 * same input batch (or an empty input batch) once we've call the subsequent
 * nodes in the pipeline to process the first set of output.  Each operator
 * could theoretically require multiple (different) iterations in this manner.
 */
bool Pipeline::midPipelineExecute(unsigned idx,
                                  std::shared_ptr<table_info> batch,
                                  OperatorResult prev_op_result, int rank) {
    // Terminate the recursion when we have processed all the operators
    // and only have the sink to go which cannot HAVE_MORE_OUTPUT.
    if (idx >= between_ops.size()) {
        // Iterating here as in the normal section below so that if the sink
        // says HAVE_MORE_OUTPUT that we can iterate with an empty batch.
        while (true) {
            DEBUG_PIPELINE_BEFORE_CONSUME(rank, sink, prev_op_result);
            OperatorResult consume_result =
                sink->ConsumeBatch(batch, prev_op_result);
            DEBUG_PIPELINE_AFTER_CONSUME(rank, sink, consume_result);
            if (consume_result == OperatorResult::FINISHED) {
                return true;
            }
            if (consume_result == OperatorResult::NEED_MORE_INPUT) {
                return false;
            }
            if (batch->nrows() != 0) {
                DEBUG_PIPELINE_CONSUME_LOOP(rank);
                batch = RetrieveTable(batch, std::vector<int64_t>());
            }
        }
    } else {
        // Get the current operator.
        std::shared_ptr<PhysicalProcessBatch>& op = between_ops[idx];
        while (true) {
            DEBUG_PIPELINE_BEFORE_PROCESS(rank, op, prev_op_result);

            // Process this batch with this operator.
            std::pair<std::shared_ptr<table_info>, OperatorResult> result =
                op->ProcessBatch(batch, prev_op_result);
            prev_op_result = result.second;

            DEBUG_PIPELINE_AFTER_PROCESS(rank, op, prev_op_result);

            // Execute subsequent operators and if any of them said that
            // no more output is needed or the current operator knows no
            // more output is needed then return true to terminate the pipeline.
            if (midPipelineExecute(idx + 1, result.first, prev_op_result,
                                   rank)) {
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
            if (batch->nrows() != 0) {
                batch = RetrieveTable(batch, std::vector<int64_t>());
            }
        }
    }
}

uint64_t Pipeline::Execute() {
    // TODO: Do we need an explicit Init phase to measure initialization time
    // outside of the time spend in constructors?

    uint64_t batches_processed = 0;
    bool finished = false;
    std::shared_ptr<table_info> batch;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    while (!finished) {
        batches_processed++;

        DEBUG_PIPELINE_BEFORE_PRODUCE(rank, source);

        // Execute the source to get the base batch
        std::pair<std::shared_ptr<table_info>, OperatorResult> result =
            source->ProduceBatch();
        batch = result.first;
        // Use NEED_MORE_INPUT for sources
        // just for compatibility with other operators' input expectations and
        // simplifying the pipeline code.
        OperatorResult produce_result =
            result.second == OperatorResult::FINISHED
                ? OperatorResult::FINISHED
                : OperatorResult::NEED_MORE_INPUT;

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
        if (!finished && result.second == OperatorResult::FINISHED) {
            batch = RetrieveTable(batch, std::vector<int64_t>());
            break;
        }
    }

    // Iterate passing empty batch to the first op until it says it is done.
    while (!finished) {
        DEBUG_PIPELINE_SOURCE_FINISHED(rank, source);
        finished = midPipelineExecute(0, batch, OperatorResult::FINISHED, rank);
    }

    // Finalize
    source->FinalizeSource();

    for (auto& op : between_ops) {
        op->FinalizeProcessBatch();
    }
    sink->FinalizeSink();

    executed = true;
    return batches_processed;
}

std::variant<std::shared_ptr<table_info>, PyObject*> Pipeline::GetResult() {
    return sink->GetResult();
}

std::shared_ptr<Pipeline> PipelineBuilder::Build(
    std::shared_ptr<PhysicalSink> sink) {
    auto pipeline = std::make_shared<Pipeline>();
    pipeline->source = source;
    pipeline->between_ops = std::move(between_ops);
    pipeline->sink = sink;
    pipeline->executed = false;
    return pipeline;
}

std::shared_ptr<Pipeline> PipelineBuilder::BuildEnd(
    std::shared_ptr<bodo::Schema> in_schema,
    std::shared_ptr<bodo::Schema> out_schema) {
    auto sink =
        std::make_shared<PhysicalResultCollector>(in_schema, out_schema);
    return Build(sink);
}
