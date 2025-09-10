#include "_pipeline.h"

#include "physical/result_collector.h"

/*
 * This has to be a recursive routine.  Each operator in the pipeline could
 * say that it HAVE_MORE_OUTPUT in which case we need to call it again for the
 * same input batch (or an empty input batch) once we've call the subsequent
 * nodes in the pipeline to process the first set of output.  Each operator
 * could theoretically require multiple (different) iterations in this manner.
 */
bool Pipeline::midPipelineExecute(unsigned idx,
                                  std::shared_ptr<table_info> batch,
                                  OperatorResult prev_op_result) {
    // Terminate the recursion when we have processed all the operators
    // and only have the sink to go which cannot HAVE_MORE_OUTPUT.
    if (idx >= between_ops.size()) {
        // Iterating here as in the normal section below so that if the sink
        // says HAVE_MORE_OUTPUT that we can iterate with an empty batch.
        while (true) {
#ifdef DEBUG_PIPELINE
            for (unsigned i = 0; i < idx; ++i)
                std::cout << " ";
            std::cout << "midPipelineExecute before ConsumeBatch "
                      << sink->ToString() << " "
                      << static_cast<int>(prev_op_result) << std::endl;
#endif
            OperatorResult consume_result =
                sink->ConsumeBatch(batch, prev_op_result);
#ifdef DEBUG_PIPELINE
            for (unsigned i = 0; i < idx; ++i)
                std::cout << " ";
            std::cout << "midPipelineExecute after ConsumeBatch "
                      << sink->ToString() << " "
                      << static_cast<int>(consume_result) << std::endl;
#endif
            if (consume_result == OperatorResult::FINISHED) {
                return true;
            }
            if (consume_result == OperatorResult::NEED_MORE_INPUT) {
                return false;
            }
            if (batch->nrows() != 0) {
#ifdef DEBUG_PIPELINE
                std::cout << "Looping in consume part of midPipelineExecute"
                          << std::endl;
#endif
                batch = RetrieveTable(batch, std::vector<int64_t>());
            }
        }
    } else {
        // Get the current operator.
        std::shared_ptr<PhysicalProcessBatch>& op = between_ops[idx];
        while (true) {
#ifdef DEBUG_PIPELINE
            for (unsigned i = 0; i < idx; ++i)
                std::cout << " ";
            std::cout << "midPipelineExecute before ProcessBatch "
                      << op->ToString() << " "
                      << static_cast<int>(prev_op_result) << std::endl;
#endif
            // Process this batch with this operator.
            std::pair<std::shared_ptr<table_info>, OperatorResult> result =
                op->ProcessBatch(batch, prev_op_result);
            prev_op_result = result.second;

#ifdef DEBUG_PIPELINE
            for (unsigned i = 0; i < idx; ++i)
                std::cout << " ";
            std::cout << "midPipelineExecute after ProcessBatch "
                      << op->ToString() << " "
                      << static_cast<int>(prev_op_result) << std::endl;
#endif
            // Execute subsequent operators and if any of them said that
            // no more output is needed or the current operator knows no
            // more output is needed then return true to terminate the pipeline.
            if (midPipelineExecute(idx + 1, result.first, prev_op_result)) {
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
    while (!finished) {
        batches_processed++;

#ifdef DEBUG_PIPELINE
        std::cout << "Pipeline::Execute before ProduceBatch "
                  << source->ToString() << std::endl;
#endif
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
#ifdef DEBUG_PIPELINE
        std::cout << "Pipeline::Execute after ProduceBatch "
                  << source->ToString() << " "
                  << static_cast<int>(produce_result) << std::endl;
#endif
        // Run the between_ops and sink of the pipeline allowing repetition
        // in the HAVE_MORE_OUTPUT case.
        finished = midPipelineExecute(0, batch, produce_result);

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
        finished = midPipelineExecute(0, batch, OperatorResult::FINISHED);
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
