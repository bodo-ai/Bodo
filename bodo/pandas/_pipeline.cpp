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
#ifdef DEBUG_PIPELINE
        for (unsigned i = 0; i < idx; ++i)
            std::cout << " ";
        std::cout << "midPipelineExecute before ConsumeBatch "
                  << sink->ToString() << std::endl;
#endif
        bodo_cpp_clock::time_point start;
        start = bodo_cpp_clock::now();
        auto ret = sink->ConsumeBatch(batch, prev_op_result) ==
                   OperatorResult::FINISHED;
        timings[sink->ToString()] +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                bodo_cpp_clock::now() - start)
                .count();
#ifdef DEBUG_PIPELINE
        for (unsigned i = 0; i < idx; ++i)
            std::cout << " ";
        std::cout << "midPipelineExecute after ConsumeBatch "
                  << sink->ToString() << std::endl;
#endif
        return ret;
    } else {
        // Get the current operator.
        std::shared_ptr<PhysicalSourceSink>& op = between_ops[idx];
        while (true) {
#ifdef DEBUG_PIPELINE
            for (unsigned i = 0; i < idx; ++i)
                std::cout << " ";
            std::cout << "midPipelineExecute before ProcessBatch "
                      << op->ToString() << std::endl;
#endif
            // Process this batch with this operator.
            bodo_cpp_clock::time_point start;
            start = bodo_cpp_clock::now();
            std::pair<std::shared_ptr<table_info>, OperatorResult> result =
                op->ProcessBatch(batch, prev_op_result);
            timings[std::to_string(idx) + " " + op->ToString()] +=
                std::chrono::duration_cast<std::chrono::microseconds>(
                    bodo_cpp_clock::now() - start)
                    .count();
            prev_op_result = result.second;

#ifdef DEBUG_PIPELINE
            for (unsigned i = 0; i < idx; ++i)
                std::cout << " ";
            std::cout << "midPipelineExecute after ProcessBatch "
                      << op->ToString() << std::endl;
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

void Pipeline::Execute() {
    // TODO: Do we need an explicit Init phase to measure initialization time
    // outside of the time spend in constructors?

    bodo_cpp_clock::time_point start;
    bool finished = false;
    while (!finished) {
        std::shared_ptr<table_info> batch;

#ifdef DEBUG_PIPELINE
        std::cout << "Pipeline::Execute before ProduceBatch "
                  << source->ToString() << std::endl;
#endif
        // Execute the source to get the base batch
        start = bodo_cpp_clock::now();
        std::pair<std::shared_ptr<table_info>, OperatorResult> result =
            source->ProduceBatch();
        timings[source->ToString()] +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                bodo_cpp_clock::now() - start)
                .count();
#ifdef DEBUG_PIPELINE
        std::cout << "Pipeline::Execute after ProduceBatch "
                  << source->ToString() << std::endl;
#endif
        batch = result.first;
        // Use NEED_MORE_INPUT for sources
        // just for compatibility with other operators' input expectations and
        // simplifying the pipeline code.
        OperatorResult produce_result =
            result.second == OperatorResult::FINISHED
                ? OperatorResult::FINISHED
                : OperatorResult::NEED_MORE_INPUT;
        // Run the between_ops and sink of the pipeline allowing repetition
        // in the HAVE_MORE_OUTPUT case.
        finished = midPipelineExecute(0, batch, produce_result);
    }

    // Finalize
    start = bodo_cpp_clock::now();
    source->Finalize();
    for (auto& op : between_ops) {
        op->Finalize();
    }
    sink->Finalize();
    timings["Finalize"] +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            bodo_cpp_clock::now() - start)
            .count();

    std::cout
        << "------------------------Pipeline timing------------------------"
        << std::endl;
    std::vector<std::string> name_order;
    name_order.push_back(source->ToString());
    for (size_t i = 0; i < between_ops.size(); ++i) {
        name_order.push_back(std::to_string(i) + " " +
                             between_ops[i]->ToString());
    }
    name_order.push_back(sink->ToString());
    for (const auto& name : name_order) {
        std::cout << name << " => " << (timings[name] / 1000000.0) << std::endl;
    }
    std::cout
        << "--------------------End Pipeline timing------------------------"
        << std::endl;
    executed = true;
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
