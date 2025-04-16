#include "_pipeline.h"

#include "physical/result_collector.h"

void Pipeline::Execute() {
    // TODO: Do we need an explicit Init phase to measure initialization time
    // outside of the time spend in constructors?

    bool finished = false;
    while (!finished) {
        finished = true;
        std::shared_ptr<table_info> batch;

        // Execute the source to get the base batch
        std::pair<std::shared_ptr<table_info>, ProducerResult> result =
            source->ProduceBatch();
        batch = result.first;
        ProducerResult produce_result = result.second;
        if (produce_result == ProducerResult::HAVE_MORE_OUTPUT) {
            finished = false;
        }

        for (std::shared_ptr<PhysicalSourceSink>& op : between_ops) {
            std::pair<std::shared_ptr<table_info>, OperatorResult> result =
                op->ProcessBatch(batch);
            batch = result.first;
            OperatorResult op_result = result.second;

            if (op_result == OperatorResult::HAVE_MORE_OUTPUT) {
                finished = false;
            }
        }

        sink->ConsumeBatch(batch);
    }

    // Finalize
    source->Finalize();
    for (auto& op : between_ops) {
        op->Finalize();
    }
    sink->Finalize();
}

std::shared_ptr<table_info> Pipeline::GetResult() { return sink->GetResult(); }

std::shared_ptr<Pipeline> PipelineBuilder::Build(
    std::shared_ptr<PhysicalSink> sink) {
    auto pipeline = std::make_shared<Pipeline>();
    pipeline->source = source;
    pipeline->between_ops = std::move(between_ops);
    pipeline->sink = sink;
    return pipeline;
}

std::shared_ptr<Pipeline> PipelineBuilder::BuildEnd(
    std::shared_ptr<arrow::Schema> out_schema) {
    auto sink = std::make_shared<PhysicalResultCollector>(
        bodo::Schema::FromArrowSchema(out_schema));
    return Build(sink);
}
