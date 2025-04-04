#include "_pipeline.h"
#include "physical/operator.h"

void Pipeline::Execute() {
    // TODO: Do we need an explicit Init phase to measure initialization time
    // outside of the time spend in constructors?

    bool finished = false;
    while (!finished) {
        finished = true;
        std::shared_ptr<table_info> batch;

        // Execute the source to get the base batch
        auto result = source->ProduceBatch();
        batch = result.first;
        ProducerResult produce_result = result.second;
        if (produce_result == ProducerResult::HAVE_MORE_OUTPUT) {
            finished = false;
        }

        for (std::shared_ptr<PhysicalSourceSink>& op : between_ops) {
            auto result = op->ProcessBatch(batch);
            batch = result.first;
            OperatorResult op_result = result.second;

            if (op_result == OperatorResult::HAVE_MORE_OUTPUT) {
                finished = false;
            }
        }

        sink->ConsumeBatch(result.first);
    }

    // Finalize
    source->Finalize();
    for (auto& op : between_ops) {
        op->Finalize();
    }
    sink->Finalize();
}
