#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "physical/operator.h"
#include "physical/result_collector.h"

/// @brief Pipeline class for executing a sequence of physical operators.
class Pipeline {
   private:
    std::shared_ptr<PhysicalSource> source;
    std::vector<std::shared_ptr<PhysicalSourceSink>> between_ops;
    std::shared_ptr<PhysicalSink> sink;

    friend class PipelineBuilder;

   public:
    /**
     * @brief Execute the pipeline and return the result (placeholder for now).
     */
    void Execute();
    PyObject* GetResult();
};

class PipelineBuilder {
   private:
    std::shared_ptr<PhysicalSource> source;
    std::vector<std::shared_ptr<PhysicalSourceSink>> between_ops;

   public:
    explicit PipelineBuilder(std::shared_ptr<PhysicalSource> _source)
        : source(std::move(_source)) {};

    // Add a physical operator to the pipeline
    void AddOperator(std::shared_ptr<PhysicalSourceSink> op) {
        between_ops.emplace_back(op);
    }

    // Build the pipeline and return it
    std::shared_ptr<Pipeline> Build(std::shared_ptr<PhysicalSink> sink) {
        auto pipeline = std::make_shared<Pipeline>();
        pipeline->source = source;
        pipeline->between_ops = std::move(between_ops);
        pipeline->sink = sink;
        return pipeline;
    }

    /// @brief Build the last pipeline for a plan, using a result collector as
    /// the sink.
    std::shared_ptr<Pipeline> BuildEnd() {
        auto sink = std::make_shared<PhysicalResultCollector>();
        return Build(sink);
    }
};
