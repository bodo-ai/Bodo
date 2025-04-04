#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "_executor.h"
#include "physical/operator.h"

class PipelineBuilder {
   private:
    std::shared_ptr<PhysicalSource> source;
    std::vector<std::shared_ptr<PhysicalSourceSink>> between_ops;
    std::optional<std::shared_ptr<PhysicalSink>> sink;

   public:
    explicit PipelineBuilder(std::shared_ptr<PhysicalSource> _source)
        : source(std::move(_source)) {};

    // Add a physical operator to the pipeline
    void AddOperator(std::shared_ptr<PhysicalOperator> op) {
        operators.push_back(std::move(op));
    }

    // Build the pipeline and return it
    std::shared_ptr<Pipeline> Build() {
        return std::make_shared<Pipeline>(operators);
    }
};

class PhysicalConverter
