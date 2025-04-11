#pragma once

#include "../libs/_table_builder.h"

#include "physical/operator.h"

class PhysicalResultCollector : public PhysicalSink {
   private:
    TableBuildBuffer buffer;

   public:
    explicit PhysicalResultCollector(std::shared_ptr<bodo::Schema> schema)
        : buffer(schema, {}) {}

    virtual ~PhysicalResultCollector() = default;

    void ConsumeBatch(std::shared_ptr<table_info> input_batch) override {
        buffer.ReserveTable(input_batch);
        buffer.UnsafeAppendBatch(input_batch);
    }

    void Finalize() override {}
};
