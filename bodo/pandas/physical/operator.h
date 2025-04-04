#pragma once

#include <memory>
#include <utility>

#include "../libs/_bodo_common.h"

/// TODO
enum class OperatorType : uint8_t {
    SOURCE,
    SINK,
    SOURCE_AND_SINK,
};

/// TODO
enum class OperatorResult : uint8_t {
    NEED_MORE_INPUT,
    HAVE_MORE_OUTPUT,
    FINISHED,
};

enum class ProducerResult : bool { HAVE_MORE_OUTPUT, FINISHED };

/**
 * @brief Physical operators to be used in the execution pipelines (NOTE: they
 * are Bodo classes and not using DuckDB).
 */
class PhysicalOperator {
   public:
    virtual OperatorType operator_type() const = 0;

    bool is_source() const { return operator_type() != OperatorType::SINK; }
    bool is_sink() const { return operator_type() != OperatorType::SOURCE; }

    // Constructor is always required for initialization
    // We should have a separate Finalize step that can throw an exception
    // as well as the destructor for cleanup
    virtual void Finalize() = 0;
};

class PhysicalSource : public PhysicalOperator {
   public:
    OperatorType operator_type() const override { return OperatorType::SOURCE; }

    virtual std::pair<std::shared_ptr<table_info>, ProducerResult>
    ProduceBatch() = 0;
};

class PhysicalSink : public PhysicalOperator {
   public:
    OperatorType operator_type() const override { return OperatorType::SINK; }

    virtual void ConsumeBatch(std::shared_ptr<table_info> input_batch) = 0;
};

class PhysicalSourceSink : public PhysicalOperator {
   public:
    OperatorType operator_type() const override {
        return OperatorType::SOURCE_AND_SINK;
    }

    virtual std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) = 0;
};
