#pragma once

#include <memory>
#include <utility>
#include "../io/arrow_reader.h"
#include "../libs/_array_utils.h"
#include "../libs/_utils.h"
#include "../libs/groupby/_groupby_ftypes.h"
#include "../libs/streaming/_groupby.h"
#include "expression.h"
#include "operator.h"

/**
 * @brief Physical node for groupby aggregation
 *
 */
class PhysicalAggregate : public PhysicalSource, public PhysicalSink {
   public:
    explicit PhysicalAggregate(
        std::shared_ptr<bodo::Schema> in_table_schema,
        std::vector<duckdb::unique_ptr<duckdb::Expression>>& agg_exprs,
        std::vector<duckdb::unique_ptr<duckdb::Expression>>& group_exprs) {
        std::vector<bool> cols_to_keep_vec(in_table_schema->ncols(), true);

        // TODO: handle keys properly
        uint64_t n_keys = 1;

        this->groupby_state = std::make_unique<GroupbyState>(
            std::make_unique<bodo::Schema>(*in_table_schema),
            // TODO
            std::vector<int32_t>({Bodo_FTypes::sum}),  // ftypes
            std::vector<int32_t>(),
            // TODO
            std::vector<int32_t>({0, 1}),  // f_in_offsets
            // TODO
            std::vector<int32_t>({1}),  // f_in_cols
            n_keys, std::vector<bool>(), std::vector<bool>(), cols_to_keep_vec,
            nullptr, get_streaming_batch_size(), true, -1, -1, -1);

        // TODO
        this->output_schema = std::make_shared<bodo::Schema>();
        output_schema->append_column(in_table_schema->column_types[0]->copy());
        output_schema->append_column(in_table_schema->column_types[1]->copy());
        output_schema->metadata = in_table_schema->metadata;
        output_schema->column_names = in_table_schema->column_names;
    }

    virtual ~PhysicalAggregate() = default;

    void Finalize() override {}

    /**
     * @brief process input tables to groupby build (populate the build
     * table)
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        bool local_is_last = prev_op_result == OperatorResult::FINISHED;
        bool request_input = true;
        bool global_is_last =
            groupby_build_consume_batch(this->groupby_state.get(), input_batch,
                                        local_is_last, true, &request_input);

        if (global_is_last) {
            return OperatorResult::FINISHED;
        }
        return request_input ? OperatorResult::NEED_MORE_INPUT
                             : OperatorResult::HAVE_MORE_OUTPUT;
    }

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        bool out_is_last = false;
        std::shared_ptr<table_info> next_batch =
            groupby_produce_output_batch_wrapper(this->groupby_state.get(),
                                                 &out_is_last, true);
        return {next_batch, out_is_last ? OperatorResult::FINISHED
                                        : OperatorResult::HAVE_MORE_OUTPUT};
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::shared_ptr<table_info> GetResult() override {
        throw std::runtime_error("GetResult called on an aggregate node.");
    }

    /**
     * @brief Get the output schema of groupby aggregation
     *
     * @return std::shared_ptr<bodo::Schema>
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

   private:
    std::shared_ptr<GroupbyState> groupby_state;
    std::shared_ptr<bodo::Schema> output_schema;
};

/**
 * @brief Physical node for count_star().
 *
 */
class PhysicalCountStar : public PhysicalSource, public PhysicalSink {
   public:
    explicit PhysicalCountStar() : local_count(0), global_count(0) {
        std::vector<std::unique_ptr<bodo::DataType>> types;
        types.emplace_back(std::make_unique<bodo::DataType>(
            bodo_array_type::arr_type_enum::NULLABLE_INT_BOOL,
            Bodo_CTypes::CTypeEnum::UINT64));
        std::vector<std::string> names = {std::string("count_star()")};
        out_schema = std::make_shared<bodo::Schema>(std::move(types), names);
        out_schema->metadata = std::make_shared<TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));
    }

    virtual ~PhysicalCountStar() = default;

    void Finalize() override {
        int result =
            MPI_Allreduce(&local_count, &global_count, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (result != MPI_SUCCESS) {
            throw std::runtime_error(
                "PhysicalCountStar::Finalize MPI_Allreduce failed.");
        }
    }

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        local_count += input_batch->nrows();
        return prev_op_result == OperatorResult::FINISHED
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    std::shared_ptr<table_info> GetResult() override {
        throw std::runtime_error(
            "GetResult called on a PhysicalCountStar node.");
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return out_schema;
    }

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        std::shared_ptr<arrow::Array> array =
            CreateOneElementArrowArray(global_count);

        std::shared_ptr<array_info> result =
            arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
        std::vector<std::shared_ptr<array_info>> cvec = {result};
        std::shared_ptr<table_info> next_batch =
            std::make_shared<table_info>(cvec);
        next_batch->metadata = out_schema->metadata;
        return {next_batch, OperatorResult::FINISHED};
    }

   private:
    uint64_t local_count, global_count;
    std::shared_ptr<bodo::Schema> out_schema;
};
