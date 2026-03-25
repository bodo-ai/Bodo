#pragma once

#include <mpi.h>
#include <algorithm>
#include <cstdint>
#include <cudf/types.hpp>
#include "../../libs/streaming/cuda_join.h"
#include "../_util.h"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "operator.h"

struct PhysicalGPUJoinMetrics {
    using time_t = MetricBase::TimerValue;
    using stat_t = MetricBase::StatValue;

    time_t init_time = 0;
    time_t consume_time = 0;
    time_t process_batch_time = 0;

    stat_t output_row_count = 0;
};

inline bool gpu_capable(duckdb::LogicalComparisonJoin& logical_join) {
    switch (logical_join.join_type) {
        case duckdb::JoinType::OUTER:
        case duckdb::JoinType::RIGHT:
        case duckdb::JoinType::LEFT:
        case duckdb::JoinType::INNER: {
            return true;
        }
        default: {
            return false;
        }
    }

    for (const duckdb::JoinCondition& cond : logical_join.conditions) {
        if (cond.IsComparison() &&
            cond.GetComparisonType() != duckdb::ExpressionType::COMPARE_EQUAL) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Physical node for join.
 *
 */
class PhysicalGPUJoin : public PhysicalGPUProcessBatch, public PhysicalGPUSink {
   private:
    bool doBroadcastJoin(duckdb::LogicalOperator& buildSide,
                         duckdb::LogicalOperator& probeSide) {
        // Get build side row width.
        uint64_t build_row_size = std::transform_reduce(
            buildSide.types.begin(), buildSide.types.end(), 0LL, std::plus<>{},
            [](auto const& s) { return GetTypeIdSize(s.InternalType()); });
        // Get probe side row width.
        uint64_t probe_row_size = std::transform_reduce(
            probeSide.types.begin(), probeSide.types.end(), 0LL, std::plus<>{},
            [](auto const& s) { return GetTypeIdSize(s.InternalType()); });
        uint64_t build_total = buildSide.estimated_cardinality * build_row_size;
        uint64_t probe_total = probeSide.estimated_cardinality * probe_row_size;

        char* bcast_threshold = std::getenv("BODO_BCAST_JOIN_THRESHOLD");
        if (bcast_threshold) {
            return static_cast<int>(build_total) < std::stoi(bcast_threshold);
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        // Do broadcast join if probe table is order of magnitude smaller than
        // probe and it fits on GPU with room for the hash table.
        return (build_total < (probe_total * 0.1)) &&
               ((build_total * 4) < total_bytes);
    }

   public:
    explicit PhysicalGPUJoin(duckdb::LogicalComparisonJoin& logical_join)
        : has_non_equi_cond(false),
          is_mark_join(logical_join.join_type == duckdb::JoinType::MARK),
          is_anti_join(logical_join.join_type == duckdb::JoinType::ANTI ||
                       logical_join.join_type == duckdb::JoinType::RIGHT_ANTI),
          is_broadcast_join(doBroadcastJoin(*logical_join.children[1],
                                            *logical_join.children[0])) {}

    PhysicalGPUJoin(duckdb::LogicalCrossProduct& logical_join,
                    const std::shared_ptr<bodo::Schema> build_table_schema,
                    const std::shared_ptr<bodo::Schema> probe_table_schema)
        : has_non_equi_cond(false),
          is_broadcast_join(doBroadcastJoin(*logical_join.children[1],
                                            *logical_join.children[0])) {
        throw std::runtime_error("Not implemented.");
    }

    /**
     * @brief Determine output schema based on the logical
     * operator and input schemas. Constructs the CudaHashJoin object.
     *
     * @param logical_join - the logical join operator
     * @param conditions - the join conditions
     * @param build_table_schema - the build table schema
     * @param probe_table_schema - the probe table schema
     */
    void buildProbeSchemas(
        duckdb::LogicalComparisonJoin& logical_join,
        duckdb::vector<duckdb::JoinCondition>& conditions,
        const std::shared_ptr<bodo::Schema> build_table_schema,
        const std::shared_ptr<bodo::Schema> probe_table_schema) {
        // Probe side
        duckdb::vector<duckdb::ColumnBinding> left_bindings =
            logical_join.children[0]->GetColumnBindings();
        // Build side
        duckdb::vector<duckdb::ColumnBinding> right_bindings =
            logical_join.children[1]->GetColumnBindings();

        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
            left_col_ref_map = getColRefMap(left_bindings);
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
            right_col_ref_map = getColRefMap(right_bindings);

        std::vector<cudf::size_type> probe_keys;
        std::vector<cudf::size_type> build_keys;

        for (const duckdb::JoinCondition& cond : logical_join.conditions) {
            if (!cond.IsComparison() ||
                cond.GetComparisonType() !=
                    duckdb::ExpressionType::COMPARE_EQUAL) {
                throw std::runtime_error(
                    "Non-equi join conditions are not supported in GPU join.");
            }
            if (cond.GetLHS().GetExpressionClass() !=
                duckdb::ExpressionClass::BOUND_COLUMN_REF) {
                throw std::runtime_error(
                    "Join condition left side is not a column reference.");
            }
            if (cond.GetRHS().GetExpressionClass() !=
                duckdb::ExpressionClass::BOUND_COLUMN_REF) {
                throw std::runtime_error(
                    "Join condition right side is not a column reference.");
            }
            auto& left_bce =
                cond.GetLHS().Cast<duckdb::BoundColumnRefExpression>();
            auto& right_bce =
                cond.GetRHS().Cast<duckdb::BoundColumnRefExpression>();
            probe_keys.push_back(left_col_ref_map[{
                left_bce.binding.table_index, left_bce.binding.column_index}]);
            build_keys.push_back(
                right_col_ref_map[{right_bce.binding.table_index,
                                   right_bce.binding.column_index}]);
        }

        // Get the indices of kept build columns
        std::set<int64_t> bound_probe_inds;
        std::set<int64_t> bound_build_inds;
        if (logical_join.left_projection_map.empty()) {
            for (duckdb::idx_t i = 0;
                 i < logical_join.children[0]->GetColumnBindings().size();
                 i++) {
                bound_probe_inds.insert(i);
            }
        } else {
            for (const auto& c : logical_join.left_projection_map) {
                bound_probe_inds.insert(c);
            }
        }

        if (logical_join.right_projection_map.empty()) {
            for (duckdb::idx_t i = 0;
                 i < logical_join.children[1]->GetColumnBindings().size();
                 i++) {
                bound_build_inds.insert(i);
            }
        } else {
            for (const auto& c : logical_join.right_projection_map) {
                bound_build_inds.insert(c);
            }
        }

        // Figure out kept columns in output
        std::vector<int64_t> build_kept_cols;
        std::vector<int64_t> probe_kept_cols;
        for (size_t idx = 0; idx < build_table_schema->ncols(); ++idx) {
            if (std::ranges::find(bound_build_inds, idx) ==
                bound_build_inds.end()) {
                continue;
            }
            build_kept_cols.push_back(idx);
        }
        for (size_t idx = 0; idx < probe_table_schema->ncols(); ++idx) {
            if (std::ranges::find(bound_probe_inds, idx) ==
                bound_probe_inds.end()) {
                continue;
            }
            probe_kept_cols.push_back(idx);
        }

        this->output_schema = std::make_shared<bodo::Schema>();
        for (const auto& kept_col : probe_kept_cols) {
            this->output_schema->column_types.push_back(
                probe_table_schema->column_types[kept_col]->copy());
            this->output_schema->column_names.push_back(
                probe_table_schema->column_names[kept_col]);
        }
        for (const auto& kept_col : build_kept_cols) {
            this->output_schema->column_types.push_back(
                build_table_schema->column_types[kept_col]->copy());
            this->output_schema->column_names.push_back(
                build_table_schema->column_names[kept_col]);
        }
        // Indexes are ignored in the Pandas merge if not joining on Indexes.
        // We designate empty metadata to indicate generating a trivial
        // RangeIndex.
        // TODO[BSE-4820]: support joining on Indexes
        this->output_schema->metadata = std::make_shared<bodo::TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));
        this->arrow_schema = this->output_schema->ToArrowSchema();

        this->cuda_join = std::make_unique<CudaHashJoin>(
            build_keys, probe_keys, build_table_schema, probe_table_schema,
            build_kept_cols, probe_kept_cols, output_schema,
            logical_join.join_type, cudf::null_equality::UNEQUAL,
            is_broadcast_join);

        assert(this->output_schema->ncols() ==
               logical_join.GetColumnBindings().size());
    }

    virtual ~PhysicalGPUJoin() = default;

    void FinalizeSink() override { cuda_join->FinalizeBuild(); }

    void FinalizeProcessBatch() override {}

    /**
     * @brief process input tables to build side of join (populate the hash
     * table)
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) override {
        bool local_is_last = prev_op_result == OperatorResult::FINISHED;

        bool global_is_last = cuda_join->BuildConsumeBatch(
            input_batch.table, input_batch.stream_event, local_is_last);

        return global_is_last ? OperatorResult::FINISHED
                              : (cuda_join->is_build_complete()
                                     ? OperatorResult::HAVE_MORE_OUTPUT
                                     : OperatorResult::NEED_MORE_INPUT);
    }

    /**
     * @brief Run join probe on the input batch
     *
     * @param input_batch input batch to probe
     * @return output batch of probe and return flag
     */
    std::pair<GPU_DATA, OperatorResult> ProcessBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) override {
        bool local_is_last = prev_op_result == OperatorResult::FINISHED;

        // TODO(ehsan): implement buffering output similar to CPU join
        bool request_input = true;
        auto [output_table, global_is_last] = cuda_join->ProbeProcessBatch(
            input_batch.table, input_batch.stream_event, se->stream,
            local_is_last);
        GPU_DATA output_gpu_data = {
            output_table != nullptr ? std::move(output_table) : nullptr,
            this->arrow_schema, se};
        if (cuda_join->shuffle_buffer_full()) {
            request_input = false;
        }

        return {output_gpu_data,
                global_is_last
                    ? OperatorResult::FINISHED
                    : (request_input ? OperatorResult::NEED_MORE_INPUT
                                     : OperatorResult::HAVE_MORE_OUTPUT)};
    }

    /**
     * @brief GetResult - just for API compatibility but should never be called
     */
    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        // Join build doesn't return output results
        throw std::runtime_error("GetResult called on a join node.");
    }

    /**
     * @brief Get the output schema of join probe
     *
     * @return std::shared_ptr<bodo::Schema>
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

    std::string ToString() override { return PhysicalGPUSink::ToString(); }

    int64_t getOpId() const { return PhysicalGPUSink::getOpId(); }

    CudaHashJoin* getJoinStatePtr() { return this->cuda_join.get(); }

   private:
    std::shared_ptr<bodo::Schema> output_schema;
    std::shared_ptr<arrow::Schema> arrow_schema;

    bool has_non_equi_cond;
    bool is_mark_join = false;
    bool is_anti_join = false;

    PhysicalGPUJoinMetrics metrics;

    std::unique_ptr<CudaHashJoin> cuda_join;
    bool is_broadcast_join = false;
};
