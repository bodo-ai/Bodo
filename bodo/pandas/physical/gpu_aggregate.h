#pragma once

#include <object.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include "../_util.h"
#include "../io/arrow_reader.h"
#include "../libs/_array_utils.h"
#include "../libs/_query_profile_collector.h"
#include "../libs/_table_builder_utils.h"
#include "../libs/_utils.h"
#include "../libs/streaming/cuda_groupby.h"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "expression.h"
#include "operator.h"

struct PhysicalGPUAggregateMetrics {
    using time_t = MetricBase::TimerValue;

    time_t init_time = 0;     // stage_0
    time_t consume_time = 0;  // stage_1
    time_t produce_time = 0;  // stage_2
};

inline bool gpu_capable(duckdb::LogicalAggregate& logical_aggregate) {
    std::cout << "gpu_capable for LogicalAggregate "
              << !logical_aggregate.groups.empty() << std::endl;
    // Temporarily don't support count_start and quantile on GPU.
    return !logical_aggregate.groups.empty();
}

inline bool gpu_capable(duckdb::LogicalDistinct& logical_distinct) {
    return true;
}

/**
 * @brief Physical node for groupby aggregation
 *
 */
class PhysicalGPUAggregate : public PhysicalGPUSource, public PhysicalGPUSink {
   public:
    explicit PhysicalGPUAggregate(std::shared_ptr<bodo::Schema> in_table_schema,
                                  duckdb::LogicalAggregate& op) {
        time_pt start_init = start_timer();
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map =
            getColRefMap(op.children[0]->GetColumnBindings());

        std::shared_ptr<bodo::Schema> in_table_schema_reordered;
        std::vector<bool> cols_to_keep_vec;
        this->initKeysAndSchema(col_ref_map, op.groups, in_table_schema,
                                in_table_schema_reordered, cols_to_keep_vec);

        std::vector<int32_t> ftypes;
        // Create input data column indices
        std::vector<int32_t> f_in_cols;
        std::vector<int32_t> f_in_offsets = {0};
        std::optional<bool> dropna = std::nullopt;

        std::vector<stream_udf_t*> udf_cfuncs;
        std::vector<int> udf_idxs;

        for (size_t i = 0; i < op.expressions.size(); i++) {
            const auto& expr = op.expressions[i];

            if (expr->type != duckdb::ExpressionType::BOUND_AGGREGATE) {
                throw std::runtime_error(
                    "Aggregate expression is not a bound aggregate: " +
                    expr->ToString());
            }
            auto& agg_expr = expr->Cast<duckdb::BoundAggregateExpression>();

            // Check if the aggregate function is supported
            bool is_udf = agg_expr.function.name.starts_with("udf");
            if (function_to_ftype.find(agg_expr.function.name) ==
                    function_to_ftype.end() &&
                !is_udf) {
                throw std::runtime_error("Unsupported aggregate function: " +
                                         agg_expr.function.name);
            }

            if (!is_udf && agg_expr.children.size() != 1 &&
                agg_expr.function.name != "size") {
                throw std::runtime_error(
                    "Aggregate expression for builtin funcs must have exactly "
                    "one child expression" +
                    expr->ToString());
            }

            size_t col_idx = 0;
            for (auto& child_expr : agg_expr.children) {
                if (child_expr->type !=
                    duckdb::ExpressionType::BOUND_COLUMN_REF) {
                    throw std::runtime_error(
                        "Aggregate expression must have only col ref "
                        "expression children" +
                        expr->ToString());
                }
                auto& colref =
                    child_expr->Cast<duckdb::BoundColumnRefExpression>();
                col_idx = col_ref_map.at(
                    {colref.binding.table_index, colref.binding.column_index});

                size_t reorder_col_idx =
                    std::find(this->input_col_inds.begin(),
                              this->input_col_inds.end(), col_idx) -
                    this->input_col_inds.begin();
                f_in_cols.push_back(reorder_col_idx);
            }
            f_in_offsets.push_back(f_in_cols.size());

            // Extract bind_info
            BodoAggFunctionData& bind_info =
                agg_expr.bind_info->Cast<BodoAggFunctionData>();

            std::unique_ptr<bodo::DataType> out_arr_type;
            if (is_udf) {
                throw std::runtime_error(
                    "GPU aggregate expressions don't support UDFs");
            } else {
                ftypes.push_back(function_to_ftype.at(agg_expr.function.name));
                std::tuple<bodo_array_type::arr_type_enum,
                           Bodo_CTypes::CTypeEnum>
                    output_dtype = get_groupby_output_dtype(
                        ftypes.back(),
                        in_table_schema->column_types[col_idx]->array_type,
                        in_table_schema->column_types[col_idx]->c_type);
                std::string timezone =
                    in_table_schema->column_types[col_idx]->timezone;
                out_arr_type = std::make_unique<bodo::DataType>(
                    std::get<0>(output_dtype), std::get<1>(output_dtype), -1,
                    -1, timezone);
            }

            this->output_schema->append_column(std::move(out_arr_type));
            this->output_schema->column_names.push_back(agg_expr.function.name);

            // NOTE: drop_na must be consistent accross all expressions
            // This is a little awkward but AFAICT there is no bind_info for
            // the LogicalAggregate.
            if (!dropna.has_value()) {
                dropna = bind_info.dropna;
            }
            if (dropna.value() != bind_info.dropna) {
                throw std::runtime_error(
                    "PhysicalGPUAggregate: All aggregate expressions must have "
                    "the same "
                    "value for the dropna parameter.");
            }
        }

        auto udf_table = alloc_table(this->output_schema->Project(udf_idxs));

        // TODO: propagate dropna value when agg columns are pruned out.
        if (!dropna.has_value()) {
            dropna = true;
        }
        this->groupby_state = std::make_unique<CudaGroupbyState>(
            std::make_unique<bodo::Schema>(*in_table_schema_reordered), ftypes,
            std::vector<int32_t>(), f_in_offsets, f_in_cols, this->keys.size(),
            std::vector<bool>(), std::vector<bool>(), cols_to_keep_vec, nullptr,
            get_streaming_batch_size(), true, -1, getOpId(), -1, false,
            std::nullopt,
            /*use_sql_rules*/ false, /* pandas_drop_na_*/ dropna.value());
        this->metrics.init_time += end_timer(start_init);
    }

    explicit PhysicalGPUAggregate(std::shared_ptr<bodo::Schema> in_table_schema,
                                  duckdb::LogicalDistinct& op) {
        time_pt start_init = start_timer();
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map =
            getColRefMap(op.children[0]->GetColumnBindings());

        std::shared_ptr<bodo::Schema> in_table_schema_reordered;
        std::vector<bool> cols_to_keep_vec;
        this->initKeysAndSchema(col_ref_map, op.distinct_targets,
                                in_table_schema, in_table_schema_reordered,
                                cols_to_keep_vec);

        std::vector<int32_t> ftypes;
        // Create input data column indices (only single data column for now)
        std::vector<int32_t> f_in_cols;

        // Offsets for the input data columns, which are trivial since we have a
        // single data column
        std::vector<int32_t> f_in_offsets(f_in_cols.size() + 1);
        std::iota(f_in_offsets.begin(), f_in_offsets.end(), 0);

        this->groupby_state = std::make_unique<CudaGroupbyState>(
            std::make_unique<bodo::Schema>(*in_table_schema_reordered), ftypes,
            std::vector<int32_t>(), f_in_offsets, f_in_cols, this->keys.size(),
            std::vector<bool>(), std::vector<bool>(), cols_to_keep_vec, nullptr,
            get_streaming_batch_size(), true, -1, getOpId(), -1, false,
            std::nullopt,
            /*use_sql_rules*/ false, /* pandas_drop_na_*/ false);
        this->metrics.init_time += end_timer(start_init);
    }

    virtual ~PhysicalGPUAggregate() = default;

    void FinalizeSink() override {}

    void FinalizeSource() override {
        QueryProfileCollector::Default().SubmitOperatorName(getOpId(),
                                                            ToString());
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 0),
            metrics.init_time);
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            metrics.consume_time);
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 2),
            metrics.produce_time);
    }

    /**
     * @brief process input tables to groupby build (populate the build
     * table)
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) override {
        time_pt start_consume = start_timer();
        bool request_input = true;
#if 0
        bool local_is_last = prev_op_result == OperatorResult::FINISHED;
        std::shared_ptr<table_info> input_batch_reordered =
            ProjectTable(input_batch, this->input_col_inds);
        bool global_is_last = groupby_state->build_consume_batch(
            input_batch_reordered, local_is_last,
            true, &request_input);

        se->event.record(se->stream);
        if (global_is_last) {
            return OperatorResult::FINISHED;
        }
#endif
        this->metrics.consume_time += end_timer(start_consume);
        return request_input ? OperatorResult::NEED_MORE_INPUT
                             : OperatorResult::HAVE_MORE_OUTPUT;
    }

    std::pair<GPU_DATA, OperatorResult> ProduceBatchGPU(
        std::shared_ptr<StreamAndEvent> se) override {
        time_pt start_produce = start_timer();
        bool out_is_last = false;
        std::shared_ptr<table_info> next_batch;
        next_batch = groupby_state->produce_output_batch(&out_is_last, true);
        this->metrics.produce_time += end_timer(start_produce);
        next_batch->column_names = this->output_schema->column_names;
        se->event.record(se->stream);
        return {next_batch, out_is_last ? OperatorResult::FINISHED
                                        : OperatorResult::HAVE_MORE_OUTPUT};
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
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

    std::string ToString() override { return PhysicalGPUSink::ToString(); }

    int64_t getOpId() const { return PhysicalGPUSink::getOpId(); }

   private:
    /**
     * @brief Initialize the key column indices for the groupby operation
     *        and the output schema.
     *
     * @param col_ref_map Mapping of column references to indices.
     * @param groups List of group expressions.
     */
    void initKeysAndSchema(
        const std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>&
            col_ref_map,
        const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups,
        const std::shared_ptr<bodo::Schema>& in_table_schema,
        std::shared_ptr<bodo::Schema>& in_table_schema_reordered,
        std::vector<bool>& cols_to_keep_vec) {
        for (const auto& expr : groups) {
            if (expr->type != duckdb::ExpressionType::BOUND_COLUMN_REF) {
                throw std::runtime_error(
                    "Groupby key expression is not a column reference: " +
                    expr->ToString());
            }
            auto& colref = expr->Cast<duckdb::BoundColumnRefExpression>();
            this->keys.push_back(col_ref_map.at(
                {colref.binding.table_index, colref.binding.column_index}));
        }

        uint64_t ncols = in_table_schema->ncols();
        initInputColumnMapping(this->input_col_inds, this->keys, ncols);

        in_table_schema_reordered =
            in_table_schema->Project(this->input_col_inds);

        cols_to_keep_vec = std::vector<bool>(ncols, true);

        // Add keys to output schema
        this->output_schema = std::make_shared<bodo::Schema>();
        for (size_t i = 0; i < this->keys.size(); i++) {
            this->output_schema->append_column(
                in_table_schema_reordered->column_types[i]->copy());
            if (in_table_schema_reordered->column_names.size() > 0) {
                this->output_schema->column_names.push_back(
                    in_table_schema_reordered->column_names[i]);
            } else {
                this->output_schema->column_names.push_back("key_" +
                                                            std::to_string(i));
            }
        }
        this->output_schema->metadata = std::make_shared<bodo::TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));
    }

    std::shared_ptr<CudaGroupbyState> groupby_state;
    std::shared_ptr<bodo::Schema> output_schema;
    std::vector<uint64_t> keys;
    PhysicalGPUAggregateMetrics metrics;
    // Mapping of input table column indices to move keys to the front.
    std::vector<int64_t> input_col_inds;

    // Map from function name to Bodo_FTypes
    static const std::map<std::string, int32_t> function_to_ftype;
};

// Definition of the static member
const std::map<std::string, int32_t> PhysicalGPUAggregate::function_to_ftype = {
    {"count", Bodo_FTypes::count}, {"max", Bodo_FTypes::max},
    {"mean", Bodo_FTypes::mean},   {"median", Bodo_FTypes::median},
    {"min", Bodo_FTypes::min},     {"nunique", Bodo_FTypes::nunique},
    {"size", Bodo_FTypes::size},   {"skew", Bodo_FTypes::skew},
    {"std", Bodo_FTypes::std},     {"sum", Bodo_FTypes::sum},
    {"var", Bodo_FTypes::var},     {"first", Bodo_FTypes::first},
    {"last", Bodo_FTypes::last}};

/**
 * @brief Physical node for count_star().
 *
 */
class PhysicalGPUCountStar : public PhysicalGPUSource, public PhysicalGPUSink {
   public:
    explicit PhysicalGPUCountStar() : local_count(0), global_count(0) {
        std::vector<std::unique_ptr<bodo::DataType>> types;
        types.emplace_back(std::make_unique<bodo::DataType>(
            bodo_array_type::arr_type_enum::NULLABLE_INT_BOOL,
            Bodo_CTypes::CTypeEnum::UINT64));
        std::vector<std::string> names = {std::string("count_star()")};
        out_schema = std::make_shared<bodo::Schema>(std::move(types), names);
        out_schema->metadata = std::make_shared<bodo::TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));
    }

    virtual ~PhysicalGPUCountStar() = default;

    void FinalizeSink() override {
        int result =
            MPI_Allreduce(&local_count, &global_count, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (result != MPI_SUCCESS) {
            throw std::runtime_error(
                "PhysicalGPUCountStar::Finalize MPI_Allreduce failed.");
        }
    }

    void FinalizeSource() override {}

    OperatorResult ConsumeBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) override {
        local_count += input_batch.table->num_rows();
        se->event.record(se->stream);
        return prev_op_result == OperatorResult::FINISHED
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        throw std::runtime_error(
            "GetResult called on a PhysicalGPUCountStar node.");
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return out_schema;
    }

    std::pair<GPU_DATA, OperatorResult> ProduceBatchGPU(
        std::shared_ptr<StreamAndEvent> se) override {
        cudf::numeric_scalar<uint64_t> s(global_count, true);
        auto col = cudf::make_column_from_scalar(s, 1, se->stream);
        std::vector<std::unique_ptr<cudf::column>> cols;
        cols.push_back(std::move(col));
        auto scalar_table = std::make_unique<cudf::table>(std::move(cols));
        se->event.record(se->stream);
        std::pair<GPU_DATA, OperatorResult> ret = std::make_pair(
            GPU_DATA(std::move(scalar_table), out_schema->ToArrowSchema(), se),
            OperatorResult::Finished);
        return ret;
    }

   private:
    uint64_t local_count, global_count;
    std::shared_ptr<bodo::Schema> out_schema;
};
