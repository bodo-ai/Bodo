#pragma once

#include <memory>
#include <stdexcept>
#include <utility>
#include "../_util.h"
#include "../io/arrow_reader.h"
#include "../libs/_array_utils.h"
#include "../libs/_utils.h"
#include "../libs/groupby/_groupby_ftypes.h"
#include "../libs/streaming/_groupby.h"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "expression.h"
#include "operator.h"

/**
 * @brief Physical node for groupby aggregation
 *
 */
class PhysicalAggregate : public PhysicalSource, public PhysicalSink {
   public:
    explicit PhysicalAggregate(std::shared_ptr<bodo::Schema> in_table_schema,
                               duckdb::LogicalAggregate& op) {
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map =
            getColRefMap(op.children[0]->GetColumnBindings());

        this->initKeys(col_ref_map, op.groups);

        uint64_t ncols = in_table_schema->ncols();
        initInputColumnMapping(this->input_col_inds, this->keys, ncols);

        std::shared_ptr<bodo::Schema> in_table_schema_reordered =
            in_table_schema->Project(this->input_col_inds);

        std::vector<bool> cols_to_keep_vec(ncols, true);

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
        this->output_schema->metadata = std::make_shared<TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));

        std::vector<int32_t> ftypes;
        // Create input data column indices (only single data column for now)
        std::vector<int32_t> f_in_cols;
        std::optional<bool> dropna = std::nullopt;
        for (const auto& expr : op.expressions) {
            if (expr->type != duckdb::ExpressionType::BOUND_AGGREGATE) {
                throw std::runtime_error(
                    "Aggregate expression is not a bound aggregate: " +
                    expr->ToString());
            }
            auto& agg_expr = expr->Cast<duckdb::BoundAggregateExpression>();
            if (agg_expr.children.size() != 1 ||
                agg_expr.children[0]->type !=
                    duckdb::ExpressionType::BOUND_COLUMN_REF) {
                throw std::runtime_error(
                    "Aggregate expression does not have a single column "
                    "reference child: " +
                    expr->ToString());
            }
            auto& colref =
                agg_expr.children[0]->Cast<duckdb::BoundColumnRefExpression>();
            size_t col_idx = col_ref_map[{colref.binding.table_index,
                                          colref.binding.column_index}];

            size_t reorder_col_idx =
                std::find(this->input_col_inds.begin(),
                          this->input_col_inds.end(), col_idx) -
                this->input_col_inds.begin();
            f_in_cols.push_back(reorder_col_idx);

            // Check if the aggregate function is supported
            if (function_to_ftype.find(agg_expr.function.name) ==
                function_to_ftype.end()) {
                throw std::runtime_error("Unsupported aggregate function: " +
                                         agg_expr.function.name);
            }

            ftypes.push_back(function_to_ftype.at(agg_expr.function.name));

            std::tuple<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>
                out_arr_type = get_groupby_output_dtype(
                    ftypes.back(),
                    in_table_schema->column_types[col_idx]->array_type,
                    in_table_schema->column_types[col_idx]->c_type);

            this->output_schema->append_column(std::make_unique<bodo::DataType>(
                std::get<0>(out_arr_type), std::get<1>(out_arr_type)));
            this->output_schema->column_names.push_back(agg_expr.function.name);

            // Extract bind_info
            BodoAggFunctionData& bind_info =
                agg_expr.bind_info->Cast<BodoAggFunctionData>();

            // NOTE: drop_na must be consistent accross all expressions
            // This is a little awkward but AFAICT there is no bind_info for
            // the LogicalAggregate.
            if (!dropna.has_value()) {
                dropna = bind_info.dropna;
            }
            if (dropna.value() != bind_info.dropna) {
                throw std::runtime_error(
                    "PhysicalAggregate: All aggregate expressions must have "
                    "the same "
                    "value for the dropna parameter.");
            }
        }

        // Offsets for the input data columns, which are trivial since we have a
        // single data column
        std::vector<int32_t> f_in_offsets(f_in_cols.size() + 1);
        std::iota(f_in_offsets.begin(), f_in_offsets.end(), 0);

        // TODO: propagate dropna value when agg columns are pruned out.
        if (!dropna.has_value()) {
            dropna = true;
        }
        this->groupby_state = std::make_unique<GroupbyState>(
            std::make_unique<bodo::Schema>(*in_table_schema_reordered), ftypes,
            std::vector<int32_t>(), f_in_offsets, f_in_cols, this->keys.size(),
            std::vector<bool>(), std::vector<bool>(), cols_to_keep_vec, nullptr,
            get_streaming_batch_size(), true, -1, -1, -1, false, std::nullopt,
            /*use_sql_rules*/ false, /* pandas_drop_na_*/ dropna.value());
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
        std::shared_ptr<table_info> input_batch_reordered =
            ProjectTable(input_batch, this->input_col_inds);
        bool global_is_last = groupby_build_consume_batch(
            this->groupby_state.get(), input_batch_reordered, local_is_last,
            true, &request_input);

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

   private:
    /**
     * @brief Initialize the key column indices for the groupby operation.
     *
     * @param col_ref_map Mapping of column references to indices.
     * @param groups List of group expressions.
     */
    void initKeys(
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map,
        const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups) {
        for (const auto& expr : groups) {
            if (expr->type != duckdb::ExpressionType::BOUND_COLUMN_REF) {
                throw std::runtime_error(
                    "Groupby key expression is not a column reference: " +
                    expr->ToString());
            }
            auto& colref = expr->Cast<duckdb::BoundColumnRefExpression>();
            this->keys.push_back(col_ref_map[{colref.binding.table_index,
                                              colref.binding.column_index}]);
        }
    }

    std::shared_ptr<GroupbyState> groupby_state;
    std::shared_ptr<bodo::Schema> output_schema;
    std::vector<uint64_t> keys;
    // Mapping of input table column indices to move keys to the front.
    std::vector<int64_t> input_col_inds;

    // Map from function name to Bodo_FTypes
    static const std::map<std::string, int32_t> function_to_ftype;
};

// Definition of the static member
const std::map<std::string, int32_t> PhysicalAggregate::function_to_ftype = {
    {"count", Bodo_FTypes::count}, {"max", Bodo_FTypes::max},
    {"mean", Bodo_FTypes::mean},   {"median", Bodo_FTypes::median},
    {"min", Bodo_FTypes::min},     {"nunique", Bodo_FTypes::nunique},
    {"size", Bodo_FTypes::size},   {"skew", Bodo_FTypes::skew},
    {"std", Bodo_FTypes::std},     {"sum", Bodo_FTypes::sum},
    {"var", Bodo_FTypes::var}};

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

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        throw std::runtime_error(
            "GetResult called on a PhysicalCountStar node.");
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return out_schema;
    }

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        std::shared_ptr<arrow::Array> array = ScalarToArrowArray(global_count);

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
