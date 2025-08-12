#pragma once

#include <Python.h>
#include <object.h>
#include <pytypedefs.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include "../_util.h"
#include "../io/arrow_reader.h"
#include "../libs/_array_utils.h"
#include "../libs/_query_profile_collector.h"
#include "../libs/_table_builder_utils.h"
#include "../libs/_utils.h"
#include "../libs/groupby/_groupby_ftypes.h"
#include "../libs/groupby/_groupby_udf.h"
#include "../libs/streaming/_groupby.h"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "expression.h"
#include "operator.h"

struct PhysicalAggregateMetrics {
    using time_t = MetricBase::TimerValue;

    time_t init_time = 0;     // stage_0
    time_t consume_time = 0;  // stage_1
    time_t produce_time = 0;  // stage_2
};

/**
 * @brief Gets a cfunc for computing the output of all UDFs.
 *
 * @param cfunc_wrapper Callable python object that takes in an int and a tuple
 * of function objects and returns the address of the cfunc.
 * @param udf_idxs The indices of the udf output columns in the output table.
 * @param funcs A list of GroupbyAggFunc objects.
 * @return udf_general_fn cfunc for applying UDFs on a grouped table.
 */
udf_general_fn get_cfunc_from_wrapper(PyObject* cfunc_wrapper,
                                      std::vector<int> udf_idxs,
                                      std::vector<PyObject*>& funcs) {
    const Py_ssize_t n = static_cast<Py_ssize_t>(funcs.size());

    PyObject* funcs_tuple = PyTuple_New(n);
    PyObject* offsets_tuple = PyTuple_New(n);

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* func_obj = funcs[static_cast<size_t>(i)];
        Py_INCREF(func_obj);
        PyTuple_SET_ITEM(funcs_tuple, i, func_obj);

        PyObject* py_offset =
            PyLong_FromLongLong(udf_idxs[static_cast<size_t>(i)]);
        PyTuple_SET_ITEM(offsets_tuple, i, py_offset);
    }

    PyObject* args = PyTuple_New(2);  // new ref

    PyTuple_SET_ITEM(args, 0, offsets_tuple);
    PyTuple_SET_ITEM(args, 1, funcs_tuple);

    // Call: cfunc_wrapper(offsets_tuple, funcs_tuple)
    PyObject* result = PyObject_Call(cfunc_wrapper, args, nullptr);
    Py_DECREF(args);

    if (!result) {
        PyErr_Print();
        throw std::runtime_error("agg: Error calling cfunc wrapper.");
    }

    if (!PyLong_Check(result)) {
        throw std::runtime_error(
            "agg: Expected cfunc wrapper to return an integer.");
    }

    return reinterpret_cast<udf_general_fn>(PyLong_AsLongLong(result));
}

/**
 * @brief Physical node for groupby aggregation
 *
 */
class PhysicalAggregate : public PhysicalSource, public PhysicalSink {
   public:
    explicit PhysicalAggregate(std::shared_ptr<bodo::Schema> in_table_schema,
                               duckdb::LogicalAggregate& op) {
        time_pt start_init = start_timer();
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map =
            getColRefMap(op.children[0]->GetColumnBindings());

        std::shared_ptr<bodo::Schema> in_table_schema_reordered;
        std::vector<bool> cols_to_keep_vec;
        this->initKeysAndSchema(col_ref_map, op.groups, in_table_schema,
                                in_table_schema_reordered, cols_to_keep_vec);

        std::vector<int32_t> ftypes;
        // Create input data column indices (only single data column for now)
        std::vector<int32_t> f_in_cols;
        std::optional<bool> dropna = std::nullopt;

        // The cfunc to call on accumulated data to compute UDF
        PyObject* cfunc_wrapper = nullptr;
        std::vector<PyObject*> udfs;
        std::vector<int> udf_idxs;

        for (size_t i = 0; i < op.expressions.size(); i++) {
            const auto& expr = op.expressions[i];

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
            size_t col_idx = col_ref_map.at(
                {colref.binding.table_index, colref.binding.column_index});

            size_t reorder_col_idx =
                std::find(this->input_col_inds.begin(),
                          this->input_col_inds.end(), col_idx) -
                this->input_col_inds.begin();
            f_in_cols.push_back(reorder_col_idx);

            // Check if the aggregate function is supported
            if (function_to_ftype.find(agg_expr.function.name) ==
                    function_to_ftype.end() &&
                !agg_expr.function.name.starts_with("udf")) {
                throw std::runtime_error("Unsupported aggregate function: " +
                                         agg_expr.function.name);
            }

            if (agg_expr.function.name.starts_with("udf")) {
                ftypes.push_back(Bodo_FTypes::gen_udf);
            } else {
                ftypes.push_back(function_to_ftype.at(agg_expr.function.name));
            }

            // Extract bind_info
            BodoAggFunctionData& bind_info =
                agg_expr.bind_info->Cast<BodoAggFunctionData>();

            // Extract out type
            auto out_arr_type = arrow_type_to_bodo_data_type(
                bind_info.out_schema->field(0)->type());

            // extract Cfunc and create a udf struct for storing callback and
            // output types
            if (ftypes.back() == Bodo_FTypes::gen_udf) {
                // callback_wrapper is a tuple of (callback_wrapper, func_arg)
                // the callback_wrapper is the same for every func, so only need
                // to extract it once.
                if (!cfunc_wrapper) {
                    cfunc_wrapper =
                        PyTuple_GET_ITEM(bind_info.callback_wrapper, 0);
                }
                udfs.push_back(PyTuple_GET_ITEM(bind_info.callback_wrapper, 1));
                udf_idxs.push_back(i + this->keys.size());
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
                    "PhysicalAggregate: All aggregate expressions must have "
                    "the same "
                    "value for the dropna parameter.");
            }
        }

        // Offsets for the input data columns, which are trivial since we have a
        // single data column
        std::vector<int32_t> f_in_offsets(f_in_cols.size() + 1);
        std::iota(f_in_offsets.begin(), f_in_offsets.end(), 0);

        // Create the udf info struct for GroupbyState
        std::optional<udfinfo_t> udf_info = std::nullopt;
        if (udfs.size()) {
            udf_general_fn agg_cfunc =
                get_cfunc_from_wrapper(cfunc_wrapper, udf_idxs, udfs);

            auto udf_table =
                alloc_table(this->output_schema->Project(udf_idxs));
            udf_info = {.udf_table_dummy = udf_table,
                        .update = nullptr,
                        .combine = nullptr,
                        .eval = nullptr,
                        .general_udf = agg_cfunc};
        }

        // TODO: propagate dropna value when agg columns are pruned out.
        if (!dropna.has_value()) {
            dropna = true;
        }
        this->groupby_state = std::make_unique<GroupbyState>(
            std::make_unique<bodo::Schema>(*in_table_schema_reordered), ftypes,
            std::vector<int32_t>(), f_in_offsets, f_in_cols, this->keys.size(),
            std::vector<bool>(), std::vector<bool>(), cols_to_keep_vec, nullptr,
            get_streaming_batch_size(), true, -1, getOpId(), -1, false,
            std::nullopt,
            /*use_sql_rules*/ false, /* pandas_drop_na_*/ dropna.value(),
            udf_info);
        this->metrics.init_time += end_timer(start_init);
    }

    explicit PhysicalAggregate(std::shared_ptr<bodo::Schema> in_table_schema,
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

        this->groupby_state = std::make_unique<GroupbyState>(
            std::make_unique<bodo::Schema>(*in_table_schema_reordered), ftypes,
            std::vector<int32_t>(), f_in_offsets, f_in_cols, this->keys.size(),
            std::vector<bool>(), std::vector<bool>(), cols_to_keep_vec, nullptr,
            get_streaming_batch_size(), true, -1, getOpId(), -1, false,
            std::nullopt,
            /*use_sql_rules*/ false, /* pandas_drop_na_*/ false);
        this->metrics.init_time += end_timer(start_init);
    }

    virtual ~PhysicalAggregate() = default;

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
    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        time_pt start_consume = start_timer();
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
        this->metrics.consume_time += end_timer(start_consume);
        return request_input ? OperatorResult::NEED_MORE_INPUT
                             : OperatorResult::HAVE_MORE_OUTPUT;
    }

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        time_pt start_produce = start_timer();
        bool out_is_last = false;
        std::shared_ptr<table_info> next_batch =
            groupby_produce_output_batch_wrapper(this->groupby_state.get(),
                                                 &out_is_last, true);
        this->metrics.produce_time += end_timer(start_produce);
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

    std::string ToString() override { return PhysicalSink::ToString(); }

    int64_t getOpId() const { return PhysicalSink::getOpId(); }

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
        this->output_schema->metadata = std::make_shared<TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));
    }

    std::shared_ptr<GroupbyState> groupby_state;
    std::shared_ptr<bodo::Schema> output_schema;
    std::vector<uint64_t> keys;
    PhysicalAggregateMetrics metrics;
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

    void FinalizeSink() override {
        int result =
            MPI_Allreduce(&local_count, &global_count, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (result != MPI_SUCCESS) {
            throw std::runtime_error(
                "PhysicalCountStar::Finalize MPI_Allreduce failed.");
        }
    }

    void FinalizeSource() override {}

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
