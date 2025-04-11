#include "_executor.h"
#include <arrow/python/api.h>
#include <arrow/python/pyarrow.h>
#include <arrow/status.h>
#include <cstdint>
#include <memory>
#include "../io/arrow_compat.h"
#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "../libs/_bodo_to_arrow.h"
#include "../libs/streaming/_shuffle.h"
#include "_plan.h"
#include "arrow/io/api.h"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "parquet/arrow/reader.h"

Executor::Executor(std::unique_ptr<duckdb::LogicalOperator> plan) {
    processNode(plan);
}

std::shared_ptr<PhysicalOperator> Executor::processNode(
    std::unique_ptr<duckdb::LogicalOperator>& plan) {
    // Convert logical plan to physical plan and create query pipelines

    // TODO: support all node types
    duckdb::LogicalOperatorType ptype = plan->type;
    switch (ptype) {
        case duckdb::LogicalOperatorType::LOGICAL_GET: {
            duckdb::LogicalGet& get_plan = plan->Cast<duckdb::LogicalGet>();

            std::shared_ptr<PhysicalOperator> physical_op =
                get_plan.bind_data->Cast<BodoScanFunctionData>()
                    .CreatePhysicalOperator();

            this->pipelines.emplace_back(
                std::vector<std::shared_ptr<PhysicalOperator>>({physical_op}));
            return physical_op;
        }
        case duckdb::LogicalOperatorType::LOGICAL_PROJECTION: {
            duckdb::LogicalProjection& proj_plan =
                plan->Cast<duckdb::LogicalProjection>();
            std::shared_ptr<PhysicalOperator> source =
                this->processNode(proj_plan.children[0]);

            std::shared_ptr<PhysicalOperator> physical_op =
                PhysicalProjection::make(proj_plan, source);

            assert(this->pipelines.size());
            this->pipelines[this->pipelines.size() - 1].operators.emplace_back(
                physical_op);
            return physical_op;
        }
        default:
            throw std::runtime_error(
                std::string(
                    "Executor doesn't handle logical operator of type ") +
                std::to_string(static_cast<int>(ptype)));
    }
}

std::shared_ptr<PhysicalOperator> PhysicalProjection::make(
    const duckdb::LogicalProjection& proj_plan,
    const std::shared_ptr<PhysicalOperator>& source) {
    // Handle UDF execution case
    if (proj_plan.expressions.size() == 1 &&
        proj_plan.expressions[0]->type ==
            duckdb::ExpressionType::BOUND_FUNCTION) {
        return std::make_shared<PhysicalRunUDF>(
            source, proj_plan.expressions[0]
                        ->Cast<duckdb::BoundFunctionExpression>()
                        .bind_info->Cast<BodoUDFFunctionData>()
                        .func);
    }

    // Process the source of this projection.
    std::vector<int64_t> selected_columns;
    // Convert BoundColumnRefExpressions in LogicalOperator.expresssions field
    // to integer selected columns.
    for (const auto& expr : proj_plan.expressions) {
        duckdb::BoundColumnRefExpression& colref =
            expr->Cast<duckdb::BoundColumnRefExpression>();
        selected_columns.push_back(colref.binding.column_index);
    }
    std::shared_ptr<PhysicalOperator> physical_op =
        std::make_shared<PhysicalProjection>(source, selected_columns);

    return physical_op;
}

std::pair<int64_t, PyObject*> Executor::execute() {
    // TODO: support multiple pipelines
    return pipelines[0].execute();
}

std::pair<int64_t, PyObject*> Pipeline::execute() {
    // TODO: support multiple operators
    std::pair<int64_t, PyObject*> last_result;
    for (std::vector<std::shared_ptr<PhysicalOperator>>::size_type i = 0;
         i < operators.size(); ++i) {
        last_result = operators[i]->execute();
        // Save result in PhysicalOperator so that downstream
        // operators with references to this PhysicalOperator
        // can use the results as input.  We can't simply
        // pass one set of pair<int64_t, PyObject*> between
        // nodes of the pipeline because some operators have
        // multiple sources.  The operators in the pipeline
        // are a post-order traversal of the operator tree.
        operators[i]->result = last_result;
    }
    return last_result;
}

std::pair<int64_t, PyObject*> PhysicalReadParquet::execute() {
    // TODO: replace with streaming Parquet read.

    auto batch = internal_reader->read_all();

    return {reinterpret_cast<int64_t>(new table_info(*batch)), pyarrow_schema};
}

std::pair<int64_t, PyObject*> PhysicalReadPandas::execute() {
    // Extract slice from pandas DataFrame
    // df.iloc[current_row:current_row+batch_size]
    // TODO: convert to streaming
    int64_t batch_size = this->num_rows;
    PyObject* iloc = PyObject_GetAttrString(df, "iloc");
    PyObject* slice =
        PySlice_New(PyLong_FromLongLong(this->current_row),
                    PyLong_FromLongLong(this->current_row + batch_size),
                    PyLong_FromLongLong(1));
    PyObject* batch = PyObject_GetItem(iloc, slice);

    // Convert pandas DataFrame to Arrow Table
    PyObject* pyarrow_module = PyImport_ImportModule("pyarrow");
    PyObject* table_func = PyObject_GetAttrString(pyarrow_module, "Table");
    PyObject* pa_table =
        PyObject_CallMethod(table_func, "from_pandas", "O", batch);

    // Unwrap Arrow table from Python object
    std::shared_ptr<arrow::Table> table =
        arrow::py::unwrap_table(pa_table).ValueOrDie();

    // Get Arrow schema for return value
    PyObject* pyarrow_schema = arrow::py::wrap_schema(table->schema());

    // Convert Arrow arrays to Bodo arrays
    auto* bodo_pool = bodo::BufferPool::DefaultPtr();
    std::shared_ptr<table_info> out_table =
        arrow_table_to_bodo(table, bodo_pool);

    // Clean up Python references
    Py_DECREF(iloc);
    Py_DECREF(slice);
    Py_DECREF(batch);
    Py_DECREF(pyarrow_module);
    Py_DECREF(table_func);
    Py_DECREF(pa_table);

    return {reinterpret_cast<int64_t>(new table_info(*out_table)),
            pyarrow_schema};
}

std::pair<int64_t, PyObject*> PhysicalProjection::execute() {
    // Get result from the source of the projection.
    std::pair<int64_t, PyObject*> src_result = src->result;
    // Get the table_info out of that result.
    table_info* src_table_info =
        reinterpret_cast<table_info*>(src_result.first);
    // Get and unwrap the arrow Table schema from the result.
    std::shared_ptr<arrow::Schema> src_schema =
        arrow::py::unwrap_schema(src_result.second).ValueOrDie();

    // Select columns from the actual data in Bodo table_info format.
    std::shared_ptr<table_info> out_table_info = ProjectTable(
        std::shared_ptr<table_info>(src_table_info), selected_columns);

    // Select those columns in arrow for schema representation.
    std::vector<std::shared_ptr<arrow::Field>> selected_fields;
    int num_fields = src_schema->num_fields();
    for (int i : selected_columns) {
        if (i >= num_fields) {
            throw std::runtime_error(std::string("Error selecting columns ") +
                                     std::to_string(i) + " " +
                                     std::to_string(num_fields));
        }
        selected_fields.push_back(src_schema->field(i));
    }
    auto out_schema = arrow::schema(selected_fields);

    PyObject* pyarrow_schema = arrow::py::wrap_schema(out_schema);
    return {reinterpret_cast<int64_t>(new table_info(*out_table_info)),
            pyarrow_schema};
}

std::pair<int64_t, PyObject*> PhysicalRunUDF::execute() {
    // Call bodo.pandas.utils.run_apply_udf() to run the UDF

    // Get input data table from the input table of the projection.
    std::pair<int64_t, PyObject*> src_result = source->result;

    // Import the bodo.pandas.utils module
    PyObject* bodo_module = PyImport_ImportModule("bodo.pandas.utils");
    if (!bodo_module) {
        PyErr_Print();
        throw std::runtime_error("Failed to import bodo.pandas.utils module");
    }

    // Call the run_apply_udf() with the table_info pointer, Arrow schema and
    // UDF function
    PyObject* result =
        PyObject_CallMethod(bodo_module, "run_apply_udf", "LOO",
                            src_result.first, src_result.second, this->func);
    if (!result) {
        PyErr_Print();
        Py_DECREF(bodo_module);
        throw std::runtime_error("Error calling run_apply_udf");
    }

    // Parse the result (assuming it returns a tuple with (table_info,
    // arrow_schema))
    if (!PyTuple_Check(result) || PyTuple_Size(result) != 2) {
        Py_DECREF(result);
        Py_DECREF(bodo_module);
        throw std::runtime_error(
            "Expected a tuple of size 2 from run_apply_udf");
    }

    // Extract the table_info pointer and arrow schema from the result
    PyObject* table_info_py = PyTuple_GetItem(result, 0);
    PyObject* arrow_schema_py = PyTuple_GetItem(result, 1);

    int64_t table_info_ptr = PyLong_AsLongLong(table_info_py);
    // Increment arrow_schema_py reference count since PyTuple_GetItem returns a
    // borrowed reference
    Py_INCREF(arrow_schema_py);

    Py_DECREF(bodo_module);
    Py_DECREF(result);

    return {table_info_ptr, arrow_schema_py};
}
