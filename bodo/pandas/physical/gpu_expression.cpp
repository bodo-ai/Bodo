#include "gpu_expression.h"
#include "_util.h"

std::variant<GPU_COLUMN, GPU_SCALAR> do_cudf_compute_binary(
    std::shared_ptr<ExprGPUResult> left_res,
    std::shared_ptr<ExprGPUResult> right_res,
    const cudf::binary_operator& comparator,
    const std::shared_ptr<cudf::data_type> result_type) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprGPUResult> left_as_array =
        std::dynamic_pointer_cast<ArrayExprGPUResult>(left_res);
    std::shared_ptr<ScalarExprGPUResult> left_as_scalar =
        std::dynamic_pointer_cast<ScalarExprGPUResult>(left_res);
    std::shared_ptr<ArrayExprGPUResult> right_as_array =
        std::dynamic_pointer_cast<ArrayExprGPUResult>(right_res);
    std::shared_ptr<ScalarExprGPUResult> right_as_scalar =
        std::dynamic_pointer_cast<ScalarExprGPUResult>(right_res);

    cudf::data_type cudf_result_type;
    if (result_type) {
        cudf_result_type = *result_type;
    }

    GPU_COLUMN res;
    if (left_as_array) {
        if (!result_type) {
            cudf_result_type = left_as_array->result->view().type();
        }
        if (right_as_array) {
            res = cudf::binary_operation(left_as_array->result->view(),
                                         right_as_array->result->view(),
                                         comparator, cudf_result_type);
        } else if (right_as_scalar) {
            res = cudf::binary_operation(left_as_array->result->view(),
                                         *(right_as_scalar->result), comparator,
                                         cudf_result_type);
        } else {
            throw std::runtime_error(
                "do_cudf_compute right is neither array nor scalar.");
        }
    } else if (left_as_scalar) {
        if (right_as_array) {
            if (!result_type) {
                cudf_result_type = right_as_array->result->view().type();
            }
            res = cudf::binary_operation(*(left_as_scalar->result),
                                         right_as_array->result->view(),
                                         comparator, cudf_result_type);
        } else if (right_as_scalar) {
            throw std::runtime_error(
                "do_cudf_compute both left and right are scalar.");
        } else {
            throw std::runtime_error(
                "do_cudf_compute right is neither array nor scalar.");
        }
    }
    {
        throw std::runtime_error(
            "do_cudf_compute left is neither array nor scalar.");
    }

    return std::move(res);
}

std::variant<GPU_COLUMN, GPU_SCALAR> do_cudf_compute_unary(
    std::shared_ptr<ExprGPUResult> left_res,
    const cudf::unary_operator& comparator,
    const arrow::compute::FunctionOptions* func_options) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprGPUResult> left_as_array =
        std::dynamic_pointer_cast<ArrayExprGPUResult>(left_res);
    std::shared_ptr<ScalarExprGPUResult> left_as_scalar =
        std::dynamic_pointer_cast<ScalarExprGPUResult>(left_res);

    if (left_as_array) {
        return cudf::unary_operation(left_as_array->result->view(), comparator);
    } else if (left_as_scalar) {
        throw std::runtime_error(
            "do_cudf_compute_unary for scalar not yet implemented.");
    } else {
        throw std::runtime_error(
            "do_cudf_compute_unary left is neither array nor scalar.");
    }
}

std::variant<GPU_COLUMN, GPU_SCALAR> do_cudf_compute_cast(
    std::shared_ptr<ExprGPUResult> left_res,
    const duckdb::LogicalType& return_type) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprGPUResult> left_as_array =
        std::dynamic_pointer_cast<ArrayExprGPUResult>(left_res);
    std::shared_ptr<ScalarExprGPUResult> left_as_scalar =
        std::dynamic_pointer_cast<ScalarExprGPUResult>(left_res);

    cudf::data_type cudf_result_type = duckdb_logicaltype_to_cudf(return_type);

    if (left_as_array) {
        return cudf::cast(left_as_array->result->view(), cudf_result_type);
    } else if (left_as_scalar) {
        throw std::runtime_error(
            "do_cudf_compute_cast cast of scalar not yet implemented.");
    } else {
        throw std::runtime_error(
            "do_cudf_compute_cast left is neither array nor scalar.");
    }
}

#if 0
arrow::Datum do_arrow_compute_binary(arrow::Datum left_res,
                                     arrow::Datum right_res,
                                     const std::string& comparator) {
    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::CallFunction(comparator, {left_res, right_res});
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_array_compute_binary: Error in Arrow compute: " +
            cmp_res.status().message());
    }

    return cmp_res.ValueOrDie();
}

arrow::Datum do_arrow_compute_unary(arrow::Datum left_res,
                                    const std::string& comparator) {
    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::CallFunction(comparator, {left_res});
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_array_compute_unary: Error in Arrow compute: " +
            cmp_res.status().message());
    }

    return cmp_res.ValueOrDie();
}

arrow::Datum do_arrow_compute_cast(arrow::Datum left_res,
                                   const duckdb::LogicalType& return_type) {
    std::shared_ptr<arrow::DataType> arrow_ret_type =
        duckdbTypeToArrow(return_type);
    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::Cast(left_res, arrow_ret_type);
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_array_compute_cast: Error in Arrow compute: " +
            cmp_res.status().message());
    }

    return cmp_res.ValueOrDie();
}
#endif

GPU_COLUMN do_cudf_compute_case(std::shared_ptr<ExprGPUResult> when_res,
                                std::shared_ptr<ExprGPUResult> then_res,
                                std::shared_ptr<ExprGPUResult> else_res) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprGPUResult> when_as_array =
        std::dynamic_pointer_cast<ArrayExprGPUResult>(when_res);
    std::shared_ptr<ScalarExprGPUResult> when_as_scalar =
        std::dynamic_pointer_cast<ScalarExprGPUResult>(when_res);
    std::shared_ptr<ArrayExprGPUResult> then_as_array =
        std::dynamic_pointer_cast<ArrayExprGPUResult>(then_res);
    std::shared_ptr<ScalarExprGPUResult> then_as_scalar =
        std::dynamic_pointer_cast<ScalarExprGPUResult>(then_res);
    std::shared_ptr<ArrayExprGPUResult> else_as_array =
        std::dynamic_pointer_cast<ArrayExprGPUResult>(else_res);
    std::shared_ptr<ScalarExprGPUResult> else_as_scalar =
        std::dynamic_pointer_cast<ScalarExprGPUResult>(else_res);

    cudf::column_view when_col;
    if (when_as_array) {
        when_col = when_as_array->result->view();
    } else if (when_as_scalar) {
        throw std::runtime_error("do_cudf_compute_case when is a scalar.");
    } else {
        throw std::runtime_error(
            "do_cudf_arrow_case when is neither array nor scalar.");
    }

    GPU_COLUMN res;
    if (then_as_array) {
        if (else_as_array) {
            res = std::move(cudf::copy_if_else(then_as_array->result->view(),
                                               else_as_array->result->view(),
                                               when_col));
        } else if (else_as_scalar) {
            res = std::move(cudf::copy_if_else(then_as_array->result->view(),
                                               *(else_as_scalar->result),
                                               when_col));
        } else {
            throw std::runtime_error(
                "do_cudf_compute_case right is neither array nor scalar.");
        }
    } else if (then_as_scalar) {
        if (else_as_array) {
            res = std::move(cudf::copy_if_else(*(then_as_scalar->result),
                                               else_as_array->result->view(),
                                               when_col));
        } else if (else_as_scalar) {
            throw std::runtime_error(
                "do_cudf_compute_case both then and else are scalar.");
        } else {
            throw std::runtime_error(
                "do_cudf_compute_case else is neither array nor scalar.");
        }
    }
    {
        throw std::runtime_error(
            "do_cudf_compute_case then is neither array nor scalar.");
    }

    return res;
}

std::shared_ptr<PhysicalGPUExpression> buildPhysicalGPUExprTree(
    duckdb::unique_ptr<duckdb::Expression>& expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>& col_ref_map,
    bool no_scalars);

std::shared_ptr<PhysicalGPUExpression> buildPhysicalGPUExprTree(
    duckdb::Expression& expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>& col_ref_map,
    bool no_scalars) {
    // Class and type here are really like the general type of the
    // expression node (expr_class) and a sub-type of that general
    // type (expr_type).
    duckdb::ExpressionClass expr_class = expr.GetExpressionClass();
    duckdb::ExpressionType expr_type = expr.GetExpressionType();

    switch (expr_class) {
        case duckdb::ExpressionClass::BOUND_COMPARISON: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundComparisonExpression>();
            // This node type has left and right children which are recursively
            // processed first and then the resulting Bodo Physical expression
            // subtrees are combined with the expression sub-type (e.g., equal,
            // greater_than, less_than) to make the Bodo PhysicalComparisonExpr.
            return std::static_pointer_cast<PhysicalGPUExpression>(
                std::make_shared<PhysicalGPUComparisonExpression>(
                    buildPhysicalGPUExprTree(bce.left, col_ref_map, no_scalars),
                    buildPhysicalGPUExprTree(bce.right, col_ref_map,
                                             no_scalars),
                    expr_type));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_COLUMN_REF: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundColumnRefExpression>();
            duckdb::ColumnBinding binding = bce.binding;
            size_t col_idx =
                col_ref_map[{binding.table_index, binding.column_index}];
            return std::static_pointer_cast<PhysicalGPUExpression>(
                std::make_shared<PhysicalGPUColumnRefExpression>(
                    col_idx, bce.GetName()));
            // binding.table_index, binding.column_index));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_CONSTANT: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundConstantExpression>();
            if (bce.value.IsNull()) {
                // Get the constant out of the duckdb node as a C++ variant.
                // Using auto since variant set will be extended.
                auto extracted_value =
                    getDefaultValueForDuckdbValueType(bce.value);
                // Return a PhysicalConstantExpression<T> where T is the actual
                // type of the value contained within bce.value.
                auto ret = std::visit(
                    [no_scalars](const auto& value) {
                        return std::static_pointer_cast<PhysicalGPUExpression>(
                            std::make_shared<PhysicalGPUNullExpression<
                                std::decay_t<decltype(value)>>>(value,
                                                                no_scalars));
                    },
                    extracted_value);
                return ret;
            } else {
                // Get the constant out of the duckdb node as a C++ variant.
                // Using auto since variant set will be extended.
                auto extracted_value = extractValue(bce.value);
                // Return a PhysicalConstantExpression<T> where T is the actual
                // type of the value contained within bce.value.
                auto ret = std::visit(
                    [no_scalars](const auto& value) {
                        return std::static_pointer_cast<PhysicalGPUExpression>(
                            std::make_shared<PhysicalGPUConstantExpression<
                                std::decay_t<decltype(value)>>>(value,
                                                                no_scalars));
                    },
                    extracted_value);
                return ret;
            }
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_CONJUNCTION: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundConjunctionExpression>();
            // This node type has left and right children which are recursively
            // processed first and then the resulting Bodo Physical expression
            // subtrees are combined with the expression sub-type (e.g., equal,
            // greater_than, less_than) to make the Bodo PhysicalComparisonExpr.
            return std::static_pointer_cast<PhysicalGPUExpression>(
                std::make_shared<PhysicalGPUConjunctionExpression>(
                    buildPhysicalGPUExprTree(bce.children[0], col_ref_map,
                                             no_scalars),
                    buildPhysicalGPUExprTree(bce.children[1], col_ref_map,
                                             no_scalars),
                    expr_type));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_OPERATOR: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& boe = expr.Cast<duckdb::BoundOperatorExpression>();
            switch (boe.children.size()) {
                case 1: {
                    return std::static_pointer_cast<PhysicalGPUExpression>(
                        std::make_shared<PhysicalGPUUnaryExpression>(
                            buildPhysicalGPUExprTree(boe.children[0],
                                                     col_ref_map, no_scalars),
                            expr_type));
                } break;
                case 2: {
                    return std::static_pointer_cast<PhysicalGPUExpression>(
                        std::make_shared<PhysicalGPUBinaryExpression>(
                            buildPhysicalGPUExprTree(boe.children[0],
                                                     col_ref_map, no_scalars),
                            buildPhysicalGPUExprTree(boe.children[1],
                                                     col_ref_map, no_scalars),
                            expr_type));
                } break;
                default:
                    throw std::runtime_error(
                        "Unsupported number of children for bound operator");
            }
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_FUNCTION: {
            // Convert the base duckdb::Expression node to its actual derived
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bfe = expr.Cast<duckdb::BoundFunctionExpression>();
            std::shared_ptr<arrow::DataType> result_type = nullptr;

            if (bfe.bind_info) {
                BodoScalarFunctionData& scalar_func_data =
                    bfe.bind_info->Cast<BodoScalarFunctionData>();
                result_type = scalar_func_data.out_schema->field(0)->type();
            }

            if (bfe.bind_info &&
                (bfe.bind_info->Cast<BodoScalarFunctionData>().args ||
                 !bfe.bind_info->Cast<BodoScalarFunctionData>()
                      .arrow_func_name.empty())) {
                BodoScalarFunctionData& scalar_func_data =
                    bfe.bind_info->Cast<BodoScalarFunctionData>();

                std::vector<std::shared_ptr<PhysicalGPUExpression>>
                    phys_children;
                for (auto& child_expr : bfe.children) {
                    phys_children.emplace_back(buildPhysicalGPUExprTree(
                        child_expr, col_ref_map, no_scalars));
                }

                if (!scalar_func_data.arrow_func_name.empty()) {
                    throw std::runtime_error("Unimplemented");
                    /*
                    return std::static_pointer_cast<PhysicalGPUExpression>(
                        std::make_shared<PhysicalGPUArrowExpression>(
                            phys_children, scalar_func_data, result_type));
                    */
                } else if (scalar_func_data.args) {
                    return std::static_pointer_cast<PhysicalGPUExpression>(
                        std::make_shared<PhysicalGPUUDFExpression>(
                            phys_children, scalar_func_data, result_type));
                }
            } else {
                switch (bfe.children.size()) {
                    case 1: {
                        return std::static_pointer_cast<PhysicalGPUExpression>(
                            std::make_shared<PhysicalGPUUnaryExpression>(
                                buildPhysicalGPUExprTree(
                                    bfe.children[0], col_ref_map, no_scalars),
                                bfe.function.name));
                    } break;
                    case 2: {
                        return std::static_pointer_cast<PhysicalGPUExpression>(
                            std::make_shared<PhysicalGPUBinaryExpression>(
                                buildPhysicalGPUExprTree(
                                    bfe.children[0], col_ref_map, no_scalars),
                                buildPhysicalGPUExprTree(
                                    bfe.children[1], col_ref_map, no_scalars),
                                bfe.function.name, result_type));
                    } break;
                    default:
                        throw std::runtime_error(
                            "Unsupported number of children " +
                            std::to_string(bfe.children.size()) +
                            " for bound function");
                }
            }
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_CAST: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundCastExpression>();
            return std::static_pointer_cast<PhysicalGPUExpression>(
                std::make_shared<PhysicalGPUCastExpression>(
                    buildPhysicalGPUExprTree(bce.child, col_ref_map,
                                             no_scalars),
                    bce.return_type));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_BETWEEN: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bbe = expr.Cast<duckdb::BoundBetweenExpression>();
            // Convert to conjunction and comparison nodes.
            std::shared_ptr<PhysicalGPUExpression> input_expr =
                buildPhysicalGPUExprTree(bbe.input, col_ref_map, no_scalars);
            std::shared_ptr<PhysicalGPUExpression> lower_expr =
                buildPhysicalGPUExprTree(bbe.lower, col_ref_map, no_scalars);
            std::shared_ptr<PhysicalGPUExpression> upper_expr =
                buildPhysicalGPUExprTree(bbe.upper, col_ref_map, no_scalars);

            std::shared_ptr<PhysicalGPUExpression> left =
                std::static_pointer_cast<PhysicalGPUExpression>(
                    std::make_shared<PhysicalGPUComparisonExpression>(
                        input_expr, lower_expr,
                        bbe.lower_inclusive
                            ? duckdb::ExpressionType::
                                  COMPARE_GREATERTHANOREQUALTO
                            : duckdb::ExpressionType::COMPARE_GREATERTHAN));

            std::shared_ptr<PhysicalGPUExpression> right =
                std::static_pointer_cast<PhysicalGPUExpression>(
                    std::make_shared<PhysicalGPUComparisonExpression>(
                        upper_expr, input_expr,
                        bbe.upper_inclusive
                            ? duckdb::ExpressionType::
                                  COMPARE_GREATERTHANOREQUALTO
                            : duckdb::ExpressionType::COMPARE_GREATERTHAN));

            return std::static_pointer_cast<PhysicalGPUExpression>(
                std::make_shared<PhysicalGPUConjunctionExpression>(
                    left, right, duckdb::ExpressionType::CONJUNCTION_AND));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_CASE: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundCaseExpression>();
            if (bce.case_checks.size() != 1) {
                throw std::runtime_error(
                    "Only single WHEN case expressions are supported.");
            }
            auto& caseCheck = bce.case_checks[0];
            return std::static_pointer_cast<PhysicalGPUExpression>(
                std::make_shared<PhysicalGPUCaseExpression>(
                    buildPhysicalGPUExprTree(caseCheck.when_expr, col_ref_map,
                                             no_scalars),
                    buildPhysicalGPUExprTree(caseCheck.then_expr, col_ref_map,
                                             no_scalars),
                    buildPhysicalGPUExprTree(bce.else_expr, col_ref_map,
                                             no_scalars)));
        } break;  // suppress wrong fallthrough error
        default:
            throw std::runtime_error(
                "Unsupported duckdb expression class " +
                std::to_string(static_cast<int>(expr_class)));
    }
    throw std::logic_error("Control should never reach here");
}

std::shared_ptr<PhysicalGPUExpression> buildPhysicalGPUExprTree(
    duckdb::unique_ptr<duckdb::Expression>& expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>& col_ref_map,
    bool no_scalars) {
    return buildPhysicalGPUExprTree(*expr, col_ref_map, no_scalars);
}

std::shared_ptr<ExprGPUResult> PhysicalGPUUDFExpression::ProcessBatch(
    GPU_DATA input_batch) {
    throw std::runtime_error(
        "PhysicalGPUUDFExpression::ProcessBatch unimplemented ");
#if 0
This is some example cudf code we may want to adapt eventually.
#include <cudf/column/column_view.hpp>
#include <cudf/transform.hpp>  // check your libcudf include path
#include <cudf/types.hpp>
#include <memory>
#include <string>

// input_col is a cudf::column_view you already have
std::string cuda_udf = R"(
extern "C" __device__ int my_udf(int x) {
  return x * 2; // example: double each element
}
)";

cudf::data_type out_type{cudf::type_id::INT32};

// single-column overload (some versions accept column_view directly)
auto out_col = cudf::transform(input_col, cuda_udf, out_type, /*is_ptx=*/false);
// out_col is std::unique_ptr<cudf::column>
End of cudf example code.

    //----------------------------------------------
    std::vector<GPU_COLUMN> child_results;
    std::vector<std::string> column_names;

    // All the sources of the UDF will be separate projections.
    // Create each one of them here.
    for (const auto& child : children) {
        std::shared_ptr<ExprGPUResult> child_res =
            child->ProcessBatch(input_batch);

        std::shared_ptr<ArrayExprGPUResult> child_as_array =
            std::dynamic_pointer_cast<ArrayExprGPUResult>(child_res);
        std::shared_ptr<ScalarExprGPUResult> child_as_scalar =
            std::dynamic_pointer_cast<ScalarExprGPUResult>(child_res);

        if (child_as_array) {
            child_results.emplace_back(child_as_array->result);
            column_names.emplace_back(child_as_array->column_name);
        } else if (child_as_scalar) {
            child_results.emplace_back(child_as_scalar->result);
            column_names.emplace_back("scalar");
        } else {
            throw std::runtime_error(
                "Child of UDF did not return an array or scalar.");
        }
    }
    // Put them all back together for the UDF to process.
    GPU_DATA udf_input = std::make_shared<table_info>(
        child_results, column_names, input_batch->metadata);

    // Actually run the UDF.
    GPU_DATA udf_output;
    if (cfunc_ptr) {
        if (cfunc_ptr == (table_udf_t)1) {
            PyThreadState* save = PyEval_SaveThread();
            cfunc_ptr = compile_future.get();
            PyEval_RestoreThread(save);
        }
        time_pt start_init_time = start_timer();
        udf_output = runCfuncScalarFunction(udf_input, cfunc_ptr);
        this->metrics.udf_execution_time += end_timer(start_init_time);
    } else {
        auto [out_temp, cpp_to_py_time, udf_time, py_to_cpp_time] =
            runPythonScalarFunction(udf_input, result_type,
                                    scalar_func_data.args,
                                    scalar_func_data.has_state, init_state);
        udf_output = out_temp;
        // Update the metrics.
        this->metrics.cpp_to_py_time += cpp_to_py_time;
        this->metrics.udf_execution_time += udf_time;
        this->metrics.py_to_cpp_time += py_to_cpp_time;
    }

    return std::make_shared<ArrayExprGPUResult>(udf_output->columns[0],
                                             udf_output->column_names[0]);
#endif
}

#if 0
std::shared_ptr<ExprGPUResult> PhysicalGPUArrowExpression::ProcessBatch(
    GPU_DATA input_batch) {
    std::shared_ptr<ExprGPUResult> res = children[0]->ProcessBatch(input_batch);
    GPU_COLUMN result;
    time_pt start_init_time = start_timer();
    std::shared_ptr
    if (scalar_func_data.arrow_func_name == "date") {
        // The Arrow compute equivalent of Series.dt.date() is year_month_day,
        // which returns a struct. To match the output dtype of Pandas, we Cast
        // to Date32 instead.
        auto cast_res = do_cudf_compute_cast(res, duckdb::LogicalType::DATE);

        std::visit([&](auto &vres) {
            using U = std::decay_t<decltype(vres)>;
            if constexpr (std::is_same_v<U, GPU_COLUMN>) {
                result = std::move(vres);
            } else if constexpr (std::is_same_v<U, GPU_SCALAR>) {
                throw std::runtime_error(
                    "Got scalar type in PhysicalGPUArrowExpression.");
            } else {
                throw std::runtime_error(
                    "Got unknown type in PhysicalGPUArrowExpression.");
            }
        }, cast_res);
    } else if (scalar_func_data.arrow_func_name == "match_substring_regex") {
        if (!PyTuple_Check(scalar_func_data.args) ||
            PyTuple_Size(scalar_func_data.args) != 1) {
            throw std::runtime_error(
                "match_substring_regex args not a 1-element tuple.");
        }

        // Get the first element (borrowed reference)
        PyObject* py_str = PyTuple_GetItem(scalar_func_data.args, 0);

        if (!PyUnicode_Check(py_str)) {
            throw std::runtime_error(
                "match_substring_regex args element is not a Python string.");
        }

        // Convert to UTF‑8 C string
        const char* c_str = PyUnicode_AsUTF8(py_str);
        if (!c_str) {
            throw std::runtime_error(
                "match_substring_regex error extracting Python string.");
        }

        arrow::compute::MatchSubstringOptions opts(c_str);
        result = do_cudf_compute_unary(res, scalar_func_data.arrow_func_name,
                                        &opts);
    } else {
        result = do_cudf_compute_unary(res, scalar_func_data.arrow_func_name);
    }
    this->metrics.arrow_compute_time += end_timer(start_init_time);
    return std::make_shared<ArrayExprGPUResult>(result, "Arrow Scalar");
}
#endif

bool PhysicalGPUExpression::join_expr(cudf::column** left_table,
                                      cudf::column** right_table,
                                      void** left_data, void** right_data,
                                      void** left_null_bitmap,
                                      void** right_null_bitmap,
                                      int64_t left_index, int64_t right_index) {
    throw std::runtime_error("PhysicalGPUExpression::join_expr unimplemented ");
#if 0
    arrow::Datum res = cur_join_expr->join_expr_internal(
        left_table, right_table, left_data, right_data, left_null_bitmap,
        right_null_bitmap, left_index, right_index);
    if (!res.is_scalar()) {
        throw std::runtime_error("join_expr_internal did not return scalar.");
    }
    if (res.scalar()->type->id() != arrow::Type::BOOL) {
        throw std::runtime_error("join_expr_internal did not return bool.");
    }
    auto bool_scalar =
        std::dynamic_pointer_cast<arrow::BooleanScalar>(res.scalar());
    if (bool_scalar && bool_scalar->is_valid) {
        return bool_scalar->value;
    } else {
        throw std::runtime_error("join_expr_internal bool is null or invalid.");
    }
#endif
}

void PhysicalGPUExpression::join_expr_batch(
    cudf::column** left_table, cudf::column** right_table, void** left_data,
    void** right_data, void** left_null_bitmap, void** right_null_bitmap,
    uint8_t* match_arr, int64_t left_index_start, int64_t left_index_end,
    int64_t right_index_start, int64_t right_index_end) {
    throw std::runtime_error("PhysicalGPUExpression::join_expr unimplemented ");
#if 0
    for (int64_t j = right_index_start; j < right_index_end; j++) {
        for (int64_t i = left_index_start; i < left_index_end; i++) {
            SetBitTo(match_arr,
                     (i - left_index_start) + (j - right_index_start),
                     join_expr(left_table, right_table, left_data, right_data,
                               left_null_bitmap, right_null_bitmap, i, j));
        }
    }
#endif
}

PhysicalGPUExpression* PhysicalGPUExpression::cur_join_expr = nullptr;
