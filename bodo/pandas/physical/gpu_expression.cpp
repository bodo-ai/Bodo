#include "gpu_expression.h"
#include <cudf/ast/expressions.hpp>
#include <memory>
#include <stdexcept>
#include <vector>
#include "_util.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"

std::variant<GPU_COLUMN, GPU_SCALAR> do_cudf_compute_binary(
    std::shared_ptr<ExprGPUResult> left_res,
    std::shared_ptr<ExprGPUResult> right_res,
    const cudf::binary_operator& comparator, std::shared_ptr<StreamAndEvent> se,
    const cudf::data_type& cudf_result_type) {
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

    GPU_COLUMN res;
    if (left_as_array) {
        if (right_as_array) {
            res = cudf::binary_operation(
                left_as_array->result->view(), right_as_array->result->view(),
                comparator, cudf_result_type, se->stream);
        } else if (right_as_scalar) {
            res = cudf::binary_operation(left_as_array->result->view(),
                                         *(right_as_scalar->result), comparator,
                                         cudf_result_type, se->stream);
        } else {
            throw std::runtime_error(
                "do_cudf_compute_binary right is neither array nor scalar.");
        }
    } else if (left_as_scalar) {
        if (right_as_array) {
            res = cudf::binary_operation(
                *(left_as_scalar->result), right_as_array->result->view(),
                comparator, cudf_result_type, se->stream);
        } else if (right_as_scalar) {
            throw std::runtime_error(
                "do_cudf_compute_binary both left and right are scalar.");
        } else {
            throw std::runtime_error(
                "do_cudf_compute_binary right is neither array nor scalar.");
        }
    } else {
        throw std::runtime_error(
            "do_cudf_compute_binary left is neither array nor scalar.");
    }

    return std::move(res);
}

std::variant<GPU_COLUMN, GPU_SCALAR> do_cudf_compute_unary(
    std::shared_ptr<ExprGPUResult> left_res,
    const cudf::unary_operator& comparator, std::shared_ptr<StreamAndEvent> se,
    const arrow::compute::FunctionOptions* func_options) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprGPUResult> left_as_array =
        std::dynamic_pointer_cast<ArrayExprGPUResult>(left_res);
    std::shared_ptr<ScalarExprGPUResult> left_as_scalar =
        std::dynamic_pointer_cast<ScalarExprGPUResult>(left_res);

    if (left_as_array) {
        return cudf::unary_operation(left_as_array->result->view(), comparator,
                                     se->stream);
    } else if (left_as_scalar) {
        throw std::runtime_error(
            "do_cudf_compute_unary for scalar not yet implemented.");
    } else {
        throw std::runtime_error(
            "do_cudf_compute_unary left is neither array nor scalar.");
    }
}

std::variant<GPU_COLUMN, GPU_SCALAR> do_cudf_compute_cast(
    std::shared_ptr<ExprGPUResult> left_res, cudf::data_type& cudf_result_type,
    std::shared_ptr<StreamAndEvent> se) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprGPUResult> left_as_array =
        std::dynamic_pointer_cast<ArrayExprGPUResult>(left_res);
    std::shared_ptr<ScalarExprGPUResult> left_as_scalar =
        std::dynamic_pointer_cast<ScalarExprGPUResult>(left_res);

    if (left_as_array) {
        return cudf::cast(left_as_array->result->view(), cudf_result_type,
                          se->stream);
    } else if (left_as_scalar) {
        throw std::runtime_error(
            "do_cudf_compute_cast cast of scalar not yet implemented.");
    } else {
        throw std::runtime_error(
            "do_cudf_compute_cast left is neither array nor scalar.");
    }
}

GPU_COLUMN do_cudf_compute_case(std::shared_ptr<ExprGPUResult> when_res,
                                std::shared_ptr<ExprGPUResult> then_res,
                                std::shared_ptr<ExprGPUResult> else_res,
                                std::shared_ptr<StreamAndEvent> se) {
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
            res = cudf::copy_if_else(then_as_array->result->view(),
                                     else_as_array->result->view(), when_col,
                                     se->stream);
        } else if (else_as_scalar) {
            res = cudf::copy_if_else(then_as_array->result->view(),
                                     *(else_as_scalar->result), when_col,
                                     se->stream);
        } else {
            throw std::runtime_error(
                "do_cudf_compute_case right is neither array nor scalar.");
        }
    } else if (then_as_scalar) {
        if (else_as_array) {
            res = cudf::copy_if_else(*(then_as_scalar->result),
                                     else_as_array->result->view(), when_col,
                                     se->stream);
        } else if (else_as_scalar) {
            throw std::runtime_error(
                "do_cudf_compute_case both then and else are scalar.");
        } else {
            throw std::runtime_error(
                "do_cudf_compute_case else is neither array nor scalar.");
        }
    } else {
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
                    expr_type, bce.return_type));
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
                    expr_type, bce.return_type));
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
                            expr_type, boe.return_type));
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
            duckdb::LogicalType result_type = bfe.return_type;

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
                    return std::static_pointer_cast<PhysicalGPUExpression>(
                        std::make_shared<PhysicalGPUArrowExpression>(
                            phys_children, scalar_func_data, result_type));
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
                            : duckdb::ExpressionType::COMPARE_GREATERTHAN,
                        bbe.return_type));

            std::shared_ptr<PhysicalGPUExpression> right =
                std::static_pointer_cast<PhysicalGPUExpression>(
                    std::make_shared<PhysicalGPUComparisonExpression>(
                        upper_expr, input_expr,
                        bbe.upper_inclusive
                            ? duckdb::ExpressionType::
                                  COMPARE_GREATERTHANOREQUALTO
                            : duckdb::ExpressionType::COMPARE_GREATERTHAN,
                        bbe.return_type));

            return std::static_pointer_cast<PhysicalGPUExpression>(
                std::make_shared<PhysicalGPUConjunctionExpression>(
                    left, right, duckdb::ExpressionType::CONJUNCTION_AND,
                    bbe.return_type));
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

bool gpu_capable(duckdb::Expression& expr) {
    duckdb::ExpressionClass expr_class = expr.GetExpressionClass();

    switch (expr_class) {
        case duckdb::ExpressionClass::BOUND_COMPARISON: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundComparisonExpression>();
            return gpu_capable(bce.left) && gpu_capable(bce.right);
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_COLUMN_REF:
            return true;
        case duckdb::ExpressionClass::BOUND_CONSTANT:
            return true;
        case duckdb::ExpressionClass::BOUND_CONJUNCTION: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundConjunctionExpression>();
            return gpu_capable(bce.children[0]) && gpu_capable(bce.children[1]);
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_OPERATOR: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& boe = expr.Cast<duckdb::BoundOperatorExpression>();
            switch (boe.children.size()) {
                case 1:
                    return gpu_capable(boe.children[0]);
                case 2:
                    return gpu_capable(boe.children[0]) &&
                           gpu_capable(boe.children[1]);
                default:
                    throw std::runtime_error(
                        "Unsupported number of children for bound operator");
            }
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_FUNCTION: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bfe = expr.Cast<duckdb::BoundFunctionExpression>();

            if (bfe.bind_info &&
                (bfe.bind_info->Cast<BodoScalarFunctionData>().args ||
                 !bfe.bind_info->Cast<BodoScalarFunctionData>()
                      .arrow_func_name.empty())) {
                BodoScalarFunctionData& scalar_func_data =
                    bfe.bind_info->Cast<BodoScalarFunctionData>();

                for (auto& child_expr : bfe.children) {
                    if (!gpu_capable(child_expr)) {
                        return false;
                    }
                }

                if (!scalar_func_data.arrow_func_name.empty()) {
                    return scalar_func_data.arrow_func_name == "ends_with" ||
                           scalar_func_data.arrow_func_name == "starts_with" ||
                           scalar_func_data.arrow_func_name ==
                               "match_substring_regex" ||
                           scalar_func_data.arrow_func_name ==
                               "match_substring_regex_first" ||
                           scalar_func_data.arrow_func_name ==
                               "utf8_slice_codeunits" ||
                           scalar_func_data.arrow_func_name == "year" ||
                           scalar_func_data.arrow_func_name == "round" ||
                           scalar_func_data.arrow_func_name == "is_in" ||
                           scalar_func_data.arrow_func_name == "is_null";
                } else if (scalar_func_data.args) {
                    return false;
                }
            } else {
                switch (bfe.children.size()) {
                    case 1:
                        return gpu_capable(bfe.children[0]);
                    case 2:
                        return gpu_capable(bfe.children[0]) &&
                               gpu_capable(bfe.children[1]);
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
            return gpu_capable(bce.child);
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_BETWEEN: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bbe = expr.Cast<duckdb::BoundBetweenExpression>();
            return gpu_capable(bbe.input) && gpu_capable(bbe.lower) &&
                   gpu_capable(bbe.upper);
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
            return gpu_capable(caseCheck.when_expr) &&
                   gpu_capable(caseCheck.then_expr) &&
                   gpu_capable(bce.else_expr);
        } break;  // suppress wrong fallthrough error
        default:
            throw std::runtime_error(
                "Unsupported duckdb expression class " +
                std::to_string(static_cast<int>(expr_class)));
    }
    throw std::logic_error("Control should never reach here");
}

bool gpu_capable(duckdb::unique_ptr<duckdb::Expression>& expr) {
    return gpu_capable(*expr);
}

std::shared_ptr<ExprGPUResult> PhysicalGPUUDFExpression::ProcessBatch(
    GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) {
    throw std::runtime_error(
        "PhysicalGPUUDFExpression::ProcessBatch unimplemented ");
}

bool PhysicalGPUExpression::join_expr(cudf::column** left_table,
                                      cudf::column** right_table,
                                      void** left_data, void** right_data,
                                      void** left_null_bitmap,
                                      void** right_null_bitmap,
                                      int64_t left_index, int64_t right_index) {
    throw std::runtime_error("PhysicalGPUExpression::join_expr unimplemented ");
}

void PhysicalGPUExpression::join_expr_batch(
    cudf::column** left_table, cudf::column** right_table, void** left_data,
    void** right_data, void** left_null_bitmap, void** right_null_bitmap,
    uint8_t* match_arr, int64_t left_index_start, int64_t left_index_end,
    int64_t right_index_start, int64_t right_index_end) {
    throw std::runtime_error("PhysicalGPUExpression::join_expr unimplemented ");
}

PhysicalGPUExpression* PhysicalGPUExpression::cur_join_expr = nullptr;

/**
 * @brief convert duckdb comparison type to cudf ast operator type
 */
cudf::ast::ast_operator comparisonTypeToCudfOp(duckdb::ExpressionType t) {
    using ET = duckdb::ExpressionType;
    switch (t) {
        case ET::COMPARE_EQUAL:
            return cudf::ast::ast_operator::EQUAL;
        case ET::COMPARE_NOTEQUAL:
            return cudf::ast::ast_operator::NOT_EQUAL;
        case ET::COMPARE_GREATERTHAN:
            return cudf::ast::ast_operator::GREATER;
        case ET::COMPARE_LESSTHAN:
            return cudf::ast::ast_operator::LESS;
        case ET::COMPARE_GREATERTHANOREQUALTO:
            return cudf::ast::ast_operator::GREATER_EQUAL;
        case ET::COMPARE_LESSTHANOREQUALTO:
            return cudf::ast::ast_operator::LESS_EQUAL;
        default:
            throw std::runtime_error(
                "comparisonTypeToCudfOp(): Unhandled comparison type");
    }
}

/**
 * @brief Convert a duckdb Value to a cudf literal and add it to the filter AST
 * expressions.
 *
 * @param value - the duckdb Value to convert
 * @param filter_ast_tree - the cudf AST expressions to which the literal will
 * be added (all components should be added to be kept alive)
 * @param filter_scalars - vector to store the created cudf scalars to keep them
 * alive
 */
void duckdbValuetoCudfLiteral(
    const duckdb::Value& value, cudf::ast::tree& filter_ast_tree,
    std::vector<std::unique_ptr<cudf::scalar>>& filter_scalars) {
    duckdb::LogicalTypeId type = value.type().id();
    switch (type) {
        case duckdb::LogicalTypeId::TINYINT: {
            auto literal_value = std::make_unique<cudf::numeric_scalar<int8_t>>(
                value.GetValue<int8_t>());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<int8_t>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::SMALLINT: {
            auto literal_value =
                std::make_unique<cudf::numeric_scalar<int16_t>>(
                    value.GetValue<int16_t>());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<int16_t>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::INTEGER: {
            auto literal_value =
                std::make_unique<cudf::numeric_scalar<int32_t>>(
                    value.GetValue<int32_t>());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<int32_t>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::BIGINT: {
            auto literal_value =
                std::make_unique<cudf::numeric_scalar<int64_t>>(
                    value.GetValue<int64_t>());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<int64_t>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::UTINYINT: {
            auto literal_value =
                std::make_unique<cudf::numeric_scalar<uint8_t>>(
                    value.GetValue<uint8_t>());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<uint8_t>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::USMALLINT: {
            auto literal_value =
                std::make_unique<cudf::numeric_scalar<uint16_t>>(
                    value.GetValue<uint16_t>());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(cudf::ast::literal(
                *static_cast<cudf::numeric_scalar<uint16_t>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::UINTEGER: {
            auto literal_value =
                std::make_unique<cudf::numeric_scalar<uint32_t>>(
                    value.GetValue<uint32_t>());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(cudf::ast::literal(
                *static_cast<cudf::numeric_scalar<uint32_t>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::UBIGINT: {
            auto literal_value =
                std::make_unique<cudf::numeric_scalar<uint64_t>>(
                    value.GetValue<uint64_t>());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(cudf::ast::literal(
                *static_cast<cudf::numeric_scalar<uint64_t>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::FLOAT: {
            auto literal_value = std::make_unique<cudf::numeric_scalar<float>>(
                value.GetValue<float>());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<float>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::DOUBLE: {
            auto literal_value = std::make_unique<cudf::numeric_scalar<double>>(
                value.GetValue<double>());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<double>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::BOOLEAN: {
            auto literal_value = std::make_unique<cudf::numeric_scalar<bool>>(
                value.GetValue<bool>());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::numeric_scalar<bool>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::VARCHAR: {
            auto literal_value = std::make_unique<cudf::string_scalar>(
                value.GetValue<std::string>());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(
                cudf::ast::literal(*static_cast<cudf::string_scalar*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::TIMESTAMP: {
            // Define a timestamp type with microsecond precision
            duckdb::timestamp_t extracted =
                value.GetValue<duckdb::timestamp_t>();
            // Create a TimestampScalar with microsecond value
            auto literal_value =
                std::make_unique<cudf::timestamp_scalar<cudf::timestamp_us>>(
                    cudf::timestamp_us{
                        cuda::std::chrono::microseconds{extracted.value}},
                    !value.IsNull());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(cudf::ast::literal(
                *static_cast<cudf::timestamp_scalar<cudf::timestamp_us>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::TIMESTAMP_MS: {
            // Define a timestamp type with millisecond precision
            duckdb::timestamp_ms_t extracted =
                value.GetValue<duckdb::timestamp_ms_t>();
            // Create a TimestampScalar with millisecond value
            auto literal_value =
                std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ms>>(
                    cudf::timestamp_ms{
                        cuda::std::chrono::milliseconds{extracted.value}},
                    !value.IsNull());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(cudf::ast::literal(
                *static_cast<cudf::timestamp_scalar<cudf::timestamp_ms>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::TIMESTAMP_SEC: {
            // Define a timestamp type with millisecond precision
            duckdb::timestamp_sec_t extracted =
                value.GetValue<duckdb::timestamp_sec_t>();
            // Create a TimestampScalar with second value
            auto literal_value =
                std::make_unique<cudf::timestamp_scalar<cudf::timestamp_s>>(
                    cudf::timestamp_s{
                        cuda::std::chrono::seconds{extracted.value}},
                    !value.IsNull());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(cudf::ast::literal(
                *static_cast<cudf::timestamp_scalar<cudf::timestamp_s>*>(
                    filter_scalars.back().get())));
            return;
        }
        case duckdb::LogicalTypeId::TIMESTAMP_NS: {
            // Define a timestamp type with nanosecond precision
            duckdb::timestamp_ns_t extracted =
                value.GetValue<duckdb::timestamp_ns_t>();
            // Create a TimestampScalar with nanosecond value
            auto literal_value =
                std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ns>>(
                    cudf::timestamp_ns{
                        cuda::std::chrono::nanoseconds{extracted.value}},
                    !value.IsNull());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(cudf::ast::literal(
                *static_cast<cudf::timestamp_scalar<cudf::timestamp_ns>*>(
                    filter_scalars.back().get())));
            return;
        }
        // TODO(ehsan): support TIMESTAMP_TZ which is not trivial since cudf
        // types don't have timezones
        case duckdb::LogicalTypeId::DATE: {
            // Define a date type
            duckdb::date_t extracted = value.GetValue<duckdb::date_t>();
            // Create a DateScalar with the date value
            auto literal_value =
                std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
                    cudf::timestamp_D{cuda::std::chrono::days{extracted.days}},
                    !value.IsNull());
            filter_scalars.push_back(std::move(literal_value));
            filter_ast_tree.push(cudf::ast::literal(
                *static_cast<cudf::timestamp_scalar<cudf::timestamp_D>*>(
                    filter_scalars.back().get())));
            return;
        }
        default:
            throw std::runtime_error(
                "duckdbValuetoCudfLiteral unhandled type." +
                std::to_string(static_cast<int>(type)));
    }
}

/**
 * @brief Convert a duckdb TableFilter to cudf AST expressions.
 *
 * @param col_idx - index of the column to which the filter applies
 * @param tf - duckdb TableFilter to convert
 * @param column_names - column names of the table (before removing unused
 * columns)
 * @param filter_ast_tree - output cudf AST expressions representing the
 * filters. All expressions should be added to be kept alive.
 * @param filter_scalars - output vector of cudf scalars representing any
 * constants in the filters. All scalars should be added to be kept alive.
 */
void tableFilterToCudfAST(
    duckdb::idx_t col_idx, duckdb::unique_ptr<duckdb::TableFilter>& tf,
    const std::vector<std::string>& column_names,
    cudf::ast::tree& filter_ast_tree,
    std::vector<std::unique_ptr<cudf::scalar>>& filter_scalars) {
    using TF = duckdb::TableFilterType;

    switch (tf->filter_type) {
        case TF::CONSTANT_COMPARISON: {
            auto cf =
                dynamic_cast_unique_ptr<duckdb::ConstantFilter>(std::move(tf));
            cudf::ast::ast_operator cmp_kind =
                comparisonTypeToCudfOp(cf->comparison_type);
            cudf::ast::column_name_reference col_ref =
                cudf::ast::column_name_reference(column_names[col_idx]);
            filter_ast_tree.push(col_ref);
            duckdbValuetoCudfLiteral(cf->constant, filter_ast_tree,
                                     filter_scalars);
            cudf::ast::operation expr = cudf::ast::operation(
                cmp_kind, filter_ast_tree[filter_ast_tree.size() - 2],
                filter_ast_tree.back());
            filter_ast_tree.push(expr);
        } break;

        case TF::CONJUNCTION_AND: {
            auto af = dynamic_cast_unique_ptr<duckdb::ConjunctionAndFilter>(
                std::move(tf));
            if (af->child_filters.size() < 2) {
                throw std::runtime_error("AND filter with <2 children");
            }
            tableFilterToCudfAST(col_idx, af->child_filters[0], column_names,
                                 filter_ast_tree, filter_scalars);
            for (std::size_t i = 1; i < af->child_filters.size(); ++i) {
                const cudf::ast::expression* prev_child =
                    &filter_ast_tree.back();
                tableFilterToCudfAST(col_idx, af->child_filters[i],
                                     column_names, filter_ast_tree,
                                     filter_scalars);
                const cudf::ast::expression* new_child =
                    &filter_ast_tree.back();
                cudf::ast::operation expr =
                    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND,
                                         *prev_child, *new_child);
                filter_ast_tree.push(expr);
            }
        } break;

        case TF::CONJUNCTION_OR: {
            auto of = dynamic_cast_unique_ptr<duckdb::ConjunctionOrFilter>(
                std::move(tf));
            if (of->child_filters.size() < 2) {
                throw std::runtime_error("OR filter with <2 children");
            }
            tableFilterToCudfAST(col_idx, of->child_filters[0], column_names,
                                 filter_ast_tree, filter_scalars);
            for (std::size_t i = 1; i < of->child_filters.size(); ++i) {
                const cudf::ast::expression* prev_child =
                    &filter_ast_tree.back();
                tableFilterToCudfAST(col_idx, of->child_filters[i],
                                     column_names, filter_ast_tree,
                                     filter_scalars);
                const cudf::ast::expression* new_child =
                    &filter_ast_tree.back();
                cudf::ast::operation expr =
                    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR,
                                         *prev_child, *new_child);
                filter_ast_tree.push(expr);
            }
        } break;

        case TF::OPTIONAL_FILTER: {
            auto of =
                dynamic_cast_unique_ptr<duckdb::OptionalFilter>(std::move(tf));
            try {
                tableFilterToCudfAST(col_idx, of->child_filter, column_names,
                                     filter_ast_tree, filter_scalars);
            } catch (...) {
                // No-op: literal true
                auto literal_value =
                    std::make_unique<cudf::numeric_scalar<bool>>(true);
                filter_scalars.push_back(std::move(literal_value));
                filter_ast_tree.push(cudf::ast::literal(
                    *static_cast<cudf::numeric_scalar<bool>*>(
                        filter_scalars.back().get())));
                cudf::ast::operation expr = cudf::ast::operation(
                    cudf::ast::ast_operator::IDENTITY, filter_ast_tree.back());
                filter_ast_tree.push(expr);
            }
        } break;

        default:
            throw std::runtime_error(
                "tableFilterToCudfAST(): Unsupported TableFilter type");
    }
}

void tableFilterSetToCudfAST(
    duckdb::TableFilterSet& filters,
    const std::vector<std::string>& column_names,
    cudf::ast::tree& filter_ast_tree,
    std::vector<std::unique_ptr<cudf::scalar>>& filter_scalars) {
    bool first = true;

    // Combine all filters with AND
    for (auto& pair : filters.filters) {
        const cudf::ast::expression* prev_cond =
            first ? nullptr : &filter_ast_tree.back();

        duckdb::idx_t col_idx = pair.first;
        auto& tf = pair.second;

        tableFilterToCudfAST(col_idx, tf, column_names, filter_ast_tree,
                             filter_scalars);

        if (first) {
            first = false;
        } else {
            filter_ast_tree.push(
                cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND,
                                     *prev_cond, filter_ast_tree.back()));
        }
    }
}

CudfASTOwner build_mixed_join_predicate(
    const std::vector<duckdb::unique_ptr<duckdb::Expression>>& exprs,
    const std::unordered_set<duckdb::idx_t>& left_table_indices,
    rmm::cuda_stream_view& stream) {
    if (exprs.empty()) {
        throw std::runtime_error(
            "build_combined_predicate: no expressions provided");
    }

    CudfASTOwner owner;

    // Convert the first expression — its root becomes the accumulator.
    const cudf::ast::expression* acc =
        &duckdb_expr_to_cudf_ast(*exprs[0], left_table_indices, owner, stream);

    // Each subsequent expression is converted into the same owner and
    // AND-ed with the accumulated root. Because all nodes live in the
    // same tree, the references remain valid.
    for (size_t i = 1; i < exprs.size(); ++i) {
        const cudf::ast::expression& rhs = duckdb_expr_to_cudf_ast(
            *exprs[i], left_table_indices, owner, stream);
        acc = &owner.push(cudf::ast::operation(
            cudf::ast::ast_operator::LOGICAL_AND, *acc, rhs));
    }

    return owner;
}

const cudf::ast::expression& duckdb_expr_to_cudf_ast(
    const duckdb::Expression& expr,
    const std::unordered_set<duckdb::idx_t>& left_table_indices,
    CudfASTOwner& owner, rmm::cuda_stream_view& stream) {
    switch (expr.expression_class) {
        case duckdb::ExpressionClass::BOUND_COLUMN_REF: {
            auto& col_ref = expr.Cast<duckdb::BoundColumnRefExpression>();

            duckdb::idx_t table_idx = col_ref.binding.table_index;
            duckdb::idx_t col_idx = col_ref.binding.column_index;

            cudf::ast::table_reference table_ref =
                left_table_indices.count(table_idx)
                    ? cudf::ast::table_reference::LEFT
                    : cudf::ast::table_reference::RIGHT;
            return owner.push(cudf::ast::column_reference(col_idx, table_ref));
        } break;

        case duckdb::ExpressionClass::BOUND_CONSTANT: {
            auto& const_expr = expr.Cast<duckdb::BoundConstantExpression>();
            const duckdb::Value& val = const_expr.value;

            owner.insert_literal(val, stream);

            return owner.tree.back();
        } break;

        case duckdb::ExpressionClass::BOUND_COMPARISON: {
            auto& cmp = expr.Cast<duckdb::BoundComparisonExpression>();

            const cudf::ast::expression& lhs = duckdb_expr_to_cudf_ast(
                *cmp.left, left_table_indices, owner, stream);
            const cudf::ast::expression& rhs = duckdb_expr_to_cudf_ast(
                *cmp.right, left_table_indices, owner, stream);

            cudf::ast::ast_operator op = duckdb_etype_to_cudf_ast_op(expr.type);
            return owner.push(cudf::ast::operation(op, lhs, rhs));
        }

        case duckdb::ExpressionClass::BOUND_CONJUNCTION: {
            auto& conj = expr.Cast<duckdb::BoundConjunctionExpression>();

            if (conj.children.size() < 2) {
                throw std::runtime_error(
                    "duckdb_expr_to_cudf_ast: conjunction has fewer than 2 "
                    "children");
            }

            cudf::ast::ast_operator op = duckdb_etype_to_cudf_ast_op(expr.type);

            const cudf::ast::expression* acc = &duckdb_expr_to_cudf_ast(
                *conj.children[0], left_table_indices, owner, stream);

            for (size_t i = 1; i < conj.children.size(); ++i) {
                const cudf::ast::expression& rhs = duckdb_expr_to_cudf_ast(
                    *conj.children[i], left_table_indices, owner, stream);
                acc = &owner.push(cudf::ast::operation(op, *acc, rhs));
            }
            return *acc;
        } break;

        case duckdb::ExpressionClass::BOUND_OPERATOR: {
            auto& op_expr = expr.Cast<duckdb::BoundOperatorExpression>();

            if (expr.type == duckdb::ExpressionType::OPERATOR_NOT) {
                if (op_expr.children.size() != 1) {
                    throw std::runtime_error(
                        "duckdb_expr_to_cudf_ast: NOT must have exactly 1 "
                        "child");
                }
                const cudf::ast::expression& child = duckdb_expr_to_cudf_ast(
                    *op_expr.children[0], left_table_indices, owner, stream);
                return owner.push(
                    cudf::ast::operation(cudf::ast::ast_operator::NOT, child));
            }

            throw std::runtime_error(
                "duckdb_expr_to_cudf_ast: unsupported BOUND_OPERATOR type " +
                std::to_string(static_cast<int>(expr.type)));
        } break;

        default:
            throw std::runtime_error(
                "duckdb_expr_to_cudf_ast: unsupported expression class " +
                std::to_string(static_cast<int>(expr.expression_class)));
    }
}

void CudfASTOwner::insert_literal(const duckdb::Value& val,
                                  rmm::cuda_stream_view& stream) {
    // Helper to push a typed literal and then transfer ownership of the
    // scalar into the owner. Must happen in this order: literal holds a
    // ref to the scalar so it must be pushed into the tree first.
    auto push_literal = [&](auto typed_scalar) {
        using ScalarT = std::decay_t<decltype(*typed_scalar)>;
        ScalarT* raw = typed_scalar.get();
        this->push(cudf::ast::literal(*raw));
        this->scalars.push_back(std::move(typed_scalar));
    };

    switch (val.type().id()) {
        case duckdb::LogicalTypeId::BOOLEAN:
            push_literal(std::make_unique<cudf::numeric_scalar<int8_t>>(
                static_cast<int8_t>(val.GetValue<bool>()), true, stream));
            break;
        case duckdb::LogicalTypeId::TINYINT:
            push_literal(std::make_unique<cudf::numeric_scalar<int8_t>>(
                val.GetValue<int8_t>(), true, stream));
            break;
        case duckdb::LogicalTypeId::SMALLINT:
            push_literal(std::make_unique<cudf::numeric_scalar<int16_t>>(
                val.GetValue<int16_t>(), true, stream));
            break;
        case duckdb::LogicalTypeId::INTEGER:
            push_literal(std::make_unique<cudf::numeric_scalar<int32_t>>(
                val.GetValue<int32_t>(), true, stream));
            break;
        case duckdb::LogicalTypeId::BIGINT:
            push_literal(std::make_unique<cudf::numeric_scalar<int64_t>>(
                val.GetValue<int64_t>(), true, stream));
            break;
        case duckdb::LogicalTypeId::UTINYINT:
            push_literal(std::make_unique<cudf::numeric_scalar<uint8_t>>(
                val.GetValue<uint8_t>(), true, stream));
            break;
        case duckdb::LogicalTypeId::USMALLINT:
            push_literal(std::make_unique<cudf::numeric_scalar<uint16_t>>(
                val.GetValue<uint16_t>(), true, stream));
            break;
        case duckdb::LogicalTypeId::UINTEGER:
            push_literal(std::make_unique<cudf::numeric_scalar<uint32_t>>(
                val.GetValue<uint32_t>(), true, stream));
            break;
        case duckdb::LogicalTypeId::UBIGINT:
            push_literal(std::make_unique<cudf::numeric_scalar<uint64_t>>(
                val.GetValue<uint64_t>(), true, stream));
            break;
        case duckdb::LogicalTypeId::FLOAT:
            push_literal(std::make_unique<cudf::numeric_scalar<float>>(
                val.GetValue<float>(), true, stream));
            break;
        case duckdb::LogicalTypeId::DOUBLE:
            push_literal(std::make_unique<cudf::numeric_scalar<double>>(
                val.GetValue<double>(), true, stream));
            break;
        case duckdb::LogicalTypeId::VARCHAR:
            push_literal(std::make_unique<cudf::string_scalar>(
                val.GetValue<std::string>(), true, stream));
            break;
        case duckdb::LogicalTypeId::DATE:
            push_literal(
                std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
                    cudf::timestamp_D{
                        cudf::duration_D{val.GetValue<int32_t>()}},
                    true, stream));
            break;
        case duckdb::LogicalTypeId::TIMESTAMP:
            push_literal(
                std::make_unique<cudf::timestamp_scalar<cudf::timestamp_us>>(
                    cudf::timestamp_us{
                        cudf::duration_us{val.GetValue<int64_t>()}},
                    true, stream));
            break;
        default:
            throw std::runtime_error(
                "duckdb_value_to_cudf_ast_literal: unsupported duckdb type " +
                val.type().ToString());
    }
}

cudf::ast::ast_operator duckdb_etype_to_cudf_ast_op(
    duckdb::ExpressionType etype) {
    switch (etype) {
        case duckdb::ExpressionType::COMPARE_EQUAL:
            return cudf::ast::ast_operator::EQUAL;
        case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            return cudf::ast::ast_operator::NOT_EQUAL;
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            return cudf::ast::ast_operator::GREATER;
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
            return cudf::ast::ast_operator::LESS;
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            return cudf::ast::ast_operator::GREATER_EQUAL;
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return cudf::ast::ast_operator::LESS_EQUAL;
        case duckdb::ExpressionType::CONJUNCTION_AND:
            return cudf::ast::ast_operator::LOGICAL_AND;
        case duckdb::ExpressionType::CONJUNCTION_OR:
            return cudf::ast::ast_operator::LOGICAL_OR;
        case duckdb::ExpressionType::OPERATOR_NOT:
            return cudf::ast::ast_operator::NOT;
        default:
            throw std::runtime_error(
                "duckdb_etype_to_cudf_ast_op: unsupported ExpressionType " +
                std::to_string(static_cast<int>(etype)));
    }
}
