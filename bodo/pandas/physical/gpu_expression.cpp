#include "gpu_expression.h"
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
            res = std::move(cudf::copy_if_else(then_as_array->result->view(),
                                               else_as_array->result->view(),
                                               when_col, se->stream));
        } else if (else_as_scalar) {
            res = std::move(cudf::copy_if_else(then_as_array->result->view(),
                                               *(else_as_scalar->result),
                                               when_col, se->stream));
        } else {
            throw std::runtime_error(
                "do_cudf_compute_case right is neither array nor scalar.");
        }
    } else if (then_as_scalar) {
        if (else_as_array) {
            res = std::move(cudf::copy_if_else(*(then_as_scalar->result),
                                               else_as_array->result->view(),
                                               when_col, se->stream));
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
                    throw std::runtime_error("Unimplemented");
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
                    throw std::runtime_error("Unimplemented");
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

// Helper constructors
std::unique_ptr<CudfExpr> make_column_ref(int col_idx) {
    auto e = std::make_unique<CudfExpr>();
    e->kind = CudfExpr::Kind::COLUMN_REF;
    e->column_index = col_idx;
    return e;
}

std::unique_ptr<CudfExpr> make_literal(const duckdb::Value& v) {
    auto e = std::make_unique<CudfExpr>();
    e->kind = CudfExpr::Kind::LITERAL;
    e->literal = v;
    return e;
}

std::unique_ptr<CudfExpr> make_binary(CudfExpr::Kind k,
                                      std::unique_ptr<CudfExpr> lhs,
                                      std::unique_ptr<CudfExpr> rhs) {
    auto e = std::make_unique<CudfExpr>();
    e->kind = k;
    e->left = std::move(lhs);
    e->right = std::move(rhs);
    return e;
}

std::unique_ptr<CudfExpr> make_and(std::unique_ptr<CudfExpr> lhs,
                                   std::unique_ptr<CudfExpr> rhs) {
    return make_binary(CudfExpr::Kind::AND, std::move(lhs), std::move(rhs));
}

std::unique_ptr<CudfExpr> make_or(std::unique_ptr<CudfExpr> lhs,
                                  std::unique_ptr<CudfExpr> rhs) {
    return make_binary(CudfExpr::Kind::OR, std::move(lhs), std::move(rhs));
}

// Map DuckDB comparison type → CudfExpr::Kind
CudfExpr::Kind comparisonTypeToCudfKind(duckdb::ExpressionType t) {
    using ET = duckdb::ExpressionType;
    switch (t) {
        case ET::COMPARE_EQUAL:
            return CudfExpr::Kind::EQ;
        case ET::COMPARE_NOTEQUAL:
            return CudfExpr::Kind::NE;
        case ET::COMPARE_GREATERTHAN:
            return CudfExpr::Kind::GT;
        case ET::COMPARE_LESSTHAN:
            return CudfExpr::Kind::LT;
        case ET::COMPARE_GREATERTHANOREQUALTO:
            return CudfExpr::Kind::GE;
        case ET::COMPARE_LESSTHANOREQUALTO:
            return CudfExpr::Kind::LE;
        default:
            throw std::runtime_error("Unhandled comparison type");
    }
}

std::unique_ptr<cudf::scalar> duckdbValueToCudfScalar(
    const duckdb::Value& value, std::shared_ptr<StreamAndEvent> se) {
    duckdb::LogicalTypeId type = value.type().id();
    switch (type) {
        case duckdb::LogicalTypeId::TINYINT:
            return std::make_unique<cudf::numeric_scalar<int8_t>>(
                value.GetValue<int8_t>(), !value.IsNull(), se->stream);
        case duckdb::LogicalTypeId::SMALLINT:
            return std::make_unique<cudf::numeric_scalar<int16_t>>(
                value.GetValue<int16_t>(), !value.IsNull(), se->stream);
        case duckdb::LogicalTypeId::INTEGER:
            return std::make_unique<cudf::numeric_scalar<int32_t>>(
                value.GetValue<int32_t>(), !value.IsNull(), se->stream);
        case duckdb::LogicalTypeId::BIGINT:
            return std::make_unique<cudf::numeric_scalar<int64_t>>(
                value.GetValue<int64_t>(), !value.IsNull(), se->stream);
        case duckdb::LogicalTypeId::UTINYINT:
            return std::make_unique<cudf::numeric_scalar<uint8_t>>(
                value.GetValue<uint8_t>(), !value.IsNull(), se->stream);
        case duckdb::LogicalTypeId::USMALLINT:
            return std::make_unique<cudf::numeric_scalar<uint16_t>>(
                value.GetValue<uint16_t>(), !value.IsNull(), se->stream);
        case duckdb::LogicalTypeId::UINTEGER:
            return std::make_unique<cudf::numeric_scalar<uint32_t>>(
                value.GetValue<uint32_t>(), !value.IsNull(), se->stream);
        case duckdb::LogicalTypeId::UBIGINT:
            return std::make_unique<cudf::numeric_scalar<uint64_t>>(
                value.GetValue<uint64_t>(), !value.IsNull(), se->stream);
        case duckdb::LogicalTypeId::FLOAT:
            return std::make_unique<cudf::numeric_scalar<float>>(
                value.GetValue<float>(), !value.IsNull(), se->stream);
        case duckdb::LogicalTypeId::DOUBLE:
            return std::make_unique<cudf::numeric_scalar<double>>(
                value.GetValue<double>(), !value.IsNull(), se->stream);
        case duckdb::LogicalTypeId::BOOLEAN:
            return std::make_unique<cudf::numeric_scalar<bool>>(
                value.GetValue<bool>(), !value.IsNull(), se->stream);
        case duckdb::LogicalTypeId::VARCHAR:
            return std::make_unique<cudf::string_scalar>(
                value.GetValue<std::string>(), !value.IsNull(), se->stream);
        case duckdb::LogicalTypeId::TIMESTAMP:
        case duckdb::LogicalTypeId::TIMESTAMP_TZ: {
            // Define a timestamp type with microsecond precision
            duckdb::timestamp_t extracted =
                value.GetValue<duckdb::timestamp_t>();
            // Create a TimestampScalar with microsecond value
            return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_us>>(
                cudf::timestamp_us{
                    cuda::std::chrono::microseconds{extracted.value}},
                !value.IsNull(), se->stream);
        } break;
        case duckdb::LogicalTypeId::TIMESTAMP_MS: {
            // Define a timestamp type with millisecond precision
            duckdb::timestamp_ms_t extracted =
                value.GetValue<duckdb::timestamp_ms_t>();
            // Create a TimestampScalar with millisecond value
            return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ms>>(
                cudf::timestamp_ms{
                    cuda::std::chrono::milliseconds{extracted.value}},
                !value.IsNull(), se->stream);
        } break;
        case duckdb::LogicalTypeId::TIMESTAMP_SEC: {
            // Define a timestamp type with second precision
            duckdb::timestamp_sec_t extracted =
                value.GetValue<duckdb::timestamp_sec_t>();
            // Create a TimestampScalar with second value
            return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_s>>(
                cudf::timestamp_s{cuda::std::chrono::seconds{extracted.value}},
                !value.IsNull(), se->stream);
        } break;
        case duckdb::LogicalTypeId::TIMESTAMP_NS: {
            // Define a timestamp type with nanosecond precision
            duckdb::timestamp_ns_t extracted =
                value.GetValue<duckdb::timestamp_ns_t>();
            // Create a TimestampScalar with nanosecond value
            return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ns>>(
                cudf::timestamp_ns{
                    cuda::std::chrono::nanoseconds{extracted.value}},
                !value.IsNull(), se->stream);
        } break;
        case duckdb::LogicalTypeId::DATE: {
            // Define a date type
            duckdb::date_t extracted = value.GetValue<duckdb::date_t>();
            // Create a DateScalar with the date value
            return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
                cudf::timestamp_D{cuda::std::chrono::days{extracted.days}},
                !value.IsNull(), se->stream);
        } break;
        default:
            throw std::runtime_error("extractValue unhandled type." +
                                     std::to_string(static_cast<int>(type)));
    }
}

std::unique_ptr<cudf::column> make_literal_column(
    duckdb::Value const& v, cudf::size_type n_rows,
    std::shared_ptr<StreamAndEvent> se) {
    return cudf::make_column_from_scalar(*duckdbValueToCudfScalar(v, se),
                                         n_rows, se->stream);
}

std::unique_ptr<cudf::column> eval_cudf_expr(
    const CudfExpr& expr, cudf::table_view input,
    std::shared_ptr<StreamAndEvent> se) {
    using Kind = CudfExpr::Kind;

    switch (expr.kind) {
        case Kind::COLUMN_REF: {
            // Return a *copy* of the column as a new owning column
            return std::make_unique<cudf::column>(
                input.column(expr.column_index), se->stream);
        }

        case Kind::LITERAL: {
            return make_literal_column(expr.literal, input.num_rows(), se);
        }

        case Kind::EQ:
        case Kind::NE:
        case Kind::LT:
        case Kind::LE:
        case Kind::GT:
        case Kind::GE:
        case Kind::AND:
        case Kind::OR: {
            auto lhs_col = eval_cudf_expr(*expr.left, input, se);
            auto rhs_col = eval_cudf_expr(*expr.right, input, se);

            cudf::binary_operator op;
            switch (expr.kind) {
                case Kind::EQ:
                    op = cudf::binary_operator::EQUAL;
                    break;
                case Kind::NE:
                    op = cudf::binary_operator::NOT_EQUAL;
                    break;
                case Kind::LT:
                    op = cudf::binary_operator::LESS;
                    break;
                case Kind::LE:
                    op = cudf::binary_operator::LESS_EQUAL;
                    break;
                case Kind::GT:
                    op = cudf::binary_operator::GREATER;
                    break;
                case Kind::GE:
                    op = cudf::binary_operator::GREATER_EQUAL;
                    break;
                case Kind::AND:
                    op = cudf::binary_operator::LOGICAL_AND;
                    break;
                case Kind::OR:
                    op = cudf::binary_operator::LOGICAL_OR;
                    break;
                default:
                    throw std::runtime_error("Unexpected binary op");
            }

            auto result = cudf::binary_operation(
                lhs_col->view(), rhs_col->view(), op,
                cudf::data_type{cudf::type_id::BOOL8}, se->stream);
            return result;
        }
    }

    throw std::runtime_error("Unknown CudfExpr kind");
}

std::unique_ptr<CudfExpr> tableFilterToCudfExpr(
    duckdb::idx_t col_idx, duckdb::unique_ptr<duckdb::TableFilter>& tf) {
    using TF = duckdb::TableFilterType;

    switch (tf->filter_type) {
        case TF::CONSTANT_COMPARISON: {
            auto cf =
                dynamic_cast_unique_ptr<duckdb::ConstantFilter>(std::move(tf));
            auto cmp_kind = comparisonTypeToCudfKind(cf->comparison_type);
            auto col_ref = make_column_ref(static_cast<int>(col_idx));
            auto lit = make_literal(cf->constant);
            return make_binary(cmp_kind, std::move(col_ref), std::move(lit));
        } break;  // prevent fallthrough error

        case TF::CONJUNCTION_AND: {
            auto af = dynamic_cast_unique_ptr<duckdb::ConjunctionAndFilter>(
                std::move(tf));
            if (af->child_filters.size() < 2) {
                throw std::runtime_error("AND filter with <2 children");
            }
            auto expr = tableFilterToCudfExpr(col_idx, af->child_filters[0]);
            for (std::size_t i = 1; i < af->child_filters.size(); ++i) {
                expr = make_and(
                    std::move(expr),
                    tableFilterToCudfExpr(col_idx, af->child_filters[i]));
            }
            return expr;
        } break;  // prevent fallthrough error

        case TF::CONJUNCTION_OR: {
            auto of = dynamic_cast_unique_ptr<duckdb::ConjunctionOrFilter>(
                std::move(tf));
            if (of->child_filters.size() < 2) {
                throw std::runtime_error("OR filter with <2 children");
            }
            auto expr = tableFilterToCudfExpr(col_idx, of->child_filters[0]);
            for (std::size_t i = 1; i < of->child_filters.size(); ++i) {
                expr = make_or(
                    std::move(expr),
                    tableFilterToCudfExpr(col_idx, of->child_filters[i]));
            }
            return expr;
        } break;  // prevent fallthrough error

        case TF::OPTIONAL_FILTER: {
            auto of =
                dynamic_cast_unique_ptr<duckdb::OptionalFilter>(std::move(tf));
            try {
                return tableFilterToCudfExpr(col_idx, of->child_filter);
            } catch (...) {
                // No-op: literal true
                return make_literal(duckdb::Value::BOOLEAN(true));
            }
        } break;  // prevent fallthrough error

        default:
            throw std::runtime_error("Unsupported TableFilterType");
    }
}

std::unique_ptr<CudfExpr> tableFilterSetToCudf(
    duckdb::TableFilterSet& filters, const std::map<int, int>& column_map) {
    std::unique_ptr<CudfExpr> root_expr;

    if (filters.filters.empty()) {
        return nullptr;
    }

    // Combine all filters with AND
    for (auto& pair : filters.filters) {
        duckdb::idx_t col_idx = pair.first;
        auto column_map_iter = column_map.find(col_idx);
        // If there are selected columns then use the column map to
        // get the mapping from the TableFilterSet column number in
        // col_idx to what column it will be after columns are
        // selected via cudf.
        if (column_map_iter != column_map.end()) {
            col_idx = column_map_iter->second;
        } else {
            throw std::runtime_error(
                "tableFilterSetToCudf(): col_idx not found in column_map");
        }
        auto& tf = pair.second;

        auto sub_expr = tableFilterToCudfExpr(col_idx, tf);

        if (!root_expr) {
            root_expr = std::move(sub_expr);
        } else {
            root_expr = make_and(std::move(root_expr), std::move(sub_expr));
        }
    }

    return root_expr;
}

std::unique_ptr<cudf::table> CudfExpr::eval(
    cudf::table& input, std::shared_ptr<StreamAndEvent> se) {
    // Evaluate expression to a BOOL8 mask column
    auto mask_col = eval_cudf_expr(*this, input.view(), se);

    // Apply boolean mask
    auto filtered =
        cudf::apply_boolean_mask(input.view(), mask_col->view(), se->stream);
    return filtered;
}
