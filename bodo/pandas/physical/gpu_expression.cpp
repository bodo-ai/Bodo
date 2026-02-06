#include "gpu_expression.h"
#include "_util.h"

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
