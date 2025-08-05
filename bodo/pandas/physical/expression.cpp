#include "expression.h"
#include <iostream>
#include "_util.h"

std::shared_ptr<arrow::Array> prepare_arrow_compute(
    std::shared_ptr<array_info> arr) {
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
    return bodo_array_to_arrow(bodo::BufferPool::DefaultPtr(), arr,
                               false /*convert_timedelta_to_int64*/, "",
                               time_unit, false, /*downcast_time_ns_to_us*/
                               bodo::default_buffer_memory_manager());
}

// String specialization
std::shared_ptr<arrow::Array> ScalarToArrowArray(const std::string& value,
                                                 size_t num_elements) {
    arrow::StringBuilder builder;
    arrow::Status status;
    for (size_t i = 0; i < num_elements; ++i) {
        status = builder.Append(value);
        if (!status.ok()) {
            throw std::runtime_error("builder.Append failed.");
        }
    }
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }
    return array;
}

std::shared_ptr<arrow::Array> ScalarToArrowArray(
    const std::shared_ptr<arrow::Scalar>& value, size_t num_elements) {
    arrow::Result<std::shared_ptr<arrow::Array>> array_result =
        arrow::MakeArrayFromScalar(*value, num_elements);
    if (!array_result.ok()) {
        throw std::runtime_error("MakeArrayFromScalar failed: " +
                                 array_result.status().message());
    }
    return array_result.ValueOrDie();
}

std::shared_ptr<arrow::Array> ScalarToArrowArray(bool value,
                                                 size_t num_elements) {
    arrow::BooleanBuilder builder;
    arrow::Status status;

    for (size_t i = 0; i < num_elements; ++i) {
        // Append boolean value
        status = builder.Append(value);
        if (!status.ok()) {
            throw std::runtime_error("builder.Append failed.");
        }
    }

    // Finalize the Arrow array
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }

    return array;
}

// String specialization
std::shared_ptr<arrow::Array> NullArrowArray(const std::string& value,
                                             size_t num_elements) {
    arrow::StringBuilder builder;
    arrow::Status status;
    status = builder.AppendNulls(num_elements);
    if (!status.ok()) {
        throw std::runtime_error("builder.AppendNulls failed.");
    }
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }
    return array;
}

std::shared_ptr<arrow::Array> NullArrowArray(
    const std::shared_ptr<arrow::Scalar>& value, size_t num_elements) {
    arrow::Result<std::shared_ptr<arrow::Array>> array_result =
        arrow::MakeArrayOfNull(value->type, num_elements);
    if (!array_result.ok()) {
        throw std::runtime_error("MakeArrayFromScalar failed: " +
                                 array_result.status().message());
    }
    return array_result.ValueOrDie();
}

std::shared_ptr<arrow::Array> NullArrowArray(bool value, size_t num_elements) {
    arrow::BooleanBuilder builder;
    arrow::Status status;

    status = builder.AppendNulls(num_elements);
    if (!status.ok()) {
        throw std::runtime_error("builder.AppendNulls failed.");
    }

    // Finalize the Arrow array
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }

    return array;
}

std::shared_ptr<array_info> do_arrow_compute_binary(
    std::shared_ptr<ExprResult> left_res, std::shared_ptr<ExprResult> right_res,
    const std::string& comparator) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprResult> left_as_array =
        std::dynamic_pointer_cast<ArrayExprResult>(left_res);
    std::shared_ptr<ScalarExprResult> left_as_scalar =
        std::dynamic_pointer_cast<ScalarExprResult>(left_res);
    std::shared_ptr<ArrayExprResult> right_as_array =
        std::dynamic_pointer_cast<ArrayExprResult>(right_res);
    std::shared_ptr<ScalarExprResult> right_as_scalar =
        std::dynamic_pointer_cast<ScalarExprResult>(right_res);

    arrow::Datum src1;
    if (left_as_array) {
        src1 = arrow::Datum(prepare_arrow_compute(left_as_array->result));
    } else if (left_as_scalar) {
        src1 = arrow::MakeScalar(prepare_arrow_compute(left_as_scalar->result)
                                     ->GetScalar(0)
                                     .ValueOrDie());
    } else {
        throw std::runtime_error(
            "do_arrow_compute left is neither array nor scalar.");
    }

    arrow::Datum src2;
    if (right_as_array) {
        src2 = arrow::Datum(prepare_arrow_compute(right_as_array->result));
    } else if (right_as_scalar) {
        src2 = arrow::MakeScalar(prepare_arrow_compute(right_as_scalar->result)
                                     ->GetScalar(0)
                                     .ValueOrDie());
    } else {
        throw std::runtime_error(
            "do_arrow_compute right is neither array nor scalar.");
    }

    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::CallFunction(comparator, {src1, src2});
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_array_compute_binary: Error in Arrow compute: " +
            cmp_res.status().message());
    }

    return arrow_array_to_bodo(cmp_res.ValueOrDie().make_array(),
                               bodo::BufferPool::DefaultPtr());
}

std::shared_ptr<array_info> do_arrow_compute_unary(
    std::shared_ptr<ExprResult> left_res, const std::string& comparator) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprResult> left_as_array =
        std::dynamic_pointer_cast<ArrayExprResult>(left_res);
    std::shared_ptr<ScalarExprResult> left_as_scalar =
        std::dynamic_pointer_cast<ScalarExprResult>(left_res);

    arrow::Datum src1;
    if (left_as_array) {
        src1 = arrow::Datum(prepare_arrow_compute(left_as_array->result));
    } else if (left_as_scalar) {
        src1 = arrow::MakeScalar(prepare_arrow_compute(left_as_scalar->result)
                                     ->GetScalar(0)
                                     .ValueOrDie());
    } else {
        throw std::runtime_error(
            "do_arrow_compute left is neither array nor scalar.");
    }

    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::CallFunction(comparator, {src1});
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_array_compute_unary: Error in Arrow compute: " +
            cmp_res.status().message());
    }

    return arrow_array_to_bodo(cmp_res.ValueOrDie().make_array(),
                               bodo::BufferPool::DefaultPtr());
}

std::shared_ptr<array_info> do_arrow_compute_cast(
    std::shared_ptr<ExprResult> left_res,
    const duckdb::LogicalType& return_type) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprResult> left_as_array =
        std::dynamic_pointer_cast<ArrayExprResult>(left_res);
    std::shared_ptr<ScalarExprResult> left_as_scalar =
        std::dynamic_pointer_cast<ScalarExprResult>(left_res);

    arrow::Datum src1;
    if (left_as_array) {
        src1 = arrow::Datum(prepare_arrow_compute(left_as_array->result));
    } else if (left_as_scalar) {
        src1 = arrow::MakeScalar(prepare_arrow_compute(left_as_scalar->result)
                                     ->GetScalar(0)
                                     .ValueOrDie());
    } else {
        throw std::runtime_error(
            "do_arrow_compute left is neither array nor scalar.");
    }

    std::shared_ptr<arrow::DataType> arrow_ret_type =
        duckdbTypeToArrow(return_type);
    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::Cast(src1, arrow_ret_type);
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_array_compute_cast: Error in Arrow compute: " +
            cmp_res.status().message());
    }

    return arrow_array_to_bodo(cmp_res.ValueOrDie().make_array(),
                               bodo::BufferPool::DefaultPtr());
}

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

std::shared_ptr<PhysicalExpression> buildPhysicalExprTree(
    duckdb::unique_ptr<duckdb::Expression>& expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>& col_ref_map,
    bool no_scalars) {
    // Class and type here are really like the general type of the
    // expression node (expr_class) and a sub-type of that general
    // type (expr_type).
    duckdb::ExpressionClass expr_class = expr->GetExpressionClass();
    duckdb::ExpressionType expr_type = expr->GetExpressionType();

    switch (expr_class) {
        case duckdb::ExpressionClass::BOUND_COMPARISON: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr->Cast<duckdb::BoundComparisonExpression>();
            // This node type has left and right children which are recursively
            // processed first and then the resulting Bodo Physical expression
            // subtrees are combined with the expression sub-type (e.g., equal,
            // greater_than, less_than) to make the Bodo PhysicalComparisonExpr.
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalComparisonExpression>(
                    buildPhysicalExprTree(bce.left, col_ref_map, no_scalars),
                    buildPhysicalExprTree(bce.right, col_ref_map, no_scalars),
                    expr_type));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_COLUMN_REF: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr->Cast<duckdb::BoundColumnRefExpression>();
            duckdb::ColumnBinding binding = bce.binding;
            size_t col_idx =
                col_ref_map[{binding.table_index, binding.column_index}];
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalColumnRefExpression>(col_idx,
                                                              bce.GetName()));
            // binding.table_index, binding.column_index));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_CONSTANT: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr->Cast<duckdb::BoundConstantExpression>();
            if (bce.value.IsNull()) {
                // Get the constant out of the duckdb node as a C++ variant.
                // Using auto since variant set will be extended.
                auto extracted_value =
                    getDefaultValueForDuckdbValueType(bce.value);
                // Return a PhysicalConstantExpression<T> where T is the actual
                // type of the value contained within bce.value.
                auto ret = std::visit(
                    [no_scalars](const auto& value) {
                        return std::static_pointer_cast<PhysicalExpression>(
                            std::make_shared<PhysicalNullExpression<
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
                        return std::static_pointer_cast<PhysicalExpression>(
                            std::make_shared<PhysicalConstantExpression<
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
            auto& bce = expr->Cast<duckdb::BoundConjunctionExpression>();
            // This node type has left and right children which are recursively
            // processed first and then the resulting Bodo Physical expression
            // subtrees are combined with the expression sub-type (e.g., equal,
            // greater_than, less_than) to make the Bodo PhysicalComparisonExpr.
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalConjunctionExpression>(
                    buildPhysicalExprTree(bce.children[0], col_ref_map,
                                          no_scalars),
                    buildPhysicalExprTree(bce.children[1], col_ref_map,
                                          no_scalars),
                    expr_type));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_OPERATOR: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& boe = expr->Cast<duckdb::BoundOperatorExpression>();
            switch (boe.children.size()) {
                case 1: {
                    return std::static_pointer_cast<PhysicalExpression>(
                        std::make_shared<PhysicalUnaryExpression>(
                            buildPhysicalExprTree(boe.children[0], col_ref_map,
                                                  no_scalars),
                            expr_type));
                } break;
                case 2: {
                    return std::static_pointer_cast<PhysicalExpression>(
                        std::make_shared<PhysicalBinaryExpression>(
                            buildPhysicalExprTree(boe.children[0], col_ref_map,
                                                  no_scalars),
                            buildPhysicalExprTree(boe.children[1], col_ref_map,
                                                  no_scalars),
                            expr_type));
                } break;
                default:
                    throw std::runtime_error(
                        "Unsupported number of children for bound operator");
            }
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_FUNCTION: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            printf("In buildPhysicalExprTree\n");
            auto& bfe = expr->Cast<duckdb::BoundFunctionExpression>();
            if (bfe.bind_info) {
                BodoScalarFunctionData& scalar_func_data =
                    bfe.bind_info->Cast<BodoScalarFunctionData>();

                std::vector<std::shared_ptr<PhysicalExpression>> phys_children;
                for (auto& child_expr : bfe.children) {
                    phys_children.emplace_back(buildPhysicalExprTree(
                        child_expr, col_ref_map, no_scalars));
                }

                const std::shared_ptr<arrow::DataType>& result_type =
                    scalar_func_data.out_schema->field(0)->type();

                if (!scalar_func_data.arrow_func_name.empty()) {
                    return std::static_pointer_cast<PhysicalExpression>(
                        std::make_shared<PhysicalArrowExpression>(
                            phys_children, scalar_func_data, result_type));
                } else if (scalar_func_data.args) {
                    return std::static_pointer_cast<PhysicalExpression>(
                        std::make_shared<PhysicalUDFExpression>(
                            phys_children, scalar_func_data, result_type));
                }
            } else {
                switch (bfe.children.size()) {
                    case 1: {
                        return std::static_pointer_cast<PhysicalExpression>(
                            std::make_shared<PhysicalUnaryExpression>(
                                buildPhysicalExprTree(bfe.children[0],
                                                      col_ref_map, no_scalars),
                                bfe.function.name));
                    } break;
                    case 2: {
                        return std::static_pointer_cast<PhysicalExpression>(
                            std::make_shared<PhysicalBinaryExpression>(
                                buildPhysicalExprTree(bfe.children[0],
                                                      col_ref_map, no_scalars),
                                buildPhysicalExprTree(bfe.children[1],
                                                      col_ref_map, no_scalars),
                                bfe.function.name));
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
            auto& bce = expr->Cast<duckdb::BoundCastExpression>();
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalCastExpression>(
                    buildPhysicalExprTree(bce.child, col_ref_map, no_scalars),
                    bce.return_type));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_BETWEEN: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bbe = expr->Cast<duckdb::BoundBetweenExpression>();
            // Convert to conjunction and comparison nodes.
            std::shared_ptr<PhysicalExpression> input_expr =
                buildPhysicalExprTree(bbe.input, col_ref_map, no_scalars);
            std::shared_ptr<PhysicalExpression> lower_expr =
                buildPhysicalExprTree(bbe.lower, col_ref_map, no_scalars);
            std::shared_ptr<PhysicalExpression> upper_expr =
                buildPhysicalExprTree(bbe.upper, col_ref_map, no_scalars);

            std::shared_ptr<PhysicalExpression> left = std::static_pointer_cast<
                PhysicalExpression>(
                std::make_shared<PhysicalComparisonExpression>(
                    input_expr, lower_expr,
                    bbe.lower_inclusive
                        ? duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO
                        : duckdb::ExpressionType::COMPARE_GREATERTHAN));

            std::shared_ptr<PhysicalExpression> right =
                std::static_pointer_cast<PhysicalExpression>(
                    std::make_shared<PhysicalComparisonExpression>(
                        upper_expr, input_expr,
                        bbe.upper_inclusive
                            ? duckdb::ExpressionType::
                                  COMPARE_GREATERTHANOREQUALTO
                            : duckdb::ExpressionType::COMPARE_GREATERTHAN));

            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalConjunctionExpression>(
                    left, right, duckdb::ExpressionType::CONJUNCTION_AND));
        } break;  // suppress wrong fallthrough error
        default:
            throw std::runtime_error(
                "Unsupported duckdb expression class " +
                std::to_string(static_cast<int>(expr_class)));
    }
    throw std::logic_error("Control should never reach here");
}

std::shared_ptr<ExprResult> PhysicalUDFExpression::ProcessBatch(
    std::shared_ptr<table_info> input_batch) {
    std::vector<std::shared_ptr<array_info>> child_results;
    std::vector<std::string> column_names;

    // All the sources of the UDF will be separate projections.
    // Create each one of them here.
    for (const auto& child : children) {
        std::shared_ptr<ExprResult> child_res =
            child->ProcessBatch(input_batch);

        std::shared_ptr<ArrayExprResult> child_as_array =
            std::dynamic_pointer_cast<ArrayExprResult>(child_res);
        if (!child_as_array) {
            throw std::runtime_error("Child of UDF did not return an array.");
        }
        child_results.emplace_back(child_as_array->result);
        column_names.emplace_back(child_as_array->column_name);
    }
    // Put them all back together for the UDF to process.
    std::shared_ptr<table_info> udf_input = std::make_shared<table_info>(
        child_results, column_names, input_batch->metadata);

    // Actually run the UDF.
    std::shared_ptr<table_info> udf_output;
    if (cfunc_ptr) {
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

    return std::make_shared<ArrayExprResult>(udf_output->columns[0],
                                             udf_output->column_names[0]);
}

std::shared_ptr<ExprResult> PhysicalArrowExpression::ProcessBatch(
    std::shared_ptr<table_info> input_batch) {
    std::shared_ptr<ExprResult> res = children[0]->ProcessBatch(input_batch);
    printf("%s\n", scalar_func_data.arrow_func_name.c_str());
    auto result = do_arrow_compute_unary(res, scalar_func_data.arrow_func_name);
    return std::make_shared<ArrayExprResult>(result, "Arrow Scalar");
}

bool PhysicalExpression::join_expr(array_info** left_table,
                                   array_info** right_table, void** left_data,
                                   void** right_data, void** left_null_bitmap,
                                   void** right_null_bitmap, int64_t left_index,
                                   int64_t right_index) {
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
}

PhysicalExpression* PhysicalExpression::cur_join_expr = nullptr;
