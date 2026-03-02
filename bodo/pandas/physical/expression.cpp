#include "expression.h"
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
    arrow::Result<std::shared_ptr<arrow::Array>> array_result;
    if (value == nullptr || value->is_valid == false) {
        array_result = arrow::MakeArrayOfNull(
            value ? value->type : arrow::null(), num_elements);
    } else {
        array_result = arrow::MakeArrayFromScalar(*value, num_elements);
    }
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

/**
 * @brief Change nulls in arrow Datum to given val.
 *
 */
arrow::Datum fill_null(arrow::Datum& src, arrow::Datum& val) {
    if (src.is_array() || src.is_chunked_array()) {
        auto mask_result = arrow::compute::IsNull(src);
        if (!mask_result.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_binary: Error in Arrow compute: " +
                mask_result.status().message());
        }
        arrow::Datum mask = mask_result.ValueOrDie();

        arrow::Result<arrow::Datum> src_res =
            arrow::compute::ReplaceWithMask(src, mask, val);
        if (!src_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_binary: Error in Arrow compute: " +
                src_res.status().message());
        }
        return src_res.ValueOrDie();
    } else if (src.is_scalar()) {
        auto scalar = src.scalar();
        if (scalar->is_valid) {
            return src;
        } else {
            return val;
        }
    } else {
        throw std::runtime_error(
            "fill_null can't handle non-array or scalar datum type.");
    }
}

/**
 * @brief Returns true if arrow datatype can hold a NaN.
 *
 */
inline bool canHoldNan(std::shared_ptr<arrow::DataType> type) {
    return type->id() == arrow::Type::HALF_FLOAT ||
           type->id() == arrow::Type::FLOAT ||
           type->id() == arrow::Type::DOUBLE;
}

arrow::Datum MakeNanScalar(const std::shared_ptr<arrow::DataType>& dtype) {
    arrow::Result<std::shared_ptr<arrow::Scalar>> res;

    if (dtype->id() == arrow::Type::FLOAT) {
        res = arrow::MakeScalar(dtype, static_cast<float>(std::nan("")));
    } else if (dtype->id() == arrow::Type::DOUBLE) {
        res = arrow::MakeScalar(dtype, static_cast<double>(std::nan("")));
    } else if (dtype->id() == arrow::Type::HALF_FLOAT) {
        res = arrow::MakeScalar(dtype, static_cast<float>(std::nan("")));
    } else {
        throw std::runtime_error("DataType does not support NaN");
    }

    if (!res.ok()) {
        throw std::runtime_error("MakeScalar failed: " +
                                 res.status().ToString());
    }

    return arrow::Datum(res.ValueOrDie());
}

std::shared_ptr<array_info> do_arrow_compute_binary(
    std::shared_ptr<ExprResult> left_res, std::shared_ptr<ExprResult> right_res,
    const std::string& comparator,
    const std::shared_ptr<arrow::DataType> result_type) {
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

    std::shared_ptr<arrow::DataType> src1_dtype = src1.type();
    std::shared_ptr<arrow::DataType> src2_dtype = src2.type();

    // If type is float then match Pandas and convert NA to NaN.
    if (canHoldNan(src1_dtype)) {
        arrow::Datum null_scalar = MakeNanScalar(src1_dtype);
        src1 = fill_null(src1, null_scalar);
    }
    if (canHoldNan(src2_dtype)) {
        arrow::Datum null_scalar = MakeNanScalar(src2_dtype);
        src2 = fill_null(src2, null_scalar);
    }

    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::CallFunction(comparator, {src1, src2});
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_arrow_compute_binary cmp_res: Error in Arrow compute: " +
            cmp_res.status().message());
    }
    auto cmp_datum = cmp_res.ValueOrDie();

    std::shared_ptr<arrow::DataType> cmp_dtype = cmp_datum.type();
    // Bodo checks is_na with NULL but NaN in Pandas considered NA
    // so convert all NaN into NA.
    if (canHoldNan(cmp_dtype)) {
        // Convert NaN into NULLs.
        auto mask_result = arrow::compute::IsNan(cmp_datum);
        if (!mask_result.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_binary mask_result: Error in Arrow "
                "compute: " +
                mask_result.status().message());
        }
        arrow::Datum mask = mask_result.ValueOrDie();

        auto null_scalar = arrow::MakeNullScalar(cmp_datum.type());

        cmp_res = arrow::compute::ReplaceWithMask(cmp_datum, mask,
                                                  arrow::Datum(null_scalar));
        if (!cmp_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_binary post replacewithmask: Error in Arrow "
                "compute: " +
                cmp_res.status().message());
        }
    }

    if (result_type && cmp_dtype != result_type) {
        // Cast to result type if available and different from current type.
        arrow::Result<arrow::Datum> cast_res =
            arrow::compute::Cast(cmp_datum, result_type);
        if (!cast_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_binary cast_res: Error in Arrow compute: " +
                cast_res.status().message());
        }
        cmp_res = cast_res;
    }

    std::shared_ptr<arrow::Array> arrow_arr = cmp_res.ValueOrDie().make_array();
    return arrow_array_to_bodo(arrow_arr, bodo::BufferPool::DefaultPtr());
}

std::shared_ptr<array_info> do_arrow_compute_unary(
    std::shared_ptr<ExprResult> left_res, const std::string& comparator,
    const arrow::compute::FunctionOptions* func_options) {
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

    // Special handling for is_not_null since it is not supported directly
    // by Arrow compute.
    if (comparator == "is_not_null") {
        arrow::Result<arrow::Datum> is_null_res =
            arrow::compute::CallFunction("is_null", {src1}, func_options);
        if (!is_null_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_array_compute_unary: Error in Arrow compute: " +
                is_null_res.status().message());
        }

        // Invert the boolean array
        arrow::Result<arrow::Datum> invert_res =
            arrow::compute::CallFunction("invert", {is_null_res.ValueOrDie()});
        if (!invert_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_array_compute_unary: Error in Arrow compute Invert: " +
                invert_res.status().message());
        }
        return arrow_array_to_bodo(invert_res.ValueOrDie().make_array(),
                                   bodo::BufferPool::DefaultPtr());
    }

    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::CallFunction(comparator, {src1}, func_options);
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

std::shared_ptr<array_info> do_arrow_compute_case(
    std::shared_ptr<ExprResult> when_res, std::shared_ptr<ExprResult> then_res,
    std::shared_ptr<ExprResult> else_res) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprResult> when_as_array =
        std::dynamic_pointer_cast<ArrayExprResult>(when_res);
    std::shared_ptr<ScalarExprResult> when_as_scalar =
        std::dynamic_pointer_cast<ScalarExprResult>(when_res);
    std::shared_ptr<ArrayExprResult> then_as_array =
        std::dynamic_pointer_cast<ArrayExprResult>(then_res);
    std::shared_ptr<ScalarExprResult> then_as_scalar =
        std::dynamic_pointer_cast<ScalarExprResult>(then_res);
    std::shared_ptr<ArrayExprResult> else_as_array =
        std::dynamic_pointer_cast<ArrayExprResult>(else_res);
    std::shared_ptr<ScalarExprResult> else_as_scalar =
        std::dynamic_pointer_cast<ScalarExprResult>(else_res);

    arrow::Datum src1;
    if (when_as_array) {
        std::shared_ptr<arrow::Array> arr =
            prepare_arrow_compute(when_as_array->result);

        src1 = arrow::Datum(arr);
    } else if (when_as_scalar) {
        src1 = arrow::MakeScalar(prepare_arrow_compute(when_as_scalar->result)
                                     ->GetScalar(0)
                                     .ValueOrDie());
    } else {
        throw std::runtime_error(
            "do_arrow_compute when is neither array nor scalar.");
    }

    arrow::Datum src2;
    if (then_as_array) {
        src2 = arrow::Datum(prepare_arrow_compute(then_as_array->result));
    } else if (then_as_scalar) {
        src2 = arrow::MakeScalar(prepare_arrow_compute(then_as_scalar->result)
                                     ->GetScalar(0)
                                     .ValueOrDie());
    } else {
        throw std::runtime_error(
            "do_arrow_compute then is neither array nor scalar.");
    }

    arrow::Datum src3;
    if (else_as_array) {
        src3 = arrow::Datum(prepare_arrow_compute(else_as_array->result));
    } else if (else_as_scalar) {
        src3 = arrow::MakeScalar(prepare_arrow_compute(else_as_scalar->result)
                                     ->GetScalar(0)
                                     .ValueOrDie());
    } else {
        throw std::runtime_error(
            "do_arrow_compute else is neither array nor scalar.");
    }

    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::CallFunction("if_else", {src1, src2, src3});
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_array_compute_case: Error in Arrow compute: " +
            cmp_res.status().message());
    }

    return arrow_array_to_bodo(cmp_res.ValueOrDie().make_array(),
                               bodo::BufferPool::DefaultPtr());
}

std::shared_ptr<PhysicalExpression> buildPhysicalExprTree(
    duckdb::unique_ptr<duckdb::Expression>& expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>& col_ref_map,
    bool no_scalars);

std::shared_ptr<PhysicalExpression> buildPhysicalExprTree(
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
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalComparisonExpression>(
                    buildPhysicalExprTree(bce.left, col_ref_map, no_scalars),
                    buildPhysicalExprTree(bce.right, col_ref_map, no_scalars),
                    expr_type));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_COLUMN_REF: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundColumnRefExpression>();
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
            auto& bce = expr.Cast<duckdb::BoundConjunctionExpression>();
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
            auto& boe = expr.Cast<duckdb::BoundOperatorExpression>();
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

                std::vector<std::shared_ptr<PhysicalExpression>> phys_children;
                for (auto& child_expr : bfe.children) {
                    phys_children.emplace_back(buildPhysicalExprTree(
                        child_expr, col_ref_map, no_scalars));
                }

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
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalCastExpression>(
                    buildPhysicalExprTree(bce.child, col_ref_map, no_scalars),
                    bce.return_type));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_BETWEEN: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bbe = expr.Cast<duckdb::BoundBetweenExpression>();
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
        case duckdb::ExpressionClass::BOUND_CASE: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundCaseExpression>();
            if (bce.case_checks.size() != 1) {
                throw std::runtime_error(
                    "Only single WHEN case expressions are supported.");
            }
            auto& caseCheck = bce.case_checks[0];
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalCaseExpression>(
                    buildPhysicalExprTree(caseCheck.when_expr, col_ref_map,
                                          no_scalars),
                    buildPhysicalExprTree(caseCheck.then_expr, col_ref_map,
                                          no_scalars),
                    buildPhysicalExprTree(bce.else_expr, col_ref_map,
                                          no_scalars)));
        } break;  // suppress wrong fallthrough error
        default:
            throw std::runtime_error(
                "Unsupported duckdb expression class " +
                std::to_string(static_cast<int>(expr_class)));
    }
    throw std::logic_error("Control should never reach here");
}

std::shared_ptr<PhysicalExpression> buildPhysicalExprTree(
    duckdb::unique_ptr<duckdb::Expression>& expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>& col_ref_map,
    bool no_scalars) {
    return buildPhysicalExprTree(*expr, col_ref_map, no_scalars);
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
        std::shared_ptr<ScalarExprResult> child_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(child_res);

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
    std::shared_ptr<table_info> udf_input = std::make_shared<table_info>(
        child_results, column_names, input_batch->metadata);

    // Actually run the UDF.
    std::shared_ptr<table_info> udf_output;
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

    return std::make_shared<ArrayExprResult>(udf_output->columns[0],
                                             udf_output->column_names[0]);
}

std::shared_ptr<ExprResult> PhysicalArrowExpression::ProcessBatch(
    std::shared_ptr<table_info> input_batch) {
    std::shared_ptr<ExprResult> res = children[0]->ProcessBatch(input_batch);
    std::shared_ptr<array_info> result;
    time_pt start_init_time = start_timer();
    if (scalar_func_data.arrow_func_name == "date") {
        // The Arrow compute equivalent of Series.dt.date() is year_month_day,
        // which returns a struct. To match the output dtype of Pandas, we Cast
        // to Date32 instead.
        result = do_arrow_compute_cast(res, duckdb::LogicalType::DATE);
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
        result = do_arrow_compute_unary(res, scalar_func_data.arrow_func_name,
                                        &opts);
    } else {
        result = do_arrow_compute_unary(res, scalar_func_data.arrow_func_name);
    }
    this->metrics.arrow_compute_time += end_timer(start_init_time);
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

void PhysicalExpression::join_expr_batch(
    array_info** left_table, array_info** right_table, void** left_data,
    void** right_data, void** left_null_bitmap, void** right_null_bitmap,
    uint8_t* match_arr, int64_t left_index_start, int64_t left_index_end,
    int64_t right_index_start, int64_t right_index_end) {
    for (int64_t j = right_index_start; j < right_index_end; j++) {
        for (int64_t i = left_index_start; i < left_index_end; i++) {
            SetBitTo(match_arr,
                     (i - left_index_start) + (j - right_index_start),
                     join_expr(left_table, right_table, left_data, right_data,
                               left_null_bitmap, right_null_bitmap, i, j));
        }
    }
}

PhysicalExpression* PhysicalExpression::cur_join_expr = nullptr;

template <typename ArrowType, typename ModOp>
arrow::Status ModImpl(arrow::compute::KernelContext* ctx,
                      const arrow::compute::ExecSpan& batch,
                      arrow::compute::ExecResult* out) {
    using CType = typename ArrowType::c_type;
    using ScalarType = typename arrow::TypeTraits<ArrowType>::ScalarType;

    // Extract left array (it has to be an array).
    const arrow::ArraySpan& left = batch[0].array;
    // Extract right element...could be scalar or array.
    const arrow::compute::ExecValue& right_span = batch[1];

    // Make sure it's an array.
    if (!left.type) {
        throw std::runtime_error("ModInt left.type not valid.");
    }

    arrow::Status status;
    // Get raw pointers to values and null bits for left array.
    const CType* left_values = left.GetValues<CType>(1);
    const uint8_t* left_valid_bits = left.buffers[0].data;

    auto is_valid_bit = [](const uint8_t* bits, int64_t offset,
                           int64_t i) -> bool {
        return !bits || arrow::bit_util::GetBit(bits, offset + i);
    };

    // Output array is preallocated and comes in as an ArraySpan in the
    // out->value variant.
    arrow::ArraySpan& out_span = std::get<arrow::ArraySpan>(out->value);
    // Get raw pointers to values and null bits of the output array.
    CType* out_values = out_span.GetValues<CType>(1);
    uint8_t* out_valid_bits = out_span.buffers[0].data;
    int64_t offset = out_span.offset;

    auto set_valid = [](uint8_t* bits, int64_t i) {
        if (bits)
            arrow::bit_util::SetBit(bits, i);
    };

    auto clear_valid = [](uint8_t* bits, int64_t i) {
        if (bits)
            arrow::bit_util::ClearBit(bits, i);
    };

    if (right_span.is_scalar()) {
        // Right side is a scalar
        const arrow::Scalar* scalar = right_span.scalar;
        if (!scalar || !scalar->is_valid) {
            // If scalar is null, all outputs are null
            for (int64_t i = 0; i < left.length; ++i) {
                clear_valid(out_valid_bits, offset + i);
            }
        } else {
            // Get right value as a scalar.
            const ScalarType& sc = right_span.scalar_as<ScalarType>();
            // Extract value from the scalar.
            CType r = sc.value;
            // For each element of the left array.
            for (int64_t i = 0; i < left.length; ++i) {
                if (!is_valid_bit(left_valid_bits, left.offset, i)) {
                    clear_valid(out_valid_bits, offset + i);
                } else {
                    // Get ith element of left.
                    CType l = left_values[i];
                    // Calculate modulus operator.
                    CType res = ModOp::apply(l, r);
                    // Assign result.
                    out_values[i] = res;
                    // Indicate index has valid data.
                    set_valid(out_valid_bits, offset + i);
                }
            }
        }
    } else {
        // Right side is an array so extract it.
        const arrow::ArraySpan& right = right_span.array;
        // Get raw pointers to values and null bits for right array.
        const CType* right_values = right.GetValues<CType>(1);
        const uint8_t* right_valid_bits = right.buffers[0].data;
        // For each element of the left array.
        for (int64_t i = 0; i < left.length; ++i) {
            if (!is_valid_bit(left_valid_bits, left.offset, i) ||
                !is_valid_bit(right_valid_bits, right.offset, i)) {
                clear_valid(out_valid_bits, offset + i);
            } else {
                // Get corresponding ith elements of left and right arrays.
                CType l = left_values[i];
                CType r = right_values[i];
                // Calculate modulus operator.
                CType res = ModOp::apply(l, r);
                // Assign result.
                out_values[i] = res;
                // Indicate index has valid data.
                set_valid(out_valid_bits, offset + i);
            }
        }
    }

    return arrow::Status::OK();
}

struct NativeMod {
    template <typename T>
    static T apply(T l, T r) {
        return (r == 0 ? 0 : (l % r));
    }
};

struct AltMod {
    template <typename T>
    static T apply(T l, T r) {
        return (r == 0 ? 0 : (l - ((int64_t)(l / r) * r)));
    }
};

void RegisterMod(arrow::compute::FunctionRegistry* registry) {
    // Declare the binary arrow compute function named "bodo_mod".
    auto func = std::make_shared<arrow::compute::ScalarFunction>(
        "bodo_mod", arrow::compute::Arity::Binary(),
        arrow::compute::FunctionDoc{
            "Modulo of two arrays", "Returns lhs % rhs", {"lhs", "rhs"}});

    // Declare int32,int32->int32 mod kernel.
    arrow::compute::ScalarKernel kernel32(
        {arrow::compute::InputType(arrow::int32()),
         arrow::compute::InputType(arrow::int32())},
        arrow::compute::OutputType(arrow::int32()),
        ModImpl<arrow::Int32Type, NativeMod>);
    // Declare int64,int64->int64 mod kernel.
    arrow::compute::ScalarKernel kernel64(
        {arrow::compute::InputType(arrow::int64()),
         arrow::compute::InputType(arrow::int64())},
        arrow::compute::OutputType(arrow::int64()),
        ModImpl<arrow::Int64Type, NativeMod>);
    // Declare float,float->float mod kernel.
    arrow::compute::ScalarKernel floatkernel32(
        {arrow::compute::InputType(arrow::float32()),
         arrow::compute::InputType(arrow::float32())},
        arrow::compute::OutputType(arrow::float32()),
        ModImpl<arrow::FloatType, AltMod>);
    // Declare double,double->double mod kernel.
    arrow::compute::ScalarKernel floatkernel64(
        {arrow::compute::InputType(arrow::float64()),
         arrow::compute::InputType(arrow::float64())},
        arrow::compute::OutputType(arrow::float64()),
        ModImpl<arrow::DoubleType, AltMod>);

    arrow::Status status;
    // Add all the above kernels to the function.
    status = func->AddKernel(kernel32);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod 32 AddKernel failed.");
    }
    status = func->AddKernel(kernel64);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod 64 AddKernel failed.");
    }
    status = func->AddKernel(floatkernel32);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod 32 AddKernel failed.");
    }
    status = func->AddKernel(floatkernel64);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod 64 AddKernel failed.");
    }
    // Register the function.
    status = registry->AddFunction(std::move(func));
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod AddFunction failed.");
    }
}

void EnsureModRegistered() {
    static std::once_flag flag;
    // Register the mod arrow compute function only once.
    std::call_once(flag, [] {
        auto* registry = arrow::compute::GetFunctionRegistry();
        RegisterMod(registry);
    });
}
