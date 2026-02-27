#pragma once

template <typename EXPR_TYPE, typename BUILD_EXPR>
inline std::shared_ptr<bodo::Schema> getProjectionOutputSchema(
    std::vector<duckdb::ColumnBinding>& source_cols,
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& exprs,
    std::shared_ptr<bodo::Schema> input_schema,
    std::vector<std::string>& col_names,
    std::vector<std::shared_ptr<EXPR_TYPE>>& physical_exprs,
    BUILD_EXPR builder) {
    // Map of column bindings to column indices in physical input table
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map =
        getColRefMap(source_cols);

    // Create the output schema from expressions
    std::shared_ptr<bodo::Schema> output_schema =
        std::make_shared<bodo::Schema>();
    for (size_t expr_idx = 0; expr_idx < exprs.size(); ++expr_idx) {
        auto& expr = exprs[expr_idx];
        physical_exprs.emplace_back(builder(expr, col_ref_map));

        if (expr->type == duckdb::ExpressionType::BOUND_COLUMN_REF) {
            auto& colref = expr->Cast<duckdb::BoundColumnRefExpression>();
            size_t col_idx = col_ref_map[{colref.binding.table_index,
                                          colref.binding.column_index}];
            std::unique_ptr<bodo::DataType> col_type =
                input_schema->column_types[col_idx]->copy();
            output_schema->append_column(std::move(col_type));
            if (input_schema->column_names.size() > 0) {
                col_names.emplace_back(input_schema->column_names[col_idx]);
            } else {
                col_names.emplace_back(colref.GetName());
            }
        } else if (expr->type == duckdb::ExpressionType::BOUND_FUNCTION) {
            auto& func_expr = expr->Cast<duckdb::BoundFunctionExpression>();
            if (func_expr.bind_info) {
                BodoScalarFunctionData& scalar_func_data =
                    func_expr.bind_info->Cast<BodoScalarFunctionData>();

                std::unique_ptr<bodo::DataType> col_type =
                    bodo::Schema::FromArrowSchema(scalar_func_data.out_schema)
                        ->column_types[0]
                        ->copy();
                output_schema->append_column(std::move(col_type));
                col_names.emplace_back(
                    scalar_func_data.out_schema->field(0)->name());
            } else {
                if (func_expr.function.name == "floor") {
                    output_schema->append_column(
                        input_schema->column_types[expr_idx]->copy());
                    col_names.emplace_back("floor");
                } else {
                    std::shared_ptr<arrow::DataType> arrow_type =
                        duckdbTypeToArrow(func_expr.return_type);
                    output_schema->append_column(
                        arrow_type_to_bodo_data_type(arrow_type));
                    col_names.emplace_back(func_expr.function.name);
                }
            }
        } else if (expr->type == duckdb::ExpressionType::VALUE_CONSTANT) {
            auto& const_expr = expr->Cast<duckdb::BoundConstantExpression>();

            std::unique_ptr<bodo::DataType> col_type =
                arrow_type_to_bodo_data_type(
                    duckdbValueToArrowType(const_expr.value))
                    ->copy();
            output_schema->append_column(std::move(col_type));
            col_names.emplace_back(const_expr.value.ToString());
        } else if (expr->type == duckdb::ExpressionType::COMPARE_EQUAL ||
                   expr->type == duckdb::ExpressionType::COMPARE_NOTEQUAL ||
                   expr->type == duckdb::ExpressionType::COMPARE_LESSTHAN ||
                   expr->type == duckdb::ExpressionType::COMPARE_GREATERTHAN ||
                   expr->type ==
                       duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO ||
                   expr->type ==
                       duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO) {
            std::unique_ptr<bodo::DataType> col_type =
                std::make_unique<bodo::DataType>(
                    bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::CTypeEnum::_BOOL);
            output_schema->append_column(std::move(col_type));
            if (input_schema->column_names.size() > 0) {
                col_names.emplace_back(input_schema->column_names[0]);
            } else {
                col_names.emplace_back("comp");
            }
        } else if (expr->type == duckdb::ExpressionType::CONJUNCTION_AND ||
                   expr->type == duckdb::ExpressionType::CONJUNCTION_OR) {
            std::unique_ptr<bodo::DataType> col_type =
                std::make_unique<bodo::DataType>(
                    bodo_array_type::NULLABLE_INT_BOOL,
                    Bodo_CTypes::CTypeEnum::_BOOL);
            output_schema->append_column(std::move(col_type));
            if (input_schema->column_names.size() > 0) {
                col_names.emplace_back(input_schema->column_names[0]);
            } else {
                col_names.emplace_back("conjunction");
            }
        } else if (expr->type == duckdb::ExpressionType::CASE_EXPR) {
            auto& case_expr = expr->Cast<duckdb::BoundCaseExpression>();

            // TODO(ehsan): cast to common type of inputs if needed to match
            // compute?
            std::unique_ptr<bodo::DataType> col_type =
                arrow_type_to_bodo_data_type(
                    duckdbTypeToArrow(case_expr.return_type))
                    ->copy();
            output_schema->append_column(std::move(col_type));
            if (input_schema->column_names.size() > 0) {
                col_names.emplace_back(input_schema->column_names[0]);
            } else {
                col_names.emplace_back("Case");
            }
        } else if (expr->type == duckdb::ExpressionType::OPERATOR_CAST) {
            auto& bce = expr->Cast<duckdb::BoundCastExpression>();

            std::unique_ptr<bodo::DataType> col_type =
                arrow_type_to_bodo_data_type(duckdbTypeToArrow(bce.return_type))
                    ->copy();
            output_schema->append_column(std::move(col_type));
            if (input_schema->column_names.size() > 0) {
                col_names.emplace_back(input_schema->column_names[0]);
            } else {
                col_names.emplace_back("Cast");
            }
        } else if (expr->type == duckdb::ExpressionType::OPERATOR_NOT) {
            std::unique_ptr<bodo::DataType> col_type =
                arrow_type_to_bodo_data_type(
                    duckdbTypeToArrow(duckdb::LogicalType::BOOLEAN))
                    ->copy();
            output_schema->append_column(std::move(col_type));
            if (input_schema->column_names.size() > 0) {
                col_names.emplace_back(input_schema->column_names[0]);
            } else {
                col_names.emplace_back("Not");
            }
        } else {
            throw std::runtime_error(
                "Unsupported expression type in projection " +
                std::to_string(static_cast<int>(expr->type)) + " " +
                expr->ToString());
        }
    }
    output_schema->column_names = col_names;
    output_schema->metadata = input_schema->metadata;
    return output_schema;
}
