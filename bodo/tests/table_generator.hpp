#pragma once

#include "../io/arrow_reader.h"
#include "../libs/_table_builder.h"
#include "./test.hpp"
#include "arrow/builder.h"

namespace {
/**
 * @brief convert a std::vector<T> into an appropriately typed arrow::Array
 * @tparam T type of the vector passed in
 * @param data the data that the returned array should contain
 * @return arrow::Array containing a copy of data's contents
 **/
template <typename T>
std::shared_ptr<arrow::Array> vecToArrowArray(const std::vector<T>& data) {
    using Builder = arrow::CTypeTraits<T>::BuilderType;
    Builder builder;
    for (const auto& element : data) {
        CHECK_ARROW_MEM(builder.Append(element), "");
    }
    std::shared_ptr<arrow::Array> ret;
    CHECK_ARROW_MEM(builder.Finish(&ret), "");
    return ret;
}

/**
 * @brief Helper function that pushes a single input vector into a vector
 * of arrow::Arrays and arrow::Fields.
 * @tparam T inner type of input_column
 * @param arrays vector of arrow Arrays to append to as output
 * @param fields vector of arrow Fields to append to as output
 * @param names list of names to use as column names for the fields
 * @param input_column data to convert to an arrow array
 **/
template <typename T>
void _cppToArrow_inner(std::vector<std::shared_ptr<arrow::Array>>& arrays,
                       std::vector<std::shared_ptr<arrow::Field>>& fields,
                       const std::vector<std::string>& names,
                       const std::vector<T>& input_column) {
    size_t i = arrays.size();
    bodo::tests::check(i < names.size(),
                       "More input columns than column names");

    auto field =
        arrow::field(names[i], arrow::CTypeTraits<T>::type_singleton());
    auto array = vecToArrowArray(input_column);

    arrays.push_back(array);
    fields.push_back(field);
}

/**
 * @brief Helper function that converts a variadic list of input vectors into a
 * vector of arrow::Arrays and arrow::Fields. This overload is the base case.
 * @tparam T
 * @param arrays vector of arrow Arrays to append to as output
 * @param fields vector of arrow Fields to append to as output
 * @param names
 * @param is_nullable
 * @param str_as_dict_cols
 * @param input_column
 **/
template <typename T>
void _cppToArrow(std::vector<std::shared_ptr<arrow::Array>>& arrays,
                 std::vector<std::shared_ptr<arrow::Field>>& fields,
                 const std::vector<std::string>& names,
                 const std::vector<bool>& is_nullable,
                 const std::set<std::string>& str_as_dict_cols,
                 const std::vector<T>& input_column) {
    _cppToArrow_inner(arrays, fields, names, input_column);

    // Verify that the final lengths of arrays/fields matches the lengths of
    // names/is_nullable
    bodo::tests::check(
        arrays.size() == names.size(),
        "Number of input columns did not match number of column names");
    bodo::tests::check(
        arrays.size() == is_nullable.size(),
        "Number of input columns did not match number of is_nullable values");
    // This check should never fail
    bodo::tests::check(arrays.size() == fields.size());
}

/**
 * @brief Helper function that converts a variadic list of input vectors into a
 * vector of arrow::Arrays and arrow::Fields.
 * @tparam T inner type of input_column, should be automatically inferred in
 * most cases.
 * @tparam Ts... parameter pack for variadic arguments, should be automatically
 * inferred in most cases.
 * @param arrays vector of arrow Arrays to append to as output - this is taken
 * as an out param to make the recursion easier to express
 * @param fields vector of arrow Fields to append to as output - this is taken
 * as an out param to make the recursion easier to express
 * @param names list of column names to use in the table. The order of names
 * must match the order in which the variadic arguments are passed in.
 * @param is_nullable see TableBuilder::TableBuilder
 * @param str_as_dict_cols see TableBuilder::TableBuilder
 * @param input_column first input vector to be used as the data for a column
 * @param input_columns optional variadic list of the remaining columns
 **/
template <typename T, typename... Ts>
void _cppToArrow(std::vector<std::shared_ptr<arrow::Array>>& arrays,
                 std::vector<std::shared_ptr<arrow::Field>>& fields,
                 const std::vector<std::string>& names,
                 const std::vector<bool>& is_nullable,
                 const std::set<std::string>& str_as_dict_cols,
                 const std::vector<T>& input_column,
                 const std::vector<Ts>&... input_columns) {
    _cppToArrow_inner(arrays, fields, names, input_column);
    _cppToArrow(arrays, fields, names, is_nullable, str_as_dict_cols,
                input_columns...);
}
}  // namespace

namespace bodo {
namespace tests {
/**
 * @brief Construct a Bodo Table from a variadic list of C++ vectors.
 * @tparam T inner type of input_column, should be automatically inferred in
 * most cases.
 * @tparam Ts... parameter pack for variadic arguments, should be automatically
 * inferred in most cases.
 * @param names list of column names to use in the table. The order of names
 * must match the order in which the variadic arguments are passed in.
 * @param is_nullable see TableBuilder::TableBuilder
 * @param str_as_dict_cols see TableBuilder::TableBuilder
 * @param input_column first input vector to be used as the data for a column
 * @param input_columns optional variadic list of the remaining columns
 * @return
 **/
template <typename T, typename... Ts>
std::unique_ptr<table_info> cppToBodo(
    const std::vector<std::string>& names, const std::vector<bool>& is_nullable,
    const std::set<std::string>& str_as_dict_cols,
    const std::vector<T>& input_column,
    const std::vector<Ts>&... input_columns) {
    // This method will first collect all of (input_column, input_columns...)
    // into an arrow::Table and then use TableBuilder to convert that into a
    // table_info.

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    // arrays and fields will be populated by the call below
    _cppToArrow(arrays, fields, names, is_nullable, str_as_dict_cols,
                input_column, input_columns...);
    auto schema = arrow::schema(std::move(fields));
    auto arrow_table = arrow::Table::Make(schema, arrays);

    std::vector<int> selected_fields;
    for (size_t i = 0; i < arrays.size(); i++) {
        selected_fields.push_back(i);
    }
    TableBuilder builder(schema, selected_fields, input_column.size(),
                         is_nullable, str_as_dict_cols, true);
    builder.append(std::move(arrow_table));

    // Check that the append added all expected rows
    bodo::tests::check(builder.get_rem_rows() == 0,
                       "Unexpected remaining rows");

    auto* table = builder.get_table();

    bodo::tests::check(table->ncols() == names.size(),
                       "Incorrect final number of columns");
    bodo::tests::check(table->nrows() == input_column.size(),
                       "Incorrect final number of rows");
    return std::unique_ptr<table_info>(table);
}
}  // namespace tests
}  // namespace bodo
