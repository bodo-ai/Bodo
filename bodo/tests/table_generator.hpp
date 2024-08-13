#pragma once

#include <string>
#include "../io/arrow_reader.h"
#include "../libs/_bodo_to_arrow.h"
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

template <typename T>
void _cppToArrow_inner(std::vector<std::shared_ptr<arrow::Array>>& arrays,
                       std::vector<std::shared_ptr<arrow::Field>>& fields,
                       const std::vector<std::string>& names,
                       const std::vector<T>& input_column);

/**
 * @brief Specialization of _cppToArrow_inner for std::vector<std::string>.
 *
 * note that maybe_unused is needed because clang doesn't realize that this is
 * being invoked as a specialization of the other declaration.
 **/
template <>
[[maybe_unused]] void _cppToArrow_inner<std::vector<std::string>>(
    std::vector<std::shared_ptr<arrow::Array>>& arrays,
    std::vector<std::shared_ptr<arrow::Field>>& fields,
    const std::vector<std::string>& names,
    const std::vector<std::vector<std::string>>& input_column) {
    size_t i = arrays.size();
    bodo::tests::check(i < names.size(),
                       "More input columns than column names");

    std::vector<std::string> inner;
    std::vector<offset_t> offsets;

    offsets.push_back(0);
    for (const auto& vec : input_column) {
        inner.insert(inner.end(), vec.begin(), vec.end());
        offset_t new_offset = offsets.back() + vec.size();
        offsets.push_back(new_offset);
    }
    auto inner_array = vecToArrowArray(inner);
    auto offset_buf = arrow::Buffer::FromVector(offsets);

    auto inner_type = std::make_shared<arrow::Field>(
        "element", arrow::CTypeTraits<std::string>::type_singleton());
    auto type = arrow::large_list(inner_type);
    auto field = arrow::field(names[i], type);
    auto array = std::make_shared<arrow::LargeListArray>(
        type, input_column.size(), offset_buf, inner_array);

    arrays.push_back(array);
    fields.push_back(field);
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
 * @param pool: a buffer pool to deal with allocations.
 * @param mm: a memory manager to deal with allocations.
 * @param input_column first input vector to be used as the data for a column
 * @param input_columns optional variadic list of the remaining columns
 * @return
 **/
template <typename T, typename... Ts>
std::shared_ptr<table_info> cppToBodo(
    const std::vector<std::string>& names, const std::vector<bool>& is_nullable,
    const std::set<std::string>& str_as_dict_cols,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm,
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

    std::shared_ptr<arrow::RecordBatch> batch =
        arrow_table->CombineChunksToBatch().ValueOrDie();
    std::shared_ptr<table_info> bodo_table =
        arrow_recordbatch_to_bodo(batch, input_column.size());

    // Adjust the schema to replace the desired columns with dictionary encoded
    // columns, and nullable with numpy.
    const std::shared_ptr<bodo::Schema>& old_bodo_schema = bodo_table->schema();
    std::shared_ptr<bodo::Schema> new_bodo_schema =
        std::make_shared<bodo::Schema>();
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    for (size_t col_idx = 0; col_idx < old_bodo_schema->column_types.size();
         col_idx++) {
        std::unique_ptr<DataType> new_type =
            old_bodo_schema->column_types[col_idx]->copy();
        if (new_type->array_type == bodo_array_type::NULLABLE_INT_BOOL &&
            !is_nullable[col_idx]) {
            new_type = std::make_unique<DataType>(bodo_array_type::NUMPY,
                                                  new_type->c_type);
            // A distastfeful but effective way to transform the array type so
            // we can/ insert into the table build buffer.
            bodo_table->columns[col_idx]->arr_type = bodo_array_type::NUMPY;
        } else if (str_as_dict_cols.count(schema->field(col_idx)->name()) > 0) {
            new_type = std::make_unique<DataType>(bodo_array_type::DICT,
                                                  Bodo_CTypes::STRING);
        }
        new_bodo_schema->append_column(new_type->copy());
        dict_builders.push_back(
            create_dict_builder_for_array(new_type->copy(), true));
    }

    for (auto& it : bodo_table->columns) {
        dict_builders.push_back(create_dict_builder_for_array(it, true));
    }
    TableBuildBuffer builder(new_bodo_schema, dict_builders, pool, mm);
    builder.UnifyTablesAndAppend(bodo_table, dict_builders);

    std::shared_ptr<table_info> table = builder.data_table;

    bodo::tests::check(table->ncols() == names.size(),
                       "Incorrect final number of columns");
    bodo::tests::check(table->nrows() == input_column.size(),
                       "Incorrect final number of rows");

    return table;
}

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
std::shared_ptr<table_info> cppToBodo(
    const std::vector<std::string>& names, const std::vector<bool>& is_nullable,
    const std::set<std::string>& str_as_dict_cols,
    const std::vector<T>& input_column,
    const std::vector<Ts>&... input_columns) {
    return cppToBodo(
        names, is_nullable, str_as_dict_cols, bodo::BufferPool::DefaultPtr(),
        bodo::default_buffer_memory_manager(), input_column, input_columns...);
}

/**
 * @brief Construct a Bodo Array from a cpp vector using cppToBodo.
 * @tparam T inner type of the input vector, should be automatically inferred in
 * most cases.
 * @param vec the cpp vector to turn into an array info
 * @param is_nullable see TableBuilder::TableBuilder
 * @return
 **/
template <typename T = int64_t>
std::shared_ptr<array_info> cppToBodoArr(
    std::vector<T> vec, bool is_nullable = false,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    std::shared_ptr<table_info> vec_as_table =
        cppToBodo({"A"}, {is_nullable}, {}, pool, mm, std::move(vec));
    return vec_as_table->columns[0];
}

}  // namespace tests
}  // namespace bodo
