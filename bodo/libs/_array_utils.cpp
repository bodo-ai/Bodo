// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_array_utils.h"

#include <mpi.h>
#include <iostream>
#include <string>

#include "_decimal_ext.h"

/**
 * Append values from a byte buffer to a primitive array builder.
 * @param values: pointer to buffer containing data
 * @param offset: offset of starting element (not byte units)
 * @param length: number of elements to copy
 * @param builder: primitive array builder holding output array
 * @param valid_elems: non-zero if elem i is non-null
 */
void append_to_primitive(const uint8_t* values, int64_t offset, int64_t length,
                         arrow::ArrayBuilder* builder,
                         const std::vector<uint8_t>& valid_elems) {
    arrow::Type::type typ = builder->type()->id();
    if (typ == arrow::Type::INT8) {
        auto typed_builder =
            dynamic_cast<arrow::NumericBuilder<arrow::Int8Type>*>(builder);
        (void)typed_builder->AppendValues((int8_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::UINT8) {
        auto typed_builder =
            dynamic_cast<arrow::NumericBuilder<arrow::UInt8Type>*>(builder);
        (void)typed_builder->AppendValues((uint8_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::INT16) {
        auto typed_builder =
            dynamic_cast<arrow::NumericBuilder<arrow::Int16Type>*>(builder);
        (void)typed_builder->AppendValues((int16_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::UINT16) {
        auto typed_builder =
            dynamic_cast<arrow::NumericBuilder<arrow::UInt16Type>*>(builder);
        (void)typed_builder->AppendValues((uint16_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::INT32) {
        auto typed_builder =
            dynamic_cast<arrow::NumericBuilder<arrow::Int32Type>*>(builder);
        (void)typed_builder->AppendValues((int32_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::UINT32) {
        auto typed_builder =
            dynamic_cast<arrow::NumericBuilder<arrow::UInt32Type>*>(builder);
        (void)typed_builder->AppendValues((uint32_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::INT64) {
        auto typed_builder =
            dynamic_cast<arrow::NumericBuilder<arrow::Int64Type>*>(builder);
        (void)typed_builder->AppendValues((int64_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::UINT64) {
        auto typed_builder =
            dynamic_cast<arrow::NumericBuilder<arrow::UInt64Type>*>(builder);
        (void)typed_builder->AppendValues((uint64_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::FLOAT) {
        auto typed_builder =
            dynamic_cast<arrow::NumericBuilder<arrow::FloatType>*>(builder);
        (void)typed_builder->AppendValues((float*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::DOUBLE) {
        auto typed_builder =
            dynamic_cast<arrow::NumericBuilder<arrow::DoubleType>*>(builder);
        (void)typed_builder->AppendValues((double*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::DECIMAL) {
        auto typed_builder = dynamic_cast<arrow::Decimal128Builder*>(builder);
        (void)typed_builder->AppendValues(
            (uint8_t*)values + BYTES_PER_DECIMAL * offset, length,
            valid_elems.data());
    } else {
        std::string err_msg = "append_to_primitive : Unsupported type " +
                              builder->type()->ToString();
        Bodo_PyErr_SetString(PyExc_RuntimeError, err_msg.c_str());
    }
}

/**
 * Generate Arrow array on the fly by appending values from a given input
 * array.
 * Can be called multiple times with different input arrays but they must
 * be of the same type.
 * The first call should pass the base (root) input array, range of rows
 * to append to output, and a builder. The builder must have the same type as
 * the input array. For example, if the input array is list(list(string))
 * the builder must be for list(list(string)). When the builder is created
 * (outside of this function) it has an empty array. Each time this function
 * is called the builder will append new values to its array.
 *
 * Note that this is a recursive algorithm and works with nested arrays.
 *
 * @param input_array : current input array in the tree of arrays
 * @param start_offset : starting offset in input_array
 * @param end_offset : ending offset in input_array
 * @param builder : array builder of the same type as input_array. constructs
                    the output array.
 */
void append_to_out_array(std::shared_ptr<arrow::Array> input_array,
                         int64_t start_offset, int64_t end_offset,
                         arrow::ArrayBuilder* builder) {
    // TODO check for nulls and append nulls
#if OFFSET_BITWIDTH == 32
    if (input_array->type_id() == arrow::Type::LIST) {
        // TODO: assert builder.type() == LIST
        std::shared_ptr<arrow::ListArray> list_array =
            std::dynamic_pointer_cast<arrow::ListArray>(input_array);
        auto list_builder = dynamic_cast<arrow::ListBuilder*>(builder);
#else
    if (input_array->type_id() == arrow::Type::LARGE_LIST) {
        // TODO: assert builder.type() == LARGE_LIST
        std::shared_ptr<arrow::LargeListArray> list_array =
            std::dynamic_pointer_cast<arrow::LargeListArray>(input_array);
        auto list_builder = dynamic_cast<arrow::LargeListBuilder*>(builder);
#endif
        arrow::ArrayBuilder* child_builder = list_builder->value_builder();

        for (int64_t idx = start_offset; idx < end_offset; idx++) {
            if (list_array->IsNull(idx)) {
                (void)list_builder->AppendNull();
                continue;
            }
            (void)list_builder->Append();  // indicate list boundary
            // TODO optimize
            append_to_out_array(list_array->values(),  // child array
                                list_array->value_offset(idx),
                                list_array->value_offset(idx + 1),
                                child_builder);
        }
    } else if (input_array->type_id() == arrow::Type::STRUCT) {
        // TODO: assert builder.type() == STRUCT
        auto struct_array =
            std::dynamic_pointer_cast<arrow::StructArray>(input_array);
        auto struct_type =
            std::dynamic_pointer_cast<arrow::StructType>(struct_array->type());
        auto struct_builder = dynamic_cast<arrow::StructBuilder*>(builder);
        for (int64_t idx = start_offset; idx < end_offset; idx++) {
            if (struct_array->IsNull(idx)) {
                (void)struct_builder->AppendNull();
                continue;
            }
            for (int i = 0; i < struct_type->num_fields();
                 i++) {  // each field is an array
                arrow::ArrayBuilder* field_builder = builder->child(i);
                append_to_out_array(struct_array->field(i), idx, idx + 1,
                                    field_builder);
            }
            (void)struct_builder->Append();
        }
        // finished appending (end_offset - start_offset) structs
        // struct_builder->AppendValues(end_offset - start_offset, NULLPTR);
#if OFFSET_BITWIDTH == 32
    } else if (input_array->type_id() == arrow::Type::STRING) {
        auto str_array =
            std::dynamic_pointer_cast<arrow::StringArray>(input_array);
        auto str_builder = dynamic_cast<arrow::StringBuilder*>(builder);
#else
    } else if (input_array->type_id() == arrow::Type::LARGE_STRING) {
        auto str_array =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(input_array);
        auto str_builder = dynamic_cast<arrow::LargeStringBuilder*>(builder);
#endif
        int64_t num_elems = end_offset - start_offset;
        // TODO: optimize
        for (int64_t i = 0; i < num_elems; i++) {
            if (str_array->IsNull(start_offset + i))
                (void)str_builder->AppendNull();
            else
                (void)str_builder->AppendValues(
                    {str_array->GetString(start_offset + i)});
        }
    } else {
        int64_t num_elems = end_offset - start_offset;
        // assume this is array of primitive values
        // TODO: decimal, date, etc.
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(input_array);
        std::vector<uint8_t> valid_elems(num_elems, 0);
        // TODO: more efficient way of getting null data?
        size_t j = 0;
        for (int64_t i = start_offset; i < start_offset + num_elems; i++)
            valid_elems[j++] = !primitive_array->IsNull(i);
        append_to_primitive(primitive_array->values()->data(), start_offset,
                            num_elems, builder, valid_elems);
    }
}

template <typename F>
array_info* RetrieveArray_SingleColumn_F(array_info* in_arr, F f,
                                         size_t nRowOut) {
    array_info* out_arr = NULL;
    bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
    Bodo_CTypes::CTypeEnum dtype = in_arr->dtype;
    if (arr_type == bodo_array_type::LIST_STRING) {
        // In the first case of STRING, we have to deal with offsets first so we
        // need one first loop to determine the needed length. In the second
        // loop, the assignation is made. If the entries are missing then the
        // bitmask is set to false.
        std::vector<offset_t> ListSizes_index(nRowOut);
        std::vector<offset_t> ListSizes_data(nRowOut);
        int64_t tot_size_index = 0;
        int64_t tot_size_data = 0;
        offset_t* index_offsets = (offset_t*)in_arr->data3;
        offset_t* data_offsets = (offset_t*)in_arr->data2;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = f(iRow);
            offset_t size_index = 0;
            offset_t size_data = 0;
            // To allow NaN values in the column
            if (idx >= 0) {
                offset_t start_offset_index = index_offsets[idx];
                offset_t end_offset_index = index_offsets[idx + 1];
                size_index = end_offset_index - start_offset_index;
                offset_t start_offset_data = data_offsets[start_offset_index];
                offset_t end_offset_data = data_offsets[end_offset_index];
                size_data = end_offset_data - start_offset_data;
            }
            ListSizes_index[iRow] = size_index;
            ListSizes_data[iRow] = size_data;
            tot_size_index += size_index;
            tot_size_data += size_data;
        }
        out_arr = alloc_array(nRowOut, tot_size_index, tot_size_data, arr_type,
                              dtype, 0, 0);
        uint8_t* out_sub_null_bitmask = (uint8_t*)out_arr->sub_null_bitmask;
        offset_t pos_index = 0;
        offset_t pos_data = 0;
        offset_t* out_index_offsets = (offset_t*)out_arr->data3;
        offset_t* out_data_offsets = (offset_t*)out_arr->data2;
        char* out_data1 = out_arr->data1;
        out_data_offsets[0] = 0;
        uint8_t* in_sub_null_bitmask = (uint8_t*)in_arr->sub_null_bitmask;
        offset_t* in_index_offsets = (offset_t*)in_arr->data3;
        offset_t* in_data_offsets = (offset_t*)in_arr->data2;
        char* in_data1 = in_arr->data1;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = f(iRow);
            offset_t size_index = ListSizes_index[iRow];
            offset_t size_data = ListSizes_data[iRow];
            out_index_offsets[iRow] = pos_index;
            // To allow NaN values in the column.
            bool bit = false;
            if (idx >= 0) {
                offset_t start_index_offset = in_index_offsets[idx];
                offset_t start_data_offset =
                    in_data_offsets[start_index_offset];
                for (offset_t u = 0; u < size_index; u++) {
                    offset_t len_str =
                        in_data_offsets[start_index_offset + u + 1] -
                        in_data_offsets[start_index_offset + u];
                    out_data_offsets[pos_index + u + 1] =
                        out_data_offsets[pos_index + u] + len_str;
                    bool bit =
                        GetBit(in_sub_null_bitmask, start_index_offset + u);
                    SetBitTo(out_sub_null_bitmask, pos_index + u, bit);
                }
                memcpy(&out_data1[pos_data], &in_data1[start_data_offset],
                       size_data);
                bit = in_arr->get_null_bit(idx);
            }
            pos_index += size_index;
            pos_data += size_data;
            out_arr->set_null_bit(iRow, bit);
        }
        out_index_offsets[nRowOut] = pos_index;
    }
    if (arr_type == bodo_array_type::STRING) {
        // In the first case of STRING, we have to deal with offsets first so we
        // need one first loop to determine the needed length. In the second
        // loop, the assignation is made. If the entries are missing then the
        // bitmask is set to false.
        int64_t n_chars = 0;
        std::vector<offset_t> ListSizes(nRowOut);
        offset_t* in_offsets = (offset_t*)in_arr->data2;
        char* in_data1 = in_arr->data1;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = f(iRow);
            offset_t size = 0;
            // To allow NaN values in the column
            if (idx >= 0) {
                offset_t start_offset = in_offsets[idx];
                offset_t end_offset = in_offsets[idx + 1];
                size = end_offset - start_offset;
            }
            ListSizes[iRow] = size;
            n_chars += size;
        }
        out_arr = alloc_array(nRowOut, n_chars, -1, arr_type, dtype, 0, 0);
        offset_t* out_offsets = (offset_t*)out_arr->data2;
        char* out_data1 = out_arr->data1;
        offset_t pos = 0;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = f(iRow);
            offset_t size = ListSizes[iRow];
            out_offsets[iRow] = pos;
            // To allow NaN values in the column.
            bool bit = false;
            if (idx >= 0) {
                offset_t start_offset = in_offsets[idx];
                char* out_ptr = out_data1 + pos;
                char* in_ptr = in_data1 + start_offset;
                memcpy(out_ptr, in_ptr, size);
                pos += size;
                bit = in_arr->get_null_bit(idx);
            }
            out_arr->set_null_bit(iRow, bit);
        }
        out_offsets[nRowOut] = pos;
    }
    if (arr_type == bodo_array_type::DICT) {
        // TODO refactor: this is mostly the same as NULLABLE_INT_BOOL

        array_info* in_indices = in_arr->info2;
        char* in_data1 = in_indices->data1;
        array_info* out_indices = alloc_array(
            nRowOut, -1, -1, in_indices->arr_type, in_indices->dtype, 0, 0);
        char* out_data1 = out_indices->data1;
        uint64_t siztype = numpy_item_size[in_indices->dtype];
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = f(iRow);
            // To allow NaN values in the column.
            bool bit = false;
            if (idx >= 0) {
                char* out_ptr = out_data1 + siztype * iRow;
                char* in_ptr = in_data1 + siztype * idx;
                memcpy(out_ptr, in_ptr, siztype);
                bit = in_indices->get_null_bit(idx);
            }
            out_indices->set_null_bit(iRow, bit);
        }
        out_arr = new array_info(
            bodo_array_type::DICT, in_arr->dtype, out_indices->length, -1, -1,
            NULL, NULL, NULL, out_indices->null_bitmask,
            NULL, NULL, NULL, NULL, 0, 0, 0, in_arr->has_global_dictionary,
            in_arr->info1, out_indices);
        // input and output share the same dictionary array
        incref_array(in_arr->info1);
    }
    if (arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // In the case of NULLABLE array, we do a single loop for
        // assigning the arrays.
        // We do not need to reassign the pointers, only their size
        // suffices for the copy.
        // In the case of missing array a value of false is assigned
        // to the bitmask.
        char* in_data1 = in_arr->data1;
        out_arr = alloc_array(nRowOut, -1, -1, arr_type, dtype, 0, 0);
        char* out_data1 = out_arr->data1;
        uint64_t siztype = numpy_item_size[dtype];
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = f(iRow);
            // To allow NaN values in the column.
            bool bit = false;
            if (idx >= 0) {
                char* out_ptr = out_data1 + siztype * iRow;
                char* in_ptr = in_data1 + siztype * idx;
                memcpy(out_ptr, in_ptr, siztype);
                bit = in_arr->get_null_bit(idx);
            }
            out_arr->set_null_bit(iRow, bit);
        }
    }
    if (arr_type == bodo_array_type::INTERVAL) {
        // Interval array requires copying left and right data
        uint64_t siztype = numpy_item_size[dtype];
        std::vector<char> vectNaN = RetrieveNaNentry(dtype);
        char* left_data = in_arr->data1;
        char* right_data = in_arr->data2;
        out_arr = alloc_array(nRowOut, -1, -1, arr_type, dtype, 0, 0);
        char* out_left_data = out_arr->data1;
        char* out_right_data = out_arr->data2;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = f(iRow);
            char* out_left_ptr = out_left_data + siztype * iRow;
            char* out_right_ptr = out_right_data + siztype * iRow;
            char* in_left_ptr;
            char* in_right_ptr;
            // To allow NaN values in the column.
            if (idx >= 0) {
                in_left_ptr = left_data + siztype * idx;
                in_right_ptr = right_data + siztype * idx;
            } else {
                in_left_ptr = vectNaN.data();
                in_right_ptr = vectNaN.data();
            }
            memcpy(out_left_ptr, in_left_ptr, siztype);
            memcpy(out_right_ptr, in_right_ptr, siztype);
        }
    }
    if (arr_type == bodo_array_type::CATEGORICAL) {
        // In the case of CATEGORICAL array we have only to put a single
        // entry. For the NaN entry the value is -1.
        uint64_t siztype = numpy_item_size[dtype];
        std::vector<char> vectNaN =
            RetrieveNaNentry(dtype);  // returns a -1 for integer values.
        char* in_data1 = in_arr->data1;
        int64_t num_categories = in_arr->num_categories;
        out_arr = alloc_categorical(nRowOut, dtype, num_categories);
        char* out_data1 = out_arr->data1;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = f(iRow);
            char* out_ptr = out_data1 + siztype * iRow;
            char* in_ptr;
            // To allow NaN values in the column.
            if (idx >= 0)
                in_ptr = in_data1 + siztype * idx;
            else
                in_ptr = vectNaN.data();
            memcpy(out_ptr, in_ptr, siztype);
        }
    }
    if (arr_type == bodo_array_type::NUMPY) {
        // In the case of NUMPY array we have only to put a single
        // entry.
        uint64_t siztype = numpy_item_size[dtype];
        std::vector<char> vectNaN = RetrieveNaNentry(dtype);
        char* in_data1 = in_arr->data1;
        out_arr = alloc_array(nRowOut, -1, -1, arr_type, dtype, 0, 0);
        char* out_data1 = out_arr->data1;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = f(iRow);
            char* out_ptr = out_data1 + siztype * iRow;
            char* in_ptr;
            // To allow NaN values in the column.
            if (idx >= 0)
                in_ptr = in_data1 + siztype * idx;
            else
                in_ptr = vectNaN.data();
            memcpy(out_ptr, in_ptr, siztype);
        }
    }
    if (arr_type == bodo_array_type::ARROW) {
        // Arrow builder for output array. builds it dynamically (buffer
        // sizes are not known in advance)
        std::unique_ptr<arrow::ArrayBuilder> builder;
        std::shared_ptr<arrow::Array> in_arrow_array = in_arr->array;
        (void)arrow::MakeBuilder(arrow::default_memory_pool(),
                                 in_arrow_array->type(), &builder);
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = f(iRow);
            // append value in position 'row' of input array to builder's
            // array (this is a recursive algorithm, can traverse nested
            // arrays)
            // To allow NaN values in the column.
            if (idx >= 0)
                append_to_out_array(in_arrow_array, idx, idx + 1,
                                    builder.get());
            else
                (void)builder->AppendNull();
        }

        // get final output array from builder
        std::shared_ptr<arrow::Array> out_arrow_array;
        // TODO: assert builder is not null (at least one row added)
        (void)builder->Finish(&out_arrow_array);
        out_arr =
            new array_info(bodo_array_type::ARROW, Bodo_CTypes::INT8 /*dummy*/,
                           nRowOut, -1, -1, NULL, NULL, NULL, NULL, NULL,
                           /*meminfo TODO*/ NULL, NULL, out_arrow_array);
    }
    return out_arr;
};

array_info* RetrieveArray_SingleColumn(array_info* in_arr,
                                       std::vector<int64_t> const& ListIdx) {
    return RetrieveArray_SingleColumn_F(
        in_arr, [&](size_t iRow) -> int64_t { return ListIdx[iRow]; },
        ListIdx.size());
}

array_info* RetrieveArray_SingleColumn_arr(array_info* in_arr,
                                           array_info* idx_arr) {
    if (idx_arr->dtype != Bodo_CTypes::UINT64) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "UINT64 is the only index type allowed");
        return nullptr;
    }
    size_t siz = idx_arr->length;
    return RetrieveArray_SingleColumn_F(
        in_arr,
        [&](size_t idx) -> int64_t {
            int64_t val = idx_arr->at<uint64_t>(idx);
            return val;
        },
        siz);
}

array_info* RetrieveArray_TwoColumns(
    array_info* const& arr1, array_info* const& arr2,
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> const& ListPairWrite,
    int const& ChoiceColumn, bool const& map_integer_type) {
    if ((arr1 != nullptr) && (arr2 != nullptr) &&
        (arr1->arr_type == bodo_array_type::DICT) &&
        (arr2->arr_type == bodo_array_type::DICT) &&
        (arr1->info1 != arr2->info1)) {
        throw std::runtime_error(
            "RetrieveArray_TwoColumns: don't know if arrays have unified "
            "dictionary");
    }
    size_t nRowOut = ListPairWrite.size();
    array_info* out_arr = NULL;
    /* The function for computing the returning values
     * In the output is the column index to use and the row index to use.
     * The row index may be -1 though.
     *
     * @param the row index in the output
     * @return the pair (column,row) to be used.
     */
    auto get_iRow =
        [&](size_t const& iRowIn) -> std::pair<array_info*, std::ptrdiff_t> {
        std::pair<std::ptrdiff_t, std::ptrdiff_t> pairLRcolumn =
            ListPairWrite[iRowIn];
        if (ChoiceColumn == 0) return {arr1, pairLRcolumn.first};
        if (ChoiceColumn == 1) return {arr2, pairLRcolumn.second};
        if (pairLRcolumn.first != -1) return {arr1, pairLRcolumn.first};
        return {arr2, pairLRcolumn.second};
    };
    // eshift is the in_table index used for the determination
    // of arr_type and dtype of the returned column.
    array_info* ref_column = nullptr;
    if (ChoiceColumn == 0) ref_column = arr1;
    if (ChoiceColumn == 1) ref_column = arr2;
    if (ChoiceColumn == 2) ref_column = arr1;
    bodo_array_type::arr_type_enum arr_type = ref_column->arr_type;
    Bodo_CTypes::CTypeEnum dtype = ref_column->dtype;
    if (arr_type == bodo_array_type::LIST_STRING) {
        // In the first case of STRING, we have to deal with offsets first so we
        // need one first loop to determine the needed length. In the second
        // loop, the assignation is made. If the entries are missing then the
        // bitmask is set to false.
        std::vector<offset_t> ListSizes_index(nRowOut);
        std::vector<offset_t> ListSizes_data(nRowOut);
        int64_t tot_size_index = 0;
        int64_t tot_size_data = 0;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<array_info*, std::ptrdiff_t> ArrRow = get_iRow(iRow);
            offset_t size_index = 0;
            offset_t size_data = 0;
            if (ArrRow.second >= 0) {
                offset_t* index_offsets = (offset_t*)ArrRow.first->data3;
                offset_t* data_offsets = (offset_t*)ArrRow.first->data2;
                offset_t start_offset_index = index_offsets[ArrRow.second];
                offset_t end_offset_index = index_offsets[ArrRow.second + 1];
                size_index = end_offset_index - start_offset_index;
                offset_t start_offset_data = data_offsets[start_offset_index];
                offset_t end_offset_data = data_offsets[end_offset_index];
                size_data = end_offset_data - start_offset_data;
            }
            ListSizes_index[iRow] = size_index;
            ListSizes_data[iRow] = size_data;
            tot_size_index += size_index;
            tot_size_data += size_data;
        }
        out_arr = alloc_array(nRowOut, tot_size_index, tot_size_data, arr_type,
                              dtype, 0, 0);
        uint8_t* out_sub_null_bitmask = (uint8_t*)out_arr->sub_null_bitmask;
        offset_t pos_index = 0;
        offset_t pos_data = 0;
        offset_t* out_index_offsets = (offset_t*)out_arr->data3;
        offset_t* out_data_offsets = (offset_t*)out_arr->data2;
        out_data_offsets[0] = 0;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<array_info*, std::ptrdiff_t> ArrRow = get_iRow(iRow);
            offset_t size_index = ListSizes_index[iRow];
            offset_t size_data = ListSizes_data[iRow];
            out_index_offsets[iRow] = pos_index;
            bool bit = false;
            if (ArrRow.second >= 0) {
                array_info* e_col = ArrRow.first;
                size_t i_row = ArrRow.second;
                offset_t* in_index_offsets = (offset_t*)e_col->data3;
                offset_t* in_data_offsets = (offset_t*)e_col->data2;
                char* data1 = e_col->data1;
                uint8_t* in_sub_null_bitmask =
                    (uint8_t*)e_col->sub_null_bitmask;
                offset_t start_index_offset = in_index_offsets[i_row];
                offset_t start_data_offset =
                    in_data_offsets[start_index_offset];
                for (offset_t u = 0; u < size_index; u++) {
                    offset_t len_str =
                        in_data_offsets[start_index_offset + u + 1] -
                        in_data_offsets[start_index_offset + u];
                    out_data_offsets[pos_index + u + 1] =
                        out_data_offsets[pos_index + u] + len_str;
                    bool bit =
                        GetBit(in_sub_null_bitmask, start_index_offset + u);
                    SetBitTo(out_sub_null_bitmask, pos_index + u, bit);
                }
                memcpy(&out_arr->data1[pos_data], &data1[start_data_offset],
                       size_data);
                bit = e_col->get_null_bit(ArrRow.second);
            }
            pos_index += size_index;
            pos_data += size_data;
            out_arr->set_null_bit(iRow, bit);
        }
        out_index_offsets[nRowOut] = pos_index;
    }
    if (arr_type == bodo_array_type::STRING) {
        // In the first case of STRING, we have to deal with offsets first so we
        // need one first loop to determine the needed length. In the second
        // loop, the assignation is made. If the entries are missing then the
        // bitmask is set to false.
        int64_t n_chars = 0;
        std::vector<offset_t> ListSizes(nRowOut);
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<array_info*, std::ptrdiff_t> ArrRow = get_iRow(iRow);
            offset_t size = 0;
            if (ArrRow.second >= 0) {
                offset_t* in_offsets = (offset_t*)ArrRow.first->data2;
                offset_t end_offset = in_offsets[ArrRow.second + 1];
                offset_t start_offset = in_offsets[ArrRow.second];
                size = end_offset - start_offset;
            }
            ListSizes[iRow] = size;
            n_chars += size;
        }
        out_arr = alloc_array(nRowOut, n_chars, -1, arr_type, dtype, 0, 0);
        offset_t pos = 0;
        offset_t* out_offsets = (offset_t*)out_arr->data2;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<array_info*, std::ptrdiff_t> ArrRow = get_iRow(iRow);
            offset_t size = ListSizes[iRow];
            out_offsets[iRow] = pos;
            bool bit = false;
            if (ArrRow.second >= 0) {
                array_info* e_col = ArrRow.first;
                offset_t* in_offsets = (offset_t*)e_col->data2;
                offset_t start_offset = in_offsets[ArrRow.second];
                char* out_ptr = out_arr->data1 + pos;
                char* in_ptr = e_col->data1 + start_offset;
                memcpy(out_ptr, in_ptr, size);
                pos += size;
                bit = e_col->get_null_bit(ArrRow.second);
            }
            out_arr->set_null_bit(iRow, bit);
        }
        out_offsets[nRowOut] = pos;
    }
    if (arr_type == bodo_array_type::DICT) {
        // TODO refactor? this is mostly the same logic as NULLABLE_INT_BOOL
        //if (is_parallel && !ref_column->has_global_dictionary)
        //    throw std::runtime_error("RetrieveArray_TwoColumns: reference column doesn't have a global dictionary");
        array_info* out_indices =
            alloc_array(nRowOut, -1, -1, ref_column->info2->arr_type,
                        ref_column->info2->dtype, 0, 0);
        uint64_t siztype = numpy_item_size[ref_column->info2->dtype];
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<array_info*, std::ptrdiff_t> ArrRow = get_iRow(iRow);
            bool bit = false;
            if (ArrRow.second >= 0) {
                array_info* e_col = ArrRow.first;
                char* out_ptr = out_indices->data1 + siztype * iRow;
                char* in_ptr = e_col->info2->data1 + siztype * ArrRow.second;
                memcpy(out_ptr, in_ptr, siztype);
                bit = e_col->info2->get_null_bit(ArrRow.second);
            }
            out_indices->set_null_bit(iRow, bit);
        }
        out_arr = new array_info(bodo_array_type::DICT, ref_column->dtype,
                                 out_indices->length, -1, -1, NULL, NULL, NULL,
                                 out_indices->null_bitmask,
                                 NULL, NULL, NULL, NULL, 0, 0, 0,
                                 ref_column->has_global_dictionary,
                                 ref_column->info1, out_indices);
        incref_array(ref_column->info1);
    }
    if (arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // In the case of NULLABLE array, we do a single loop for
        // assigning the arrays.
        // We do not need to reassign the pointers, only their size
        // suffices for the copy.
        // In the case of missing array a value of false is assigned
        // to the bitmask.
        out_arr = alloc_array(nRowOut, -1, -1, arr_type, dtype, 0, 0);
        uint64_t siztype = numpy_item_size[dtype];
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<array_info*, std::ptrdiff_t> ArrRow = get_iRow(iRow);
            bool bit = false;
            if (ArrRow.second >= 0) {
                array_info* e_col = ArrRow.first;
                char* out_ptr = out_arr->data1 + siztype * iRow;
                char* in_ptr = e_col->data1 + siztype * ArrRow.second;
                memcpy(out_ptr, in_ptr, siztype);
                bit = e_col->get_null_bit(ArrRow.second);
            }
            out_arr->set_null_bit(iRow, bit);
        }
    }
    if (arr_type == bodo_array_type::CATEGORICAL) {
        // In the case of CATEGORICAL array we have only to put a single
        // entry. For the NaN entry the value is -1.
        uint64_t siztype = numpy_item_size[dtype];
        std::vector<char> vectNaN =
            RetrieveNaNentry(dtype);  // returns a -1 for integer values.
        int64_t num_categories = ref_column->num_categories;
        out_arr = alloc_categorical(nRowOut, dtype, num_categories);
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<array_info*, std::ptrdiff_t> ArrRow = get_iRow(iRow);
            //
            char* out_ptr = out_arr->data1 + siztype * iRow;
            char* in_ptr;
            if (ArrRow.second >= 0)
                in_ptr = ArrRow.first->data1 + siztype * ArrRow.second;
            else
                in_ptr = vectNaN.data();
            memcpy(out_ptr, in_ptr, siztype);
        }
    }
    if (arr_type == bodo_array_type::NUMPY) {
        // In the case of NUMPY array we have only to put a single
        // entry.
        // In the case of missing data we have to assign a NaN and that is
        // not easy in general and done in the RetrieveNaNentry.
        // According to types:
        // ---signed integer: value -1
        // ---unsigned integer: value 0
        // ---floating point: std::nan as here both notions match.
        uint64_t siztype = numpy_item_size[dtype];
        if (!map_integer_type) {
            std::vector<char> vectNaN = RetrieveNaNentry(dtype);
            out_arr = alloc_array(nRowOut, -1, -1, arr_type, dtype, 0, 0);
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                std::pair<array_info*, std::ptrdiff_t> ArrRow = get_iRow(iRow);
                //
                char* out_ptr = out_arr->data1 + siztype * iRow;
                char* in_ptr;
                if (ArrRow.second >= 0)
                    in_ptr = ArrRow.first->data1 + siztype * ArrRow.second;
                else
                    in_ptr = vectNaN.data();
                memcpy(out_ptr, in_ptr, siztype);
            }
        } else {
            bodo_array_type::arr_type_enum arr_type_o =
                bodo_array_type::NULLABLE_INT_BOOL;
            out_arr = alloc_array(nRowOut, -1, -1, arr_type_o, dtype, 0, 0);
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                std::pair<array_info*, std::ptrdiff_t> ArrRow = get_iRow(iRow);
                //
                bool bit = false;
                if (ArrRow.second >= 0) {
                    char* out_ptr = out_arr->data1 + siztype * iRow;
                    char* in_ptr =
                        ArrRow.first->data1 + siztype * ArrRow.second;
                    memcpy(out_ptr, in_ptr, siztype);
                    bit = true;
                }
                out_arr->set_null_bit(iRow, bit);
            }
        }
    }
    if (arr_type == bodo_array_type::ARROW) {
        // Arrow builder for output array. builds it dynamically (buffer
        // sizes are not known in advance)
        std::unique_ptr<arrow::ArrayBuilder> builder;
        std::shared_ptr<arrow::Array> in_arr_typ = ref_column->array;
        (void)arrow::MakeBuilder(arrow::default_memory_pool(),
                                 in_arr_typ->type(), &builder);
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<array_info*, std::ptrdiff_t> ArrRow = get_iRow(iRow);
            if (ArrRow.second >= 0) {
                // non-null value for output
                std::shared_ptr<arrow::Array> in_arr = ArrRow.first->array;
                size_t row = ArrRow.second;
                // append value in position 'row' of input array to builder's
                // array (this is a recursive algorithm, can traverse nested
                // arrays)
                append_to_out_array(in_arr, row, row + 1, builder.get());
            } else {
                // null value for output
                (void)builder->AppendNull();
            }
        }

        // get final output array from builder
        std::shared_ptr<arrow::Array> out_arrow_array;
        // TODO: assert builder is not null (at least one row added)
        (void)builder->Finish(&out_arrow_array);
        out_arr =
            new array_info(bodo_array_type::ARROW, Bodo_CTypes::INT8 /*dummy*/,
                           nRowOut, -1, -1, NULL, NULL, NULL, NULL, NULL,
                           /*meminfo TODO*/ NULL, NULL, out_arrow_array);
    }
    return out_arr;
};

table_info* RetrieveTable(table_info* const& in_table,
                          std::vector<int64_t> const& ListIdx,
                          int const& n_cols_arg) {
    std::vector<array_info*> out_arrs;
    size_t n_cols;
    if (n_cols_arg == -1)
        n_cols = (size_t)in_table->ncols();
    else
        n_cols = n_cols_arg;
    for (size_t i_col = 0; i_col < n_cols; i_col++) {
        array_info* in_arr = in_table->columns[i_col];
        out_arrs.emplace_back(RetrieveArray_SingleColumn(in_arr, ListIdx));
        // Standard stealing of reference (see shuffle_table_kernel for
        // discussion)
        decref_array(in_arr);
    }
    return new table_info(out_arrs);
}

template <typename T>
std::pair<int, bool> process_arrow_bitmap(bool const& na_position_bis,
                                          T const& arrow1, int64_t pos1,
                                          T const& arrow2, int64_t pos2) {
    bool bit1 = !arrow1->IsNull(pos1);
    bool bit2 = !arrow2->IsNull(pos2);
    if (bit1 && !bit2) {
        int val = -1;
        if (na_position_bis) val = 1;
        return {val, true};
    }
    if (!bit1 && bit2) {
        int val = 1;
        if (na_position_bis) val = -1;
        return {val, true};
    }
    return {0, bit1};
}

int ComparisonArrowColumn(std::shared_ptr<arrow::Array> const& arr1,
                          int64_t pos1_s, int64_t pos1_e,
                          std::shared_ptr<arrow::Array> const& arr2,
                          int64_t pos2_s, int64_t pos2_e,
                          bool const& na_position_bis) {
    auto process_length = [](int const& len1, int const& len2) -> int {
        if (len1 > len2) return -1;
        if (len1 < len2) return 1;
        return 0;
    };
#if OFFSET_BITWIDTH == 32
    if (arr1->type_id() == arrow::Type::LIST) {
        std::shared_ptr<arrow::ListArray> list_array1 =
            std::dynamic_pointer_cast<arrow::ListArray>(arr1);
        std::shared_ptr<arrow::ListArray> list_array2 =
            std::dynamic_pointer_cast<arrow::ListArray>(arr2);
#else
    if (arr1->type_id() == arrow::Type::LARGE_LIST) {
        std::shared_ptr<arrow::LargeListArray> list_array1 =
            std::dynamic_pointer_cast<arrow::LargeListArray>(arr1);
        std::shared_ptr<arrow::LargeListArray> list_array2 =
            std::dynamic_pointer_cast<arrow::LargeListArray>(arr2);
#endif
        int64_t len1 = pos1_e - pos1_s;
        int64_t len2 = pos2_e - pos2_s;
        int64_t min_len = std::min(len1, len2);
        for (int64_t idx = 0; idx < min_len; idx++) {
            std::pair<int, bool> epair =
                process_arrow_bitmap(na_position_bis, list_array1, pos1_s + idx,
                                     list_array2, pos2_s + idx);
            if (epair.first != 0) return epair.first;
            if (epair.second) {
                int n_pos1_s = list_array1->value_offset(pos1_s + idx);
                int n_pos1_e = list_array1->value_offset(pos1_s + idx + 1);
                int n_pos2_s = list_array2->value_offset(pos2_s + idx);
                int n_pos2_e = list_array2->value_offset(pos2_s + idx + 1);
                int test = ComparisonArrowColumn(
                    list_array1->values(), n_pos1_s, n_pos1_e,
                    list_array2->values(), n_pos2_s, n_pos2_e, na_position_bis);
                if (test) return test;
            }
        }
        return process_length(len1, len2);
    } else if (arr1->type_id() == arrow::Type::STRUCT) {
        auto struct_array1 =
            std::dynamic_pointer_cast<arrow::StructArray>(arr1);
        auto struct_array2 =
            std::dynamic_pointer_cast<arrow::StructArray>(arr2);
        auto struct_type1 =
            std::dynamic_pointer_cast<arrow::StructType>(struct_array1->type());
        auto struct_type2 =
            std::dynamic_pointer_cast<arrow::StructType>(struct_array2->type());
        int num_fields = struct_type1->num_fields();
        int64_t len1 = pos1_e - pos1_s;
        int64_t len2 = pos2_e - pos2_s;
        int64_t min_len = std::min(len1, len2);
        for (int64_t idx = 0; idx < min_len; idx++) {
            int n_pos1_s = pos1_s + idx;
            int n_pos1_e = pos1_s + idx + 1;
            int n_pos2_s = pos2_s + idx;
            int n_pos2_e = pos2_s + idx + 1;
            std::pair<int, bool> epair =
                process_arrow_bitmap(na_position_bis, struct_array1, n_pos1_s,
                                     struct_array2, n_pos2_s);
            if (epair.first != 0) return epair.first;
            if (epair.second) {
                for (int i = 0; i < num_fields; i++) {
                    int test = ComparisonArrowColumn(
                        struct_array1->field(i), n_pos1_s, n_pos1_e,
                        struct_array2->field(i), n_pos2_s, n_pos2_e,
                        na_position_bis);
                    if (test) return test;
                }
            }
        }
        return process_length(len1, len2);
#if OFFSET_BITWIDTH == 32
    } else if (arr1->type_id() == arrow::Type::STRING) {
        auto str_array1 = std::dynamic_pointer_cast<arrow::StringArray>(arr1);
        auto str_array2 = std::dynamic_pointer_cast<arrow::StringArray>(arr2);
#else
    } else if (arr1->type_id() == arrow::Type::LARGE_STRING) {
        auto str_array1 =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(arr1);
        auto str_array2 =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(arr2);
#endif
        int64_t len1 = pos1_e - pos1_s;
        int64_t len2 = pos2_e - pos2_s;
        int64_t min_len = std::min(len1, len2);
        for (int64_t idx = 0; idx < min_len; idx++) {
            int n_pos1_s = pos1_s + idx;
            int n_pos2_s = pos2_s + idx;
            std::pair<int, bool> epair = process_arrow_bitmap(
                na_position_bis, str_array1, n_pos1_s, str_array2, n_pos2_s);
            if (epair.first != 0) return epair.first;
            if (epair.second) {
                std::string str1 = str_array1->GetString(pos1_s + idx);
                std::string str2 = str_array2->GetString(pos2_s + idx);
                size_t len1 = str1.size();
                size_t len2 = str2.size();
                size_t minlen = std::min(len1, len2);
                int test = std::memcmp(str2.c_str(), str1.c_str(), minlen);
                if (test) return test;
                // If not, we may be able to conclude via the string length.
                if (len1 > len2) return -1;
                if (len1 < len2) return 1;
            }
        }
        return process_length(len1, len2);
    } else {
        auto primitive_array1 =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(arr1);
        auto primitive_array2 =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(arr2);
        int64_t len1 = pos1_e - pos1_s;
        int64_t len2 = pos2_e - pos2_s;
        int64_t min_len = std::min(len1, len2);
        for (int64_t idx = 0; idx < min_len; idx++) {
            int n_pos1_s = pos1_s + idx;
            int n_pos2_s = pos2_s + idx;
            std::pair<int, bool> epair =
                process_arrow_bitmap(na_position_bis, primitive_array1,
                                     n_pos1_s, primitive_array2, n_pos2_s);
            if (epair.first != 0) return epair.first;
            if (epair.second) {
                int test = 0;
                // TODO: implement other types
                Bodo_CTypes::CTypeEnum bodo_typ = arrow_to_bodo_type(primitive_array1->type());
                size_t siz_typ = numpy_item_size[bodo_typ];
                char* ptr1 = (char*)primitive_array1->values()->data() +
                             siz_typ * n_pos1_s;
                char* ptr2 = (char*)primitive_array2->values()->data() +
                             siz_typ * n_pos2_s;
                test = NumericComparison(bodo_typ, ptr1, ptr2, na_position_bis);
                if (test) return test;
            }
        }
        return process_length(len1, len2);
    }
}

bool TestEqualColumn(array_info* arr1, int64_t pos1, array_info* arr2,
                     int64_t pos2, bool is_na_equal) {
    if (arr1->arr_type == bodo_array_type::ARROW) {
        // TODO: Handle is_na_equal in Arrow arrays
        int64_t pos1_s = pos1;
        int64_t pos1_e = pos1 + 1;
        int64_t pos2_s = pos2;
        int64_t pos2_e = pos2 + 1;
        bool na_position_bis = true;  // This value has no importance
        int val =
            ComparisonArrowColumn(arr1->array, pos1_s, pos1_e, arr2->array,
                                  pos2_s, pos2_e, na_position_bis);
        return val == 0;
    }
    if (arr1->arr_type == bodo_array_type::NUMPY ||
        arr1->arr_type == bodo_array_type::CATEGORICAL) {
        // In the case of NUMPY, we compare the values for concluding.
        uint64_t siztype = numpy_item_size[arr1->dtype];
        char* ptr1 = arr1->data1 + siztype * pos1;
        char* ptr2 = arr2->data1 + siztype * pos2;
        if (memcmp(ptr1, ptr2, siztype) != 0) return false;
        // Check for null if NA is not considered equal
        if (!is_na_equal) {
            if (arr1->arr_type == bodo_array_type::CATEGORICAL) {
                // Categorical null values are represented as -1
                return !isnan_categorical_ptr(arr1->dtype, ptr1);
            } else if (arr1->dtype == Bodo_CTypes::FLOAT32) {
                return !isnan(*((float*)ptr1));
            } else if (arr1->dtype == Bodo_CTypes::FLOAT64) {
                return !isnan(*((double*)ptr1));
            } else if (arr1->dtype == Bodo_CTypes::DATETIME ||
                       arr1->dtype == Bodo_CTypes::TIMEDELTA) {
                return (*((int64_t*)ptr1)) !=
                       std::numeric_limits<int64_t>::min();
            }
        }
    }
    if (arr1->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
        arr1->arr_type == bodo_array_type::DICT) {
        if (arr1->arr_type == bodo_array_type::DICT) {
            if (arr1->info1 != arr2->info1) {
                throw std::runtime_error(
                    "TestEqualColumn: don't know if arrays have unified "
                    "dictionary");
            }
            arr1 = arr1->info2;
            arr2 = arr2->info2;
        }
        // NULLABLE case. We need to consider the bitmask and the values.
        bool bit1 = arr1->get_null_bit(pos1);
        bool bit2 = arr2->get_null_bit(pos2);
        // If one bitmask is T and the other the reverse then they are
        // clearly not equal.
        if (bit1 != bit2) return false;
        // If both bitmasks are false, then it does not matter what value
        // they are storing. Comparison is the same as for NUMPY.
        if (bit1) {
            uint64_t siztype = numpy_item_size[arr1->dtype];
            char* ptr1 = arr1->data1 + siztype * pos1;
            char* ptr2 = arr2->data1 + siztype * pos2;
            if (memcmp(ptr1, ptr2, siztype) != 0) return false;
        } else {
            return is_na_equal;
        }
    }
    if (arr1->arr_type == bodo_array_type::LIST_STRING) {
        // For STRING case we need to deal bitmask and the values.
        bool bit1 = arr1->get_null_bit(pos1);
        bool bit2 = arr2->get_null_bit(pos2);
        uint8_t* sub_null_bitmask1 = (uint8_t*)arr1->sub_null_bitmask;
        uint8_t* sub_null_bitmask2 = (uint8_t*)arr2->sub_null_bitmask;
        // (1): If bitmasks are different then we conclude they are not equal.
        if (bit1 != bit2) return false;
        // If bitmasks are both false, then no need to compare the string
        // values.
        if (bit1) {
            // Here we consider the shifts in data2 for the comparison.
            offset_t* data3_1 = (offset_t*)arr1->data3;
            offset_t* data3_2 = (offset_t*)arr2->data3;
            offset_t len1 = data3_1[pos1 + 1] - data3_1[pos1];
            offset_t len2 = data3_2[pos2 + 1] - data3_2[pos2];
            // (2): If number of strings are different the they are different
            if (len1 != len2) return false;
            // (3): Checking that the string lengths are the same
            offset_t* data2_1 = (offset_t*)arr1->data2;
            offset_t* data2_2 = (offset_t*)arr2->data2;
            offset_t pos1_prev = data3_1[pos1];
            offset_t pos2_prev = data3_2[pos2];
            for (offset_t u = 0; u < len1; u++) {
                offset_t len1_str =
                    data2_1[pos1_prev + u + 1] - data2_1[pos1_prev + u];
                offset_t len2_str =
                    data2_2[pos2_prev + u + 1] - data2_2[pos2_prev + u];
                if (len1_str != len2_str) return false;
                bool str_bit1 = GetBit(sub_null_bitmask1, pos1_prev + u);
                bool str_bit2 = GetBit(sub_null_bitmask2, pos2_prev + u);
                if (str_bit1 != str_bit2) return false;
            }
            offset_t tot_nb_char =
                data2_1[data3_1[pos1 + 1]] - data2_1[data3_1[pos1]];
            // (4): Checking the data1 array
            offset_t pos1_B = data2_1[data3_1[pos1]];
            offset_t pos2_B = data2_2[data3_2[pos2]];
            char* data1_1_comp = arr1->data1 + pos1_B;
            char* data1_2_comp = arr2->data1 + pos2_B;
            if (memcmp(data1_1_comp, data1_2_comp, tot_nb_char) != 0)
                return false;
        } else {
            return is_na_equal;
        }
    }
    if (arr1->arr_type == bodo_array_type::STRING) {
        // For STRING case we need to deal bitmask and the values.
        bool bit1 = arr1->get_null_bit(pos1);
        bool bit2 = arr2->get_null_bit(pos2);
        // If bitmasks are different then we conclude they are not equal.
        if (bit1 != bit2) return false;
        // If bitmasks are both false, then no need to compare the string
        // values.
        if (bit1) {
            // Here we consider the shifts in data2 for the comparison.
            offset_t* data2_1 = (offset_t*)arr1->data2;
            offset_t* data2_2 = (offset_t*)arr2->data2;
            offset_t len1 = data2_1[pos1 + 1] - data2_1[pos1];
            offset_t len2 = data2_2[pos2 + 1] - data2_2[pos2];
            // If string lengths are different then they are different.
            if (len1 != len2) return false;
            // Now we iterate over the characters for the comparison.
            offset_t pos1_prev = data2_1[pos1];
            offset_t pos2_prev = data2_2[pos2];
            char* data1_1 = arr1->data1 + pos1_prev;
            char* data1_2 = arr2->data1 + pos2_prev;
            if (memcmp(data1_1, data1_2, len1) != 0) return false;
        } else {
            return is_na_equal;
        }
    }
    return true;
};

/** This function is used to determine if the value in a Categorical pointer
 * (pointer to a single value in a CategoricalArrayType) isnan.
 * @param the data type for the codes.
 * @param the Categorical Pointer
 * @returns if the value stored at the ptr is nan
 */
inline bool isnan_categorical_ptr(int dtype, char* ptr) {
    switch (dtype) {
        case Bodo_CTypes::INT8:
            return isnan_categorical<int8_t, Bodo_CTypes::INT8>(
                *((const int8_t*)ptr));
        case Bodo_CTypes::INT16:
            return isnan_categorical<int16_t, Bodo_CTypes::INT16>(
                *((const int16_t*)ptr));
        case Bodo_CTypes::INT32:
            return isnan_categorical<int32_t, Bodo_CTypes::INT32>(
                *((const int32_t*)ptr));
        case Bodo_CTypes::INT64:
            return isnan_categorical<int64_t, Bodo_CTypes::INT64>(
                *((const int64_t*)ptr));

        default:
            throw std::runtime_error(
                "_array_utils.h::NumericComparison: Invalid dtype put on "
                "CategoricalArrayType.");
    }
}

int KeyComparisonAsPython_Column(bool const& na_position_bis, array_info* arr1,
                                 size_t const& iRow1, array_info* arr2,
                                 size_t const& iRow2) {
    if (arr1->arr_type == bodo_array_type::ARROW) {
        int64_t pos1_s = iRow1;
        int64_t pos1_e = iRow1 + 1;
        int64_t pos2_s = iRow2;
        int64_t pos2_e = iRow2 + 1;
        return ComparisonArrowColumn(arr1->array, pos1_s, pos1_e, arr2->array,
                                     pos2_s, pos2_e, na_position_bis);
    }
    if (arr1->arr_type == bodo_array_type::NUMPY) {
        // In the case of NUMPY, we compare the values for concluding.
        uint64_t siztype = numpy_item_size[arr1->dtype];
        char* ptr1 = arr1->data1 + (siztype * iRow1);
        char* ptr2 = arr2->data1 + (siztype * iRow2);
        return NumericComparison(arr1->dtype, ptr1, ptr2, na_position_bis);
    }
    auto process_bits = [&](bool bit1, bool bit2) -> int {
        if (bit1 && !bit2) {
            if (na_position_bis) return 1;
            return -1;
        }
        if (!bit1 && bit2) {
            if (na_position_bis) return -1;
            return 1;
        }
        return 0;
    };
    if (arr1->arr_type == bodo_array_type::CATEGORICAL) {
        // In the case of CATEGORICAL, we need to check for null
        uint64_t siztype = numpy_item_size[arr1->dtype];
        char* ptr1 = arr1->data1 + (siztype * iRow1);
        char* ptr2 = arr2->data1 + (siztype * iRow2);
        bool is_not_na1 = !isnan_categorical_ptr(arr1->dtype, ptr1);
        bool is_not_na2 = !isnan_categorical_ptr(arr2->dtype, ptr2);
        int reply = process_bits(is_not_na1, is_not_na2);
        if (reply != 0) return reply;
        if (is_not_na1) {
            return NumericComparison(arr1->dtype, ptr1, ptr2, na_position_bis);
        }
    }
    if (arr1->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // NULLABLE case. We need to consider the bitmask and the values.
        uint8_t* null_bitmask1 = (uint8_t*)arr1->null_bitmask;
        uint8_t* null_bitmask2 = (uint8_t*)arr2->null_bitmask;
        bool bit1 = GetBit(null_bitmask1, iRow1);
        bool bit2 = GetBit(null_bitmask2, iRow2);
        // If one bitmask is T and the other the reverse then they are
        // clearly not equal.
        int reply = process_bits(bit1, bit2);
        if (reply != 0) return reply;
        // If both bitmasks are false, then it does not matter what value
        // they are storing. Comparison is the same as for NUMPY.
        if (bit1) {
            uint64_t siztype = numpy_item_size[arr1->dtype];
            char* ptr1 = arr1->data1 + (siztype * iRow1);
            char* ptr2 = arr2->data1 + (siztype * iRow2);
            int test =
                NumericComparison(arr1->dtype, ptr1, ptr2, na_position_bis);
            if (test) return test;
        }
    }
    if (arr1->arr_type == bodo_array_type::LIST_STRING) {
        // For LIST_STRING case we need to deal bitmask and the values.
        bool bit1 = arr1->get_null_bit(iRow1);
        bool bit2 = arr2->get_null_bit(iRow2);
        uint8_t* sub_null_bitmask1 = (uint8_t*)arr1->sub_null_bitmask;
        uint8_t* sub_null_bitmask2 = (uint8_t*)arr2->sub_null_bitmask;
        // If bitmasks are different then we can conclude the comparison
        int reply = process_bits(bit1, bit2);
        if (reply != 0) return reply;
        if (bit1) {  // here bit1 = bit2
            offset_t* data3_1 = (offset_t*)arr1->data3;
            offset_t* data3_2 = (offset_t*)arr2->data3;
            offset_t* data2_1 = (offset_t*)arr1->data2;
            offset_t* data2_2 = (offset_t*)arr2->data2;
            // Computing the number of strings and their minimum
            offset_t nb_string1 = data3_1[iRow1 + 1] - data3_1[iRow1];
            offset_t nb_string2 = data3_2[iRow2 + 1] - data3_2[iRow2];
            offset_t min_nb_string = nb_string1;
            if (nb_string2 < nb_string1) min_nb_string = nb_string2;
            // Iterating over the number of strings.
            for (size_t istr = 0; istr < min_nb_string; istr++) {
                size_t pos1_prev = data2_1[istr + data3_1[iRow1]];
                size_t pos2_prev = data2_2[istr + data3_2[iRow2]];
                size_t len1 = data2_1[istr + 1 + data3_1[iRow1]] -
                              data2_1[istr + data3_1[iRow1]];
                size_t len2 = data2_2[istr + 1 + data3_2[iRow2]] -
                              data2_2[istr + data3_2[iRow2]];
                bool str_bit1 =
                    GetBit(sub_null_bitmask1, data3_1[iRow1] + istr);
                bool str_bit2 =
                    GetBit(sub_null_bitmask2, data3_2[iRow2] + istr);
                int reply_str = process_bits(str_bit1, str_bit2);
                if (reply_str != 0) return reply_str;
                if (str_bit1) {  // here str_bit1 = str_bit2
                    offset_t minlen = len1;
                    if (len2 < len1) minlen = len2;
                    char* data1_1 = arr1->data1 + pos1_prev;
                    char* data1_2 = arr2->data1 + pos2_prev;
                    // We check the strings for comparison and check if we can
                    // conclude.
                    int test = std::memcmp(data1_2, data1_1, minlen);
                    if (test) return test;
                    // If not, we may be able to conclude via their length
                    if (len1 > len2) return -1;
                    if (len1 < len2) return 1;
                }
            }
            // If the number of strings is different then we can conclude.
            if (nb_string1 > nb_string2) return -1;
            if (nb_string1 < nb_string2) return 1;
        }
    }
    if (arr1->arr_type == bodo_array_type::STRING) {
        // For STRING case we need to deal bitmask and the values.
        bool bit1 = arr1->get_null_bit(iRow1);
        bool bit2 = arr2->get_null_bit(iRow2);
        // If bitmasks are different then we can conclude the comparison
        int reply = process_bits(bit1, bit2);
        if (reply != 0) return reply;
        // If bitmasks are both false, then no need to compare the string
        // values.
        if (bit1) {
            // Here we consider the shifts in data2 for the comparison.
            offset_t* data2_1 = (offset_t*)arr1->data2;
            offset_t* data2_2 = (offset_t*)arr2->data2;
            offset_t len1 = data2_1[iRow1 + 1] - data2_1[iRow1];
            offset_t len2 = data2_2[iRow2 + 1] - data2_2[iRow2];
            // Compute minimal length
            offset_t minlen = len1;
            if (len2 < len1) minlen = len2;
            // From the common characters, we may be able to conclude.
            offset_t pos1_prev = data2_1[iRow1];
            offset_t pos2_prev = data2_2[iRow2];
            char* data1_1 = (char*)arr1->data1 + pos1_prev;
            char* data1_2 = (char*)arr2->data1 + pos2_prev;
            int test = std::memcmp(data1_2, data1_1, minlen);
            if (test) return test;
            // If not, we may be able to conclude via the string length.
            if (len1 > len2) return -1;
            if (len1 < len2) return 1;
        }
    }
    return 0;
}

bool KeyComparisonAsPython(size_t const& n_key, int64_t* vect_ascending,
                           std::vector<array_info*> const& columns1,
                           size_t const& shift_key1, size_t const& iRow1,
                           std::vector<array_info*> const& columns2,
                           size_t const& shift_key2, size_t const& iRow2,
                           int64_t* na_position) {
    // iteration over the list of key for the comparison.
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        bool ascending = vect_ascending[iKey];
        bool na_last = na_position[iKey];
        bool na_position_bis = (!na_last) ^ ascending;
        int test = KeyComparisonAsPython_Column(
            na_position_bis, columns1[shift_key1 + iKey], iRow1,
            columns2[shift_key2 + iKey], iRow2);
        if (test) {
            if (ascending) return test > 0;
            return test < 0;
        }
    }
    // If all keys are equal then we return false
    return false;
};

// ----------------------- Debug functions -----------------------

/** Printing the string expression of an entry in the column
 *
 * @param dtype: the data type on input
 * @param ptrdata: The pointer to the data (its length is determined by dtype)
 * @return The string on output.
 */
std::string GetStringExpression(Bodo_CTypes::CTypeEnum const& dtype,
                                char* ptrdata, int scale) {
    if (dtype == Bodo_CTypes::_BOOL) {
        bool* ptr = (bool*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::INT8) {
        int8_t* ptr = (int8_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::UINT8) {
        uint8_t* ptr = (uint8_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::INT16) {
        int16_t* ptr = (int16_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::UINT16) {
        uint16_t* ptr = (uint16_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::INT32) {
        int32_t* ptr = (int32_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::UINT32) {
        uint32_t* ptr = (uint32_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::INT64) {
        int64_t* ptr = (int64_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::UINT64) {
        uint64_t* ptr = (uint64_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::DATE || dtype == Bodo_CTypes::DATETIME ||
        dtype == Bodo_CTypes::TIMEDELTA) {
        int64_t* ptr = (int64_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::FLOAT32) {
        float* ptr = (float*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::DECIMAL) {
        decimal_value_cpp* val = (decimal_value_cpp*)ptrdata;
        return decimal_value_cpp_to_std_string(*val, scale);
    }
    if (dtype == Bodo_CTypes::FLOAT64) {
        double* ptr = (double*)ptrdata;
        return std::to_string(*ptr);
    }
    return "no matching type";
}

template <typename T>
void DEBUG_append_to_primitive_T(const T* values, int64_t offset,
                                 int64_t length, std::string& string_builder,
                                 const std::vector<uint8_t>& valid_elems) {
    string_builder += "[";
    for (int64_t i = 0; i < length; i++) {
        if (i > 0) string_builder += ",";
        if (valid_elems[i]) {
            T val = values[offset + i];
            string_builder += std::to_string(val);
        } else {
            string_builder += "None";
        }
    }
    string_builder += "]";
}

void DEBUG_append_to_primitive_decimal(
    const decimal_value_cpp* values, int64_t offset, int64_t length,
    std::string& string_builder, const std::vector<uint8_t>& valid_elems) {
    string_builder += "[";
    for (int64_t i = 0; i < length; i++) {
        if (i > 0) string_builder += ",";
        if (valid_elems[i]) {
            decimal_value_cpp val = values[offset + i];
            int scale = 18;
            string_builder += decimal_value_cpp_to_std_string(val, scale);
        } else {
            string_builder += "None";
        }
    }
    string_builder += "]";
}

void DEBUG_append_to_primitive(arrow::Type::type const& type,
                               const uint8_t* values, int64_t offset,
                               int64_t length, std::string& string_builder,
                               const std::vector<uint8_t>& valid_elems) {
    if (type == arrow::Type::INT8) {
        DEBUG_append_to_primitive_T((int8_t*)values, offset, length,
                                    string_builder, valid_elems);
    } else if (type == arrow::Type::UINT8) {
        DEBUG_append_to_primitive_T((uint8_t*)values, offset, length,
                                    string_builder, valid_elems);
    } else if (type == arrow::Type::INT16) {
        DEBUG_append_to_primitive_T((int16_t*)values, offset, length,
                                    string_builder, valid_elems);
    } else if (type == arrow::Type::UINT16) {
        DEBUG_append_to_primitive_T((uint16_t*)values, offset, length,
                                    string_builder, valid_elems);
    } else if (type == arrow::Type::INT32) {
        DEBUG_append_to_primitive_T((int32_t*)values, offset, length,
                                    string_builder, valid_elems);
    } else if (type == arrow::Type::UINT32) {
        DEBUG_append_to_primitive_T((uint32_t*)values, offset, length,
                                    string_builder, valid_elems);
    } else if (type == arrow::Type::INT64) {
        DEBUG_append_to_primitive_T((int64_t*)values, offset, length,
                                    string_builder, valid_elems);
    } else if (type == arrow::Type::UINT64) {
        DEBUG_append_to_primitive_T((uint64_t*)values, offset, length,
                                    string_builder, valid_elems);
    } else if (type == arrow::Type::FLOAT) {
        DEBUG_append_to_primitive_T((float*)values, offset, length,
                                    string_builder, valid_elems);
    } else if (type == arrow::Type::DOUBLE) {
        DEBUG_append_to_primitive_T((double*)values, offset, length,
                                    string_builder, valid_elems);
    } else if (type == arrow::Type::DECIMAL) {
        DEBUG_append_to_primitive_decimal((decimal_value_cpp*)values, offset,
                                          length, string_builder, valid_elems);
    } else {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Unsupported primitive type building arrow array");
    }
}

void DEBUG_append_to_out_array(std::shared_ptr<arrow::Array> input_array,
                               int64_t start_offset, int64_t end_offset,
                               std::string& string_builder) {
    // TODO check for nulls and append nulls
#if OFFSET_BITWIDTH == 32
    if (input_array->type_id() == arrow::Type::LIST) {
        std::shared_ptr<arrow::ListArray> list_array =
            std::dynamic_pointer_cast<arrow::ListArray>(input_array);
#else
    if (input_array->type_id() == arrow::Type::LARGE_LIST) {
        std::shared_ptr<arrow::LargeListArray> list_array =
            std::dynamic_pointer_cast<arrow::LargeListArray>(input_array);
#endif

        string_builder += "[";
        for (int64_t idx = start_offset; idx < end_offset; idx++) {
            if (idx > start_offset) {
                string_builder += ", ";
            }
            if (list_array->IsNull(idx)) {
                string_builder += "None";
            } else {
                DEBUG_append_to_out_array(
                    list_array->values(), list_array->value_offset(idx),
                    list_array->value_offset(idx + 1), string_builder);
            }
        }
        string_builder += "]";
    } else if (input_array->type_id() == arrow::Type::STRUCT) {
        // TODO: assert builder.type() == STRUCT
        auto struct_array =
            std::dynamic_pointer_cast<arrow::StructArray>(input_array);
        auto struct_type =
            std::dynamic_pointer_cast<arrow::StructType>(struct_array->type());
        for (int64_t idx = start_offset; idx < end_offset; idx++) {
            if (idx > start_offset) string_builder += ",";
            if (struct_array->IsNull(idx)) {
                string_builder += "None";
                continue;
            }
            string_builder += "{";
            for (int i = 0; i < struct_type->num_fields();
                 i++) {  // each field is an array
                if (i > 0) string_builder += ", ";
                DEBUG_append_to_out_array(struct_array->field(i), idx, idx + 1,
                                          string_builder);
            }
            string_builder += "}";
        }
#if OFFSET_BITWIDTH == 32
    } else if (input_array->type_id() == arrow::Type::STRING) {
        auto str_array =
            std::dynamic_pointer_cast<arrow::StringArray>(input_array);
#else
    } else if (input_array->type_id() == arrow::Type::LARGE_STRING) {
        auto str_array =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(input_array);
#endif
        string_builder += "[";
        for (int64_t i = start_offset; i < end_offset; i++) {
            if (i > 0) string_builder += ", ";
            if (str_array->IsNull(i))
                string_builder += "None";
            else
                string_builder += "\"" + str_array->GetString(i) + "\"";
        }
        string_builder += "]";
    } else {
        int64_t num_elems = end_offset - start_offset;
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(input_array);
        std::vector<uint8_t> valid_elems(num_elems, 0);
        size_t j = 0;
        for (int64_t i = start_offset; i < start_offset + num_elems; i++)
            valid_elems[j++] = !primitive_array->IsNull(i);
        arrow::Type::type type = primitive_array->type()->id();
        DEBUG_append_to_primitive(type, primitive_array->values()->data(),
                                  start_offset, num_elems, string_builder,
                                  valid_elems);
    }
}

#undef DEBUG_DEBUG  // Yes, it is a concept

std::vector<std::string> GetColumn_as_ListString(array_info* arr) {
    size_t nRow = arr->length;
    std::vector<std::string> ListStr(nRow);
    std::string strOut;
#ifdef DEBUG_DEBUG
    std::cout << "Beginning of DEBUG_PrintColumn\n";
#endif
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
#ifdef DEBUG_DEBUG
        std::cout << "DEBUG, Case NULLABLE_INT_BOOL\n";
#endif
        uint64_t siztype = numpy_item_size[arr->dtype];
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = arr->get_null_bit(iRow);
            if (bit) {
                char* ptrdata1 = &(arr->data1[siztype * iRow]);
                strOut = GetStringExpression(arr->dtype, ptrdata1, arr->scale);
            } else {
                strOut = "false";
            }
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::NUMPY) {
#ifdef DEBUG_DEBUG
        std::cout << "DEBUG, Case NUMPY\n";
#endif
        uint64_t siztype = numpy_item_size[arr->dtype];
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            char* ptrdata1 = &(arr->data1[siztype * iRow]);
            strOut = GetStringExpression(arr->dtype, ptrdata1, arr->scale);
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::STRING) {
#ifdef DEBUG_DEBUG
        std::cout << "DEBUG, Case STRING\n";
#endif
        offset_t* data2 = (offset_t*)arr->data2;
        char* data1 = arr->data1;
#ifdef DEBUG_DEBUG
        std::cout << "DEBUG, We have the pointers\n";
#endif
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = arr->get_null_bit(iRow);
            if (bit) {
                offset_t start_pos = data2[iRow];
                offset_t end_pos = data2[iRow + 1];
                offset_t len = end_pos - start_pos;
#ifdef DEBUG_DEBUG
                std::cout << "start_pos=" << start_pos << " end_pos=" << end_pos
                          << " len=" << len << "\n";
#endif
                char* strname;
                strname = new char[len + 1];
                for (offset_t i = 0; i < len; i++) {
                    strname[i] = data1[start_pos + i];
                }
                strname[len] = '\0';
                strOut = strname;
                delete[] strname;
            } else {
                strOut = "false";
            }
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::LIST_STRING) {
#ifdef DEBUG_DEBUG
        std::cout << "DEBUG, Case LIST_STRING\n";
#endif
        offset_t* index_offset = (offset_t*)arr->data3;
        offset_t* data_offset = (offset_t*)arr->data2;
        uint8_t* sub_null_bitmask = (uint8_t*)arr->sub_null_bitmask;
        char* data1 = arr->data1;
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = arr->get_null_bit(iRow);
            if (bit) {
                strOut = "[";
                offset_t len = index_offset[iRow + 1] - index_offset[iRow];
                for (offset_t u = 0; u < len; u++) {
                    bool str_bit =
                        GetBit(sub_null_bitmask, index_offset[iRow] + u);
                    if (u > 0) strOut += ",";
                    if (str_bit) {
                        offset_t start_pos =
                            data_offset[index_offset[iRow] + u];
                        offset_t end_pos =
                            data_offset[index_offset[iRow] + u + 1];
                        offset_t lenB = end_pos - start_pos;
                        char* strname;
                        strname = new char[lenB + 1];
                        for (offset_t i = 0; i < lenB; i++)
                            strname[i] = data1[start_pos + i];
                        strname[lenB] = '\0';
                        strOut += strname;
                        delete[] strname;
                    } else {
                        strOut += "None";
                    }
                }
                strOut += "]";
            } else {
                strOut = "false";
            }
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::ARROW) {
#ifdef DEBUG_DEBUG
        std::cout << "DEBUG, Case ARROW\n";
#endif
        std::shared_ptr<arrow::Array> in_arr = arr->array;
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            strOut = "";
            DEBUG_append_to_out_array(in_arr, iRow, iRow + 1, strOut);
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::CATEGORICAL) {
#ifdef DEBUG_DEBUG
        std::cout << "DEBUG, Case CATEGORICAL\n";
#endif
        uint64_t siztype = numpy_item_size[arr->dtype];
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            char* ptrdata1 = &(arr->data1[siztype * iRow]);
            strOut = GetStringExpression(arr->dtype, ptrdata1, arr->scale);
            ListStr[iRow] = strOut;
        }
    }
    return ListStr;
}

void DEBUG_PrintVectorArrayInfo(std::ostream& os,
                                std::vector<array_info*> const& ListArr) {
    int nCol = ListArr.size();
    if (nCol == 0) {
        os << "number of columns nCol=" << nCol << " Nothing to print\n";
        return;
    }
    std::vector<int> ListLen(nCol);
    int nRowMax = 0;
    for (int iCol = 0; iCol < nCol; iCol++) {
        int nRow = ListArr[iCol]->length;
        if (nRow > nRowMax) nRowMax = nRow;
        ListLen[iCol] = nRow;
    }
    std::vector<std::vector<std::string>> ListListStr;
    for (int iCol = 0; iCol < nCol; iCol++) {
        std::vector<std::string> LStr = GetColumn_as_ListString(ListArr[iCol]);
        for (int iRow = ListLen[iCol]; iRow < nRowMax; iRow++)
            LStr.emplace_back("");
        ListListStr.emplace_back(LStr);
    }
    std::vector<std::string> ListStrOut(nRowMax);
    for (int iRow = 0; iRow < nRowMax; iRow++) {
        std::string str = std::to_string(iRow) + " :";
        ListStrOut[iRow] = str;
    }
    for (int iCol = 0; iCol < nCol; iCol++) {
        std::vector<int> ListLen(nRowMax);
        size_t maxlen = 0;
        for (int iRow = 0; iRow < nRowMax; iRow++) {
            size_t elen = ListListStr[iCol][iRow].size();
            ListLen[iRow] = elen;
            if (elen > maxlen) maxlen = elen;
        }
        for (int iRow = 0; iRow < nRowMax; iRow++) {
            std::string str = ListStrOut[iRow] + " " + ListListStr[iCol][iRow];
            size_t diff = maxlen - ListLen[iRow];
            for (size_t u = 0; u < diff; u++) str += " ";
            ListStrOut[iRow] = str;
        }
    }
    for (int iRow = 0; iRow < nRowMax; iRow++) os << ListStrOut[iRow] << "\n";
}

void DEBUG_PrintSetOfColumn(std::ostream& os,
                            std::vector<array_info*> const& ListArr) {
    int nCol = ListArr.size();
    if (nCol == 0) {
        os << "number of columns nCol=" << nCol << " Nothing to print\n";
        return;
    }
    std::vector<int> ListLen(nCol);
    int nRowMax = 0;
    os << "nCol=" << nCol << " List of number of rows:";
    for (int iCol = 0; iCol < nCol; iCol++) {
        int nRow = ListArr[iCol]->length;
        os << " " << nRow;
        if (nRow > nRowMax) nRowMax = nRow;
        ListLen[iCol] = nRow;
    }
    os << "\n";
    DEBUG_PrintVectorArrayInfo(os, ListArr);
}

std::string GetDtype_as_string(Bodo_CTypes::CTypeEnum const& dtype) {
    if (dtype == Bodo_CTypes::INT8) return "INT8";
    if (dtype == Bodo_CTypes::UINT8) return "UINT8";
    if (dtype == Bodo_CTypes::INT16) return "INT16";
    if (dtype == Bodo_CTypes::UINT16) return "UINT16";
    if (dtype == Bodo_CTypes::INT32) return "INT32";
    if (dtype == Bodo_CTypes::UINT32) return "UINT32";
    if (dtype == Bodo_CTypes::INT64) return "INT64";
    if (dtype == Bodo_CTypes::UINT64) return "UINT64";
    if (dtype == Bodo_CTypes::FLOAT32) return "FLOAT32";
    if (dtype == Bodo_CTypes::FLOAT64) return "FLOAT64";
    if (dtype == Bodo_CTypes::STRING) return "STRING";
    if (dtype == Bodo_CTypes::BINARY) return "BINARY";
    if (dtype == Bodo_CTypes::_BOOL) return "_BOOL";
    if (dtype == Bodo_CTypes::DECIMAL) return "DECIMAL";
    if (dtype == Bodo_CTypes::DATE) return "DATE";
    if (dtype == Bodo_CTypes::DATETIME) return "DATETIME";
    if (dtype == Bodo_CTypes::TIMEDELTA) return "TIMEDELTA";
    return "unmatching dtype";
}

std::string GetArrType_as_string(bodo_array_type::arr_type_enum arr_type) {
    if (arr_type == bodo_array_type::NUMPY) return "NUMPY";
    if (arr_type == bodo_array_type::STRING) return "STRING";
    if (arr_type == bodo_array_type::NULLABLE_INT_BOOL) return "NULLABLE";
    if (arr_type == bodo_array_type::LIST_STRING) return "LIST_STRING";
    if (arr_type == bodo_array_type::ARROW) return "ARROW";
    if (arr_type == bodo_array_type::CATEGORICAL) return "CATEGORICAL";
    return "Uncovered case of GetDtypeString\n";
}

void DEBUG_PrintRefct(std::ostream& os,
                      std::vector<array_info*> const& ListArr) {
    int nCol = ListArr.size();
    auto GetNRTinfo = [](NRT_MemInfo* meminf) -> std::string {
        if (meminf == NULL) return "NULL";
        return "(refct=" + std::to_string(meminf->refct) + ")";
    };
    for (int iCol = 0; iCol < nCol; iCol++) {
        os << "iCol=" << iCol << " : "
           << GetArrType_as_string(ListArr[iCol]->arr_type)
           << " dtype=" << GetDtype_as_string(ListArr[iCol]->dtype)
           << " : meminfo=" << GetNRTinfo(ListArr[iCol]->meminfo)
           << " meminfo_bitmask=" << GetNRTinfo(ListArr[iCol]->meminfo_bitmask)
           << "\n";
    }
}

void DEBUG_PrintColumn(std::ostream& os, array_info* arr) {
    int n_rows = arr->length;
    os << "ARRAY_INFO: Column n=" << n_rows
       << " arr=" << GetArrType_as_string(arr->arr_type)
       << " dtype=" << GetDtype_as_string(arr->dtype) << "\n";
    std::vector<std::string> LStr = GetColumn_as_ListString(arr);
    for (int i_row = 0; i_row < n_rows; i_row++)
        os << "i_row=" << i_row << " S=" << LStr[i_row] << "\n";
}

void DEBUG_PrintColumn(std::ostream& os, multiple_array_info* arr) {
    int n_rows = arr->length;
    os << "MULTIPLE_ARRAY_INFO: Column n=" << n_rows
       << " arr=" << GetArrType_as_string(arr->arr_type)
       << " dtype=" << GetDtype_as_string(arr->dtype) << "\n";
    std::vector<array_info*> total_arr = arr->vect_arr;
    for (auto& earr : arr->vect_access) total_arr.push_back(earr);
    DEBUG_PrintVectorArrayInfo(os, total_arr);
}

/**
 * Used for a custom reduction to merge all the HyperLogLog registers across all
 * ranks.
 *
 * The body of this function is what is done in the `HyperLogLog.merge()`
 * function with some decoration to deal with MPI.
 */
void MPI_hyper_log_log_merge(void* in, void* inout, int* len,
                             MPI_Datatype* dptr) {
    uint8_t* M_in = reinterpret_cast<uint8_t*>(in);
    uint8_t* M_inout = reinterpret_cast<uint8_t*>(inout);
    // The loop below comes from libs/hyperloglog.hpp:merge() (currently like
    // 161)
    for (int r = 0; r < *len; ++r) {
        if (M_inout[r] < M_in[r]) {
            M_inout[r] |= M_in[r];
        }
    }
}

// Passing bit width = 20 to HyperLogLog (impacts accuracy and execution
// time). 30 is extremely slow. 20 seems to be about as fast as 10 and
// more accurate. With 20 it is pretty fast, faster than calculating
// the hashes with our MurmurHash3_x64_32, and uses 1 MB of memory
#define HLL_SIZE 20

size_t get_nunique_hashes(uint32_t const* const hashes, const size_t len,
                          bool is_parallel) {
    tracing::Event ev("get_nunique_hashes", is_parallel);
    hll::HyperLogLog hll(HLL_SIZE);
    hll.addAll(hashes, len);
    const size_t est = std::min(static_cast<size_t>(hll.estimate()), len);
    ev.add_attribute("estimate", est);
    return est;
}

std::pair<size_t, size_t> get_nunique_hashes_global(
    uint32_t const* const hashes, const size_t len, bool is_parallel) {
    tracing::Event ev("get_nunique_hashes_global", is_parallel);
    tracing::Event ev_local("get_nunique_hashes_local", is_parallel);
    hll::HyperLogLog hll(HLL_SIZE);
    hll.addAll(hashes, len);
    size_t local_est = static_cast<size_t>(hll.estimate());
    ev.add_attribute("local_estimate", local_est);
    ev_local.finalize();

    // To get a global estimate of the cardinality we first do a custom
    // reduction of the "registers" in the HyperLogLog. Once the registers
    // have been reduced to rank 0, we overwrite the local register in the
    // hll object, and then compute the estimate.
    //
    // Note: the merge for HLL-HIP is much more complicated and so isn't a
    // trivial replacement. It certainly could be tested, but would require
    // writing more code.
    int my_rank = std::numeric_limits<int>::max();
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    std::vector<uint8_t> hll_registers(hll.data().size(), 0);
    {
        MPI_Op mpi_hll_op;
        MPI_Op_create(&MPI_hyper_log_log_merge, true, &mpi_hll_op);
        MPI_Reduce(hll.data().data(), hll_registers.data(), hll.data().size(),
                   MPI_UNSIGNED_CHAR, mpi_hll_op, 0, MPI_COMM_WORLD);
        MPI_Op_free(&mpi_hll_op);
    }

    // Cast to known MPI-compatible type since size_t is implementation
    // defined.
    unsigned long local_len = static_cast<unsigned long>(len);
    unsigned long global_len = 0;
    MPI_Allreduce(&local_len, &global_len, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                  MPI_COMM_WORLD);

    unsigned long est;
    if (my_rank == 0) {
        hll.overwrite_registers(std::move(hll_registers));
        using std::min;
        est = min(global_len, static_cast<unsigned long>(hll.estimate()));
    }
    MPI_Bcast(&est, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    // cast to `size_t` to avoid compilation error on Windows
    ev.add_attribute("g_global_estimate", static_cast<size_t>(est));
    return {local_est, est};
}
