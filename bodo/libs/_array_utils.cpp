// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_array_utils.h"
#include "_decimal_ext.h"
#include <iostream>
#include <string>

/**
 * Append values from a byte buffer to a primitive array builder.
 * @param values: pointer to buffer containing data
 * @param offset: offset of starting element (not byte units)
 * @param length: number of elements to copy
 * @param builder: primitive array builder holding output array
 * @param valid_elems: non-zero if elem i is non-null
 */
void append_to_primitive(const uint8_t *values, int64_t offset, int64_t length,
                         arrow::ArrayBuilder *builder,
                         const std::vector<uint8_t> &valid_elems) {
    if (builder->type()->id() == arrow::Type::INT32) {
        auto typed_builder = dynamic_cast<arrow::NumericBuilder<arrow::Int32Type>*>(builder);
        typed_builder->AppendValues((int32_t*)values + offset, length, valid_elems.data());
    } else if (builder->type()->id() == arrow::Type::DOUBLE) {
        auto typed_builder = dynamic_cast<arrow::NumericBuilder<arrow::DoubleType>*>(builder);
        typed_builder->AppendValues((double*)values + offset, length, valid_elems.data());
    } else {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Unsupported primitive type building arrow array");
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
                         arrow::ArrayBuilder *builder) {
    // TODO check for nulls and append nulls
    if (input_array->type_id() == arrow::Type::LIST) {
        // TODO: assert builder.type() == LIST
        std::shared_ptr<arrow::ListArray> list_array = std::dynamic_pointer_cast<arrow::ListArray>(input_array);
        auto list_builder = dynamic_cast<arrow::ListBuilder*>(builder);
        arrow::ArrayBuilder *child_builder = list_builder->value_builder();

        for (int64_t idx = start_offset; idx < end_offset; idx++) {
            if (list_array->IsNull(idx)) {
                list_builder->AppendNull();
                continue;
            }
            list_builder->Append();  // indicate list boundary
            // TODO optimize
            append_to_out_array(list_array->values(),  // child array
                                list_array->value_offset(idx), list_array->value_offset(idx + 1),
                                child_builder);
        }
    } else if (input_array->type_id() == arrow::Type::STRUCT) {
        // TODO: assert builder.type() == STRUCT
        auto struct_array = std::dynamic_pointer_cast<arrow::StructArray>(input_array);
        auto struct_type = std::dynamic_pointer_cast<arrow::StructType>(struct_array->type());
        auto struct_builder = dynamic_cast<arrow::StructBuilder*>(builder);
        for (int64_t idx = start_offset; idx < end_offset; idx++) {
            if (struct_array->IsNull(idx)) {
                struct_builder->AppendNull();
                continue;
            }
            for (int i=0; i < struct_type->num_children(); i++) {  // each field is an array
                arrow::ArrayBuilder *field_builder = builder->child(i);
                append_to_out_array(struct_array->field(i), idx, idx + 1, field_builder);
            }
            struct_builder->Append();
        }
        // finished appending (end_offset - start_offset) structs
        //struct_builder->AppendValues(end_offset - start_offset, NULLPTR);
    } else if (input_array->type_id() == arrow::Type::STRING) {
        auto str_array = std::dynamic_pointer_cast<arrow::StringArray>(input_array);
        auto str_builder = dynamic_cast<arrow::StringBuilder*>(builder);
        int64_t num_elems = end_offset - start_offset;
        // TODO: optimize
        for (int64_t i=0; i < num_elems; i++) {
            if (str_array->IsNull(start_offset + i))
                str_builder->AppendNull();
            else
                str_builder->AppendValues({str_array->GetString(start_offset + i)});
        }
    } else {
        int64_t num_elems = end_offset - start_offset;
        // assume this is array of primitive values
        // TODO: decimal, date, etc.
        auto primitive_array = std::dynamic_pointer_cast<arrow::PrimitiveArray>(input_array);
        std::vector<uint8_t> valid_elems(num_elems, 0);
        // TODO: more efficient way of getting null data?
        size_t j=0;
        for (int64_t i = start_offset; i < start_offset + num_elems; i++)
            valid_elems[j++] = !primitive_array->IsNull(i);
        append_to_primitive(primitive_array->values()->data(), start_offset, num_elems, builder, valid_elems);
    }
}

array_info* RetrieveArray(
    table_info* const& in_table,
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> const& ListPairWrite,
    size_t const& shift1, size_t const& shift2, int const& ChoiceColumn,
    bool const& map_integer_type) {
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
        [&](size_t const& iRowIn) -> std::pair<size_t, std::ptrdiff_t> {
        std::pair<std::ptrdiff_t, std::ptrdiff_t> pairLRcolumn =
            ListPairWrite[iRowIn];
        if (ChoiceColumn == 0) return {shift1, pairLRcolumn.first};
        if (ChoiceColumn == 1) return {shift2, pairLRcolumn.second};
        if (pairLRcolumn.first != -1) return {shift1, pairLRcolumn.first};
        return {shift2, pairLRcolumn.second};
    };
    // eshift is the in_table index used for the determination
    // of arr_type and dtype of the returned column.
    size_t eshift;
    if (ChoiceColumn == 0) eshift = shift1;
    if (ChoiceColumn == 1) eshift = shift2;
    if (ChoiceColumn == 2) eshift = shift1;
    bodo_array_type::arr_type_enum arr_type =
        in_table->columns[eshift]->arr_type;
    Bodo_CTypes::CTypeEnum dtype = in_table->columns[eshift]->dtype;
    if (arr_type == bodo_array_type::LIST_STRING) {
        // In the first case of STRING, we have to deal with offsets first so we
        // need one first loop to determine the needed length. In the second
        // loop, the assignation is made. If the entries are missing then the
        // bitmask is set to false.
        std::vector<uint32_t> ListSizes_index(nRowOut);
        std::vector<uint32_t> ListSizes_data(nRowOut);
        int64_t tot_size_index = 0;
        int64_t tot_size_data = 0;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
            uint32_t size_index = 0;
            uint32_t size_data = 0;
            if (pairShiftRow.second >= 0) {
                uint32_t* index_offsets =
                    (uint32_t*)in_table->columns[pairShiftRow.first]->data3;
                uint32_t* data_offsets =
                    (uint32_t*)in_table->columns[pairShiftRow.first]->data2;
                uint32_t start_offset_index = index_offsets[pairShiftRow.second];
                uint32_t end_offset_index = index_offsets[pairShiftRow.second + 1];
                size_index = end_offset_index - start_offset_index;
                uint32_t start_offset_data = data_offsets[start_offset_index];
                uint32_t end_offset_data = data_offsets[end_offset_index];
                size_data = end_offset_data - start_offset_data;
            }
            ListSizes_index[iRow] = size_index;
            ListSizes_data[iRow] = size_data;
            tot_size_index += size_index;
            tot_size_data += size_data;
        }
        out_arr = alloc_array(nRowOut, tot_size_index, tot_size_data, arr_type, dtype, 0);
        uint8_t* out_null_bitmask = (uint8_t*)out_arr->null_bitmask;
        uint32_t pos_index = 0;
        uint32_t pos_data = 0;
        uint32_t* out_index_offsets = (uint32_t*)out_arr->data3;
        uint32_t* out_data_offsets = (uint32_t*)out_arr->data2;
        out_data_offsets[0] = 0;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
            uint32_t size_index = ListSizes_index[iRow];
            uint32_t size_data  = ListSizes_data[iRow];
            out_index_offsets[iRow] = pos_index;
            bool bit = false;
            if (pairShiftRow.second >= 0) {
                size_t i_col = pairShiftRow.first;
                size_t i_row = pairShiftRow.second;
                uint8_t* in_null_bitmask = (uint8_t*)in_table->columns[i_col]->null_bitmask;
                uint32_t* in_index_offsets = (uint32_t*)in_table->columns[i_col]->data3;
                uint32_t* in_data_offsets = (uint32_t*)in_table->columns[i_col]->data2;
                char* data1 = in_table->columns[i_col]->data1;
                uint32_t start_index_offset = in_index_offsets[i_row];
                uint32_t start_data_offset = in_data_offsets[start_index_offset];
                for (uint32_t u=0; u<size_index; u++) {
                  uint32_t len_str = in_data_offsets[start_index_offset+u+1] - in_data_offsets[start_index_offset+u];
                  out_data_offsets[pos_index+u+1] = out_data_offsets[pos_index+u] + len_str;
                }
                memcpy(&out_arr->data1[pos_data], &data1[start_data_offset], size_data);
                bit = GetBit(in_null_bitmask, pairShiftRow.second);
            }
            pos_index += size_index;
            pos_data += size_data;
            SetBitTo(out_null_bitmask, iRow, bit);
        }
        out_index_offsets[nRowOut] = pos_index;
    }
    if (arr_type == bodo_array_type::STRING) {
        // In the first case of STRING, we have to deal with offsets first so we
        // need one first loop to determine the needed length. In the second
        // loop, the assignation is made. If the entries are missing then the
        // bitmask is set to false.
        int64_t n_chars = 0;
        std::vector<uint32_t> ListSizes(nRowOut);
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
            uint32_t size = 0;
            if (pairShiftRow.second >= 0) {
                uint32_t* in_offsets =
                    (uint32_t*)in_table->columns[pairShiftRow.first]->data2;
                uint32_t end_offset = in_offsets[pairShiftRow.second + 1];
                uint32_t start_offset = in_offsets[pairShiftRow.second];
                size = end_offset - start_offset;
            }
            ListSizes[iRow] = size;
            n_chars += size;
        }
        out_arr = alloc_array(nRowOut, n_chars, -1, arr_type, dtype, 0);
        uint8_t* out_null_bitmask = (uint8_t*)out_arr->null_bitmask;
        uint32_t pos = 0;
        uint32_t* out_offsets = (uint32_t*)out_arr->data2;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
            uint32_t size = ListSizes[iRow];
            out_offsets[iRow] = pos;
            bool bit = false;
            if (pairShiftRow.second >= 0) {
                uint8_t* in_null_bitmask =
                    (uint8_t*)in_table->columns[pairShiftRow.first]
                        ->null_bitmask;
                uint32_t* in_offsets =
                    (uint32_t*)in_table->columns[pairShiftRow.first]->data2;
                uint32_t start_offset = in_offsets[pairShiftRow.second];
                for (uint32_t i = 0; i < size; i++) {
                    out_arr->data1[pos] = in_table->columns[pairShiftRow.first]
                                              ->data1[start_offset];
                    pos++;
                    start_offset++;
                }
                bit = GetBit(in_null_bitmask, pairShiftRow.second);
            }
            SetBitTo(out_null_bitmask, iRow, bit);
        }
        out_offsets[nRowOut] = pos;
    }
    if (arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // In the case of NULLABLE array, we do a single loop for
        // assigning the arrays.
        // We do not need to reassign the pointers, only their size
        // suffices for the copy.
        // In the case of missing array a value of false is assigned
        // to the bitmask.
        out_arr = alloc_array(nRowOut, -1, -1, arr_type, dtype, 0);
        uint8_t* out_null_bitmask = (uint8_t*)out_arr->null_bitmask;
        uint64_t siztype = numpy_item_size[dtype];
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
            bool bit = false;
            if (pairShiftRow.second >= 0) {
                uint8_t* in_null_bitmask =
                    (uint8_t*)in_table->columns[pairShiftRow.first]
                        ->null_bitmask;
                for (uint64_t u = 0; u < siztype; u++)
                    out_arr->data1[siztype * iRow + u] =
                        in_table->columns[pairShiftRow.first]
                            ->data1[siztype * pairShiftRow.second + u];
                bit = GetBit(in_null_bitmask, pairShiftRow.second);
            }
            SetBitTo(out_null_bitmask, iRow, bit);
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
            out_arr = alloc_array(nRowOut, -1, -1, arr_type, dtype, 0);
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
                //
                if (pairShiftRow.second >= 0) {
                    for (uint64_t u = 0; u < siztype; u++)
                        out_arr->data1[siztype * iRow + u] =
                            in_table->columns[pairShiftRow.first]->data1[siztype * pairShiftRow.second + u];
                } else {
                    for (uint64_t u = 0; u < siztype; u++)
                        out_arr->data1[siztype * iRow + u] = vectNaN[u];
                }
            }
        }
        else {
            bodo_array_type::arr_type_enum arr_type_o = bodo_array_type::NULLABLE_INT_BOOL;
            out_arr = alloc_array(nRowOut, -1, -1, arr_type_o, dtype, 0);
            uint8_t* out_null_bitmask = (uint8_t*)out_arr->null_bitmask;
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
                //
                bool bit = false;
                if (pairShiftRow.second >= 0) {
                    for (uint64_t u = 0; u < siztype; u++)
                        out_arr->data1[siztype * iRow + u] =
                            in_table->columns[pairShiftRow.first]
                                ->data1[siztype * pairShiftRow.second + u];
                    bit = true;
                }
                SetBitTo(out_null_bitmask, iRow, bit);
            }
        }
    }
    if (arr_type == bodo_array_type::ARROW) {
        // Arrow builder for output array. builds it dynamically (buffer
        // sizes are not known in advance)
        std::unique_ptr<arrow::ArrayBuilder> builder;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
            if (pairShiftRow.second >= 0) {
                // non-null value for output
                std::shared_ptr<arrow::Array> in_arr = in_table->columns[pairShiftRow.first]->array;
                if (!builder)
                    // make array builder of same type as the input array.
                    // works for nested arrays of any type
                    arrow::MakeBuilder(arrow::default_memory_pool(), in_arr->type(), &builder);
                size_t row = pairShiftRow.second;
                // append value in position 'row' of input array to builder's array
                // (this is a recursive algorithm, can traverse nested arrays)
                append_to_out_array(in_arr, row, row + 1, builder.get());
            } else {
                // null value for output
                if (!builder) {
                    std::shared_ptr<arrow::Array> in_arr = in_table->columns[pairShiftRow.first]->array;
                    arrow::MakeBuilder(arrow::default_memory_pool(), in_arr->type(), &builder);
                }
                builder->AppendNull();
            }
        }

        // get final output array from builder
        std::shared_ptr<arrow::Array> out_arrow_array;
        // TODO: assert builder is not null (at least one row added)
        builder->Finish(&out_arrow_array);
        out_arr = new array_info(bodo_array_type::ARROW, Bodo_CTypes::INT8/*dummy*/, -1,
                      -1, -1, NULL, NULL, NULL, NULL, /*meminfo TODO*/NULL,
                      NULL, out_arrow_array);
    }
    return out_arr;
};

table_info* RetrieveTable(
    table_info* const& in_table,
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> const&
    ListPairWrite, int const& n_cols_arg) {
    std::vector<array_info*> out_arrs;
    bool map_integer_type = false;
    size_t n_cols;
    if (n_cols_arg == -1)
      n_cols = (size_t)in_table->ncols();
    else
      n_cols = n_cols_arg;
    for (size_t i_col = 0; i_col < n_cols; i_col++)
        out_arrs.emplace_back(RetrieveArray(in_table, ListPairWrite, i_col, -1,
                                            0, map_integer_type));
    return new table_info(out_arrs);
}

bool TestEqualColumn(array_info* arr1, int64_t pos1, array_info* arr2,
                     int64_t pos2) {
    if (arr1->arr_type == bodo_array_type::NUMPY) {
        // In the case of NUMPY, we compare the values for concluding.
        uint64_t siztype = numpy_item_size[arr1->dtype];
        char* ptr1 = arr1->data1 + siztype * pos1;
        char* ptr2 = arr2->data1 + siztype * pos2;
        if (memcmp(ptr1, ptr2, siztype) != 0) return false;
    }
    if (arr1->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // NULLABLE case. We need to consider the bitmask and the values.
        uint8_t* null_bitmask1 = (uint8_t*)arr1->null_bitmask;
        uint8_t* null_bitmask2 = (uint8_t*)arr2->null_bitmask;
        bool bit1 = GetBit(null_bitmask1, pos1);
        bool bit2 = GetBit(null_bitmask2, pos2);
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
        }
    }
    if (arr1->arr_type == bodo_array_type::LIST_STRING) {
        // For STRING case we need to deal bitmask and the values.
        uint8_t* null_bitmask1 = (uint8_t*)arr1->null_bitmask;
        uint8_t* null_bitmask2 = (uint8_t*)arr2->null_bitmask;
        bool bit1 = GetBit(null_bitmask1, pos1);
        bool bit2 = GetBit(null_bitmask2, pos2);
        // (1): If bitmasks are different then we conclude they are not equal.
        if (bit1 != bit2) return false;
        // If bitmasks are both false, then no need to compare the string
        // values.
        if (bit1) {
            // Here we consider the shifts in data2 for the comparison.
            uint32_t* data3_1 = (uint32_t*)arr1->data3;
            uint32_t* data3_2 = (uint32_t*)arr2->data3;
            uint32_t len1 = data3_1[pos1 + 1] - data3_1[pos1];
            uint32_t len2 = data3_2[pos2 + 1] - data3_2[pos2];
            // (2): If number of strings are different the they are different
            if (len1 != len2) return false;
            // (3): Checking that the string lengths are the same
            uint32_t* data2_1 = (uint32_t*)arr1->data2;
            uint32_t* data2_2 = (uint32_t*)arr2->data2;
            uint32_t pos1_prev = data3_1[pos1];
            uint32_t pos2_prev = data3_2[pos2];
            for (uint32_t u=0; u<len1; u++) {
              uint32_t len1_str = data2_1[pos1_prev + u + 1] - data2_1[pos1_prev + u];
              uint32_t len2_str = data2_2[pos2_prev + u + 1] - data2_2[pos2_prev + u];
              if (len1_str != len2_str) return false;
            }
            uint32_t tot_nb_char = data2_1[data3_1[pos1 + 1]] - data2_1[data3_1[pos1]];
            // (4): Checking the data1 array
            uint32_t pos1_B = data2_1[data3_1[pos1]];
            uint32_t pos2_B = data2_2[data3_2[pos2]];
            char* data1_1_comp = arr1->data1 + pos1_B;
            char* data1_2_comp = arr2->data1 + pos2_B;
            if (memcmp(data1_1_comp, data1_2_comp, tot_nb_char) != 0) return false;
        }
    }
    if (arr1->arr_type == bodo_array_type::STRING) {
        // For STRING case we need to deal bitmask and the values.
        uint8_t* null_bitmask1 = (uint8_t*)arr1->null_bitmask;
        uint8_t* null_bitmask2 = (uint8_t*)arr2->null_bitmask;
        bool bit1 = GetBit(null_bitmask1, pos1);
        bool bit2 = GetBit(null_bitmask2, pos2);
        // If bitmasks are different then we conclude they are not equal.
        if (bit1 != bit2) return false;
        // If bitmasks are both false, then no need to compare the string
        // values.
        if (bit1) {
            // Here we consider the shifts in data2 for the comparison.
            uint32_t* data2_1 = (uint32_t*)arr1->data2;
            uint32_t* data2_2 = (uint32_t*)arr2->data2;
            uint32_t len1 = data2_1[pos1 + 1] - data2_1[pos1];
            uint32_t len2 = data2_2[pos2 + 1] - data2_2[pos2];
            // If string lengths are different then they are different.
            if (len1 != len2) return false;
            // Now we iterate over the characters for the comparison.
            uint32_t pos1_prev = data2_1[pos1];
            uint32_t pos2_prev = data2_2[pos2];
            char* data1_1 = arr1->data1 + pos1_prev;
            char* data1_2 = arr2->data1 + pos2_prev;
            if (memcmp(data1_1, data1_2, len1) != 0) return false;
        }
    }
    return true;
};

bool KeyComparisonAsPython(size_t const& n_key, int64_t* vect_ascending,
                           std::vector<array_info*> const& columns1,
                           size_t const& shift_key1, size_t const& iRow1,
                           std::vector<array_info*> const& columns2,
                           size_t const& shift_key2, size_t const& iRow2,
                           bool const& na_position) {
    // iteration over the list of key for the comparison.
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        bool ascending = vect_ascending[iKey];
        auto ProcessOutput = [&](int const& value) -> bool {
            if (ascending) {
                return value > 0;
            }
            return value < 0;
        };
        bool na_position_bis = (!na_position) ^ ascending;
        if (columns1[shift_key1 + iKey]->arr_type == bodo_array_type::NUMPY) {
            // In the case of NUMPY, we compare the values for concluding.
            uint64_t siztype =
                numpy_item_size[columns1[shift_key1 + iKey]->dtype];
            char* ptr1 = columns1[shift_key1 + iKey]->data1 + (siztype * iRow1);
            char* ptr2 = columns2[shift_key2 + iKey]->data1 + (siztype * iRow2);
            int test = NumericComparison(columns1[shift_key1 + iKey]->dtype,
                                         ptr1, ptr2, na_position_bis);
            if (test != 0) return ProcessOutput(test);
        }
        if (columns1[shift_key1 + iKey]->arr_type ==
            bodo_array_type::NULLABLE_INT_BOOL) {
            // NULLABLE case. We need to consider the bitmask and the values.
            uint8_t* null_bitmask1 =
                (uint8_t*)columns1[shift_key1 + iKey]->null_bitmask;
            uint8_t* null_bitmask2 =
                (uint8_t*)columns2[shift_key2 + iKey]->null_bitmask;
            bool bit1 = GetBit(null_bitmask1, iRow1);
            bool bit2 = GetBit(null_bitmask2, iRow2);
            // If one bitmask is T and the other the reverse then they are
            // clearly not equal.
            if (bit1 && !bit2) {
                if (na_position_bis) return ProcessOutput(1);
                return ProcessOutput(-1);
            }
            if (!bit1 && bit2) {
                if (na_position_bis) return ProcessOutput(-1);
                return ProcessOutput(1);
            }
            // If both bitmasks are false, then it does not matter what value
            // they are storing. Comparison is the same as for NUMPY.
            if (bit1) {
                uint64_t siztype =
                    numpy_item_size[columns1[shift_key1 + iKey]->dtype];
                char* ptr1 =
                    columns1[shift_key1 + iKey]->data1 + (siztype * iRow1);
                char* ptr2 =
                    columns2[shift_key2 + iKey]->data1 + (siztype * iRow2);
                int test = NumericComparison(columns1[shift_key1 + iKey]->dtype,
                                             ptr1, ptr2, na_position_bis);
                if (test != 0) return ProcessOutput(test);
            }
        }
        if (columns1[shift_key1 + iKey]->arr_type == bodo_array_type::LIST_STRING) {
            // For LIST_STRING case we need to deal bitmask and the values.
            uint8_t* null_bitmask1 =
                (uint8_t*)columns1[shift_key1 + iKey]->null_bitmask;
            uint8_t* null_bitmask2 =
                (uint8_t*)columns2[shift_key2 + iKey]->null_bitmask;
            bool bit1 = GetBit(null_bitmask1, iRow1);
            bool bit2 = GetBit(null_bitmask2, iRow2);
            // If bitmasks are different then we can conclude the comparison
            if (bit1 && !bit2) {
                if (na_position_bis) return ProcessOutput(1);
                return ProcessOutput(-1);
            }
            if (!bit1 && bit2) {
                if (na_position_bis) return ProcessOutput(-1);
                return ProcessOutput(1);
            }
            if (bit1) {
                uint32_t* data3_1 =
                    (uint32_t*)columns1[shift_key1 + iKey]->data3;
                uint32_t* data3_2 =
                    (uint32_t*)columns2[shift_key2 + iKey]->data3;
                uint32_t* data2_1 =
                    (uint32_t*)columns1[shift_key1 + iKey]->data2;
                uint32_t* data2_2 =
                    (uint32_t*)columns2[shift_key2 + iKey]->data2;
                // Computing the number of strings and their minimum
                uint32_t nb_string1 = data3_1[iRow1 + 1] - data3_1[iRow1];
                uint32_t nb_string2 = data3_2[iRow2 + 1] - data3_2[iRow2];
                uint32_t min_nb_string = nb_string1;
                if (nb_string2 < nb_string1) min_nb_string = nb_string2;
                // Iterating over the number of strings.
                for (size_t istr=0; istr<min_nb_string; istr++) {
                    size_t pos1_prev = data2_1[istr + data3_1[iRow1]];
                    size_t pos2_prev = data2_2[istr + data3_2[iRow2]];
                    size_t len1=data2_1[istr + 1 + data3_1[iRow1]] - data2_1[istr + data3_1[iRow1]];
                    size_t len2=data2_2[istr + 1 + data3_2[iRow2]] - data2_2[istr + data3_2[iRow2]];
                    uint32_t minlen = len1;
                    if (len2 < len1) minlen = len2;
                    char* data1_1 = columns1[shift_key1 + iKey]->data1 + pos1_prev;
                    char* data1_2 = columns2[shift_key2 + iKey]->data1 + pos2_prev;
                    // We check the strings for comparison and check if we can conclude.
                    int test = std::strncmp(data1_2, data1_1, minlen);
                    if (test != 0) return ProcessOutput(test);
                    // If not, we may be able to conclude via their length
                    if (len1 > len2) return ProcessOutput(-1);
                    if (len1 < len2) return ProcessOutput(1);
                }
                // If the number of strings is different then we can conclude.
                if (nb_string1 > nb_string2) return ProcessOutput(-1);
                if (nb_string1 < nb_string2) return ProcessOutput(1);
            }
        }
        if (columns1[shift_key1 + iKey]->arr_type == bodo_array_type::STRING) {
            // For STRING case we need to deal bitmask and the values.
            uint8_t* null_bitmask1 =
                (uint8_t*)columns1[shift_key1 + iKey]->null_bitmask;
            uint8_t* null_bitmask2 =
                (uint8_t*)columns2[shift_key2 + iKey]->null_bitmask;
            bool bit1 = GetBit(null_bitmask1, iRow1);
            bool bit2 = GetBit(null_bitmask2, iRow2);
            // If bitmasks are different then we can conclude the comparison
            if (bit1 && !bit2) {
                if (na_position_bis) return ProcessOutput(1);
                return ProcessOutput(-1);
            }
            if (!bit1 && bit2) {
                if (na_position_bis) return ProcessOutput(-1);
                return ProcessOutput(1);
            }
            // If bitmasks are both false, then no need to compare the string
            // values.
            if (bit1) {
                // Here we consider the shifts in data2 for the comparison.
                uint32_t* data2_1 =
                    (uint32_t*)columns1[shift_key1 + iKey]->data2;
                uint32_t* data2_2 =
                    (uint32_t*)columns2[shift_key2 + iKey]->data2;
                uint32_t len1 = data2_1[iRow1 + 1] - data2_1[iRow1];
                uint32_t len2 = data2_2[iRow2 + 1] - data2_2[iRow2];
                // Compute minimal length
                uint32_t minlen = len1;
                if (len2 < len1) minlen = len2;
                // From the common characters, we may be able to conclude.
                uint32_t pos1_prev = data2_1[iRow1];
                uint32_t pos2_prev = data2_2[iRow2];
                char* data1_1 =
                    (char*)columns1[shift_key1 + iKey]->data1 + pos1_prev;
                char* data1_2 =
                    (char*)columns2[shift_key2 + iKey]->data1 + pos2_prev;
                int test = std::strncmp(data1_2, data1_1, minlen);
                if (test != 0) return ProcessOutput(test);
                // If not, we may be able to conclude via the string length.
                if (len1 > len2) return ProcessOutput(-1);
                if (len1 < len2) return ProcessOutput(1);
            }
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
    if (dtype == Bodo_CTypes::DATE ||
        dtype == Bodo_CTypes::DATETIME ||
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

std::vector<std::string> DEBUG_PrintColumn(array_info* arr) {
    size_t nRow = arr->length;
    std::vector<std::string> ListStr(nRow);
    std::string strOut;
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        uint64_t siztype = numpy_item_size[arr->dtype];
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = GetBit(null_bitmask, iRow);
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
        uint64_t siztype = numpy_item_size[arr->dtype];
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            char* ptrdata1 = &(arr->data1[siztype * iRow]);
            strOut = GetStringExpression(arr->dtype, ptrdata1, arr->scale);
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::STRING) {
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        uint32_t* data2 = (uint32_t*)arr->data2;
        char* data1 = arr->data1;
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = GetBit(null_bitmask, iRow);
            if (bit) {
                uint32_t start_pos = data2[iRow];
                uint32_t end_pos = data2[iRow + 1];
                uint32_t len = end_pos - start_pos;
                char* strname;
                strname = new char[len + 1];
                for (uint32_t i = 0; i < len; i++) {
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
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        uint32_t* data3 = (uint32_t*)arr->data3;
        uint32_t* data2 = (uint32_t*)arr->data2;
        char* data1 = arr->data1;
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = GetBit(null_bitmask, iRow);
            if (bit) {
                strOut = "[";
                uint32_t len=data3[iRow+1] - data3[iRow];
                for (uint32_t u=0; u<len; u++) {
                    uint32_t start_pos = data2[data3[iRow] + u];
                    uint32_t end_pos = data2[data3[iRow] + u + 1];
                    uint32_t lenB = end_pos - start_pos;
                    char* strname;
                    strname = new char[lenB + 1];
                    for (uint32_t i = 0; i < lenB; i++) {
                        strname[i] = data1[start_pos + i];
                    }
                    strname[lenB] = '\0';
                    if (u>0)
                        strOut += ",";
                    strOut += strname;
                    delete[] strname;
                }
                strOut += "]";
            } else {
                strOut = "false";
            }
            ListStr[iRow] = strOut;
        }
    }
    return ListStr;
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
    os << "List of number of rows:";
    for (int iCol = 0; iCol < nCol; iCol++) {
        int nRow = ListArr[iCol]->length;
        os << " " << nRow;
        if (nRow > nRowMax) nRowMax = nRow;
        ListLen[iCol] = nRow;
    }
    os << "\n";
    std::vector<std::vector<std::string>> ListListStr;
    for (int iCol = 0; iCol < nCol; iCol++) {
        std::vector<std::string> LStr = DEBUG_PrintColumn(ListArr[iCol]);
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


std::string GetDtype_as_string(Bodo_CTypes::CTypeEnum const& dtype)
{
  if (dtype == Bodo_CTypes::INT8)
    return "INT8";
  if (dtype == Bodo_CTypes::UINT8)
    return "UINT8";
  if (dtype == Bodo_CTypes::INT16)
    return "INT16";
  if (dtype == Bodo_CTypes::UINT16)
    return "UINT16";
  if (dtype == Bodo_CTypes::INT32)
    return "INT32";
  if (dtype == Bodo_CTypes::UINT32)
    return "UINT32";
  if (dtype == Bodo_CTypes::INT64)
    return "INT64";
  if (dtype == Bodo_CTypes::UINT64)
    return "UINT64";
  if (dtype == Bodo_CTypes::FLOAT32)
    return "FLOAT32";
  if (dtype == Bodo_CTypes::FLOAT64)
    return "FLOAT64";
  if (dtype == Bodo_CTypes::STRING)
    return "STRING";
  if (dtype == Bodo_CTypes::_BOOL)
    return "_BOOL";
  if (dtype == Bodo_CTypes::DECIMAL)
    return "DECIMAL";
  if (dtype == Bodo_CTypes::DATE)
    return "DATE";
  if (dtype == Bodo_CTypes::DATETIME)
    return "DATETIME";
  if (dtype == Bodo_CTypes::TIMEDELTA)
    return "TIMEDELTA";
  return "unmatching dtype";
}

void DEBUG_PrintRefct(std::ostream& os,
                      std::vector<array_info*> const& ListArr) {
    int nCol = ListArr.size();
    auto GetType = [](bodo_array_type::arr_type_enum arr_type) -> std::string {
        if (arr_type == bodo_array_type::NULLABLE_INT_BOOL) return "NULLABLE";
        if (arr_type == bodo_array_type::NUMPY) return "NUMPY";
        if (arr_type == bodo_array_type::STRING) return "STRING";
        if (arr_type == bodo_array_type::LIST_STRING) return "LIST_STRING";
        return "Uncovered case in DEBUG_PrintRefct\n";
    };
    auto GetNRTinfo = [](NRT_MemInfo* meminf) -> std::string {
        if (meminf == NULL) return "NULL";
        return "(refct=" + std::to_string(meminf->refct) + ")";
    };
    for (int iCol = 0; iCol < nCol; iCol++) {
        os << "iCol=" << iCol << " : " << GetType(ListArr[iCol]->arr_type)
           << " dtype=" << GetDtype_as_string(ListArr[iCol]->dtype)
           << " : meminfo=" << GetNRTinfo(ListArr[iCol]->meminfo)
           << " meminfo_bitmask=" << GetNRTinfo(ListArr[iCol]->meminfo_bitmask)
           << "\n";
    }
}
