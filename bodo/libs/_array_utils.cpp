// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_array_utils.h"
#include <iostream>
#include <string>

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
        out_arr = alloc_array(nRowOut, n_chars, arr_type, dtype, 0);
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
        out_arr = alloc_array(nRowOut, -1, arr_type, dtype, 0);
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
            out_arr = alloc_array(nRowOut, -1, arr_type, dtype, 0);
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
                //
                if (pairShiftRow.second >= 0) {
                    for (uint64_t u = 0; u < siztype; u++)
                        out_arr->data1[siztype * iRow + u] =
                            in_table->columns[pairShiftRow.first]
                                ->data1[siztype * pairShiftRow.second + u];
                } else {
                    for (uint64_t u = 0; u < siztype; u++)
                        out_arr->data1[siztype * iRow + u] = vectNaN[u];
                }
            }
        } else {
            bodo_array_type::arr_type_enum arr_type_o =
                bodo_array_type::NULLABLE_INT_BOOL;
            out_arr = alloc_array(nRowOut, -1, arr_type_o, dtype, 0);
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
                                char* ptrdata) {
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
    if (dtype == Bodo_CTypes::UINT64 || dtype == Bodo_CTypes::DATE ||
        dtype == Bodo_CTypes::DATETIME) {
        uint64_t* ptr = (uint64_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::FLOAT32) {
        float* ptr = (float*)ptrdata;
        return std::to_string(*ptr);
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
                strOut = GetStringExpression(arr->dtype, ptrdata1);
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
            strOut = GetStringExpression(arr->dtype, ptrdata1);
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
    return ListStr;
}

void DEBUG_PrintSetOfColumn(std::ostream& os,
                            std::vector<array_info*> const& ListArr) {
    int nCol = ListArr.size();
    if (nCol == 0) {
        os << "Nothing to print really\n";
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

void DEBUG_PrintRefct(std::ostream& os,
                      std::vector<array_info*> const& ListArr) {
    int nCol = ListArr.size();
    auto GetType = [](bodo_array_type::arr_type_enum arr_type) -> std::string {
        if (arr_type == bodo_array_type::NULLABLE_INT_BOOL) return "NULLABLE";
        if (arr_type == bodo_array_type::NUMPY) return "NUMPY";
        return "STRING";
    };
    auto GetNRTinfo = [](NRT_MemInfo* meminf) -> std::string {
        if (meminf == NULL) return "NULL";
        return "(refct=" + std::to_string(meminf->refct) + ")";
    };
    for (int iCol = 0; iCol < nCol; iCol++) {
        os << "iCol=" << iCol << " : " << GetType(ListArr[iCol]->arr_type)
           << " dtype=" << ListArr[iCol]->dtype
           << " : meminfo=" << GetNRTinfo(ListArr[iCol]->meminfo)
           << " meminfo_bitmask=" << GetNRTinfo(ListArr[iCol]->meminfo_bitmask)
           << "\n";
    }
}
