#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "streaming/_join.h"

#include <arrow/array/array_nested.h>
#include <arrow/builder.h>
#include <arrow/compute/cast.h>
#include <arrow/type_fwd.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <complex>
#include <iostream>
#include <span>
#include <string>
#include <unordered_map>

#include "_datetime_utils.h"
#include "_decimal_ext.h"
#include "_mpi.h"
#include "_table_builder_utils.h"
#include "vendored/hyperloglog.hpp"

/**
 * Append values from a byte buffer to a primitive array builder.
 * @param primitive_array: The arrow primitive array
 * @param offset: offset of starting element (not byte units)
 * @param length: number of elements to copy
 * @param builder: primitive array builder holding output array
 * @param valid_elems: non-zero if elem i is non-null
 */
void append_to_primitive(
    std::shared_ptr<arrow::PrimitiveArray> const& primitive_array,
    int64_t offset, int64_t length, arrow::ArrayBuilder* builder,
    const std::span<const uint8_t> valid_elems) {
    arrow::Type::type typ = builder->type()->id();
    const uint8_t* values = primitive_array->values()->data();
    if (typ == arrow::Type::BOOL) {
        auto typed_builder = static_cast<arrow::BooleanBuilder*>(builder);
        // Boolean arrays reuse the null bitmap data
        (void)typed_builder->AppendValues((uint8_t*)values, length,
                                          primitive_array->null_bitmap_data(),
                                          offset);
    } else if (typ == arrow::Type::INT8) {
        auto typed_builder =
            static_cast<arrow::NumericBuilder<arrow::Int8Type>*>(builder);
        (void)typed_builder->AppendValues((int8_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::UINT8) {
        auto typed_builder =
            static_cast<arrow::NumericBuilder<arrow::UInt8Type>*>(builder);
        (void)typed_builder->AppendValues((uint8_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::INT16) {
        auto typed_builder =
            static_cast<arrow::NumericBuilder<arrow::Int16Type>*>(builder);
        (void)typed_builder->AppendValues((int16_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::UINT16) {
        auto typed_builder =
            static_cast<arrow::NumericBuilder<arrow::UInt16Type>*>(builder);
        (void)typed_builder->AppendValues((uint16_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::INT32) {
        auto typed_builder =
            static_cast<arrow::NumericBuilder<arrow::Int32Type>*>(builder);
        (void)typed_builder->AppendValues((int32_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::UINT32) {
        auto typed_builder =
            static_cast<arrow::NumericBuilder<arrow::UInt32Type>*>(builder);
        (void)typed_builder->AppendValues((uint32_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::INT64) {
        auto typed_builder =
            static_cast<arrow::NumericBuilder<arrow::Int64Type>*>(builder);
        (void)typed_builder->AppendValues((int64_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::UINT64) {
        auto typed_builder =
            static_cast<arrow::NumericBuilder<arrow::UInt64Type>*>(builder);
        (void)typed_builder->AppendValues((uint64_t*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::FLOAT) {
        auto typed_builder =
            static_cast<arrow::NumericBuilder<arrow::FloatType>*>(builder);
        (void)typed_builder->AppendValues((float*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::DOUBLE) {
        auto typed_builder =
            static_cast<arrow::NumericBuilder<arrow::DoubleType>*>(builder);
        (void)typed_builder->AppendValues((double*)values + offset, length,
                                          valid_elems.data());
    } else if (typ == arrow::Type::DECIMAL) {
        auto typed_builder = static_cast<arrow::Decimal128Builder*>(builder);
        (void)typed_builder->AppendValues(
            (uint8_t*)values + BYTES_PER_DECIMAL * offset, length,
            valid_elems.data());
    } else if (typ == arrow::Type::DATE32) {
        auto typed_builder = static_cast<arrow::Date32Builder*>(builder);
        (void)typed_builder->AppendValues((int32_t*)values + offset, length,
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
        auto list_builder = static_cast<arrow::ListBuilder*>(builder);
#else
    if (input_array->type_id() == arrow::Type::LARGE_LIST) {
        // TODO: assert builder.type() == LARGE_LIST
        std::shared_ptr<arrow::LargeListArray> list_array =
            std::dynamic_pointer_cast<arrow::LargeListArray>(input_array);
        auto list_builder = static_cast<arrow::LargeListBuilder*>(builder);
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
        auto struct_builder = static_cast<arrow::StructBuilder*>(builder);
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
        auto str_builder = static_cast<arrow::StringBuilder*>(builder);
#else
    } else if (input_array->type_id() == arrow::Type::LARGE_STRING) {
        auto str_array =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(input_array);
        auto str_builder = static_cast<arrow::LargeStringBuilder*>(builder);
#endif
        int64_t num_elems = end_offset - start_offset;
        // TODO: optimize
        for (int64_t i = 0; i < num_elems; i++) {
            if (str_array->IsNull(start_offset + i)) {
                (void)str_builder->AppendNull();
            } else {
                (void)str_builder->AppendValues(
                    {str_array->GetString(start_offset + i)});
            }
        }
    } else if (input_array->type_id() == arrow::Type::MAP) {
        std::shared_ptr<arrow::MapArray> map_array =
            std::dynamic_pointer_cast<arrow::MapArray>(input_array);
        auto map_builder = static_cast<arrow::MapBuilder*>(builder);

        arrow::ArrayBuilder* key_builder = map_builder->key_builder();
        arrow::ArrayBuilder* item_builder = map_builder->item_builder();

        for (int64_t idx = start_offset; idx < end_offset; idx++) {
            if (map_array->IsNull(idx)) {
                (void)map_builder->AppendNull();
                continue;
            }

            (void)map_builder->Append();  // indicate list boundary

            append_to_out_array(map_array->keys(), map_array->value_offset(idx),
                                map_array->value_offset(idx + 1), key_builder);

            append_to_out_array(map_array->items(),
                                map_array->value_offset(idx),
                                map_array->value_offset(idx + 1), item_builder);
        }

    } else if (input_array->type_id() == arrow::Type::DICTIONARY) {
        auto dict_array =
            std::dynamic_pointer_cast<arrow::DictionaryArray>(input_array);
        auto dict_indices =
            std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(
                dict_array->indices());

        // NOTE: Arrow as of v11.0 uses DictionaryBuilder instead of
        // Dictionary32Builder, which requires int64 index inputs instead of
        // int32 (seems like a bug).
        auto dict_builder =
            static_cast<arrow::DictionaryBuilder<arrow::LargeStringType>*>(
                builder);

        int64_t num_elems = end_offset - start_offset;
        for (int64_t i = 0; i < num_elems; i++) {
            if (dict_array->IsNull(start_offset + i)) {
                (void)dict_builder->AppendNull();
            } else {
                // Convert int32 index to int64 as expected by DictionaryBuilder
                int64_t index = static_cast<int64_t>(
                    *(dict_indices->raw_values() + (start_offset + i)));
                (void)dict_builder->AppendIndices(&index, 1);
            }
        }

    } else {
        int64_t num_elems = end_offset - start_offset;
        int64_t vect_size;
        // assume this is array of primitive values
        // TODO: decimal, date, etc.
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(input_array);
        if (primitive_array->type()->id() == arrow::Type::BOOL) {
            // Note the AppendValues API used by BOOL indexes into the
            // null bitmap directly so we don't use valid_elems to track
            // null values.
            vect_size = 0;
        } else {
            vect_size = num_elems;
        }
        bodo::vector<uint8_t> valid_elems(vect_size, 0);
        // TODO: more efficient way of getting null data?
        size_t j = 0;
        if (primitive_array->type()->id() != arrow::Type::BOOL) {
            // Nullable boolean arrays use a bitmap just reuse the bitmap.
            for (int64_t i = start_offset; i < start_offset + num_elems; i++) {
                valid_elems[j++] = !primitive_array->IsNull(i);
            }
        }
        append_to_primitive(primitive_array, start_offset, num_elems, builder,
                            valid_elems);
    }
}

/**
 * @brief Create a Numpy array by selecting elements from input array as
 * provided by index array in_arr_idxs.
 *
 * @tparam IndexT type of index to retrieve, must be int32_t or int64_t
 * @param in_arr Numpy array with input data
 * @param in_arr_idxs array of indices into input array to create the output
 * array
 * @param nRowOut number of rows in output array
 * @param use_nullable_arr use nullable integer/float data type if input data is
 * Numpy integer/float
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 * @return std::shared_ptr<array_info> output data array as specified by input
 */
template <typename IndexT>
std::shared_ptr<array_info> RetrieveArray_SingleColumn_F_numpy(
    std::shared_ptr<array_info> in_arr, const IndexT* in_arr_idxs,
    size_t nRowOut, bool use_nullable_arr = false,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    std::shared_ptr<array_info> out_arr = nullptr;
    bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
    assert(arr_type == bodo_array_type::NUMPY);
    Bodo_CTypes::CTypeEnum dtype = in_arr->dtype;
    uint64_t siztype = numpy_item_size[dtype];
    std::vector<char> vectNaN = RetrieveNaNentry(dtype);
    char* in_data1 = in_arr->data1<bodo_array_type::NUMPY>();
    // use nullable int/float/bool array if dtype is integer/float/bool and
    // use_nullable_arr is specified
    if (use_nullable_arr &&
        (is_integer(dtype) || is_float(dtype) || dtype == Bodo_CTypes::_BOOL)) {
        out_arr = alloc_array_top_level(
            nRowOut, -1, -1, bodo_array_type::NULLABLE_INT_BOOL, dtype, -1, 0,
            0, false, false, false, pool, std::move(mm));
        char* out_data1 = out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
        if (dtype == Bodo_CTypes::_BOOL) {
            // Boolean needs a special implementation because the output
            // array stores 1 bit per boolean.
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                int64_t idx = in_arr_idxs[iRow];
                bool bit = false;
                if (idx >= 0) {
                    bool data_bit = in_data1[idx] != 0;
                    SetBitTo((uint8_t*)out_data1, iRow, data_bit);
                    bit = true;
                }
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow,
                                                                          bit);
            }
        } else {
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                int64_t idx = in_arr_idxs[iRow];
                char* out_ptr = out_data1 + siztype * iRow;
                char* in_ptr;
                bool bit = false;
                if (idx >= 0) {
                    in_ptr = in_data1 + siztype * idx;
                    memcpy(out_ptr, in_ptr, siztype);
                    bit = true;
                }
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow,
                                                                          bit);
            }
        }
    } else {
        out_arr =
            alloc_array_top_level(nRowOut, -1, -1, arr_type, dtype, -1, 0, 0,
                                  false, false, false, pool, std::move(mm));

        if (siztype == sizeof(int32_t)) {
            using T = int32_t;
            T na_val;
            memcpy(&na_val, (T*)vectNaN.data(), sizeof(T));
            T* out_data1 = (T*)out_arr->data1<bodo_array_type::NUMPY>();
            T* in_data1 = (T*)in_arr->data1<bodo_array_type::NUMPY>();

            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                int64_t idx = in_arr_idxs[iRow];
                out_data1[iRow] = (idx >= 0) ? in_data1[idx] : na_val;
            }
            return out_arr;
        }

        // TODO: refactor duplicate code
        if (siztype == sizeof(int64_t)) {
            using T = int64_t;
            T na_val;
            memcpy(&na_val, (T*)vectNaN.data(), sizeof(T));
            T* out_data1 = (T*)out_arr->data1<bodo_array_type::NUMPY>();
            T* in_data1 = (T*)in_arr->data1<bodo_array_type::NUMPY>();

            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                int64_t idx = in_arr_idxs[iRow];
                out_data1[iRow] = (idx >= 0) ? in_data1[idx] : na_val;
            }
            return out_arr;
        }

        char* out_data1 = out_arr->data1<bodo_array_type::NUMPY>();
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = in_arr_idxs[iRow];
            char* out_ptr = out_data1 + siztype * iRow;
            char* in_ptr;
            // To allow NaN values in the column.
            if (idx >= 0) {
                in_ptr = in_data1 + siztype * idx;
            } else {
                in_ptr = vectNaN.data();
            }
            memcpy(out_ptr, in_ptr, siztype);
        }
    }
    return out_arr;
}

/**
 * @brief Create a nullable array by selecting elements from input array as
 * provided by index array in_arr_idxs.
 *
 * @tparam IndexT type of index to retrieve, must be int32_t or int64_t
 * @param in_arr Nullable array with input data
 * @param in_arr_idxs array of indices into input array to create the output
 * array
 * @param nRowOut number of rows in output array
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 * @return std::shared_ptr<array_info> output data array as specified by input
 */
template <typename IndexT>
std::shared_ptr<array_info> RetrieveArray_SingleColumn_F_nullable(
    std::shared_ptr<array_info> in_arr, const IndexT* in_arr_idxs,
    size_t nRowOut,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
    assert(arr_type == bodo_array_type::NULLABLE_INT_BOOL);
    Bodo_CTypes::CTypeEnum dtype = in_arr->dtype;
    std::shared_ptr<array_info> out_arr =
        alloc_array_top_level(nRowOut, -1, -1, arr_type, dtype, -1, 0, 0, false,
                              false, false, pool, std::move(mm));
    uint64_t siztype = numpy_item_size[dtype];

    if (dtype == Bodo_CTypes::_BOOL) {
        // Nullable boolean arrays use 1 bit per boolean
        // so we need a specialized loop.
        uint8_t* in_data1 =
            (uint8_t*)in_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
        uint8_t* out_data1 =
            (uint8_t*)out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = in_arr_idxs[iRow];
            bool null_bit = false;
            if (idx >= 0) {
                bool data_bit = GetBit(in_data1, idx);
                SetBitTo(out_data1, iRow, data_bit);
                null_bit =
                    in_arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                        idx);
            }
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow,
                                                                      null_bit);
        }
        return out_arr;
    }

    if (siztype == sizeof(int32_t)) {
        using T = int32_t;
        T* in_data1 = (T*)in_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
        T* out_data1 = (T*)out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = in_arr_idxs[iRow];
            bool bit = false;
            if (idx >= 0) {
                out_data1[iRow] = in_data1[idx];
                bit = in_arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    idx);
            } else {
                // set index value of dict-encoded string arrays to zero in case
                // some other code accesses it by mistake. see
                // https://bodo.atlassian.net/browse/BE-4146
                out_data1[iRow] = 0;
            }
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow,
                                                                      bit);
        }
        return out_arr;
    }

    // TODO: refactor duplicate code
    if (siztype == sizeof(int64_t)) {
        using T = int64_t;
        T* in_data1 = (T*)in_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
        T* out_data1 = (T*)out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            int64_t idx = in_arr_idxs[iRow];
            bool bit = false;
            if (idx >= 0) {
                out_data1[iRow] = in_data1[idx];
                bit = in_arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    idx);
            }
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow,
                                                                      bit);
        }
        return out_arr;
    }

    char* in_data1 = in_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
    char* out_data1 = out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
    for (size_t iRow = 0; iRow < nRowOut; iRow++) {
        int64_t idx = in_arr_idxs[iRow];
        bool bit = false;
        char* out_ptr = out_data1 + siztype * iRow;
        if (idx >= 0) {
            char* in_ptr = in_data1 + siztype * idx;
            memcpy(out_ptr, in_ptr, siztype);
            bit = in_arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(idx);
        }
        out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow, bit);
    }

    return out_arr;
}

/**
 * @brief Create a TimestampTZ array by selecting elements from input array as
 * provided by index array in_arr_idxs.
 *
 * @tparam IndexT type of index to retrieve, must be int32_t or int64_t
 * @param in_arr TimestampTZ array with input data
 * @param in_arr_idxs array of indices into input array to create the output
 * array
 * @param nRowOut number of rows in output array
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 * @return std::shared_ptr<array_info> output data array as specified by input
 */
template <typename IndexT>
std::shared_ptr<array_info> RetrieveArray_SingleColumn_F_timestamptz(
    std::shared_ptr<array_info> in_arr, const IndexT* in_arr_idxs,
    size_t nRowOut,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
    assert(arr_type == bodo_array_type::TIMESTAMPTZ);
    Bodo_CTypes::CTypeEnum dtype = in_arr->dtype;
    std::shared_ptr<array_info> out_arr =
        alloc_array_top_level(nRowOut, -1, -1, arr_type, dtype, -1, 0, 0, false,
                              false, false, pool, std::move(mm));
    using T1 = typename dtype_to_type<Bodo_CTypes::INT64>::type;
    using T2 = typename dtype_to_type<Bodo_CTypes::INT16>::type;
    T1* in_ts_data = (T1*)in_arr->data1<bodo_array_type::TIMESTAMPTZ>();
    T1* out_ts_data = (T1*)out_arr->data1<bodo_array_type::TIMESTAMPTZ>();
    T2* in_offset_data = (T2*)in_arr->data2<bodo_array_type::TIMESTAMPTZ>();
    T2* out_offset_data = (T2*)out_arr->data2<bodo_array_type::TIMESTAMPTZ>();
    for (size_t iRow = 0; iRow < nRowOut; iRow++) {
        int64_t idx = in_arr_idxs[iRow];
        bool bit = false;
        if (idx >= 0) {
            out_ts_data[iRow] = in_ts_data[idx];
            out_offset_data[iRow] = in_offset_data[idx];
            bit = in_arr->get_null_bit<bodo_array_type::TIMESTAMPTZ>(idx);
        }
        out_arr->set_null_bit<bodo_array_type::TIMESTAMPTZ>(iRow, bit);
    }
    return out_arr;
}

template <typename IndexT>
std::shared_ptr<array_info> RetrieveArray_SingleColumn_F(
    std::shared_ptr<array_info> in_arr, const IndexT* in_arr_idxs,
    size_t nRowOut, bool use_nullable_arr,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    std::shared_ptr<array_info> out_arr = nullptr;
    bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
    Bodo_CTypes::CTypeEnum dtype = in_arr->dtype;
    switch (arr_type) {
        case bodo_array_type::STRING: {
            // In the first case of STRING, we have to deal with offsets first
            // so we need one first loop to determine the needed length. In the
            // second loop, the assignation is made. If the entries are missing
            // then the bitmask is set to false.
            int64_t n_chars = 0;
            bodo::vector<offset_t> ListSizes(nRowOut, pool);
            offset_t* in_offsets =
                (offset_t*)in_arr->data2<bodo_array_type::STRING>();
            char* in_data1 = in_arr->data1<bodo_array_type::STRING>();
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                int64_t idx = in_arr_idxs[iRow];
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
            out_arr = alloc_array_top_level(nRowOut, n_chars, -1, arr_type,
                                            dtype, -1, 0, 0, false, false,
                                            false, pool, std::move(mm));
            offset_t* out_offsets =
                (offset_t*)out_arr->data2<bodo_array_type::STRING>();
            char* out_data1 = out_arr->data1<bodo_array_type::STRING>();
            offset_t pos = 0;
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                int64_t idx = in_arr_idxs[iRow];
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
                    bit = in_arr->get_null_bit<bodo_array_type::STRING>(idx);
                }
                out_arr->set_null_bit<bodo_array_type::STRING>(iRow, bit);
            }
            out_offsets[nRowOut] = pos;
            break;
        }
        case bodo_array_type::DICT: {
            std::shared_ptr<array_info> in_indices = in_arr->child_arrays[1];
            std::shared_ptr<array_info> out_indices =
                RetrieveArray_SingleColumn_F_nullable(
                    in_indices, in_arr_idxs, nRowOut, pool, std::move(mm));
            out_arr =
                create_dict_string_array(in_arr->child_arrays[0], out_indices);
            break;
        }
        case bodo_array_type::NULLABLE_INT_BOOL: {
            out_arr = RetrieveArray_SingleColumn_F_nullable(
                in_arr, in_arr_idxs, nRowOut, pool, std::move(mm));
            break;
        }
        case bodo_array_type::TIMESTAMPTZ: {
            out_arr = RetrieveArray_SingleColumn_F_timestamptz(
                in_arr, in_arr_idxs, nRowOut, pool, std::move(mm));
            break;
        }
        case bodo_array_type::INTERVAL: {
            // Interval array requires copying left and right data
            uint64_t siztype = numpy_item_size[dtype];
            std::vector<char> vectNaN = RetrieveNaNentry(dtype);
            char* left_data = in_arr->data1<bodo_array_type::INTERVAL>();
            char* right_data = in_arr->data2<bodo_array_type::INTERVAL>();
            out_arr = alloc_array_top_level(nRowOut, -1, -1, arr_type, dtype,
                                            -1, 0, 0, false, false, false, pool,
                                            std::move(mm));
            char* out_left_data = out_arr->data1<bodo_array_type::INTERVAL>();
            char* out_right_data = out_arr->data2<bodo_array_type::INTERVAL>();
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                int64_t idx = in_arr_idxs[iRow];
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
            break;
        }
        case bodo_array_type::CATEGORICAL: {
            // In the case of CATEGORICAL array we have only to put a single
            // entry. For the NaN entry the value is -1.
            uint64_t siztype = numpy_item_size[dtype];
            std::vector<char> vectNaN =
                RetrieveNaNentry(dtype);  // returns a -1 for integer values.
            char* in_data1 = in_arr->data1<bodo_array_type::CATEGORICAL>();
            int64_t num_categories = in_arr->num_categories;
            out_arr = alloc_categorical(nRowOut, dtype, num_categories, pool,
                                        std::move(mm));
            char* out_data1 = out_arr->data1<bodo_array_type::CATEGORICAL>();
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                int64_t idx = in_arr_idxs[iRow];
                char* out_ptr = out_data1 + siztype * iRow;
                char* in_ptr;
                // To allow NaN values in the column.
                if (idx >= 0) {
                    in_ptr = in_data1 + siztype * idx;
                } else {
                    in_ptr = vectNaN.data();
                }
                memcpy(out_ptr, in_ptr, siztype);
            }
            break;
        }
        case bodo_array_type::NUMPY: {
            out_arr = RetrieveArray_SingleColumn_F_numpy(
                in_arr, in_arr_idxs, nRowOut, use_nullable_arr, pool,
                std::move(mm));
            break;
        }
        case bodo_array_type::ARRAY_ITEM:
        case bodo_array_type::STRUCT:
        case bodo_array_type::MAP: {
            // NOTE: We don't support using custom pool for these dtypes.
            //   These probably need to be re-written to not go through
            //   Arrow anyway.
            // Arrow builder for output array. builds it dynamically (buffer
            // sizes are not known in advance)
            std::unique_ptr<arrow::ArrayBuilder> builder;
            std::shared_ptr<arrow::Array> in_arrow_array = to_arrow(in_arr);
            (void)arrow::MakeBuilder(pool, in_arrow_array->type(), &builder);
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                int64_t idx = in_arr_idxs[iRow];
                // append value in position 'row' of input array to builder's
                // array (this is a recursive algorithm, can traverse nested
                // arrays)
                // To allow NaN values in the column.
                if (idx >= 0) {
                    append_to_out_array(in_arrow_array, idx, idx + 1,
                                        builder.get());
                } else {
                    (void)builder->AppendNull();
                }
            }

            // get final output array from builder
            std::shared_ptr<arrow::Array> out_arrow_array;
            // TODO: assert builder is not null (at least one row added)
            (void)builder->Finish(&out_arrow_array);

            // Pass input array to reuse its dictionaries since builder
            // doesn't set dictionaries.
            out_arr = arrow_array_to_bodo(out_arrow_array, pool, -1, in_arr);
            break;
        }
        default:
            throw std::runtime_error(
                "RetrieveArray_SingleColumn_F not implemented for this array "
                "type: " +
                GetArrType_as_string(arr_type));
    }
    out_arr->precision = in_arr->precision;
    return out_arr;
};

std::shared_ptr<array_info> RetrieveArray_SingleColumn(
    std::shared_ptr<array_info> in_arr, const std::span<const int32_t> ListIdx,
    bool use_nullable_arr, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    return RetrieveArray_SingleColumn_F<int32_t>(
        std::move(in_arr), ListIdx.data(), ListIdx.size(), use_nullable_arr,
        pool, std::move(mm));
}

std::shared_ptr<array_info> RetrieveArray_SingleColumn(
    std::shared_ptr<array_info> in_arr, const std::span<const int64_t> ListIdx,
    bool use_nullable_arr, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    return RetrieveArray_SingleColumn_F<int64_t>(
        std::move(in_arr), ListIdx.data(), ListIdx.size(), use_nullable_arr,
        pool, std::move(mm));
}

std::shared_ptr<array_info> RetrieveArray_SingleColumn_arr(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<array_info> idx_arr,
    bool use_nullable_arr, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    if (idx_arr->dtype != Bodo_CTypes::UINT64) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "UINT64 is the only index type allowed");
        return nullptr;
    }
    size_t siz = idx_arr->length;
    return RetrieveArray_SingleColumn_F<int64_t>(
        std::move(in_arr), (int64_t*)idx_arr->data1(), siz, use_nullable_arr,
        pool, std::move(mm));
}

std::shared_ptr<array_info> RetrieveArray_TwoColumns(
    std::shared_ptr<array_info> const& arr1,
    std::shared_ptr<array_info> const& arr2,
    const std::span<const int64_t> short_write_idxs,
    const std::span<const int64_t> long_write_idxs,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    if ((arr1 != nullptr) && (arr2 != nullptr) &&
        (arr1->arr_type == bodo_array_type::DICT) &&
        (arr2->arr_type == bodo_array_type::DICT) &&
        !is_matching_dictionary(arr1->child_arrays[0], arr2->child_arrays[0])) {
        throw std::runtime_error(
            "RetrieveArray_TwoColumns: don't know if arrays have unified "
            "dictionary");
    }
    size_t nRowOut = long_write_idxs.size();
    std::shared_ptr<array_info> out_arr = nullptr;
    /* The function for computing the returning values
     * In the output is the column index to use and the row index to use.
     * The row index may be -1 though.
     *
     * @param the row index in the output
     * @return the pair (column,row) to be used.
     */
    auto get_iRow = [&](size_t const& iRowIn)
        -> std::pair<std::shared_ptr<array_info>, int64_t> {
        int64_t short_val = short_write_idxs[iRowIn];
        if (short_val != -1) {
            return {arr1, short_val};
        } else {
            return {arr2, long_write_idxs[iRowIn]};
        }
    };
    // eshift is the in_table index used for the determination
    // of arr_type and dtype of the returned column.
    bodo_array_type::arr_type_enum arr_type = arr1->arr_type;
    Bodo_CTypes::CTypeEnum dtype = arr1->dtype;
    if (arr_type == bodo_array_type::STRING) {
        // In the first case of STRING, we have to deal with offsets first so we
        // need one first loop to determine the needed length. In the second
        // loop, the assignation is made. If the entries are missing then the
        // bitmask is set to false.
        int64_t n_chars = 0;
        bodo::vector<offset_t> ListSizes(nRowOut);
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<std::shared_ptr<array_info>, int64_t> ArrRow =
                get_iRow(iRow);
            offset_t size = 0;
            if (ArrRow.second >= 0) {
                offset_t* in_offsets =
                    (offset_t*)ArrRow.first->data2<bodo_array_type::STRING>();
                offset_t end_offset = in_offsets[ArrRow.second + 1];
                offset_t start_offset = in_offsets[ArrRow.second];
                size = end_offset - start_offset;
            }
            ListSizes[iRow] = size;
            n_chars += size;
        }
        out_arr = alloc_array_top_level(nRowOut, n_chars, -1, arr_type, dtype);
        offset_t pos = 0;
        offset_t* out_offsets =
            (offset_t*)out_arr->data2<bodo_array_type::STRING>();
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<std::shared_ptr<array_info>, int64_t> ArrRow =
                get_iRow(iRow);
            offset_t size = ListSizes[iRow];
            out_offsets[iRow] = pos;
            bool bit = false;
            if (ArrRow.second >= 0) {
                std::shared_ptr<array_info> e_col = ArrRow.first;
                offset_t* in_offsets =
                    (offset_t*)e_col->data2<bodo_array_type::STRING>();
                offset_t start_offset = in_offsets[ArrRow.second];
                char* out_ptr = out_arr->data1<bodo_array_type::STRING>() + pos;
                char* in_ptr =
                    e_col->data1<bodo_array_type::STRING>() + start_offset;
                memcpy(out_ptr, in_ptr, size);
                pos += size;
                bit =
                    e_col->get_null_bit<bodo_array_type::STRING>(ArrRow.second);
            }
            out_arr->set_null_bit<bodo_array_type::STRING>(iRow, bit);
        }
        out_offsets[nRowOut] = pos;
    }
    if (arr_type == bodo_array_type::DICT) {
        // TODO refactor? this is mostly the same logic as NULLABLE_INT_BOOL
        // if (is_parallel && !arr1->child_arrays[0]->is_globally_replicated)
        //    throw std::runtime_error("RetrieveArray_TwoColumns: reference
        //    column doesn't have a global dictionary");
        std::shared_ptr<array_info> out_indices = alloc_array_top_level(
            nRowOut, -1, -1, arr1->child_arrays[1]->arr_type,
            arr1->child_arrays[1]->dtype);
        uint64_t siztype = numpy_item_size[arr1->child_arrays[1]->dtype];
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<std::shared_ptr<array_info>, int64_t> ArrRow =
                get_iRow(iRow);
            bool bit = false;
            char* out_ptr =
                out_indices->data1<bodo_array_type::NULLABLE_INT_BOOL>() +
                siztype * iRow;
            if (ArrRow.second >= 0) {
                std::shared_ptr<array_info> e_col = ArrRow.first;
                char* in_ptr =
                    e_col->child_arrays[1]
                        ->data1<bodo_array_type::NULLABLE_INT_BOOL>() +
                    siztype * ArrRow.second;
                memcpy(out_ptr, in_ptr, siztype);
                bit = e_col->child_arrays[1]
                          ->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                              ArrRow.second);
            } else {
                // set index value to zero in case some other code accesses it
                // by mistake. see https://bodo.atlassian.net/browse/BE-4146
                memset(out_ptr, 0, siztype);
            }
            out_indices->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow,
                                                                          bit);
        }
        out_arr = create_dict_string_array(arr1->child_arrays[0], out_indices);
    }
    if (arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // In the case of NULLABLE array, we do a single loop for
        // assigning the arrays.
        // We do not need to reassign the pointers, only their size
        // suffices for the copy.
        // In the case of missing array a value of false is assigned
        // to the bitmask.
        out_arr = alloc_array_top_level(nRowOut, -1, -1, arr_type, dtype);
        if (dtype == Bodo_CTypes::_BOOL) {
            // Nullable boolean arrays store 1 bit per boolean so we
            // need to use a different loop.
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                std::pair<std::shared_ptr<array_info>, int64_t> ArrRow =
                    get_iRow(iRow);
                bool null_bit = false;
                if (ArrRow.second >= 0) {
                    std::shared_ptr<array_info> e_col = ArrRow.first;
                    bool data_bit = GetBit(
                        (uint8_t*)
                            e_col->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                        ArrRow.second);
                    SetBitTo((uint8_t*)out_arr
                                 ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                             iRow, data_bit);
                    null_bit =
                        e_col->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                            ArrRow.second);
                }
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    iRow, null_bit);
            }
        } else {
            uint64_t siztype = numpy_item_size[dtype];
            for (size_t iRow = 0; iRow < nRowOut; iRow++) {
                std::pair<std::shared_ptr<array_info>, int64_t> ArrRow =
                    get_iRow(iRow);
                bool bit = false;
                if (ArrRow.second >= 0) {
                    std::shared_ptr<array_info> e_col = ArrRow.first;
                    char* out_ptr =
                        out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>() +
                        siztype * iRow;
                    char* in_ptr =
                        e_col->data1<bodo_array_type::NULLABLE_INT_BOOL>() +
                        siztype * ArrRow.second;
                    memcpy(out_ptr, in_ptr, siztype);
                    bit =
                        e_col->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                            ArrRow.second);
                }
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow,
                                                                          bit);
            }
        }
    }
    if (arr_type == bodo_array_type::CATEGORICAL) {
        // In the case of CATEGORICAL array we have only to put a single
        // entry. For the NaN entry the value is -1.
        uint64_t siztype = numpy_item_size[dtype];
        std::vector<char> vectNaN =
            RetrieveNaNentry(dtype);  // returns a -1 for integer values.
        int64_t num_categories = arr1->num_categories;
        out_arr = alloc_categorical(nRowOut, dtype, num_categories);
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<std::shared_ptr<array_info>, int64_t> ArrRow =
                get_iRow(iRow);
            //
            char* out_ptr =
                out_arr->data1<bodo_array_type::CATEGORICAL>() + siztype * iRow;
            char* in_ptr;
            if (ArrRow.second >= 0) {
                in_ptr = ArrRow.first->data1<bodo_array_type::CATEGORICAL>() +
                         siztype * ArrRow.second;
            } else {
                in_ptr = vectNaN.data();
            }
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
        std::vector<char> vectNaN = RetrieveNaNentry(dtype);
        out_arr = alloc_array_top_level(nRowOut, -1, -1, arr_type, dtype);
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<std::shared_ptr<array_info>, int64_t> ArrRow =
                get_iRow(iRow);
            //
            char* out_ptr =
                out_arr->data1<bodo_array_type::NUMPY>() + siztype * iRow;
            char* in_ptr;
            if (ArrRow.second >= 0) {
                in_ptr = ArrRow.first->data1<bodo_array_type::NUMPY>() +
                         siztype * ArrRow.second;
            } else {
                in_ptr = vectNaN.data();
            }
            memcpy(out_ptr, in_ptr, siztype);
        }
    }
    if (arr_type == bodo_array_type::STRUCT ||
        arr_type == bodo_array_type::ARRAY_ITEM ||
        arr_type == bodo_array_type::MAP) {
        // Arrow builder for output array. builds it dynamically (buffer
        // sizes are not known in advance)
        std::unique_ptr<arrow::ArrayBuilder> builder;
        std::shared_ptr<arrow::Array> in_arr_typ = to_arrow(arr1);
        (void)arrow::MakeBuilder(pool, in_arr_typ->type(), &builder);
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<std::shared_ptr<array_info>, int64_t> ArrRow =
                get_iRow(iRow);
            if (ArrRow.second >= 0) {
                // non-null value for output
                std::shared_ptr<arrow::Array> in_arr = to_arrow(ArrRow.first);
                size_t row = ArrRow.second;
                // append value in position 'row' of input array to
                // builder's array (this is a recursive algorithm, can
                // traverse nested arrays)
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

        out_arr = arrow_array_to_bodo(out_arrow_array, pool);
    }
    if (arr_type == bodo_array_type::TIMESTAMPTZ) {
        out_arr = alloc_array_top_level(nRowOut, -1, -1, arr_type, dtype);
        uint64_t size_ts_type = numpy_item_size[Bodo_CTypes::INT64];
        uint64_t size_offset_type = numpy_item_size[Bodo_CTypes::INT16];
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<std::shared_ptr<array_info>, int64_t> ArrRow =
                get_iRow(iRow);
            bool bit = false;
            if (ArrRow.second >= 0) {
                std::shared_ptr<array_info> e_col = ArrRow.first;
                char* out_ts_ptr =
                    out_arr->data1<bodo_array_type::TIMESTAMPTZ>() +
                    size_ts_type * iRow;
                char* in_ts_ptr = e_col->data1<bodo_array_type::TIMESTAMPTZ>() +
                                  size_ts_type * ArrRow.second;
                char* out_offset_ptr =
                    out_arr->data2<bodo_array_type::TIMESTAMPTZ>() +
                    size_offset_type * iRow;
                char* in_offset_ptr =
                    e_col->data1<bodo_array_type::TIMESTAMPTZ>() +
                    size_offset_type * ArrRow.second;

                memcpy(out_ts_ptr, in_ts_ptr, size_ts_type);
                memcpy(out_offset_ptr, in_offset_ptr, size_offset_type);
                bit = e_col->get_null_bit<bodo_array_type::TIMESTAMPTZ>(
                    ArrRow.second);
            }
            out_arr->set_null_bit<bodo_array_type::TIMESTAMPTZ>(iRow, bit);
        }
    }
    return out_arr;
}

template <typename IndexT>
std::shared_ptr<table_info> DoRetrieveTable(
    std::shared_ptr<table_info> const in_table,
    const std::span<const IndexT> ListIdx, int const& n_cols_arg,
    const bool use_nullable_arr, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    std::vector<std::string> col_names;
    size_t n_cols;
    if (n_cols_arg == -1) {
        n_cols = (size_t)in_table->ncols();
    } else {
        n_cols = n_cols_arg;
    }
    if (in_table->column_names.size() > 0) {
        col_names.reserve(n_cols);
    }
    for (size_t i_col = 0; i_col < n_cols; i_col++) {
        std::shared_ptr<array_info> in_arr = in_table->columns[i_col];
        out_arrs.emplace_back(RetrieveArray_SingleColumn(
            std::move(in_arr), ListIdx, use_nullable_arr, pool, mm));
        // Release reference (and potentially memory) for the column from this
        // table if this is the last table reference.
        reset_col_if_last_table_ref(in_table, i_col);
        if (in_table->column_names.size() > 0) {
            col_names.push_back(in_table->column_names[i_col]);
        }
    }
    return std::make_shared<table_info>(out_arrs, col_names,
                                        in_table->metadata);
}

std::shared_ptr<table_info> RetrieveTable(
    std::shared_ptr<table_info> const in_table,
    const std::span<const int32_t> ListIdx, int const& n_cols_arg,
    const bool use_nullable_arr, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    return DoRetrieveTable<int32_t>(in_table, ListIdx, n_cols_arg,
                                    use_nullable_arr, pool, mm);
}

std::shared_ptr<table_info> RetrieveTable(
    std::shared_ptr<table_info> const in_table,
    const std::span<const int64_t> ListIdx, int const& n_cols_arg,
    const bool use_nullable_arr, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    return DoRetrieveTable<int64_t>(in_table, ListIdx, n_cols_arg,
                                    use_nullable_arr, pool, mm);
}

template <typename IndexT>
std::shared_ptr<table_info> DoRetrieveTable(
    std::shared_ptr<table_info> const in_table,
    const std::span<const IndexT> rowInds, std::vector<uint64_t> const& colInds,
    const bool use_nullable_arr, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    std::vector<std::string> col_names;
    if (in_table->column_names.size() > 0) {
        col_names.reserve(colInds.size());
    }
    for (size_t i_col : colInds) {
        std::shared_ptr<array_info> in_arr = in_table->columns[i_col];
        out_arrs.emplace_back(RetrieveArray_SingleColumn(
            std::move(in_arr), rowInds, use_nullable_arr, pool, mm));
        // Release reference (and potentially memory) for the column from this
        // table if this is the last table reference.
        reset_col_if_last_table_ref(in_table, i_col);
        if (in_table->column_names.size() > 0) {
            col_names.push_back(in_table->column_names[i_col]);
        }
    }
    return std::make_shared<table_info>(out_arrs, col_names,
                                        in_table->metadata);
}

std::shared_ptr<table_info> RetrieveTable(
    std::shared_ptr<table_info> const in_table,
    const std::span<const int32_t> rowInds,
    std::vector<uint64_t> const& colInds, const bool use_nullable_arr,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    return DoRetrieveTable<int32_t>(in_table, rowInds, colInds,
                                    use_nullable_arr, pool, mm);
}

std::shared_ptr<table_info> RetrieveTable(
    std::shared_ptr<table_info> const in_table,
    const std::span<const int64_t> rowInds,
    std::vector<uint64_t> const& colInds, const bool use_nullable_arr,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    return DoRetrieveTable<int64_t>(in_table, rowInds, colInds,
                                    use_nullable_arr, pool, mm);
}

std::shared_ptr<table_info> RetrieveTable(
    std::shared_ptr<table_info> const in_table,
    std::shared_ptr<array_info> row_bitmask) {
    // convert bitmask to selection vector
    std::vector<int64_t> rows_to_keep(
        std::max(in_table->nrows(), static_cast<uint64_t>(1)), -1);
    size_t next_idx = 0;
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        rows_to_keep[next_idx] = i_row;
        size_t delta =
            arrow::bit_util::GetBit(
                row_bitmask
                    ->data1<bodo_array_type::NULLABLE_INT_BOOL, uint8_t>(),
                i_row)
                ? 1
                : 0;
        next_idx += delta;
    }
    rows_to_keep.resize(next_idx);
    if (rows_to_keep.size() == in_table->nrows()) {
        // Nothing was filtered out, so skip the copy.
        return in_table;
    }
    return RetrieveTable(std::move(in_table), std::move(rows_to_keep));
}

std::shared_ptr<table_info> ProjectTable(
    std::shared_ptr<table_info> const in_table,
    const std::span<const int64_t> column_indices) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    std::vector<std::string> col_names;
    if (in_table->column_names.size() > 0) {
        col_names.reserve(column_indices.size());
    }
    for (size_t i_col : column_indices) {
        out_arrs.emplace_back(in_table->columns[i_col]);
        if (in_table->column_names.size() > 0) {
            col_names.push_back(in_table->column_names[i_col]);
        }
    }
    return std::make_shared<table_info>(out_arrs, in_table->nrows(), col_names,
                                        in_table->metadata);
}

std::shared_ptr<table_info> ProjectTable(
    std::shared_ptr<table_info> const in_table, size_t first_n_cols) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    size_t n_cols = std::min(first_n_cols, in_table->columns.size());
    std::vector<std::string> col_names;
    if (in_table->column_names.size() > 0) {
        col_names.reserve(n_cols);
    }

    for (size_t i = 0; i < n_cols; i++) {
        out_arrs.emplace_back(in_table->columns[i]);
        if (in_table->column_names.size() > 0) {
            col_names.push_back(in_table->column_names[i]);
        }
    }
    return std::make_shared<table_info>(out_arrs, in_table->nrows(), col_names,
                                        in_table->metadata);
}

// @brief Compare the bitmap of two arrow arrays at a given position
// @param na_position_bis True if NA should be considered greater than any value
// @param arrow1 First array
// @param pos1 Position in first array
// @param arrow2 Second array
// @param pos2 Position in second array
// @param is_na_equal True if NA should be considered equal to NA
template <typename T>
std::pair<int, bool> process_arrow_bitmap(bool const& na_position_bis,
                                          T const& arrow1, int64_t pos1,
                                          T const& arrow2, int64_t pos2,
                                          bool const& is_na_equal) {
    bool bit1 = !arrow1->IsNull(pos1);
    bool bit2 = !arrow2->IsNull(pos2);
    // if either bit is null and !is_na_equal, they are not equal
    if ((!bit1 || !bit2) && !is_na_equal) {
        return {-1, false};
    }
    if (bit1 && !bit2) {
        int val = -1;
        if (na_position_bis) {
            val = 1;
        }
        return {val, true};
    }
    if (!bit1 && bit2) {
        int val = 1;
        if (na_position_bis) {
            val = -1;
        }
        return {val, true};
    }
    return {0, bit1};
}

std::shared_ptr<arrow::ArrayData> get_sort_indices_of_slice_arrow(
    std::shared_ptr<arrow::Array> const& arr, int64_t pos_s, int64_t pos_e) {
    // Get a slice of the array containing only the selected elements
    std::shared_ptr<arrow::Array> slice = arr->Slice(pos_s, pos_e - pos_s);
    arrow::Result<arrow::Datum> sort_indices_res =
        arrow::compute::CallFunction("array_sort_indices", {slice});
    if (!sort_indices_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            fmt::format("get_sort_indices: Error sorting array: {}",
                        sort_indices_res.status().message()));
    }
    return sort_indices_res.ValueOrDie().array();
}

int CompareNAValues(bool na_position_bis, bool bit1, bool bit2) {
    if (bit1 && !bit2) {
        if (na_position_bis) {
            return 1;
        } else {
            return -1;
        }
    }
    if (!bit1 && bit2) {
        if (na_position_bis) {
            return -1;
        } else {
            return 1;
        }
    }
    return 0;
}

template <bodo_array_type::arr_type_enum T>
int KeyComparisonAsPython_Column_impl(bool const& na_position_bis,
                                      const std::shared_ptr<array_info>& arr1,
                                      size_t const& iRow1,
                                      const std::shared_ptr<array_info>& arr2,
                                      size_t const& iRow2);

template <>
int KeyComparisonAsPython_Column_impl<bodo_array_type::NUMPY>(
    bool const& na_position_bis, const std::shared_ptr<array_info>& arr1,
    size_t const& iRow1, const std::shared_ptr<array_info>& arr2,
    size_t const& iRow2) {
    // In the case of NUMPY, we compare the values for concluding.
    uint64_t siztype = numpy_item_size[arr1->dtype];
    char* ptr1 = arr1->data1<bodo_array_type::NUMPY>() + (siztype * iRow1);
    char* ptr2 = arr2->data1() + (siztype * iRow2);
    return NumericComparison(arr1->dtype, ptr1, ptr2, na_position_bis);
}

template <>
int KeyComparisonAsPython_Column_impl<bodo_array_type::CATEGORICAL>(
    bool const& na_position_bis, const std::shared_ptr<array_info>& arr1,
    size_t const& iRow1, const std::shared_ptr<array_info>& arr2,
    size_t const& iRow2) {
    // In the case of CATEGORICAL, we need to check for null
    uint64_t siztype = numpy_item_size[arr1->dtype];
    char* ptr1 =
        arr1->data1<bodo_array_type::CATEGORICAL>() + (siztype * iRow1);
    char* ptr2 = arr2->data1() + (siztype * iRow2);
    bool is_not_na1 = !isnan_categorical_ptr(arr1->dtype, ptr1);
    bool is_not_na2 = !isnan_categorical_ptr(arr2->dtype, ptr2);
    int reply = CompareNAValues(na_position_bis, is_not_na1, is_not_na2);
    if (reply != 0) {
        return reply;
    }

    if (is_not_na1) {
        return NumericComparison(arr1->dtype, ptr1, ptr2, na_position_bis);
    }
    return 0;
}

template <>
int KeyComparisonAsPython_Column_impl<bodo_array_type::NULLABLE_INT_BOOL>(
    bool const& na_position_bis, const std::shared_ptr<array_info>& arr1,
    size_t const& iRow1, const std::shared_ptr<array_info>& arr2,
    size_t const& iRow2) {
    // NULLABLE case. We need to consider the bitmask and the values.
    uint8_t* null_bitmask1 =
        (uint8_t*)arr1->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>();
    uint8_t* null_bitmask2 = (uint8_t*)arr2->null_bitmask();
    bool bit1 = GetBit(null_bitmask1, iRow1);
    bool bit2 = GetBit(null_bitmask2, iRow2);
    // If one bitmask is T and the other the reverse then they are
    // clearly not equal.
    int reply = CompareNAValues(na_position_bis, bit1, bit2);
    if (reply != 0) {
        return reply;
    }
    // If both bitmasks are false, then it does not matter what value
    // they are storing. Comparison is the same as for NUMPY.
    if (bit1) {
        if (arr1->dtype == Bodo_CTypes::_BOOL) {
            uint8_t* data1 =
                (uint8_t*)arr1->data1<bodo_array_type::NULLABLE_INT_BOOL>();
            uint8_t* data2 = (uint8_t*)arr2->data1();
            bool bit1 = arrow::bit_util::GetBit(data1, iRow1);
            bool bit2 = arrow::bit_util::GetBit(data2, iRow2);
            if (bit1 == bit2) {
                return 0;
            } else if (bit1) {
                // bit1 is true and bit2 is false
                // So (val1 < val2) == False
                return -1;
            } else {
                // bit1 is false and bit2 is true
                // So (val1 < val2) == True
                return 1;
            }
        } else {
            uint64_t siztype = numpy_item_size[arr1->dtype];
            char* ptr1 = arr1->data1<bodo_array_type::NULLABLE_INT_BOOL>() +
                         (siztype * iRow1);
            char* ptr2 = arr2->data1() + (siztype * iRow2);
            int test =
                NumericComparison(arr1->dtype, ptr1, ptr2, na_position_bis);
            return test;
        }
    }
    return 0;
}

template <>
int KeyComparisonAsPython_Column_impl<bodo_array_type::TIMESTAMPTZ>(
    bool const& na_position_bis, const std::shared_ptr<array_info>& arr1,
    size_t const& iRow1, const std::shared_ptr<array_info>& arr2,
    size_t const& iRow2) {
    // NULLABLE case. We need to consider the bitmask and the values.
    uint8_t* null_bitmask1 =
        (uint8_t*)arr1->null_bitmask<bodo_array_type::TIMESTAMPTZ>();
    uint8_t* null_bitmask2 =
        (uint8_t*)arr2->null_bitmask<bodo_array_type::TIMESTAMPTZ>();
    bool bit1 = GetBit(null_bitmask1, iRow1);
    bool bit2 = GetBit(null_bitmask2, iRow2);
    // If one bitmask is T and the other the reverse then they are
    // clearly not equal.
    int reply = CompareNAValues(na_position_bis, bit1, bit2);
    if (reply != 0) {
        return reply;
    }
    // If bit1 is true it means that both rows are non-null, so
    // we need to compare their UTC timestamps.
    if (bit1) {
        uint64_t siztype = numpy_item_size[arr1->dtype];
        char* ptr1 =
            arr1->data1<bodo_array_type::TIMESTAMPTZ>() + (siztype * iRow1);
        char* ptr2 =
            arr2->data1<bodo_array_type::TIMESTAMPTZ>() + (siztype * iRow2);
        int test = NumericComparison(arr1->dtype, ptr1, ptr2, na_position_bis);
        return test;
    }

    return 0;
}

template <>
int KeyComparisonAsPython_Column_impl<bodo_array_type::STRING>(
    bool const& na_position_bis, const std::shared_ptr<array_info>& arr1,
    size_t const& iRow1, const std::shared_ptr<array_info>& arr2,
    size_t const& iRow2) {
    // For STRING case we need to deal bitmask and the values.
    assert(arr2->arr_type == bodo_array_type::STRING);
    bool bit1 = arr1->get_null_bit<bodo_array_type::STRING>(iRow1);
    bool bit2 = arr2->get_null_bit<bodo_array_type::STRING>(iRow2);
    // If bitmasks are different then we can conclude the comparison
    int reply = CompareNAValues(na_position_bis, bit1, bit2);
    if (reply != 0) {
        return reply;
    }
    // If bitmasks are both false, then no need to compare the string
    // values.
    if (bit1) {
        // Here we consider the shifts in data2 for the comparison.
        offset_t* data2_1 = (offset_t*)arr1->data2<bodo_array_type::STRING>();
        offset_t* data2_2 = (offset_t*)arr2->data2<bodo_array_type::STRING>();
        offset_t len1 = data2_1[iRow1 + 1] - data2_1[iRow1];
        offset_t len2 = data2_2[iRow2 + 1] - data2_2[iRow2];
        // Compute minimal length
        offset_t minlen = len1;
        if (len2 < len1) {
            minlen = len2;
        }
        // From the common characters, we may be able to conclude.
        offset_t pos1_prev = data2_1[iRow1];
        offset_t pos2_prev = data2_2[iRow2];
        char* data1_1 =
            (char*)arr1->data1<bodo_array_type::STRING>() + pos1_prev;
        char* data1_2 = (char*)arr2->data1() + pos2_prev;
        int test = std::memcmp(data1_2, data1_1, minlen);
        if (test) {
            return test;
        }
        // If not, we may be able to conclude via the string length.
        if (len1 > len2) {
            return -1;
        }
        if (len1 < len2) {
            return 1;
        }
    }
    return 0;
}

template <>
int KeyComparisonAsPython_Column_impl<bodo_array_type::DICT>(
    bool const& na_position_bis, const std::shared_ptr<array_info>& arr1,
    size_t const& iRow1, const std::shared_ptr<array_info>& arr2,
    size_t const& iRow2) {
    assert(arr2->arr_type == bodo_array_type::DICT);
    if (!is_matching_dictionary(arr1->child_arrays[0], arr2->child_arrays[0])) {
        throw std::runtime_error(
            "KeyComparisonAsPython_Column: don't know if arrays "
            "have unified dictionary");
    }
    // Since arr1->child_arrays[0] == arr2->child_arrays[0] (if arr2 is
    // DICT)
    std::shared_ptr<array_info> arr_dict = arr1->child_arrays[0];
    std::shared_ptr<array_info> arr1_indices = arr1->child_arrays[1];
    std::shared_ptr<array_info> arr2_indices = arr2->child_arrays[1];

    if (arr_dict->is_locally_sorted) {
        // In case of sorted dictionaries, we can simply compare the
        // indices
        return KeyComparisonAsPython_Column(na_position_bis, arr1_indices,
                                            iRow1, arr2_indices, iRow2);
    }
    // If the dictionary is not sorted, then do the regular STRING like
    // check

    // NULLABLE case. We need to consider the bitmask and the values (of
    // the indices)
    bool bit1 =
        arr1_indices->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow1);
    bool bit2 =
        arr2_indices->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow2);
    // If one bitmask is T and the other the reverse then they are
    // clearly not equal.
    int reply = CompareNAValues(na_position_bis, bit1, bit2);
    if (reply != 0) {
        return reply;
    }

    // Currently we assume there are no null values in the dictionary
    // itself, so no special handling is needed, but this might change
    // in the future.

    // If both bitmasks are false, then it does not matter what value
    // they are storing.
    if (bit1) {
        // Get the index for the dict array
        int32_t new_iRow1 =
            arr1_indices
                ->at<dict_indices_t, bodo_array_type::NULLABLE_INT_BOOL>(iRow1);
        int32_t new_iRow2 =
            arr2_indices
                ->at<dict_indices_t, bodo_array_type::NULLABLE_INT_BOOL>(iRow2);

        // Now we compare the dict entries (same as the STRING logic)
        return KeyComparisonAsPython_Column(na_position_bis, arr_dict,
                                            new_iRow1, arr_dict, new_iRow2);
    }
    return 0;
}

template <>
int KeyComparisonAsPython_Column_impl<bodo_array_type::STRUCT>(
    bool const& na_position_bis, const std::shared_ptr<array_info>& arr1,
    size_t const& iRow1, const std::shared_ptr<array_info>& arr2,
    size_t const& iRow2) {
    bool bit1 = arr1->get_null_bit<bodo_array_type::STRUCT>(iRow1);
    bool bit2 = arr2->get_null_bit<bodo_array_type::STRUCT>(iRow2);
    int na_comp = CompareNAValues(na_position_bis, bit1, bit2);
    if (na_comp != 0) {
        return na_comp;
    }
    size_t num_fields1 = arr1->child_arrays.size();
    size_t num_fields2 = arr2->child_arrays.size();
    if (num_fields1 > num_fields2) {
        return -1;
    } else if (num_fields1 < num_fields2) {
        return 1;
    }

    for (size_t i = 0; i < num_fields1; i++) {
        int res =
            KeyComparisonAsPython_Column(na_position_bis, arr1->child_arrays[i],
                                         iRow1, arr2->child_arrays[i], iRow2);
        if (res != 0) {
            return res;
        }
    }
    // The structs are equal
    return 0;
}

template <>
int KeyComparisonAsPython_Column_impl<bodo_array_type::ARRAY_ITEM>(
    bool const& na_position_bis, const std::shared_ptr<array_info>& arr1,
    size_t const& iRow1, const std::shared_ptr<array_info>& arr2,
    size_t const& iRow2) {
    bool bit1 = arr1->get_null_bit<bodo_array_type::ARRAY_ITEM>(iRow1);
    bool bit2 = arr2->get_null_bit<bodo_array_type::ARRAY_ITEM>(iRow2);
    int na_comp = CompareNAValues(na_position_bis, bit1, bit2);
    if (na_comp != 0) {
        return na_comp;
    }
    auto data1 = arr1->child_arrays[0];
    auto offsets1 = (offset_t*)arr1->data1<bodo_array_type::ARRAY_ITEM>();

    auto data2 = arr2->child_arrays[0];
    auto offsets2 = (offset_t*)arr2->data1<bodo_array_type::ARRAY_ITEM>();

    offset_t start1 = offsets1[iRow1];
    offset_t start2 = offsets2[iRow2];
    offset_t length1 = offsets1[iRow1 + 1] - start1;
    offset_t length2 = offsets2[iRow2 + 1] - start2;

    for (size_t i = 0; i < std::min(length1, length2); i++) {
        int res = KeyComparisonAsPython_Column(na_position_bis, data1,
                                               start1 + i, data2, start2 + i);
        if (res != 0) {
            return res;
        }
    }

    if (length1 > length2) {
        return -1;
    } else if (length1 < length2) {
        return 1;
    }
    // The lists are equal
    return 0;
}

template <>
int KeyComparisonAsPython_Column_impl<bodo_array_type::MAP>(
    bool const& na_position_bis, const std::shared_ptr<array_info>& arr1,
    size_t const& iRow1, const std::shared_ptr<array_info>& arr2,
    size_t const& iRow2) {
    bool bit1 = arr1->get_null_bit<bodo_array_type::MAP>(iRow1);
    bool bit2 = arr2->get_null_bit<bodo_array_type::MAP>(iRow2);
    int na_comp = CompareNAValues(na_position_bis, bit1, bit2);
    if (na_comp != 0) {
        return na_comp;
    }
    std::shared_ptr<array_info> list1 = arr1->child_arrays[0];
    auto* offsets1 = (offset_t*)list1->data1<bodo_array_type::ARRAY_ITEM>();
    std::shared_ptr<array_info> list2 = arr2->child_arrays[0];
    auto* offsets2 = (offset_t*)list2->data1<bodo_array_type::ARRAY_ITEM>();

    offset_t start1 = offsets1[iRow1];
    offset_t start2 = offsets2[iRow2];
    offset_t length1 = offsets1[iRow1 + 1] - start1;
    offset_t length2 = offsets2[iRow2 + 1] - start2;

    // We need to compare ignoring the order the keys themselves, so we sort
    // the range of keys corresponding the rows being compared first
    std::vector<int64_t> vect_ascending{0};
    std::vector<int64_t> na_position{0};

    std::vector<std::shared_ptr<array_info>> key_col1{
        list1->child_arrays[0]->child_arrays[0]};
    auto key_table1 = std::make_shared<table_info>(std::move(key_col1));
    auto sorted_idxs1 = sort_values_table_local_get_indices<int64_t>(
        std::move(key_table1), 1, vect_ascending.data(), na_position.data(),
        false, start1, length1);

    std::vector<std::shared_ptr<array_info>> key_col2{
        list2->child_arrays[0]->child_arrays[0]};
    auto key_table2 = std::make_shared<table_info>(std::move(key_col2));
    auto sorted_idxs2 = sort_values_table_local_get_indices<int64_t>(
        std::move(key_table2), 1, vect_ascending.data(), na_position.data(),
        false, start2, length2);

    for (size_t i = 0; i < std::min(length1, length2); i++) {
        int kv_comp = KeyComparisonAsPython_Column(
            na_position_bis, list1->child_arrays[0], sorted_idxs1[i],
            list2->child_arrays[0], sorted_idxs2[i]);
        if (kv_comp != 0) {
            return kv_comp;
        }
    }

    if (length1 > length2) {
        return -1;
    } else if (length1 < length2) {
        return 1;
    }
    return 0;
}

int KeyComparisonAsPython_Column(bool const& na_position_bis,
                                 const std::shared_ptr<array_info>& arr1,
                                 size_t const& iRow1,
                                 const std::shared_ptr<array_info>& arr2,
                                 size_t const& iRow2) {
    assert(arr2->arr_type == arr1->arr_type);
    switch (arr1->arr_type) {
#define IMPL_COMPARISON(type)                                                 \
    case type:                                                                \
        return KeyComparisonAsPython_Column_impl<type>(na_position_bis, arr1, \
                                                       iRow1, arr2, iRow2)

        IMPL_COMPARISON(bodo_array_type::STRUCT);
        IMPL_COMPARISON(bodo_array_type::ARRAY_ITEM);
        IMPL_COMPARISON(bodo_array_type::MAP);
        IMPL_COMPARISON(bodo_array_type::NUMPY);
        IMPL_COMPARISON(bodo_array_type::CATEGORICAL);
        IMPL_COMPARISON(bodo_array_type::NULLABLE_INT_BOOL);
        IMPL_COMPARISON(bodo_array_type::TIMESTAMPTZ);
        IMPL_COMPARISON(bodo_array_type::STRING);
        IMPL_COMPARISON(bodo_array_type::DICT);

        default:
            throw std::runtime_error("Unsupported type for comparison: " +
                                     GetArrType_as_string(arr1->arr_type));
#undef DISPATCH_COMPARISION
    }
}

bool KeyComparisonAsPython(
    size_t const& n_key, const int64_t* vect_ascending,
    std::vector<std::shared_ptr<array_info>> const& columns1,
    size_t const& shift_key1, size_t const& iRow1,
    std::vector<std::shared_ptr<array_info>> const& columns2,
    size_t const& shift_key2, size_t const& iRow2, const int64_t* na_position) {
    // iteration over the list of key for the comparison.
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        bool ascending = vect_ascending[iKey];
        bool na_last = na_position[iKey];
        bool na_position_bis = (!na_last) ^ ascending;
        int test = KeyComparisonAsPython_Column(
            na_position_bis, columns1[shift_key1 + iKey], iRow1,
            columns2[shift_key2 + iKey], iRow2);
        if (test) {
            if (ascending) {
                return test > 0;
            }
            return test < 0;
        }
    }
    // If all keys are equal then we return false
    return false;
}

bool KeyComparisonAsPython(
    size_t const& n_key, const int64_t* vect_ascending,
    std::vector<std::shared_ptr<array_info>> const& columns1,
    size_t const& iRow1,
    std::vector<std::shared_ptr<array_info>> const& columns2,
    size_t const& iRow2, const int64_t* na_position) {
    return KeyComparisonAsPython(n_key, vect_ascending, columns1, 0, iRow1,
                                 columns2, 0, iRow2, na_position);
}

/**
 * @brief Helper function for create_temp_null_bitmask_for_array to fill the
 * bitmask for a NUMPY array.
 *
 * @param[out] bitmask Bitmask array to fill. It is pre-allocated to the
 * correct size and just needs to be filled up.
 * @param arr The array to create bitmask for.
 *
 * @tparam DType The dtype for the array. Only certain dtypes that can have
 * have sentinel nulls are supported.
 */
template <Bodo_CTypes::CTypeEnum DType>
    requires(NullSentinelDtype<DType>)
void fill_null_bitmask_numpy(uint8_t* bitmask,
                             const std::shared_ptr<array_info> arr) {
    using T = typename dtype_to_type<DType>::type;
    assert(arr->arr_type == bodo_array_type::NUMPY);
    T* data = (T*)(arr->data1<bodo_array_type::NUMPY>());
    for (size_t i = 0; i < arr->length; i++) {
        SetBitTo(bitmask, i, !isnan_alltype<T, DType>(data[i]));
    }
}

/**
 * @brief Helper function for bitwise_and_null_bitmasks to create a temp
 * null bitmask for the provided array. Only NUMPY, CATEGORICAL and ARROW
 * array types are supported. For other types, there's either a null-bitmask
 * already, or the array types are not supported as join keys (which is the
 * use case for this).
 *
 * @param arr Array to build bitmask for.
 * @return uint8_t* Bitmask constructed for the array. Caller is responsible
 * for cleaning up the returned array using delete[].
 */
uint8_t* create_temp_null_bitmask_for_array(
    const std::shared_ptr<array_info> arr) {
    uint64_t length = arr->length;
    int64_t n_bytes = ((length + 7) >> 3);
    uint8_t* bitmask = new uint8_t[n_bytes];

    switch (arr->arr_type) {
        case bodo_array_type::NUMPY: {
            // Only some dtypes have the concept of a sentinel null,
            // for the rest, there's no concept of a null, so we set all
            // bits to not null.
            if (arr->dtype == Bodo_CTypes::FLOAT32) {
                fill_null_bitmask_numpy<Bodo_CTypes::FLOAT32>(bitmask, arr);
            } else if (arr->dtype == Bodo_CTypes::FLOAT64) {
                fill_null_bitmask_numpy<Bodo_CTypes::FLOAT64>(bitmask, arr);
            } else if (arr->dtype == Bodo_CTypes::DATETIME) {
                fill_null_bitmask_numpy<Bodo_CTypes::DATETIME>(bitmask, arr);
            } else if (arr->dtype == Bodo_CTypes::TIMEDELTA) {
                fill_null_bitmask_numpy<Bodo_CTypes::TIMEDELTA>(bitmask, arr);
            } else {
                // Set all bits to not null.
                memset(bitmask, 0xff, n_bytes);
            }
            break;
        }
        case bodo_array_type::CATEGORICAL: {
            uint64_t siztype = numpy_item_size[arr->dtype];
            char* data_ptr = arr->data1<bodo_array_type::CATEGORICAL>();
            for (size_t i = 0; i < arr->length; i++) {
                SetBitTo(bitmask, i,
                         !isnan_categorical_ptr(arr->dtype,
                                                data_ptr + (siztype * i)));
            }
            break;
        }
        case bodo_array_type::ARRAY_ITEM:
        case bodo_array_type::STRUCT:
        case bodo_array_type::MAP: {
            // For robustness, we do a for loop and use the IsNull function
            // to determine if an element is null. IsNull correctly handles
            // all cases including no nulls, all nulls (Null array type),
            // etc.
            for (size_t i = 0; i < arr->length; i++) {
                SetBitTo(bitmask, i, !to_arrow(arr)->IsNull(i));
            }

            // XXX If we want, we can do something smarter like:
            // const uint8_t* arrow_null_bitmask =
            //     arr->to_arrow()->null_bitmap_data();
            // if (arrow_null_bitmask != nullptr)
            //     memcpy(bitmask, arrow_null_bitmask, n_bytes);
            // else {
            //     if (arr->to_arrow()->null_count() ==
            //     arr->to_arrow()->length())
            //         // all null case
            //         memset(bitmask, 0, n_bytes);
            //     else
            //         // no null case
            //         memset(bitmask, 0xff, n_bytes);
            // }

            break;
        }
        default:
            // For cases like INTERVAL and ARRAY_ITEM.
            // These cannot be key columns at this point anyway (see
            // KeyComparisonAsPython_Column).
            throw std::runtime_error(
                "create_temp_null_bitmask_for_array not supported for "
                "array "
                "type: " +
                GetArrType_as_string(arr->arr_type));
    }

    return bitmask;
};

uint8_t* bitwise_and_null_bitmasks(
    const std::vector<std::shared_ptr<array_info>>& arrays,
    const bool is_parallel) {
    tracing::Event ev("bitwise_and_null_bitmasks", is_parallel);
    ev.add_attribute("num_arrays", arrays.size());
    uint64_t length = arrays[0]->length;
    ev.add_attribute("nrows", length);
    int64_t n_bytes = ((length + 7) >> 3);
    uint8_t* final_bitmask = new uint8_t[n_bytes];
    // XXX Might want to optimize for the single key case.
    memset(final_bitmask, 0xff, n_bytes);  // Start off as not nulls

    for (std::shared_ptr<array_info> arr : arrays) {
        uint8_t* arr_null_bitmask;
        bool free_bitmask = false;
        if (arr->null_bitmask()) {
            // Use existing bitmask if one exists
            arr_null_bitmask = (uint8_t*)(arr->null_bitmask());
        } else {
            // Create a temporary one for NUMPY, CATEGORICAL and ARROW
            // arrays. The remaining array types are not supported as join
            // keys.
            arr_null_bitmask = create_temp_null_bitmask_for_array(arr);
            free_bitmask = true;
        }
        // Do a bitwise AND on all the bits, one byte at a time.
        // XXX The compiler should vectorize it automatically
        // (hopefully)?
        for (int i = 0; i < n_bytes; i++) {
            final_bitmask[i] &= arr_null_bitmask[i];
        }
        if (free_bitmask) {
            delete[] arr_null_bitmask;
        }
    }

    return final_bitmask;
};

// ----------------------- Debug functions -----------------------

/** Printing the string expression of an entry in the column
 *
 * @param dtype: the data type on input
 * @param ptrdata: The pointer to the data (its length is determined
 * by dtype)
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
    if (dtype == Bodo_CTypes::DATE || dtype == Bodo_CTypes::INT32) {
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
    // TODO: [BE-4106] Split Time into Time32 and Time64
    if (dtype == Bodo_CTypes::DATETIME || dtype == Bodo_CTypes::TIMEDELTA ||
        dtype == Bodo_CTypes::TIME || dtype == Bodo_CTypes::TIMESTAMPTZ) {
        int64_t* ptr = (int64_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::FLOAT32) {
        float* ptr = (float*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::DECIMAL) {
        __int128_t* val = (__int128_t*)ptrdata;
        return int128_decimal_to_std_string(*val, scale);
    }
    if (dtype == Bodo_CTypes::FLOAT64) {
        double* ptr = (double*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::COMPLEX128) {
        std::complex<double>* ptr = (std::complex<double>*)ptrdata;
        return std::to_string(ptr->real()) + "+" + std::to_string(ptr->imag()) +
               "j";
    }
    if (dtype == Bodo_CTypes::COMPLEX64) {
        std::complex<float>* ptr = (std::complex<float>*)ptrdata;
        return std::to_string(ptr->real()) + "+" + std::to_string(ptr->imag()) +
               "j";
    }
    return "no matching type";
}

std::string GetTimestamptzString(const std::shared_ptr<array_info>& arr,
                                 size_t idx) {
    assert(arr->arr_type == bodo_array_type::TIMESTAMPTZ);
    int64_t ts_utc =
        ((int64_t*)(arr->data1<bodo_array_type::TIMESTAMPTZ>()))[idx];
    int16_t offset_mintues_raw =
        ((int16_t*)(arr->data2<bodo_array_type::TIMESTAMPTZ>()))[idx];
    int16_t offset_hours = std::abs(offset_mintues_raw) / 60;
    int16_t offset_minutes = std::abs(offset_mintues_raw) % 60;
    std::string sign = (offset_mintues_raw > 0 ? " +" : " -");
    // Convert the offset to ±hh:mm format
    std::string hour_str = std::to_string(offset_hours);
    std::string minute_str = std::to_string(offset_minutes);
    if (hour_str.size() == 1) {
        hour_str = "0" + hour_str;
    }
    if (minute_str.size() == 1) {
        minute_str = "0" + minute_str;
    }
    // Extract the second & subsecond components, then convert
    // the seconds to a timestamp (format: "YYYY-MM-DD HH:MM:SS")
    int64_t subsecond = ts_utc % 1000000000;
    auto seconds = (time_t)(ts_utc / 1000000000);
    auto ts = std::gmtime(&seconds);
    char ts_buff[32];
    std::strftime(ts_buff, 32, "%Y-%m-%d %T", ts);
    // Convert the subsecond components into a length-9 string
    // by left-padding with zeros
    std::string subsecond_str = std::to_string(subsecond);
    while (subsecond_str.size() < 9) {
        subsecond_str = "0" + subsecond_str;
    }
    return std::string(ts_buff) + "." + subsecond_str + sign + hour_str + ":" +
           minute_str;
}

template <typename T>
void DEBUG_append_to_primitive_T(const T* values, int64_t offset,
                                 int64_t length, std::string& string_builder,
                                 const std::vector<uint8_t>& valid_elems) {
    string_builder += "[";
    for (int64_t i = 0; i < length; i++) {
        if (i > 0) {
            string_builder += ",";
        }
        if (valid_elems[i]) {
            T val = values[offset + i];
            string_builder += std::to_string(val);
        } else {
            string_builder += "None";
        }
    }
    string_builder += "]";
}

void DEBUG_append_to_primitive_boolean(
    const uint8_t* values, int64_t offset, int64_t length,
    std::string& string_builder, const std::vector<uint8_t>& valid_elems) {
    string_builder += "[";
    for (int64_t i = 0; i < length; i++) {
        if (i > 0) {
            string_builder += ",";
        }
        if (valid_elems[i]) {
            bool bit = arrow::bit_util::GetBit(values, offset + i);
            string_builder += std::to_string(bit);
        } else {
            string_builder += "None";
        }
    }
    string_builder += "]";
}

void DEBUG_append_to_primitive_decimal(
    const __int128_t* values, int64_t offset, int64_t length,
    std::string& string_builder, const std::vector<uint8_t>& valid_elems) {
    string_builder += "[";
    for (int64_t i = 0; i < length; i++) {
        if (i > 0) {
            string_builder += ",";
        }
        if (valid_elems[i]) {
            __int128_t val = values[offset + i];
            int scale = 18;
            string_builder += int128_decimal_to_std_string(val, scale);
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
    if (type == arrow::Type::BOOL) {
        DEBUG_append_to_primitive_boolean((uint8_t*)values, offset, length,
                                          string_builder, valid_elems);
    } else if (type == arrow::Type::INT8) {
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
        DEBUG_append_to_primitive_decimal((__int128_t*)values, offset, length,
                                          string_builder, valid_elems);
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
    } else if (input_array->type_id() == arrow::Type::MAP) {
        auto map_array =
            std::dynamic_pointer_cast<arrow::MapArray>(input_array);
        for (int64_t idx = start_offset; idx < end_offset; idx++) {
            if (idx > start_offset) {
                string_builder += ",";
            }
            if (map_array->IsNull(idx)) {
                string_builder += "None";
                continue;
            } else {
                DEBUG_append_to_out_array(
                    map_array->values(), map_array->value_offset(idx),
                    map_array->value_offset(idx + 1), string_builder);
            }
        }
    } else if (input_array->type_id() == arrow::Type::STRUCT) {
        // TODO: assert builder.type() == STRUCT
        auto struct_array =
            std::dynamic_pointer_cast<arrow::StructArray>(input_array);
        auto struct_type =
            std::dynamic_pointer_cast<arrow::StructType>(struct_array->type());
        for (int64_t idx = start_offset; idx < end_offset; idx++) {
            if (idx > start_offset) {
                string_builder += ",";
            }
            if (struct_array->IsNull(idx)) {
                string_builder += "None";
                continue;
            }
            string_builder += "{";
            for (int i = 0; i < struct_type->num_fields();
                 i++) {  // each field is an array
                if (i > 0) {
                    string_builder += ", ";
                }
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
            if (i > 0) {
                string_builder += ", ";
            }
            if (str_array->IsNull(i)) {
                string_builder += "None";
            } else {
                string_builder += "\"" + str_array->GetString(i) + "\"";
            }
        }
        string_builder += "]";
    } else if (input_array->type_id() == arrow::Type::DICTIONARY) {
        auto str_array =
            std::dynamic_pointer_cast<arrow::DictionaryArray>(input_array);
        auto dict_arr = std::static_pointer_cast<arrow::LargeStringArray>(
            str_array->dictionary());

        string_builder += "[";
        for (int64_t i = start_offset; i < end_offset; i++) {
            if (i > 0) {
                string_builder += ", ";
            }
            if (str_array->IsNull(i)) {
                string_builder += "None";
            } else {
                string_builder +=
                    "\"" + dict_arr->GetString(str_array->GetValueIndex(i)) +
                    "\"";
            }
        }
        string_builder += "]";
    } else {
        int64_t num_elems = end_offset - start_offset;
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(input_array);
        std::vector<uint8_t> valid_elems(num_elems, 0);
        size_t j = 0;
        for (int64_t i = start_offset; i < start_offset + num_elems; i++) {
            valid_elems[j++] = !primitive_array->IsNull(i);
        }
        arrow::Type::type type = primitive_array->type()->id();
        DEBUG_append_to_primitive(type, primitive_array->values()->data(),
                                  start_offset, num_elems, string_builder,
                                  valid_elems);
    }
}

#undef DEBUG_DEBUG  // Yes, it is a concept

bodo::vector<std::string> GetColumn_as_ListString(
    const std::shared_ptr<array_info> arr) {
    size_t nRow = arr->length;
    bodo::vector<std::string> ListStr(nRow);
    std::string strOut;
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        if (arr->dtype == Bodo_CTypes::_BOOL) {
            for (size_t iRow = 0; iRow < nRow; iRow++) {
                bool bit =
                    arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow);
                if (bit) {
                    bool data_bit = GetBit(
                        (uint8_t*)
                            arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                        iRow);
                    strOut = std::to_string(data_bit);
                } else {
                    strOut = "NA";
                }
                ListStr[iRow] = strOut;
            }
        } else {
            uint64_t siztype = numpy_item_size[arr->dtype];
            for (size_t iRow = 0; iRow < nRow; iRow++) {
                bool bit =
                    arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow);
                if (bit) {
                    char* ptrdata1 =
                        &(arr->data1<bodo_array_type::NULLABLE_INT_BOOL>()
                              [siztype * iRow]);
                    strOut =
                        GetStringExpression(arr->dtype, ptrdata1, arr->scale);
                } else {
                    strOut = "NA";
                }
                ListStr[iRow] = strOut;
            }
        }
    }
    if (arr->arr_type == bodo_array_type::NUMPY) {
        uint64_t siztype = numpy_item_size[arr->dtype];
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            char* ptrdata1 =
                &(arr->data1<bodo_array_type::NUMPY>()[siztype * iRow]);
            strOut = GetStringExpression(arr->dtype, ptrdata1, arr->scale);
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::DICT) {
        nRow = arr->child_arrays[1]->length;
        bodo::vector<std::string> dataStr(nRow);
        dataStr = GetColumn_as_ListString(arr->child_arrays[0]);
        bodo::vector<std::string> indexStr(nRow);
        indexStr = GetColumn_as_ListString(arr->child_arrays[1]);
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = arr->get_null_bit<bodo_array_type::DICT>(iRow);
            if (!bit) {
                strOut = "NA";
            } else {
                strOut = dataStr[std::stoi(indexStr[iRow])];
            }
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::STRING) {
        offset_t* data2 = (offset_t*)arr->data2<bodo_array_type::STRING>();
        char* data1 = arr->data1<bodo_array_type::STRING>();
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = arr->get_null_bit<bodo_array_type::STRING>(iRow);
            if (bit) {
                offset_t start_pos = data2[iRow];
                offset_t end_pos = data2[iRow + 1];
                offset_t len = end_pos - start_pos;
                char* strname;
                strname = new char[len + 1];
                for (offset_t i = 0; i < len; i++) {
                    strname[i] = data1[start_pos + i];
                }
                strname[len] = '\0';
                strOut = strname;
                delete[] strname;
            } else {
                strOut = "NA";
            }
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::STRUCT ||
        arr->arr_type == bodo_array_type::ARRAY_ITEM ||
        arr->arr_type == bodo_array_type::MAP) {
        std::shared_ptr<arrow::Array> in_arr = to_arrow(arr);
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            strOut = "";
            DEBUG_append_to_out_array(in_arr, iRow, iRow + 1, strOut);
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::CATEGORICAL) {
        uint64_t siztype = numpy_item_size[arr->dtype];
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            char* ptrdata1 =
                &(arr->data1<bodo_array_type::CATEGORICAL>()[siztype * iRow]);
            strOut = GetStringExpression(arr->dtype, ptrdata1, arr->scale);
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::TIMESTAMPTZ) {
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = arr->get_null_bit<bodo_array_type::TIMESTAMPTZ>(iRow);
            if (bit) {
                strOut = GetTimestamptzString(arr, iRow);
            } else {
                strOut = "NA";
            }
            ListStr[iRow] = strOut;
        }
    }
    return ListStr;
}

void DEBUG_PrintVectorArrayInfo(
    std::ostream& os, std::vector<std::shared_ptr<array_info>> const& ListArr) {
    int nCol = ListArr.size();
    if (nCol == 0) {
        os << "number of columns nCol=" << nCol << " Nothing to print\n";
        return;
    }
    std::vector<int> ListLen(nCol);
    int nRowMax = 0;
    for (int iCol = 0; iCol < nCol; iCol++) {
        int nRow = ListArr[iCol]->length;
        if (nRow > nRowMax) {
            nRowMax = nRow;
        }
        ListLen[iCol] = nRow;
    }
    bodo::vector<bodo::vector<std::string>> ListListStr;
    for (int iCol = 0; iCol < nCol; iCol++) {
        bodo::vector<std::string> LStr = GetColumn_as_ListString(ListArr[iCol]);
        for (int iRow = ListLen[iCol]; iRow < nRowMax; iRow++) {
            LStr.emplace_back("");
        }
        ListListStr.emplace_back(LStr);
    }
    bodo::vector<std::string> ListStrOut(nRowMax);
    for (int iRow = 0; iRow < nRowMax; iRow++) {
        std::string str = std::to_string(iRow) + " :";
        ListStrOut[iRow] = str;
    }
    for (int iCol = 0; iCol < nCol; iCol++) {
        bodo::vector<int> ListLen(nRowMax);
        size_t maxlen = 0;
        for (int iRow = 0; iRow < nRowMax; iRow++) {
            size_t elen = ListListStr[iCol][iRow].size();
            ListLen[iRow] = elen;
            if (elen > maxlen) {
                maxlen = elen;
            }
        }
        for (int iRow = 0; iRow < nRowMax; iRow++) {
            std::string str = ListStrOut[iRow] + " " + ListListStr[iCol][iRow];
            size_t diff = maxlen - ListLen[iRow];
            for (size_t u = 0; u < diff; u++) {
                str += " ";
            }
            ListStrOut[iRow] = str;
        }
    }
    for (int iRow = 0; iRow < nRowMax; iRow++) {
        os << ListStrOut[iRow] << "\n";
    }
}

void DEBUG_PrintSetOfColumn(
    std::ostream& os, std::vector<std::shared_ptr<array_info>> const& ListArr) {
    int nCol = ListArr.size();
    if (nCol == 0) {
        os << "number of columns nCol=" << nCol << " Nothing to print\n";
        return;
    }

    for (int iCol = 0; iCol < nCol; iCol++) {
        os << "Column " << iCol << " : "
           << "arr_type=" << GetArrType_as_string(ListArr[iCol]->arr_type)
           << " dtype=" << GetDtype_as_string(ListArr[iCol]->dtype)
           << std::endl;
    }

    std::vector<int> ListLen(nCol);
    int nRowMax = 0;
    os << "nCol=" << nCol << " List of number of rows:";
    for (int iCol = 0; iCol < nCol; iCol++) {
        int nRow = ListArr[iCol]->length;
        os << " " << nRow;
        if (nRow > nRowMax) {
            nRowMax = nRow;
        }
        ListLen[iCol] = nRow;
    }
    os << "\n";
    DEBUG_PrintVectorArrayInfo(os, ListArr);
}

void DEBUG_PrintTable(std::ostream& os, const table_info* table,
                      bool print_column_names) {
    if (print_column_names) {
        for (auto& val : table->column_names) {
            os << val << " ";
        }
        os << "\n";
    }
    DEBUG_PrintSetOfColumn(os, table->columns);
}

void DEBUG_PrintTable(std::ostream& os,
                      const std::shared_ptr<const table_info>& table,
                      bool print_column_names) {
    DEBUG_PrintTable(os, table.get(), print_column_names);
}

void DEBUG_PrintUnorderedMap(std::ostream& os,
                             std::unordered_map<uint64_t, uint64_t> map) {
    for (auto& it : map) {
        os << it.first << ": " << it.second << "\n";
    }
}

template <typename T>
std::string DEBUG_VectorToString(const std::vector<T>& vec) {
    std::string str = "[";
    for (const T& i : vec) {
        str += fmt::format("{}, ", std::to_string(i));
    }
    str += "]";
    return str;
}

// Explicitly initialize the required templates for loader to be able to
// find them statically.
template std::string DEBUG_VectorToString(const std::vector<int32_t>& vec);
template std::string DEBUG_VectorToString(const std::vector<int64_t>& vec);
template std::string DEBUG_VectorToString(const std::vector<uint32_t>& vec);
template std::string DEBUG_VectorToString(const std::vector<uint64_t>& vec);

void DEBUG_PrintRefct(std::ostream& os,
                      std::vector<std::shared_ptr<array_info>> const& ListArr) {
    int nCol = ListArr.size();
    auto GetNRTinfo = [](NRT_MemInfo* meminf) -> std::string {
        if (meminf == nullptr) {
            return "NULL";
        }
        return "(refct=" + std::to_string(meminf->refct) + ")";
    };
    for (int iCol = 0; iCol < nCol; iCol++) {
        os << "iCol=" << iCol << " : "
           << GetArrType_as_string(ListArr[iCol]->arr_type)
           << " dtype=" << GetDtype_as_string(ListArr[iCol]->dtype);
        if (ListArr[iCol]->arr_type == bodo_array_type::DICT) {
            // Print details from child arrays in the dict case.
            for (auto buffer : ListArr[iCol]->child_arrays[0]->buffers) {
                os << " : child_arrays[0] meminfo="
                   << GetNRTinfo(buffer->getMeminfo()) << "\n";
            }
            for (auto buffer : ListArr[iCol]->child_arrays[1]->buffers) {
                os << " : child_arrays[1] meminfo="
                   << GetNRTinfo(buffer->getMeminfo()) << "\n";
            }
        } else {
            for (auto buffer : ListArr[iCol]->buffers) {
                os << " : meminfo=" << GetNRTinfo(buffer->getMeminfo()) << "\n";
            }
        }
    }
}

void DEBUG_PrintColumn(std::ostream& os,
                       const std::shared_ptr<array_info> arr) {
    int n_rows = arr->length;
    os << "ARRAY_INFO: Column n=" << n_rows
       << " arr=" << GetArrType_as_string(arr->arr_type)
       << " dtype=" << GetDtype_as_string(arr->dtype) << "\n";
    if (arr->arr_type == bodo_array_type::STRUCT) {
        os << "Fields: ";
        for (auto f : arr->field_names) {
            os << f << " ";
        }
        os << "\n";
    }
    bodo::vector<std::string> LStr = GetColumn_as_ListString(arr);
    for (int i_row = 0; i_row < n_rows; i_row++) {
        os << "i_row=" << i_row << " S=" << LStr[i_row] << "\n";
    }
}

/**
 * Used for a custom reduction to merge all the HyperLogLog
 * registers across all ranks.
 *
 * The body of this function is what is done in the
 * `HyperLogLog.merge()` function with some decoration to deal with
 * MPI.
 */
void MPI_hyper_log_log_merge(void* in, void* inout, int* len,
                             MPI_Datatype* dptr) {
    uint8_t* M_in = reinterpret_cast<uint8_t*>(in);
    uint8_t* M_inout = reinterpret_cast<uint8_t*>(inout);
    // The loop below comes from libs/vendored/hyperloglog.hpp:merge()
    // (currently like 161)
    for (int r = 0; r < *len; ++r) {
        if (M_inout[r] < M_in[r]) {
            M_inout[r] |= M_in[r];
        }
    }
}

size_t get_nunique_hashes(const std::shared_ptr</*const*/ uint32_t[]>& hashes,
                          const size_t len, bool is_parallel) {
    tracing::Event ev("get_nunique_hashes", is_parallel);
    hll::HyperLogLog hll(HLL_SIZE);
    hll.addAll(hashes, len);
    const size_t est = std::min(static_cast<size_t>(hll.estimate()), len);
    ev.add_attribute("estimate", est);
    return est;
}

std::pair<size_t, size_t> get_nunique_hashes_global(
    const std::shared_ptr<uint32_t[]> hashes, const size_t len,
    bool is_parallel) {
    tracing::Event ev("get_nunique_hashes_global", is_parallel);
    tracing::Event ev_local("get_nunique_hashes_local", is_parallel);
    hll::HyperLogLog hll(HLL_SIZE);
    hll.addAll(hashes, len);
    size_t local_est = static_cast<size_t>(hll.estimate());
    ev.add_attribute("local_estimate", local_est);
    ev_local.finalize();

    // To get a global estimate of the cardinality we first do a
    // custom reduction of the "registers" in the HyperLogLog. Once
    // the registers have been reduced to rank 0, we overwrite the
    // local register in the hll object, and then compute the
    // estimate.
    //
    // Note: the merge for HLL-HIP is much more complicated and so
    // isn't a trivial replacement. It certainly could be tested,
    // but would require writing more code.
    int my_rank = std::numeric_limits<int>::max();
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    std::vector<uint8_t> hll_registers(hll.data().size(), 0);
    {
        MPI_Op mpi_hll_op;
        CHECK_MPI(MPI_Op_create(&MPI_hyper_log_log_merge, true, &mpi_hll_op),
                  "get_nunique_hashes_global: MPI error on MPI_Op_create:");
        CHECK_MPI(MPI_Reduce(hll.data().data(), hll_registers.data(),
                             hll.data().size(), MPI_UNSIGNED_CHAR, mpi_hll_op,
                             0, MPI_COMM_WORLD),
                  "get_nunique_hashes_global: MPI error on MPI_Reduce:");
        CHECK_MPI(MPI_Op_free(&mpi_hll_op),
                  "get_nunique_hashes_global: MPI error on MPI_Op_free:");
    }

    uint64_t local_len = static_cast<uint64_t>(len);
    uint64_t global_len = 0;
    CHECK_MPI(MPI_Allreduce(&local_len, &global_len, 1, MPI_UNSIGNED_LONG_LONG,
                            MPI_SUM, MPI_COMM_WORLD),
              "get_nunique_hashes_global: MPI error on MPI_Allreduce:");

    uint64_t est;
    if (my_rank == 0) {
        hll.overwrite_registers(std::move(hll_registers));
        using std::min;
        est = min(global_len, static_cast<uint64_t>(hll.estimate()));
    }
    CHECK_MPI(MPI_Bcast(&est, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD),
              "get_nunique_hashes_global: MPI error on MPI_Bcast:");
    // cast to `size_t` to avoid compilation error on Windows
    ev.add_attribute("g_global_estimate", static_cast<size_t>(est));
    return {local_est, est};
}

std::shared_ptr<table_info> concat_tables(
    const std::vector<std::shared_ptr<table_info>>& table_chunks,
    const bool input_is_unpinned) {
    assert(table_chunks.size() > 0);

    auto schema = table_chunks[0]->schema();
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    for (auto& col : schema->column_types) {
        // Note that none of the columns are "keys" from the perspective of the
        // dictionary builder, which is referring to keys for hashing/join
        dict_builders.emplace_back(
            create_dict_builder_for_array(col->copy(), false));
    }

    TableBuildBuffer table_builder(std::move(schema), dict_builders);
    table_builder.ReserveTable(table_chunks, input_is_unpinned);

    for (auto& table : table_chunks) {
        if (input_is_unpinned) {
            table->pin();
        }
        std::shared_ptr<table_info> unified_table =
            unify_dictionary_arrays_helper(table, dict_builders, 0, false);
        // TODO(aneesh) we should take ownership of the vector passed in the
        // clear the pointer to the original table here to try to deallocate it
        // earlier.
        table_builder.UnsafeAppendBatch(std::move(unified_table));
        if (input_is_unpinned) {
            table->unpin();
        }
    }

    return table_builder.data_table;
}

std::shared_ptr<array_info> concat_arrays(
    std::vector<std::shared_ptr<array_info>>& arrays) {
    // Create dummy tables to pass to concat_tables
    std::vector<std::shared_ptr<table_info>> dummy_tables(arrays.size());
    for (size_t i = 0; i < arrays.size(); i++) {
        dummy_tables[i] = std::make_shared<table_info>();
        dummy_tables[i]->columns.push_back(arrays[i]);
    }

    // Concat the dummy tables and retrieve the concatenated column
    std::shared_ptr<table_info> concatenated_table =
        concat_tables(dummy_tables);
    std::shared_ptr<array_info> concatenated_arr =
        concatenated_table->columns[0];

    return concatenated_arr;
}

std::shared_ptr<table_info> concat_tables(
    std::vector<std::shared_ptr<table_info>>&& table_chunks,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    const bool input_is_unpinned) {
    auto schema = table_chunks[0]->schema();
    TableBuildBuffer table_builder(std::move(schema), dict_builders);
    table_builder.ReserveTable(table_chunks, input_is_unpinned);

    for (auto& table : table_chunks) {
        if (input_is_unpinned) {
            table->pin();
        }
        table_builder.UnsafeAppendBatch(table);
        if (input_is_unpinned) {
            table->unpin();
        }
        table.reset();
    }

    return table_builder.data_table;
}

std::string array_val_to_str(std::shared_ptr<array_info> arr, size_t idx) {
    switch (arr->dtype) {
        case Bodo_CTypes::INT8:
            return std::to_string(arr->at<int8_t>(idx));
        case Bodo_CTypes::UINT8:
            return std::to_string(arr->at<uint8_t>(idx));
        case Bodo_CTypes::INT32:
            return std::to_string(arr->at<int32_t>(idx));
        case Bodo_CTypes::UINT32:
            return std::to_string(arr->at<uint32_t>(idx));
        case Bodo_CTypes::INT64:
            return std::to_string(arr->at<int64_t>(idx));
        case Bodo_CTypes::UINT64:
            return std::to_string(arr->at<uint64_t>(idx));
        case Bodo_CTypes::FLOAT32:
            return std::to_string(arr->at<float>(idx));
        case Bodo_CTypes::FLOAT64:
            return std::to_string(arr->at<double>(idx));
        case Bodo_CTypes::INT16:
            return std::to_string(arr->at<int16_t>(idx));
        case Bodo_CTypes::UINT16:
            return std::to_string(arr->at<uint16_t>(idx));
        case Bodo_CTypes::STRING: {
            if (arr->arr_type == bodo_array_type::DICT) {
                // In case of dictionary encoded string array
                // get the string value by indexing into the dictionary
                return array_val_to_str(
                    arr->child_arrays[0],
                    arr->child_arrays[1]
                        ->at<dict_indices_t,
                             bodo_array_type::NULLABLE_INT_BOOL>(idx));
            }
            offset_t* offsets =
                (offset_t*)arr->data2<bodo_array_type::STRING>();
            return std::string(
                arr->data1<bodo_array_type::STRING>() + offsets[idx],
                offsets[idx + 1] - offsets[idx]);
        }
        case Bodo_CTypes::DATE: {
            int64_t day = arr->at<int32_t>(idx);
            int64_t year = days_to_yearsdays(&day);
            int64_t month;
            get_month_day(year, day, &month, &day);
            std::string date_str;
            date_str.reserve(10);
            date_str += std::to_string(year) + "-";
            if (month < 10) {
                date_str += "0";
            }
            date_str += std::to_string(month) + "-";
            if (day < 10) {
                date_str += "0";
            }
            date_str += std::to_string(day);
            return date_str;
        }
        case Bodo_CTypes::_BOOL:
            bool val;
            if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                val = GetBit(
                    (uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                    idx);
            } else {
                val = arr->at<bool>(idx);
            }
            if (val) {
                return "True";
            } else {
                return "False";
            }
        default: {
            std::vector<char> error_msg(100);
            snprintf(error_msg.data(), error_msg.size(),
                     "array_val_to_str not implemented for dtype %d",
                     arr->dtype);
            throw std::runtime_error(error_msg.data());
        }
    }
}
