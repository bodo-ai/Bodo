#include "_bodo_common.h"

#include <arrow/array.h>
#include <complex>

#include <fmt/format.h>
#include "_array_utils.h"
#include "_bodo_to_arrow.h"
#include "_datetime_utils.h"
#include "_distributed.h"

// for numpy arrays, this maps dtype to sizeof(dtype)
// Order should match Bodo_CTypes::CTypeEnum
const std::vector<size_t> numpy_item_size({
    sizeof(int8_t),    // INT8
    sizeof(uint8_t),   // UINT8
    sizeof(int32_t),   // INT32
    sizeof(uint32_t),  // UINT32
    sizeof(int64_t),   // INT64
    sizeof(float),     // FLOAT32
    sizeof(double),    // FLOAT64
    sizeof(uint64_t),  // UINT64
    sizeof(int16_t),   // INT16
    sizeof(uint16_t),  // UINT16
    0,                 // STRING
    // Note: This is only true for Numpy Boolean arrays
    // and should be removed when we purge Numpy Boolean arrays
    // from C++.
    sizeof(bool),       // _BOOL
    BYTES_PER_DECIMAL,  // DECIMAL
    sizeof(int32_t),    // DATE
    // TODO: [BE-4106] TIME size should depend on precision.
    sizeof(int64_t),    // TIME
    sizeof(int64_t),    // DATETIME
    sizeof(int64_t),    // TIMEDELTA
    BYTES_PER_DECIMAL,  // INT128
    0,                  // LIST
    0,                  // STRUCT
    0,                  // BINARY
    // std::complex is bit compatible with fftw_complex and C99 fftw_complex
    // https://www.fftw.org/fftw3_doc/Complex-numbers.html
    sizeof(std::complex<float>),   // COMPLEX64
    sizeof(std::complex<double>),  // COMPLEX128
    0,                             // MAP
    sizeof(int64_t),               // TIMESTAMPTZ data1
});
void bodo_common_init() {
    static bool initialized = false;
    if (initialized) {
        return;
    }
    initialized = true;

    if (numpy_item_size.size() != Bodo_CTypes::_numtypes) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Incorrect number of bodo item sizes!");
        return;
    }

    PyObject* np_mod = PyImport_ImportModule("numpy");
    PyObject* dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "bool");
    if ((size_t)PyNumber_AsSsize_t(
            PyObject_GetAttrString(dtype_obj, "itemsize"), NULL) !=
        sizeof(bool)) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "bool size mismatch between C++ and NumPy!");
        return;
    }
    dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "float32");
    if ((size_t)PyNumber_AsSsize_t(
            PyObject_GetAttrString(dtype_obj, "itemsize"), NULL) !=
        sizeof(float)) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "float32 size mismatch between C++ and NumPy!");
        return;
    }
    dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "float64");
    if ((size_t)PyNumber_AsSsize_t(
            PyObject_GetAttrString(dtype_obj, "itemsize"), NULL) !=
        sizeof(double)) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "float64 size mismatch between C++ and NumPy!");
        return;
    }
}

void Bodo_PyErr_SetString(PyObject* type, const char* message) {
    PyErr_SetString(type, message);
    throw std::runtime_error(message);
}

Bodo_CTypes::CTypeEnum arrow_to_bodo_type(arrow::Type::type type) {
    switch (type) {
        case arrow::Type::INT8:
            return Bodo_CTypes::INT8;
        case arrow::Type::UINT8:
            return Bodo_CTypes::UINT8;
        case arrow::Type::INT16:
            return Bodo_CTypes::INT16;
        case arrow::Type::UINT16:
            return Bodo_CTypes::UINT16;
        case arrow::Type::INT32:
            return Bodo_CTypes::INT32;
        case arrow::Type::UINT32:
            return Bodo_CTypes::UINT32;
        case arrow::Type::INT64:
            return Bodo_CTypes::INT64;
        case arrow::Type::UINT64:
            return Bodo_CTypes::UINT64;
        case arrow::Type::FLOAT:
            return Bodo_CTypes::FLOAT32;
        case arrow::Type::DOUBLE:
            return Bodo_CTypes::FLOAT64;
        case arrow::Type::DECIMAL:
            return Bodo_CTypes::DECIMAL;
        case arrow::Type::TIMESTAMP:
            return Bodo_CTypes::DATETIME;
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
            // The main difference between
            // STRING and LARGE_STRING is the offset size
            // (uint32 for STRING and uint64 for LARGE_STRING).
            // We use 64 bit offsets for everything in Bodo,
            // so these two types are equivalent for us.
            return Bodo_CTypes::STRING;
        case arrow::Type::BINARY:
            return Bodo_CTypes::BINARY;
        case arrow::Type::DATE32:
            return Bodo_CTypes::DATE;
        case arrow::Type::TIME32:
            return Bodo_CTypes::TIME;
        case arrow::Type::TIME64:
            return Bodo_CTypes::TIME;
        case arrow::Type::BOOL:
            return Bodo_CTypes::_BOOL;
        // TODO Timedelta
        default: {
            // TODO: Construct the type from the id
            throw std::runtime_error("arrow_to_bodo_type");
        }
    }
}

// --------------------- bodo::DataType and bodo::Schema ---------------------

static const char* arr_type_to_str(bodo_array_type::arr_type_enum arr_type) {
    if (arr_type == bodo_array_type::NUMPY) {
        return "NUMPY";
    } else if (arr_type == bodo_array_type::STRING) {
        return "STRING";
    } else if (arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        return "NULLABLE";
    } else if (arr_type == bodo_array_type::ARRAY_ITEM) {
        return "ARRAY_ITEM";
    } else if (arr_type == bodo_array_type::STRUCT) {
        return "STRUCT";
    } else if (arr_type == bodo_array_type::MAP) {
        return "MAP";
    } else if (arr_type == bodo_array_type::CATEGORICAL) {
        return "CATEGORICAL";
    } else if (arr_type == bodo_array_type::DICT) {
        return "DICT";
    } else if (arr_type == bodo_array_type::TIMESTAMPTZ) {
        return "TIMESTAMPTZ";
    } else {
        return "UNKNOWN";
    }
}

static const char* dtype_to_str(Bodo_CTypes::CTypeEnum dtype) {
    if (dtype == Bodo_CTypes::INT8) {
        return "INT8";
    } else if (dtype == Bodo_CTypes::UINT8) {
        return "UINT8";
    } else if (dtype == Bodo_CTypes::INT16) {
        return "INT16";
    } else if (dtype == Bodo_CTypes::UINT16) {
        return "UINT16";
    } else if (dtype == Bodo_CTypes::INT32) {
        return "INT32";
    } else if (dtype == Bodo_CTypes::UINT32) {
        return "UINT32";
    } else if (dtype == Bodo_CTypes::INT64) {
        return "INT64";
    } else if (dtype == Bodo_CTypes::UINT64) {
        return "UINT64";
    } else if (dtype == Bodo_CTypes::FLOAT32) {
        return "FLOAT32";
    } else if (dtype == Bodo_CTypes::FLOAT64) {
        return "FLOAT64";
    } else if (dtype == Bodo_CTypes::STRING) {
        return "STRING";
    } else if (dtype == Bodo_CTypes::BINARY) {
        return "BINARY";
    } else if (dtype == Bodo_CTypes::_BOOL) {
        return "_BOOL";
    } else if (dtype == Bodo_CTypes::DECIMAL) {
        return "DECIMAL";
    } else if (dtype == Bodo_CTypes::INT128) {
        return "INT128";
    } else if (dtype == Bodo_CTypes::DATE) {
        return "DATE";
    } else if (dtype == Bodo_CTypes::DATETIME) {
        return "DATETIME";
    } else if (dtype == Bodo_CTypes::TIMEDELTA) {
        return "TIMEDELTA";
    } else if (dtype == Bodo_CTypes::TIME) {
        return "TIME";
    } else if (dtype == Bodo_CTypes::LIST) {
        return "LIST";
    } else if (dtype == Bodo_CTypes::STRUCT) {
        return "STRUCT";
    } else if (dtype == Bodo_CTypes::MAP) {
        return "MAP";
    } else if (dtype == Bodo_CTypes::COMPLEX64) {
        return "COMPLEX64";
    } else if (dtype == Bodo_CTypes::COMPLEX128) {
        return "COMPLEX128";
    } else if (dtype == Bodo_CTypes::TIMESTAMPTZ) {
        return "TIMESTAMPTZ";
    } else {
        return "UNKNOWN";
    }
}

std::string GetDtype_as_string(Bodo_CTypes::CTypeEnum const& dtype) {
    return dtype_to_str(dtype);
}

std::string GetArrType_as_string(bodo_array_type::arr_type_enum arr_type) {
    return arr_type_to_str(arr_type);
}

namespace bodo {

std::unique_ptr<DataType> DataType::copy() const {
    if (this->is_array()) {
        const ArrayType* this_as_array = static_cast<const ArrayType*>(this);
        return std::make_unique<ArrayType>(this_as_array->value_type->copy());

    } else if (this->is_map()) {
        const MapType* this_as_map = static_cast<const MapType*>(this);
        return std::make_unique<MapType>(this_as_map->key_type->copy(),
                                         this_as_map->value_type->copy());
    } else if (this->is_struct()) {
        const StructType* this_as_struct = static_cast<const StructType*>(this);
        std::vector<std::unique_ptr<DataType>> new_child_types;
        new_child_types.reserve(this_as_struct->child_types.size());
        for (const auto& t : this_as_struct->child_types) {
            new_child_types.push_back(t->copy());
        }
        return std::make_unique<StructType>(std::move(new_child_types));

    } else {
        return std::make_unique<DataType>(this->array_type, this->c_type);
    }
}

std::unique_ptr<DataType> DataType::to_nullable_type() const {
    if (this->is_array()) {
        const ArrayType* this_as_array = static_cast<const ArrayType*>(this);
        return std::make_unique<ArrayType>(
            this_as_array->value_type->to_nullable_type());

    } else if (this->is_map()) {
        const MapType* this_as_map = static_cast<const MapType*>(this);
        return std::make_unique<MapType>(
            this_as_map->key_type->to_nullable_type(),
            this_as_map->value_type->to_nullable_type());
    } else if (this->is_struct()) {
        const StructType* this_as_struct = static_cast<const StructType*>(this);
        std::vector<std::unique_ptr<DataType>> new_child_types;
        new_child_types.reserve(this_as_struct->child_types.size());
        for (const auto& t : this_as_struct->child_types) {
            new_child_types.push_back(t->to_nullable_type());
        }
        return std::make_unique<StructType>(std::move(new_child_types));

    } else {
        bodo_array_type::arr_type_enum arr_type = this->array_type;
        Bodo_CTypes::CTypeEnum dtype = this->c_type;
        if ((arr_type == bodo_array_type::NUMPY) &&
            (is_integer(dtype) || is_float(dtype) ||
             dtype == Bodo_CTypes::_BOOL)) {
            arr_type = bodo_array_type::NULLABLE_INT_BOOL;
        }
        return std::make_unique<DataType>(arr_type, dtype);
    }
}

void DataType::to_string_inner(std::string& out) {
    out += arr_type_to_str(this->array_type);
    out += "[";
    out += dtype_to_str(this->c_type);
    out += "]";
}

void ArrayType::to_string_inner(std::string& out) {
    out += arr_type_to_str(this->array_type);
    out += "[";
    this->value_type->to_string_inner(out);
    out += "]";
}

void StructType::to_string_inner(std::string& out) {
    out += arr_type_to_str(this->array_type);
    out += "[";

    for (size_t i = 0; i < child_types.size(); i++) {
        if (i != 0) {
            out += ", ";
        }
        out += std::to_string(i);
        out += ": ";
        child_types[i]->to_string_inner(out);
    }

    out += "]";
}

void MapType::to_string_inner(std::string& out) {
    out += arr_type_to_str(this->array_type);
    out += "[";
    this->key_type->to_string_inner(out);
    out += ", ";
    this->value_type->to_string_inner(out);
    out += "]";
}

void DataType::Serialize(std::vector<int8_t>& arr_array_types,
                         std::vector<int8_t>& arr_c_types) const {
    arr_array_types.push_back(array_type);
    arr_c_types.push_back(c_type);
}

void ArrayType::Serialize(std::vector<int8_t>& arr_array_types,
                          std::vector<int8_t>& arr_c_types) const {
    arr_array_types.push_back(array_type);
    arr_c_types.push_back(c_type);
    value_type->Serialize(arr_array_types, arr_c_types);
}

void StructType::Serialize(std::vector<int8_t>& arr_array_types,
                           std::vector<int8_t>& arr_c_types) const {
    arr_array_types.push_back(array_type);
    arr_c_types.push_back(c_type);
    arr_array_types.push_back(child_types.size());
    arr_c_types.push_back(child_types.size());

    for (auto& child_type : child_types) {
        child_type->Serialize(arr_array_types, arr_c_types);
    }
}

void MapType::Serialize(std::vector<int8_t>& arr_array_types,
                        std::vector<int8_t>& arr_c_types) const {
    arr_array_types.push_back(array_type);
    arr_c_types.push_back(c_type);
    key_type->Serialize(arr_array_types, arr_c_types);
    value_type->Serialize(arr_array_types, arr_c_types);
}

static std::unique_ptr<DataType> from_byte_helper(
    const std::span<const int8_t> arr_array_types,
    const std::span<const int8_t> arr_c_types, size_t& i) {
    auto array_type =
        static_cast<bodo_array_type::arr_type_enum>(arr_array_types[i]);
    auto c_type = static_cast<Bodo_CTypes::CTypeEnum>(arr_c_types[i]);
    i += 1;

    if (array_type == bodo_array_type::ARRAY_ITEM) {
        auto inner_arr = from_byte_helper(arr_array_types, arr_c_types, i);
        return std::make_unique<ArrayType>(std::move(inner_arr));

    } else if (array_type == bodo_array_type::STRUCT) {
        std::vector<std::unique_ptr<DataType>> child_types;
        size_t num_fields = static_cast<size_t>(arr_array_types[i]);
        i += 1;

        for (size_t j = 0; j < num_fields; j++) {
            child_types.push_back(
                from_byte_helper(arr_array_types, arr_c_types, i));
        }
        return std::make_unique<StructType>(std::move(child_types));
    } else if (array_type == bodo_array_type::MAP) {
        std::unique_ptr<DataType> key_arr =
            from_byte_helper(arr_array_types, arr_c_types, i);
        std::unique_ptr<DataType> value_arr =
            from_byte_helper(arr_array_types, arr_c_types, i);
        return std::make_unique<MapType>(std::move(key_arr),
                                         std::move(value_arr));
    } else {
        return std::make_unique<DataType>(array_type, c_type);
    }
}

std::unique_ptr<DataType> DataType::Deserialize(
    const std::span<const int8_t> arr_array_types,
    const std::span<const int8_t> arr_c_types) {
    size_t i = 0;
    return from_byte_helper(arr_array_types, arr_c_types, i);
}

Schema::Schema() : column_types() {}
Schema::Schema(const Schema& other) {
    this->column_types.reserve(other.column_types.size());
    for (const auto& t : other.column_types) {
        this->column_types.push_back(t->copy());
    }
}

Schema::Schema(Schema&& other) {
    this->column_types = std::move(other.column_types);
}

Schema::Schema(std::vector<std::unique_ptr<bodo::DataType>>&& column_types_)
    : column_types(std::move(column_types_)) {}
void Schema::insert_column(const int8_t arr_array_type, const int8_t arr_c_type,
                           const size_t idx) {
    size_t i = 0;
    this->insert_column(from_byte_helper(std::vector<int8_t>({arr_array_type}),
                                         std::vector<int8_t>({arr_c_type}), i),
                        idx);
}

void Schema::insert_column(std::unique_ptr<DataType>&& col, const size_t idx) {
    this->column_types.insert(column_types.begin() + idx, std::move(col));
}

void Schema::append_column(std::unique_ptr<DataType>&& col) {
    this->column_types.emplace_back(std::move(col));
}
void Schema::append_column(const int8_t arr_array_type,
                           const int8_t arr_c_type) {
    size_t i = 0;
    this->append_column(from_byte_helper(std::vector<int8_t>({arr_array_type}),
                                         std::vector<int8_t>({arr_c_type}), i));
}

void Schema::append_schema(std::unique_ptr<Schema>&& other_schema) {
    for (auto& col : other_schema->column_types) {
        this->column_types.push_back(std::move(col));
    }
}
size_t Schema::ncols() const { return this->column_types.size(); }

std::unique_ptr<Schema> Schema::Deserialize(
    const std::span<const int8_t> arr_array_types,
    const std::span<const int8_t> arr_c_types) {
    std::unique_ptr<Schema> schema = std::make_unique<Schema>();
    schema->column_types.reserve(arr_array_types.size());
    size_t i = 0;

    while (i < arr_array_types.size()) {
        schema->column_types.push_back(
            from_byte_helper(arr_array_types, arr_c_types, i));
    }

    return schema;
}

std::pair<std::vector<int8_t>, std::vector<int8_t>> Schema::Serialize() const {
    std::vector<int8_t> arr_array_types;
    std::vector<int8_t> arr_c_types;

    for (auto& arr_type : column_types) {
        arr_type->Serialize(arr_array_types, arr_c_types);
    }

    return std::pair(arr_array_types, arr_c_types);
}

std::string Schema::ToString() {
    std::string out;
    for (size_t i = 0; i < this->column_types.size(); i++) {
        if (i > 0) {
            out += "\n";
        }
        out += fmt::format("{}: {}", i, this->column_types[i]->ToString());
    }
    return out;
}

}  // namespace bodo

// ---------------------------------------------------------------------------

std::shared_ptr<arrow::Array> to_arrow(const std::shared_ptr<array_info> arr) {
    std::shared_ptr<arrow::Array> arrow_arr;
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
    bodo_array_to_arrow(bodo::BufferPool::DefaultPtr(), std::move(arr),
                        &arrow_arr, false /*convert_timedelta_to_int64*/, "",
                        time_unit, false /*downcast_time_ns_to_us*/,
                        bodo::default_buffer_memory_manager());
    return arrow_arr;
}

std::unique_ptr<BodoBuffer> AllocateBodoBuffer(
    const int64_t size, bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    NRT_MemInfo* meminfo =
        NRT_MemInfo_alloc_safe_aligned_pool(size, ALIGNMENT, pool);
    return std::make_unique<BodoBuffer>((uint8_t*)meminfo->data, size, meminfo,
                                        false, pool, std::move(mm));
}

std::unique_ptr<BodoBuffer> AllocateBodoBuffer(
    const int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t itemsize = numpy_item_size[typ_enum];
    int64_t size = length * itemsize;
    return AllocateBodoBuffer(size, pool, std::move(mm));
}

std::string array_info::val_to_str(size_t idx) {
    switch (dtype) {
        case Bodo_CTypes::INT8:
            return std::to_string(this->at<int8_t>(idx));
        case Bodo_CTypes::UINT8:
            return std::to_string(this->at<uint8_t>(idx));
        case Bodo_CTypes::INT32:
            return std::to_string(this->at<int32_t>(idx));
        case Bodo_CTypes::UINT32:
            return std::to_string(this->at<uint32_t>(idx));
        case Bodo_CTypes::INT64:
            return std::to_string(this->at<int64_t>(idx));
        case Bodo_CTypes::UINT64:
            return std::to_string(this->at<uint64_t>(idx));
        case Bodo_CTypes::FLOAT32:
            return std::to_string(this->at<float>(idx));
        case Bodo_CTypes::FLOAT64:
            return std::to_string(this->at<double>(idx));
        case Bodo_CTypes::INT16:
            return std::to_string(this->at<int16_t>(idx));
        case Bodo_CTypes::UINT16:
            return std::to_string(this->at<uint16_t>(idx));
        case Bodo_CTypes::STRING: {
            if (this->arr_type == bodo_array_type::DICT) {
                // In case of dictionary encoded string array
                // get the string value by indexing into the dictionary
                return this->child_arrays[0]->val_to_str(
                    this->child_arrays[1]
                        ->at<dict_indices_t,
                             bodo_array_type::NULLABLE_INT_BOOL>(idx));
            }
            offset_t* offsets =
                (offset_t*)this->data2<bodo_array_type::STRING>();
            return std::string(
                this->data1<bodo_array_type::STRING>() + offsets[idx],
                offsets[idx + 1] - offsets[idx]);
        }
        case Bodo_CTypes::DATE: {
            int64_t day = this->at<int32_t>(idx);
            int64_t year = days_to_yearsdays(&day);
            int64_t month;
            get_month_day(year, day, &month, &day);
            std::string date_str;
            date_str.reserve(10);
            date_str += std::to_string(year) + "-";
            if (month < 10)
                date_str += "0";
            date_str += std::to_string(month) + "-";
            if (day < 10)
                date_str += "0";
            date_str += std::to_string(day);
            return date_str;
        }
        case Bodo_CTypes::_BOOL:
            bool val;
            if (this->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                val = GetBit(
                    (uint8_t*)this->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                    idx);
            } else {
                val = this->at<bool>(idx);
            }
            if (val) {
                return "True";
            } else {
                return "False";
            }
        default: {
            std::vector<char> error_msg(100);
            snprintf(error_msg.data(), error_msg.size(),
                     "val_to_str not implemented for dtype %d", dtype);
            throw std::runtime_error(error_msg.data());
        }
    }
}

std::unique_ptr<array_info> alloc_numpy(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t size = length * numpy_item_size[typ_enum];
    std::unique_ptr<BodoBuffer> buffer =
        AllocateBodoBuffer(size, pool, std::move(mm));
    return std::make_unique<array_info>(
        bodo_array_type::NUMPY, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>({std::move(buffer)}));
}

std::unique_ptr<array_info> alloc_interval_array(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t size = length * numpy_item_size[typ_enum];
    std::unique_ptr<BodoBuffer> left_buffer =
        AllocateBodoBuffer(size, pool, mm);
    std::unique_ptr<BodoBuffer> right_buffer =
        AllocateBodoBuffer(size, pool, std::move(mm));
    return std::make_unique<array_info>(
        bodo_array_type::INTERVAL, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(left_buffer), std::move(right_buffer)}));
}

std::unique_ptr<array_info> alloc_array_item(
    int64_t n_arrays, std::shared_ptr<array_info> inner_arr,
    int64_t extra_null_bytes, bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    std::unique_ptr<BodoBuffer> offsets_buffer =
        AllocateBodoBuffer(n_arrays + 1, Bodo_CType_offset, pool, mm);
    int64_t n_bytes = ((n_arrays + 7) >> 3) + extra_null_bytes;
    std::unique_ptr<BodoBuffer> null_bitmap_buffer =
        AllocateBodoBuffer(n_bytes, Bodo_CTypes::UINT8, pool, std::move(mm));
    // Set offset buffer to all zeros. Not setting this will result in known
    // errors as we use offsets[size] as a reference when appending new rows.
    memset(offsets_buffer->mutable_data(), 0,
           (n_arrays + 1) * sizeof(offset_t));
    // Set null bitmask to all ones to avoid unexpected issues
    memset(null_bitmap_buffer->mutable_data(), 0xff, n_bytes);
    return std::make_unique<array_info>(
        bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST, n_arrays,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(offsets_buffer), std::move(null_bitmap_buffer)}),
        std::vector<std::shared_ptr<array_info>>({inner_arr}));
}

std::unique_ptr<array_info> alloc_struct(
    int64_t length, std::vector<std::shared_ptr<array_info>> child_arrays,
    int64_t extra_null_bytes, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    std::unique_ptr<BodoBuffer> buffer_bitmask =
        AllocateBodoBuffer(n_bytes, pool, std::move(mm));
    // Set null bitmask to all ones to avoid unexpected issues
    memset(buffer_bitmask->mutable_data(), 0xff, n_bytes);
    return std::make_unique<array_info>(
        bodo_array_type::STRUCT, Bodo_CTypes::CTypeEnum::STRUCT, length,
        std::vector<std::shared_ptr<BodoBuffer>>({std::move(buffer_bitmask)}),
        std::move(child_arrays));
}

std::unique_ptr<array_info> alloc_map(int64_t n_rows,
                                      std::shared_ptr<array_info> inner_arr) {
    return std::make_unique<array_info>(
        bodo_array_type::MAP, Bodo_CTypes::MAP, n_rows,
        std::vector<std::shared_ptr<BodoBuffer>>({}),
        std::vector<std::shared_ptr<array_info>>({std::move(inner_arr)}));
}

std::unique_ptr<array_info> alloc_categorical(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t num_categories,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t size = length * numpy_item_size[typ_enum];
    std::unique_ptr<BodoBuffer> buffer =
        AllocateBodoBuffer(size, pool, std::move(mm));
    return std::make_unique<array_info>(
        bodo_array_type::CATEGORICAL, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>({std::move(buffer)}),
        std::vector<std::shared_ptr<array_info>>({}), 0, 0, num_categories);
}

std::unique_ptr<array_info> alloc_nullable_array(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    int64_t size;
    if (typ_enum == Bodo_CTypes::_BOOL) {
        // Boolean arrays store 1 bit per element.
        // Note extra_null_bytes are used for padding when we need
        // to shuffle and split the entries into separate bytes, so
        // we need these for the data as well.
        size = n_bytes;
    } else {
        size = length * numpy_item_size[typ_enum];
    }
    std::unique_ptr<BodoBuffer> buffer = AllocateBodoBuffer(size, pool, mm);
    std::unique_ptr<BodoBuffer> buffer_bitmask =
        AllocateBodoBuffer(n_bytes * sizeof(uint8_t), pool, std::move(mm));
    return std::make_unique<array_info>(
        bodo_array_type::NULLABLE_INT_BOOL, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(buffer), std::move(buffer_bitmask)}));
}

std::unique_ptr<array_info> alloc_nullable_array_no_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    // Same as alloc_nullable_array but we set the null_bitmask
    // such that there are no null values in the output.
    // Useful for cases like allocating indices array of dictionary-encoded
    // string arrays such as input_file_name column where nulls are not possible
    std::unique_ptr<array_info> arr = alloc_nullable_array(
        length, typ_enum, extra_null_bytes, pool, std::move(mm));
    size_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    memset(arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(), 0xff,
           n_bytes);  // null not possible
    return arr;
}

std::unique_ptr<array_info> alloc_nullable_array_all_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    // Same as alloc_nullable_array but we set the null_bitmask
    // such that all values are null values in the output.
    // Useful for cases like the iceberg void transform.
    std::unique_ptr<array_info> arr = alloc_nullable_array(
        length, typ_enum, extra_null_bytes, pool, std::move(mm));
    size_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    memset(arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(), 0x00,
           n_bytes);  // all nulls
    return arr;
}

std::unique_ptr<array_info> alloc_string_array(
    Bodo_CTypes::CTypeEnum typ_enum, int64_t length, int64_t n_chars,
    int64_t array_id, int64_t extra_null_bytes, bool is_globally_replicated,
    bool is_locally_unique, bool is_locally_sorted,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // allocate data/offsets/null_bitmap arrays
    std::unique_ptr<BodoBuffer> data_buffer =
        AllocateBodoBuffer(n_chars, Bodo_CTypes::UINT8, pool, mm);
    std::unique_ptr<BodoBuffer> offsets_buffer =
        AllocateBodoBuffer(length + 1, Bodo_CType_offset, pool, mm);
    int64_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    std::unique_ptr<BodoBuffer> null_bitmap_buffer =
        AllocateBodoBuffer(n_bytes, Bodo_CTypes::UINT8, pool, std::move(mm));
    // setting all to non-null to avoid unexpected issues
    memset(null_bitmap_buffer->mutable_data(), 0xff, n_bytes);

    // set offsets for boundaries
    offset_t* offsets_ptr = (offset_t*)offsets_buffer->mutable_data();
    offsets_ptr[0] = 0;
    offsets_ptr[length] = n_chars;

    // Generate a valid array id
    if (array_id < 0) {
        array_id = generate_array_id(length);
    }

    return std::make_unique<array_info>(
        bodo_array_type::STRING, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(data_buffer), std::move(offsets_buffer),
             std::move(null_bitmap_buffer)}),
        std::vector<std::shared_ptr<array_info>>({}), 0, 0, 0, array_id,
        is_globally_replicated, is_locally_unique, is_locally_sorted);
}

std::unique_ptr<array_info> alloc_dict_string_array(
    int64_t length, int64_t n_keys, int64_t n_chars_keys,
    int64_t extra_null_bytes, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // dictionary
    std::shared_ptr<array_info> dict_data_arr =
        alloc_string_array(Bodo_CTypes::CTypeEnum::STRING, n_keys, n_chars_keys,
                           -1, 0, false, false, false, pool, mm);
    // indices
    std::shared_ptr<array_info> indices_data_arr = alloc_nullable_array(
        length, Bodo_CTypes::INT32, extra_null_bytes, pool, std::move(mm));

    return std::make_unique<array_info>(
        bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING, length,
        std::vector<std::shared_ptr<BodoBuffer>>({}),
        std::vector<std::shared_ptr<array_info>>(
            {dict_data_arr, indices_data_arr}));
}

std::unique_ptr<array_info> alloc_timestamptz_array(
    int64_t length, int64_t extra_null_bytes, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    auto timestamp_buffer =
        AllocateBodoBuffer(length, Bodo_CTypes::CTypeEnum::DATETIME, pool, mm);
    // tz offset in ns
    auto offset_buffer =
        AllocateBodoBuffer(length, Bodo_CTypes::CTypeEnum::INT16, pool, mm);

    int64_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    auto null_bitmap_buffer =
        AllocateBodoBuffer(n_bytes * sizeof(uint8_t), pool, mm);
    // setting all to non-null to avoid unexpected issues
    memset(null_bitmap_buffer->mutable_data(), 0xff, n_bytes);

    auto arr_type = bodo_array_type::arr_type_enum::TIMESTAMPTZ;
    auto dtype = Bodo_CTypes::CTypeEnum::TIMESTAMPTZ;
    return std::make_unique<array_info>(
        arr_type, dtype, length,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(timestamp_buffer), std::move(offset_buffer),
             std::move(null_bitmap_buffer)}),
        std::vector<std::shared_ptr<array_info>>({}));
}

std::unique_ptr<array_info> create_string_array(
    Bodo_CTypes::CTypeEnum typ_enum, bodo::vector<uint8_t> const& null_bitmap,
    bodo::vector<std::string> const& list_string, int64_t array_id,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    size_t len = list_string.size();
    // Calculate the number of characters for allocating the string.
    size_t nb_char = 0;
    bodo::vector<std::string>::const_iterator iter = list_string.begin();
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        if (GetBit(null_bitmap.data(), i_grp)) {
            nb_char += iter->size();
        }
        iter++;
    }
    std::unique_ptr<array_info> out_col =
        alloc_string_array(typ_enum, len, nb_char, array_id, 0, false, false,
                           false, pool, std::move(mm));
    // update string array payload to reflect change
    char* data_o = out_col->data1<bodo_array_type::STRING>();
    offset_t* offsets_o = (offset_t*)out_col->data2<bodo_array_type::STRING>();
    offset_t pos = 0;
    iter = list_string.begin();
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        offsets_o[i_grp] = pos;
        bool bit = GetBit(null_bitmap.data(), i_grp);
        if (bit) {
            size_t len_str = size_t(iter->size());
            memcpy(data_o, iter->data(), len_str);
            data_o += len_str;
            pos += len_str;
        }
        out_col->set_null_bit<bodo_array_type::STRING>(i_grp, bit);
        iter++;
    }
    offsets_o[len] = pos;
    return out_col;
}

std::unique_ptr<array_info> create_list_string_array(
    bodo::vector<uint8_t> const& null_bitmap,
    bodo::vector<bodo::vector<std::pair<std::string, bool>>> const&
        list_list_pair) {
    size_t len = list_list_pair.size();
    // Determining the number of characters in output.
    size_t nb_string = 0;
    size_t nb_char = 0;
    bodo::vector<bodo::vector<std::pair<std::string, bool>>>::const_iterator
        iter = list_list_pair.begin();
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        if (GetBit(null_bitmap.data(), i_grp)) {
            bodo::vector<std::pair<std::string, bool>> e_list = *iter;
            nb_string += e_list.size();
            for (auto& e_str : e_list) {
                nb_char += e_str.first.size();
            }
        }
        iter++;
    }
    // Allocation needs to be done through alloc_array_item, which allocates
    // with meminfos and same data structs that Python uses. We need to
    // re-allocate here because number of strings and chars has been determined
    // here (previous out_col was just an empty dummy allocation).

    std::unique_ptr<array_info> new_out_col = alloc_array_item(
        len,
        alloc_string_array(Bodo_CTypes::CTypeEnum::STRING, nb_string, nb_char));

    uint8_t* sub_null_bitmask_o = (uint8_t*)new_out_col->child_arrays[0]
                                      ->null_bitmask<bodo_array_type::STRING>();
    char* data_o =
        new_out_col->child_arrays[0]->data1<bodo_array_type::STRING>();
    offset_t* data_offsets_o = (offset_t*)new_out_col->child_arrays[0]
                                   ->data2<bodo_array_type::STRING>();
    offset_t* index_offsets_o =
        (offset_t*)new_out_col->data1<bodo_array_type::ARRAY_ITEM>();

    // Writing the list_strings in output
    data_offsets_o[0] = 0;
    offset_t pos_index = 0;
    offset_t pos_data = 0;
    iter = list_list_pair.begin();
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        bool bit = GetBit(null_bitmap.data(), i_grp);
        new_out_col->set_null_bit<bodo_array_type::ARRAY_ITEM>(i_grp, bit);
        index_offsets_o[i_grp] = pos_index;
        if (bit) {
            bodo::vector<std::pair<std::string, bool>> e_list = *iter;
            offset_t n_string = e_list.size();
            for (offset_t i_str = 0; i_str < n_string; i_str++) {
                std::string& estr = e_list[i_str].first;
                offset_t n_char = estr.size();
                memcpy(data_o, estr.data(), n_char);
                data_o += n_char;
                pos_data++;
                data_offsets_o[pos_data] =
                    data_offsets_o[pos_data - 1] + n_char;
                bool bit = e_list[i_str].second;
                SetBitTo(sub_null_bitmask_o, pos_index + i_str, bit);
            }
            pos_index += n_string;
        }
        iter++;
    }
    index_offsets_o[len] = pos_index;
    return new_out_col;
}

std::unique_ptr<array_info> create_dict_string_array(
    std::shared_ptr<array_info> dict_arr,
    std::shared_ptr<array_info> indices_arr) {
    std::unique_ptr<array_info> out_col = std::make_unique<array_info>(
        bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING,
        indices_arr->length, std::vector<std::shared_ptr<BodoBuffer>>({}),
        std::vector<std::shared_ptr<array_info>>({dict_arr, indices_arr}));
    return out_col;
}

/**
 * Allocates memory for string allocation as a NRT_MemInfo
 */
NRT_MemInfo* alloc_meminfo(int64_t length) {
    return NRT_MemInfo_alloc_safe(length);
}

/**
 * @brief allocate a numpy array payload
 *
 * @param length number of elements
 * @param typ_enum dtype of elements
 * @return numpy_arr_payload
 */
numpy_arr_payload allocate_numpy_payload(int64_t length,
                                         Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t itemsize = numpy_item_size[typ_enum];
    int64_t size = length * itemsize;
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    char* data = (char*)meminfo->data;
    return make_numpy_array_payload(meminfo, NULL, length, itemsize, data,
                                    length, itemsize);
}

/**
 * @brief decref numpy array stored in payload and free if refcount becomes
 * zero.
 *
 * @param arr
 */
void decref_numpy_payload(numpy_arr_payload arr) {
    if (arr.meminfo->refct != -1) {
        arr.meminfo->refct--;
    }
    if (arr.meminfo->refct == 0) {
        NRT_MemInfo_call_dtor(arr.meminfo);
    }
}

std::unique_ptr<array_info> alloc_array_top_level(
    int64_t length, int64_t n_sub_elems, int64_t n_sub_sub_elems,
    bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
    int64_t array_id, int64_t extra_null_bytes, int64_t num_categories,
    bool is_globally_replicated, bool is_locally_unique, bool is_locally_sorted,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    switch (arr_type) {
        case bodo_array_type::STRING:
            return alloc_string_array(dtype, length, n_sub_elems, array_id,
                                      extra_null_bytes, is_globally_replicated,
                                      is_locally_unique, is_locally_sorted,
                                      pool, std::move(mm));

        case bodo_array_type::NULLABLE_INT_BOOL:
            return alloc_nullable_array(length, dtype, extra_null_bytes, pool,
                                        std::move(mm));

        case bodo_array_type::INTERVAL:
            return alloc_interval_array(length, dtype, pool, std::move(mm));

        case bodo_array_type::NUMPY:
            return alloc_numpy(length, dtype, pool, std::move(mm));

        case bodo_array_type::CATEGORICAL:
            return alloc_categorical(length, dtype, num_categories, pool,
                                     std::move(mm));

        case bodo_array_type::DICT:
            return alloc_dict_string_array(length, n_sub_elems, n_sub_sub_elems,
                                           extra_null_bytes, pool,
                                           std::move(mm));
        case bodo_array_type::TIMESTAMPTZ:
            return alloc_timestamptz_array(length, extra_null_bytes, pool,
                                           std::move(mm));
        case bodo_array_type::ARRAY_ITEM:
            return alloc_array_item(length, nullptr, extra_null_bytes, pool,
                                    std::move(mm));
        case bodo_array_type::STRUCT:
            return alloc_struct(length, {}, extra_null_bytes, pool,
                                std::move(mm));
        case bodo_array_type::MAP: {
            std::unique_ptr<array_info> inner_array = alloc_array_item(
                length, nullptr, extra_null_bytes, pool, std::move(mm));
            return alloc_map(length, std::move(inner_array));
        }
        default:
            throw std::runtime_error("alloc_array: array type (" +
                                     GetArrType_as_string(arr_type) +
                                     ") not supported");
    }
}

std::unique_ptr<array_info> alloc_array_like(std::shared_ptr<array_info> in_arr,
                                             bool reuse_dictionaries) {
    bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
    Bodo_CTypes::CTypeEnum dtype = in_arr->dtype;
    if (arr_type == bodo_array_type::ARRAY_ITEM) {
        return alloc_array_item(0,
                                alloc_array_like(in_arr->child_arrays.front()));
    } else if (arr_type == bodo_array_type::STRUCT) {
        std::vector<std::shared_ptr<array_info>> child_arrays;
        child_arrays.reserve(in_arr->child_arrays.size());
        for (std::shared_ptr<array_info> child_array : in_arr->child_arrays) {
            child_arrays.push_back(alloc_array_like(child_array));
        }
        return alloc_struct(0, std::move(child_arrays));
    } else if (arr_type == bodo_array_type::MAP) {
        std::unique_ptr<array_info> array_item_arr = alloc_array_item(
            0, alloc_array_like(
                   in_arr->child_arrays.front()->child_arrays.front()));
        return std::make_unique<array_info>(
            bodo_array_type::MAP, Bodo_CTypes::MAP, 0,
            std::vector<std::shared_ptr<BodoBuffer>>({}),
            std::vector<std::shared_ptr<array_info>>(
                {std::move(array_item_arr)}));
    } else {
        std::unique_ptr<array_info> out_arr =
            alloc_array_top_level(0, 0, 0, arr_type, dtype);
        // For dict encoded columns, re-use the same dictionary if
        // reuse_dictionaries = true
        if (reuse_dictionaries && (arr_type == bodo_array_type::DICT)) {
            out_arr->child_arrays.front() = in_arr->child_arrays.front();
        }
        return out_arr;
    }
}

int64_t arrow_array_memory_size(std::shared_ptr<arrow::Array> arr) {
    int64_t n_rows = arr->length();
    int64_t n_bytes = (n_rows + 7) >> 3;
#if OFFSET_BITWIDTH == 32
    if (arr->type_id() == arrow::Type::LIST) {
        std::shared_ptr<arrow::ListArray> list_arr =
            std::dynamic_pointer_cast<arrow::ListArray>(arr);
#else
    if (arr->type_id() == arrow::Type::LARGE_LIST) {
        std::shared_ptr<arrow::LargeListArray> list_arr =
            std::dynamic_pointer_cast<arrow::LargeListArray>(arr);
#endif
        int64_t siz_offset = sizeof(offset_t) * (n_rows + 1);
        int64_t siz_null_bitmap = n_bytes;
        std::shared_ptr<arrow::Array> arr_values = list_arr->values();
        return siz_offset + siz_null_bitmap +
               arrow_array_memory_size(arr_values);
    }
    if (arr->type_id() == arrow::Type::MAP) {
        std::shared_ptr<arrow::MapArray> map_arr =
            std::dynamic_pointer_cast<arrow::MapArray>(arr);
        int64_t siz_offset = sizeof(uint32_t) * (n_rows + 1);
        int64_t siz_null_bitmap = n_bytes;
        int64_t total_siz = siz_offset + siz_null_bitmap;
        total_siz += arrow_array_memory_size(map_arr->keys());
        total_siz += arrow_array_memory_size(map_arr->items());
        return total_siz;
    }
    if (arr->type_id() == arrow::Type::STRUCT) {
        std::shared_ptr<arrow::StructArray> struct_arr =
            std::dynamic_pointer_cast<arrow::StructArray>(arr);
        auto struct_type =
            std::dynamic_pointer_cast<arrow::StructType>(struct_arr->type());
        int64_t num_fields = struct_type->num_fields();
        int64_t total_siz = n_bytes;
        for (int64_t i_field = 0; i_field < num_fields; i_field++)
            total_siz += arrow_array_memory_size(struct_arr->field(i_field));
        return total_siz;
    }
#if OFFSET_BITWIDTH == 32
    if (arr->type_id() == arrow::Type::STRING) {
        std::shared_ptr<arrow::StringArray> string_array =
            std::dynamic_pointer_cast<arrow::StringArray>(arr);
#else
    if (arr->type_id() == arrow::Type::LARGE_STRING) {
        std::shared_ptr<arrow::LargeStringArray> string_array =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(arr);
#endif
        int64_t siz_offset = sizeof(offset_t) * (n_rows + 1);
        int64_t siz_null_bitmap = n_bytes;
        int64_t siz_character = string_array->value_offset(n_rows);
        return siz_offset + siz_null_bitmap + siz_character;
    } else {
        int64_t siz_null_bitmap = n_bytes;
        std::shared_ptr<arrow::PrimitiveArray> prim_arr =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(arr);
        std::shared_ptr<arrow::DataType> arrow_type = prim_arr->type();
        int64_t siz_primitive_data;
        if (arrow_type->id() == arrow::Type::BOOL) {
            // Arrow boolean arrays store 1 bit per boolean
            siz_primitive_data = n_bytes;
        } else {
            Bodo_CTypes::CTypeEnum bodo_typ =
                arrow_to_bodo_type(prim_arr->type()->id());
            int64_t siz_typ = numpy_item_size[bodo_typ];
            siz_primitive_data = siz_typ * n_rows;
        }
        return siz_null_bitmap + siz_primitive_data;
    }
}

int64_t array_memory_size(std::shared_ptr<array_info> earr,
                          bool include_dict_size) {
    if (earr->arr_type == bodo_array_type::NUMPY ||
        earr->arr_type == bodo_array_type::CATEGORICAL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        return siztype * earr->length;
    } else if (earr->arr_type == bodo_array_type::DICT) {
        // Not all functions want to consider the size of the dictionary.
        int64_t dict_size =
            include_dict_size
                ? array_memory_size(earr->child_arrays[0], include_dict_size)
                : 0;
        return dict_size +
               array_memory_size(earr->child_arrays[1], include_dict_size);
    } else if (earr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        int64_t n_bytes = ((earr->length + 7) >> 3);
        if (earr->dtype == Bodo_CTypes::_BOOL) {
            // Nullable boolean arrays store 1 bit per boolean.
            return n_bytes * 2;
        } else {
            uint64_t siztype = numpy_item_size[earr->dtype];
            return n_bytes + siztype * earr->length;
        }
    } else if (earr->arr_type == bodo_array_type::TIMESTAMPTZ) {
        int64_t timestamp_bytes =
            earr->length * numpy_item_size[Bodo_CTypes::INT64];
        int64_t offset_bytes =
            earr->length * numpy_item_size[Bodo_CTypes::INT16];
        int64_t null_bytes = arrow::bit_util::BytesForBits(earr->length);
        return timestamp_bytes + offset_bytes + null_bytes;
    } else if (earr->arr_type == bodo_array_type::STRING) {
        int64_t n_bytes = ((earr->length + 7) >> 3);
        return earr->n_sub_elems() + sizeof(offset_t) * (earr->length + 1) +
               n_bytes;
    } else if (earr->arr_type == bodo_array_type::ARRAY_ITEM) {
        int64_t n_bytes = ((earr->length + 7) >> 3);
        return n_bytes + sizeof(offset_t) * (earr->length + 1) +
               array_memory_size(earr->child_arrays.front(), include_dict_size);
    } else if (earr->arr_type == bodo_array_type::STRUCT) {
        int64_t n_bytes = ((earr->length + 7) >> 3), child_array_size = 0;
        for (const std::shared_ptr<array_info>& child_array :
             earr->child_arrays) {
            child_array_size +=
                array_memory_size(child_array, include_dict_size);
        }
        return n_bytes + child_array_size;
    } else if (earr->arr_type == bodo_array_type::MAP) {
        return array_memory_size(earr->child_arrays.front(), include_dict_size);
    }
    throw std::runtime_error(
        "Array Type: " + GetArrType_as_string(earr->arr_type) +
        " not covered in array_memory_size()");
}

int64_t table_local_memory_size(const std::shared_ptr<table_info>& table,
                                bool include_dict_size) {
    int64_t local_size = 0;
    for (auto& arr : table->columns) {
        local_size += array_memory_size(arr, include_dict_size);
    }
    return local_size;
}

int64_t table_global_memory_size(const std::shared_ptr<table_info>& table) {
    int64_t local_size = table_local_memory_size(table, false);
    int64_t global_size;
    MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG_LONG_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_size;
}

std::shared_ptr<array_info> copy_array(std::shared_ptr<array_info> earr,
                                       bool shallow_copy_child_arrays) {
    std::shared_ptr<array_info> farr;
    // Copy child arrays
    if (earr->arr_type == bodo_array_type::DICT) {
        std::shared_ptr<array_info> dictionary =
            copy_array(earr->child_arrays[0]);
        std::shared_ptr<array_info> indices = copy_array(earr->child_arrays[1]);
        farr = create_dict_string_array(dictionary, indices);
    } else if (earr->arr_type == bodo_array_type::ARRAY_ITEM) {
        farr = alloc_array_item(earr->length,
                                shallow_copy_child_arrays
                                    ? earr->child_arrays.front()
                                    : copy_array(earr->child_arrays.front()));
    } else if (earr->arr_type == bodo_array_type::STRUCT) {
        std::vector<std::shared_ptr<array_info>> child_arrays(
            earr->child_arrays);
        if (!shallow_copy_child_arrays) {
            for (size_t i = 0; i < child_arrays.size(); ++i) {
                child_arrays[i] = copy_array(earr->child_arrays[i]);
            }
        }
        farr = alloc_struct(earr->length, child_arrays);
    } else if (earr->arr_type == bodo_array_type::MAP) {
        std::shared_ptr<array_info> arr_item_arr =
            shallow_copy_child_arrays ? earr->child_arrays.front()
                                      : copy_array(earr->child_arrays.front());
        farr = alloc_map(arr_item_arr->length, arr_item_arr);
    } else {
        farr = alloc_array_top_level(
            earr->length, earr->n_sub_elems(), 0, earr->arr_type, earr->dtype,
            earr->arr_type == bodo_array_type::STRING ? earr->array_id : -1, 0,
            earr->num_categories, earr->is_globally_replicated,
            earr->is_locally_unique, earr->is_locally_sorted);
    }
    // Copy buffers
    if (earr->arr_type == bodo_array_type::NUMPY ||
        earr->arr_type == bodo_array_type::CATEGORICAL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        memcpy(farr->data1(), earr->data1(), siztype * earr->length);
    } else if (earr->arr_type == bodo_array_type::INTERVAL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        memcpy(farr->data1<bodo_array_type::INTERVAL>(),
               earr->data1<bodo_array_type::INTERVAL>(),
               siztype * earr->length);
        memcpy(farr->data2<bodo_array_type::INTERVAL>(),
               earr->data2<bodo_array_type::INTERVAL>(),
               siztype * earr->length);
    } else if (earr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        int64_t data_copy_size;
        int64_t n_bytes = ((earr->length + 7) >> 3);
        if (earr->dtype == Bodo_CTypes::_BOOL) {
            data_copy_size = n_bytes;
        } else {
            data_copy_size = earr->length * numpy_item_size[earr->dtype];
        }
        memcpy(farr->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
               earr->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
               data_copy_size);
        memcpy(farr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
               earr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
               n_bytes);
    } else if (earr->arr_type == bodo_array_type::TIMESTAMPTZ) {
        int64_t timestamp_bytes =
            earr->length * numpy_item_size[Bodo_CTypes::TIMESTAMPTZ];
        int64_t offset_bytes =
            earr->length * numpy_item_size[Bodo_CTypes::INT16];
        int64_t n_bytes = ((earr->length + 7) >> 3);
        memcpy(farr->data1<bodo_array_type::TIMESTAMPTZ>(),
               earr->data1<bodo_array_type::TIMESTAMPTZ>(), timestamp_bytes);
        memcpy(farr->data2<bodo_array_type::TIMESTAMPTZ>(),
               earr->data2<bodo_array_type::TIMESTAMPTZ>(), offset_bytes);
        memcpy(farr->null_bitmask<bodo_array_type::TIMESTAMPTZ>(),
               earr->null_bitmask<bodo_array_type::TIMESTAMPTZ>(), n_bytes);
    } else if (earr->arr_type == bodo_array_type::STRING) {
        memcpy(farr->data1<bodo_array_type::STRING>(),
               earr->data1<bodo_array_type::STRING>(), earr->n_sub_elems());
        memcpy(farr->data2<bodo_array_type::STRING>(),
               earr->data2<bodo_array_type::STRING>(),
               sizeof(offset_t) * (earr->length + 1));
        int64_t n_bytes = ((earr->length + 7) >> 3);
        memcpy(farr->null_bitmask<bodo_array_type::STRING>(),
               earr->null_bitmask<bodo_array_type::STRING>(), n_bytes);
    } else if (earr->arr_type == bodo_array_type::ARRAY_ITEM) {
        memcpy(farr->data1<bodo_array_type::ARRAY_ITEM>(),
               earr->data1<bodo_array_type::ARRAY_ITEM>(),
               sizeof(offset_t) * (earr->length + 1));
        int64_t n_bytes = ((earr->length + 7) >> 3);
        memcpy(farr->null_bitmask<bodo_array_type::ARRAY_ITEM>(),
               earr->null_bitmask<bodo_array_type::ARRAY_ITEM>(), n_bytes);
    } else if (earr->arr_type == bodo_array_type::MAP) {
        // Map array has no buffers to copy
    } else if (earr->arr_type == bodo_array_type::STRUCT) {
        int64_t n_bytes = ((earr->length + 7) >> 3);
        memcpy(farr->null_bitmask<bodo_array_type::STRUCT>(),
               earr->null_bitmask<bodo_array_type::STRUCT>(), n_bytes);
    } else {
        throw std::runtime_error(
            "copy_array: " + GetArrType_as_string(earr->arr_type) +
            " array not supported yet!");
    }
    return farr;
}

size_t get_expected_bits_per_entry(bodo_array_type::arr_type_enum arr_type,
                                   Bodo_CTypes::CTypeEnum c_type) {
    // TODO: Handle nested data structure and categorical data types seperately
    size_t nullable;
    switch (arr_type) {
        case bodo_array_type::DICT:
            return 33;  // Dictionaries are fixed 32 bits + 1 null bit
        case bodo_array_type::NUMPY:
        case bodo_array_type::INTERVAL:
        case bodo_array_type::CATEGORICAL:
            nullable = 0;
            break;
        case bodo_array_type::STRING:
        case bodo_array_type::NULLABLE_INT_BOOL:
        case bodo_array_type::TIMESTAMPTZ:
        case bodo_array_type::STRUCT:
        case bodo_array_type::ARRAY_ITEM:
        case bodo_array_type::MAP:
            nullable = 1;
            break;
        default:
            throw std::runtime_error(
                "get_expected_bits_per_entry: Invalid array type!");
    }
    switch (c_type) {
        case Bodo_CTypes::_BOOL:
            return nullable + (arr_type == bodo_array_type::NUMPY ? 8 : 1);
        case Bodo_CTypes::INT8:
        case Bodo_CTypes::UINT8:
        case Bodo_CTypes::INT16:
        case Bodo_CTypes::UINT16:
        case Bodo_CTypes::INT32:
        case Bodo_CTypes::UINT32:
        case Bodo_CTypes::INT64:
        case Bodo_CTypes::UINT64:
        case Bodo_CTypes::INT128:
        case Bodo_CTypes::FLOAT32:
        case Bodo_CTypes::FLOAT64:
        case Bodo_CTypes::DECIMAL:
        case Bodo_CTypes::DATE:
        case Bodo_CTypes::TIME:
        case Bodo_CTypes::DATETIME:
        case Bodo_CTypes::TIMEDELTA:
        case Bodo_CTypes::TIMESTAMPTZ:
            return (nullable + numpy_item_size[c_type]) << 3;
        case Bodo_CTypes::STRING:
        case Bodo_CTypes::LIST:
        case Bodo_CTypes::MAP:
        case Bodo_CTypes::STRUCT:
        case Bodo_CTypes::BINARY:
            return nullable +
                   256;  // 32 bytes estimate for unknown or variable size types
        default:
            throw std::runtime_error(
                "get_expected_bits_per_entry: Invalid C type!");
    }
}

size_t get_row_bytes(const std::shared_ptr<bodo::Schema>& schema) {
    size_t row_bits = 0;
    for (const std::unique_ptr<bodo::DataType>& data_type :
         schema->column_types) {
        row_bits += get_expected_bits_per_entry(data_type->array_type,
                                                data_type->c_type);
    }
    return (row_bits + 7) >> 3;
}

/**
 * Free underlying array of array_info pointer and delete the pointer.
 * Called from Python.
 */
void delete_info(array_info* arr) { delete arr; }

/**
 * Delete table pointer and its column array_info pointers (but not the arrays).
 * Called from Python.
 */
void delete_table(table_info* table) { delete table; }

table_info* cpp_table_map_to_list(table_info* table) {
    std::vector<std::shared_ptr<array_info>> new_columns;
    for (std::shared_ptr<array_info>& v : table->columns) {
        // Get underlying list(struct) array for map array
        if (v->arr_type == bodo_array_type::MAP) {
            new_columns.emplace_back(v->child_arrays[0]);
        } else {
            new_columns.emplace_back(v);
        }
    }
    table_info* out_table = new table_info(new_columns);
    delete table;
    return out_table;
}

void decref_meminfo(MemInfo* meminfo) {
    if (meminfo != NULL && meminfo->refct != -1) {
        meminfo->refct--;
        if (meminfo->refct == 0) {
            NRT_MemInfo_call_dtor(meminfo);
        }
    }
}

void incref_meminfo(MemInfo* meminfo) {
    if (meminfo != NULL && meminfo->refct != -1) {
        meminfo->refct++;
    }
}

void reset_col_if_last_table_ref(std::shared_ptr<table_info> const& table,
                                 size_t col_idx) {
    if (table.use_count() == 1) {
        table->columns[col_idx].reset();
    }
}

void clear_all_cols_if_last_table_ref(
    std::shared_ptr<table_info> const& table) {
    if (table.use_count() == 1) {
        table->columns.clear();
    }
}

std::tuple<std::vector<int8_t>, std::vector<int8_t>>
get_dtypes_arr_types_from_table(const std::shared_ptr<table_info>& table) {
    std::vector<int8_t> arr_c_types;
    std::vector<int8_t> arr_array_types;

    for (size_t i = 0; i < table->columns.size(); i++) {
        _get_dtypes_arr_types_from_array(table->columns[i], arr_c_types,
                                         arr_array_types);
    }
    return std::make_tuple(arr_c_types, arr_array_types);
}

void _get_dtypes_arr_types_from_array(const std::shared_ptr<array_info>& array,
                                      std::vector<int8_t>& arr_c_types,
                                      std::vector<int8_t>& arr_array_types) {
    arr_c_types.push_back((int8_t)array->dtype);
    arr_array_types.push_back((int8_t)array->arr_type);

    if (array->arr_type == bodo_array_type::STRUCT) {
        arr_c_types.push_back(array->child_arrays.size());
        arr_array_types.push_back(array->child_arrays.size());
        for (auto child : array->child_arrays) {
            _get_dtypes_arr_types_from_array(child, arr_c_types,
                                             arr_array_types);
        }
    }
    if (array->arr_type == bodo_array_type::ARRAY_ITEM) {
        _get_dtypes_arr_types_from_array(array->child_arrays[0], arr_c_types,
                                         arr_array_types);
    }
    if (array->arr_type == bodo_array_type::MAP) {
        // Handle key and value arrays
        _get_dtypes_arr_types_from_array(
            array->child_arrays[0]->child_arrays[0]->child_arrays[0],
            arr_c_types, arr_array_types);
        _get_dtypes_arr_types_from_array(
            array->child_arrays[0]->child_arrays[0]->child_arrays[1],
            arr_c_types, arr_array_types);
    }
}

std::tuple<std::vector<int8_t>, std::vector<int8_t>>
get_dtypes_arr_types_from_array(const std::shared_ptr<array_info>& array) {
    std::vector<int8_t> arr_c_types, arr_array_types;
    _get_dtypes_arr_types_from_array(array, arr_c_types, arr_array_types);
    return {arr_c_types, arr_array_types};
}

/**
 * @brief Get start index of next column e.g. given the following inputs:
 * arr_array_types = [ARRAY_ITEM, NULLABLE_INT_BOOL, ARRAY_ITEM, ARRAY_ITEM,
 * NULLABLE_INT_BOOL, NULLABLE_INT_BOOL] idx = 2 The function will output 5
 *
 * @param arr_array_types The span of array types
 * @param idx The index of the current column
 * @return The index of next column
 */
size_t get_next_col_idx(const std::span<const int8_t>& arr_array_types,
                        size_t idx) {
    if (idx >= arr_array_types.size()) {
        throw std::runtime_error("get_next_col_idx: index " +
                                 std::to_string(idx) + " out of bound!");
    }
    if (arr_array_types[idx] == bodo_array_type::ARRAY_ITEM) {
        do {
            ++idx;
        } while (idx < arr_array_types.size() &&
                 arr_array_types[idx] == bodo_array_type::ARRAY_ITEM);
        if (idx >= arr_array_types.size()) {
            throw std::runtime_error(
                "The last array type cannot be ARRAY_ITEM: inner array type "
                "needs to be provided!");
        }
        return get_next_col_idx(arr_array_types, idx);
    } else if (arr_array_types[idx] == bodo_array_type::STRUCT) {
        int8_t tot = arr_array_types[idx + 1];
        idx += 2;
        while (tot--) {
            idx = get_next_col_idx(arr_array_types, idx);
        }
        return idx;
    } else if (arr_array_types[idx] == bodo_array_type::MAP) {
        idx++;
        idx = get_next_col_idx(arr_array_types, idx);
        idx = get_next_col_idx(arr_array_types, idx);
        return idx;
    } else {
        return idx + 1;
    }
}

std::vector<size_t> get_col_idx_map(
    const std::span<const int8_t>& arr_array_types) {
    std::vector<size_t> col_idx_map;
    for (size_t i = 0; i < arr_array_types.size();
         i = get_next_col_idx(arr_array_types, i)) {
        col_idx_map.push_back(i);
    }
    return col_idx_map;
}

// get memory alloc/free info from _meminfo.h
size_t get_stats_alloc() { return NRT_MemSys_get_stats_alloc(); }
size_t get_stats_free() { return NRT_MemSys_get_stats_free(); }
size_t get_stats_mi_alloc() { return NRT_MemSys_get_stats_mi_alloc(); }
size_t get_stats_mi_free() { return NRT_MemSys_get_stats_mi_free(); }

// Dictionary utilities

/**
 * @brief Generate a new local id for a dictionary. These
 * can be used to identify if dictionaries are "equivalent"
 * because they share an id. Other than ==, a particular
 * id has no significance.
 *
 * @param length The length of the dictionary being assigned
 * the id. All dictionaries of length 0 should get the same
 * id.
 * @return int64_t The new id that is generated.
 */
static int64_t generate_array_id_state(int64_t length) {
    static int64_t id_counter = 1;
    if (length == 0) {
        // Ensure we can identify all length 0 dictionaries
        // and that all can be unified without transposing.
        return 0;
    } else {
        return id_counter++;
    }
}

int64_t generate_array_id(int64_t length) {
    return generate_array_id_state(length);
}

extern "C" {

PyMODINIT_FUNC PyInit_ext(void) {
    PyObject* m;
    MOD_DEF(m, "ext", "No docs", NULL);
    if (m == NULL)
        return NULL;

    bodo_common_init();

    SetAttrStringFromPyInit(m, hdist);
    SetAttrStringFromPyInit(m, hstr_ext);
    SetAttrStringFromPyInit(m, decimal_ext);
    SetAttrStringFromPyInit(m, quantile_alg);
    SetAttrStringFromPyInit(m, lateral);
    SetAttrStringFromPyInit(m, lead_lag);
    SetAttrStringFromPyInit(m, crypto_funcs);
    SetAttrStringFromPyInit(m, hdatetime_ext);
    SetAttrStringFromPyInit(m, hio);
    SetAttrStringFromPyInit(m, array_ext);
    SetAttrStringFromPyInit(m, s3_reader);
    SetAttrStringFromPyInit(m, fsspec_reader);
    SetAttrStringFromPyInit(m, hdfs_reader);

    SetAttrStringFromPyInit(m, _hdf5);
    SetAttrStringFromPyInit(m, arrow_cpp);
    SetAttrStringFromPyInit(m, csv_cpp);
    SetAttrStringFromPyInit(m, json_cpp);
    SetAttrStringFromPyInit(m, memory_budget_cpp);
    SetAttrStringFromPyInit(m, stream_join_cpp);
    SetAttrStringFromPyInit(m, stream_groupby_cpp);
    SetAttrStringFromPyInit(m, stream_dict_encoding_cpp);
    SetAttrStringFromPyInit(m, table_builder_cpp);
    SetAttrStringFromPyInit(m, fft_cpp);
    SetAttrStringFromPyInit(m, query_profile_collector_cpp);
    SetAttrStringFromPyInit(m, uuid_cpp);
#ifdef BUILD_WITH_V8
    SetAttrStringFromPyInit(m, javascript_udf_cpp);
#endif

#ifdef IS_TESTING
    SetAttrStringFromPyInit(m, test_cpp);
#endif

    SetAttrStringFromPyInit(m, listagg);

    return m;
}

} /* extern "C" */
