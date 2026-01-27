#include "_bodo_common.h"

#include <complex>
#include <memory>
#include <stdexcept>
#include <string>

#include <arrow/array.h>
#include <arrow/type.h>
#include <fmt/format.h>
#include "_distributed.h"
#include "_memory.h"
#include "arrow/util/key_value_metadata.h"

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

    // Get the default buffer pool pointer from Python and set the global
    // pointer
    PyObject* memory_module = PyImport_ImportModule("bodo.memory_cpp");
    if (memory_module == nullptr) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Failed to import bodo.memory_cpp module!");
        return;
    }

    PyObject* pool_ptr_obj =
        PyObject_CallMethod(memory_module, "default_buffer_pool_ptr", nullptr);
    if (pool_ptr_obj == nullptr) {
        Py_DECREF(memory_module);
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Failed to call default_buffer_pool_ptr()!");
        return;
    }

    int64_t memory_pool_ptr = PyLong_AsLongLong(pool_ptr_obj);
    if (memory_pool_ptr == -1 && PyErr_Occurred()) {
        Py_DECREF(pool_ptr_obj);
        Py_DECREF(memory_module);
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Failed to convert pool pointer to integer!");
        return;
    }

    bodo::init_buffer_pool_ptr(memory_pool_ptr);

    Py_DECREF(pool_ptr_obj);

    // Get the default memsys pointer from Python and set the global
    // pointer
    PyObject* memsys_ptr_obj =
        PyObject_CallMethod(memory_module, "default_memsys_ptr", nullptr);
    if (memsys_ptr_obj == nullptr) {
        Py_DECREF(memory_module);
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Failed to call default_memsys_ptr()!");
        return;
    }

    int64_t memory_memsys_ptr = PyLong_AsLongLong(memsys_ptr_obj);
    if (memory_memsys_ptr == -1 && PyErr_Occurred()) {
        Py_DECREF(pool_ptr_obj);
        Py_DECREF(memory_module);
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Failed to convert memsys pointer to integer!");
        return;
    }

    global_memsys = reinterpret_cast<MemSys*>(memory_memsys_ptr);

    Py_DECREF(memsys_ptr_obj);
    Py_DECREF(memory_module);

    if (numpy_item_size.size() != Bodo_CTypes::_numtypes) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Incorrect number of bodo item sizes!");
        return;
    }

    PyObject* np_mod = PyImport_ImportModule("numpy");
    PyObject* dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "bool");
    if ((size_t)PyNumber_AsSsize_t(
            PyObject_GetAttrString(dtype_obj, "itemsize"), nullptr) !=
        sizeof(bool)) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "bool size mismatch between C++ and NumPy!");
        return;
    }
    dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "float32");
    if ((size_t)PyNumber_AsSsize_t(
            PyObject_GetAttrString(dtype_obj, "itemsize"), nullptr) !=
        sizeof(float)) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "float32 size mismatch between C++ and NumPy!");
        return;
    }
    dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "float64");
    if ((size_t)PyNumber_AsSsize_t(
            PyObject_GetAttrString(dtype_obj, "itemsize"), nullptr) !=
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
        case arrow::Type::DURATION:
            return Bodo_CTypes::TIMEDELTA;
        default: {
            // TODO: Construct the type from the id
            throw std::runtime_error("arrow_to_bodo_type");
        }
    }
}

std::unique_ptr<bodo::DataType> arrow_type_to_bodo_data_type(
    const std::shared_ptr<arrow::DataType> arrow_type) {
    switch (arrow_type->id()) {
        // String array
        case arrow::Type::LARGE_STRING:
        case arrow::Type::STRING: {
            return std::make_unique<bodo::DataType>(bodo_array_type::STRING,
                                                    Bodo_CTypes::STRING);
        }
        // Binary array
        case arrow::Type::LARGE_BINARY:
        case arrow::Type::BINARY: {
            return std::make_unique<bodo::DataType>(bodo_array_type::STRING,
                                                    Bodo_CTypes::BINARY);
        }
        // array(item) array
        case arrow::Type::LARGE_LIST:
        case arrow::Type::LIST: {
            assert(arrow_type->num_fields() == 1);
            std::unique_ptr<bodo::DataType> inner =
                arrow_type_to_bodo_data_type(arrow_type->field(0)->type());
            return std::make_unique<bodo::ArrayType>(std::move(inner));
        }
        // map array
        case arrow::Type::MAP: {
            std::shared_ptr<arrow::MapType> map_type =
                std::static_pointer_cast<arrow::MapType>(arrow_type);
            std::unique_ptr<bodo::DataType> key_type =
                arrow_type_to_bodo_data_type(map_type->key_type());
            std::unique_ptr<bodo::DataType> value_type =
                arrow_type_to_bodo_data_type(map_type->item_type());
            return std::make_unique<bodo::MapType>(std::move(key_type),
                                                   std::move(value_type));
        }
        // struct array
        case arrow::Type::STRUCT: {
            std::vector<std::unique_ptr<bodo::DataType>> field_types;
            for (int i = 0; i < arrow_type->num_fields(); i++) {
                field_types.push_back(
                    arrow_type_to_bodo_data_type(arrow_type->field(i)->type()));
            }
            return std::make_unique<bodo::StructType>(std::move(field_types));
        }
        // all fixed-size nullable types
        case arrow::Type::DOUBLE:
        case arrow::Type::FLOAT:
        case arrow::Type::BOOL:
        case arrow::Type::UINT64:
        case arrow::Type::INT64:
        case arrow::Type::UINT32:
        case arrow::Type::DATE32:
        case arrow::Type::DURATION:
        case arrow::Type::INT32:
        case arrow::Type::UINT16:
        case arrow::Type::INT16:
        case arrow::Type::UINT8:
        case arrow::Type::INT8: {
            return std::make_unique<bodo::DataType>(
                bodo_array_type::NULLABLE_INT_BOOL,
                arrow_to_bodo_type(arrow_type->id()));
        }
        case arrow::Type::TIMESTAMP: {
            auto arrow_timestamp_type =
                std::static_pointer_cast<arrow::TimestampType>(arrow_type);
            return std::make_unique<bodo::DataType>(
                bodo_array_type::NULLABLE_INT_BOOL,
                arrow_to_bodo_type(arrow_type->id()), -1, -1,
                arrow_timestamp_type->timezone());
        }

        case arrow::Type::TIME32:
        case arrow::Type::TIME64: {
            std::shared_ptr<arrow::TimeType> time_type =
                std::static_pointer_cast<arrow::TimeType>(arrow_type);
            int8_t precision;
            switch (time_type->unit()) {
                case arrow::TimeUnit::SECOND:
                    precision = 0;
                    break;
                case arrow::TimeUnit::MILLI:
                    precision = 3;
                    break;
                case arrow::TimeUnit::MICRO:
                    precision = 6;
                    break;
                case arrow::TimeUnit::NANO:
                    precision = 9;
                    break;
                default:
                    throw std::runtime_error(
                        "Unsupported time unit passed to "
                        "arrow_type_to_bodo_data_type: " +
                        time_type->ToString());
            }
            return std::make_unique<bodo::DataType>(
                bodo_array_type::NULLABLE_INT_BOOL,
                arrow_to_bodo_type(arrow_type->id()), precision);
        }

        // decimal array
        case arrow::Type::DECIMAL128: {
            auto arrow_decimal_type =
                std::static_pointer_cast<arrow::Decimal128Type>(arrow_type);
            return std::make_unique<bodo::DataType>(
                bodo_array_type::NULLABLE_INT_BOOL,
                arrow_to_bodo_type(arrow_type->id()),
                arrow_decimal_type->precision(), arrow_decimal_type->scale());
        }
        // dictionary-encoded array
        case arrow::Type::DICTIONARY: {
            return std::make_unique<bodo::DataType>(bodo_array_type::DICT,
                                                    Bodo_CTypes::STRING);
        }
        // null array
        case arrow::Type::NA: {
            // null array is currently stored as string array in C++
            return std::make_unique<bodo::DataType>(bodo_array_type::STRING,
                                                    Bodo_CTypes::STRING);
        }
        case arrow::Type::EXTENSION: {
            // Cast the type to an ExtensionArray to access the extension name
            auto ext_type =
                std::static_pointer_cast<arrow::ExtensionType>(arrow_type);
            auto name = ext_type->extension_name();
            if (name == "arrow_timestamp_tz") {
                return std::make_unique<bodo::DataType>(
                    bodo_array_type::TIMESTAMPTZ, Bodo_CTypes::TIMESTAMPTZ);
            }
            [[fallthrough]];
        }
        default:
            throw std::runtime_error(
                "arrow_type_to_bodo_data_type(): Arrow type " +
                arrow_type->ToString() + " not supported");
    }
}

int32_t decimal_precision_to_integer_bytes(int32_t precision) {
    if (precision < 3) {
        return 1;
    }
    if (precision < 5) {
        return 2;
    }
    if (precision < 9) {
        return 4;
    }
    if (precision < 18) {
        return 8;
    }
    return 16;
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
        return std::make_unique<DataType>(this->array_type, this->c_type,
                                          this->precision, this->scale,
                                          this->timezone);
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
        return std::make_unique<DataType>(arr_type, dtype, this->precision,
                                          this->scale, this->timezone);
    }
}

void DataType::to_string_inner(std::string& out) {
    out += arr_type_to_str(this->array_type);
    out += "[";
    out += dtype_to_str(this->c_type);
    // for decimals we want to add the precision and scale as well
    if (this->c_type == Bodo_CTypes::DECIMAL) {
        out += fmt::format("({},{})", this->precision, this->scale);
    }
    if (this->c_type == Bodo_CTypes::DATETIME && this->timezone.length() > 0) {
        out += fmt::format("({})", this->timezone);
    }
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
    // if it is a type decimal also push back the scale and precision
    if (c_type == Bodo_CTypes::DECIMAL) {
        arr_array_types.push_back(precision);
        arr_array_types.push_back(scale);
        arr_c_types.push_back(precision);
        arr_c_types.push_back(scale);
    } else if (array_type == bodo_array_type::NULLABLE_INT_BOOL &&
               c_type == Bodo_CTypes::DATETIME) {
        // For Datetime types, append the length of the timezone
        // string and then the characters as int8_t.
        if (timezone.size() > 255) {
            throw std::runtime_error(
                "String too long for 1-byte length prefix");
        }
        uint8_t len = static_cast<uint8_t>(timezone.size());
        arr_array_types.push_back(static_cast<int8_t>(len));
        arr_array_types.insert(arr_array_types.end(), timezone.begin(),
                               timezone.end());
        arr_c_types.push_back(static_cast<int8_t>(len));
        arr_c_types.insert(arr_c_types.end(), timezone.begin(), timezone.end());
    }
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

std::shared_ptr<::arrow::Field> DataType::ToArrowType(std::string& name) const {
    if (array_type == bodo_array_type::STRING) {
        if (c_type == Bodo_CTypes::STRING) {
            return std::make_shared<::arrow::Field>(name, arrow::large_utf8(),
                                                    true);
        } else {
            assert(c_type == Bodo_CTypes::BINARY);
            return std::make_shared<::arrow::Field>(name, arrow::large_binary(),
                                                    true);
        }

    } else if (array_type == bodo_array_type::DICT) {
        return std::make_shared<::arrow::Field>(
            name, arrow::dictionary(arrow::int32(), arrow::large_utf8()), true);
    } else if (array_type == bodo_array_type::TIMESTAMPTZ) {
        throw std::runtime_error("TIMESTAMPTZ is not supported in Arrow");
    }

    bool is_nullable = array_type == bodo_array_type::NULLABLE_INT_BOOL;
    std::shared_ptr<arrow::DataType> dtype;
    switch (c_type) {
        case Bodo_CTypes::INT8:
            dtype = arrow::int8();
            break;
        case Bodo_CTypes::UINT8:
            dtype = arrow::uint8();
            break;
        case Bodo_CTypes::INT16:
            dtype = arrow::int16();
            break;
        case Bodo_CTypes::UINT16:
            dtype = arrow::uint16();
            break;
        case Bodo_CTypes::INT32:
            dtype = arrow::int32();
            break;
        case Bodo_CTypes::UINT32:
            dtype = arrow::uint32();
            break;
        case Bodo_CTypes::INT64:
            dtype = arrow::int64();
            break;
        case Bodo_CTypes::UINT64:
            dtype = arrow::uint64();
            break;
        case Bodo_CTypes::FLOAT32:
            dtype = arrow::float32();
            break;
        case Bodo_CTypes::FLOAT64:
            dtype = arrow::float64();
            break;
        case Bodo_CTypes::_BOOL:
            dtype = arrow::boolean();
            break;
        case Bodo_CTypes::DATE:
            dtype = arrow::date32();
            break;
        // TODO: check precision
        case Bodo_CTypes::TIME:
            dtype = arrow::time64(arrow::TimeUnit::NANO);
            break;
        case Bodo_CTypes::DATETIME:
            if (timezone.length() > 0) {
                dtype = arrow::timestamp(arrow::TimeUnit::NANO, timezone);
            } else {
                dtype = arrow::timestamp(arrow::TimeUnit::NANO);
            }
            break;
        case Bodo_CTypes::TIMEDELTA:
            dtype = arrow::duration(arrow::TimeUnit::NANO);
            break;
        case Bodo_CTypes::DECIMAL:
            dtype = arrow::decimal128(precision, scale);
            break;
        default: {
            throw std::runtime_error("ToArrowType: unsupported dtype " +
                                     std::string(dtype_to_str(c_type)));
        }
    }

    return std::make_shared<::arrow::Field>(name, dtype, is_nullable);
}

std::shared_ptr<::arrow::DataType> DataType::ToArrowDataType() const {
    std::string dummy = "dummy";
    std::shared_ptr<::arrow::Field> field = ToArrowType(dummy);
    return field->type();
}

std::shared_ptr<::arrow::Field> ArrayType::ToArrowType(
    std::string& name) const {
    std::string element_name = "element";
    return std::make_shared<::arrow::Field>(
        name, arrow::large_list(this->value_type->ToArrowType(element_name)),
        true);
}

std::shared_ptr<::arrow::Field> StructType::ToArrowType(
    std::string& name) const {
    std::vector<std::shared_ptr<::arrow::Field>> fields;
    for (size_t i = 0; i < child_types.size(); i++) {
        std::string field_name = fmt::format("field_{}", i);
        fields.push_back(child_types[i]->ToArrowType(field_name));
    }

    return std::make_shared<::arrow::Field>(name, arrow::struct_(fields), true);
}

std::shared_ptr<::arrow::Field> MapType::ToArrowType(std::string& name) const {
    std::string key_name = "key";
    std::string value_name = "value";

    return std::make_shared<::arrow::Field>(
        name,
        arrow::map(this->key_type->ToArrowType(key_name)->type(),
                   this->value_type->ToArrowType(value_name)),
        true);
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
    } else if (c_type == Bodo_CTypes::DECIMAL) {
        uint8_t precision = arr_c_types[i];
        uint8_t scale = arr_c_types[i + 1];
        i += 2;
        return std::make_unique<DataType>(array_type, c_type, precision, scale);
    } else if (c_type == Bodo_CTypes::DATETIME &&
               array_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint8_t len = static_cast<uint8_t>(arr_array_types[i]);
        i += 1;

        if (len == 0) {
            return std::make_unique<DataType>(array_type, c_type, -1, -1, "");
        }

        std::string timezone(reinterpret_cast<const char*>(&arr_array_types[i]),
                             len);
        i += len;

        return std::make_unique<DataType>(array_type, c_type, -1, -1, timezone);
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

TableMetadata TableMetadata::append(TableMetadata const& other) const {
    std::vector<std::string> new_keys(this->keys);
    std::vector<std::string> new_values(this->values);
    new_keys.insert(new_keys.end(), other.keys.begin(), other.keys.end());
    new_values.insert(new_values.end(), other.values.begin(),
                      other.values.end());
    return TableMetadata{.keys = new_keys, .values = new_values};
}

Schema::Schema() : column_types() {}
Schema::Schema(const Schema& other) {
    this->column_types.reserve(other.column_types.size());
    for (const auto& t : other.column_types) {
        this->column_types.push_back(t->copy());
    }
    this->column_names = other.column_names;
    this->metadata = other.metadata;
}

Schema::Schema(Schema&& other) {
    this->column_types = std::move(other.column_types);
    this->column_names = std::move(other.column_names);
    this->metadata = std::move(other.metadata);
}

Schema::Schema(std::vector<std::unique_ptr<bodo::DataType>>&& column_types_)
    : column_types(std::move(column_types_)) {}

Schema::Schema(std::vector<std::unique_ptr<bodo::DataType>>&& column_types_,
               std::vector<std::string> column_names)
    : column_types(std::move(column_types_)), column_names(column_names) {}

Schema::Schema(std::vector<std::unique_ptr<bodo::DataType>>&& column_types_,
               std::vector<std::string> column_names,
               std::shared_ptr<bodo::TableMetadata> metadata)
    : column_types(std::move(column_types_)),
      column_names(column_names),
      metadata(metadata) {}

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
    for (auto& col_name : other_schema->column_names) {
        this->column_names.push_back(std::move(col_name));
    }
    if (this->metadata) {
        this->metadata = std::make_shared<TableMetadata>(
            this->metadata->append(*other_schema->metadata));
    } else {
        this->metadata = other_schema->metadata;
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

std::string Schema::ToString(bool use_col_names) {
    std::string out;
    for (size_t i = 0; i < this->column_types.size(); i++) {
        if (i > 0) {
            out += "\n";
        }
        out += fmt::format("{}: {}", i, this->column_types[i]->ToString());
        if (use_col_names && i < this->column_names.size()) {
            out += " " + this->column_names[i];
        }
    }
    return out;
}

std::unique_ptr<Schema> Schema::Project(size_t first_n) const {
    std::vector<std::unique_ptr<DataType>> dtypes;
    std::vector<std::string> col_names;
    dtypes.reserve(first_n);
    if (this->column_names.size() > 0) {
        col_names.reserve(first_n);
    }
    for (size_t i = 0; i < std::min(first_n, this->column_types.size()); i++) {
        dtypes.push_back(this->column_types[i]->copy());
        if (this->column_names.size() > 0) {
            col_names.push_back(this->column_names[i]);
        }
    }
    return std::make_unique<Schema>(std::move(dtypes), std::move(col_names),
                                    this->metadata);
}

template <typename T>
    requires(std::integral<T> && !std::same_as<T, bool>)
std::unique_ptr<Schema> Schema::Project(
    const std::vector<T>& column_indices) const {
    std::vector<std::unique_ptr<DataType>> dtypes;
    std::vector<std::string> col_names;
    dtypes.reserve(column_indices.size());
    if (this->column_names.size() > 0) {
        col_names.reserve(column_indices.size());
    }
    for (T col_idx : column_indices) {
        assert(static_cast<size_t>(col_idx) < this->column_types.size());
        dtypes.push_back(this->column_types[col_idx]->copy());
        if (this->column_names.size() > 0) {
            col_names.push_back(this->column_names[col_idx]);
        }
    }
    return std::make_unique<Schema>(std::move(dtypes), std::move(col_names),
                                    this->metadata);
}

// Explicit template instantiations
template std::unique_ptr<Schema> Schema::Project<int>(
    const std::vector<int>& column_indices) const;
template std::unique_ptr<Schema> Schema::Project<int64_t>(
    const std::vector<int64_t>& column_indices) const;
template std::unique_ptr<Schema> Schema::Project<uint64_t>(
    const std::vector<uint64_t>& column_indices) const;

std::shared_ptr<arrow::Schema> Schema::ToArrowSchema() const {
    if (this->column_names.size() == 0 && this->ncols() != 0) {
        throw std::runtime_error(
            "Schema::ToArrowSchema: column names not available");
    }

    if (!this->metadata) {
        throw std::runtime_error(
            "Schema::ToArrowSchema: metadata not available");
    }

    std::vector<std::shared_ptr<::arrow::Field>> fields;
    fields.reserve(this->column_types.size());

    uint32_t idx = 0;
    for (size_t i = 0; i < this->column_types.size(); i++) {
        const std::unique_ptr<DataType>& data_type = this->column_types[i];
        std::string name = fmt::format("field_{}", idx);
        // Use table name if available
        if (this->column_names.size() > 0) {
            name = this->column_names[i];
        }
        fields.push_back(data_type->ToArrowType(name));
        idx++;
    }
    std::shared_ptr<const arrow::KeyValueMetadata> arrow_metadata =
        std::make_shared<const arrow::KeyValueMetadata>(this->metadata->keys,
                                                        this->metadata->values);
    return std::make_shared<arrow::Schema>(fields, arrow_metadata);
}

std::shared_ptr<Schema> Schema::FromArrowSchema(
    std::shared_ptr<::arrow::Schema> schema) {
    std::vector<std::unique_ptr<DataType>> column_types;
    for (const auto& field : schema->fields()) {
        // TODO: support dictionary-encoded arrays
        auto bodo_type = arrow_type_to_bodo_data_type(field->type());
        column_types.push_back(bodo_type->copy());
    }
    std::shared_ptr<TableMetadata> metadata;
    if (schema->metadata()) {
        metadata = std::make_shared<TableMetadata>(
            schema->metadata()->keys(), schema->metadata()->values());
    } else {
        metadata = std::make_shared<TableMetadata>(std::vector<std::string>{},
                                                   std::vector<std::string>{});
    }
    return std::make_shared<Schema>(std::move(column_types),
                                    schema->field_names(), metadata);
}

std::unique_ptr<Schema> Schema::copy() const {
    std::vector<std::unique_ptr<DataType>> column_types_copy;
    column_types_copy.reserve(this->column_types.size());
    for (const auto& t : this->column_types) {
        column_types_copy.push_back(t->copy());
    }
    return std::make_unique<Schema>(std::move(column_types_copy),
                                    this->column_names, this->metadata);
}

}  // namespace bodo

// ---------------------------------------------------------------------------

#if defined(_WIN32)

// Constructor for float/double to int128 conversion
template <FloatOrDouble T>
__int128_t::__int128_t(T in_val) {
    _Word[0] = _Word[1] = 0;

    // Return 0 for NaN and infinity special cases
    if (std::isnan(in_val) || std::isinf(in_val)) {
        return;
    }

    // Check the sign and get magnitude
    bool negative = (in_val < 0.0f);
    double mag =
        negative ? -static_cast<double>(in_val) : static_cast<double>(in_val);

    // If magnitude < 1, the integer part is 0
    if (mag < 1.0L) {
        return;
    }

    // Saturate if mag >= 2^127
    static const double TWO_POW_127 = std::ldexpl((double)1.0, 127);
    if (mag >= TWO_POW_127) {
        // Saturate to INT128_MAX or INT128_MIN
        if (!negative) {
            _Word[0] = 0xFFFFFFFFFFFFFFFFULL;
            _Word[1] = 0x7FFFFFFFFFFFFFFFULL;
        } else {
            _Word[0] = 0ULL;
            _Word[1] = static_cast<int64_t>(0x8000000000000000ULL);
        }
        return;
    }

    // Divide the long double by 2^64 to get the "high" part.
    // floorl(...) ensures we only keep the integer part.
    static const double TWO_POW_64 = std::ldexpl((double)1.0, 64);

    long double hiPart = std::floorl(mag / TWO_POW_64);
    long double loPart = mag - hiPart * TWO_POW_64;

    uint64_t lo64 = static_cast<uint64_t>(loPart);
    uint64_t hi64 = static_cast<uint64_t>(hiPart);

    _Word[0] = lo64;
    _Word[1] = hi64;

    if (negative) {
        // -x = ~x + 1
        uint64_t negLo = ~_Word[0] + 1ULL;
        uint64_t negHi = ~_Word[1];
        if (negLo == 0ULL) {
            // carry into high part
            negHi += 1;
        }
        _Word[0] = negLo;
        _Word[1] = negHi;
    }
}

template __int128_t::__int128_t(float in_val);
template __int128_t::__int128_t(double in_val);

// Helper for int128 for float/double conversion
template <FloatOrDouble T>
T __int128_t::int128_to_float() const {
    const __int128_t& value = *this;

    bool negative = (value < __int128_t(0));
    __int128_t mag = negative ? static_cast<__int128_t>(-value)
                              : static_cast<__int128_t>(value);

    // Combine (high64 * 2^64 + low64) in a higher-precision type
    static const double TWO_POW_64 = std::ldexpl((double)1.0, 64);
    double temp = static_cast<double>(value._Word[1]) * TWO_POW_64 +
                  static_cast<double>(value._Word[0]);

    // If outside float range, clamp or set to infinity.
    if (temp > std::numeric_limits<T>::max()) {
        temp = std::numeric_limits<T>::infinity();
    }

    // Convert to float and restore sign
    T result = static_cast<T>(temp);
    if (negative) {
        result = -result;
    }

    return result;
}

template float __int128_t::int128_to_float<float>() const;
template double __int128_t::int128_to_float<double>() const;

#endif

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

std::unique_ptr<array_info> alloc_numpy_array_all_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    // Note: This check is derived from SQLNASentinelDtype
    if (typ_enum == Bodo_CTypes::DATETIME ||
        typ_enum == Bodo_CTypes::TIMEDELTA) {
        std::unique_ptr<array_info> arr =
            alloc_numpy(length, typ_enum, pool, mm);
        int64_t* data = (int64_t*)arr->data1<bodo_array_type::NUMPY>();
        for (int64_t i = 0; i < length; i++) {
            // Set all values to the sentinel NULL value.
            data[i] = std::numeric_limits<int64_t>::min();
        }
        return arr;
    } else {
        throw std::runtime_error(
            "alloc_numpy_array_all_nulls called on a NUMPY array without null "
            "support. Possible type mismatch.");
    }
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

std::unique_ptr<array_info> alloc_array_item_all_nulls(
    int64_t n_arrays, std::shared_ptr<array_info> inner_arr,
    int64_t extra_null_bytes, bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    std::unique_ptr<array_info> arr = alloc_array_item(
        n_arrays, inner_arr, extra_null_bytes, pool, std::move(mm));
    int64_t n_bytes = ((n_arrays + 7) >> 3) + extra_null_bytes;
    // set to all null
    memset(arr->null_bitmask<bodo_array_type::ARRAY_ITEM>(), 0x00, n_bytes);
    // set offsets to all 0
    offset_t* offsets_ptr =
        (offset_t*)arr->data1<bodo_array_type::ARRAY_ITEM>();
    memset(offsets_ptr, 0, (n_arrays + 1) * sizeof(offset_t));
    return arr;
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

std::unique_ptr<array_info> alloc_categorical_array_all_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t num_categories,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    std::unique_ptr<array_info> arr = alloc_categorical(
        length, typ_enum, num_categories, pool, std::move(mm));
    int64_t size = length * numpy_item_size[typ_enum];
    // Null is -1 for Categorical arrays, so set all values to -1.
    memset(arr->data1<bodo_array_type::CATEGORICAL>(), 0xff, size);
    return arr;
}

std::unique_ptr<array_info> alloc_nullable_array(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm, std::string timezone) {
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
            {std::move(buffer), std::move(buffer_bitmask)}),
        std::vector<std::shared_ptr<array_info>>({}), 0, 0, 0, -1, false, false,
        false, 0, std::vector<std::string>({}), timezone);
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

std::unique_ptr<array_info> alloc_string_array_all_nulls(
    Bodo_CTypes::CTypeEnum typ_enum, int64_t length, int64_t extra_null_bytes,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    std::unique_ptr<array_info> arr =
        alloc_string_array(typ_enum, length, 0, -1, extra_null_bytes, false,
                           false, false, pool, std::move(mm));
    int64_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    // set to all null
    memset(arr->null_bitmask(), 0x00, n_bytes);
    // set offsets to all 0
    offset_t* offsets_ptr = (offset_t*)arr->data2<bodo_array_type::STRING>();
    memset(offsets_ptr, 0, (length + 1) * sizeof(offset_t));
    return arr;
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

std::unique_ptr<array_info> alloc_dict_string_array_all_nulls(
    int64_t length, int64_t extra_null_bytes, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    std::unique_ptr<array_info> arr = alloc_dict_string_array(
        length, 0, 0, extra_null_bytes, pool, std::move(mm));
    size_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    memset(arr->null_bitmask<bodo_array_type::DICT>(), 0x00, n_bytes);
    return arr;
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

std::unique_ptr<array_info> alloc_timestamptz_array_all_nulls(
    int64_t length, int64_t extra_null_bytes, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    std::unique_ptr<array_info> arr =
        alloc_timestamptz_array(length, extra_null_bytes, pool, std::move(mm));
    size_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    memset(arr->null_bitmask<bodo_array_type::TIMESTAMPTZ>(), 0x00, n_bytes);
    return arr;
}

std::unique_ptr<array_info> alloc_all_null_array_top_level(
    int64_t length, bodo_array_type::arr_type_enum arr_type,
    Bodo_CTypes::CTypeEnum dtype, int64_t extra_null_bytes,
    int64_t num_categories, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    switch (arr_type) {
        case bodo_array_type::NUMPY:
            return alloc_numpy_array_all_nulls(length, dtype, pool, mm);
        case bodo_array_type::STRING:
            return alloc_string_array_all_nulls(dtype, length, extra_null_bytes,
                                                pool, mm);
        case bodo_array_type::NULLABLE_INT_BOOL:
            return alloc_nullable_array_all_nulls(length, dtype,
                                                  extra_null_bytes, pool, mm);
        case bodo_array_type::CATEGORICAL:
            return alloc_categorical_array_all_nulls(length, dtype,
                                                     num_categories, pool, mm);
        case bodo_array_type::DICT:
            return alloc_dict_string_array_all_nulls(length, dtype, pool, mm);
        case bodo_array_type::TIMESTAMPTZ:
            return alloc_timestamptz_array_all_nulls(length, extra_null_bytes,
                                                     pool, mm);
        default:
            // Interval arrays don't seem to have a valid "null" representation.
            // TODO: How do we handle array item array? We can make the output
            // null easily, but it seems like we need the inner array to have a
            // consistent type.
            // TODO: How do we handle struct? Conceptually it seems like Python
            // will expect n arrays of all nulls rather than nulls at the upper
            // most level.
            throw std::runtime_error("Unsupported array type");
    }
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
    return make_numpy_array_payload(meminfo, nullptr, length, itemsize, data,
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

std::unique_ptr<array_info> alloc_array_like(
    std::shared_ptr<array_info> in_arr, bool reuse_dictionaries,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
    Bodo_CTypes::CTypeEnum dtype = in_arr->dtype;
    if (arr_type == bodo_array_type::ARRAY_ITEM) {
        return alloc_array_item(
            0, alloc_array_like(in_arr->child_arrays.front(), true, pool, mm),
            0, pool, mm);
    } else if (arr_type == bodo_array_type::STRUCT) {
        std::vector<std::shared_ptr<array_info>> child_arrays;
        child_arrays.reserve(in_arr->child_arrays.size());
        for (std::shared_ptr<array_info> child_array : in_arr->child_arrays) {
            child_arrays.push_back(
                alloc_array_like(child_array, true, pool, mm));
        }
        return alloc_struct(0, std::move(child_arrays));
    } else if (arr_type == bodo_array_type::MAP) {
        std::unique_ptr<array_info> array_item_arr = alloc_array_item(
            0,
            alloc_array_like(in_arr->child_arrays.front()->child_arrays.front(),
                             true, pool, mm),
            0, pool, mm);
        return std::make_unique<array_info>(
            bodo_array_type::MAP, Bodo_CTypes::MAP, 0,
            std::vector<std::shared_ptr<BodoBuffer>>({}),
            std::vector<std::shared_ptr<array_info>>(
                {std::move(array_item_arr)}));
    } else {
        std::unique_ptr<array_info> out_arr = alloc_array_top_level(
            0, 0, 0, arr_type, dtype, -1, 0, 0, false, false, false, pool, mm);
        out_arr->precision = in_arr->precision;
        out_arr->scale = in_arr->scale;
        // For dict encoded columns, re-use the same dictionary if
        // reuse_dictionaries = true
        if (reuse_dictionaries && (arr_type == bodo_array_type::DICT)) {
            out_arr->child_arrays.front() = in_arr->child_arrays.front();
        }
        return out_arr;
    }
}

int64_t array_memory_size(std::shared_ptr<array_info> earr,
                          bool include_dict_size, bool include_children,
                          bool approximate_string_size) {
    if (earr->arr_type == bodo_array_type::NUMPY ||
        earr->arr_type == bodo_array_type::CATEGORICAL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        return siztype * earr->length;
    } else if (earr->arr_type == bodo_array_type::DICT) {
        // Not all functions want to consider the size of the dictionary.
        int64_t dict_size =
            include_dict_size
                ? array_memory_size(earr->child_arrays[0], include_dict_size,
                                    include_children, approximate_string_size)
                : 0;
        return dict_size +
               array_memory_size(earr->child_arrays[1], include_dict_size,
                                 include_children, approximate_string_size);
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
        int64_t num_chars = 0;
        if (approximate_string_size) {
            num_chars = earr->buffers[0]->getMeminfo()->size - earr->offset;
        } else {
            num_chars = earr->n_sub_elems();
        }
        return num_chars + sizeof(offset_t) * (earr->length + 1) + n_bytes;
    } else if (earr->arr_type == bodo_array_type::ARRAY_ITEM) {
        int64_t n_bytes = ((earr->length + 7) >> 3);
        return n_bytes + sizeof(offset_t) * (earr->length + 1) +
               (include_children
                    ? array_memory_size(earr->child_arrays.front(),
                                        include_dict_size, include_children)
                    : 0);
    } else if (earr->arr_type == bodo_array_type::STRUCT) {
        int64_t n_bytes = ((earr->length + 7) >> 3), child_array_size = 0;
        if (include_children) {
            for (const std::shared_ptr<array_info>& child_array :
                 earr->child_arrays) {
                child_array_size += array_memory_size(
                    child_array, include_dict_size, include_children,
                    approximate_string_size);
            }
        }
        return n_bytes + child_array_size;
    } else if (earr->arr_type == bodo_array_type::MAP) {
        return include_children
                   ? array_memory_size(earr->child_arrays.front(),
                                       include_dict_size, include_children,
                                       approximate_string_size)
                   : 0;
    }
    throw std::runtime_error(
        "Array Type: " + GetArrType_as_string(earr->arr_type) +
        " not covered in array_memory_size()");
}

int64_t array_dictionary_memory_size(std::shared_ptr<array_info> earr) {
    if (earr->arr_type == bodo_array_type::DICT) {
        return array_memory_size(earr->child_arrays[0], true, true);
    } else if (earr->arr_type == bodo_array_type::MAP) {
        return array_dictionary_memory_size(earr->child_arrays.front());
    } else if (earr->arr_type == bodo_array_type::ARRAY_ITEM) {
        return array_dictionary_memory_size(earr->child_arrays.front());
    } else if (earr->arr_type == bodo_array_type::STRUCT) {
        int64_t dict_size = 0;
        for (auto& child_arr : earr->child_arrays) {
            dict_size += array_dictionary_memory_size(child_arr);
        }
        return dict_size;
    }
    return 0;
}

int64_t table_local_memory_size(const std::shared_ptr<table_info>& table,
                                bool include_dict_size,
                                bool approximate_string_size) {
    int64_t local_size = 0;
    for (auto& arr : table->columns) {
        local_size += array_memory_size(arr, include_dict_size, true,
                                        approximate_string_size);
    }
    return local_size;
}

int64_t table_local_dictionary_memory_size(
    const std::shared_ptr<table_info>& table) {
    int64_t local_size = 0;
    for (auto& arr : table->columns) {
        local_size += array_dictionary_memory_size(arr);
    }
    return local_size;
}

int64_t table_global_memory_size(const std::shared_ptr<table_info>& table) {
    int64_t local_size = table_local_memory_size(table, false);
    int64_t global_size;
    CHECK_MPI(MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG_LONG_INT,
                            MPI_SUM, MPI_COMM_WORLD),
              "table_global_memory_size: MPI error on MPI_Allreduce:");
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
        farr->scale = earr->scale;
        farr->precision = earr->precision;
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
            return nullable + (numpy_item_size[c_type] * 8);
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
    if (meminfo != nullptr && meminfo->refct != -1) {
        meminfo->refct--;
        if (meminfo->refct == 0) {
            NRT_MemInfo_call_dtor(meminfo);
        }
    }
}

void incref_meminfo(MemInfo* meminfo) {
    if (meminfo != nullptr && meminfo->refct != -1) {
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

    for (const auto& column : table->columns) {
        _get_dtypes_arr_types_from_array(column, arr_c_types, arr_array_types);
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

std::string get_bodo_version() {
    // Load the module
    PyObject* bodo_mod = PyImport_ImportModule("bodo");

    // Get the version attribute
    PyObject* version = PyObject_GetAttrString(bodo_mod, "__version__");
    if (version == nullptr) {
        throw std::runtime_error("Unable to retrieve bodo version");
    }

    // Convert to C++
    const char* version_str = (char*)PyUnicode_DATA(version);
    size_t version_length = PyUnicode_GET_LENGTH(version);
    std::string result(version_str, version_length);

    Py_DECREF(bodo_mod);
    Py_DECREF(version);
    return result;
}
