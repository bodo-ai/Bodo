#include "_javascript_udf.h"
#include <fmt/format.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include "_bodo_common.h"
#include "_utils.h"
PUSH_IGNORED_COMPILER_ERROR("-Wtemplate-id-cdtor")
#include "include/v8-date.h"
#include "include/v8-exception.h"
#include "include/v8-function.h"
#include "include/v8-initialization.h"
#include "include/v8-local-handle.h"
#include "include/v8-primitive.h"
POP_IGNORED_COMPILER_ERROR()

static v8::Isolate::CreateParams create_params;
static bool v8_initialized = false;
static const char *enable_mem_debug_env_var = std::getenv("BODO_DEBUG_V8_MEM");
static const bool enable_mem_debug =
    enable_mem_debug_env_var != nullptr &&
    std::string(enable_mem_debug_env_var) == "1";

/** @brief Create a new isolate configured
 * for throughput over latency
 * @return The new isolate
 */
v8::Isolate *create_new_isolate() {
    v8::Isolate *isolate = v8::Isolate::New(create_params);
    // Favor throughput over latency
    isolate->SetRAILMode(v8::PERFORMANCE_LOAD);
    return isolate;
}

void init_v8() {
    if (!v8_initialized) {
        v8::V8::SetFlagsFromString("--single-threaded");
        v8_platform_isolate_instance = std::make_shared<v8_platform_isolate>();
        // Use a single threaded platform so we don't have contention
        v8_platform_isolate_instance->platform =
            v8::platform::NewSingleThreadedDefaultPlatform();
        v8::V8::InitializePlatform(
            v8_platform_isolate_instance->platform.get());

        v8::V8::Initialize();
        // Create a new Isolate and make it the current one, we'll only ever
        // need one.
        create_params.array_buffer_allocator_shared =
            std::shared_ptr<v8::ArrayBuffer::Allocator>(
                v8::ArrayBuffer::Allocator::NewDefaultAllocator());
        uint64_t max_mem = get_physically_installed_memory();
        create_params.constraints.ConfigureDefaults(max_mem, 0);
        if (enable_mem_debug) {
            std::cerr << "Configuring V8 with "
                      << BytesToHumanReadableString(max_mem) << " of memory"
                      << std::endl;
            std::cerr << "Set V8 constraints to " << std::endl;
            std::cerr << "  max_old_space_size: "
                      << BytesToHumanReadableString(
                             create_params.constraints
                                 .max_old_generation_size_in_bytes())
                      << std::endl;
            std::cerr << "  max_young_space_size: "
                      << BytesToHumanReadableString(
                             create_params.constraints
                                 .max_young_generation_size_in_bytes())
                      << std::endl;
            if (create_params.constraints.stack_limit() != nullptr) {
                std::cerr << "  stack_limit: "
                          << create_params.constraints.stack_limit()
                          << std::endl;
            }
            std::cerr << std::endl;
        }
        v8_initialized = true;
        v8_platform_isolate_instance->isolate = create_new_isolate();
    }
}

std::unique_ptr<JavaScriptFunction> JavaScriptFunction::create(
    std::string _body, std::vector<std::string> _arg_names,
    std::unique_ptr<bodo::DataType> _return_type) {
    init_v8();
    v8::HandleScope handle_scope(v8_platform_isolate_instance->isolate);
    std::unique_ptr<JavaScriptFunction> f = std::unique_ptr<JavaScriptFunction>(
        new JavaScriptFunction(_body, _arg_names, std::move(_return_type)));
    return f;
}

JavaScriptFunction::JavaScriptFunction(
    std::string _body, std::vector<std::string> _arg_names,
    std::unique_ptr<bodo::DataType> _return_type)
    : return_type(std::move(_return_type)),
      arg_names(_arg_names),
      context(v8_platform_isolate_instance->isolate,
              v8::Context::New(v8_platform_isolate_instance->isolate)) {
    v8::Local<v8::Context> local_context =
        context.Get(v8_platform_isolate_instance->isolate);

    // Disallow code generation from strings to prevent eval, this matches
    // Snowflake's behavior
    local_context->AllowCodeGenerationFromStrings(false);

    v8::Isolate::Scope isolate_scope(v8_platform_isolate_instance->isolate);
    // Create the handle scope and set the active context
    v8::HandleScope handle_scope(v8_platform_isolate_instance->isolate);
    v8::Context::Scope context_scope(local_context);
    v8::TryCatch try_catch(v8_platform_isolate_instance->isolate);

    std::mt19937 gen;
    std::uniform_int_distribution<> distrib(65, 90);
    char random_function_name_char_arr[this->size_rand_names];
    for (size_t i = 0; i < this->size_rand_names; i++) {
        random_function_name_char_arr[i] = distrib(gen);
    }
    this->random_function_name =
        std::string(random_function_name_char_arr, this->size_rand_names);

    std::string script_str =
        fmt::format("function {}({}) {{ {} }}", this->random_function_name,
                    fmt::join(this->arg_names, ", "), _body);
    v8::Local<v8::String> source =
        v8::String::NewFromUtf8(v8_platform_isolate_instance->isolate,
                                script_str.c_str())
            .ToLocalChecked();

    v8::MaybeLocal<v8::Script> maybe_script =
        v8::Script::Compile(local_context, source);
    CHECK_V8_EXCEPTION(v8_platform_isolate_instance->isolate, local_context,
                       try_catch, "Error initializing JavaScript UDF");
    v8::Local<v8::Script> script = maybe_script.ToLocalChecked();

    script->Run(local_context).IsEmpty();
    CHECK_V8_EXCEPTION(v8_platform_isolate_instance->isolate, local_context,
                       try_catch, "Error initializing JavaScript UDF");

    v8::Local<v8::String> v8_func_name =
        v8::String::NewFromUtf8(v8_platform_isolate_instance->isolate,
                                this->random_function_name.c_str())
            .ToLocalChecked();
    v8::Local<v8::Value> v8_func_value = local_context->Global()
                                             ->Get(local_context, v8_func_name)
                                             .ToLocalChecked();
    assert(v8_func_value->IsFunction());
    this->v8_func.Reset(v8_platform_isolate_instance->isolate,
                        v8::Local<v8::Function>::Cast(v8_func_value));
    if (enable_mem_debug) {
        v8::HeapStatistics heap_stats;
        v8_platform_isolate_instance->isolate->GetHeapStatistics(&heap_stats);
        std::cerr << "UDF created. V8 heap physical size: "
                  << BytesToHumanReadableString(
                         heap_stats.total_physical_size())
                  << std::endl;
    }
}

JavaScriptFunction *create_javascript_udf_py_entry(
    char *body, int32_t body_len, array_info *arg_names,
    int8_t *return_array_type, int8_t *return_array_c_type,
    int32_t size_return_array_type) {
    try {
        init_v8();
        auto return_datatype = bodo::DataType::Deserialize(
            std::vector(return_array_type,
                        return_array_type + size_return_array_type),
            std::vector(return_array_c_type,
                        return_array_c_type + size_return_array_type));
        assert(arg_names->arr_type == bodo_array_type::STRING);
        assert(arg_names->dtype == Bodo_CTypes::STRING);
        std::vector<std::string> arg_names_vec;
        arg_names_vec.reserve(arg_names->length);
        for (size_t i = 0; i < arg_names->length; ++i) {
            offset_t start =
                arg_names->data2<bodo_array_type::STRING, offset_t>()[i];
            offset_t end =
                arg_names->data2<bodo_array_type::STRING, offset_t>()[i + 1];
            arg_names_vec.push_back(
                std::string(arg_names->data1<bodo_array_type::STRING>() + start,
                            end - start));
        }
        v8::Isolate::Scope isolate_scope(v8_platform_isolate_instance->isolate);
        v8::HandleScope handle_scope(v8_platform_isolate_instance->isolate);
        JavaScriptFunction *f =
            new JavaScriptFunction(std::string(body, body_len), arg_names_vec,
                                   std::move(return_datatype));
        return f;

    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
}

void delete_javascript_udf_py_entry(JavaScriptFunction *func) {
    try {
        delete func;
        if (enable_mem_debug) {
            v8::HeapStatistics heap_stats;
            v8_platform_isolate_instance->isolate->GetHeapStatistics(
                &heap_stats);
            std::cerr << "Deleted JavaScript UDF. V8 heap physical size: "
                      << BytesToHumanReadableString(
                             heap_stats.total_physical_size())
                      << std::endl;
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

// Convert numeric args from Bodo to V8
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(numpy_array<arr_type> || nullable_array<arr_type>)
void convert_bodo_to_v8_helper(v8::Local<v8::Context> &local_context,
                               const std::shared_ptr<array_info> &src_arr,
                               std::vector<v8::Local<v8::Value>> &arg_col) {
    using T = typename dtype_to_type<dtype>::type;
    for (size_t i = 0; i < src_arr->length; i++) {
        if (non_null_at<arr_type, T, dtype>(*src_arr, i)) {
            T arg_val = src_arr->data1<arr_type, T>()[i];
            arg_col.emplace_back(v8::Number::New(
                v8_platform_isolate_instance->isolate, arg_val));
        } else {
            arg_col.emplace_back(
                v8::Null(v8_platform_isolate_instance->isolate));
        }
    }
}

// Convert date arguments from Bodo to V8
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires((numpy_array<arr_type> || nullable_array<arr_type>) &&
             dtype == Bodo_CTypes::DATE)
void convert_bodo_to_v8_helper(v8::Local<v8::Context> &local_context,
                               const std::shared_ptr<array_info> &src_arr,
                               std::vector<v8::Local<v8::Value>> &arg_col) {
    using T = typename dtype_to_type<dtype>::type;
    for (size_t i = 0; i < src_arr->length; i++) {
        if (non_null_at<arr_type, T, dtype>(*src_arr, i)) {
            T arg_val = src_arr->data1<arr_type, T>()[i];
            // Convert from days since unix epoch to ms since unix epoch.
            int64_t arg_ms = ((int64_t)arg_val) * 24 * 60 * 60 * 1000;
            arg_col.emplace_back(
                v8::Date::New(local_context, arg_ms).ToLocalChecked());
        } else {
            arg_col.emplace_back(
                v8::Null(v8_platform_isolate_instance->isolate));
        }
    }
}

// Convert nullable booleans from Bodo to V8
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(nullable_array<arr_type> && bool_dtype<dtype>)
void convert_bodo_to_v8_helper(v8::Local<v8::Context> &local_context,
                               const std::shared_ptr<array_info> &src_arr,
                               std::vector<v8::Local<v8::Value>> &arg_col) {
    for (size_t i = 0; i < src_arr->length; i++) {
        if (src_arr->get_null_bit<arr_type>(i)) {
            bool arg_val = GetBit(src_arr->data1<arr_type, uint8_t>(), i);
            arg_col.emplace_back(v8::Boolean::New(
                v8_platform_isolate_instance->isolate, arg_val));
        } else {
            arg_col.emplace_back(
                v8::Null(v8_platform_isolate_instance->isolate));
        }
    }
}

// Convert numpy booleans from Bodo to V8
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(numpy_array<arr_type> && bool_dtype<dtype>)
void convert_bodo_to_v8_helper(v8::Local<v8::Context> &local_context,
                               const std::shared_ptr<array_info> &src_arr,
                               std::vector<v8::Local<v8::Value>> &arg_col) {
    for (size_t i = 0; i < src_arr->length; i++) {
        bool arg_val = src_arr->data1<arr_type, uint8_t>()[i];
        arg_col.emplace_back(
            v8::Boolean::New(v8_platform_isolate_instance->isolate, arg_val));
    }
}

// Convert regular strings from Bodo to V8
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(string_array<arr_type>)
void convert_bodo_to_v8_helper(v8::Local<v8::Context> &local_context,
                               const std::shared_ptr<array_info> &src_arr,
                               std::vector<v8::Local<v8::Value>> &arg_col) {
    char *raw_data_ptr = src_arr->data1<bodo_array_type::STRING>();
    offset_t *offsets = src_arr->data2<bodo_array_type::STRING, offset_t>();
    for (size_t i = 0; i < src_arr->length; i++) {
        if (src_arr->get_null_bit<arr_type>(i)) {
            offset_t start_offset = offsets[i];
            offset_t end_offset = offsets[i + 1];
            offset_t len = end_offset - start_offset;
            v8::MaybeLocal<v8::String> maybeLocalStr = v8::String::NewFromUtf8(
                v8_platform_isolate_instance->isolate,
                &raw_data_ptr[start_offset], v8::NewStringType::kNormal, len);
            arg_col.emplace_back(maybeLocalStr.ToLocalChecked());
        } else {
            arg_col.emplace_back(
                v8::Null(v8_platform_isolate_instance->isolate));
        }
    }
}

// Convert binary data from Bodo to V8
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(string_array<arr_type> && dtype == Bodo_CTypes::BINARY)
void convert_bodo_to_v8_helper(v8::Local<v8::Context> &local_context,
                               const std::shared_ptr<array_info> &src_arr,
                               std::vector<v8::Local<v8::Value>> &arg_col) {
    char *raw_data_ptr = src_arr->data1<bodo_array_type::STRING>();
    offset_t *offsets = src_arr->data2<bodo_array_type::STRING, offset_t>();
    for (size_t i = 0; i < src_arr->length; i++) {
        if (src_arr->get_null_bit<arr_type>(i)) {
            offset_t start_offset = offsets[i];
            offset_t end_offset = offsets[i + 1];
            offset_t len = end_offset - start_offset;
            v8::Local<v8::ArrayBuffer> localBuffer = v8::ArrayBuffer::New(
                v8_platform_isolate_instance->isolate, len);
            memcpy(localBuffer->Data(), &raw_data_ptr[start_offset], len);
            v8::Local<v8::Uint8Array> localBinary =
                v8::Uint8Array::New(localBuffer, 0, len);
            arg_col.emplace_back(localBinary);
        } else {
            arg_col.emplace_back(
                v8::Null(v8_platform_isolate_instance->isolate));
        }
    }
}

// Convert dictionary-encoded strings from Bodo to V8
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
    requires(dict_array<arr_type>)
void convert_bodo_to_v8_helper(v8::Local<v8::Context> &local_context,
                               const std::shared_ptr<array_info> &src_arr,
                               std::vector<v8::Local<v8::Value>> &arg_col) {
    std::shared_ptr<array_info> string_arr = src_arr->child_arrays[0];
    std::shared_ptr<array_info> idx_arr = src_arr->child_arrays[1];
    char *raw_data_ptr = string_arr->data1<bodo_array_type::STRING>();
    offset_t *offsets = string_arr->data2<bodo_array_type::STRING, offset_t>();
    dict_indices_t *indices =
        idx_arr->data1<bodo_array_type::NULLABLE_INT_BOOL, dict_indices_t>();
    for (size_t i = 0; i < src_arr->length; i++) {
        if (src_arr->get_null_bit<arr_type>(i)) {
            int64_t dict_idx = indices[i];
            offset_t start_offset = offsets[dict_idx];
            offset_t end_offset = offsets[dict_idx + 1];
            offset_t len = end_offset - start_offset;
            v8::MaybeLocal<v8::String> maybeLocalStr = v8::String::NewFromUtf8(
                v8_platform_isolate_instance->isolate,
                &raw_data_ptr[start_offset], v8::NewStringType::kNormal, len);
            arg_col.emplace_back(maybeLocalStr.ToLocalChecked());
        } else {
            arg_col.emplace_back(
                v8::Null(v8_platform_isolate_instance->isolate));
        }
    }
}

// Convert a Bodo array to a vector of V8 values
void convert_bodo_to_v8(v8::Local<v8::Context> &local_context,
                        const std::shared_ptr<array_info> &src_arr,
                        std::vector<v8::Local<v8::Value>> &arg_col) {
#define bodo_to_v8_case(arr_type, dtype)                                   \
    case dtype:                                                            \
        convert_bodo_to_v8_helper<arr_type, dtype>(local_context, src_arr, \
                                                   arg_col);               \
        break;
    switch (src_arr->arr_type) {
        case bodo_array_type::NUMPY: {
            switch (src_arr->dtype) {
                bodo_to_v8_case(bodo_array_type::NUMPY, Bodo_CTypes::UINT8);
                bodo_to_v8_case(bodo_array_type::NUMPY, Bodo_CTypes::UINT16);
                bodo_to_v8_case(bodo_array_type::NUMPY, Bodo_CTypes::UINT32);
                bodo_to_v8_case(bodo_array_type::NUMPY, Bodo_CTypes::UINT64);
                bodo_to_v8_case(bodo_array_type::NUMPY, Bodo_CTypes::INT8);
                bodo_to_v8_case(bodo_array_type::NUMPY, Bodo_CTypes::INT16);
                bodo_to_v8_case(bodo_array_type::NUMPY, Bodo_CTypes::INT32);
                bodo_to_v8_case(bodo_array_type::NUMPY, Bodo_CTypes::INT64);
                bodo_to_v8_case(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT32);
                bodo_to_v8_case(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64);
                bodo_to_v8_case(bodo_array_type::NUMPY, Bodo_CTypes::_BOOL);
                bodo_to_v8_case(bodo_array_type::NUMPY, Bodo_CTypes::DATE);
                default:
                    throw std::runtime_error(
                        "execute_javascript_udf: unsupported argument dtype "
                        "for numpy array " +
                        GetArrType_as_string(src_arr->dtype));
            }
            break;
        }
        case bodo_array_type::NULLABLE_INT_BOOL: {
            switch (src_arr->dtype) {
                bodo_to_v8_case(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::UINT8);
                bodo_to_v8_case(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::UINT16);
                bodo_to_v8_case(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::UINT32);
                bodo_to_v8_case(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::UINT64);
                bodo_to_v8_case(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT8);
                bodo_to_v8_case(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT16);
                bodo_to_v8_case(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT32);
                bodo_to_v8_case(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT64);
                bodo_to_v8_case(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::FLOAT32);
                bodo_to_v8_case(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::FLOAT64);
                bodo_to_v8_case(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::_BOOL);
                bodo_to_v8_case(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::DATE);
                default: {
                    throw std::runtime_error(
                        "execute_javascript_udf: unsupported argument dtype "
                        "for nullable array " +
                        GetArrType_as_string(src_arr->dtype));
                }
            }
            break;
        }
        case bodo_array_type::STRING: {
            switch (src_arr->dtype) {
                bodo_to_v8_case(bodo_array_type::STRING, Bodo_CTypes::STRING);
                bodo_to_v8_case(bodo_array_type::STRING, Bodo_CTypes::BINARY);
                default: {
                    throw std::runtime_error(
                        "execute_javascript_udf: unsupported argument dtype "
                        "for string array " +
                        GetArrType_as_string(src_arr->dtype));
                }
            }
            break;
        }
        case bodo_array_type::DICT: {
            switch (src_arr->dtype) {
                bodo_to_v8_case(bodo_array_type::DICT, Bodo_CTypes::STRING);
                default: {
                    throw std::runtime_error(
                        "execute_javascript_udf: unsupported argument dtype "
                        "for dictionary encoded array " +
                        GetArrType_as_string(src_arr->dtype));
                }
            }
            break;
        }
        default: {
            throw std::runtime_error(
                "execute_javascript_udf: unsupported argument array type " +
                GetArrType_as_string(src_arr->arr_type));
        }
    }
}

// Templated helper for execute_javascript_udf to handle array/dtype-specific
// logic
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype>
void execute_javascript_udf_body(
    v8::Local<v8::Context> local_context, v8::Local<v8::Function> local_v8_func,
    std::shared_ptr<ArrayBuildBuffer> ret, size_t out_count, size_t argc,
    const std::vector<std::vector<v8::Local<v8::Value>>> &arg_columns) {
    v8::TryCatch trycatch(v8_platform_isolate_instance->isolate);
    for (size_t i = 0; i < out_count; ++i) {
        // Place all of the arguments from the current row in an array
        std::vector<v8::Local<v8::Value>> argv;
        for (size_t arg = 0; arg < argc; arg++) {
            argv.emplace_back(arg_columns[arg][i]);
        }
        // Call the UDF
        auto pre_result = local_v8_func->Call(
            local_context, local_context->Global(), argc, argv.data());
        CHECK_V8_EXCEPTION(
            local_context->GetIsolate(), local_context, trycatch,
            "execute_javascript_udf_body: executing functin failed")
        auto result = pre_result.ToLocalChecked();

        // Write the result to the Bodo array
        append_v8_handle<arr_type, dtype>(local_context, result, ret, trycatch);
    }
}

std::shared_ptr<array_info> execute_javascript_udf(
    JavaScriptFunction *func,
    const std::vector<std::shared_ptr<array_info>> &args) {
    v8::Isolate::Scope isolate_scope(v8_platform_isolate_instance->isolate);
    // Create the handle scope and set the active context
    v8::HandleScope handle_scope(v8_platform_isolate_instance->isolate);
    v8::Local<v8::Context> local_context =
        func->context.Get(v8_platform_isolate_instance->isolate);
    v8::Context::Scope context_scope(local_context);
    v8::Local<v8::Function> local_v8_func =
        func->v8_func.Get(v8_platform_isolate_instance->isolate);

    size_t argc = args.size();
    size_t out_count = argc == 0 ? 1 : args[0]->length;

    // Allocate the output array/builder
    std::shared_ptr<array_info> ret = alloc_array_top_level(
        0, 0, 0, func->return_type->array_type, func->return_type->c_type);
    std::shared_ptr<ArrayBuildBuffer> arr_builder =
        std::make_shared<ArrayBuildBuffer>(ret);
    arr_builder->ReserveSize(out_count);

    // For each argument, construct a column of values converted to V8
    std::vector<std::vector<v8::Local<v8::Value>>> arg_columns(argc);
    for (size_t arg_idx = 0; arg_idx < argc; arg_idx++) {
        convert_bodo_to_v8(local_context, args[arg_idx], arg_columns[arg_idx]);
    }

#define execute_js_udf_dtype_case(arr_type, dtype)                      \
    case dtype: {                                                       \
        execute_javascript_udf_body<arr_type, dtype>(                   \
            local_context, local_v8_func, arr_builder, out_count, argc, \
            arg_columns);                                               \
        break;                                                          \
    }
    switch (func->return_type->array_type) {
        case bodo_array_type::NUMPY: {
            switch (func->return_type->c_type) {
                execute_js_udf_dtype_case(bodo_array_type::NUMPY,
                                          Bodo_CTypes::UINT8);
                execute_js_udf_dtype_case(bodo_array_type::NUMPY,
                                          Bodo_CTypes::UINT16);
                execute_js_udf_dtype_case(bodo_array_type::NUMPY,
                                          Bodo_CTypes::UINT32);
                execute_js_udf_dtype_case(bodo_array_type::NUMPY,
                                          Bodo_CTypes::UINT64);
                execute_js_udf_dtype_case(bodo_array_type::NUMPY,
                                          Bodo_CTypes::INT8);
                execute_js_udf_dtype_case(bodo_array_type::NUMPY,
                                          Bodo_CTypes::INT16);
                execute_js_udf_dtype_case(bodo_array_type::NUMPY,
                                          Bodo_CTypes::INT32);
                execute_js_udf_dtype_case(bodo_array_type::NUMPY,
                                          Bodo_CTypes::INT64);
                execute_js_udf_dtype_case(bodo_array_type::NUMPY,
                                          Bodo_CTypes::FLOAT32);
                execute_js_udf_dtype_case(bodo_array_type::NUMPY,
                                          Bodo_CTypes::FLOAT64);
                execute_js_udf_dtype_case(bodo_array_type::NUMPY,
                                          Bodo_CTypes::_BOOL);
                execute_js_udf_dtype_case(bodo_array_type::NUMPY,
                                          Bodo_CTypes::DATE);
                default: {
                    throw std::runtime_error(
                        "execute_javascript_udf: unsupported output dtype " +
                        GetArrType_as_string(func->return_type->c_type));
                }
            }
            break;
        }
        case bodo_array_type::NULLABLE_INT_BOOL: {
            switch (func->return_type->c_type) {
                execute_js_udf_dtype_case(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::UINT8);
                execute_js_udf_dtype_case(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::UINT16);
                execute_js_udf_dtype_case(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::UINT32);
                execute_js_udf_dtype_case(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::UINT64);
                execute_js_udf_dtype_case(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::INT8);
                execute_js_udf_dtype_case(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::INT16);
                execute_js_udf_dtype_case(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::INT32);
                execute_js_udf_dtype_case(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::INT64);
                execute_js_udf_dtype_case(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::FLOAT32);
                execute_js_udf_dtype_case(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::FLOAT64);
                execute_js_udf_dtype_case(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::_BOOL);
                execute_js_udf_dtype_case(bodo_array_type::NULLABLE_INT_BOOL,
                                          Bodo_CTypes::DATE);
                default: {
                    throw std::runtime_error(
                        "execute_javascript_udf: unsupported output dtype " +
                        GetArrType_as_string(func->return_type->c_type));
                }
            }
            break;
        }
        case bodo_array_type::STRING: {
            switch (func->return_type->c_type) {
                execute_js_udf_dtype_case(bodo_array_type::STRING,
                                          Bodo_CTypes::STRING);
                execute_js_udf_dtype_case(bodo_array_type::STRING,
                                          Bodo_CTypes::BINARY);
                default: {
                    throw std::runtime_error(
                        "execute_javascript_udf: unsupported output dtype "
                        "for string array " +
                        GetArrType_as_string(func->return_type->c_type));
                }
            }
            break;
        }
        case bodo_array_type::DICT: {
            switch (func->return_type->c_type) {
                execute_js_udf_dtype_case(bodo_array_type::DICT,
                                          Bodo_CTypes::STRING);
                default: {
                    throw std::runtime_error(
                        "execute_javascript_udf: unsupported output dtype "
                        "for dictionary encoded array " +
                        GetArrType_as_string(func->return_type->c_type));
                }
            }
            break;
        }
        default: {
            throw std::runtime_error(
                "execute_javascript_udf: unsupported output array type " +
                GetArrType_as_string(func->return_type->array_type));
        }
    }

    if (enable_mem_debug) {
        v8::HeapStatistics heap_stats;
        v8_platform_isolate_instance->isolate->GetHeapStatistics(&heap_stats);
        std::cerr << "Executed JavaScript UDF. V8 heap physical size: "
                  << BytesToHumanReadableString(
                         heap_stats.total_physical_size())
                  << std::endl;
    }

    return ret;
}

// Assumes length of args == func->arg_names.size()
array_info *execute_javascript_udf_py_entry(JavaScriptFunction *func,
                                            table_info *args) {
    try {
        if (args == nullptr) {
            return new array_info(*execute_javascript_udf(func, {}));
        } else {
            std::unique_ptr<table_info> args_ptr =
                std::unique_ptr<table_info>(args);
            return new array_info(
                *execute_javascript_udf(func, args_ptr->columns));
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
}

PyMODINIT_FUNC PyInit_javascript_udf_cpp(void) {
    PyObject *m;
    MOD_DEF(m, "javascript_udf_cpp", "No docs", NULL);
    if (m == NULL)
        return NULL;

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, create_javascript_udf_py_entry);
    SetAttrStringFromVoidPtr(m, delete_javascript_udf_py_entry);
    SetAttrStringFromVoidPtr(m, execute_javascript_udf_py_entry);

    return m;
}
