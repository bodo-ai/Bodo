#include <arrow/util/bit_util.h>
#include <fmt/format.h>
#include <stdexcept>
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_dict_builder.h"
#include "_table_builder.h"
#include "include/libplatform/libplatform.h"
#include "include/v8-context.h"
#include "include/v8-exception.h"
#include "include/v8-isolate.h"
#include "include/v8-persistent-handle.h"
#include "include/v8-script.h"
#include "include/v8-typed-array.h"

#ifndef CHECK_v8_EXCEPTION
#define CHECK_V8_EXCEPTION(isolate, context, try_catch, error_prefix)         \
    if (try_catch.HasCaught()) [[unlikely]] {                                 \
        v8::String::Utf8Value exception(isolate, try_catch.Exception());      \
        v8::Local<v8::Message> message = try_catch.Message();                 \
        if (message.IsEmpty()) {                                              \
            throw std::runtime_error(                                         \
                fmt::format("{}\n{}", error_prefix, *exception));             \
        } else {                                                              \
            v8::String::Utf8Value filename(                                   \
                isolate, message->GetScriptOrigin().ResourceName());          \
            int linenum = message->GetLineNumber(context).FromMaybe(-1);      \
            if (linenum == -1) {                                              \
                throw std::runtime_error(fmt::format(                         \
                    "{}\n{}: {}", error_prefix, *filename, *exception));      \
            } else {                                                          \
                throw std::runtime_error(fmt::format("{}\n{}:{}: {}",         \
                                                     error_prefix, *filename, \
                                                     linenum, *exception));   \
            }                                                                 \
        }                                                                     \
    }
#endif

// Keep the global platform and isolate together so we can clean up properly
// the isolate must be destroyed before the platform
struct v8_platform_isolate {
    v8::Isolate *isolate;
    std::unique_ptr<v8::Platform> platform;
    ~v8_platform_isolate() {
        v8::platform::NotifyIsolateShutdown(this->platform.get(),
                                            this->isolate);
        this->isolate->Dispose();
    }
};
static std::shared_ptr<v8_platform_isolate> v8_platform_isolate_instance;

struct JavaScriptFunction {
    // The type of the return array
    const std::unique_ptr<bodo::DataType> return_type;
    // The names of the arguments
    const std::vector<std::string> arg_names;
    // The context to execute the function in
    const v8::Global<v8::Context> context;
    // This is a global function so we can call it from any context, contains
    // the passed in body
    v8::Global<v8::Function> v8_func;
    // The event to be used for tracing
    tracing::ResumableEvent tracing_event;
    static std::unique_ptr<JavaScriptFunction> create(
        std::string _body, std::vector<std::string> _arg_names,
        std::unique_ptr<bodo::DataType> _return_type);
    // Marked as friend to allow create_javascript_udf_py_entry to call the
    // constructor, avoids using create which returns a unique pointer when we
    // need a raw pointer
    friend JavaScriptFunction *create_javascript_udf_py_entry(
        char *body, int32_t body_len, array_info *arg_names,
        int8_t *return_array_type, int8_t *return_array_c_type,
        int32_t size_return_array_type);

   private:
    // This is a random function name to avoid collisions with other functions
    const size_t size_rand_names = 32;
    std::string random_function_name;
    /**
     * Private because it needs an active handle_scope, create() should be used
     * instead
     * @param _body The JavaScript code as a string, will be wrapped in a
     * function
     * @param _arg_names The names of the arguments
     * @param _return_type The type of the return array
     */
    JavaScriptFunction(std::string _body, std::vector<std::string> _arg_names,
                       std::unique_ptr<bodo::DataType> _return_type);
};

/**
 * @brief Idempotent function to initialize V8
 */
void init_v8();

/**
 * @brief Python entrypoint to create a JavaScript UDF from a string and return
 * type. This function just wraps body into a string and deserializes the return
 * array type before passing to the constructor.
 * @param body The JavaScript code
 * @param body_len The length of the JavaScript code
 * @param arg_names The names of the arguments, should be a string array_info
 * with one element per argument
 * @param return_array_type The type of the return array in serialized form
 * @param return_array_c_type The dtype of the return array in serialized form
 * @return The JavaScript UDF
 */
JavaScriptFunction *create_javascript_udf_py_entry(
    char *body, int32_t body_len, array_info *arg_names,
    int8_t *return_array_type, int8_t *return_array_c_type,
    int32_t size_return_array_type);

/**
 * @brief Delete a JavaScript UDF
 * @param func The JavaScript UDF
 */
void delete_javascript_udf_py_entry(JavaScriptFunction *func);

/**
 * @brief Python entry point to execute a JavaScript UDF
 * This function just wraps raw pointers before passing to
 * execute_javascript_udf
 * @param f The JavaScript UDF
 * @param args The arguments, must be the same number as f->arg_names.size()
 * @return The resulting array_info
 */
array_info *execute_javascript_udf_py_entry(JavaScriptFunction *f,
                                            table_info *args);

/**
 * @brief Calls the JavaScriptFunction with the supplied args and returns the
 * result
 * @param func JavaScriptFunction to execute
 * @param args The arguments to call the function with, each array should be the
 * same length, or length 1 for scalars.
 * @return Output array, each element is cast to func->return_type. It's of
 * length args[0] or 1 if there are no args
 */
std::shared_ptr<array_info> execute_javascript_udf(
    JavaScriptFunction *func,
    const std::vector<std::shared_ptr<array_info>> &args);

/**
 * @brief Append a v8::Value into an array_info, assumes arr has enough capacity
 * @tparam arr_arr_type The type of the array_info
 * @tparam arr_c_type The C type of the array_info
 * @param ctx The v8 context
 * @param obj The v8 value to append
 * @param arr The ArrayBuildBuffer to append into
 * @param idx The index to append into
 * @param dict_builder The dictionary builder to use, only used for dict, can be
 * nullptr otherwise
 * This function is specialized for c types that require a javascript bigint
 * It can handle bigints that are 1 word
 */
template <bodo_array_type::arr_type_enum arr_arr_type,
          Bodo_CTypes::CTypeEnum arr_c_type>
    requires(arr_arr_type == bodo_array_type::NUMPY &&
             (arr_c_type == Bodo_CTypes::INT64 ||
              arr_c_type == Bodo_CTypes::UINT64))
void append_v8_handle(v8::Local<v8::Context> ctx, v8::Local<v8::Value> obj,
                      const std::shared_ptr<ArrayBuildBuffer> &arr,
                      v8::TryCatch &trycatch) {
    size_t idx = arr->size;
    using val_t = dtype_to_type<arr_c_type>::type;
    val_t val;
    // For some reason you can't convert a Number to a BigInt directly
    if (obj->IsNumber()) {
        // Since we know this is a number we don't need to check
        // the trycatch
        val = static_cast<val_t>(obj->NumberValue(ctx).ToChecked());
    } else {
        auto maybe_js_val = obj->ToBigInt(ctx);
        CHECK_V8_EXCEPTION(ctx->GetIsolate(), ctx, trycatch,
                           "append_v8_handle: ToBigInt failed")
        auto js_val = maybe_js_val.ToLocalChecked();
        // Cast to the correct type based on the array type
        // and truncate if necessary
        if constexpr (arr_c_type == Bodo_CTypes::INT64) {
            val = js_val->Int64Value();
        } else if constexpr (arr_c_type == Bodo_CTypes::UINT64) {
            val = js_val->Uint64Value();
        }
    }
    arr->data_array->data1<arr_arr_type, val_t>()[idx] = val;
    CHECK_ARROW_MEM(
        arr->data_array->buffers[0]->SetSize((arr->size + 1) * sizeof(val_t)),
        "append_v8_handle: SetSize failed")
    arr->data_array->length += 1;
}

/**
 * @brief Append a v8::Value into an array_info, assumes arr has enough capacity
 * @tparam arr_arr_type The type of the array_info
 * @tparam arr_c_type The C type of the array_info
 * @param ctx The v8 context
 * @param obj The v8 value to append
 * @param arr The ArrayBuildBuffer to append into
 * @param idx The index to append into
 * @param dict_builder The dictionary builder to use, only used for dict, can be
 * nullptr otherwise
 * This function is specialized for integer types that are not 64 bits long
 */
template <bodo_array_type::arr_type_enum arr_arr_type,
          Bodo_CTypes::CTypeEnum arr_c_type>
    requires(arr_arr_type == bodo_array_type::NUMPY &&
             (arr_c_type == Bodo_CTypes::INT32 ||
              arr_c_type == Bodo_CTypes::UINT32 ||
              arr_c_type == Bodo_CTypes::INT16 ||
              arr_c_type == Bodo_CTypes::UINT16 ||
              arr_c_type == Bodo_CTypes::INT8 ||
              arr_c_type == Bodo_CTypes::UINT8))
void append_v8_handle(v8::Local<v8::Context> ctx, v8::Local<v8::Value> obj,
                      const std::shared_ptr<ArrayBuildBuffer> &arr,
                      v8::TryCatch &trycatch) {
    size_t idx = arr->size;
    auto maybe_js_val = obj->ToInteger(ctx);
    CHECK_V8_EXCEPTION(ctx->GetIsolate(), ctx, trycatch,
                       "append_v8_handle: ToInteger failed")
    auto js_val = maybe_js_val.ToLocalChecked();
    using val_t = dtype_to_type<arr_c_type>::type;
    val_t val;

    // Cast to the correct type based on the array type
    // and truncate if necessary
    if constexpr (!is_unsigned_integer(arr_c_type)) {
        auto maybe_val = js_val->Int32Value(ctx);
        CHECK_V8_EXCEPTION(ctx->GetIsolate(), ctx, trycatch,
                           "append_v8_handle: To(U)Int32Value failed")
        val = static_cast<val_t>(maybe_val.ToChecked());
    } else {
        auto maybe_val = js_val->Uint32Value(ctx);
        CHECK_V8_EXCEPTION(ctx->GetIsolate(), ctx, trycatch,
                           "append_v8_handle: To(U)Int32Value failed")
        val = static_cast<val_t>(maybe_val.ToChecked());
    }
    arr->data_array->data1<arr_arr_type, val_t>()[idx] = val;
    CHECK_ARROW_MEM(
        arr->data_array->buffers[0]->SetSize((arr->size + 1) * sizeof(val_t)),
        "append_v8_handle: SetSize failed")
    arr->data_array->length += 1;
}

/**
 * @brief Append a v8::Value into an array_info, assumes arr has enough capacity
 * @tparam arr_arr_type The type of the array_info
 * @tparam arr_c_type The C type of the array_info
 * @param ctx The v8 context
 * @param obj The v8 value to append
 * @param arr The ArrayBuildBuffer to append into
 * @param idx The index to append into
 * @param dict_builder The dictionary builder to use, only used for dict, can be
 * nullptr otherwise
 *  This function is specialized for float types
 */
template <bodo_array_type::arr_type_enum arr_arr_type,
          Bodo_CTypes::CTypeEnum arr_c_type>
    requires(arr_arr_type == bodo_array_type::NUMPY &&
             (arr_c_type == Bodo_CTypes::FLOAT32 ||
              arr_c_type == Bodo_CTypes::FLOAT64))
void append_v8_handle(v8::Local<v8::Context> ctx, v8::Local<v8::Value> obj,
                      const std::shared_ptr<ArrayBuildBuffer> &arr,
                      v8::TryCatch &trycatch) {
    size_t idx = arr->size;
    auto maybe_js_val = obj->ToNumber(ctx);
    CHECK_V8_EXCEPTION(ctx->GetIsolate(), ctx, trycatch,
                       "append_v8_handle: ToNumber failed")
    auto js_val = maybe_js_val.ToLocalChecked();
    using val_t = dtype_to_type<arr_c_type>::type;
    val_t val;
    val = static_cast<val_t>(js_val->Value());
    arr->data_array->data1<arr_arr_type, val_t>()[idx] = val;
    CHECK_ARROW_MEM(
        arr->data_array->buffers[0]->SetSize((arr->size + 1) * sizeof(val_t)),
        "append_v8_handle: SetSize failed")
    arr->data_array->length += 1;
}

/**
 * @brief Append a v8::Value into an array_info, assumes arr has enough capacity
 * @tparam arr_arr_type The type of the array_info
 * @tparam arr_c_type The C type of the array_info
 * @param ctx The v8 context
 * @param obj The v8 value to append
 * @param arr The ArrayBuildBuffer to append into
 * @param idx The index to append into
 * @param dict_builder The dictionary builder to use, only used for dict, can be
 * nullptr otherwise
 * This function is specialized for boolean types in NUMPY arrays
 */
template <bodo_array_type::arr_type_enum arr_arr_type,
          Bodo_CTypes::CTypeEnum arr_c_type>
    requires(arr_arr_type == bodo_array_type::NUMPY &&
             arr_c_type == Bodo_CTypes::_BOOL)
void append_v8_handle(v8::Local<v8::Context> ctx, v8::Local<v8::Value> obj,
                      const std::shared_ptr<ArrayBuildBuffer> &arr,
                      v8::TryCatch &trycatch) {
    size_t idx = arr->size;
    auto js_val = obj->ToBoolean(ctx->GetIsolate());
    arr->data_array->data1<arr_arr_type, bool>()[idx] = js_val->Value();
    CHECK_ARROW_MEM(
        arr->data_array->buffers[0]->SetSize((arr->size + 1) * sizeof(bool)),
        "append_v8_handle: SetSize failed")
    arr->data_array->length += 1;
}

/**
 * @brief Append a v8::Value into an array_info, assumes arr has enough capacity
 * @tparam arr_arr_type The type of the array_info
 * @tparam arr_c_type The C type of the array_info
 * @param ctx The v8 context
 * @param obj The v8 value to append
 * @param arr The ArrayBuildBuffer to append into
 * @param idx The index to append into
 * @param dict_builder The dictionary builder to use, only used for dict, can be
 * nullptr otherwise
 * This function is specialized for DATE type in NUMPY arrays
 */
template <bodo_array_type::arr_type_enum arr_arr_type,
          Bodo_CTypes::CTypeEnum arr_c_type>
    requires(arr_arr_type == bodo_array_type::NUMPY &&
             arr_c_type == Bodo_CTypes::DATE)
void append_v8_handle(v8::Local<v8::Context> ctx, v8::Local<v8::Value> obj,
                      const std::shared_ptr<ArrayBuildBuffer> &arr,
                      v8::TryCatch &trycatch) {
    size_t idx = arr->size;
    auto maybe_js_val = obj->NumberValue(ctx);
    CHECK_V8_EXCEPTION(ctx->GetIsolate(), ctx, trycatch,
                       "append_v8_handle: NumberValue failed")
    auto js_val = maybe_js_val.ToChecked();
    using val_t = dtype_to_type<arr_c_type>::type;
    // Convert from milliseconds to days since epoch
    constexpr val_t days_to_ms = 24 * 60 * 60 * 1000;
    arr->data_array->data1<arr_arr_type, val_t>()[idx] =
        val_t(js_val / days_to_ms);
    CHECK_ARROW_MEM(
        arr->data_array->buffers[0]->SetSize((arr->size + 1) * sizeof(val_t)),
        "append_v8_handle: SetSize failed")
    arr->data_array->length += 1;
}

/**
 * @brief Append a v8::Value into an array_info, assumes arr has enough capacity
 * for the offsets, handles resizing the data buffer
 * @tparam arr_arr_type The type of the array_info
 * @tparam arr_c_type The C type of the array_info
 * @param ctx The v8 context
 * @param obj The v8 value to append
 * @param arr The ArrayBuildBuffer to append into
 * @param idx The index to append into
 * @param dict_builder The dictionary builder to use, only used for dict, can be
 * nullptr otherwise
 * This function is specialized for string arrays of string data
 */
template <bodo_array_type::arr_type_enum arr_arr_type,
          Bodo_CTypes::CTypeEnum arr_c_type>
    requires(arr_arr_type == bodo_array_type::STRING &&
             arr_c_type == Bodo_CTypes::STRING)
void append_v8_handle(v8::Local<v8::Context> ctx, v8::Local<v8::Value> obj,
                      const std::shared_ptr<ArrayBuildBuffer> &arr,
                      v8::TryCatch &trycatch) {
    size_t idx = arr->size;
    if (obj->IsNullOrUndefined()) {
        arr->data_array->data2<arr_arr_type, offset_t>()[idx + 1] =
            arr->data_array->data2<arr_arr_type, offset_t>()[idx];
        arr->data_array->set_null_bit<arr_arr_type>(idx, false);
    } else {
        auto js_string = v8::String::Utf8Value(ctx->GetIsolate(), obj);
        assert(*js_string != nullptr);
        arr->ReserveSpaceForStringAppend(js_string.length());
        memcpy(arr->data_array->data1<arr_arr_type>() +
                   arr->data_array->data2<arr_arr_type, offset_t>()[idx],
               *js_string, js_string.length() * sizeof(char));
        arr->data_array->data2<arr_arr_type, offset_t>()[idx + 1] =
            arr->data_array->data2<arr_arr_type, offset_t>()[idx] +
            js_string.length();
        arr->data_array->set_null_bit<arr_arr_type>(idx, true);
    }
    CHECK_ARROW_MEM(
        arr->data_array->buffers[0]->SetSize(
            arr->data_array->data2<arr_arr_type, offset_t>()[idx + 1]),
        "append_v8_handle: SetSize failed")
    CHECK_ARROW_MEM(arr->data_array->buffers[1]->SetSize((arr->size + 1) *
                                                         sizeof(offset_t)),
                    "append_v8_handle: SetSize failed")
    CHECK_ARROW_MEM(arr->data_array->buffers[2]->SetSize(
                        arrow::bit_util::BytesForBits(arr->size + 1)),
                    "append_v8_handle: SetSize failed")
    arr->data_array->length += 1;
}

/**
 * @brief Append a v8::Value into an array_info, assumes arr has enough capacity
 * for the offsets, handles resizing the data buffer
 * @tparam arr_arr_type The type of the array_info
 * @tparam arr_c_type The C type of the array_info
 * @param ctx The v8 context
 * @param obj The v8 value to append
 * @param arr The ArrayBuildBuffer to append into
 * @param idx The index to append into
 * @param dict_builder The dictionary builder to use, only used for dict, can be
 * nullptr otherwise
 * This function is specialized for string arrays of string data
 */
template <bodo_array_type::arr_type_enum arr_arr_type,
          Bodo_CTypes::CTypeEnum arr_c_type>
    requires(arr_arr_type == bodo_array_type::STRING &&
             arr_c_type == Bodo_CTypes::BINARY)
void append_v8_handle(v8::Local<v8::Context> ctx, v8::Local<v8::Value> obj,
                      const std::shared_ptr<ArrayBuildBuffer> &arr,
                      v8::TryCatch &trycatch) {
    size_t idx = arr->size;
    if (obj->IsNullOrUndefined()) {
        arr->data_array->data2<arr_arr_type, offset_t>()[idx + 1] =
            arr->data_array->data2<arr_arr_type, offset_t>()[idx];
        arr->data_array->set_null_bit<arr_arr_type>(idx, false);
    } else {
        v8::Uint8Array *js_binary_arr = v8::Uint8Array::Cast(*obj);
        assert(js_binary_arr != nullptr);
        arr->ReserveSpaceForStringAppend(js_binary_arr->ByteLength());
        memcpy(arr->data_array->data1<arr_arr_type>() +
                   arr->data_array->data2<arr_arr_type, offset_t>()[idx],
               js_binary_arr->Buffer()->Data(),
               js_binary_arr->ByteLength() * sizeof(char));
        arr->data_array->data2<arr_arr_type, offset_t>()[idx + 1] =
            arr->data_array->data2<arr_arr_type, offset_t>()[idx] +
            js_binary_arr->ByteLength();
        arr->data_array->set_null_bit<arr_arr_type>(idx, true);
    }
    CHECK_ARROW_MEM(
        arr->data_array->buffers[0]->SetSize(
            arr->data_array->data2<arr_arr_type, offset_t>()[idx + 1]),
        "append_v8_handle: SetSize failed")
    CHECK_ARROW_MEM(arr->data_array->buffers[1]->SetSize((arr->size + 1) *
                                                         sizeof(offset_t)),
                    "append_v8_handle: SetSize failed")
    CHECK_ARROW_MEM(arr->data_array->buffers[2]->SetSize(
                        arrow::bit_util::BytesForBits(arr->size + 1)),
                    "append_v8_handle: SetSize failed")
    arr->data_array->length += 1;
}

/**
 * @brief Append a v8::Value into an array_info, assumes arr has enough capacity
 * @tparam arr_arr_type The type of the array_info
 * @tparam arr_c_type The C type of the array_info
 * @param ctx The v8 context
 * @param obj The v8 value to append
 * @param arr The ArrayBuildBuffer to append into
 * @param idx The index to append into
 * @param dict_builder The dictionary builder to use, only used for dict, can be
 * nullptr otherwise
 * This function is specialized for nullable arrays that are not boolean
 * It reuses the numpy implementations and adds null handling
 */
template <bodo_array_type::arr_type_enum arr_arr_type,
          Bodo_CTypes::CTypeEnum arr_c_type>
    requires(arr_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
             arr_c_type != Bodo_CTypes::_BOOL)
void append_v8_handle(v8::Local<v8::Context> ctx, v8::Local<v8::Value> obj,
                      const std::shared_ptr<ArrayBuildBuffer> &arr,
                      v8::TryCatch &trycatch) {
    size_t idx = arr->size;
    if (obj->IsNullOrUndefined()) {
        arr->data_array->set_null_bit<arr_arr_type>(idx, false);
        // Not calling the numpy append function because we need to set the null
        // so we need to resize here
        CHECK_ARROW_MEM(arr->data_array->buffers[0]->SetSize(
                            numpy_item_size[arr_c_type] * (arr->size + 1)),
                        "append_v8_handle: SetSize failed")
        arr->data_array->length += 1;
    } else {
        append_v8_handle<bodo_array_type::NUMPY, arr_c_type>(ctx, obj, arr,
                                                             trycatch);
        arr->data_array->set_null_bit<arr_arr_type>(idx, true);
    }
    // Only set size for null array, the rest are handled by the numpy function
    CHECK_ARROW_MEM(arr->data_array->buffers[1]->SetSize(
                        arrow::bit_util::BytesForBits(arr->size + 1)),
                    "append_v8_handle: SetSize failed")
}

/**
 * @brief Append a v8::Value into an array_info, assumes arr has enough capacity
 * @tparam arr_arr_type The type of the array_info
 * @tparam arr_c_type The C type of the array_info
 * @param ctx The v8 context
 * @param obj The v8 value to append
 * @param arr The ArrayBuildBuffer to append into
 * @param idx The index to append into
 * @param dict_builder The dictionary builder to use, only used for dict, can be
 * nullptr otherwise
 * This function is specialized for nullable arrays of bools, these
 * need special handling since the data array is a bitmap
 */
template <bodo_array_type::arr_type_enum arr_arr_type,
          Bodo_CTypes::CTypeEnum arr_c_type>
    requires(arr_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
             arr_c_type == Bodo_CTypes::_BOOL)
void append_v8_handle(v8::Local<v8::Context> ctx, v8::Local<v8::Value> obj,
                      const std::shared_ptr<ArrayBuildBuffer> &arr,
                      v8::TryCatch &trycatch) {
    size_t idx = arr->size;
    if (obj->IsNullOrUndefined()) {
        arr->data_array->set_null_bit<arr_arr_type>(idx, false);
    } else {
        auto js_val = obj->ToBoolean(ctx->GetIsolate());
        bool bit_is_set = js_val->Value();
        arrow::bit_util::SetBitTo(
            arr->data_array->data1<arr_arr_type, uint8_t>(), idx, bit_is_set);
        arr->data_array->set_null_bit<arr_arr_type>(idx, true);
    }
    CHECK_ARROW_MEM(arr->data_array->buffers[0]->SetSize(
                        arrow::bit_util::BytesForBits(arr->size + 1)),
                    "append_v8_handle: SetSize failed")
    CHECK_ARROW_MEM(arr->data_array->buffers[1]->SetSize(
                        arrow::bit_util::BytesForBits(arr->size + 1)),
                    "append_v8_handle: SetSize failed")
    arr->data_array->length += 1;
}

/**
 * @brief Append a v8::Value into an array_info, assumes arr has enough capacity
 * for the indices, handles resizing the dictionary if necessary
 * @tparam arr_arr_type The type of the array_info
 * @tparam arr_c_type The C type of the array_info
 * @param ctx The v8 context
 * @param obj The v8 value to append
 * @param arr The ArrayBuildBuffer to append into
 * @param idx The index to append into
 * @param dict_builder The dictionary builder to use, only used for dict, can be
 * nullptr otherwise
 * This function is specialized for dict arrays of strings and requires a valid
 * DictionaryBuilder. It adds the string to the dict if necessary and sets the
 * index of string in the dictionary in the index array
 */
template <bodo_array_type::arr_type_enum arr_arr_type,
          Bodo_CTypes::CTypeEnum arr_c_type>
    requires(arr_arr_type == bodo_array_type::DICT &&
             arr_c_type == Bodo_CTypes::STRING)
void append_v8_handle(v8::Local<v8::Context> ctx, v8::Local<v8::Value> obj,
                      const std::shared_ptr<ArrayBuildBuffer> &arr,
                      v8::TryCatch &trycatch) {
    size_t idx = arr->size;
    if (obj->IsNullOrUndefined()) {
        arr->data_array->set_null_bit<arr_arr_type>(idx, false);
    } else {
        auto js_string = v8::String::Utf8Value(ctx->GetIsolate(), obj);
        assert(*js_string != nullptr);
        // Append the string into the dictionary builder and get the index
        dict_indices_t dict_idx =
            arr->dict_builder->InsertIfNotExists(*js_string);
        // Append the index into the index array
        arr->data_array->child_arrays[1]
            ->data1<bodo_array_type::NULLABLE_INT_BOOL, dict_indices_t>()[idx] =
            dict_idx;
        // Set the null bit, the dict array just uses the index array's null bit
        arr->data_array->set_null_bit<arr_arr_type>(idx, true);
    }
    CHECK_ARROW_MEM(arr->data_array->child_arrays[1]->buffers[0]->SetSize(
                        (arr->size + 1) * sizeof(dict_indices_t)),
                    "append_v8_handle: SetSize failed")
    CHECK_ARROW_MEM(arr->data_array->child_arrays[1]->buffers[1]->SetSize(
                        arrow::bit_util::BytesForBits(arr->size + 1)),
                    "append_v8_handle: SetSize failed")
    arr->data_array->length += 1;
}

/**
 * @brief Append a v8::Value into an array_info, assumes arr has enough capacity
 * for the indices, handles resizing the dictionary if necessary
 * @tparam arr_arr_type The type of the array_info
 * @tparam arr_c_type The C type of the array_info
 * @param ctx The v8 context
 * @param obj The v8 value to append
 * @param arr The ArrayBuildBuffer to append into
 * @param idx The index to append into
 * @param dict_builder The dictionary builder to use, only used for dict, can be
 * nullptr otherwise
 * This function is specialized for dict arrays of binary data and requires a
 * valid DictionaryBuilder. It adds the string to the dict if necessary and sets
 * the index of string in the dictionary in the index array
 */
template <bodo_array_type::arr_type_enum arr_arr_type,
          Bodo_CTypes::CTypeEnum arr_c_type>
    requires(arr_arr_type == bodo_array_type::DICT &&
             arr_c_type == Bodo_CTypes::BINARY)
void append_v8_handle(v8::Local<v8::Context> ctx, v8::Local<v8::Value> obj,
                      const std::shared_ptr<ArrayBuildBuffer> &arr,
                      v8::TryCatch &trycatch) {
    size_t idx = arr->size;
    if (obj->IsNullOrUndefined()) {
        arr->data_array->set_null_bit<arr_arr_type>(idx, false);
    } else {
        v8::Uint8Array *uint8_arr = v8::Uint8Array::Cast(*obj);
        assert(uint8_arr != nullptr);
        // Append the string into the dictionary builder and get the index
        dict_indices_t dict_idx =
            arr->dict_builder->InsertIfNotExists(std::string_view(
                reinterpret_cast<char *>(uint8_arr->Buffer()->Data()),
                uint8_arr->Length()));
        // Append the index into the index array
        arr->data_array->child_arrays[1]
            ->data1<bodo_array_type::NULLABLE_INT_BOOL, dict_indices_t>()[idx] =
            dict_idx;
        // Set the null bit, the dict array just uses the index array's null bit
        arr->data_array->set_null_bit<arr_arr_type>(idx, true);
    }
    CHECK_ARROW_MEM(arr->data_array->child_arrays[1]->buffers[0]->SetSize(
                        (arr->size + 1) * sizeof(dict_indices_t)),
                    "append_v8_handle: SetSize failed")
    CHECK_ARROW_MEM(arr->data_array->child_arrays[1]->buffers[1]->SetSize(
                        arrow::bit_util::BytesForBits(arr->size + 1)),
                    "append_v8_handle: SetSize failed")
    arr->data_array->length += 1;
}
