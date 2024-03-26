#include "_javascript_udf.h"
#include <fmt/format.h>
#include <random>
#include "_bodo_common.h"

#include "include/libplatform/libplatform.h"
#include "include/v8-function.h"
#include "include/v8-initialization.h"
#include "include/v8-isolate.h"
#include "include/v8-local-handle.h"
#include "include/v8-primitive.h"
#include "include/v8-script.h"

static std::unique_ptr<v8::Platform> platform;
static v8::Isolate::CreateParams create_params;
static std::shared_ptr<v8::Isolate> isolate;
static bool v8_initialized = false;

// @brief Create a new isolate wrapped in a shared pointer and set configure it
// for throughput over latency
// @return The new isolate
std::shared_ptr<v8::Isolate> create_new_isolate() {
    v8::Isolate *isolate = v8::Isolate::New(create_params);
    // Favor throughput over latency
    isolate->SetRAILMode(v8::PERFORMANCE_LOAD);
    // No delete, Isolate::Dispose should be used instead
    return std::shared_ptr<v8::Isolate>(
        isolate, [](v8::Isolate *isolate) { isolate->Dispose(); });
}

void init_v8() {
    if (!v8_initialized) {
        // Use a single threaded platform so we don't have contention
        platform = v8::platform::NewSingleThreadedDefaultPlatform();
        v8::V8::InitializePlatform(platform.get());

        v8::V8::Initialize();
        // Create a new Isolate and make it the current one, we'll only ever
        // need one.
        create_params.array_buffer_allocator_shared =
            std::shared_ptr<v8::ArrayBuffer::Allocator>(
                v8::ArrayBuffer::Allocator::NewDefaultAllocator());
        v8_initialized = true;
        isolate = create_new_isolate();
    }
}

std::unique_ptr<JavaScriptFunction> JavaScriptFunction::create(
    std::string _body, std::vector<std::string> _arg_names,
    std::unique_ptr<bodo::DataType> _return_type) {
    init_v8();
    v8::HandleScope handle_scope(isolate.get());
    std::unique_ptr<JavaScriptFunction> f = std::unique_ptr<JavaScriptFunction>(
        new JavaScriptFunction(_body, _arg_names, std::move(_return_type)));
    return f;
}

JavaScriptFunction::JavaScriptFunction(
    std::string _body, std::vector<std::string> _arg_names,
    std::unique_ptr<bodo::DataType> _return_type)
    : return_type(std::move(_return_type)),
      arg_names(_arg_names),
      context(isolate.get(), v8::Context::New(isolate.get())),
      tracing_event("JavaScript UDF") {
    v8::Isolate::Scope isolate_scope(isolate.get());
    // Create the handle scope and set the active context
    v8::HandleScope handle_scope(isolate.get());
    v8::Context::Scope context_scope(this->context.Get(isolate.get()));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(65, 90);
    char random_function_name_char_arr[this->size_rand_names];
    for (int i = 0; i < this->size_rand_names; i++) {
        random_function_name_char_arr[i] = distrib(gen);
    }
    this->random_function_name =
        std::string(random_function_name_char_arr, this->size_rand_names);

    std::string script_str =
        fmt::format("function {}({}) {{ {} }}", this->random_function_name,
                    fmt::join(this->arg_names, ", "), _body);
    v8::Local<v8::String> source =
        v8::String::NewFromUtf8(isolate.get(), script_str.c_str())
            .ToLocalChecked();
    v8::Local<v8::Script> script =
        v8::Script::Compile(this->context.Get(isolate.get()), source)
            .ToLocalChecked();
    script->Run(this->context.Get(isolate.get())).ToLocalChecked();

    v8::Local<v8::String> v8_func_name =
        v8::String::NewFromUtf8(isolate.get(),
                                this->random_function_name.c_str())
            .ToLocalChecked();
    v8::Local<v8::Context> local_context = this->context.Get(isolate.get());
    v8::Local<v8::Value> v8_func_value = local_context->Global()
                                             ->Get(local_context, v8_func_name)
                                             .ToLocalChecked();
    assert(v8_func_value->IsFunction());
    this->v8_func.Reset(isolate.get(),
                        v8::Local<v8::Function>::Cast(v8_func_value));
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
            arg_names_vec.push_back(std::string(
                arg_names->data1<bodo_array_type::STRING>()[i],
                ((offset_t *)
                     arg_names->data2<bodo_array_type::STRING>())[i + 1] -
                    ((offset_t *)
                         arg_names->data2<bodo_array_type::STRING>())[i]));
        }
        v8::Isolate::Scope isolate_scope(isolate.get());
        v8::HandleScope handle_scope(isolate.get());
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
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

std::shared_ptr<array_info> execute_javascript_udf(
    JavaScriptFunction *func, std::vector<std::unique_ptr<array_info>> args) {
    auto tracingEvent(func->tracing_event.iteration());
    // We don't support args for now.
    assert(args.size() == 0);
    v8::Isolate::Scope isolate_scope(isolate.get());
    // Create the handle scope and set the active context
    v8::HandleScope handle_scope(isolate.get());
    v8::Local<v8::Context> local_context = func->context.Get(isolate.get());
    v8::Context::Scope context_scope(local_context);
    v8::Local<v8::Function> local_v8_func = func->v8_func.Get(isolate.get());

    size_t out_count = args.size() == 0 ? 1 : args[0]->length;
    std::shared_ptr<array_info> ret =
        alloc_array_top_level(out_count, 0, 0, func->return_type->array_type,
                              func->return_type->c_type);
    for (size_t i = 0; i < out_count; ++i) {
        auto result =
            local_v8_func
                ->Call(local_context, local_context->Global(), 0, nullptr)
                .ToLocalChecked();
        int64_t retval = result->IntegerValue(local_context).ToChecked();
        // this needs to become something like
        // cast_v8_value_to_bodo_array<array_type, ctype>(result);
        reinterpret_cast<int64_t *>(ret->data1())[i] = retval;
    }

    return ret;
}

// Assumes length of args == func->arg_names.size()
array_info *execute_javascript_udf_py_entry(JavaScriptFunction *func,
                                            array_info **args) {
    try {
        std::vector<std::unique_ptr<array_info>> args_vec;
        args_vec.reserve(func->arg_names.size());
        for (size_t i = 0; i < func->arg_names.size(); ++i) {
            args_vec.push_back(std::unique_ptr<array_info>(args[i]));
        }
        return new array_info(
            *execute_javascript_udf(func, std::move(args_vec)));
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
