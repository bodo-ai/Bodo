#include "../libs/_bodo_common.h"
#include "./test.hpp"
#include "../libs/_javascript_udf.h"
#include "include/v8-context.h"
#include "include/v8-isolate.h"
#include "include/v8-local-handle.h"
#include "include/v8-primitive.h"
#include "include/v8-script.h"

static bodo::tests::suite tests([] {
    bodo::tests::test("test_basic", [] {
        init_v8();

        // Create a new Isolate and make it the current one.
        v8::Isolate::CreateParams create_params;
        create_params.array_buffer_allocator =
            v8::ArrayBuffer::Allocator::NewDefaultAllocator();
        v8::Isolate* isolate = v8::Isolate::New(create_params);
        {
            v8::Isolate::Scope isolate_scope(isolate);
            // Create a stack-allocated handle scope.
            v8::HandleScope handle_scope(isolate);
            // Create a new context.
            v8::Local<v8::Context> context = v8::Context::New(isolate);
            // Enter the context for compiling and running the hello world
            // script.
            v8::Context::Scope context_scope(context);
            {
                // Create a string containing the JavaScript source code.
                v8::Local<v8::String> source = v8::String::NewFromUtf8Literal(
                    isolate, "'Hello' + ', World!'");
                // Compile the source code.
                v8::Local<v8::Script> script =
                    v8::Script::Compile(context, source).ToLocalChecked();
                // Run the script to get the result.
                v8::Local<v8::Value> result =
                    script->Run(context).ToLocalChecked();
                // Convert the result to an UTF8 string and print it.
                v8::String::Utf8Value utf8(isolate, result);
                bodo::tests::check(std::string(*utf8) == "Hello, World!");
            }
        }
        isolate->Dispose();
        delete create_params.array_buffer_allocator;
    });
    bodo::tests::test("test_basic_JavaScriptFunction", [] {
        auto f = JavaScriptFunction::create(
            "return 2", {},
            std::make_unique<bodo::DataType>(bodo_array_type::NULLABLE_INT_BOOL,
                                             Bodo_CTypes::INT64));
        auto out_arr = execute_javascript_udf(f.get(), {});
        bodo::tests::check(out_arr->data1()[0] == 2);
    });
});
