#include "./test.hpp"
#ifdef BUILD_WITH_V8
#include "include/libplatform/libplatform.h"
#include "include/v8-context.h"
#include "include/v8-initialization.h"
#include "include/v8-isolate.h"
#include "include/v8-local-handle.h"
#include "include/v8-primitive.h"
#include "include/v8-script.h"
#endif

static bodo::tests::suite tests([] {
    bodo::tests::test("test_basic", [] {
        #ifdef BUILD_WITH_V8
        std::unique_ptr<v8::Platform> platform =
            v8::platform::NewDefaultPlatform();
        v8::V8::InitializePlatform(platform.get());
        v8::V8::Initialize();

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
        #endif
    });
});
