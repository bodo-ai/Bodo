#include <limits>
#include "../libs/_bodo_common.h"
#include "../libs/_javascript_udf.h"
#include "./test.hpp"
#include "include/v8-context.h"
#include "include/v8-date.h"
#include "include/v8-isolate.h"
#include "include/v8-local-handle.h"
#include "include/v8-primitive.h"
#include "include/v8-script.h"

static bodo::tests::suite tests([] {
    init_v8();
    bodo::tests::test("test_basic", [] {
        v8::Isolate::CreateParams create_params;
        create_params.array_buffer_allocator_shared =
            std::shared_ptr<v8::ArrayBuffer::Allocator>(
                v8::ArrayBuffer::Allocator::NewDefaultAllocator());
        v8::Isolate *isolate = v8::Isolate::New(create_params);
        {
            v8::Isolate::Scope isolate_scope(isolate);
            // Create a stack-allocated handle scope.
            v8::HandleScope handle_scope(isolate);
            // Create a new context.
            v8::Local<v8::Context> context = v8::Context::New(isolate);
            // Enter the context for compiling and running the hello world
            // script.
            v8::Context::Scope context_scope(context);
            // Create a string containing the JavaScript source code.
            v8::Local<v8::String> source =
                v8::String::NewFromUtf8Literal(isolate, "'Hello' + ', World!'");
            // Compile the source code.
            v8::Local<v8::Script> script =
                v8::Script::Compile(context, source).ToLocalChecked();
            // Run the script to get the result.
            v8::Local<v8::Value> result = script->Run(context).ToLocalChecked();
            // Convert the result to an UTF8 string and print it.
            v8::String::Utf8Value utf8(isolate, result);
            bodo::tests::check(std::string(*utf8) == "Hello, World!");
        }
        isolate->Dispose();
    });
    bodo::tests::test("test_basic_JavaScriptFunction", [] {
        auto f = JavaScriptFunction::create(
            "return 2", {},
            std::make_unique<bodo::DataType>(bodo_array_type::NULLABLE_INT_BOOL,
                                             Bodo_CTypes::INT64));
        auto out_arr = execute_javascript_udf(f.get(), {});
        bodo::tests::check(out_arr->data1()[0] == 2);
    });

    bodo::tests::test("test_javascript_to_bodo_conversion_string", [] {
        v8::Isolate::CreateParams create_params;
        create_params.array_buffer_allocator_shared =
            std::shared_ptr<v8::ArrayBuffer::Allocator>(
                v8::ArrayBuffer::Allocator::NewDefaultAllocator());
        v8::Isolate *isolate = v8::Isolate::New(create_params);
        {
            v8::Isolate::Scope isolate_scope(isolate);
            // Create a stack-allocated handle scope.
            v8::HandleScope handle_scope(isolate);
            // Create a new context.
            v8::Local<v8::Context> context = v8::Context::New(isolate);
            // Enter the context for compiling and running the hello world
            // script.
            v8::Context::Scope context_scope(context);

            v8::TryCatch trycatch(isolate);
            // Create a string
            v8::Local<v8::String> js_string =
                v8::String::NewFromUtf8Literal(isolate, "abcdef");
            // Create a bodo string array
            std::shared_ptr<array_info> bodo_str_array =
                alloc_string_array(Bodo_CTypes::STRING, 0, 0);
            // Create a string array builder
            std::shared_ptr<ArrayBuildBuffer> str_arr_builder =
                std::make_shared<ArrayBuildBuffer>(bodo_str_array);
            str_arr_builder->ReserveSize(4);
            // Convert the string to a bodo string array
            append_v8_handle<bodo_array_type::STRING, Bodo_CTypes::STRING>(
                context, js_string, str_arr_builder, trycatch);
            // Check the data, offsets and null bitmap
            bodo::tests::check(
                std::string(bodo_str_array->data1(),
                            bodo_str_array->data2<bodo_array_type::STRING,
                                                  offset_t>()[1]) == "abcdef");
            bodo::tests::check(
                bodo_str_array->data2<bodo_array_type::STRING, offset_t>()[1] ==
                6);
            bodo::tests::check(bodo_str_array->get_null_bit(0) == 1);

            // Add a second string
            js_string = v8::String::NewFromUtf8Literal(isolate, "ghijklop");
            append_v8_handle<bodo_array_type::STRING, Bodo_CTypes::STRING>(
                context, js_string, str_arr_builder, trycatch);
            // Check the data and offsets for both entries and the
            // nullbitmap for the second
            bodo::tests::check(
                std::string(
                    bodo_str_array->data1(),
                    bodo_str_array
                        ->data2<bodo_array_type::STRING, offset_t>()[2]) ==
                "abcdefghijklop");
            bodo::tests::check(
                bodo_str_array->data2<bodo_array_type::STRING, offset_t>()[1] ==
                6);
            bodo::tests::check(
                bodo_str_array->data2<bodo_array_type::STRING, offset_t>()[2] ==
                14);
            bodo::tests::check(bodo_str_array->get_null_bit(1) == 1);

            // Create a number
            v8::Local<v8::Number> js_number = v8::Number::New(isolate, 2);
            // Append the number to the string array to check casting
            append_v8_handle<bodo_array_type::STRING, Bodo_CTypes::STRING>(
                context, js_number, str_arr_builder, trycatch);
            // Check the data and offsets for all entries and the
            // nullbitmap for the third
            bodo::tests::check(
                std::string(
                    bodo_str_array->data1(),
                    bodo_str_array
                        ->data2<bodo_array_type::STRING, offset_t>()[3]) ==
                "abcdefghijklop2");
            bodo::tests::check(
                bodo_str_array->data2<bodo_array_type::STRING, offset_t>()[1] ==
                6);
            bodo::tests::check(
                bodo_str_array->data2<bodo_array_type::STRING, offset_t>()[2] ==
                14);
            bodo::tests::check(
                bodo_str_array->data2<bodo_array_type::STRING, offset_t>()[3] ==
                15);
            bodo::tests::check(bodo_str_array->get_null_bit(2) == 1);

            // Create a a null value
            v8::Local<v8::Value> js_null = v8::Null(isolate);
            // Append the null value to the string array to check casting
            append_v8_handle<bodo_array_type::STRING, Bodo_CTypes::STRING>(
                context, js_null, str_arr_builder, trycatch);
            // Check the nullbitmap for the fourth
            // and data and offsets for all entries
            bodo::tests::check(bodo_str_array->get_null_bit(3) == 0);
            bodo::tests::check(
                std::string(
                    bodo_str_array->data1(),
                    bodo_str_array
                        ->data2<bodo_array_type::STRING, offset_t>()[4]) ==
                "abcdefghijklop2");
            bodo::tests::check(
                bodo_str_array->data2<bodo_array_type::STRING, offset_t>()[1] ==
                6);
            bodo::tests::check(
                bodo_str_array->data2<bodo_array_type::STRING, offset_t>()[2] ==
                14);
            bodo::tests::check(
                bodo_str_array->data2<bodo_array_type::STRING, offset_t>()[3] ==
                15);
            bodo::tests::check(
                bodo_str_array->data2<bodo_array_type::STRING, offset_t>()[4] ==
                15);

            bodo::tests::check(bodo_str_array->length == 4);
            bodo::tests::check(str_arr_builder->size == 4);
        }
        isolate->Dispose();
    });
    bodo::tests::test("test_javascript_to_bodo_conversion_dict_string", [] {
        v8::Isolate::CreateParams create_params;
        create_params.array_buffer_allocator_shared =
            std::shared_ptr<v8::ArrayBuffer::Allocator>(
                v8::ArrayBuffer::Allocator::NewDefaultAllocator());
        v8::Isolate *isolate = v8::Isolate::New(create_params);
        {
            v8::Isolate::Scope isolate_scope(isolate);
            // Create a stack-allocated handle scope.
            v8::HandleScope handle_scope(isolate);
            // Create a new context.
            v8::Local<v8::Context> context = v8::Context::New(isolate);
            // Enter the context for compiling and running the hello world
            // script.
            v8::Context::Scope context_scope(context);

            v8::TryCatch trycatch(isolate);
            // Create a string
            v8::Local<v8::String> js_string =
                v8::String::NewFromUtf8Literal(isolate, "abcdef");
            // Create a bodo dictionary string array
            std::shared_ptr<array_info> bodo_dict_array =
                alloc_dict_string_array(0, 0, 0);
            std::shared_ptr<array_info> dictionary =
                bodo_dict_array->child_arrays[0];
            std::shared_ptr<array_info> idx_array =
                bodo_dict_array->child_arrays[1];
            // Create a dict builder
            std::shared_ptr<DictionaryBuilder> dict_builder =
                std::make_shared<DictionaryBuilder>(dictionary, false);
            // Create a dict array builder
            std::shared_ptr<ArrayBuildBuffer> dict_arr_builder =
                std::make_shared<ArrayBuildBuffer>(bodo_dict_array,
                                                   dict_builder);
            dict_arr_builder->ReserveSize(4);
            // Read the string into the dict array
            append_v8_handle<bodo_array_type::DICT, Bodo_CTypes::STRING>(
                context, js_string, dict_arr_builder, trycatch);
            // Check the data, offsets and null bitmap
            bodo::tests::check(
                std::string(dictionary->data1(),
                            dictionary->data2<bodo_array_type::STRING,
                                              offset_t>()[1]) == "abcdef");
            bodo::tests::check(
                reinterpret_cast<offset_t *>(
                    dictionary
                        ->data2<bodo_array_type::STRING, offset_t>())[1] == 6);
            bodo::tests::check(bodo_dict_array->get_null_bit(0) == 1);
            bodo::tests::check(((dict_indices_t *)idx_array->data1())[0] == 0);
            // Insert a different string
            js_string = v8::String::NewFromUtf8Literal(isolate, "ghijklop");
            append_v8_handle<bodo_array_type::DICT, Bodo_CTypes::STRING>(
                context, js_string, dict_arr_builder, trycatch);
            // Check the data, offsets and null bitmap
            bodo::tests::check(
                std::string(
                    dictionary->data1(),
                    dictionary
                        ->data2<bodo_array_type::STRING, offset_t>()[2]) ==
                "abcdefghijklop");
            bodo::tests::check(
                reinterpret_cast<offset_t *>(
                    dictionary
                        ->data2<bodo_array_type::STRING, offset_t>())[2] == 14);
            bodo::tests::check(bodo_dict_array->get_null_bit(1) == 1);
            bodo::tests::check(((dict_indices_t *)idx_array->data1())[1] == 1);
            // Insert the first string again, to check the dictionary
            // builder
            js_string = v8::String::NewFromUtf8Literal(isolate, "abcdef");
            append_v8_handle<bodo_array_type::DICT, Bodo_CTypes::STRING>(
                context, js_string, dict_arr_builder, trycatch);
            // Check the data, offsets and null bitmap
            bodo::tests::check(
                std::string(
                    dictionary->data1(),
                    dictionary
                        ->data2<bodo_array_type::STRING, offset_t>()[2]) ==
                "abcdefghijklop");
            bodo::tests::check(
                reinterpret_cast<offset_t *>(
                    dictionary
                        ->data2<bodo_array_type::STRING, offset_t>())[2] == 14);
            bodo::tests::check(bodo_dict_array->get_null_bit(2) == 1);
            bodo::tests::check(((dict_indices_t *)idx_array->data1())[2] == 0);
            bodo::tests::check(bodo_dict_array->length == 3);
        }
        isolate->Dispose();
    });
    auto big_int64_test_func = []<Bodo_CTypes::CTypeEnum ctype> {
        return [] {
            v8::Isolate::CreateParams create_params;
            create_params.array_buffer_allocator_shared =
                std::shared_ptr<v8::ArrayBuffer::Allocator>(
                    v8::ArrayBuffer::Allocator::NewDefaultAllocator());
            v8::Isolate *isolate = v8::Isolate::New(create_params);
            {
                using val_t = typename dtype_to_type<ctype>::type;
                v8::Isolate::Scope isolate_scope(isolate);
                // Create a stack-allocated handle scope.
                v8::HandleScope handle_scope(isolate);
                // Create a new context.
                v8::Local<v8::Context> context = v8::Context::New(isolate);
                // Enter the context for compiling and running the hello world
                // script.
                v8::Context::Scope context_scope(context);

                v8::TryCatch trycatch(isolate);
                // Create a bigint that fits in one word
                val_t max = std::numeric_limits<val_t>::max();
                v8::Local<v8::BigInt> js_bigint =
                    is_unsigned_integer(ctype)
                        ? v8::BigInt::NewFromUnsigned(isolate, max)
                        : v8::BigInt::New(isolate, max);
                // Create a bodo int array
                std::shared_ptr<array_info> bodo_int_array =
                    alloc_nullable_array_all_nulls(0, ctype, 0);
                // Create a int array builder
                std::shared_ptr<ArrayBuildBuffer> int_arr_builder =
                    std::make_shared<ArrayBuildBuffer>(bodo_int_array);
                int_arr_builder->ReserveSize(4);
                // Convert the bigint to a bodo int array
                append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL, ctype>(
                    context, js_bigint, int_arr_builder, trycatch);
                // Check the data and null bitmap
                bodo::tests::check(
                    bodo_int_array->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                          val_t>()[0] == max);
                bodo::tests::check(bodo_int_array->get_null_bit(0) == 1);
                // Create a numeric string
                v8::Local<v8::String> js_string =
                    v8::String::NewFromUtf8Literal(isolate,
                                                   "1234567890123456789");
                // Convert the string to a bodo uint64 array
                append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL, ctype>(
                    context, js_string, int_arr_builder, trycatch);
                // Check the data and null bitmap
                bodo::tests::check(
                    bodo_int_array->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                          val_t>()[1] == 1234567890123456789);
                bodo::tests::check(bodo_int_array->get_null_bit(1) == 1);
                // Create the min bigint
                val_t min = std::numeric_limits<val_t>::min();
                js_bigint = is_unsigned_integer(ctype)
                                ? v8::BigInt::NewFromUnsigned(isolate, min)
                                : v8::BigInt::New(isolate, min);
                // Convert the bigint to a bodo int array
                append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL, ctype>(
                    context, js_bigint, int_arr_builder, trycatch);
                // Check the data and null bitmap
                bodo::tests::check(
                    bodo_int_array->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                          val_t>()[2] == min);
                bodo::tests::check(bodo_int_array->get_null_bit(2) == 1);
                bodo::tests::check(bodo_int_array->length == 3);

                // Try an invalid string
                js_string = v8::String::NewFromUtf8Literal(isolate, "abc");
                // Convert the string to a bodo int array
                bodo::tests::check_exception(
                    [&] {
                        append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL,
                                         ctype>(context, js_string,
                                                int_arr_builder, trycatch);
                    },
                    "append_v8_handle: ToBigInt failed");
            }
            isolate->Dispose();
        };
    };

    bodo::tests::test("test_javascript_to_bodo_conversion_uint64",
                      big_int64_test_func.operator()<Bodo_CTypes::UINT64>());
    bodo::tests::test("test_javascript_to_bodo_conversion_int64",
                      big_int64_test_func.operator()<Bodo_CTypes::INT64>());

    auto int_test_func = []<Bodo_CTypes::CTypeEnum ctype> {
        return [] {
            v8::Isolate::CreateParams create_params;
            create_params.array_buffer_allocator_shared =
                std::shared_ptr<v8::ArrayBuffer::Allocator>(
                    v8::ArrayBuffer::Allocator::NewDefaultAllocator());
            v8::Isolate *isolate = v8::Isolate::New(create_params);
            {
                using val_t = typename dtype_to_type<ctype>::type;
                v8::Isolate::Scope isolate_scope(isolate);
                // Create a stack-allocated handle scope.
                v8::HandleScope handle_scope(isolate);
                // Create a new context.
                v8::Local<v8::Context> context = v8::Context::New(isolate);
                // Enter the context for compiling and running the hello world
                // script.
                v8::Context::Scope context_scope(context);

                v8::TryCatch trycatch(isolate);
                // Create a number
                v8::Local<v8::Number> js_number = v8::Number::New(isolate, 2);
                // Create a bodo int16 array
                std::shared_ptr<array_info> bodo_int_array =
                    alloc_nullable_array_all_nulls(0, ctype, 0);
                // Create a int16 array builder
                std::shared_ptr<ArrayBuildBuffer> int_arr_builder =
                    std::make_shared<ArrayBuildBuffer>(bodo_int_array);
                int_arr_builder->ReserveSize(5);
                // Convert the number to a bodo int16 array
                append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL, ctype>(
                    context, js_number, int_arr_builder, trycatch);
                // Check the data and null bitmap
                bodo::tests::check(
                    bodo_int_array->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                          val_t>()[0] == 2);
                bodo::tests::check(bodo_int_array->get_null_bit(0) == 1);
                // Create a string
                v8::Local<v8::String> js_string =
                    v8::String::NewFromUtf8(
                        isolate, is_unsigned_integer(ctype) ? "2" : "-2")
                        .ToLocalChecked();
                // Convert the string to a bodo int16 array
                append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL, ctype>(
                    context, js_string, int_arr_builder, trycatch);
                // Check the data and null bitmap
                int check_int = is_unsigned_integer(ctype) ? 2 : -2;
                bodo::tests::check(
                    bodo_int_array->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                          val_t>()[1] == check_int);
                bodo::tests::check(bodo_int_array->get_null_bit(1) == 1);
                // Create a null value
                v8::Local<v8::Value> js_null = v8::Null(isolate);
                // Convert the null value to a bodo int16 array
                append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL, ctype>(
                    context, js_null, int_arr_builder, trycatch);
                // Check the null bitmap
                bodo::tests::check(bodo_int_array->get_null_bit(2) == 0);
                // Add the numeric max and min
                js_number =
                    v8::Number::New(isolate, std::numeric_limits<val_t>::max());
                append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL, ctype>(
                    context, js_number, int_arr_builder, trycatch);
                js_number =
                    v8::Number::New(isolate, std::numeric_limits<val_t>::min());
                append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL, ctype>(
                    context, js_number, int_arr_builder, trycatch);
                // Check the data and null bitmap
                bodo::tests::check(
                    bodo_int_array->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                          val_t>()[3] ==
                    std::numeric_limits<val_t>::max());
                bodo::tests::check(bodo_int_array->get_null_bit(3) == 1);
                bodo::tests::check(
                    bodo_int_array->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                          val_t>()[4] ==
                    std::numeric_limits<val_t>::min());
                bodo::tests::check(bodo_int_array->get_null_bit(4) == 1);
                bodo::tests::check(bodo_int_array->length == 5);
            }
            isolate->Dispose();
        };
    };
    bodo::tests::test("test_javascript_to_bodo_conversion_uint8",
                      int_test_func.operator()<Bodo_CTypes::UINT8>());
    bodo::tests::test("test_javascript_to_bodo_conversion_uint16",
                      int_test_func.operator()<Bodo_CTypes::UINT16>());
    bodo::tests::test("test_javascript_to_bodo_conversion_uint32",
                      int_test_func.operator()<Bodo_CTypes::UINT32>());
    bodo::tests::test("test_javascript_to_bodo_conversion_int8",
                      int_test_func.operator()<Bodo_CTypes::INT8>());
    bodo::tests::test("test_javascript_to_bodo_conversion_int16",
                      int_test_func.operator()<Bodo_CTypes::INT16>());
    bodo::tests::test("test_javascript_to_bodo_conversion_int32",
                      int_test_func.operator()<Bodo_CTypes::INT32>());

    bodo::tests::test("test_javascript_to_bodo_conversion_float64", [] {
        v8::Isolate::CreateParams create_params;
        create_params.array_buffer_allocator_shared =
            std::shared_ptr<v8::ArrayBuffer::Allocator>(
                v8::ArrayBuffer::Allocator::NewDefaultAllocator());
        v8::Isolate *isolate = v8::Isolate::New(create_params);
        {
            v8::Isolate::Scope isolate_scope(isolate);
            // Create a stack-allocated handle scope.
            v8::HandleScope handle_scope(isolate);
            // Create a new context.
            v8::Local<v8::Context> context = v8::Context::New(isolate);
            // Enter the context for compiling and running the hello world
            // script.
            v8::Context::Scope context_scope(context);

            v8::TryCatch trycatch(isolate);
            // Create a number
            v8::Local<v8::Number> js_number = v8::Number::New(isolate, 2.2);
            // Create a bodo float64 array
            std::shared_ptr<array_info> bodo_float64_array =
                alloc_nullable_array_all_nulls(0, Bodo_CTypes::FLOAT64, 0);
            // Create a float64 array builder
            std::shared_ptr<ArrayBuildBuffer> float64_arr_builder =
                std::make_shared<ArrayBuildBuffer>(bodo_float64_array);
            float64_arr_builder->ReserveSize(4);
            // Convert the number to a bodo float64 array
            append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::FLOAT64>(
                context, js_number, float64_arr_builder, trycatch);
            // Check the data and null bitmap
            double bodo_val =
                bodo_float64_array
                    ->data1<bodo_array_type::NULLABLE_INT_BOOL, double>()[0];
            double expected_val = 2.2;
            bodo::tests::check(
                (fabs(bodo_val - expected_val) <=
                 std::numeric_limits<double>::epsilon() *
                     std::max<double>(fabs(bodo_val), fabs(expected_val))));
            bodo::tests::check(bodo_float64_array->get_null_bit(0) == 1);
            // Create a string
            v8::Local<v8::String> js_string =
                v8::String::NewFromUtf8Literal(isolate, "-2.2");
            // Convert the string to a bodo float64 array
            append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::FLOAT64>(
                context, js_string, float64_arr_builder, trycatch);
            // Check the data and null bitmap
            bodo_val =
                bodo_float64_array
                    ->data1<bodo_array_type::NULLABLE_INT_BOOL, double>()[1];
            expected_val = -2.2;
            bodo::tests::check(
                (fabs(bodo_val - expected_val) <=
                 std::numeric_limits<double>::epsilon() *
                     std::max<double>(fabs(bodo_val), fabs(expected_val))));
            bodo::tests::check(bodo_float64_array->get_null_bit(1) == 1);
            // Create a null value
            v8::Local<v8::Value> js_null = v8::Null(isolate);
            // Convert the null value to a bodo float64 array
            append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::FLOAT64>(
                context, js_null, float64_arr_builder, trycatch);
            // Check the null bitmap
            bodo::tests::check(bodo_float64_array->get_null_bit(2) == 0);
            bodo::tests::check(bodo_float64_array->length == 3);
        }
        isolate->Dispose();
    });

    bodo::tests::test("test_javascript_to_bodo_conversion_numpy_bool", [] {
        v8::Isolate::CreateParams create_params;
        create_params.array_buffer_allocator_shared =
            std::shared_ptr<v8::ArrayBuffer::Allocator>(
                v8::ArrayBuffer::Allocator::NewDefaultAllocator());
        v8::Isolate *isolate = v8::Isolate::New(create_params);
        {
            v8::Isolate::Scope isolate_scope(isolate);
            // Create a stack-allocated handle scope.
            v8::HandleScope handle_scope(isolate);
            // Create a new context.
            v8::Local<v8::Context> context = v8::Context::New(isolate);
            // Enter the context for compiling and running the hello world
            // script.
            v8::Context::Scope context_scope(context);

            v8::TryCatch trycatch(isolate);
            // Create a boolean
            v8::Local<v8::Boolean> js_bool = v8::Boolean::New(isolate, true);
            // Create a bodo numpy bool array
            std::shared_ptr<array_info> bodo_numpy_bool_array =
                alloc_numpy(0, Bodo_CTypes::_BOOL);
            // Create a numpy bool array builder
            std::shared_ptr<ArrayBuildBuffer> numpy_bool_arr_builder =
                std::make_shared<ArrayBuildBuffer>(bodo_numpy_bool_array);
            numpy_bool_arr_builder->ReserveSize(4);
            // Convert the boolean to a bodo numpy bool array
            append_v8_handle<bodo_array_type::NUMPY, Bodo_CTypes::_BOOL>(
                context, js_bool, numpy_bool_arr_builder, trycatch);
            // Check the data
            bodo::tests::check(bodo_numpy_bool_array
                                   ->data1<bodo_array_type::NUMPY, bool>()[0] ==
                               true);
            // Create a string
            v8::Local<v8::String> js_string =
                v8::String::NewFromUtf8Literal(isolate, "false");
            // Convert the string to a bodo numpy bool array
            append_v8_handle<bodo_array_type::NUMPY, Bodo_CTypes::_BOOL>(
                context, js_string, numpy_bool_arr_builder, trycatch);
            // Check the data
            bodo::tests::check(bodo_numpy_bool_array
                                   ->data1<bodo_array_type::NUMPY, bool>()[1] ==
                               true);
            // Create a number
            v8::Local<v8::Number> js_number = v8::Number::New(isolate, 0);
            // Convert the number to a bodo numpy bool array
            append_v8_handle<bodo_array_type::NUMPY, Bodo_CTypes::_BOOL>(
                context, js_number, numpy_bool_arr_builder, trycatch);
            // Check the data
            bodo::tests::check(bodo_numpy_bool_array
                                   ->data1<bodo_array_type::NUMPY, bool>()[2] ==
                               false);
            bodo::tests::check(bodo_numpy_bool_array->length == 3);
        }
        isolate->Dispose();
    });

    bodo::tests::test("test_javascript_to_bodo_conversion_nullable_bool", [] {
        v8::Isolate::CreateParams create_params;
        create_params.array_buffer_allocator_shared =
            std::shared_ptr<v8::ArrayBuffer::Allocator>(
                v8::ArrayBuffer::Allocator::NewDefaultAllocator());
        v8::Isolate *isolate = v8::Isolate::New(create_params);
        {
            v8::Isolate::Scope isolate_scope(isolate);
            // Create a stack-allocated handle scope.
            v8::HandleScope handle_scope(isolate);
            // Create a new context.
            v8::Local<v8::Context> context = v8::Context::New(isolate);
            // Enter the context for compiling and running the hello world
            // script.
            v8::Context::Scope context_scope(context);

            v8::TryCatch trycatch(isolate);
            // Create a boolean
            v8::Local<v8::Boolean> js_bool = v8::Boolean::New(isolate, true);
            // Create a bodo nullable bool array
            std::shared_ptr<array_info> bodo_nullable_bool_array =
                alloc_nullable_array_all_nulls(0, Bodo_CTypes::_BOOL, 0);
            // Create a nullable bool array builder
            std::shared_ptr<ArrayBuildBuffer> nullable_bool_arr_builder =
                std::make_shared<ArrayBuildBuffer>(bodo_nullable_bool_array);
            nullable_bool_arr_builder->ReserveSize(4);
            // Convert the boolean to a bodo nullable bool array
            append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::_BOOL>(
                context, js_bool, nullable_bool_arr_builder, trycatch);
            // Check the data
            bodo::tests::check(
                arrow::bit_util::GetBit(
                    bodo_nullable_bool_array
                        ->data1<bodo_array_type::NULLABLE_INT_BOOL, uint8_t>(),
                    0) == true);
            // Create a string
            v8::Local<v8::String> js_string =
                v8::String::NewFromUtf8Literal(isolate, "false");
            // Convert the string to a bodo nullable bool array
            append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::_BOOL>(
                context, js_string, nullable_bool_arr_builder, trycatch);
            // Check the data
            bodo::tests::check(
                arrow::bit_util::GetBit(
                    bodo_nullable_bool_array
                        ->data1<bodo_array_type::NULLABLE_INT_BOOL, uint8_t>(),
                    1) == true);
            // Create a number
            v8::Local<v8::Number> js_number = v8::Number::New(isolate, 0);
            // Convert the number to a bodo nullable bool array
            append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::_BOOL>(
                context, js_number, nullable_bool_arr_builder, trycatch);
            // Check the data
            bodo::tests::check(
                arrow::bit_util::GetBit(
                    bodo_nullable_bool_array
                        ->data1<bodo_array_type::NULLABLE_INT_BOOL, uint8_t>(),
                    2) == false);
            bodo::tests::check(bodo_nullable_bool_array->length == 3);
        }
        isolate->Dispose();
    });

    bodo::tests::test("test_javascript_to_bodo_conversion_date", [] {
        v8::Isolate::CreateParams create_params;
        create_params.array_buffer_allocator_shared =
            std::shared_ptr<v8::ArrayBuffer::Allocator>(
                v8::ArrayBuffer::Allocator::NewDefaultAllocator());
        v8::Isolate *isolate = v8::Isolate::New(create_params);
        {
            v8::Isolate::Scope isolate_scope(isolate);
            // Create a stack-allocated handle scope.
            v8::HandleScope handle_scope(isolate);
            // Create a new context.
            v8::Local<v8::Context> context = v8::Context::New(isolate);
            // Enter the context for compiling and running the hello world
            // script.
            v8::Context::Scope context_scope(context);

            v8::TryCatch trycatch(isolate);
            // Create a date Monday, May 3, 2021 12:00:00 AM
            v8::Local<v8::Value> js_date =
                v8::Date::New(context, 1620000000000).ToLocalChecked();
            // Create a bodo nullable date array
            std::shared_ptr<array_info> bodo_nullable_date_array =
                alloc_nullable_array_all_nulls(0, Bodo_CTypes::DATE, 0);
            // Create a nullable bool array builder
            std::shared_ptr<ArrayBuildBuffer> nullable_date_arr_builder =
                std::make_shared<ArrayBuildBuffer>(bodo_nullable_date_array);
            nullable_date_arr_builder->ReserveSize(4);
            using val_t = typename dtype_to_type<Bodo_CTypes::DATE>::type;
            // Convert the dateean to a bodo nullable bool array
            append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::DATE>(
                context, js_date, nullable_date_arr_builder, trycatch);
            // Check the data
            bodo::tests::check(
                bodo_nullable_date_array
                    ->data1<bodo_array_type::NULLABLE_INT_BOOL, val_t>()[0] ==
                18750);
            // Create a number
            v8::Local<v8::Number> js_number =
                v8::Number::New(isolate, 1620000000000);
            // Convert the number to a bodo nullable date array
            append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::DATE>(
                context, js_number, nullable_date_arr_builder, trycatch);
            // Check the data
            bodo::tests::check(
                bodo_nullable_date_array
                    ->data1<bodo_array_type::NULLABLE_INT_BOOL, val_t>()[1] ==
                18750);
            // Create a null value
            v8::Local<v8::Value> js_null = v8::Null(isolate);
            // Convert the null value to a bodo nullable date array
            append_v8_handle<bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::DATE>(
                context, js_null, nullable_date_arr_builder, trycatch);
            // Check the null bitmap is set only for the first two bits
            bodo::tests::check(bodo_nullable_date_array->get_null_bit(0) == 1);
            bodo::tests::check(bodo_nullable_date_array->get_null_bit(1) == 1);
            bodo::tests::check(bodo_nullable_date_array->get_null_bit(2) == 0);

            bodo::tests::check(bodo_nullable_date_array->length == 3);
        }
        isolate->Dispose();
    });
});
