#include "../libs/_bodo_common.h"
#include "../libs/_dict_builder.h"
#include "../libs/_memory.h"
#include "../libs/_table_builder.h"
#include "./test.hpp"

bodo::tests::suite dict_builder_tests([] {
    bodo::tests::test("test_dict_builder_transpose_multi_cache", [] {
        // Test that the transpose cache works correctly when
        // multiple arrays are cached, even when a dict is used after an
        // intermediate.
        std::shared_ptr<array_info> dict =
            alloc_array(0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
        DictionaryBuilder dict_builder = DictionaryBuilder(dict, false, 2);

        std::shared_ptr<array_info> arr = alloc_dict_string_array(0, 0, 0);
        std::shared_ptr<array_info> arr_dict = arr->child_arrays[0];
        arr_dict->array_id = 1;

        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_misses == 1);
        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_misses == 1);
        arr_dict->array_id = 2;
        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_misses == 2);
        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_misses == 2);
        arr_dict->array_id = 1;
        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_misses == 2);
    });
    bodo::tests::test("test_dict_builder_transpose_lru_cache", [] {
        // Test that the transpose cache evicts the least recently used
        // entry.
        std::shared_ptr<array_info> dict =
            alloc_array(0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
        DictionaryBuilder dict_builder = DictionaryBuilder(dict, false, 2);

        std::shared_ptr<array_info> arr = alloc_dict_string_array(0, 0, 0);
        std::shared_ptr<array_info> arr_dict = arr->child_arrays[0];
        arr_dict->length = 1;
        int64_t first_id = generate_array_id(1);
        int64_t second_id = generate_array_id(1);
        int64_t third_id = generate_array_id(1);

        arr_dict->array_id = first_id;
        dict_builder.UnifyDictionaryArray(arr);
        arr_dict->array_id = second_id;
        dict_builder.UnifyDictionaryArray(arr);
        arr_dict->array_id = third_id;
        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_misses == 3);

        // Ensure the second array was cached and the first was evicted.
        arr_dict->array_id = second_id;
        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_misses == 3);
        arr_dict->array_id = first_id;
        dict_builder.UnifyDictionaryArray(arr);
        std::cout << dict_builder.unify_cache_misses << std::endl;
        bodo::tests::check(dict_builder.unify_cache_misses == 4);
    });
});
