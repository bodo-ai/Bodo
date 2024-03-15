#include "../libs/_bodo_common.h"
#include "../libs/_dict_builder.h"
#include "../libs/_memory.h"
#include "./table_generator.hpp"
#include "./test.hpp"

static std::shared_ptr<table_info> strVecToTable(
    std::string column_name, const std::vector<std::string>& input_column,
    bool dict_encoded = true) {
    std::set<std::string> dict_encoded_cols;
    if (dict_encoded) {
        dict_encoded_cols.insert(column_name);
    }
    return bodo::tests::cppToBodo({column_name}, {false}, dict_encoded_cols,
                                  input_column);
};

bodo::tests::suite dict_builder_tests([] {
    bodo::tests::test("test_dict_builder_transpose_multi_cache", [] {
        // Test that the transpose cache works correctly when
        // multiple arrays are cached, even when a dict is used after an
        // intermediate.
        std::shared_ptr<array_info> dict = alloc_array_top_level(
            0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
        DictionaryBuilder dict_builder = DictionaryBuilder(dict, false, {}, 2);

        std::shared_ptr<array_info> arr = alloc_dict_string_array(0, 0, 0);
        std::shared_ptr<array_info> arr_dict = arr->child_arrays[0];
        arr_dict->array_id = 1;

        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_id_misses == 1);
        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_id_misses == 1);
        arr_dict->array_id = 2;
        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_id_misses == 2);
        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_id_misses == 2);
        arr_dict->array_id = 1;
        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_id_misses == 2);
    });
    bodo::tests::test("test_dict_builder_transpose_lru_cache", [] {
        // Test that the transpose cache evicts the least recently used
        // entry.
        std::shared_ptr<array_info> dict = alloc_array_top_level(
            0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
        DictionaryBuilder dict_builder = DictionaryBuilder(dict, false, {}, 2);

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
        bodo::tests::check(dict_builder.unify_cache_id_misses == 3);

        // Ensure the second array was cached and the first was evicted.
        arr_dict->array_id = second_id;
        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_id_misses == 3);
        arr_dict->array_id = first_id;
        dict_builder.UnifyDictionaryArray(arr);
        bodo::tests::check(dict_builder.unify_cache_id_misses == 4);

        bodo::tests::check(dict_builder.unify_cache_length_misses == 0);
    });

    bodo::tests::test("test_dict_builder_id_replacement", [] {
        std::shared_ptr<array_info> dict = alloc_array_top_level(
            0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
        DictionaryBuilder dict_builder = DictionaryBuilder(dict, false, {}, 2);
        // Assert that we start with the empty dictionary id
        bodo::tests::check(dict_builder.dict_buff->data_array->array_id == 0);

        std::vector<std::string> input_column = {
            "Str1", "Str1", "Str1", "Str1", "Str2", "Str2", "Str2", "Str2",
        };
        const std::string column_name = "A";
        auto table = strVecToTable(column_name, input_column);

        dict_builder.UnifyDictionaryArray(table->columns[0]);
        bodo::tests::check(dict_builder.unify_cache_id_misses == 1);
        auto curr_id = dict_builder.dict_buff->data_array->array_id;
        bodo::tests::check(curr_id > 0);

        dict_builder.UnifyDictionaryArray(table->columns[0]);
        bodo::tests::check(dict_builder.unify_cache_id_misses == 1);
        // assert that the id does not change on append with previously appended
        // data
        bodo::tests::check(dict_builder.dict_buff->data_array->array_id ==
                           curr_id);

        std::vector<std::string> input_column_1 = {
            "Str2", "Str2", "Str2", "Str2", "Str3", "Str3", "Str3", "Str3",
        };
        auto table_1 = strVecToTable(column_name, input_column_1);
        dict_builder.UnifyDictionaryArray(table_1->columns[0]);
        bodo::tests::check(dict_builder.unify_cache_id_misses == 2);
        auto new_id = dict_builder.dict_buff->data_array->array_id;
        // assert that the id does not change on append with new data
        bodo::tests::check(new_id == curr_id);

        bodo::tests::check(dict_builder.unify_cache_length_misses == 0);
    });
    bodo::tests::test("test_dict_builder_same_input_different_length", [] {
        /*
         * This test will create two dictionary builders (src, dst), append data
         * to one of them (src), and the unify into the second dictionary. Then
         * the test will append more data to the first dictionary, and redo the
         * unification. This should cause a cache miss because the length of the
         * dictionary has changed.
         */

        DictionaryBuilder dict_builder_dst = DictionaryBuilder(
            alloc_array_top_level(0, 0, 0, bodo_array_type::STRING,
                                  Bodo_CTypes::STRING),
            false, {}, 2);

        std::shared_ptr<array_info> src_data = alloc_array_top_level(
            0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
        DictionaryBuilder dict_builder_src =
            DictionaryBuilder(src_data, false, {}, 2);

        // In order to unify the dictionaries, we need to create a dict encoded
        // array that uses the dictionary being built by dict_builder_src
        std::shared_ptr<array_info> indices_data_arr =
            alloc_nullable_array(0, Bodo_CTypes::INT32, 0);
        auto dict_encoded_src = std::make_shared<array_info>(
            bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING, 0,
            std::vector<std::shared_ptr<BodoBuffer>>({}),
            std::vector<std::shared_ptr<array_info>>(
                {src_data, indices_data_arr}));

        // Append some initial data to src, and unify with dst
        auto table = strVecToTable("A", std::vector<std::string>{
                                            "Str1",
                                            "Str2",
                                        });
        dict_builder_src.UnifyDictionaryArray(table->columns[0]);
        // We've never seen this dictionary before, cache id miss
        dict_builder_dst.UnifyDictionaryArray(dict_encoded_src);
        bodo::tests::check(dict_builder_dst.unify_cache_length_misses == 0);
        bodo::tests::check(dict_builder_dst.unify_cache_id_misses == 1);

        // Add completely new data to src (changing the length) and redo unify
        auto table_1 = strVecToTable("A", std::vector<std::string>{
                                              "Str3",
                                              "Str4",
                                          });
        dict_builder_src.UnifyDictionaryArray(table_1->columns[0]);
        // We've seen this dictionary before - but the length has changed,
        // cache length miss
        dict_builder_dst.UnifyDictionaryArray(dict_encoded_src);
        bodo::tests::check(dict_builder_dst.unify_cache_length_misses == 1);
        bodo::tests::check(dict_builder_dst.unify_cache_id_misses == 1);

        // Add data that shouldn't change the length of src
        auto table_2 = strVecToTable("A", std::vector<std::string>{
                                              "Str1",
                                              "Str4",
                                          });
        dict_builder_src.UnifyDictionaryArray(table_2->columns[0]);
        // We've seen this dictionary before - and the length has not
        // changed, cache hit
        dict_builder_dst.UnifyDictionaryArray(dict_encoded_src);
        bodo::tests::check(dict_builder_dst.unify_cache_length_misses == 1);
        bodo::tests::check(dict_builder_dst.unify_cache_id_misses == 1);

        // Assert that all strings are already in dst
        bodo::tests::check(dict_builder_dst.dict_buff->data_array->length == 4);
    });

    bodo::tests::test("test_append_string_array_to_builder", [] {
        /*
         * This test will create two dictionary builders (src, dst), append data
         * to one of them (src), and the unify into the second dictionary. Then
         * the test will append more data to the first dictionary, and redo the
         * unification. This should cause a cache miss because the length of the
         * dictionary has changed.
         */

        DictionaryBuilder dict_builder = DictionaryBuilder(
            alloc_array_top_level(0, 0, 0, bodo_array_type::STRING,
                                  Bodo_CTypes::STRING),
            false);
        // Assert that the dictionary is empty
        bodo::tests::check(dict_builder.dict_buff->data_array->array_id <= 0);
        bodo::tests::check(dict_builder.dict_buff->data_array->length == 0,
                           "expected length = 0");

        auto table_0 = strVecToTable("A", {}, false);
        dict_builder.UnifyDictionaryArray(table_0->columns[0]);
        // Assert that the dictionary is still empty
        bodo::tests::check(dict_builder.dict_buff->data_array->array_id <= 0);
        bodo::tests::check(dict_builder.dict_buff->data_array->length == 0);

        auto table_1 = strVecToTable("A",
                                     std::vector<std::string>{
                                         "Str1",
                                         "Str2",
                                         "Str3",
                                         "Str3",
                                         "Str3",
                                     },
                                     false);
        dict_builder.UnifyDictionaryArray(table_1->columns[0]);
        // Assert that the dictionary is non-empty and has a valid id
        bodo::tests::check(dict_builder.dict_buff->data_array->array_id > 0);
        bodo::tests::check(dict_builder.dict_buff->data_array->length == 3);
    });
});
