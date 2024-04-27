#include "../libs/_bodo_to_arrow.h"
#include "../libs/_theta_sketches.h"
#include "./test.hpp"

// Helper utility to create nullable arrays used for testing.
// Creates a nullable column from vectors of values and nulls
template <Bodo_CTypes::CTypeEnum dtype, typename T>
    requires(dtype != Bodo_CTypes::_BOOL)
std::shared_ptr<array_info> nullable_array_from_vector(
    std::vector<T> numbers, std::vector<bool> nulls) {
    size_t length = numbers.size();
    auto result = alloc_nullable_array_no_nulls(length, dtype);
    T *buffer = result->data1<bodo_array_type::NULLABLE_INT_BOOL, T>();
    for (size_t i = 0; i < length; i++) {
        if (nulls[i]) {
            buffer[i] = (T)numbers[i];
        } else {
            result->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
        }
    }
    return result;
}

// Variant of nullable_array_from_vector to build a string array from vectors
std::shared_ptr<array_info> string_array_from_vector(
    bodo::vector<std::string> strings, bodo::vector<bool> nulls) {
    size_t length = strings.size();

    bodo::vector<uint8_t> null_bitmask((length + 7) >> 3, 0);
    for (size_t i = 0; i < length; i++) {
        SetBitTo(null_bitmask.data(), i, nulls[i]);
    }
    return create_string_array(Bodo_CTypes::STRING, null_bitmask, strings, -1);
}

// Verifies that two floats x and y are approximately equal, with a
// relative tolerance of epsilon relative to x. The default is 0.01
// which means it checks if y is within 1% of x.
void check_approx(double x, double y, double epsilon = 0.01) {
    double lower, upper;
    if (x < 0) {
        lower = x * (1 + epsilon);
        upper = x * (1 - epsilon);
    } else {
        lower = x * (1 - epsilon);
        upper = x * (1 + epsilon);
    }
    bodo::tests::check(lower <= y);
    bodo::tests::check(upper >= y);
}

static bodo::tests::suite tests([] {
    /********************************************************************************/
    /* Start of tests copied from theta_sketch_test.cpp in the theta_sketch
     * library */
    /********************************************************************************/
    bodo::tests::test("library_tests_empty", [] {
        datasketches::update_theta_sketch update_sketch =
            datasketches::update_theta_sketch::builder().build();
        bodo::tests::check(update_sketch.is_empty());
        bodo::tests::check(!update_sketch.is_estimation_mode());
        bodo::tests::check(update_sketch.get_theta() == 1.0);
        bodo::tests::check(update_sketch.get_estimate() == 0.0);
        bodo::tests::check(update_sketch.get_lower_bound(1) == 0.0);
        bodo::tests::check(update_sketch.get_upper_bound(1) == 0.0);
        bodo::tests::check(update_sketch.is_ordered());

        datasketches::compact_theta_sketch compact_sketch =
            update_sketch.compact();
        bodo::tests::check(compact_sketch.is_empty());
        bodo::tests::check(!compact_sketch.is_estimation_mode());
        bodo::tests::check(compact_sketch.get_theta() == 1.0);
        bodo::tests::check(compact_sketch.get_estimate() == 0.0);
        bodo::tests::check(compact_sketch.get_lower_bound(1) == 0.0);
        bodo::tests::check(compact_sketch.get_upper_bound(1) == 0.0);
        bodo::tests::check(compact_sketch.is_ordered());

        // empty is forced to be ordered
        bodo::tests::check(update_sketch.compact(false).is_ordered());
    });
    bodo::tests::test("library_tests_no_retained", [] {
        // Test copied from theta_sketch_test.cpp in the theta_sketch library
        datasketches::update_theta_sketch update_sketch =
            datasketches::update_theta_sketch::builder().set_p(0.001f).build();
        update_sketch.update(1);
        bodo::tests::check(update_sketch.get_num_retained() == 0);
        bodo::tests::check(!update_sketch.is_empty());
        bodo::tests::check(update_sketch.is_estimation_mode());
        bodo::tests::check(update_sketch.get_estimate() == 0.0);
        bodo::tests::check(update_sketch.get_lower_bound(1) == 0.0);
        bodo::tests::check(update_sketch.get_upper_bound(1) > 0);

        datasketches::compact_theta_sketch compact_sketch =
            update_sketch.compact();
        bodo::tests::check(compact_sketch.get_num_retained() == 0);
        bodo::tests::check(!compact_sketch.is_empty());
        bodo::tests::check(compact_sketch.is_estimation_mode());
        bodo::tests::check(compact_sketch.get_estimate() == 0.0);
        bodo::tests::check(compact_sketch.get_lower_bound(1) == 0.0);
        bodo::tests::check(compact_sketch.get_upper_bound(1) > 0);

        update_sketch.reset();
        bodo::tests::check(update_sketch.is_empty());
        bodo::tests::check(!update_sketch.is_estimation_mode());
        bodo::tests::check(update_sketch.get_theta() == 1.0);
        bodo::tests::check(update_sketch.get_estimate() == 0.0);
        bodo::tests::check(update_sketch.get_lower_bound(1) == 0.0);
        bodo::tests::check(update_sketch.get_upper_bound(1) == 0.0);
    });
    bodo::tests::test("library_tests_singleton", [] {
        // Test copied from theta_sketch_test.cpp in the theta_sketch library
        datasketches::update_theta_sketch update_sketch =
            datasketches::update_theta_sketch::builder().build();
        update_sketch.update(1);
        bodo::tests::check(!update_sketch.is_empty());
        bodo::tests::check(!update_sketch.is_estimation_mode());
        bodo::tests::check(update_sketch.get_theta() == 1.0);
        bodo::tests::check(update_sketch.get_estimate() == 1.0);
        bodo::tests::check(update_sketch.get_lower_bound(1) == 1.0);
        bodo::tests::check(update_sketch.get_upper_bound(1) == 1.0);
        bodo::tests::check(update_sketch.is_ordered());  // one item is ordered

        datasketches::compact_theta_sketch compact_sketch =
            update_sketch.compact();
        bodo::tests::check(!compact_sketch.is_empty());
        bodo::tests::check(!compact_sketch.is_estimation_mode());
        bodo::tests::check(compact_sketch.get_theta() == 1.0);
        bodo::tests::check(compact_sketch.get_estimate() == 1.0);
        bodo::tests::check(compact_sketch.get_lower_bound(1) == 1.0);
        bodo::tests::check(compact_sketch.get_upper_bound(1) == 1.0);
        bodo::tests::check(compact_sketch.is_ordered());

        // single item is forced to be ordered
        bodo::tests::check(update_sketch.compact(false).is_ordered());
    });
    bodo::tests::test("library_tests_estimation", [] {
        // Test copied from theta_sketch_test.cpp in the theta_sketch library
        datasketches::update_theta_sketch update_sketch =
            datasketches::update_theta_sketch::builder()
                .set_resize_factor(
                    datasketches::update_theta_sketch::resize_factor::X1)
                .build();
        const int n = 8000;
        for (int i = 0; i < n; i++)
            update_sketch.update(i);
        double lower_estimate = ((double)n) * 0.99;
        double upper_estimate = ((double)n) * 1.01;
        bodo::tests::check(!update_sketch.is_empty());
        bodo::tests::check(update_sketch.is_estimation_mode());
        bodo::tests::check(update_sketch.get_theta() < 1.0);
        bodo::tests::check(update_sketch.get_estimate() >= lower_estimate);
        bodo::tests::check(update_sketch.get_estimate() <= upper_estimate);
        bodo::tests::check(update_sketch.get_lower_bound(1) < n);
        bodo::tests::check(update_sketch.get_upper_bound(1) > n);

        const uint32_t k = 1 << datasketches::theta_constants::DEFAULT_LG_K;
        bodo::tests::check(update_sketch.get_num_retained() >= k);
        update_sketch.trim();
        bodo::tests::check(update_sketch.get_num_retained() == k);

        datasketches::compact_theta_sketch compact_sketch =
            update_sketch.compact();
        bodo::tests::check(!compact_sketch.is_empty());
        bodo::tests::check(compact_sketch.is_ordered());
        bodo::tests::check(compact_sketch.is_estimation_mode());
        bodo::tests::check(compact_sketch.get_theta() < 1.0);
        bodo::tests::check(compact_sketch.get_estimate() >= lower_estimate);
        bodo::tests::check(compact_sketch.get_estimate() <= upper_estimate);
        bodo::tests::check(compact_sketch.get_lower_bound(1) < n);
        bodo::tests::check(compact_sketch.get_upper_bound(1) > n);
    });
    bodo::tests::test("library_tests_serialize_deserialize_compressed", [] {
        // Test copied from theta_sketch_test.cpp in the theta_sketch library
        auto update_sketch =
            datasketches::update_theta_sketch::builder().build();
        for (int i = 0; i < 10000; i++)
            update_sketch.update(i);
        auto compact_sketch = update_sketch.compact();

        auto bytes = compact_sketch.serialize_compressed();
        {
            // deserialize bytes
            auto deserialized_sketch =
                datasketches::compact_theta_sketch::deserialize(bytes.data(),
                                                                bytes.size());
            bodo::tests::check(deserialized_sketch.get_num_retained() ==
                               compact_sketch.get_num_retained());
            bodo::tests::check(deserialized_sketch.get_theta() ==
                               compact_sketch.get_theta());
            auto iter = deserialized_sketch.begin();
            for (const auto key : compact_sketch) {
                bodo::tests::check(*iter == key);
                ++iter;
            }
        }
        {
            // wrap bytes
            auto wrapped_sketch =
                datasketches::wrapped_compact_theta_sketch::wrap(bytes.data(),
                                                                 bytes.size());
            bodo::tests::check(wrapped_sketch.get_num_retained() ==
                               compact_sketch.get_num_retained());
            bodo::tests::check(wrapped_sketch.get_theta() ==
                               compact_sketch.get_theta());
            auto iter = wrapped_sketch.begin();
            for (const auto key : compact_sketch) {
                bodo::tests::check(*iter == key);
                ++iter;
            }
        }

        std::stringstream s(std::ios::in | std::ios::out | std::ios::binary);
        compact_sketch.serialize_compressed(s);
        auto deserialized_sketch =
            datasketches::compact_theta_sketch::deserialize(s);
        bodo::tests::check(deserialized_sketch.get_num_retained() ==
                           compact_sketch.get_num_retained());
        bodo::tests::check(deserialized_sketch.get_theta() ==
                           compact_sketch.get_theta());
        auto iter = deserialized_sketch.begin();
        for (const auto key : compact_sketch) {
            bodo::tests::check(*iter == key);
            ++iter;
        }
    });
    /******************************************************************************/
    /* End of tests copied from theta_sketch_test.cpp in the theta_sketch
     * library */
    /******************************************************************************/
    bodo::tests::test("test_collection_init", [] {
        // Verifies that init_theta_sketches produces a vector of the correct
        // length and with the correct combination of null / non-null pointers.
        auto test_fn = [](const std::vector<bool> &ndv_cols) {
            size_t n_cols = ndv_cols.size();
            auto sketch_collection = init_theta_sketches(ndv_cols);
            for (size_t col_idx = 0; col_idx < n_cols; col_idx++) {
                bodo::tests::check(ndv_cols[col_idx] ==
                                   sketch_collection[col_idx].has_value());
            }
            delete[] sketch_collection;
        };
        test_fn({});
        test_fn({true});
        test_fn({false});
        test_fn({true, true});
        test_fn({true, false});
        test_fn({false, true});
        test_fn({false, false});
        test_fn({true, true, true});
        test_fn({true, true, false});
        test_fn({true, false, true});
        test_fn({true, false, false});
        test_fn({false, true, true});
        test_fn({false, true, false});
        test_fn({false, false, true});
        test_fn({false, false, false});
    });
    bodo::tests::test("test_singleton_numeric_single_update", [] {
        // Tests creating a singleton theta sketch and adding
        // a single batch of numeric data then estimating
        // the number of distinct entries.
        auto sketch_collection = init_theta_sketches({true});
        auto A1 = nullable_array_from_vector<Bodo_CTypes::INT32, int32_t>(
            {0, 1, 4, -1, 9, 6, 5, -1, 6, 9, 4},
            {true, true, true, false, true, true, true, false, true, true,
             true});
        std::shared_ptr<table_info> T1 = std::make_shared<table_info>();
        T1->columns.push_back(A1);
        auto arrow_table = bodo_table_to_arrow(T1);
        for (int i = 0; i < arrow_table->num_columns(); i++) {
            auto column = arrow_table->column(i);
            update_theta_sketches(sketch_collection, column, i);
        }
        bodo::tests::check(!sketch_collection[0].value().is_empty());
        bodo::tests::check(!sketch_collection[0].value().is_estimation_mode());
        bodo::tests::check(sketch_collection[0].value().get_theta() == 1.0);
        bodo::tests::check(sketch_collection[0].value().get_estimate() == 6.0);
        bodo::tests::check(sketch_collection[0].value().get_lower_bound(1) ==
                           6.0);
        bodo::tests::check(sketch_collection[0].value().get_upper_bound(1) ==
                           6.0);
        bodo::tests::check(!sketch_collection[0].value().is_ordered());
        delete[] sketch_collection;
    });
    bodo::tests::test("test_singleton_numeric_multiple_update", [] {
        // Tests creating a singleton theta sketch and adding
        // a multiple batches of numeric data then estimating
        // the number of distinct entries.
        auto sketch_collection = init_theta_sketches({true});
        // Create and insert 100 batches of data with a total of
        // 29850 rows and 8743 distinct non-null entries.
        for (size_t batch = 0; batch < 100; batch++) {
            size_t batch_size = 300 + ((batch % 10) * (batch % 10)) - 30;
            std::vector<int64_t> data;
            std::vector<bool> nulls;
            for (size_t i = 0; i < batch_size; i++) {
                if ((i + 1) * (i + batch) % 17 > 0) {
                    int64_t value = 0;
                    for (size_t j = 1; j < 16; j++) {
                        value += (i % j) + batch;
                        if (j % 3 == 0)
                            value <<= 1;
                    }
                    data.push_back(value);
                    nulls.push_back(true);
                } else {
                    data.push_back(-1);
                    nulls.push_back(false);
                }
            }
            auto A1 = nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>(
                data, nulls);
            std::shared_ptr<table_info> T1 = std::make_shared<table_info>();
            T1->columns.push_back(A1);
            auto arrow_table = bodo_table_to_arrow(T1);
            for (int i = 0; i < arrow_table->num_columns(); i++) {
                auto column = arrow_table->column(i);
                update_theta_sketches(sketch_collection, column, i);
            }
        }
        bodo::tests::check(!sketch_collection[0].value().is_empty());
        bodo::tests::check(sketch_collection[0].value().is_estimation_mode());
        check_approx(sketch_collection[0].value().get_theta(), 0.538182);
        check_approx(sketch_collection[0].value().get_estimate(), 8320.61);
        check_approx(sketch_collection[0].value().get_lower_bound(1), 8235.12);
        check_approx(sketch_collection[0].value().get_upper_bound(1), 8406.97);
        delete[] sketch_collection;
    });
    bodo::tests::test("test_multiple_numeric_multiple_update", [] {
        // Tests creating a collection of multiple theta sketches, including
        // one column that does not have a sketch, and adding a multiple batches
        // of numeric data then estimating the number of distinct entries.
        auto sketch_collection = init_theta_sketches({true, false, true});
        // Create and insert 100 batches of data with the following patterns:
        // Column 0: Total of 48924 rows, all of them unique
        // Column 1: Same as column 0
        // COlumn 2: Total of 48924, 349 of which are unique non-nulls
        size_t row_id = 0;
        for (size_t batch = 0; batch < 100; batch++) {
            size_t batch_size = 500 + ((batch % 11) * (batch % 13)) - 40;
            std::vector<int64_t> data_0;
            std::vector<bool> nulls_0;
            std::vector<int64_t> data_2;
            std::vector<bool> nulls_2;
            for (size_t i = 0; i < batch_size; i++) {
                data_0.push_back(row_id);
                nulls_0.push_back(true);
                if (i % 100 > 0) {
                    int64_t row_id_copy = row_id;
                    int64_t value = 1;
                    while (row_id_copy > 0) {
                        value *= row_id_copy % 10;
                        row_id_copy /= 10;
                    }
                    data_2.push_back(value);
                    nulls_2.push_back(true);
                } else {
                    data_0.push_back(-1);
                    nulls_2.push_back(false);
                }
                row_id++;
            }
            auto A1 = nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>(
                data_0, nulls_0);
            auto A2 = nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>(
                data_0, nulls_0);
            auto A3 = nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>(
                data_2, nulls_2);
            std::shared_ptr<table_info> T1 = std::make_shared<table_info>();
            T1->columns.push_back(A1);
            T1->columns.push_back(A2);
            T1->columns.push_back(A3);
            auto arrow_table = bodo_table_to_arrow(T1);
            for (int i = 0; i < arrow_table->num_columns(); i++) {
                auto column = arrow_table->column(i);
                update_theta_sketches(sketch_collection, column, i);
            }
        }
        bodo::tests::check(!sketch_collection[0].value().is_empty());
        bodo::tests::check(sketch_collection[0].value().is_estimation_mode());
        check_approx(sketch_collection[0].value().get_theta(), 0.152023);
        check_approx(sketch_collection[0].value().get_estimate(), 48275.6);
        check_approx(sketch_collection[0].value().get_lower_bound(2), 47245.1);
        check_approx(sketch_collection[0].value().get_lower_bound(1), 47755.7);
        check_approx(sketch_collection[0].value().get_upper_bound(1), 48801.1);
        check_approx(sketch_collection[0].value().get_upper_bound(2), 49328.5);

        bodo::tests::check(!sketch_collection[2].value().is_empty());
        bodo::tests::check(!sketch_collection[2].value().is_estimation_mode());
        bodo::tests::check(sketch_collection[2].value().get_theta() == 1.0);
        bodo::tests::check(sketch_collection[2].value().get_estimate() ==
                           349.0);
        bodo::tests::check(sketch_collection[2].value().get_lower_bound(1) ==
                           349.0);
        bodo::tests::check(sketch_collection[2].value().get_upper_bound(1) ==
                           349.0);
        delete[] sketch_collection;
    });
    bodo::tests::test("test_singleton_string_single_update", [] {
        // Tests creating a singleton theta sketch and adding
        // a single batch of string data then estimating
        // the number of distinct entries.
        auto sketch_collection = init_theta_sketches({true});
        auto A1 = string_array_from_vector(
            {"Alphabet", "", "Soup", "", "Is", "", "Alcoholic", "", "Alphabet"},
            {true, false, true, false, true, false, true, false, true});
        std::shared_ptr<table_info> T1 = std::make_shared<table_info>();
        T1->columns.push_back(A1);
        auto arrow_table = bodo_table_to_arrow(T1);
        for (int i = 0; i < arrow_table->num_columns(); i++) {
            auto column = arrow_table->column(i);
            update_theta_sketches(sketch_collection, column, i);
        }
        bodo::tests::check(!sketch_collection[0].value().is_empty());
        bodo::tests::check(!sketch_collection[0].value().is_estimation_mode());
        bodo::tests::check(sketch_collection[0].value().get_theta() == 1.0);
        bodo::tests::check(sketch_collection[0].value().get_estimate() == 4.0);
        bodo::tests::check(sketch_collection[0].value().get_lower_bound(1) ==
                           4.0);
        bodo::tests::check(sketch_collection[0].value().get_upper_bound(1) ==
                           4.0);
        bodo::tests::check(!sketch_collection[0].value().is_ordered());
        delete[] sketch_collection;
    });
    bodo::tests::test("test_singleton_string_multiple_update", [] {
        // Tests creating a singleton theta sketch and adding
        // a multiple batches of string data then estimating
        // the number of distinct entries.
        auto sketch_collection = init_theta_sketches({true});
        // Create and insert 100 batches of data with a total of
        // 51810 rows and 16746 distinct non-null strings.
        size_t row_id = 32;
        for (size_t batch = 0; batch < 100; batch++) {
            size_t batch_size = 500 + ((batch % 17) * (batch % 11)) - 21;
            bodo::vector<std::string> data;
            bodo::vector<bool> nulls;
            for (size_t i = 0; i < batch_size; i++) {
                if ((i * batch) % 101 < 100) {
                    data.push_back(
                        std::to_string(row_id * row_id).substr(0, 4) +
                        std::to_string(batch).substr(0, 1));
                    nulls.push_back(true);
                } else {
                    data.push_back("");
                    nulls.push_back(false);
                }
                row_id++;
            }
            auto A1 = string_array_from_vector(data, nulls);
            std::shared_ptr<table_info> T1 = std::make_shared<table_info>();
            T1->columns.push_back(A1);
            auto arrow_table = bodo_table_to_arrow(T1);
            for (int i = 0; i < arrow_table->num_columns(); i++) {
                auto column = arrow_table->column(i);
                update_theta_sketches(sketch_collection, column, i);
            }
        }
        bodo::tests::check(!sketch_collection[0].value().is_empty());
        bodo::tests::check(sketch_collection[0].value().is_estimation_mode());
        check_approx(sketch_collection[0].value().get_theta(), 0.279546);
        check_approx(sketch_collection[0].value().get_estimate(), 16923.8);
        check_approx(sketch_collection[0].value().get_lower_bound(1), 16714.0);
        check_approx(sketch_collection[0].value().get_upper_bound(1), 17136.3);
        delete[] sketch_collection;
    });
    bodo::tests::test("test_singleton_dict_multiple_update", [] {
        // Tests creating a singleton theta sketch and adding
        // a multiple batches of dictionary encoded data then estimating
        // the number of distinct entries.
        auto sketch_collection = init_theta_sketches({true});
        // Create and insert 100 batches of data with a total of
        // 111383 rows and 9595 distinct non-null strings, including
        // many dict rows that are never used.
        bodo::vector<std::string> data;
        bodo::vector<bool> nulls;
        for (size_t i = 0; i < 20000; i++) {
            data.push_back(std::to_string(i));
            nulls.push_back(true);
        }
        auto dict_arr = string_array_from_vector(data, nulls);
        size_t row_id = 0;
        for (size_t batch = 0; batch < 100; batch++) {
            size_t batch_size = 1000 + ((batch % 31) * (batch % 39)) - 120;
            std::vector<dict_indices_t> indices;
            std::vector<bool> nulls;
            for (size_t i = 0; i < batch_size; i++) {
                if ((i * batch) % 201 < 200) {
                    int64_t val = (row_id * row_id * row_id) % 20000;
                    indices.push_back(val);
                    nulls.push_back(true);
                } else {
                    indices.push_back(0);
                    nulls.push_back(false);
                }
                row_id++;
            }
            std::shared_ptr<array_info> index_arr =
                nullable_array_from_vector<Bodo_CTypes::INT32, dict_indices_t>(
                    indices, nulls);
            std::shared_ptr<table_info> T1 = std::make_shared<table_info>();
            T1->columns.push_back(
                create_dict_string_array(dict_arr, index_arr));
            auto arrow_table = bodo_table_to_arrow(T1);
            auto column = arrow_table->column(0);
            auto dict_hits = get_dictionary_hits(column);
            update_theta_sketches(sketch_collection, column, 0, dict_hits);
        }
        bodo::tests::check(!sketch_collection[0].value().is_empty());
        bodo::tests::check(sketch_collection[0].value().is_estimation_mode());
        check_approx(sketch_collection[0].value().get_theta(), 0.535667);
        check_approx(sketch_collection[0].value().get_estimate(), 9472.3);
        check_approx(sketch_collection[0].value().get_lower_bound(2), 9291.37);
        check_approx(sketch_collection[0].value().get_lower_bound(1), 9380.69);
        check_approx(sketch_collection[0].value().get_upper_bound(1), 9564.78);
        check_approx(sketch_collection[0].value().get_upper_bound(2), 9656.71);
        delete[] sketch_collection;
    });
    bodo::tests::test("test_singleton_numeric_multiple_merge", [] {
        // Tests creating singleton theta sketches and adding
        // a batch of numeric data to each, merging them, then
        // estimating the number of distinct entries as well as
        // the serialization.
        auto sketch_collection_0 = init_theta_sketches({true});
        auto sketch_collection_1 = init_theta_sketches({true});
        auto sketch_collection_2 = init_theta_sketches({true});
        auto sketch_collection_3 = init_theta_sketches({true});
        // Collection 0: insert 8 values, 4 of which are unique
        auto A0 = nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>(
            {1, 1, 2, 1, 1, 2, 3, 4},
            {true, true, true, true, true, true, true, true});
        std::shared_ptr<table_info> T0 = std::make_shared<table_info>();
        T0->columns.push_back(A0);
        auto arrow_table_0 = bodo_table_to_arrow(T0);
        for (int i = 0; i < arrow_table_0->num_columns(); i++) {
            auto column = arrow_table_0->column(i);
            update_theta_sketches(sketch_collection_0, column, i);
        }
        // Collection 1: insert 8 values, 2 duplicates of #0 and 2 unique
        auto A1 = nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>(
            {3, 4, 5, 6, 3, 4, 5, 6},
            {true, true, true, true, true, true, true, true});
        std::shared_ptr<table_info> T1 = std::make_shared<table_info>();
        T1->columns.push_back(A1);
        auto arrow_table_1 = bodo_table_to_arrow(T1);
        for (int i = 0; i < arrow_table_1->num_columns(); i++) {
            auto column = arrow_table_1->column(i);
            update_theta_sketches(sketch_collection_1, column, i);
        }
        // Collection 2: insert 1,000 values, all unique except for 1 overlap
        // with #1
        std::vector<int64_t> data2;
        std::vector<bool> nulls2;
        for (size_t row = 0; row < 1000; row++) {
            data2.push_back(6 + row);
            nulls2.push_back(true);
        }
        auto A2 = nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>(
            data2, nulls2);
        std::shared_ptr<table_info> T2 = std::make_shared<table_info>();
        T2->columns.push_back(A2);
        auto arrow_table_2 = bodo_table_to_arrow(T2);
        for (int i = 0; i < arrow_table_2->num_columns(); i++) {
            auto column = arrow_table_2->column(i);
            update_theta_sketches(sketch_collection_2, column, i);
        }
        // Collection 3: insert 10,000 values, all unique (no overlap with
        // #0/#1/#2)
        std::vector<int64_t> data3;
        std::vector<bool> nulls3;
        for (size_t row = 0; row < 10000; row++) {
            data3.push_back(10000 + row);
            nulls3.push_back(true);
        }
        auto A3 = nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>(
            data3, nulls3);
        std::shared_ptr<table_info> T3 = std::make_shared<table_info>();
        T3->columns.push_back(A3);
        auto arrow_table_3 = bodo_table_to_arrow(T3);
        for (int i = 0; i < arrow_table_3->num_columns(); i++) {
            auto column = arrow_table_3->column(i);
            update_theta_sketches(sketch_collection_3, column, i);
        }

        immutable_theta_sketch_collection_t compact_0 =
            compact_theta_sketches(sketch_collection_0, 1);
        immutable_theta_sketch_collection_t compact_1 =
            compact_theta_sketches(sketch_collection_1, 1);
        immutable_theta_sketch_collection_t compact_2 =
            compact_theta_sketches(sketch_collection_2, 1);
        immutable_theta_sketch_collection_t compact_3 =
            compact_theta_sketches(sketch_collection_3, 1);

        // First test: merge collections 0 and 1, observe the estimates
        immutable_theta_sketch_collection_t merge_01 =
            merge_theta_sketches({compact_0, compact_1});
        bodo::tests::check(!merge_01[0].value().is_empty());
        bodo::tests::check(!merge_01[0].value().is_estimation_mode());
        check_approx(merge_01[0].value().get_theta(), 1.0);
        check_approx(merge_01[0].value().get_estimate(), 6.0);
        check_approx(merge_01[0].value().get_lower_bound(1), 6.0);
        check_approx(merge_01[0].value().get_upper_bound(1), 6.0);

        // Second test: serialize the merged #0/#1, then deserialize,
        // then confirm the estimates are intact.
        auto serialized_01 = serialize_theta_sketches(merge_01);
        immutable_theta_sketch_collection_t deserialized_01 =
            deserialize_theta_sketches(serialized_01);
        bodo::tests::check(!deserialized_01[0].value().is_empty());
        bodo::tests::check(!deserialized_01[0].value().is_estimation_mode());
        check_approx(deserialized_01[0].value().get_theta(), 1.0);
        check_approx(deserialized_01[0].value().get_estimate(), 6.0);
        check_approx(deserialized_01[0].value().get_lower_bound(1), 6.0);
        check_approx(deserialized_01[0].value().get_upper_bound(1), 6.0);

        // Third test: merge with #2, confirm the accuracy of the estimates
        immutable_theta_sketch_collection_t merge_012 =
            merge_theta_sketches({deserialized_01, compact_2});
        bodo::tests::check(!merge_012[0].value().is_empty());
        bodo::tests::check(!merge_012[0].value().is_estimation_mode());
        check_approx(merge_012[0].value().get_theta(), 1.0);
        check_approx(merge_012[0].value().get_estimate(), 1005.0);
        check_approx(merge_012[0].value().get_lower_bound(1), 1005.0);
        check_approx(merge_012[0].value().get_upper_bound(1), 1005.0);

        // Fourth test: serialize the merged #0/#1/#2, then deserialize,
        // then confirm the estimates are intact.
        auto serialized_012 = serialize_theta_sketches(merge_012);
        immutable_theta_sketch_collection_t deserialized_012 =
            deserialize_theta_sketches(serialized_012);
        bodo::tests::check(!deserialized_012[0].value().is_empty());
        bodo::tests::check(!deserialized_012[0].value().is_estimation_mode());
        check_approx(deserialized_012[0].value().get_theta(), 1.0);
        check_approx(deserialized_012[0].value().get_estimate(), 1005.0);
        check_approx(deserialized_012[0].value().get_lower_bound(1), 1005.0);
        check_approx(deserialized_012[0].value().get_upper_bound(1), 1005.0);

        // Fifth test: merge with #3, confirm the accuracy of the estimates
        immutable_theta_sketch_collection_t merge_0123 =
            merge_theta_sketches({deserialized_012, compact_3});
        bodo::tests::check(!merge_0123[0].value().is_empty());
        bodo::tests::check(merge_0123[0].value().is_estimation_mode());
        check_approx(merge_0123[0].value().get_theta(), 0.372599);
        check_approx(merge_0123[0].value().get_estimate(), 10993);
        check_approx(merge_0123[0].value().get_lower_bound(1), 10856);
        check_approx(merge_0123[0].value().get_upper_bound(1), 11131.8);

        // Sixth test: serialize the merged #0/#1/#2/#3, then deserialize,
        // then confirm the estimates are intact.
        auto serialized_0123 = serialize_theta_sketches(merge_0123);
        immutable_theta_sketch_collection_t deserialized_0123 =
            deserialize_theta_sketches(serialized_0123);
        bodo::tests::check(!deserialized_0123[0].value().is_empty());
        bodo::tests::check(deserialized_0123[0].value().is_estimation_mode());
        check_approx(deserialized_0123[0].value().get_theta(), 0.372599);
        check_approx(deserialized_0123[0].value().get_estimate(), 10993);
        check_approx(deserialized_0123[0].value().get_lower_bound(1), 10856);
        check_approx(deserialized_0123[0].value().get_upper_bound(1), 11131.8);

        delete[] sketch_collection_0;
        delete[] sketch_collection_1;
        delete[] sketch_collection_2;
        delete[] sketch_collection_3;
    });
    bodo::tests::test("test_parallel_merge_singleton", [] {
        // Tests creating singleton theta sketches across multiple
        // ranks, adding a batch of numeric data to each, then merging
        // them and estimating the number of distinct.
        int cp, np;
        MPI_Comm_rank(MPI_COMM_WORLD, &cp);
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        auto current_rank = (size_t)cp;
        auto num_ranks = (size_t)np;
        auto sketch_collection = init_theta_sketches({true});

        // There are 10 distinct values (0-100) that are distributed
        // by mod-ing by the number of ranks.
        std::vector<int64_t> data;
        std::vector<bool> nulls;
        for (size_t i = 0; i < 100; i++) {
            if ((i % num_ranks == current_rank) ||
                (i % num_ranks == current_rank + 1)) {
                for (size_t n = 0; n < num_ranks - current_rank; n++) {
                    data.push_back(i);
                    nulls.push_back(true);
                }
            }
        }
        auto A0 = nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>(
            data, nulls);
        std::shared_ptr<table_info> T0 = std::make_shared<table_info>();
        T0->columns.push_back(A0);
        auto arrow_table_0 = bodo_table_to_arrow(T0);
        for (int i = 0; i < arrow_table_0->num_columns(); i++) {
            auto column = arrow_table_0->column(i);
            update_theta_sketches(sketch_collection, column, i);
        }

        // Compact and merge the results across ranks
        auto immutable_collection =
            compact_theta_sketches(sketch_collection, 1);
        auto merged_result =
            merge_parallel_theta_sketches(immutable_collection);

        // Verify that there are a total of 100 distinct values, but only
        // on rank zero since the others should have received a nullptr.
        if (current_rank == 0) {
            bodo::tests::check(merged_result.size() == 1);
            bodo::tests::check(!merged_result[0].value().is_empty());
            bodo::tests::check(!merged_result[0].value().is_estimation_mode());
            check_approx(merged_result[0].value().get_theta(), 1.0);
            check_approx(merged_result[0].value().get_estimate(), 100.0);
            check_approx(merged_result[0].value().get_lower_bound(1), 100.0);
            check_approx(merged_result[0].value().get_upper_bound(1), 100.0);
        } else {
            bodo::tests::check(merged_result.size() == 0);
        }
        delete[] sketch_collection;
    });
    bodo::tests::test("test_parallel_merge_multiple", [] {
        // Tests creating multiple theta sketches across multiple
        // ranks, adding a batch of numeric data to each, then merging
        // them and estimating the number of distinct.
        int cp, np;
        MPI_Comm_rank(MPI_COMM_WORLD, &cp);
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        auto current_rank = (size_t)cp;
        auto num_ranks = (size_t)np;
        auto sketch_collection = init_theta_sketches({true, false, true});

        // Overall: 10,000 rows
        // Collection 0: 41 unique values across all the ranks
        // Colleciton 1: No theta sketch
        // Collection 2: Every row is unique
        std::vector<int64_t> data0;
        std::vector<int64_t> data2;
        std::vector<bool> nulls;
        for (size_t i = 0; i < 10000; i++) {
            if (i % num_ranks == current_rank) {
                data0.push_back(i % 41);
                data2.push_back(i);
                nulls.push_back(true);
            }
        }
        auto A0 = nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>(
            data0, nulls);
        auto A2 = nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>(
            data2, nulls);
        std::shared_ptr<table_info> T = std::make_shared<table_info>();
        T->columns.push_back(A0);
        T->columns.push_back(A0);
        T->columns.push_back(A2);
        auto arrow_table = bodo_table_to_arrow(T);
        for (int i = 0; i < arrow_table->num_columns(); i++) {
            auto column = arrow_table->column(i);
            update_theta_sketches(sketch_collection, column, i);
        }

        // Compact and merge the results across ranks
        auto immutable_collection =
            compact_theta_sketches(sketch_collection, 3);
        auto merged_result =
            merge_parallel_theta_sketches(immutable_collection);

        // Verify the expected results, but only on rank zero since the others
        // should have received a nullptr.
        if (current_rank == 0) {
            bodo::tests::check(merged_result.size() == 3);
            bodo::tests::check(merged_result[0].has_value());
            bodo::tests::check(!merged_result[1].has_value());
            bodo::tests::check(merged_result[2].has_value());
            bodo::tests::check(!merged_result[0].value().is_empty());
            bodo::tests::check(!merged_result[0].value().is_estimation_mode());
            check_approx(merged_result[0].value().get_theta(), 1.0);
            check_approx(merged_result[0].value().get_estimate(), 41.0);
            check_approx(merged_result[0].value().get_lower_bound(1), 41.0);
            check_approx(merged_result[0].value().get_upper_bound(1), 41.0);
            bodo::tests::check(!merged_result[2].value().is_empty());
            bodo::tests::check(merged_result[2].value().is_estimation_mode());
            check_approx(merged_result[2].value().get_theta(), 0.413831);
            check_approx(merged_result[2].value().get_estimate(), 9897.76);
            check_approx(merged_result[2].value().get_lower_bound(1), 9778.36);
            check_approx(merged_result[2].value().get_upper_bound(1), 10018.6);
        } else {
            bodo::tests::check(merged_result.size() == 0);
        }
        delete[] sketch_collection;
    });
});
