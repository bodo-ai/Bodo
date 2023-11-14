/// Tests of bodo::tests::cppToBodo

#include "./table_generator.hpp"
#include "./test.hpp"

bodo::tests::suite table_generator_tests([] {
    bodo::tests::test("test_table_generator", [] {
        auto table =
            bodo::tests::cppToBodo({"A", "B"}, {false, true}, {"A"},
                                   std::vector<std::string>{"Hello", "world"},
                                   std::vector<int64_t>{0, 1});
        bodo::tests::check(table->columns.size() == 2);

        bodo::tests::check(table->columns[0]->length == 2);
        bodo::tests::check(table->columns[0]->arr_type ==
                           bodo_array_type::arr_type_enum::DICT);
        bodo::tests::check(table->columns[0]->dtype ==
                           Bodo_CTypes::CTypeEnum::STRING);

        bodo::tests::check(table->columns[1]->length == 2);
        bodo::tests::check(table->columns[1]->arr_type ==
                           bodo_array_type::arr_type_enum::NULLABLE_INT_BOOL);
        bodo::tests::check(table->columns[1]->dtype ==
                           Bodo_CTypes::CTypeEnum::INT64);
    });
    bodo::tests::test("wrong_number_of_input_cols", [] {
        bool exception_raised = false;
        try {
            // Passed in three names but only two columns
            auto table = bodo::tests::cppToBodo(
                {"A", "B", "C"}, {false, true, true}, {},
                std::vector<int64_t>{0}, std::vector<int64_t>{1});
        } catch (std::runtime_error &e) {
            exception_raised = true;
        }
        bodo::tests::check(exception_raised);
    });
    bodo::tests::test("too_many_variadic_args", [] {
        bool exception_raised = false;
        try {
            // Passed in three names but only two columns
            auto table = bodo::tests::cppToBodo(
                {"A"}, {false}, {}, std::vector<int64_t>{0},
                std::vector<int64_t>{1}, std::vector<int64_t>{2});
        } catch (std::runtime_error &e) {
            exception_raised = true;
        }
        bodo::tests::check(exception_raised);
    });
    bodo::tests::test("wrong_number_of_is_nullables", [] {
        bool exception_raised = false;
        try {
            // Passed in three names but only two columns
            auto table = bodo::tests::cppToBodo(
                {"A", "B", "C"}, {false, true}, {}, std::vector<int64_t>{0},
                std::vector<int64_t>{1}, std::vector<int64_t>{2});
        } catch (std::runtime_error &e) {
            exception_raised = true;
        }
        bodo::tests::check(exception_raised);
    });
});
