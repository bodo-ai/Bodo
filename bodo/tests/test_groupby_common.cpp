
#include "../libs/_bodo_common.h"
#include "../libs/groupby/_groupby_common.h"
#include "./table_generator.hpp"
#include "./test.hpp"

static bodo::tests::suite tests([] {
    bodo::tests::test("alloc_init_keys empty", [] {
        std::shared_ptr<table_info> table =
            bodo::tests::cppToBodo({"A", "B"}, {false, true}, {"A"},
                                   std::vector<std::string>{"Hello", "world"},
                                   std::vector<int64_t>{0, 1});
        std::vector<std::shared_ptr<table_info>> from_tables{table};
        std::vector<grouping_info> grp_infos;
        auto out_table = std::make_shared<table_info>();
        alloc_init_keys(from_tables, out_table, grp_infos, 0, 0);
        bodo::tests::check(out_table->columns.size() == 0);
    });
    bodo::tests::test("alloc_keys_one_table_one_group", [] {
        std::shared_ptr<table_info> table = bodo::tests::cppToBodo(
            {"A"}, {false}, {},
            std::vector<std::vector<std::string>>{{"Hello", "world"},
                                                  {"Goodbye", "universe"}});

        grouping_info grp_info;
        grp_info.group_to_first_row.push_back(0);

        auto out_table = std::make_shared<table_info>();
        alloc_init_keys({table}, out_table, {grp_info}, 1, 1);
        bodo::tests::check(out_table->columns.size() == 1);

        auto arrow_arr = to_arrow(out_table->columns[0]);
        std::shared_ptr<arrow::LargeListArray> list_arr =
            std::dynamic_pointer_cast<arrow::LargeListArray>(arrow_arr);

        bodo::tests::check(list_arr->value_offset(0) == 0);
        bodo::tests::check(list_arr->value_offset(1) == 2);

        std::shared_ptr<arrow::LargeStringArray> inner_arr =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(
                list_arr->values());
        bodo::tests::check(inner_arr->GetString(0) == "Hello");
        bodo::tests::check(inner_arr->GetString(1) == "world");
    });
    bodo::tests::test("alloc_keys_one_table_two_groups", [] {
        std::shared_ptr<table_info> table = bodo::tests::cppToBodo(
            {"A"}, {false}, {},
            std::vector<std::vector<std::string>>{{"Hello", "world"},
                                                  {"Goodbye", "universe"}});

        grouping_info grp_info;
        grp_info.group_to_first_row.push_back(0);
        grp_info.group_to_first_row.push_back(1);

        auto out_table = std::make_shared<table_info>();
        alloc_init_keys({table}, out_table, {grp_info}, 1, 2);
        bodo::tests::check(out_table->columns.size() == 1);

        auto arrow_arr = to_arrow(out_table->columns[0]);
        std::shared_ptr<arrow::LargeListArray> list_arr =
            std::dynamic_pointer_cast<arrow::LargeListArray>(arrow_arr);

        bodo::tests::check(list_arr->value_offset(0) == 0);
        bodo::tests::check(list_arr->value_offset(1) == 2);
        bodo::tests::check(list_arr->value_offset(2) == 4);

        std::shared_ptr<arrow::LargeStringArray> inner_arr =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(
                list_arr->values());
        bodo::tests::check(inner_arr->GetString(0) == "Hello");
        bodo::tests::check(inner_arr->GetString(1) == "world");
        bodo::tests::check(inner_arr->GetString(2) == "Goodbye");
        bodo::tests::check(inner_arr->GetString(3) == "universe");
    });

    bodo::tests::test("alloc_keys_two_tables_two_groups", [] {
        std::shared_ptr<table_info> table0 = bodo::tests::cppToBodo(
            {"A"}, {false}, {},
            std::vector<std::vector<std::string>>{{"Hello", "world"},
                                                  {"Goodbye", "universe"}});
        std::shared_ptr<table_info> table1 = bodo::tests::cppToBodo(
            {"A"}, {false}, {},
            std::vector<std::vector<std::string>>{{"one_a", "one_b", "one_c"},
                                                  {"two_a", "two_b", "two_c"}});

        grouping_info grp_info0;
        grp_info0.group_to_first_row.push_back(0);
        grp_info0.group_to_first_row.push_back(-1);
        grouping_info grp_info1;
        grp_info1.group_to_first_row.push_back(-1);
        grp_info1.group_to_first_row.push_back(1);

        auto out_table = std::make_shared<table_info>();
        alloc_init_keys({table0, table1}, out_table, {grp_info0, grp_info1}, 1,
                        2);
        bodo::tests::check(out_table->columns.size() == 1);

        auto arrow_arr = to_arrow(out_table->columns[0]);
        std::shared_ptr<arrow::LargeListArray> list_arr =
            std::dynamic_pointer_cast<arrow::LargeListArray>(arrow_arr);

        bodo::tests::check(list_arr->value_offset(0) == 0);
        bodo::tests::check(list_arr->value_offset(1) == 2);
        bodo::tests::check(list_arr->value_offset(2) == 5);

        std::shared_ptr<arrow::LargeStringArray> inner_arr =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(
                list_arr->values());
        bodo::tests::check(inner_arr->GetString(0) == "Hello");
        bodo::tests::check(inner_arr->GetString(1) == "world");
        bodo::tests::check(inner_arr->GetString(2) == "two_a");
        bodo::tests::check(inner_arr->GetString(3) == "two_b");
        bodo::tests::check(inner_arr->GetString(4) == "two_c");
    });
});
