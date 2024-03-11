#include "../libs/_bodo_common.h"
#include "../libs/_memory.h"
#include "../libs/_table_builder.h"
#include "./test.hpp"

bodo::tests::suite table_builder_tests([] {
    bodo::tests::test("test_table_builder_estimated_size", [] {
        std::cout << std::endl;

        std::vector<int8_t> arr_c_types{Bodo_CTypes::CTypeEnum::INT64};
        std::vector<int8_t> arr_array_types{
            bodo_array_type::arr_type_enum::NULLABLE_INT_BOOL};
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders{nullptr};
        std::shared_ptr<bodo::Schema> schema =
            bodo::Schema::Deserialize(arr_array_types, arr_c_types);

        TableBuildBuffer builder(schema, dict_builders);
        size_t estimated_size = builder.EstimatedSize();
        bodo::tests::check(estimated_size == 0);

        const size_t num_rows = 10;
        // Create a table with a single column and `num_rows` rows
        auto array = alloc_array_top_level(
            num_rows, 0, 0, bodo_array_type::arr_type_enum::NULLABLE_INT_BOOL,
            Bodo_CTypes::CTypeEnum::INT64);
        auto* data = (int64_t*)(array->data1());
        for (size_t i = 0; i < num_rows; i++) {
            data[i] = i;
        }
        auto table = alloc_table(schema);
        table->columns[0] = std::move(array);

        // Assert that when add to the builder, the expected size is non-zero
        builder.UnifyTablesAndAppend(table, dict_builders);
        estimated_size = builder.EstimatedSize();
        bodo::tests::check(estimated_size > 0);

        // Assert that as we continue to add to the table, the size grows
        // monotonically
        for (size_t i = 0; i < 10; i++) {
            builder.UnifyTablesAndAppend(table, dict_builders);
            size_t new_size = builder.EstimatedSize();
            bodo::tests::check(new_size >= estimated_size);
            estimated_size = new_size;
        }
    });
});
