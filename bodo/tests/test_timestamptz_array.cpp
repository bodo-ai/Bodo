#include "../libs/_bodo_common.h"
#include "./test.hpp"

bodo::tests::suite timestamptz_array_tests([] {
    bodo::tests::test("test_timestamptz_array_allocation", [] {
        std::unique_ptr<array_info> arr = alloc_array_top_level(
            0, 0, 0, bodo_array_type::arr_type_enum::TIMESTAMPTZ,
            Bodo_CTypes::CTypeEnum::DATETIME);

        bodo::tests::check(arr->buffers[0] != nullptr);
        bodo::tests::check(arr->buffers[1] != nullptr);
        bodo::tests::check(arr->buffers[2] != nullptr);
    });
});
