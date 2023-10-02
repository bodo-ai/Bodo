/// Example file showing how to use the C++ testing infrastructure

#include "./test.hpp"

bodo::tests::suite tests([] {
    bodo::tests::before_each([] {
        // Example before_each - this is optional and can be omitted entirely
    });

    bodo::tests::after_each([] {
        // Example after_each - this is optional and can be omitted entirely
    });

    bodo::tests::test("test_example", [] { bodo::tests::check(true); });
});
