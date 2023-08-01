/// Example file showing how to use the C++ testing infrastructure

#include "./test.hpp"

bodo::tests::suite tests([] {
    bodo::tests::test("test_example", [] { bodo::tests::check(true); });
});
