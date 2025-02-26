/// Tests of the C++ testing infrastructure

#include "./test.hpp"

bodo::tests::suite test_framework_tests([] {

// Skip on Windows since the use of stderr seems to cause hangs on Windows CI
#ifndef _WIN32
    bodo::tests::test("Registering before/after_each after tests", [] {
        // This test constructs a fake suite and registers tests manually to
        // verify that before/after_each cannot be registered after a test is.
        bodo::tests::suite test_suite([] {}, false);
        std::vector<std::string> markers = {};
        test_suite.add_test("foo", [] {}, markers, 0);

        bool caught_exception = false;
        try {
            test_suite.before_each([] {});
        } catch (std::exception* e) {
            caught_exception = true;
        }
        bodo::tests::check(caught_exception);

        caught_exception = false;
        try {
            test_suite.after_each([] {});
        } catch (std::exception* e) {
            caught_exception = true;
        }
        bodo::tests::check(caught_exception);
    });

    bodo::tests::test("Registering before/after_each wraps tests", [] {
        // This test constructs a fake suite and registers tests manually to
        // verify that before/after_each is called before/after each test
        std::vector<std::string> events;
        bodo::tests::suite test_suite([] {}, false);
        std::vector<std::string> markers = {};
        test_suite.before_each([&] { events.emplace_back("before"); });
        test_suite.after_each([&] { events.emplace_back("after"); });
        test_suite.add_test(
            "test0", [&] { events.emplace_back("test0"); }, markers, 0);
        test_suite.add_test(
            "test1", [&] { events.emplace_back("test1"); }, markers, 1);

        auto& tests = test_suite.tests();
        tests.at("test0").func_();
        tests.at("test1").func_();

        // Check that all events happened in the expected order
        bodo::tests::check(events.size() == 6);
        bodo::tests::check(events[0] == "before");
        bodo::tests::check(events[1] == "test0");
        bodo::tests::check(events[2] == "after");
        bodo::tests::check(events[3] == "before");
        bodo::tests::check(events[4] == "test1");
        bodo::tests::check(events[5] == "after");
    });

#endif
});
