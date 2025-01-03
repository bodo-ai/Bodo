/// Simple test suite for bodo C++ code.
///
/// This currently uses a simple exception based 'check' function to check for
/// invariants.
///
/// In the future, it may be worth integrating a 'real' testing library like
/// gtest, boost ut, or boost test. At the time of writing, these were deemed
/// not worth the effort.

#pragma once

#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <source_location>
#include <string>
#include <vector>

#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"

namespace bodo::tests {

/// @brief A single test case. Use the 'test()' function instead of constructing
/// this directly.
class test_case {
   public:
    /// @brief Construct a test case based on a callable and a line number
    /// @param f Callable that can be used to run the test. Should throw an
    /// exception on failure
    /// @param lineno Line in the file where the test is defined
    test_case(std::function<void()> f, int lineno,
              std::vector<std::string> &markers)
        : func_(f), lineno_(lineno), markers_(markers) {}

    std::function<void()> func_;
    int lineno_;
    std::vector<std::string> markers_;
};

/// @brief Logically groups tests into groups. Each c++ test file should contain
/// one suite.
class suite {
   public:
    /// @brief Construct a test suite. Each c++ test file should contain one
    /// global 'suite' that is statically defined (so as not to cause name
    /// conflicts).
    ///
    /// For example, a test_example.cpp file may contain:
    ///
    ///  static bodo::test::suite examples([]{
    ///     // define tests here usinge test()
    ///  })
    ///
    /// @tparam T Callable type
    /// @param initializer This function is called to construct all tests in the
    /// suite. It should call the `test` function to define individual tests.
    /// @param no_set_current for testing only - skips modifying global suite
    /// registry
    /// @param location Automatically set to the current source location
    template <typename T>
    suite(T initializer, bool no_set_current = false,
          const std::source_location location = std::source_location::current())
        : filename_(location.file_name()) {
        if (!no_set_current) {
            set_current(this);
        }
        initializer();
    }

    static suite *get_current();
    static const std::vector<suite *> &get_all();

    /// @brief Add a test into the suite. Don't call this directly. Instead use
    /// 'test' in the initializer argument to the suite constructor.
    /// @tparam T The callable type
    /// @param nm Test name (the thing that goes after the :: in a pytest test
    /// name)
    /// @param func Callable to run the test. Should throw an exception on error
    /// @param lineno Line in the source file where the test was defined.
    template <typename T>
    void add_test(const std::string &nm, T func,
                  std::vector<std::string> &markers, int lineno) {
        auto wrapped_test = [&]() {
            if (before_each_) {
                (*before_each_)();
            }

            func();

            if (after_each_) {
                (*after_each_)();
            }
        };
        tests_.insert(
            std::make_pair(nm, test_case(wrapped_test, lineno, markers)));
    }

    /// @brief Register a callback to fire after every test in the suite
    /// Note that this must come before any test definitions.
    /// @tparam T The callable type
    /// @param func Callable to run after every test
    template <typename T>
    void after_each(T func) {
        if (!tests_.empty()) {
            std::cerr << "Cannot register after_each after registering tests"
                      << std::endl;
            throw new std::exception();
        }
        after_each_ = func;
    }

    /// @brief Register a callback to fire before every test in the suite
    /// Note that this must come before any test definitions.
    /// @tparam T The callable type
    /// @param func Callable to run before every test
    template <typename T>
    void before_each(T func) {
        if (!tests_.empty()) {
            std::cerr << "Cannot register before_each after registering tests"
                      << std::endl;
            throw new std::exception();
        }
        before_each_ = func;
    }

    /// @brief Get all tests by name
    /// @return An std::map mapping test names to test_case's
    const std::map<std::string, test_case> &tests() const { return tests_; }

    /// @brief Get the name of the file the suite was defined in
    /// @return The name of the file from which the suite was constructed.
    const std::string &name() const { return filename_; }

   private:
    std::string filename_;
    std::map<std::string, test_case> tests_;

    std::optional<std::function<void()>> before_each_;
    std::optional<std::function<void()>> after_each_;

    static void set_current(suite *st);
};

/// @brief Define a test case. This should only be called in the initializer
/// callback to the suite constructor.
/// @tparam TestFunc Callable type for the test case
/// @param nm Name of the test. This is what goes after the :: in the pytest
/// name
/// @param func The callable to run when this test is invoked. Should throw an
/// exception if the test fails.
/// @param location The location in the source that defines this test. Do not
/// set. Defaults to the calling source location.
template <typename TestFunc>
void test(const std::string &nm, TestFunc func,
          std::vector<std::string> markers = {},
          std::source_location location = std::source_location::current()) {
    suite::get_current()->add_test(nm, func, markers, location.line());
}

/// @brief Register a callback to fire after every test in the suite
/// Note that this must come before any test definitions.
/// @tparam TestFunc The callable type
/// @param func Callable to run after every test
template <typename TestFunc>
void after_each(TestFunc func) {
    suite::get_current()->after_each(func);
}

/// @brief Register a callback to fire before every test in the suite
/// Note that this must come before any test definitions.
/// @tparam TestFunc The callable type
/// @param func Callable to run before every test
template <typename TestFunc>
void before_each(TestFunc func) {
    suite::get_current()->before_each(func);
}

/// @brief Helper function to throw an exception and report an error if a
/// condition fails
/// @param x The condition to check. If true, simply returns. Otherwise, fails
/// and reports an error
/// @param location Do not set. Set automatically to the calling source
/// location.
void check(bool x, const std::source_location location =
                       std::source_location::current());

/// @brief Same as check(x, location) but customize the error message
/// @param x The condition to check
/// @param message The message to use instead of the default
/// @param location Do not set. Set automatically to the calling source
/// location.
void check(
    bool x, const char *message,
    const std::source_location location = std::source_location::current());

/** @brief Variant of check that ensures all ranks pass.
 */
void check_parallel(bool x, const std::source_location location =
                                std::source_location::current());

/// @brief Helper function to check if an exception was thrown by input callable
/// @param f Callable function to check if exception was raised in
/// @param expected_msg_start The expected start of the exception message
void check_exception(std::function<void()> f, const char *expected_msg_start);

}  // namespace bodo::tests

// Create an array of the desired type without directly setting the null bit
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(nullable_array<ArrType>)
std::shared_ptr<array_info> make_arr(size_t n) {
    return alloc_nullable_array(n, DType);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(numpy_array<ArrType>)
std::shared_ptr<array_info> make_arr(size_t n) {
    return alloc_numpy(n, DType);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(string_array<ArrType>)
std::shared_ptr<array_info> make_arr(size_t n) {
    return alloc_string_array(DType, n, 0);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(dict_array<ArrType>)
std::shared_ptr<array_info> make_arr(size_t n) {
    return alloc_dict_string_array(n, 0, 0);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(ArrType == bodo_array_type::ARRAY_ITEM)
std::shared_ptr<array_info> make_arr(size_t n) {
    return alloc_array_item(n, alloc_numpy(0, DType));
}

// Create an array of the desired type with all null elements
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(nullable_array<ArrType>)
std::shared_ptr<array_info> make_all_null_arr(size_t n) {
    return alloc_nullable_array_all_nulls(n, DType);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(numpy_array<ArrType>)
std::shared_ptr<array_info> make_all_null_arr(size_t n) {
    return alloc_numpy(n, DType);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(string_array<ArrType>)
std::shared_ptr<array_info> make_all_null_arr(size_t n) {
    return alloc_string_array_all_nulls(DType, n);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(dict_array<ArrType>)
std::shared_ptr<array_info> make_all_null_arr(size_t n) {
    return alloc_dict_string_array_all_nulls(n);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(array_item_array<ArrType>)
std::shared_ptr<array_info> make_all_null_arr(size_t n) {
    return alloc_array_item_all_nulls(n, alloc_numpy(0, DType));
}

using empty_return_enum = enum {
    ZERO,
    ONE,
    NULL_OUTPUT,
    EMPTY_STRING,
    EMPTY_ARRAY,
    EMPTY_MAP,
};

// Create an array of the desired type with all 0 elements. Only works
// on nullable/numpy arrays with a numeric dtype.
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          empty_return_enum RetType>
    requires(RetType == empty_return_enum::ZERO)
std::shared_ptr<array_info> make_result_output(size_t n) {
    using T = typename dtype_to_type<DType>::type;
    std::shared_ptr<array_info> res = make_arr<ArrType, DType>(n);
    for (size_t i = 0; i < n; i++) {
        set_non_null<ArrType, T, DType>(*res, i);
        set_arr_item<ArrType, T, DType>(*res, i, static_cast<T>(0));
    }
    return res;
}

// Create an array of the desired type with all 1 elements. Only works
// on nullable/numpy arrays with a numeric dtype.
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          empty_return_enum RetType>
    requires(RetType == empty_return_enum::ONE)
std::shared_ptr<array_info> make_result_output(size_t n) {
    using T = typename dtype_to_type<DType>::type;
    std::shared_ptr<array_info> res = make_arr<ArrType, DType>(n);
    for (size_t i = 0; i < n; i++) {
        set_non_null<ArrType, T, DType>(*res, i);
        set_arr_item<ArrType, T, DType>(*res, i, static_cast<T>(1));
    }
    return res;
}

// Create an array of the desired type with all null elements. Should work
// on all array types.
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          empty_return_enum RetType>
    requires(RetType == empty_return_enum::NULL_OUTPUT)
std::shared_ptr<array_info> make_result_output(size_t n) {
    return make_all_null_arr<ArrType, DType>(n);
}

// Create an array of the desired type with all empty string elements. Only
// works when return type is a string array.
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          empty_return_enum RetType>
    requires(RetType == empty_return_enum::EMPTY_STRING)
std::shared_ptr<array_info> make_result_output(size_t n) {
    bodo::vector<uint8_t> nulls((n + 7) >> 3, 0xff);
    bodo::vector<std::string> strings(n, "");
    return create_string_array(DType, nulls, strings);
}

// Create an array of the desired type with all empty array elements. Only
// works when return type is an array item array.
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          empty_return_enum RetType>
    requires(RetType == empty_return_enum::EMPTY_ARRAY)
std::shared_ptr<array_info> make_result_output(size_t n) {
    std::shared_ptr<array_info> inner_arr;
    if (DType == Bodo_CTypes::STRUCT) {
        std::shared_ptr<array_info> dummy_inner =
            make_result_output<bodo_array_type::STRING, Bodo_CTypes::STRING,
                               empty_return_enum::EMPTY_STRING>(0);
        inner_arr = alloc_struct(0, {dummy_inner, dummy_inner});
    } else {
        inner_arr = alloc_numpy(0, DType);
    }
    std::shared_ptr<array_info> array_arr = alloc_array_item(n, inner_arr);
    offset_t *offsets =
        (offset_t *)(array_arr->buffers[0]->mutable_data() + array_arr->offset);
    for (size_t i = 0; i < n + 1; i++) {
        offsets[i] = 0;
    }
    return array_arr;
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          empty_return_enum RetType>
    requires(RetType == empty_return_enum::EMPTY_MAP)
std::shared_ptr<array_info> make_result_output(size_t n) {
    std::shared_ptr<array_info> inner_arr =
        make_result_output<bodo_array_type::ARRAY_ITEM, Bodo_CTypes::STRUCT,
                           empty_return_enum::EMPTY_ARRAY>(n);
    return alloc_map(n, inner_arr);
}
