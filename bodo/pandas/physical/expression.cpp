#include "expression.h"

duckdb::ExpressionType exprSwitchLeftRight(duckdb::ExpressionType etype) {
    switch (etype) {
        case duckdb::ExpressionType::COMPARE_EQUAL:
        case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            return etype;
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
            return duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO;
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            return duckdb::ExpressionType::COMPARE_LESSTHAN;
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            return duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO;
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return duckdb::ExpressionType::COMPARE_GREATERTHAN;
        default:
            throw std::runtime_error(
                "switchLeftRight doesn't handle expression type " +
                std::to_string(static_cast<int>(etype)));
    }
}

void compare_one_array_string(std::shared_ptr<array_info> arr1,
                              const std::string data2, uint8_t *output,
                              const std::function<bool(int)> &comparator) {
    bool na_position = false;

    assert(arr1->arr_type == bodo_array_type::STRING);
    bodo::vector<std::string> svec = {data2};
    std::vector<bool> nulls = {true};
    std::shared_ptr<array_info> arr2 =
        string_array_from_vector(svec, nulls, Bodo_CTypes::STRING);
    for (uint64_t i = 0; i < arr1->length; ++i) {
        int test = KeyComparisonAsPython_Column(na_position, arr1, i, arr2, 0);
        SetBitTo(output, i, comparator(test));
    }
}

void compare_two_array(std::shared_ptr<array_info> arr1,
                       const std::shared_ptr<array_info> arr2, uint8_t *output,
                       const std::function<bool(int)> &comparator) {
    int64_t n_rows = arr1->length;
    uint64_t arr1_siztype = numpy_item_size[arr1->dtype];
    char *arr1_data1 = arr1->data1();
    char *arr1_data1_end = arr1_data1 + (n_rows * arr1_siztype);
    char *arr2_data1 = arr2->data1();
    uint64_t arr2_siztype = numpy_item_size[arr2->dtype];
    assert(arr1->length == arr2->length);
    assert(arr1->dtype == arr2->dtype);
    bool na_position = false;

    if (is_numerical(arr1->dtype)) {
        std::function<int(const char *, const char *, bool const &)> ncfunc =
            getNumericComparisonFunc(arr1->dtype);
        for (uint64_t i = 0; arr1_data1 < arr1_data1_end;
             arr1_data1 += arr1_siztype, arr2_data1 += arr2_siztype, ++i) {
            int test = ncfunc(arr1_data1, arr2_data1, na_position);
            SetBitTo(output, i, comparator(test));
        }
    } else if (arr1->arr_type == bodo_array_type::STRING) {
        for (uint64_t i = 0; i < arr1->length; ++i) {
            int test =
                KeyComparisonAsPython_Column(na_position, arr1, i, arr2, i);
            SetBitTo(output, i, comparator(test));
        }
    }
}

std::function<bool(int)> equal_test = [](int test) { return test == 0; };
std::function<bool(int)> not_equal_test = [](int test) { return test != 0; };
std::function<bool(int)> greater_test = [](int test) { return test < 0; };
std::function<bool(int)> less_test = [](int test) { return test > 0; };
std::function<bool(int)> greater_equal_test = [](int test) {
    return test <= 0;
};
std::function<bool(int)> less_equal_test = [](int test) { return test >= 0; };
