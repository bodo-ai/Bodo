#include <memory>
#include <sstream>
#include <vector>
#include "../libs/_bodo_common.h"
#include "./test.hpp"
#include "table_generator.hpp"

// tests that serialization/deserialization works correctly for nested types
static bodo::tests::suite tests([] {
    bodo::tests::test("test_serialize_deserialize", [] {
        // creates a struct type that has decimals nested at various levels and
        // checks that ->deserialize ->serialize is equal to the serialized
        // type.
        bodo_array_type::arr_type_enum nullable_int_bool =
            bodo_array_type::NULLABLE_INT_BOOL;
        Bodo_CTypes::CTypeEnum decimal = Bodo_CTypes::DECIMAL;

        std::vector<uint8_t> scales({12, 13, 14});
        std::vector<uint8_t> precisions({31, 30, 29});
        std::vector<std::unique_ptr<bodo::DataType>> decimals;

        for (size_t i = 0; i < scales.size(); i++) {
            decimals.push_back(bodo::DataType(nullable_int_bool, decimal,
                                              precisions[i], scales[i])
                                   .copy());
        }
        auto date_type =
            std::make_unique<bodo::DataType>(bodo_array_type::NULLABLE_INT_BOOL,
                                             Bodo_CTypes::DATETIME, -1, -1, "");
        auto date_type_tz = std::make_unique<bodo::DataType>(
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::DATETIME, -1, -1,
            "UTC");

        bodo::ArrayType array1(decimals[0]->copy());
        bodo::ArrayType array2(date_type->copy());
        bodo::MapType map1(array2.copy(), decimals[1]->copy());

        std::vector<std::unique_ptr<bodo::DataType>> child_types;
        child_types.push_back(array1.copy());
        child_types.push_back(map1.copy());
        child_types.push_back(date_type_tz->copy());
        child_types.push_back(decimals[2]->copy());

        bodo::StructType struct1(std::move(child_types));
        std::vector<int8_t> array_types, c_types;
        struct1.Serialize(array_types, c_types);

        std::shared_ptr<array_info> array_types_info =
            bodo::tests::cppToBodoArr(array_types);
        std::shared_ptr<array_info> c_types_info =
            bodo::tests::cppToBodoArr(c_types);

        std::stringstream ss_arr, ss_ctype;
        DEBUG_PrintColumn(ss_arr, array_types_info);
        DEBUG_PrintColumn(ss_ctype, c_types_info);

        // deserialize then serialize
        std::unique_ptr<bodo::DataType> deserialized_type =
            bodo::DataType::Deserialize(array_types, c_types);

        std::vector<int8_t> new_arr_types, new_c_types;
        deserialized_type->Serialize(new_arr_types, new_c_types);

        std::shared_ptr<array_info> new_array_types_info =
            bodo::tests::cppToBodoArr(array_types);
        std::shared_ptr<array_info> new_c_types_info =
            bodo::tests::cppToBodoArr(c_types);

        std::stringstream new_ss_arr, new_ss_ctype;
        DEBUG_PrintColumn(new_ss_arr, new_array_types_info);
        DEBUG_PrintColumn(new_ss_ctype, new_c_types_info);

        // The Serialized struct type should match with these as well
        std::shared_ptr<array_info> expected_array_types =
            bodo::tests::cppToBodoArr(
                std::vector<int8_t>({bodo_array_type::STRUCT,
                                     /* struct fields*/ 4,
                                     bodo_array_type::ARRAY_ITEM,
                                     bodo_array_type::NULLABLE_INT_BOOL,
                                     static_cast<int8_t>(precisions[0]),
                                     static_cast<int8_t>(scales[0]),
                                     bodo_array_type::MAP,
                                     bodo_array_type::ARRAY_ITEM,
                                     bodo_array_type::NULLABLE_INT_BOOL,
                                     /* date_type tz*/ 0,
                                     bodo_array_type::NULLABLE_INT_BOOL,
                                     static_cast<int8_t>(precisions[1]),
                                     static_cast<int8_t>(scales[1]),
                                     bodo_array_type::NULLABLE_INT_BOOL,
                                     /* date_type_tz tz*/ 3,
                                     85,
                                     84,
                                     67,
                                     bodo_array_type::NULLABLE_INT_BOOL,
                                     static_cast<int8_t>(precisions[2]),
                                     static_cast<int8_t>(scales[2])}));
        std::shared_ptr<array_info> expected_c_types =
            bodo::tests::cppToBodoArr(
                std::vector<int8_t>({Bodo_CTypes::STRUCT,
                                     4,
                                     Bodo_CTypes::LIST,
                                     Bodo_CTypes::DECIMAL,
                                     static_cast<int8_t>(precisions[0]),
                                     static_cast<int8_t>(scales[0]),
                                     Bodo_CTypes::MAP,
                                     Bodo_CTypes::LIST,
                                     Bodo_CTypes::DATETIME,
                                     0,
                                     Bodo_CTypes::DECIMAL,
                                     static_cast<int8_t>(precisions[1]),
                                     static_cast<int8_t>(scales[1]),
                                     Bodo_CTypes::DATETIME,
                                     /* date_type_tz tz*/ 3,
                                     85,
                                     84,
                                     67,
                                     Bodo_CTypes::DECIMAL,
                                     static_cast<int8_t>(precisions[2]),
                                     static_cast<int8_t>(scales[2])}));

        std::stringstream ss_arr_expected, ss_ctype_expected;
        DEBUG_PrintColumn(ss_arr_expected, expected_array_types);
        DEBUG_PrintColumn(ss_ctype_expected, expected_c_types);

        bodo::tests::check(ss_arr.str() == ss_arr_expected.str());
        bodo::tests::check(ss_ctype.str() == ss_ctype_expected.str());

        bodo::tests::check(new_ss_arr.str() == ss_arr.str());
        bodo::tests::check(new_ss_ctype.str() == ss_ctype.str());
    });
});
