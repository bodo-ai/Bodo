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

        std::vector<uint8_t> scales({12, 13, 14, 15});
        std::vector<uint8_t> precisions({31, 30, 29, 28});
        std::vector<std::unique_ptr<bodo::DataType>> decimals;

        for (size_t i = 0; i < scales.size(); i++) {
            decimals.push_back(bodo::DataType(nullable_int_bool, decimal,
                                              precisions[i], scales[i])
                                   .copy());
        }

        bodo::ArrayType array1(decimals[0]->copy());
        bodo::ArrayType array2(decimals[1]->copy());
        bodo::MapType map1(array2.copy(), decimals[2]->copy());

        std::vector<std::unique_ptr<bodo::DataType>> child_types;
        child_types.push_back(array1.copy());
        child_types.push_back(map1.copy());
        child_types.push_back(decimals[3]->copy());

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
                {bodo_array_type::STRUCT, 3, bodo_array_type::ARRAY_ITEM,
                 bodo_array_type::NULLABLE_INT_BOOL, precisions[0], scales[0],
                 bodo_array_type::MAP, bodo_array_type::ARRAY_ITEM,
                 bodo_array_type::NULLABLE_INT_BOOL, precisions[1], scales[1],
                 bodo_array_type::NULLABLE_INT_BOOL, precisions[2], scales[2],
                 bodo_array_type::NULLABLE_INT_BOOL, precisions[3], scales[3]});
        std::shared_ptr<array_info> expected_c_types =
            bodo::tests::cppToBodoArr(
                {Bodo_CTypes::STRUCT, 3, Bodo_CTypes::LIST,
                 Bodo_CTypes::DECIMAL, precisions[0], scales[0],
                 Bodo_CTypes::MAP, Bodo_CTypes::LIST, Bodo_CTypes::DECIMAL,
                 precisions[1], scales[1], Bodo_CTypes::DECIMAL, precisions[2],
                 scales[2], Bodo_CTypes::DECIMAL, precisions[3], scales[3]});

        std::stringstream ss_arr_expected, ss_ctype_expected;
        DEBUG_PrintColumn(ss_arr_expected, new_array_types_info);
        DEBUG_PrintColumn(ss_ctype_expected, new_c_types_info);

        bodo::tests::check(ss_arr.str() == ss_arr_expected.str());
        bodo::tests::check(ss_ctype.str() == ss_ctype_expected.str());

        bodo::tests::check(new_ss_arr.str() == ss_arr.str());
        bodo::tests::check(new_ss_ctype.str() == ss_ctype.str());
    });
});
