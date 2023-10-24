#include "../libs/_bodo_common.h"
#include "./test.hpp"

static bodo::tests::suite tests([] {
    bodo::tests::test("test_basic", [] {
        std::vector<int8_t> arr_array_types{
            bodo_array_type::NULLABLE_INT_BOOL,
            bodo_array_type::STRING,
            bodo_array_type::NUMPY,
            bodo_array_type::STRING,
            bodo_array_type::LIST_STRING,
            bodo_array_type::DICT,
        };

        std::vector<int8_t> arr_c_types{
            Bodo_CTypes::INT64,  Bodo_CTypes::STRING,      Bodo_CTypes::_BOOL,
            Bodo_CTypes::BINARY, Bodo_CTypes::LIST_STRING, Bodo_CTypes::STRING,
        };

        auto schema = bodo::Schema::Deserialize(arr_array_types, arr_c_types);
        bodo::tests::check(schema->column_types.size() == 6);

        bodo::tests::check(schema->column_types[0]->array_type ==
                           bodo_array_type::NULLABLE_INT_BOOL);
        bodo::tests::check(schema->column_types[0]->c_type ==
                           Bodo_CTypes::INT64);
        bodo::tests::check(schema->column_types[0]->ToString() ==
                           "NULLABLE[INT64]");

        bodo::tests::check(schema->column_types[1]->array_type ==
                           bodo_array_type::STRING);
        bodo::tests::check(schema->column_types[1]->c_type ==
                           Bodo_CTypes::STRING);
        bodo::tests::check(schema->column_types[1]->ToString() ==
                           "STRING[STRING]");

        bodo::tests::check(schema->column_types[2]->array_type ==
                           bodo_array_type::NUMPY);
        bodo::tests::check(schema->column_types[2]->c_type ==
                           Bodo_CTypes::_BOOL);
        bodo::tests::check(schema->column_types[2]->ToString() ==
                           "NUMPY[_BOOL]");

        bodo::tests::check(schema->column_types[3]->array_type ==
                           bodo_array_type::STRING);
        bodo::tests::check(schema->column_types[3]->c_type ==
                           Bodo_CTypes::BINARY);
        bodo::tests::check(schema->column_types[3]->ToString() ==
                           "STRING[BINARY]");

        bodo::tests::check(schema->column_types[4]->array_type ==
                           bodo_array_type::LIST_STRING);
        bodo::tests::check(schema->column_types[4]->c_type ==
                           Bodo_CTypes::LIST_STRING);
        bodo::tests::check(schema->column_types[4]->ToString() ==
                           "LIST_STRING[LIST_STRING]");

        bodo::tests::check(schema->column_types[5]->array_type ==
                           bodo_array_type::DICT);
        bodo::tests::check(schema->column_types[5]->c_type ==
                           Bodo_CTypes::STRING);
        bodo::tests::check(schema->column_types[5]->ToString() ==
                           "DICT[STRING]");

        auto [out_array_types, out_c_types] = schema->Serialize();
        bodo::tests::check(out_array_types.size() == 6);
        bodo::tests::check(out_c_types.size() == 6);

        for (size_t i = 0; i < 6; i++) {
            bodo::tests::check(out_array_types[i] == arr_array_types[i]);
            bodo::tests::check(out_c_types[i] == arr_c_types[i]);
        }
    });

    bodo::tests::test("test_nested", [] {
        std::vector<int8_t> arr_array_types{
            bodo_array_type::NULLABLE_INT_BOOL,
            bodo_array_type::ARRAY_ITEM,
            bodo_array_type::STRING,
            bodo_array_type::STRUCT,
            3,
            bodo_array_type::NUMPY,
            bodo_array_type::LIST_STRING,
            bodo_array_type::DICT,
        };

        std::vector<int8_t> arr_c_types{
            Bodo_CTypes::INT64,
            Bodo_CTypes::LIST,
            Bodo_CTypes::STRING,
            Bodo_CTypes::STRUCT,
            3,
            Bodo_CTypes::_BOOL,
            Bodo_CTypes::LIST_STRING,
            Bodo_CTypes::STRING,
        };

        auto schema = bodo::Schema::Deserialize(arr_array_types, arr_c_types);
        bodo::tests::check(schema->column_types.size() == 3);

        bodo::tests::check(schema->column_types[0]->array_type ==
                           bodo_array_type::NULLABLE_INT_BOOL);
        bodo::tests::check(schema->column_types[0]->c_type ==
                           Bodo_CTypes::INT64);
        bodo::tests::check(schema->column_types[0]->ToString() ==
                           "NULLABLE[INT64]");

        bodo::tests::check(schema->column_types[1]->array_type ==
                           bodo_array_type::ARRAY_ITEM);
        bodo::tests::check(schema->column_types[1]->c_type ==
                           Bodo_CTypes::LIST);
        auto list_type =
            static_cast<bodo::ArrayType*>(schema->column_types[1].get());
        bodo::tests::check(list_type->value_type->array_type ==
                           bodo_array_type::STRING);
        bodo::tests::check(list_type->value_type->c_type ==
                           Bodo_CTypes::STRING);
        bodo::tests::check(schema->column_types[1]->ToString() ==
                           "ARRAY_ITEM[STRING[STRING]]");

        bodo::tests::check(schema->column_types[2]->array_type ==
                           bodo_array_type::STRUCT);
        bodo::tests::check(schema->column_types[2]->c_type ==
                           Bodo_CTypes::STRUCT);
        auto struct_type =
            static_cast<bodo::StructType*>(schema->column_types[2].get());

        bodo::tests::check(struct_type->child_types[0]->array_type ==
                           bodo_array_type::NUMPY);
        bodo::tests::check(struct_type->child_types[0]->c_type ==
                           Bodo_CTypes::_BOOL);

        bodo::tests::check(struct_type->child_types[1]->array_type ==
                           bodo_array_type::LIST_STRING);
        bodo::tests::check(struct_type->child_types[1]->c_type ==
                           Bodo_CTypes::LIST_STRING);

        bodo::tests::check(struct_type->child_types[2]->array_type ==
                           bodo_array_type::DICT);
        bodo::tests::check(struct_type->child_types[2]->c_type ==
                           Bodo_CTypes::STRING);
        bodo::tests::check(schema->column_types[2]->ToString() ==
                           "STRUCT[0: NUMPY[_BOOL], 1: "
                           "LIST_STRING[LIST_STRING], 2: DICT[STRING]]");

        auto [out_array_types, out_c_types] = schema->Serialize();
        bodo::tests::check(out_array_types.size() == 8);
        bodo::tests::check(out_c_types.size() == 8);

        for (size_t i = 0; i < 8; i++) {
            bodo::tests::check(out_array_types[i] == arr_array_types[i]);
            bodo::tests::check(out_c_types[i] == arr_c_types[i]);
        }
    });

    bodo::tests::test("test_multiple_nesting", [] {
        std::vector<int8_t> arr_array_types{
            bodo_array_type::NULLABLE_INT_BOOL,
            bodo_array_type::ARRAY_ITEM,
            bodo_array_type::STRUCT,
            2,
            bodo_array_type::NUMPY,
            bodo_array_type::ARRAY_ITEM,
            bodo_array_type::DICT,
            bodo_array_type::STRING,
        };

        std::vector<int8_t> arr_c_types{
            Bodo_CTypes::INT32,  Bodo_CTypes::LIST,
            Bodo_CTypes::STRUCT, 2,
            Bodo_CTypes::_BOOL,  Bodo_CTypes::LIST,
            Bodo_CTypes::STRING, Bodo_CTypes::BINARY,
        };

        auto schema = bodo::Schema::Deserialize(arr_array_types, arr_c_types);
        bodo::tests::check(schema->column_types.size() == 3);
        bodo::tests::check(schema->column_types[0]->array_type ==
                           bodo_array_type::NULLABLE_INT_BOOL);
        bodo::tests::check(schema->column_types[0]->c_type ==
                           Bodo_CTypes::INT32);
        bodo::tests::check(schema->column_types[0]->ToString() ==
                           "NULLABLE[INT32]");

        bodo::tests::check(schema->column_types[2]->array_type ==
                           bodo_array_type::STRING);
        bodo::tests::check(schema->column_types[2]->c_type ==
                           Bodo_CTypes::BINARY);
        bodo::tests::check(schema->column_types[2]->ToString() ==
                           "STRING[BINARY]");

        bodo::tests::check(schema->column_types[1]->array_type ==
                           bodo_array_type::ARRAY_ITEM);
        bodo::tests::check(schema->column_types[1]->c_type ==
                           Bodo_CTypes::LIST);
        auto list_type =
            static_cast<bodo::ArrayType*>(schema->column_types[1].get());

        bodo::tests::check(list_type->value_type->array_type ==
                           bodo_array_type::STRUCT);
        bodo::tests::check(list_type->value_type->c_type ==
                           Bodo_CTypes::STRUCT);
        auto struct_type =
            static_cast<bodo::StructType*>(list_type->value_type.get());

        bodo::tests::check(struct_type->child_types[0]->array_type ==
                           bodo_array_type::NUMPY);
        bodo::tests::check(struct_type->child_types[0]->c_type ==
                           Bodo_CTypes::_BOOL);

        bodo::tests::check(struct_type->child_types[1]->array_type ==
                           bodo_array_type::ARRAY_ITEM);
        bodo::tests::check(struct_type->child_types[1]->c_type ==
                           Bodo_CTypes::LIST);
        auto list_type2 =
            static_cast<bodo::ArrayType*>(struct_type->child_types[1].get());

        bodo::tests::check(list_type2->value_type->array_type ==
                           bodo_array_type::DICT);
        bodo::tests::check(list_type2->value_type->c_type ==
                           Bodo_CTypes::STRING);
        bodo::tests::check(
            schema->column_types[1]->ToString() ==
            "ARRAY_ITEM[STRUCT[0: NUMPY[_BOOL], 1: ARRAY_ITEM[DICT[STRING]]]]");

        auto [out_array_types, out_c_types] = schema->Serialize();
        bodo::tests::check(out_array_types.size() == 8);
        bodo::tests::check(out_c_types.size() == 8);

        for (size_t i = 0; i < 8; i++) {
            bodo::tests::check(out_array_types[i] == arr_array_types[i]);
            bodo::tests::check(out_c_types[i] == arr_c_types[i]);
        }
    });
});
