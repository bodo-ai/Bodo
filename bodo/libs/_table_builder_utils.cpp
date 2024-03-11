#include "_table_builder_utils.h"

#include "_bodo_common.h"
#include "_bodo_to_arrow.h"

std::unique_ptr<array_info> alloc_empty_array(
    const std::unique_ptr<bodo::DataType>& datatype,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    if (datatype->is_array()) {
        auto array_type = static_cast<bodo::ArrayType*>(datatype.get());
        auto inner_arr = alloc_empty_array(array_type->value_type, pool, mm);
        return alloc_array_item(0, std::move(inner_arr), 0, pool, mm);

    } else if (datatype->is_struct()) {
        auto struct_type = static_cast<bodo::StructType*>(datatype.get());

        std::vector<std::shared_ptr<array_info>> arrs;
        for (auto& datatype : struct_type->child_types) {
            arrs.push_back(alloc_empty_array(datatype, pool, mm));
        };

        return alloc_struct(0, arrs, 0, pool, mm);
    } else if (datatype->is_map()) {
        bodo::MapType* map_type = static_cast<bodo::MapType*>(datatype.get());
        std::unique_ptr<array_info> key_arr =
            alloc_empty_array(map_type->key_type, pool, mm);
        std::unique_ptr<array_info> value_arr =
            alloc_empty_array(map_type->value_type, pool, mm);
        std::unique_ptr<array_info> struct_arr = alloc_struct(
            0, {std::move(key_arr), std::move(value_arr)}, 0, pool, mm);
        std::unique_ptr<array_info> array_item_arr =
            alloc_array_item(0, std::move(struct_arr), 0, pool, mm);
        return alloc_map(0, std::move(array_item_arr));
    } else {
        return alloc_array_top_level(0, 0, 0, datatype->array_type,
                                     datatype->c_type, -1, 0, 0, false, false,
                                     false, pool, mm);
    }
}

std::shared_ptr<table_info> alloc_table(
    const std::shared_ptr<bodo::Schema>& schema, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> arrays;

    for (auto& arr_type : schema->column_types) {
        arrays.push_back(alloc_empty_array(arr_type, pool, mm));
    }

    return std::make_shared<table_info>(arrays);
}

std::shared_ptr<table_info> alloc_table_like(
    const std::shared_ptr<table_info>& table, const bool reuse_dictionaries) {
    std::vector<std::shared_ptr<array_info>> arrays;
    arrays.reserve(table->ncols());
    for (auto& in_arr : table->columns) {
        arrays.push_back(alloc_array_like(in_arr));
    }
    return std::make_shared<table_info>(arrays);
}
