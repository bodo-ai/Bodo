#include "_table_builder_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"

std::unique_ptr<array_info> alloc_empty_array(
    const std::unique_ptr<bodo::DataType>& datatype,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    if (datatype->is_array()) {
        auto array_type = static_cast<bodo::ArrayType*>(datatype.get());
        auto inner_arr = alloc_empty_array(array_type->value_type, pool, mm);
        return alloc_array_item(0, std::move(inner_arr), pool, mm);

    } else if (datatype->is_struct()) {
        auto struct_type = static_cast<bodo::StructType*>(datatype.get());

        std::vector<std::shared_ptr<array_info>> arrs;
        for (auto& datatype : struct_type->child_types) {
            arrs.push_back(alloc_empty_array(datatype, pool, mm));
        };

        return alloc_struct(0, arrs, pool, mm);
    } else {
        return alloc_array_top_level(0, 0, 0, datatype->array_type,
                                     datatype->c_type, -1, 0, 0, false, false,
                                     false, pool, mm);
    }
}

std::shared_ptr<table_info> alloc_table(
    const std::vector<int8_t>& arr_c_types,
    const std::vector<int8_t>& arr_array_types, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    const std::span<const int8_t> arr_types_span{arr_array_types};
    const std::span<const int8_t> c_types_span{arr_c_types};

    auto schema = bodo::Schema::Deserialize(arr_types_span, c_types_span);
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
