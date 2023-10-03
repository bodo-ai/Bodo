#include "_table_builder_utils.h"
#include "_bodo_common.h"

std::shared_ptr<table_info> alloc_table(
    const std::vector<int8_t>& arr_c_types,
    const std::vector<int8_t>& arr_array_types, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> arrays;
    std::vector<size_t> col_to_idx_map(get_col_idx_map(arr_array_types));

    for (size_t i = 0; i < col_to_idx_map.size(); i++) {
        size_t start_idx = col_to_idx_map[i];
        size_t end_idx =
            (i == col_to_idx_map.size() - 1 ? arr_array_types.size()
                                            : col_to_idx_map[i + 1]);
        if (arr_array_types[start_idx] == bodo_array_type::ARRAY_ITEM) {
            arrays.push_back(alloc_array_item(
                0, start_idx, end_idx, arr_array_types, arr_c_types, pool, mm));
        } else if (arr_array_types[start_idx] == bodo_array_type::STRUCT) {
            arrays.push_back(alloc_struct(
                0, start_idx, end_idx, arr_array_types, arr_c_types, pool, mm));
        } else {
            arrays.push_back(alloc_array(
                0, 0, 0,
                (bodo_array_type::arr_type_enum)arr_array_types[start_idx],
                (Bodo_CTypes::CTypeEnum)arr_c_types[start_idx], -1, 0, 0, false,
                false, false, pool, mm));
        }
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
