#include "_table_builder_utils.h"

std::shared_ptr<table_info> alloc_table(
    const std::vector<int8_t>& arr_c_types,
    const std::vector<int8_t>& arr_array_types, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> arrays;

    for (size_t i = 0; i < arr_c_types.size(); i++) {
        bodo_array_type::arr_type_enum arr_type =
            (bodo_array_type::arr_type_enum)arr_array_types[i];
        Bodo_CTypes::CTypeEnum dtype = (Bodo_CTypes::CTypeEnum)arr_c_types[i];

        arrays.push_back(alloc_array(0, 0, 0, arr_type, dtype, -1, 0, 0, false,
                                     false, false, pool, mm));
    }
    return std::make_shared<table_info>(arrays);
}

std::shared_ptr<table_info> alloc_table_like(
    const std::shared_ptr<table_info>& table, const bool reuse_dictionaries) {
    std::vector<std::shared_ptr<array_info>> arrays;
    for (size_t i = 0; i < table->ncols(); i++) {
        bodo_array_type::arr_type_enum arr_type = table->columns[i]->arr_type;
        Bodo_CTypes::CTypeEnum dtype = table->columns[i]->dtype;
        arrays.push_back(alloc_array(0, 0, 0, arr_type, dtype));
        // For dict encoded columns, re-use the same dictionary
        // if reuse_dictionaries = true
        if (reuse_dictionaries && (arr_type == bodo_array_type::DICT)) {
            arrays[i]->child_arrays[0] = table->columns[i]->child_arrays[0];
        }
    }
    return std::make_shared<table_info>(arrays);
}
