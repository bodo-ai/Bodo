#include "_table_builder_utils.h"

#include "_array_build_buffer.h"
#include "_bodo_common.h"
#include "_dict_builder.h"

std::unique_ptr<array_info> alloc_empty_array(
    const std::unique_ptr<bodo::DataType>& datatype,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm,
    std::shared_ptr<DictionaryBuilder> dict_builder) {
    if (datatype->is_array()) {
        auto array_type = static_cast<bodo::ArrayType*>(datatype.get());
        auto inner_arr = alloc_empty_array(
            array_type->value_type, pool, mm,
            dict_builder ? dict_builder->child_dict_builders[0] : nullptr);
        return alloc_array_item(0, std::move(inner_arr), 0, pool, mm);

    } else if (datatype->is_struct()) {
        auto struct_type = static_cast<bodo::StructType*>(datatype.get());

        std::vector<std::shared_ptr<array_info>> arrs;
        for (size_t i = 0; i < struct_type->child_types.size(); i++) {
            auto& datatype = struct_type->child_types[i];
            arrs.push_back(alloc_empty_array(
                datatype, pool, mm,
                dict_builder ? dict_builder->child_dict_builders[i] : nullptr));
        };

        return alloc_struct(0, arrs, 0, pool, mm);
    } else if (datatype->is_map()) {
        bodo::MapType* map_type = static_cast<bodo::MapType*>(datatype.get());
        std::shared_ptr<DictionaryBuilder> list_dict_builder = nullptr;
        std::shared_ptr<DictionaryBuilder> struct_dict_builder = nullptr;
        std::shared_ptr<DictionaryBuilder> key_dict_builder = nullptr;
        std::shared_ptr<DictionaryBuilder> value_dict_builder = nullptr;
        if (dict_builder) {
            list_dict_builder = dict_builder->child_dict_builders[0];
            struct_dict_builder = list_dict_builder->child_dict_builders[0];
            key_dict_builder = struct_dict_builder->child_dict_builders[0];
            value_dict_builder = struct_dict_builder->child_dict_builders[1];
        }

        std::unique_ptr<array_info> key_arr =
            alloc_empty_array(map_type->key_type, pool, mm, key_dict_builder);
        std::unique_ptr<array_info> value_arr = alloc_empty_array(
            map_type->value_type, pool, mm, value_dict_builder);
        std::unique_ptr<array_info> struct_arr = alloc_struct(
            0, {std::move(key_arr), std::move(value_arr)}, 0, pool, mm);
        std::unique_ptr<array_info> array_item_arr =
            alloc_array_item(0, std::move(struct_arr), 0, pool, mm);
        return alloc_map(0, std::move(array_item_arr));
    } else {
        std::unique_ptr<array_info> array_out = alloc_array_top_level(
            0, 0, 0, datatype->array_type, datatype->c_type, -1, 0, 0, false,
            false, false, pool, mm);
        array_out->precision = datatype->precision;
        array_out->scale = datatype->scale;
        if (dict_builder) {
            assert(datatype->array_type == bodo_array_type::DICT);
            array_out->child_arrays[0] = dict_builder->dict_buff->data_array;
        }
        return array_out;
    }
}

std::shared_ptr<table_info> alloc_table(
    const std::shared_ptr<bodo::Schema>& schema, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm,
    std::vector<std::shared_ptr<DictionaryBuilder>>* dict_builders) {
    std::vector<std::shared_ptr<array_info>> arrays;

    for (size_t i = 0; i < schema->column_types.size(); i++) {
        auto& arr_type = schema->column_types[i];
        arrays.push_back(alloc_empty_array(
            arr_type, pool, mm,
            dict_builders == nullptr ? nullptr : (*dict_builders)[i]));
    }

    return std::make_shared<table_info>(arrays, 0, schema->column_names,
                                        schema->metadata);
}

std::shared_ptr<table_info> alloc_table_like(
    const std::shared_ptr<table_info>& table, const bool reuse_dictionaries,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> arrays;
    arrays.reserve(table->ncols());
    for (auto& in_arr : table->columns) {
        arrays.push_back(alloc_array_like(in_arr, true, pool, mm));
    }
    return std::make_shared<table_info>(arrays, table->nrows(),
                                        table->column_names, table->metadata);
}

std::shared_ptr<table_info> alloc_table_like(
    const std::shared_ptr<bodo::Schema>& schema, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    std::vector<std::shared_ptr<array_info>> arrays;
    arrays.reserve(schema->ncols());
    for (auto& dtype : schema->column_types) {
        arrays.emplace_back(alloc_empty_array(dtype));
    }
    return std::make_shared<table_info>(arrays, 0, schema->column_names,
                                        schema->metadata);
}

std::shared_ptr<table_info> unify_dictionary_arrays_helper(
    const std::shared_ptr<table_info>& in_table,
    std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    uint64_t n_keys, bool only_transpose_existing_on_key_cols) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(in_table->ncols());
    for (size_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        std::shared_ptr<array_info> out_arr;
        if (dict_builders[i] == nullptr) {
            out_arr = in_arr;
        } else {
            if (only_transpose_existing_on_key_cols && (i < n_keys)) {
                out_arr = dict_builders[i]->TransposeExisting(in_arr);
            } else {
                out_arr = dict_builders[i]->UnifyDictionaryArray(in_arr);
            }
        }
        out_arrs.emplace_back(out_arr);
    }
    return std::make_shared<table_info>(out_arrs);
}
