
#include "_listagg.h"
#include "_array_operations.h"
#include "_bodo_common.h"

/**
 * @brief Collection of templated helper functions that
 */
template <bodo_array_type::arr_type_enum arr_typ>
struct listagg_seq_utils {
    /**
     * @brief Returns the string array for the input column.
     */
    inline static std::shared_ptr<array_info> get_string_arr(
        const std::shared_ptr<array_info> &in_col) {
        if constexpr (arr_typ == bodo_array_type::arr_type_enum::DICT) {
            return in_col->child_arrays[0];
        } else {
            return in_col;
        }
    }

    /**
     * @brief Determines the offset into the string/offset array
     * for index i of the input column.
     */
    inline static int32_t get_offset_idx(
        const std::shared_ptr<array_info> &in_col, size_t i) {
        if constexpr (arr_typ == bodo_array_type::arr_type_enum::DICT) {
            const std::shared_ptr<array_info> &dict_indices_array =
                in_col->child_arrays[1];
            return getv<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(
                dict_indices_array, i);
        } else {
            return i;
        }
    }
};

/**
 * @brief Determines the total number of characters that are needed to fully
 * contain the output string.
 */
template <bodo_array_type::arr_type_enum arr_typ>
size_t get_total_chars_needed_for_output(
    std::shared_ptr<array_info> sorted_agg_col, size_t separator_length) {
    size_t num_null_elements = 0;
    size_t num_char_from_existing_strings = 0;
    std::shared_ptr<array_info> string_array =
        listagg_seq_utils<arr_typ>::get_string_arr(sorted_agg_col);
    assert(string_array->arr_type == bodo_array_type::STRING);
    offset_t *offsets_ptr =
        (offset_t *)string_array->data2<bodo_array_type::STRING>();

    const uint8_t *null_bitmap = (uint8_t *)sorted_agg_col->null_bitmask();

    for (uint64_t i = 0; i < sorted_agg_col->length; i++) {
        if (!GetBit(null_bitmap, i)) {
            num_null_elements++;
        } else {
            int32_t offset_idx =
                listagg_seq_utils<arr_typ>::get_offset_idx(sorted_agg_col, i);
            num_char_from_existing_strings +=
                offsets_ptr[offset_idx + 1] - offsets_ptr[offset_idx];
        }
    }

    size_t num_chars_due_to_sep;
    if (sorted_agg_col->length > num_null_elements) {
        num_chars_due_to_sep = (separator_length * (sorted_agg_col->length -
                                                    num_null_elements - 1));
    } else {
        num_chars_due_to_sep = 0;
    }

    //+1 for null terminator
    size_t output_size =
        num_chars_due_to_sep + num_char_from_existing_strings + 1;

    return output_size;
}

/**
 * @brief Copies the elements from sorted_agg_col into the output_string,
 * separated by the separator argument.
 * @param sorted_agg_col The sorted column to copy into the output string
 * @param[in, out] output_string The output string to copy the elements
 * from sorted_agg_col and the separators into. This function assumes that
 * the size of the allocated output_string is sufficient to contain the output.
 * @param[in] separator The separator to use between elements of the
 * sorted_agg_col
 * @tparam arr_typ The type of the sorted_agg_col
 * @tparam append_separator If true, append the separator in between
 * each of the output elements
 */
template <bodo_array_type::arr_type_enum arr_typ, bool append_separator>
void copy_array_to_output_string(
    const std::shared_ptr<array_info> &sorted_agg_col, char *output_string,
    const std::string &separator) {
    std::shared_ptr<array_info> string_array =
        listagg_seq_utils<arr_typ>::get_string_arr(sorted_agg_col);
    assert(string_array->arr_type == bodo_array_type::STRING);
    offset_t *offsets_ptr =
        (offset_t *)string_array->data2<bodo_array_type::STRING>();
    char *data_ptr = (char *)string_array->data1<bodo_array_type::STRING>();
    size_t offset_into_output_string = 0;
    bool seen_non_null = false;

    const uint8_t *null_bitmap = (uint8_t *)sorted_agg_col->null_bitmask();

    size_t separator_length = separator.size();
    for (uint64_t i = 0; i < sorted_agg_col->length; i++) {
        if (!GetBit(null_bitmap, i)) {
            continue;
        }
        if constexpr (append_separator) {
            if (seen_non_null) {
                memcpy(output_string + offset_into_output_string,
                       separator.c_str(), separator_length);
                offset_into_output_string += separator_length;
            }
        }
        seen_non_null = true;

        int32_t real_idx =
            listagg_seq_utils<arr_typ>::get_offset_idx(sorted_agg_col, i);
        offset_t str_len = offsets_ptr[real_idx + 1] - offsets_ptr[real_idx];
        memcpy(output_string + offset_into_output_string,
               data_ptr + offsets_ptr[real_idx], str_len);
        offset_into_output_string += str_len;
    }
}

template <bodo_array_type::arr_type_enum arr_typ>
char *listagg_seq(std::shared_ptr<table_info> in_table,
                  const std::string &separator, int num_order_cols,
                  bool *ascending, bool *na_position,
                  int64_t *output_string_size_ptr) {
    // If the input table is empty, just return empty string
    if (in_table->nrows() == 0) {
        char *retval = new char[1];
        strcpy(retval, "");
        return retval;
    }

    // Step 1: Sort the input table
    assert((uint64_t)num_order_cols == in_table->ncols());

    // Convert bool* to int64_t*
    std::vector<int64_t> ascending_real(num_order_cols);
    std::vector<int64_t> na_position_real(num_order_cols);

    // Initialize the group ordering
    // Ignore the value for the separator argument and data argument
    // present at the beginning.
    // (they are dummy values, required to get the rest of this to work)
    for (int64_t i = 0; i < num_order_cols; i++) {
        ascending_real[i] = ascending[i];
        na_position_real[i] = na_position[i];
    }

    // new initializes to 0
    std::vector<int64_t> dead_keys(num_order_cols);
    for (int64_t i = 0; i < num_order_cols; i++) {
        // Mark all the sorting keys as dead.
        dead_keys[i] = 1;
    }

    // Sort the table
    std::shared_ptr<table_info> sorted_table =
        sort_values_table_local(in_table, num_order_cols, ascending_real.data(),
                                na_position_real.data(), dead_keys.data(),
                                // TODO: set this correctly
                                false /* This is just used for tracing */);

    std::shared_ptr<array_info> sorted_agg_col = sorted_table->columns[0];

    // Step 2: Allocate the output string

    size_t separator_length = separator.size();
    size_t output_string_length = get_total_chars_needed_for_output<arr_typ>(
        sorted_agg_col, separator_length);

    *output_string_size_ptr = output_string_length - 1;

    // must be char*, since we're returning this to python
    // output_string_length includes the null terminator
    char *output_string = new char[output_string_length];

    output_string[output_string_length - 1] = '\0';  // null terminator

    // Step 3: copy the data from the input column into the output string

    if (separator_length == 0) {
        copy_array_to_output_string<arr_typ, false>(sorted_agg_col,
                                                    output_string, separator);
    } else {
        copy_array_to_output_string<arr_typ, true>(sorted_agg_col,
                                                   output_string, separator);
    }

    // Step 4: Return the result
    return output_string;
}

/**
 * Wrapper for listagg_seq that is callable from python.
 * See listagg_seq for argument/output descriptions.
 */
char *listagg_seq_py(table_info *raw_in_table, char *separator,
                     int num_order_cols, bool *ascending, bool *na_position,
                     int64_t separator_len, int64_t *output_string_size) {
    std::shared_ptr<table_info> in_table =
        std::shared_ptr<table_info>(raw_in_table);

    std::string separator_as_string(separator, separator_len);

    switch (raw_in_table->columns[num_order_cols]->arr_type) {
        case bodo_array_type::arr_type_enum::DICT:
            return listagg_seq<bodo_array_type::arr_type_enum::DICT>(
                in_table, separator_as_string, num_order_cols, ascending,
                na_position, output_string_size);
            break;
        case bodo_array_type::arr_type_enum::STRING:
            return listagg_seq<bodo_array_type::arr_type_enum::STRING>(
                in_table, separator_as_string, num_order_cols, ascending,
                na_position, output_string_size);
            break;
        default:
            throw std::runtime_error(
                "Internal error in _listagg.cpp: listagg_seq_py: "
                "aggregate array must always be "
                "string or dictionary type.");
    }
}

// Initialize lead_lag_seq_py function for usage with python
PyMODINIT_FUNC PyInit_listagg(void) {
    PyObject *m;
    MOD_DEF(m, "listagg", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, listagg_seq_py);

    return m;
}
