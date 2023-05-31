// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef _ARRAY_HASH_H_INCLUDED
#define _ARRAY_HASH_H_INCLUDED

#include "_array_utils.h"
#include "_bodo_common.h"
#include "_murmurhash3.h"

#define SEED_HASH_PARTITION 0xb0d01289
#define SEED_HASH_MULTIKEY 0xb0d01288
#define SEED_HASH_DROPDUPLI 0xb0d01287
#define SEED_HASH_JOIN 0xb0d01286
#define SEED_HASH_GROUPBY_SHUFFLE 0xb0d01285
#define SEED_HASH_PIVOT_SHUFFLE 0xb0d01285
#define SEED_HASH_CONTAINER 0xb0d01284

void hash_array_combine(std::unique_ptr<uint32_t[]>& out_hashes,
                        std::shared_ptr<array_info> array, size_t n_rows,
                        const uint32_t seed, bool global_dict_needed,
                        bool is_parallel);

/**
 * Function for the computation of hashes for keys
 *
 * @param key_arrs: input keys to hashThe hashes on output.
 * @param seed: the seed of the computation.
 * @param is_parallel: whether we run in parallel or not.
 * @return hash keys
 *
 */
std::unique_ptr<uint32_t[]> hash_keys(
    std::vector<std::shared_ptr<array_info>> const& key_arrs,
    const uint32_t seed, bool is_parallel, bool global_dict_needed = true);

std::unique_ptr<uint32_t[]> coherent_hash_keys(
    std::vector<std::shared_ptr<array_info>> const& key_arrs,
    std::vector<std::shared_ptr<array_info>> const& ref_key_arrs,
    const uint32_t seed, bool is_parallel);

void hash_array(std::unique_ptr<uint32_t[]>& out_hashes,
                std::shared_ptr<array_info> array, size_t n_rows,
                const uint32_t seed, bool is_parallel, bool global_dict_needed,
                bool use_murmurhash = false);

/**
 * Function for the getting table keys and returning its hashes
 *
 * @param in_table: the input table
 * @param num_keys : the number of keys
 * @param seed: the seed of the computation.
 * @param is_parallel: whether we run in parallel or not.
 * @return hash keys
 *
 */
inline std::unique_ptr<uint32_t[]> hash_keys_table(
    std::shared_ptr<table_info> in_table, size_t num_keys, uint32_t seed,
    bool is_parallel) {
    tracing::Event ev("hash_keys_table", is_parallel);
    std::vector<std::shared_ptr<array_info>> key_arrs(
        in_table->columns.begin(), in_table->columns.begin() + num_keys);
    return hash_keys(key_arrs, seed, is_parallel);
}

inline std::unique_ptr<uint32_t[]> coherent_hash_keys_table(
    std::shared_ptr<table_info> in_table, std::shared_ptr<table_info> ref_table,
    size_t num_keys, uint32_t seed, bool is_parallel) {
    tracing::Event ev("coherent_hash_keys_table", is_parallel);
    std::vector<std::shared_ptr<array_info>> key_arrs(
        in_table->columns.begin(), in_table->columns.begin() + num_keys);
    std::vector<std::shared_ptr<array_info>> ref_key_arrs(
        ref_table->columns.begin(), ref_table->columns.begin() + num_keys);
    return coherent_hash_keys(key_arrs, ref_key_arrs, seed, is_parallel);
}

/**
 * Multi column key that can be used to hash the value of multiple columns
 * in a DataFrame row in C++ hash tables/sets.
 * NOTE: This assumes the key columns are the first columns in the table.
 */
struct multi_col_key {
    uint32_t hash;
    std::shared_ptr<table_info> table;
    int64_t row;
    bool is_parallel;

    multi_col_key(uint32_t _hash, std::shared_ptr<table_info> _table,
                  int64_t _row, bool _is_parallel)
        : hash(_hash), table(_table), row(_row), is_parallel(_is_parallel) {}

    bool operator==(const multi_col_key& other) const {
        for (int64_t i = 0; i < table->num_keys; i++) {
            std::shared_ptr<array_info> c1 = table->columns[i];
            std::shared_ptr<array_info> c2 = other.table->columns[i];
            size_t size_type;
            switch (c1->arr_type) {
                case bodo_array_type::ARRAY_ITEM:
                case bodo_array_type::STRUCT: {
                    int64_t pos1_s = row;
                    int64_t pos1_e = row + 1;
                    int64_t pos2_s = other.row;
                    int64_t pos2_e = other.row + 1;
                    bool na_position_bis = true;
                    int test = ComparisonArrowColumn(
                        to_arrow(c1), pos1_s, pos1_e, to_arrow(c2), pos2_s,
                        pos2_e, na_position_bis);
                    if (test != 0) {
                        return false;
                    }
                }
                    continue;
                case bodo_array_type::DICT: {
                    // Require the dictionary to always have unique values. If
                    // the data is distributed it must also be global.
                    if (c1->has_deduped_local_dictionary &&
                        c2->has_deduped_local_dictionary &&
                        (!is_parallel || c1->has_global_dictionary) &&
                        (!other.is_parallel || c2->has_global_dictionary)) {
                        if (c1->child_arrays[0] != c2->child_arrays[0]) {
                            throw std::runtime_error(
                                "multi-key-hashing dictionary the columns are "
                                "not unified.");
                        }
                        if (c1->get_null_bit(row) !=
                            c2->get_null_bit(other.row)) {
                            return false;
                        }
                        if (!c1->get_null_bit(row)) {
                            // If both are null then continue because
                            // the dictionary contains garbage
                            continue;
                        }
                        bool match =
                            c1->child_arrays[1]->at<dict_indices_t>(row) ==
                            c2->child_arrays[1]->at<dict_indices_t>(other.row);
                        if (!match) {
                            return false;
                        }
                        continue;
                    } else {
                        throw std::runtime_error(
                            "multi-key-hashing dictionary array requires "
                            "global dictionary of unique values");
                    }
                }
                    continue;
                case bodo_array_type::NULLABLE_INT_BOOL:
                    if (c1->get_null_bit(row) != c2->get_null_bit(other.row)) {
                        return false;
                    }
                    if (!c1->get_null_bit(row)) {
                        continue;
                    }
                    if (c1->dtype == Bodo_CTypes::_BOOL) {
                        // Nullable bools are stored as 1 bit
                        if (GetBit((uint8_t*)c1->data1(), row) !=
                            GetBit((uint8_t*)c2->data1(), other.row)) {
                            return false;
                        }
                    } else {
                        size_type = numpy_item_size[c1->dtype];
                        if (memcmp(c1->data1() + size_type * row,
                                   c2->data1() + size_type * other.row,
                                   size_type) != 0) {
                            return false;
                        }
                    }
                    continue;

                case bodo_array_type::CATEGORICAL:  // Even in missing case
                                                    // (value -1) this works
                case bodo_array_type::NUMPY:
                    size_type = numpy_item_size[c1->dtype];
                    if (memcmp(c1->data1() + size_type * row,
                               c2->data1() + size_type * other.row,
                               size_type) != 0) {
                        return false;
                    }
                    continue;
                case bodo_array_type::STRING: {
                    uint8_t* c1_null_bitmask = (uint8_t*)c1->null_bitmask();
                    uint8_t* c2_null_bitmask = (uint8_t*)c2->null_bitmask();
                    if (GetBit(c1_null_bitmask, row) !=
                        GetBit(c2_null_bitmask, other.row)) {
                        return false;
                    }
                    offset_t* c1_offsets = (offset_t*)c1->data2();
                    offset_t* c2_offsets = (offset_t*)c2->data2();
                    offset_t c1_str_len = c1_offsets[row + 1] - c1_offsets[row];
                    offset_t c2_str_len =
                        c2_offsets[other.row + 1] - c2_offsets[other.row];
                    if (c1_str_len != c2_str_len) {
                        return false;
                    }
                    char* c1_str = c1->data1() + c1_offsets[row];
                    char* c2_str = c2->data1() + c2_offsets[other.row];
                    if (memcmp(c1_str, c2_str, c1_str_len) != 0) {
                        return false;
                    }
                }
                    continue;
                case bodo_array_type::LIST_STRING: {
                    uint8_t* c1_null_bitmask = (uint8_t*)c1->null_bitmask();
                    uint8_t* c2_null_bitmask = (uint8_t*)c2->null_bitmask();
                    if (GetBit(c1_null_bitmask, row) !=
                        GetBit(c2_null_bitmask, other.row)) {
                        return false;
                    }
                    uint8_t* c1_sub_null_bitmask =
                        (uint8_t*)c1->sub_null_bitmask();
                    uint8_t* c2_sub_null_bitmask =
                        (uint8_t*)c2->sub_null_bitmask();
                    offset_t* c1_index_offsets = (offset_t*)c1->data3();
                    offset_t* c2_index_offsets = (offset_t*)c2->data3();
                    offset_t* c1_data_offsets = (offset_t*)c1->data2();
                    offset_t* c2_data_offsets = (offset_t*)c2->data2();
                    // Comparing the number of strings.
                    offset_t c1_index_len =
                        c1_index_offsets[row + 1] - c1_index_offsets[row];
                    offset_t c2_index_len = c2_index_offsets[other.row + 1] -
                                            c2_index_offsets[other.row];
                    if (c1_index_len != c2_index_len) {
                        return false;
                    }
                    // comparing the length of the strings.
                    for (offset_t u = 0; u < c1_index_len; u++) {
                        offset_t size_data1 =
                            c1_data_offsets[c1_index_offsets[row] + u + 1] -
                            c1_data_offsets[c1_index_offsets[row] + u];
                        offset_t size_data2 =
                            c2_data_offsets[c2_index_offsets[other.row] + u +
                                            1] -
                            c2_data_offsets[c2_index_offsets[other.row] + u];
                        if (size_data1 != size_data2) {
                            return false;
                        }
                        bool str_bit1 = GetBit(c1_sub_null_bitmask,
                                               c1_index_offsets[row] + u);
                        bool str_bit2 = GetBit(c2_sub_null_bitmask,
                                               c2_index_offsets[other.row] + u);
                        if (str_bit1 != str_bit2) {
                            return false;
                        }
                    }
                    // Now comparing the strings. Their length is the same since
                    // we pass above check
                    offset_t common_len =
                        c1_data_offsets[c1_index_offsets[row + 1]] -
                        c1_data_offsets[c1_index_offsets[row]];
                    char* c1_strB =
                        c1->data1() + c1_data_offsets[c1_index_offsets[row]];
                    char* c2_strB =
                        c2->data1() +
                        c2_data_offsets[c2_index_offsets[other.row]];
                    if (memcmp(c1_strB, c2_strB, common_len) != 0)
                        return false;
                    continue;
                }
                default: {
                    throw std::runtime_error(
                        "multi_col_key_hash : Unsupported type");
                }
                    continue;
            }
        }
        return true;
    }
};

struct multi_col_key_hash {
    std::size_t operator()(const multi_col_key& k) const { return k.hash; }
};

/**
 * Unifies dictionaries of DICT arrays arr1 and arr2
 * If the arrays have a reference to the same dictionary then
 * this function does nothing and returns.
 * Replaces old dictionary with the new one.
 * Updates the indices to conform to the new dictionary.
 */
void unify_dictionaries(std::shared_ptr<array_info> arr1,
                        std::shared_ptr<array_info> arr2, bool arr1_is_parallel,
                        bool arr2_is_parallel);

// For hashing function involving dictionaries of dictionary-encoded arrays
struct HashDict {
    uint32_t operator()(const size_t iRow) const {
        if (iRow < global_array_rows) {
            return global_array_hashes[iRow];
        } else {
            return local_array_hashes[iRow - global_array_rows];
        }
    }
    const size_t global_array_rows;
    const std::unique_ptr<uint32_t[]>& global_array_hashes;
    const std::unique_ptr<uint32_t[]>& local_array_hashes;
};

// For key comparison involving dictionaries of dictionary-encoded arrays
struct KeyEqualDict {
    bool operator()(const size_t iRowA, const size_t iRowB) const {
        size_t jRowA, jRowB;
        std::shared_ptr<array_info> dict_A, dict_B;
        if (iRowA < global_array_rows) {
            dict_A = global_dictionary;
            jRowA = iRowA;
        } else {
            dict_A = local_dictionary;
            jRowA = iRowA - global_array_rows;
        }
        if (iRowB < global_array_rows) {
            dict_B = global_dictionary;
            jRowB = iRowB;
        } else {
            dict_B = local_dictionary;
            jRowB = iRowB - global_array_rows;
        }
        // TODO inline?
        return TestEqualColumn(dict_A, jRowA, dict_B, jRowB, true);
    }
    // global_dict_len, global_dictionary, local_dictionary
    size_t global_array_rows;
    std::shared_ptr<array_info> global_dictionary;
    std::shared_ptr<array_info> local_dictionary;
};

/**
 * @brief Unifies dictionaries of 1 or more DICT arrays. If all
 * arrays have a reference to the same dictionary then this function
 * does nothing and returns.
 * Replaces old dictionaries with the new one.
 * Updates the indices to conform to the new dictionary.
 *
 * @param arrs A vector of DICT arrays that need to be unified. These must
 * have already removed any duplicates and be consistent across ranks.
 * @param is_parallels If arrs[i] is parallel then is_parallels[i] is true.
 * This is used for checking the global condition for each array.
 */
void unify_several_dictionaries(std::vector<std::shared_ptr<array_info>>& arrs,
                                std::vector<bool>& is_parallels);

// Hash comparison for comparing multiple arrays where each array is inserted
// one at a time.
struct HashMultiArray {
    uint32_t operator()(const std::pair<size_t, size_t> hash_info) const {
        const size_t iRow = hash_info.first;
        const size_t iTable = hash_info.second;
        return hashes[iTable][iRow];
    }
    std::vector<std::shared_ptr<uint32_t[]>>& hashes;
};

// Equality comparison for comparing multiple arrays where each array is
// inserted one at a time.
struct MultiArrayInfoEqual {
    bool operator()(const std::pair<size_t, size_t> hash_info1,
                    const std::pair<size_t, size_t> hash_info2) const {
        const size_t iRowA = hash_info1.first;
        const size_t iTableA = hash_info1.second;
        const size_t iRowB = hash_info2.first;
        const size_t iTableB = hash_info2.second;
        std::shared_ptr<array_info> dict_A = arrs[iTableA];
        std::shared_ptr<array_info> dict_B = arrs[iTableB];
        // TODO inline?
        return TestEqualColumn(dict_A, iRowA, dict_B, iRowB, true);
    }
    std::vector<std::shared_ptr<array_info>>& arrs;
};

/**
 * @brief functor for comparing two elements (one from each of the two arrays).
 */
class ElementComparator {
   public:
    // Store data pointers to avoid extra struct access in performance critical
    // code
    ElementComparator(const std::shared_ptr<array_info>& arr1_,
                      const std::shared_ptr<array_info>& arr2_) {
        // Store the index data for dict encoded arrays since index comparison
        // is enough in case of unified dictionaries. Only use when the
        // dictionaries are the same (and are deduped since otherwise the index
        // comparisons are not accurate).
        if (arr1_->arr_type == bodo_array_type::DICT &&
            arr2_->arr_type == bodo_array_type::DICT) {
            if (arr1_->child_arrays[0] != arr2_->child_arrays[0]) {
                throw std::runtime_error(
                    "ElementComparator: don't know if arrays have "
                    "unified dictionary.");
            }
            if (!(arr1_->has_deduped_local_dictionary)) {
                throw std::runtime_error(
                    "ElementComparator: Dictionary is not deduplicated.");
            }
            this->arr1 = arr1_->child_arrays[1];
            this->arr2 = arr2_->child_arrays[1];
        } else {
            this->arr1 = arr1_;
            this->arr2 = arr2_;
        }
        this->data_ptr_1 = this->arr1->data1();
        this->data_ptr_2 = this->arr2->data1();
        this->null_bitmask_1 = (uint8_t*)this->arr1->null_bitmask();
        this->null_bitmask_2 = (uint8_t*)this->arr2->null_bitmask();
    }

    // Numpy arrays with nullable dtypes (float, datetime/timedelta)
    // NAs equal case
    template <bodo_array_type::arr_type_enum ArrType,
              Bodo_CTypes::CTypeEnum DType, bool is_na_equal>
        requires(ArrType == bodo_array_type::NUMPY &&
                 NullSentinelDtype<DType> && is_na_equal)
    constexpr bool operator()(const int64_t iRowA, const int64_t iRowB) const {
        using T = typename dtype_to_type<DType>::type;
        T* data1 = (T*)this->data_ptr_1;
        T* data2 = (T*)this->data_ptr_2;
        T val1 = data1[iRowA];
        T val2 = data2[iRowB];
        bool isna1 = isnan_alltype<T, DType>(val1);
        bool isna2 = isnan_alltype<T, DType>(val2);
        return (isna1 && isna2) || (!isna1 && !isna2 && val1 == val2);
    }

    // NAs not equal case
    template <bodo_array_type::arr_type_enum ArrType,
              Bodo_CTypes::CTypeEnum DType, bool is_na_equal>
        requires(ArrType == bodo_array_type::NUMPY &&
                 NullSentinelDtype<DType> && !is_na_equal)
    constexpr bool operator()(const int64_t iRowA, const int64_t iRowB) const {
        using T = typename dtype_to_type<DType>::type;
        T* data1 = (T*)this->data_ptr_1;
        T* data2 = (T*)this->data_ptr_2;
        T val1 = data1[iRowA];
        T val2 = data2[iRowB];
        bool isna1 = isnan_alltype<T, DType>(val1);
        bool isna2 = isnan_alltype<T, DType>(val2);
        return !isna1 && !isna2 && (val1 == val2);
    }

    // Numpy arrays with non-nullable dtypes (integer, boolean)
    // 'is_na_equal' doesn't matter.
    template <bodo_array_type::arr_type_enum ArrType,
              Bodo_CTypes::CTypeEnum DType, bool is_na_equal = true>
        requires(ArrType == bodo_array_type::NUMPY && !NullSentinelDtype<DType>)
    constexpr bool operator()(const int64_t iRowA, const int64_t iRowB) const {
        using T = typename dtype_to_type<DType>::type;
        T* data1 = (T*)this->data_ptr_1;
        T* data2 = (T*)this->data_ptr_2;
        return data1[iRowA] == data2[iRowB];
    }

    // Nullable non-boolean arrays
    // NAs equal case:
    template <bodo_array_type::arr_type_enum ArrType,
              Bodo_CTypes::CTypeEnum DType, bool is_na_equal>
        requires(ArrType == bodo_array_type::NULLABLE_INT_BOOL &&
                 DType != Bodo_CTypes::CTypeEnum::_BOOL && is_na_equal)
    constexpr bool operator()(const int64_t iRowA, const int64_t iRowB) const {
        using T = typename dtype_to_type<DType>::type;
        T* data1 = (T*)this->data_ptr_1;
        T* data2 = (T*)this->data_ptr_2;
        T val1 = data1[iRowA];
        T val2 = data2[iRowB];
        bool isna1 = !GetBit(this->null_bitmask_1, iRowA);
        bool isna2 = !GetBit(this->null_bitmask_2, iRowB);
        return (isna1 && isna2) || (!isna1 && !isna2 && val1 == val2);
    }

    // NAs not equal case:
    template <bodo_array_type::arr_type_enum ArrType,
              Bodo_CTypes::CTypeEnum DType, bool is_na_equal>
        requires(ArrType == bodo_array_type::NULLABLE_INT_BOOL &&
                 DType != Bodo_CTypes::_BOOL && !is_na_equal)
    constexpr bool operator()(const int64_t iRowA, const int64_t iRowB) const {
        using T = typename dtype_to_type<DType>::type;
        T* data1 = (T*)this->data_ptr_1;
        T* data2 = (T*)this->data_ptr_2;
        T val1 = data1[iRowA];
        T val2 = data2[iRowB];
        bool isna1 = !GetBit(this->null_bitmask_1, iRowA);
        bool isna2 = !GetBit(this->null_bitmask_2, iRowB);
        return !isna1 && !isna2 && (val1 == val2);
    }

    // Nullable boolean arrays
    // NA equal case:
    template <bodo_array_type::arr_type_enum ArrType,
              Bodo_CTypes::CTypeEnum DType, bool is_na_equal>
        requires(ArrType == bodo_array_type::NULLABLE_INT_BOOL &&
                 DType == Bodo_CTypes::_BOOL && is_na_equal)
    constexpr bool operator()(const int64_t iRowA, const int64_t iRowB) const {
        bool val1 = GetBit((uint8_t*)this->data_ptr_1, iRowA);
        bool val2 = GetBit((uint8_t*)this->data_ptr_2, iRowB);
        bool isna1 = !GetBit(this->null_bitmask_1, iRowA);
        bool isna2 = !GetBit(this->null_bitmask_2, iRowB);
        return (isna1 && isna2) || (!isna1 && !isna2 && val1 == val2);
    }

    // NAs not equal case:
    template <bodo_array_type::arr_type_enum ArrType,
              Bodo_CTypes::CTypeEnum DType, bool is_na_equal>
        requires(ArrType == bodo_array_type::NULLABLE_INT_BOOL &&
                 DType == Bodo_CTypes::CTypeEnum::_BOOL && !is_na_equal)
    constexpr bool operator()(const int64_t iRowA, const int64_t iRowB) const {
        bool val1 = GetBit((uint8_t*)this->data_ptr_1, iRowA);
        bool val2 = GetBit((uint8_t*)this->data_ptr_2, iRowB);
        bool isna1 = !GetBit(this->null_bitmask_1, iRowA);
        bool isna2 = !GetBit(this->null_bitmask_2, iRowB);
        return !isna1 && !isna2 && (val1 == val2);
    }

    // Dict-encoded string arrays
    // NAs equal case:
    template <bodo_array_type::arr_type_enum ArrType,
              Bodo_CTypes::CTypeEnum DType, bool is_na_equal>
        requires(ArrType == bodo_array_type::DICT && is_na_equal)
    constexpr bool operator()(const int64_t iRowA, const int64_t iRowB) const {
        using T = DICT_INDEX_C_TYPE;
        T* data1 = (T*)this->data_ptr_1;
        T* data2 = (T*)this->data_ptr_2;
        T val1 = data1[iRowA];
        T val2 = data2[iRowB];
        bool isna1 = !GetBit(this->null_bitmask_1, iRowA);
        bool isna2 = !GetBit(this->null_bitmask_2, iRowB);
        return (isna1 && isna2) || (!isna1 && !isna2 && val1 == val2);
    }

    // NAs not equal case:
    template <bodo_array_type::arr_type_enum ArrType,
              Bodo_CTypes::CTypeEnum DType, bool is_na_equal>
        requires(ArrType == bodo_array_type::DICT && !is_na_equal)
    constexpr bool operator()(const int64_t iRowA, const int64_t iRowB) const {
        using T = DICT_INDEX_C_TYPE;
        T* data1 = (T*)this->data_ptr_1;
        T* data2 = (T*)this->data_ptr_2;
        T val1 = data1[iRowA];
        T val2 = data2[iRowB];
        bool isna1 = !GetBit(this->null_bitmask_1, iRowA);
        bool isna2 = !GetBit(this->null_bitmask_2, iRowB);
        return !isna1 && !isna2 && (val1 == val2);
    }

    // generic comparator, fall back to runtime type checks with TestEqualColumn
    template <bodo_array_type::arr_type_enum ArrType,
              Bodo_CTypes::CTypeEnum DType, bool is_na_equal>
        requires(ArrType != bodo_array_type::NUMPY &&
                 ArrType != bodo_array_type::NULLABLE_INT_BOOL &&
                 ArrType != bodo_array_type::DICT)
    constexpr bool operator()(const int64_t iRowA, const int64_t iRowB) const {
        return TestEqualColumn(this->arr1, iRowA, this->arr2, iRowB,
                               is_na_equal);
    }

   private:
    std::shared_ptr<array_info> arr1;
    std::shared_ptr<array_info> arr2;
    const char* data_ptr_1;
    const char* data_ptr_2;
    const uint8_t* null_bitmask_1;
    const uint8_t* null_bitmask_2;
};

/**
 * @brief Key equality comparator class (for hash map use) that is specialized
 * for one key to make common cases faster (e.g. Int32, Int64, DATE, DICT).
 *
 * @tparam ArrType array's type (e.g. bodo_array_type::NULLABLE_INT_BOOL)
 * @tparam DType array's dtype (e.g. Bodo_CTypes::INT32)
 * @tparam is_na_equal Whether NAs are considered equal
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool is_na_equal>
class KeysEqualComparatorOneKey {
   public:
    KeysEqualComparatorOneKey(const std::shared_ptr<array_info> arr)
        : cmp(arr, arr) {}

    constexpr bool operator()(const int64_t iRowA,
                              const int64_t iRowB) const noexcept {
        return (
            cmp.template operator()<ArrType, DType, is_na_equal>(iRowA, iRowB));
    }

   private:
    const ElementComparator cmp;
};

/**
 * @brief Key equality comparator class (for hash map use) that is specialized
 * for two keys to make common cases faster (e.g. Int32/DATE, Int64/Int64).
 *
 * @tparam ArrType1 first array's type (e.g. bodo_array_type::NULLABLE_INT_BOOL)
 * @tparam DType1 first array's dtype (e.g. Bodo_CTypes::INT32)
 * @tparam ArrType2 second array's type (e.g. bodo_array_type::NUMPY)
 * @tparam DType2 first array's dtype (e.g. Bodo_CTypes::DATE)
 * @tparam is_na_equal Whether NAs are considered equal
 */
template <bodo_array_type::arr_type_enum ArrType1,
          Bodo_CTypes::CTypeEnum DType1,
          bodo_array_type::arr_type_enum ArrType2,
          Bodo_CTypes::CTypeEnum DType2, bool is_na_equal>
class KeysEqualComparatorTwoKeys {
   public:
    KeysEqualComparatorTwoKeys(const std::shared_ptr<array_info> arr1,
                               const std::shared_ptr<array_info> arr2)
        : cmp1(arr1, arr1), cmp2(arr2, arr2) {}

    constexpr bool operator()(const int64_t iRowA,
                              const int64_t iRowB) const noexcept {
        return (cmp1.template operator()<ArrType1, DType1, is_na_equal>(
                   iRowA, iRowB)) &&
               (cmp2.template operator()<ArrType2, DType2, is_na_equal>(iRowA,
                                                                        iRowB));
    }

   private:
    const ElementComparator cmp1;
    const ElementComparator cmp2;
};

/**
 * @brief Join Key equality comparator class (for hash map use) that is
 * specialized for one key to make common cases faster (e.g. Int32, Int64, DATE,
 * DICT).
 *
 * @tparam ArrType : array's type (e.g. bodo_array_type::NULLABLE_INT_BOOL)
 * @tparam DType : array's dtype (e.g. Bodo_CTypes::INT32)
 * @tparam is_na_equal : Whether NAs are considered equal
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool is_na_equal>
class JoinKeysEqualComparatorOneKey {
   public:
    JoinKeysEqualComparatorOneKey(const std::shared_ptr<array_info>& arr1,
                                  const std::shared_ptr<array_info>& arr2)
        : cmp_arr1(arr1, arr1),
          cmp_arr2(arr2, arr2),
          cmp_arr1_arr2(arr1, arr2),
          cmp_arr2_arr1(arr2, arr1),
          arr1_len(arr1->length) {}

    constexpr bool operator()(int64_t iRowA, int64_t iRowB) const noexcept {
        // A row can refer to either arr1 (build table) or arr2 (probe table).
        // If iRow < arr1_len then it is in the build table
        //    at index iRow.
        // If iRow >= arr1_len then it is in the probe table
        //    at index (iRow - arr1_len).
        if (iRowA < arr1_len) {
            if (iRowB < arr1_len) {
                // Determine if NA columns should match. They should always
                // match when populating the hash map with the build table.
                // When comparing the build and probe tables this depends on
                // is_na_equal.
                return cmp_arr1.template operator()<ArrType, DType, true>(
                    iRowA, iRowB);
            } else {
                iRowB = iRowB - arr1_len;
                return cmp_arr1_arr2.template
                operator()<ArrType, DType, is_na_equal>(iRowA, iRowB);
            }
        } else {
            if (iRowB < arr1_len) {
                iRowA = iRowA - arr1_len;
                return cmp_arr2_arr1.template
                operator()<ArrType, DType, is_na_equal>(iRowA, iRowB);
            } else {
                iRowA = iRowA - arr1_len;
                iRowB = iRowB - arr1_len;
                // Same logic here regarding is_na_equal
                return cmp_arr2.template operator()<ArrType, DType, true>(
                    iRowA, iRowB);
            }
        }
    }

   private:
    const ElementComparator cmp_arr1;
    const ElementComparator cmp_arr2;
    const ElementComparator cmp_arr1_arr2;
    const ElementComparator cmp_arr2_arr1;
    const int64_t arr1_len;
};

template <bodo_array_type::arr_type_enum ArrType1,
          Bodo_CTypes::CTypeEnum DType1,
          bodo_array_type::arr_type_enum ArrType2,
          Bodo_CTypes::CTypeEnum DType2, bool is_na_equal>
class JoinKeysEqualComparatorTwoKeys {
   public:
    JoinKeysEqualComparatorTwoKeys(const std::shared_ptr<array_info>& t1_k1,
                                   const std::shared_ptr<array_info>& t1_k2,
                                   const std::shared_ptr<array_info>& t2_k1,
                                   const std::shared_ptr<array_info>& t2_k2)
        : cmp_k1_t1(t1_k1, t1_k1),
          cmp_k1_t2(t2_k1, t2_k1),
          cmp_k1_t1_t2(t1_k1, t2_k1),
          cmp_k1_t2_t1(t2_k1, t1_k1),
          cmp_k2_t1(t1_k2, t1_k2),
          cmp_k2_t2(t2_k2, t2_k2),
          cmp_k2_t1_t2(t1_k2, t2_k2),
          cmp_k2_t2_t1(t2_k2, t1_k2),
          t1_len(t1_k1->length) {}

    constexpr bool operator()(int64_t iRowA, int64_t iRowB) const noexcept {
        // A row can refer to either t1 (build table) or t2 (probe table).
        // If iRow < t1_len then it is in the build table
        //    at index iRow.
        // If iRow >= t1_len then it is in the probe table
        //    at index (iRow - t1_len).
        if (iRowA < t1_len) {
            if (iRowB < t1_len) {
                // Determine if NA columns should match. They should always
                // match when populating the hash map with the build table.
                // When comparing the build and probe tables this depends on
                // is_na_equal.
                return (cmp_k1_t1.template operator()<ArrType1, DType1, true>(
                           iRowA, iRowB)) &&
                       (cmp_k2_t1.template operator()<ArrType2, DType2, true>(
                           iRowA, iRowB));
            } else {
                iRowB = iRowB - t1_len;
                return (cmp_k1_t1_t2
                            .template operator()<ArrType1, DType1, is_na_equal>(
                                iRowA, iRowB)) &&
                       (cmp_k2_t1_t2
                            .template operator()<ArrType2, DType2, is_na_equal>(
                                iRowA, iRowB));
            }
        } else {
            if (iRowB < t1_len) {
                iRowA = iRowA - t1_len;
                return (cmp_k1_t2_t1
                            .template operator()<ArrType1, DType1, is_na_equal>(
                                iRowA, iRowB)) &&
                       (cmp_k2_t2_t1
                            .template operator()<ArrType2, DType2, is_na_equal>(
                                iRowA, iRowB));
            } else {
                iRowA = iRowA - t1_len;
                iRowB = iRowB - t1_len;
                // Same logic here regarding is_na_equal
                return (cmp_k1_t2.template operator()<ArrType1, DType1, true>(
                           iRowA, iRowB)) &&
                       (cmp_k2_t2.template operator()<ArrType2, DType2, true>(
                           iRowA, iRowB));
            }
        }
    }

   private:
    const ElementComparator cmp_k1_t1;
    const ElementComparator cmp_k1_t2;
    const ElementComparator cmp_k1_t1_t2;
    const ElementComparator cmp_k1_t2_t1;
    const ElementComparator cmp_k2_t1;
    const ElementComparator cmp_k2_t2;
    const ElementComparator cmp_k2_t1_t2;
    const ElementComparator cmp_k2_t2_t1;
    const int64_t t1_len;
};

/**
 * @brief Invokes `operator()` template of input functor with type instantiation
 * based on the specified arr_type and dtype. Similar to:
 * https://github.com/rapidsai/cudf/blob/c4a1389bca6f2fd521bd5e768eda7407aa3e66b5/cpp/include/cudf/utilities/type_dispatcher.hpp#L440
 *
 * @tparam Functor functor with operator() to call
 * @tparam Ts types of input arguments to operator()
 * @param arr_type array type for type instantiation
 * @param dtype dtype for type instantiation
 * @param is_na_equal Whether NAs are considered equal.
 * @param f input functor
 * @param args arguments to pass to operator()
 * @return constexpr decltype(auto) return type inferred from operator()
 */
template <typename Functor, typename... Ts>
inline constexpr decltype(auto) type_dispatcher(
    bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
    const bool is_na_equal, Functor& f, Ts&&... args) {
#ifndef DISPATCH_CASE
#define DISPATCH_CASE(ARRAY_TYPE, DTYPE)                            \
    case DTYPE: {                                                   \
        if (is_na_equal) {                                          \
            return f.template operator()<ARRAY_TYPE, DTYPE, true>(  \
                std::forward<Ts>(args)...);                         \
        } else {                                                    \
            return f.template operator()<ARRAY_TYPE, DTYPE, false>( \
                std::forward<Ts>(args)...);                         \
        }                                                           \
    }
#endif

    switch (arr_type) {
        case bodo_array_type::NUMPY:
            switch (dtype) {
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::_BOOL);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::INT8);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::UINT8);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::INT16);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::UINT16);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::INT32);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::UINT32);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::INT64);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::UINT64);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT32);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::INT128);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::DECIMAL);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::DATE);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::TIME);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME);
                DISPATCH_CASE(bodo_array_type::NUMPY, Bodo_CTypes::TIMEDELTA);
                default:
                    throw std::runtime_error("invalid dtype for Numpy arrays " +
                                             std::to_string(dtype));
            }
        case bodo_array_type::NULLABLE_INT_BOOL:
            switch (dtype) {
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::_BOOL);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT8);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::UINT8);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT16);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::UINT16);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT32);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::UINT32);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT64);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::UINT64);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::FLOAT32);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::FLOAT64);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT128);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::DECIMAL);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::DATE);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::TIME);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::DATETIME);
                DISPATCH_CASE(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::TIMEDELTA);
                default:
                    throw std::runtime_error(
                        "invalid dtype for nullable arrays " +
                        std::to_string(dtype));
            }
        case bodo_array_type::STRING:
            switch (dtype) {
                DISPATCH_CASE(bodo_array_type::STRING, Bodo_CTypes::STRING);
                DISPATCH_CASE(bodo_array_type::STRING, Bodo_CTypes::BINARY);
                default:
                    throw std::runtime_error(
                        "invalid dtype for string arrays " +
                        std::to_string(dtype));
            }
        case bodo_array_type::DICT:
            switch (dtype) {
                DISPATCH_CASE(bodo_array_type::DICT, Bodo_CTypes::STRING);
                default:
                    throw std::runtime_error("invalid dtype for dict arrays " +
                                             std::to_string(dtype));
            }
        case bodo_array_type::CATEGORICAL:
            switch (dtype) {
                // categorical index numbers are integers
                DISPATCH_CASE(bodo_array_type::CATEGORICAL, Bodo_CTypes::INT8);
                DISPATCH_CASE(bodo_array_type::CATEGORICAL, Bodo_CTypes::INT16);
                DISPATCH_CASE(bodo_array_type::CATEGORICAL, Bodo_CTypes::INT32);
                DISPATCH_CASE(bodo_array_type::CATEGORICAL, Bodo_CTypes::INT64);
                default:
                    throw std::runtime_error(
                        "invalid dtype for categorical arrays " +
                        std::to_string(dtype));
            }
        case bodo_array_type::LIST_STRING:
            switch (dtype) {
                DISPATCH_CASE(bodo_array_type::LIST_STRING,
                              Bodo_CTypes::LIST_STRING);
                default:
                    throw std::runtime_error(
                        "invalid dtype for list string arrays " +
                        std::to_string(dtype));
            }
        case bodo_array_type::ARRAY_ITEM:
            switch (dtype) {
                DISPATCH_CASE(bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST);
                default:
                    throw std::runtime_error(
                        "invalid dtype for array(item) arrays " +
                        std::to_string(dtype));
            }
        case bodo_array_type::STRUCT:
            switch (dtype) {
                DISPATCH_CASE(bodo_array_type::STRUCT, Bodo_CTypes::STRUCT);
                default:
                    throw std::runtime_error(
                        "invalid dtype for struct arrays " +
                        std::to_string(dtype));
            }
        default:
            throw std::runtime_error("invalid array type " +
                                     GetArrType_as_string(arr_type));
    }
}
#undef DISPATCH_CASE

/**
 * @brief General key equality comparator class for hash map use.
 * Checks data types in runtime for each key column and calls ElementComparator.
 * Specialized classes such as KeysEqualComparatorTwoKeys should be used instead
 * whenever possible to avoid the runtime type check costs. Similar to:
 * https://github.com/rapidsai/cudf/blob/c4a1389bca6f2fd521bd5e768eda7407aa3e66b5/cpp/include/cudf/table/experimental/row_operators.cuh#L1140
 *
 */
class KeysEqualComparator {
   public:
    using cmp_func_t =
        std::function<bool(const int64_t, const int64_t, const int64_t)>;

    KeysEqualComparator(const int64_t n_keys,
                        const std::shared_ptr<table_info> table,
                        const bool is_na_equal)
        : n_keys{n_keys}, table{std::move(table)}, is_na_equal(is_na_equal) {
        // Create all the ElementComparator instances up front.
        this->cmps.reserve(n_keys);
        for (int64_t key_i = 0; key_i < n_keys; key_i++) {
            this->cmps.emplace_back(table->columns[key_i],
                                    table->columns[key_i]);
        }

        this->equal_elements = [=, this](const int64_t key_i,
                                         const int64_t iRowA,
                                         const int64_t iRowB) {
            const std::shared_ptr<array_info>& arr =
                this->table->columns[key_i];
            return type_dispatcher(arr->arr_type, arr->dtype, this->is_na_equal,
                                   this->cmps[key_i], iRowA, iRowB);
        };
    }

    bool operator()(const int64_t iRowA, const int64_t iRowB) const noexcept {
        for (int64_t key_i = 0; key_i < n_keys; key_i++) {
            bool is_equal = this->equal_elements(key_i, iRowA, iRowB);
            if (!is_equal) {
                return false;
            }
        }
        return true;
    }

   private:
    const int64_t n_keys;
    const std::shared_ptr<table_info> table;
    const bool is_na_equal;
    cmp_func_t equal_elements;
    std::vector<ElementComparator> cmps;
};

// ----- Simple C++ Cache for LIKE kernel dict-encoding case --------

// Hashmap type for the like kernel cache. The key is `uint64_t`, where
// the two `uint32_t` indices are bitwise concatenated into a `uint64_t`.
// The value is `int8_t`, but we will only use 3 value: 0 (false), 1 (true)
// and -1 (cache miss).
// We use regular std::hash. Using XXH3 had slightly worse performance during
// our testing.
typedef UNORD_MAP_CONTAINER<uint64_t, int8_t, std::hash<uint64_t>,
                            std::equal_to<>>
    like_kernel_cache_t;

/**
 * @brief Allocate the cache for the like kernel dict-encoding case.
 * The cache will map pair of indices (2 uint32_t bitwise concatenated into an
 * uint64_t) to a int8_t.
 *
 * @param reserve_size Size to reserve the hashmap to.
 * @return like_kernel_cache_t* Pointer to the hashmap.
 */
like_kernel_cache_t* alloc_like_kernel_cache(uint64_t reserve_size) noexcept;

/**
 * @brief Add computation output to the hashmap.
 *
 * @param cache Pointer to the hashmap
 * @param idx1 Index in the first dictionary
 * @param idx2 Index in the second dictionary
 * @param val Boolean output of the computation to cache.
 */
void add_to_like_kernel_cache(like_kernel_cache_t* cache, uint32_t idx1,
                              uint32_t idx2, bool val) noexcept;

/**
 * @brief Check if a value is in the cache.
 *
 *
 * @param cache Pointer to the hashmap
 * @param idx1 Index in the first dictionary
 * @param idx2 Index in the second dictionary
 * @return int8_t -1 if value doesn't exist in the cache, 0 if it does exist and
 * it's false and 1 if it's true.
 */
int8_t check_like_kernel_cache(like_kernel_cache_t* cache, uint32_t idx1,
                               uint32_t idx2) noexcept;

/**
 * @brief Delete the cache and free the memory.
 *
 * @param cache Pointer to the hashmap
 */
void dealloc_like_kernel_cache(like_kernel_cache_t* cache) noexcept;

#endif  // _ARRAY_HASH_H_INCLUDED
