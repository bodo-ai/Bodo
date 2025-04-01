#include <arrow/util/bit_util.h>
#include <fmt/core.h>
#include <mpi.h>
#include <sstream>
#include "../../libs/_dict_builder.h"
#include "../../libs/_distributed.h"
#include "../../libs/groupby/_groupby_ftypes.h"
#include "../../libs/streaming/_groupby.h"
#include "../../libs/streaming/_shuffle.h"
#include "../table_generator.hpp"
#include "../test.hpp"

// Mock class to test protected methods
class IncrementalShuffleStateTest : public IncrementalShuffleState {
   public:
    IncrementalShuffleStateTest(
        const std::vector<int8_t>& arr_c_types_,
        const std::vector<int8_t>& arr_array_types_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        const uint64_t n_keys_, uint64_t& curr_iter_, int64_t& sync_freq_)
        : IncrementalShuffleState(arr_c_types_, arr_array_types_,
                                  dict_builders_, n_keys_, curr_iter_,
                                  sync_freq_, -1) {};
    IncrementalShuffleStateTest(
        std::shared_ptr<bodo::Schema> schema_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        const uint64_t n_keys_, const uint64_t& curr_iter_, int64_t& sync_freq_)
        : IncrementalShuffleState(schema_, dict_builders_, n_keys_, curr_iter_,
                                  sync_freq_, -1) {};
    void ResetAfterShuffle() override {
        IncrementalShuffleState::ResetAfterShuffle();
    }
};

std::shared_ptr<table_info> test_shuffle(std::shared_ptr<table_info> table) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::unique_ptr<bodo::Schema> schema = table->schema();
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    for (size_t i = 0; i < table->ncols(); i++) {
        dict_builders.push_back(create_dict_builder_for_array(
            std::shared_ptr<bodo::DataType>(schema->column_types[i]->copy()),
            true));
        if (dict_builders[i] != nullptr) {
            table->columns[i] =
                dict_builders[i]->UnifyDictionaryArray(table->columns[i]);
        }
    }

    int64_t sync_freq = 10;
    // Test with a length % 8 != 0
    uint64_t curr_iter = 0;
    IncrementalShuffleStateTest state = IncrementalShuffleStateTest(
        std::move(schema), dict_builders, table->ncols(), curr_iter, sync_freq);
    state.Initialize(nullptr, true, MPI_COMM_WORLD);
    if (rank == 0) {
        size_t nrows = table->nrows();
        std::vector<bool> append = std::vector<bool>(nrows, true);
        state.AppendBatch(std::move(table), append);
    }
    std::shared_ptr<table_info> shuffle_table = nullptr;
    MPI_Request finish_req = MPI_REQUEST_NULL;
    int finish_flag = 0;
    do {
        auto shuffle_result = state.ShuffleIfRequired(true);
        if (state.SendRecvEmpty() && !finish_flag) {
            if (finish_req == MPI_REQUEST_NULL) {
                CHECK_MPI(MPI_Ibarrier(MPI_COMM_WORLD, &finish_req),
                          "test_shuffle: MPI error on MPI_Ibarrier:");
            } else {
                CHECK_MPI(
                    MPI_Test(&finish_req, &finish_flag, MPI_STATUS_IGNORE),
                    "test_shuffle: MPI error on MPI_Test:");
            }
        }
        if (shuffle_result.has_value()) {
            shuffle_table = shuffle_result.value();
        }
    } while (!finish_flag || !state.SendRecvEmpty());
    return shuffle_table;
}

int64_t shuffle_groupby(GroupbyIncrementalShuffleState& state) {
    int64_t n_recv_rows = 0;
    bool barrier_started = false;
    MPI_Request finish_req = MPI_REQUEST_NULL;
    int finish_flag = 0;
    do {
        auto shuffle_result = state.ShuffleIfRequired(true);
        if (state.SendRecvEmpty() && !finish_flag) {
            if (!barrier_started) {
                CHECK_MPI(MPI_Ibarrier(MPI_COMM_WORLD, &finish_req),
                          "shuffle_groupby: MPI error on MPI_Ibarrier:");
                barrier_started = true;
            } else {
                CHECK_MPI(
                    MPI_Test(&finish_req, &finish_flag, MPI_STATUS_IGNORE),
                    "shuffle_groupby: MPI error on MPI_Test:");
            }
        }
        if (shuffle_result.has_value()) {
            n_recv_rows += shuffle_result.value()->nrows();
        }
    } while (!finish_flag || !state.SendRecvEmpty());
    CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &n_recv_rows, 1, MPI_LONG_LONG_INT,
                            MPI_SUM, MPI_COMM_WORLD),
              "shuffle_groupby: MPI error on MPI_Allreduce:");
    return n_recv_rows;
}

static bodo::tests::suite tests([] {
    bodo::tests::test("test_build_table_recreate", [] {
        int npes;
        MPI_Comm_size(MPI_COMM_WORLD, &npes);
        if (npes != 1) {
            std::cout
                << "this test depends on all data shuffling to the same rank"
                << std::endl;
            return;
        }

        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders =
            std::vector<std::shared_ptr<DictionaryBuilder>>(1);
        int64_t sync_freq = 10;
        uint64_t curr_iter = 0;
        IncrementalShuffleStateTest state = IncrementalShuffleStateTest(
            {0}, {0}, dict_builders, 1, curr_iter, sync_freq);

        // Make it look like the shuffle buffer is too large
        // The utilization should already be low since we haven't added any data
        state.table_buffer->array_buffers[0]
            .data_array->buffers[0]
            ->getMeminfo()
            ->size = 100000000000000000L;

        // Keep a reference to the original table buffer
        auto* original_table_buffer = state.table_buffer.get();
        // This should recreate the table buffer
        state.ResetAfterShuffle();

        // Check that the table buffer is recreated
        bodo::tests::check(state.table_buffer->array_buffers[0]
                               .data_array->buffers[0]
                               ->getMeminfo()
                               ->size == 0);

        bodo::tests::check(state.table_buffer.get() != original_table_buffer);
    });
    bodo::tests::test("test_mrnf_shuffle_local_reduction", [] {
        int npes;
        MPI_Comm_size(MPI_COMM_WORLD, &npes);
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders =
            std::vector<std::shared_ptr<DictionaryBuilder>>(2);
        int64_t sync_freq = 10;
        int64_t curr_iter = 0;
        std::vector<int8_t> arr_arr_types = {
            bodo_array_type::NULLABLE_INT_BOOL,
            bodo_array_type::NULLABLE_INT_BOOL};
        std::vector<int8_t> arr_c_types = {Bodo_CTypes::INT32,
                                           Bodo_CTypes::UINT64};
        std::vector<std::shared_ptr<array_info>> local_input_cols = {
            alloc_nullable_array(10, (Bodo_CTypes::CTypeEnum)arr_c_types[1])};
        std::shared_ptr<BasicColSet> col_set =
            makeColSet(local_input_cols,                    // in_cols
                       nullptr,                             // index_col
                       Bodo_FTypes::min_row_number_filter,  // ftype
                       true,                                // do_combine
                       true,                                // skip_na_data
                       0,                                   // period
                       {0},                                 // transform_funcs
                       0,                                   // n_udf
                       false,                               // parallel
                       {true}, {true},  // window_na_position
                       nullptr,         // window_args
                       0,               // n_input_cols
                       nullptr,         // udf_n_redvars
                       nullptr,         // udf_table
                       0,               // udf_table_idx
                       nullptr,         // nunique_table
                       true             // use_sql_rules
            );
        auto schema = bodo::Schema::Deserialize(arr_arr_types, arr_c_types);
        std::vector<std::shared_ptr<BasicColSet>> col_sets = {col_set};
        std::vector<int32_t> dummy_f_running_value_offsets;
        GroupbyIncrementalShuffleState state = GroupbyIncrementalShuffleState(
            std::make_shared<bodo::Schema>(*schema), dict_builders, col_sets, 1,
            1, curr_iter, sync_freq, 0, false, AggregationType::MRNF,
            dummy_f_running_value_offsets, false);
        state.Initialize(nullptr, true, MPI_COMM_WORLD);
        // Table with only 1 unique key, should perform local reduction
        auto table = bodo::tests::cppToBodo({"A", "B"}, {true, true}, {},
                                            std::vector<int32_t>(100, 0),
                                            std::vector<uint64_t>(100, 1));
        std::vector<bool> append = std::vector<bool>(table->nrows(), true);
        state.AppendBatch(std::move(table), append);
        int64_t n_global_rows = shuffle_groupby(state);
        bodo::tests::check(n_global_rows == npes);

        std::vector<int32_t> keys;
        for (size_t i = 0; i < 100; ++i) {
            keys.push_back(i);
        }
        // Table with all unique keys, shouldn't perform local reduction
        auto table2 = bodo::tests::cppToBodo({"A", "B"}, {true, true}, {}, keys,
                                             std::vector<uint64_t>(100, 1));
        state.AppendBatch(std::move(table2), append);
        n_global_rows = shuffle_groupby(state);
        bodo::tests::check(n_global_rows == (npes * 100));
    });
    bodo::tests::test("test_async_shuffle_nullable_bool", [] {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto test_bool_shuffle = [&]<Bodo_CTypes::CTypeEnum first_array_dtype>(
                                     size_t array_size) {
            using ctype = dtype_to_type<first_array_dtype>::type;
            auto table = bodo::tests::cppToBodo(
                {"A", "B"}, {true, true}, {}, std::vector<ctype>(array_size, 1),
                std::vector<bool>(array_size, true));

            uint64_t n_nulls_expected = 0;
            uint64_t n_false_expected = 0;
            if (rank == 0) {
                for (size_t i = 0; i < array_size; i += 25) {
                    table->columns[0]->set_null_bit(i, false);
                    n_nulls_expected++;
                }
                for (size_t i = 0; i < array_size; i += 20) {
                    arrow::bit_util::SetBitTo(
                        (uint8_t*)table->columns[1]->data1(), i, false);
                    n_false_expected++;
                }
            }
            CHECK_MPI(
                MPI_Bcast(&n_nulls_expected, 1, MPI_UNSIGNED_LONG_LONG, 0,
                          MPI_COMM_WORLD),
                "test_async_shuffle_nullable_bool: MPI error on MPI_Bcast:");
            CHECK_MPI(
                MPI_Bcast(&n_false_expected, 1, MPI_UNSIGNED_LONG_LONG, 0,
                          MPI_COMM_WORLD),
                "test_async_shuffle_nullable_bool: MPI error on MPI_Bcast:");
            std::shared_ptr<table_info> shuffle_table =
                test_shuffle(std::move(table));
            uint64_t n_nulls = 0;
            uint64_t n_false = 0;
            uint64_t length_col1 = 0;
            uint64_t length_col2 = 0;
            if (shuffle_table.get() != nullptr) {
                length_col1 = shuffle_table->columns[0]->length;
                length_col2 = shuffle_table->columns[1]->length;
                for (size_t i = 0; i < shuffle_table->nrows(); i++) {
                    if (!shuffle_table->columns[0]->get_null_bit(i)) {
                        n_nulls++;
                    }
                    if (!arrow::bit_util::GetBit(
                            shuffle_table->columns[1]
                                ->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                        uint8_t>(),
                            i)) {
                        n_false++;
                    }
                }
            }
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &n_nulls, 1, MPI_UNSIGNED_LONG_LONG,
                              MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_nullable_bool: MPI error on "
                "MPI_Allreduce:");
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &n_false, 1, MPI_UNSIGNED_LONG_LONG,
                              MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_nullable_bool: MPI error on "
                "MPI_Allreduce:");
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &length_col1, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_nullable_bool: MPI error on "
                "MPI_Allreduce:");
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &length_col2, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_nullable_bool: MPI error on "
                "MPI_Allreduce:");
            bodo::tests::check(
                length_col1 == array_size,
                fmt::format("length_col1: {} != {}", length_col1, array_size)
                    .c_str());
            bodo::tests::check(
                n_nulls == n_nulls_expected,
                fmt::format("{} != {}", n_nulls, n_nulls_expected).c_str());
            bodo::tests::check(n_false == n_false_expected);
            bodo::tests::check(length_col2 == array_size);
        };
        for (size_t i = 1; i < 1000; i++) {
            test_bool_shuffle.operator()<Bodo_CTypes::_BOOL>(i);
        }
        for (size_t i = 1; i < 1000; i++) {
            test_bool_shuffle.operator()<Bodo_CTypes::INT32>(i);
        }
    });
    bodo::tests::test("test_async_shuffle_dict", [] {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto test_dict_shuffle = [&](size_t array_size) {
            std::vector<std::string> dict_data_vec =
                std::vector<std::string>(array_size);
            for (size_t i = 0; i < array_size; i++) {
                dict_data_vec[i] = std::to_string(i % 10);
            }
            auto table = bodo::tests::cppToBodo(
                {"A", "B"}, {false, true}, {"B"}, std::vector<int>(array_size),
                dict_data_vec);

            uint64_t n_nulls_expected = 0;
            uint64_t values_expected[10] = {};
            if (rank == 0) {
                for (size_t i = 0; i < array_size; i += 25) {
                    table->columns[1]->set_null_bit(i, false);
                    n_nulls_expected++;
                }
                for (size_t i = 0; i < array_size; i++) {
                    if (table->columns[1]->get_null_bit(i)) {
                        values_expected[i % 10]++;
                    }
                }
            }
            CHECK_MPI(MPI_Bcast(&n_nulls_expected, 1, MPI_UNSIGNED_LONG_LONG, 0,
                                MPI_COMM_WORLD),
                      "test_async_shuffle_dict: MPI error on MPI_Bcast:");
            CHECK_MPI(MPI_Bcast(&values_expected, 10, MPI_UNSIGNED_LONG_LONG, 0,
                                MPI_COMM_WORLD),
                      "test_async_shuffle_dict: MPI error on MPI_Bcast:");
            std::shared_ptr<table_info> shuffle_table =
                test_shuffle(std::move(table));
            uint64_t n_nulls = 0;
            uint64_t length_col1 = 0;
            uint64_t values[10] = {};
            if (shuffle_table.get() != nullptr) {
                length_col1 = shuffle_table->columns[1]->length;
                for (size_t i = 0; i < shuffle_table->nrows(); i++) {
                    if (!shuffle_table->columns[1]->get_null_bit(i)) {
                        n_nulls++;
                    }
                    // This only works because all of the members of the dict
                    // are 1 character so we don't need to check offsets
                    dict_indices_t dict_idx =
                        shuffle_table->columns[1]
                            ->child_arrays[1]
                            ->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                    dict_indices_t>()[i];
                    if (dict_idx >= 0) {
                        char val =
                            shuffle_table->columns[1]
                                ->child_arrays[0]
                                ->data1<bodo_array_type::STRING>()[dict_idx];
                        int idx = std::stoi(std::string({val}));
                        assert(idx < 10);
                        assert(idx >= 0);
                        values[idx]++;
                    }
                }
            }
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &n_nulls, 1, MPI_UNSIGNED_LONG_LONG,
                              MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_dict: MPI error on MPI_Allreduce:");
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &length_col1, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_dict: MPI error on MPI_Allreduce:");
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &values, 10, MPI_UNSIGNED_LONG_LONG,
                              MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_dict: MPI error on MPI_Allreduce:");

            bodo::tests::check(n_nulls == n_nulls_expected);
            for (size_t i = 0; i < 10; i++) {
                bodo::tests::check(values[i] == values_expected[i]);
            }
            bodo::tests::check(length_col1 == array_size);
        };
        for (size_t i = 1; i < 1000; i++) {
            test_dict_shuffle(i);
        }
    });
    bodo::tests::test("test_async_shuffle_array_item", [] {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto test_array_item_shuffle = [&](size_t array_size,
                                           std::shared_ptr<array_info>
                                               inner_arr) {
            std::shared_ptr<array_info> array =
                alloc_array_item(array_size, inner_arr);
            size_t curr_offset = 0;
            array->data1<bodo_array_type::ARRAY_ITEM, offset_t>()[0] = 0;
            size_t expected_inner_lens[6] = {};
            for (size_t i = 1; i < array_size + 1; i++) {
                size_t inner_len = i % 6;
                expected_inner_lens[inner_len]++;
                curr_offset += inner_len;
                array->data1<bodo_array_type::ARRAY_ITEM, offset_t>()[i] =
                    curr_offset;
            }
            bodo::tests::check(curr_offset <= array_size * 3);

            std::vector<std::shared_ptr<array_info>> cols = {array};
            auto table = std::make_unique<table_info>(cols);
            std::shared_ptr<table_info> shuffle_table =
                test_shuffle(std::move(table));
            uint64_t length_col1 = 0;
            uint64_t length_inner_arr = 0;
            uint64_t inner_lens[6] = {};
            if (shuffle_table.get() != nullptr) {
                length_col1 = shuffle_table->columns[0]->length;
                length_inner_arr =
                    shuffle_table->columns[0]->child_arrays[0]->length;

                for (size_t i = 0; i < shuffle_table->nrows(); i++) {
                    size_t inner_len =
                        shuffle_table->columns[0]
                            ->data1<bodo_array_type::ARRAY_ITEM,
                                    offset_t>()[i + 1] -
                        shuffle_table->columns[0]
                            ->data1<bodo_array_type::ARRAY_ITEM, offset_t>()[i];
                    inner_lens[inner_len]++;
                }
                bodo::tests::check(
                    shuffle_table->ncols() == 1,
                    fmt::format("ncols: {} != 1", shuffle_table->ncols())
                        .c_str());
            }

            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &length_col1, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_array_item: MPI error on MPI_Allreduce:");
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &length_inner_arr, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_array_item: MPI error on MPI_Allreduce:");
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &inner_lens, 6,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_array_item: MPI error on MPI_Allreduce:");
            bodo::tests::check(
                length_col1 == array_size,
                fmt::format("length_col1 {} != {}", length_col1, array_size)
                    .c_str());
            bodo::tests::check(length_inner_arr == curr_offset,
                               fmt::format("length_inner_arr {} != {}",
                                           length_inner_arr, curr_offset)
                                   .c_str());
            for (size_t i = 0; i < 6; i++) {
                bodo::tests::check(inner_lens[i] == expected_inner_lens[i],
                                   fmt::format("idx i: {} != {}", inner_lens[i],
                                               expected_inner_lens[i])
                                       .c_str());
            }
        };

        for (size_t i = 1; i < 1000; i++) {
            std::shared_ptr<array_info> inner_arr = alloc_array_top_level(
                i * 3 + 1, 0, 0, bodo_array_type::NULLABLE_INT_BOOL,
                Bodo_CTypes::INT32);
            test_array_item_shuffle(i, inner_arr);
        }

        for (size_t i = 1; i < 1000; i++) {
            std::shared_ptr<array_info> inner_arr = alloc_array_top_level(
                i * 3 + 1, 0, 0, bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64);
            test_array_item_shuffle(i, inner_arr);
        }

        for (size_t i = 1; i < 1000; i++) {
            std::shared_ptr<array_info> inner_arr = alloc_array_top_level(
                i * 3 + 1, 0, 0, bodo_array_type::NULLABLE_INT_BOOL,
                Bodo_CTypes::_BOOL);
            test_array_item_shuffle(i, inner_arr);
        }

        for (size_t i = 1; i < 1000; i++) {
            std::shared_ptr<array_info> inner_arr = alloc_string_array(
                Bodo_CTypes::STRING, i * 3 + 1, (i * 3 + 1) * 2);
            for (size_t j = 1; j < inner_arr->length; j++) {
                inner_arr->data2<bodo_array_type::STRING, offset_t>()[j] =
                    inner_arr
                        ->data2<bodo_array_type::STRING, offset_t>()[j - 1] +
                    1;
            }
            test_array_item_shuffle(i, inner_arr);
        }

        for (size_t i = 1; i < 1000; i++) {
            std::vector<std::string> dict_data(i * 3 + 1);
            for (size_t j = 0; j < i * 3 + 1; j++) {
                dict_data[j] = std::to_string(j % 10);
            }
            std::shared_ptr<array_info> inner_arr =
                bodo::tests::cppToBodo({"B"}, {true}, {"B"}, dict_data)
                    ->columns[0];
            test_array_item_shuffle(i, inner_arr);
        }

        for (size_t i = 1; i < 1000; i++) {
            std::shared_ptr<array_info> inner_inner_arr = alloc_array_top_level(
                i * 3 + 1, 0, 0, bodo_array_type::NULLABLE_INT_BOOL,
                Bodo_CTypes::INT32);
            std::shared_ptr<array_info> inner_arr =
                alloc_array_item((i * 3 + 1) * 2, inner_inner_arr);

            test_array_item_shuffle(i, inner_arr);
        }

        // BSE-3548 support hashing timestamp tz in nested arrays
        // for (size_t i = 1; i < 1000; i++) {
        //    std::shared_ptr<array_info> inner_arr = alloc_array_top_level(
        //        i * 3 + 1, 0, 0, bodo_array_type::TIMESTAMPTZ,
        //        Bodo_CTypes::TIMESTAMPTZ);
        //    test_array_item_shuffle(i, inner_arr);
        //}
        for (size_t i = 1; i < 1000; i++) {
            std::shared_ptr<array_info> inner_inner_arr = alloc_array_top_level(
                i * 3 + 1, 0, 0, bodo_array_type::NULLABLE_INT_BOOL,
                Bodo_CTypes::INT32);
            std::vector<std::shared_ptr<array_info>> struct_fields = {
                inner_inner_arr};
            std::shared_ptr<array_info> inner_arr =
                alloc_struct((i * 3 + 1) * 2, struct_fields);

            test_array_item_shuffle(i, inner_arr);
        }

        // List of map of int32 to float64
        for (size_t i = 1; i < 1000; i++) {
            std::shared_ptr<array_info> key_arr = alloc_array_top_level(
                i * 3 + 1, 0, 0, bodo_array_type::NULLABLE_INT_BOOL,
                Bodo_CTypes::INT32);
            std::shared_ptr<array_info> val_arr = alloc_array_top_level(
                i * 3 + 1, 0, 0, bodo_array_type::NULLABLE_INT_BOOL,
                Bodo_CTypes::FLOAT64);
            std::vector<std::shared_ptr<array_info>> struct_fields = {key_arr,
                                                                      val_arr};
            std::shared_ptr<array_info> struct_arr =
                alloc_struct((i * 3 + 1) * 2, struct_fields);
            std::shared_ptr<array_info> map_list =
                alloc_array_item((i * 3 + 1) * 2, struct_arr);
            std::shared_ptr<array_info> inner_arr =
                alloc_map((i * 3 + 1) * 2, map_list);

            test_array_item_shuffle(i, inner_arr);
        }
    });
    bodo::tests::test("test_async_shuffle_timestamptz", [] {
        auto test_timestamptz_shuffle = [&](size_t array_size) {
            std::shared_ptr<array_info> timestamptz_arr =
                alloc_timestamptz_array(array_size);
            int expected_nulls = 0;
            int expected_timestamp_sum = 0;
            int expected_int16_sum = 0;
            for (size_t i = 0; i < array_size; i++) {
                timestamptz_arr
                    ->data1<bodo_array_type::TIMESTAMPTZ,
                            dtype_to_type<Bodo_CTypes::DATETIME>::type>()[i] =
                    i;
                timestamptz_arr
                    ->data2<bodo_array_type::TIMESTAMPTZ,
                            dtype_to_type<Bodo_CTypes::INT16>::type>()[i] = i;

                // Set nulls with some non-byte aligned pattern
                if (i % 13 == 0) {
                    timestamptz_arr->set_null_bit(i, false);
                    expected_nulls++;
                } else {
                    expected_timestamp_sum += i;
                    expected_int16_sum += i;
                }
            }
            auto table_arrs =
                std::vector<std::shared_ptr<array_info>>{timestamptz_arr};
            auto table = std::make_unique<table_info>(table_arrs);
            std::shared_ptr<table_info> shuffle_table =
                test_shuffle(std::move(table));
            int nulls = 0;
            int timestamp_sum = 0;
            int int16_sum = 0;
            if (shuffle_table.get() != nullptr) {
                for (size_t i = 0; i < shuffle_table->nrows(); i++) {
                    if (shuffle_table->columns[0]->get_null_bit(i)) {
                        timestamp_sum +=
                            shuffle_table->columns[0]
                                ->data1<bodo_array_type::TIMESTAMPTZ,
                                        dtype_to_type<
                                            Bodo_CTypes::DATETIME>::type>()[i];
                        int16_sum +=
                            shuffle_table->columns[0]
                                ->data2<bodo_array_type::TIMESTAMPTZ,
                                        dtype_to_type<
                                            Bodo_CTypes::INT16>::type>()[i];
                    } else {
                        nulls++;
                    }
                }
            }
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &nulls, 1, MPI_INT, MPI_SUM,
                              MPI_COMM_WORLD),
                "test_async_shuffle_timestamptz: MPI error on MPI_Allreduce:");
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &timestamp_sum, 1, MPI_INT, MPI_SUM,
                              MPI_COMM_WORLD),
                "test_async_shuffle_timestamptz: MPI error on MPI_Allreduce:");
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &int16_sum, 1, MPI_INT, MPI_SUM,
                              MPI_COMM_WORLD),
                "test_async_shuffle_timestamptz: MPI error on MPI_Allreduce:");
            bodo::tests::check(nulls == expected_nulls);
            bodo::tests::check(timestamp_sum == expected_timestamp_sum);
            bodo::tests::check(int16_sum == expected_int16_sum);
        };
        for (size_t i = 1; i < 1000; i++) {
            test_timestamptz_shuffle(i);
        }
    });
    bodo::tests::test("test_async_shuffle_struct", [] {
        auto test_struct_shuffle = [&](size_t array_size) {
            auto int_arr = alloc_array_top_level(
                array_size, 0, 0, bodo_array_type::NUMPY, Bodo_CTypes::INT32);
            auto bool_arr = alloc_array_top_level(
                array_size, 0, 0, bodo_array_type::NULLABLE_INT_BOOL,
                Bodo_CTypes::_BOOL);
            auto string_arr =
                bodo::tests::cppToBodo({"A"}, {false}, {"A"},
                                       std::vector<std::string>(array_size))
                    ->columns[0];
            auto dict_arr =
                bodo::tests::cppToBodo({"A"}, {true}, {"A"},
                                       std::vector<std::string>(array_size))
                    ->columns[0];
            auto array_item_arr =
                alloc_array_item(array_size, std::move(int_arr));
            std::vector<std::shared_ptr<array_info>> child_arrays = {
                std::move(bool_arr), std::move(string_arr), std::move(dict_arr),
                std::move(array_item_arr)};
            auto struct_arr = alloc_struct(array_size, child_arrays);
            std::vector<std::shared_ptr<array_info>> cols = {
                std::move(struct_arr)};
            auto table = std::make_unique<table_info>(cols);
            std::shared_ptr<table_info> shuffle_table =
                test_shuffle(std::move(table));
            uint64_t length_col1 = 0;
            std::vector<uint64_t> child_lens(child_arrays.size(), 0);
            if (shuffle_table != nullptr) {
                length_col1 = shuffle_table->columns[0]->length;
                bodo::tests::check(
                    shuffle_table->columns[0]->child_arrays.size() ==
                    child_arrays.size());
                for (size_t i = 0; i < child_arrays.size(); ++i) {
                    child_lens[i] +=
                        shuffle_table->columns[0]->child_arrays[i]->length;
                }
            }
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &length_col1, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_struct: MPI error on MPI_Allreduce:");
            CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, child_lens.data(),
                                    child_lens.size(), MPI_UNSIGNED_LONG_LONG,
                                    MPI_SUM, MPI_COMM_WORLD),
                      "test_async_shuffle_struct: MPI error on MPI_Allreduce:");
            for (auto child_len : child_lens) {
                bodo::tests::check(child_len == array_size);
            }
            bodo::tests::check(
                length_col1 == array_size,
                fmt::format("length_col1 {} != {}", length_col1, array_size)
                    .c_str());
        };

        for (size_t i = 1; i < 1000; i++) {
            test_struct_shuffle(i);
        }
    });
    bodo::tests::test("test_async_shuffle_map", []() {
        auto test_map_shuffle = [&](size_t array_size) {
            std::shared_ptr<array_info> key_arr = alloc_array_top_level(
                array_size * 2, 0, 0, bodo_array_type::NUMPY,
                Bodo_CTypes::INT32);
            std::shared_ptr<array_info> val_arr = alloc_array_top_level(
                array_size * 2, 0, 0, bodo_array_type::NULLABLE_INT_BOOL,
                Bodo_CTypes::FLOAT64);

            std::vector<std::shared_ptr<array_info>> struct_fields = {key_arr,
                                                                      val_arr};
            std::shared_ptr<array_info> struct_arr =
                alloc_struct(array_size * 2, struct_fields);

            std::shared_ptr<array_info> array_item_arr =
                alloc_array_item(array_size, struct_arr);
            for (size_t i = 1; i < array_size + 1; i++) {
                array_item_arr
                    ->data1<bodo_array_type::ARRAY_ITEM, offset_t>()[i] =
                    i * 2 - 1;
            }
            size_t expected_len_keys =
                array_item_arr->data1<bodo_array_type::ARRAY_ITEM,
                                      offset_t>()[array_size];

            size_t expected_nulls = 0;
            // Set nulls with some non-byte aligned pattern
            for (size_t i = 0; i < array_size; i += 13) {
                array_item_arr->set_null_bit(i, false);
                expected_nulls++;
            }

            std::shared_ptr<array_info> map_arr =
                alloc_map(array_size, array_item_arr);

            std::vector<std::shared_ptr<array_info>> cols = {map_arr};
            auto table = std::make_unique<table_info>(cols);

            std::shared_ptr<table_info> shuffle_table =
                test_shuffle(std::move(table));
            uint64_t shuffled_nulls = 0;
            uint64_t length_col1 = 0;
            uint64_t length_keys = 0;
            if (shuffle_table != nullptr) {
                auto shuffled_child =
                    shuffle_table->columns[0]->child_arrays[0];
                length_col1 = shuffled_child->length;
                length_keys =
                    shuffled_child->child_arrays[0]->child_arrays[0]->length;
                for (size_t i = 0; i < shuffled_child->length; ++i) {
                    if (!shuffled_child->get_null_bit(i)) {
                        shuffled_nulls++;
                    }
                }
                bodo::tests::check(shuffle_table->ncols() == 1);
            }
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &length_col1, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_map: MPI error on MPI_Allreduce:");
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &shuffled_nulls, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_map: MPI error on MPI_Allreduce:");
            CHECK_MPI(
                MPI_Allreduce(MPI_IN_PLACE, &length_keys, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
                "test_async_shuffle_map: MPI error on MPI_Allreduce:");
            bodo::tests::check(shuffled_nulls == expected_nulls);
            bodo::tests::check(length_col1 == array_size);
            bodo::tests::check(length_keys == expected_len_keys);
        };

        for (size_t i = 1; i < 1000; i++) {
            test_map_shuffle(i);
        }
    });
    bodo::tests::test("test_send_dict_resize_works", []() {
        // trying to reproduce seg fault where data buff gets reallocated before
        // it can be sent.

        int my_rank;
        int n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        // test is designed for 2 ranks
        if (n_pes != 2) {
            return;
        }

        size_t array_size = 100;
        size_t num_new_rows = 100;

        std::shared_ptr<table_info> empty_table = bodo::tests::cppToBodo(
            {"A"}, {true}, {"A"}, std::vector<std::string>());

        std::vector<std::string> expected_vec(array_size);
        for (size_t i = 0; i < array_size; i++) {
            expected_vec[i] = std::to_string(i);
        }
        std::shared_ptr<table_info> expected_table =
            bodo::tests::cppToBodo({"A"}, {true}, {"A"}, expected_vec);

        // send states for rank 0
        std::vector<AsyncShuffleSendState> send_states;
        if (my_rank == 0) {
            std::vector<std::string> dict_data_vec =
                std::vector<std::string>(array_size);
            for (size_t i = 0; i < array_size; i++) {
                dict_data_vec[i] = std::to_string(i);
            }

            std::shared_ptr<table_info> initial_table =
                bodo::tests::cppToBodo({"A"}, {true}, {"A"}, dict_data_vec);

            std::shared_ptr<uint32_t[]> hashes =
                std::make_shared<uint32_t[]>(array_size, 1);

            // create table buffer and append first set of values
            std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
            dict_builders.push_back(create_dict_builder_for_array(
                initial_table->schema()->column_types[0]->copy(), false));
            TableBuildBuffer table_buffer(initial_table->schema(),
                                          dict_builders);

            table_buffer.UnifyTablesAndAppend(initial_table, dict_builders);

            // wait until after send to append new
            std::vector<std::string> new_vals(num_new_rows);
            for (size_t i = 0; i < num_new_rows; i++) {
                new_vals[i] = std::to_string(i + array_size);
            }
            std::shared_ptr<table_info> new_vals_table =
                bodo::tests::cppToBodo({"A"}, {true}, {"A"}, new_vals);

            void* loc_before =
                (void*)(dict_builders[0]->dict_buff->data_array->data1());

            send_states.push_back(shuffle_issend(
                table_buffer.data_table, hashes, nullptr, MPI_COMM_WORLD));

            table_buffer.UnifyTablesAndAppend(new_vals_table, dict_builders);
            void* loc_after =
                (void*)(dict_builders[0]->dict_buff->data_array->data1());

            // make sure the raw pointer to the data buff is actually different
            bodo::tests::check(loc_after != loc_before);
        }
        // ensure resizing happens before recv
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD),
                  "test_send_dict_resize_works: MPI error on MPI_Barrier:");

        // recv the resized dictionary encoded array column
        bool tables_match;
        if (my_rank == 1) {
            std::vector<AsyncShuffleRecvState> recv_states;
            shuffle_irecv(empty_table, MPI_COMM_WORLD, recv_states);
            std::unique_ptr<bodo::Schema> schema = empty_table->schema();

            std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
            for (size_t i = 0; i < empty_table->ncols(); i++) {
                dict_builders.push_back(create_dict_builder_for_array(
                    schema->column_types[i]->copy(), false));
            }
            TableBuildBuffer result_table(std::move(schema), dict_builders);
            IncrementalShuffleMetrics metrics;
            while (recv_states.size() != 0) {
                consume_completed_recvs(recv_states, MPI_COMM_WORLD,
                                        dict_builders, metrics, result_table);
            }

            // check that the recv'd table looks correct
            std::stringstream ss1;
            std::stringstream ss2;
            DEBUG_PrintTable(ss1, result_table.data_table);
            DEBUG_PrintTable(ss2, expected_table);
            tables_match = ss1.str() == ss2.str();
        }
        CHECK_MPI(
            MPI_Bcast(&tables_match, 1, MPI_UNSIGNED_CHAR, 1, MPI_COMM_WORLD),
            "test_send_dict_resize_works: MPI error on MPI_Bcast:");

        bodo::tests::check(tables_match);

        // cleanup
        if (my_rank == 0) {
            while (send_states.size() > 0) {
                std::erase_if(send_states, [&](AsyncShuffleSendState& s) {
                    return s.sendDone();
                });
            }
        }
    });
});
