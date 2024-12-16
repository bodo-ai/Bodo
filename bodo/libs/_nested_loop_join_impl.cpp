#include "_nested_loop_join_impl.h"
#include "_dict_builder.h"
#include "_distributed.h"
#include "_memory.h"

std::tuple<std::vector<array_info*>, std::vector<void*>, std::vector<void*>>
get_gen_cond_data_ptrs(std::shared_ptr<table_info> table) {
    std::vector<array_info*> array_infos;
    std::vector<void*> col_ptrs;
    std::vector<void*> null_bitmaps;

    // get raw array_info pointers for cond_func
    get_gen_cond_data_ptrs(table, &array_infos, &col_ptrs, &null_bitmaps);
    return std::make_tuple(array_infos, col_ptrs, null_bitmaps);
}

void get_gen_cond_data_ptrs(std::shared_ptr<table_info> table,
                            std::vector<array_info*>* array_infos,
                            std::vector<void*>* col_ptrs,
                            std::vector<void*>* null_bitmaps) {
    array_infos->reserve(table->ncols());
    col_ptrs->reserve(table->ncols());
    null_bitmaps->reserve(table->ncols());

    for (const std::shared_ptr<array_info>& arr : table->columns) {
        array_infos->push_back(arr.get());
        col_ptrs->push_back(static_cast<void*>(arr->data1()));
        null_bitmaps->push_back(static_cast<void*>(arr->null_bitmask()));
    }
}

void nested_loop_join_handle_dict_encoded(
    std::shared_ptr<table_info> left_table,
    std::shared_ptr<table_info> right_table, bool left_parallel,
    bool right_parallel) {
    // make all dictionaries global (necessary for broadcast and potentially
    // other operations)
    for (std::shared_ptr<array_info> a : left_table->columns) {
        if (a->arr_type == bodo_array_type::DICT) {
            make_dictionary_global_and_unique(a, left_parallel);
        }
    }
    for (std::shared_ptr<array_info> a : right_table->columns) {
        if (a->arr_type == bodo_array_type::DICT) {
            make_dictionary_global_and_unique(a, right_parallel);
        }
    }
}

void add_unmatched_rows(std::span<uint8_t> bit_map, size_t n_rows,
                        bodo::vector<int64_t>& table_idxs,
                        bodo::vector<int64_t>& other_table_idxs,
                        bool needs_reduction, int64_t offset) {
    if (needs_reduction) {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        MPI_Allreduce_bool_or(bit_map);
        int pos = 0;
        for (size_t i = 0; i < n_rows; i++) {
            bool bit = GetBit(bit_map.data(), i + offset);
            // distribute the replicated input table rows across ranks
            // to load balance the output
            if (!bit) {
                int node = pos % n_pes;
                if (node == myrank) {
                    table_idxs.emplace_back(i);
                    other_table_idxs.emplace_back(-1);
                }
                pos++;
            }
        }
    } else {
        for (size_t i = 0; i < n_rows; i++) {
            bool bit = GetBit(bit_map.data(), i + offset);
            if (!bit) {
                table_idxs.emplace_back(i);
                other_table_idxs.emplace_back(-1);
            }
        }
    }
}
