#include "_fft.h"
#include <mpi.h>
#include <numeric>
#include "_array_utils.h"
#include "_distributed.h"
#include "tracing.h"

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex_dtype<dtype>
void import_wisdom(int rank) {
    if (rank == 0) {
        fftw_import_wisdom_from_filename_fn<dtype>();
    }
    fftw_mpi_broadcast_wisdom_fn<dtype>(MPI_COMM_WORLD);
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex_dtype<dtype>
void export_wisdom(int rank) {
    fftw_mpi_gather_wisdom_fn<dtype>(MPI_COMM_WORLD);
    // We want to write the wisdom once on each host in case
    // rank 0 is on a different host next run.
    // This splits the  communicators into groups of ranks
    // with shared memory, which should be 1 per host.
    fftw_mpi_broadcast_wisdom_fn<dtype>(MPI_COMM_WORLD);
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shmcomm);
    int shmrank;
    MPI_Comm_rank(shmcomm, &shmrank);

    if (shmrank == 0) {
        fftw_export_wisdom_to_filename_fn<dtype>();
    }
}

/**
 * @brief Get FFTW planning flag from environment variable BODO_FFTW_PLANNING if
 * available. Otherwise, return FFTW_MEASURE by default. See //
 * https://www.fftw.org/fftw3_doc/Planner-Flags.html
 *
 * @return unsigned int FFTW planning flag (e.g. FFTW_MEASURE)
 */
unsigned int get_fftw_planning_flag() {
    char* fftw_planning_env_ = std::getenv("BODO_FFTW_PLANNING");

    if (fftw_planning_env_) {
        if (std::strcmp(fftw_planning_env_, "FFTW_ESTIMATE") == 0) {
            return FFTW_ESTIMATE;
        }
        if (std::strcmp(fftw_planning_env_, "FFTW_MEASURE") == 0) {
            return FFTW_MEASURE;
        }
        if (std::strcmp(fftw_planning_env_, "FFTW_PATIENT") == 0) {
            return FFTW_PATIENT;
        }
        if (std::strcmp(fftw_planning_env_, "FFTW_EXHAUSTIVE") == 0) {
            return FFTW_EXHAUSTIVE;
        }
        if (std::strcmp(fftw_planning_env_, "FFTW_WISDOM_ONLY") == 0) {
            return FFTW_WISDOM_ONLY;
        }
    }
    return FFTW_MEASURE;
}

/**
 * @brief Set timeout for FFTW planning which can take a long time, especially
 * on few cores. Uses BODO_FFTW_PLANNING_TIMEOUT environment variable if
 * available. Otherwises, sets to 1 hour by default (FFTW's default is
 * unlimited which can seem like hanging).
 *
 */
void set_fftw_timeout() {
    double seconds = 60.0 * 60.0;
    char* fftw_planning_env_ = std::getenv("BODO_FFTW_PLANNING_TIMEOUT");
    if (fftw_planning_env_) {
        seconds = std::stod(fftw_planning_env_);
    }
    fftw_set_timelimit(seconds);
}

/**
 * @brief Redistributes the given array to match fftw's expected distribution
 * @param arr The array to redistribute
 * @param out_arr The buffer to redistribute into
 * @param rank The rank of the current process
 * @param npes The number of processes
 * @param shape The shape of the array
 * @param local_arr_start The start of the local array as calculated by fftw
 * @param local_arr_len The length of the local array as calculated by fftw
 */
template <Bodo_CTypes::CTypeEnum dtype>
    requires complex_dtype<dtype>
void redistribute_for_fftw(std::shared_ptr<array_info>& arr,
                           std::shared_ptr<array_info>& out_arr, int rank,
                           int npes, uint64_t shape[2],
                           ptrdiff_t local_arr_start, ptrdiff_t local_arr_len) {
    tracing::Event ev = tracing::Event("redistribute_for_fftw");
    int arr_len = arr->length;
    std::vector<int> arr_lens = std::vector<int>(npes, 0);
    std::vector<int> local_arr_starts = std::vector<int>(npes + 1, 0);
    local_arr_starts[npes] = shape[0] * shape[1];
    HANDLE_MPI_ERROR(MPI_Allgather(&arr_len, 1, MPI_INT, arr_lens.data(), 1,
                                   MPI_INT, MPI_COMM_WORLD),
                     "redistribute_for_fftw: MPI error on MPI_Allgather:");
    HANDLE_MPI_ERROR(
        MPI_Allgather(&local_arr_start, 1, MPI_INT, local_arr_starts.data(), 1,
                      MPI_INT, MPI_COMM_WORLD),
        "redistribute_for_fftw: MPI error on MPI_Allgather:");
    size_t global_idx_offset = 0;
    for (int i = 0; i < rank; ++i) {
        global_idx_offset += arr_lens[i];
    }
    std::vector<int> sendoffsets = std::vector<int>(npes);
    sendoffsets[0] = 0;
    std::vector<int> sendcounts = std::vector<int>(npes, 0);
    size_t target_rank = 0;
    for (int i = 0; i < arr_len; ++i) {
        int global_idx = global_idx_offset + i;
        // Depending on the data distribution sometimes fftw decides not to
        // assign any data to a rank and gives it a local_arr_start of 0, this
        // would then always give the row to the last rank instead of the
        // assigned rank
        while (global_idx >= local_arr_starts[target_rank + 1] &&
               local_arr_starts[target_rank + 1] != 0) {
            target_rank++;
            sendoffsets[target_rank] = i;
        }
        sendcounts[target_rank]++;
    }

    std::vector<int> recvoffsets = std::vector<int>(npes, 0);
    std::vector<int> recvcounts = std::vector<int>(npes, 0);
    HANDLE_MPI_ERROR(
        MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1,
                     MPI_INT, MPI_COMM_WORLD),
        "redistribute_for_fftw: MPI error on MPI_Alltoall:");
    for (int i = 1; i < npes; ++i) {
        recvoffsets[i] = recvoffsets[i - 1] + recvcounts[i - 1];
    }

    HANDLE_MPI_ERROR(
        MPI_Alltoallv(arr->buffers[0]->mutable_data(), sendcounts.data(),
                      sendoffsets.data(), fftw_mpi_type<dtype>,
                      out_arr->buffers[0]->mutable_data(), recvcounts.data(),
                      recvoffsets.data(), fftw_mpi_type<dtype>, MPI_COMM_WORLD),
        "redistribute_for_fftw: MPI error on MPI_Alltoallv:");
    out_arr->length = local_arr_len;
}
/**
 * @brief Performs a parallel FFT2 on the given array, of shape shape.
 * This is templated on the dtype of the array, and will call the appropriate
 * fftw functions for the given dtype.
 * @param arr The array to perform the FFT on
 * @param shape The shape of the array
 * @param parallel Whether arr is distributed or not
 * @return  The result of the FFT
 */
template <Bodo_CTypes::CTypeEnum dtype>
    requires complex_dtype<dtype>
std::shared_ptr<array_info> fft2(std::shared_ptr<array_info> arr,
                                 uint64_t shape[2], bool parallel) {
    tracing::Event ev = tracing::Event("fft2");
    fftw_init_fn<dtype>();

    int npes, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    import_wisdom<dtype>(rank);

    if (parallel) {
        // Get the global shape
        HANDLE_MPI_ERROR(MPI_Allreduce(MPI_IN_PLACE, &shape[0], 1, MPI_UINT64_T,
                                       MPI_SUM, MPI_COMM_WORLD),
                         "fft2: MPI error on MPI_Allreduce:");
    }
    // When processing an array of shape (n, 1) or (1, n) fftw doesn't support
    // the (n, 1) case so we need to swap the shape and then
    // swap it back after the fft
    bool swapped_shape = false;
    if (shape[1] == 1) {
        std::swap(shape[0], shape[1]);
        swapped_shape = true;
    }
    ptrdiff_t local_alloc, local_n0, local_0_start;

    local_alloc = fftw_local_size_2d_fn<dtype>(
        shape[0], shape[1], MPI_COMM_WORLD, &local_n0, &local_0_start);
    // Our 1d distribution does not handle nitems / nranks != 0 the same way as
    // fftw so we always have to redistribute and return 1d_var
    uint64_t local_arr_len = local_n0 * shape[1];
    uint64_t local_arr_start = local_0_start * shape[1];

    std::shared_ptr<array_info> out_arr =
        alloc_array_top_level(arr->length, 0, 0, arr->arr_type, dtype);
    // FFTW might need more memory than the array has for scratch space
    CHECK_ARROW_MEM(out_arr->buffers[0]->Reserve(
                        local_alloc * sizeof(fftw_complex_type<dtype>)),
                    "fft2: failed to allocate memory for output");

    // FFTW hangs in this case
    if (shape[0] * shape[1] < 2) {
        memcpy(out_arr->buffers[0]->mutable_data(),
               arr->buffers[0]->mutable_data(),
               sizeof(fftw_complex_type<dtype>) * arr->length);
        if (parallel) {
            shape[0] = local_n0;
        }
        return out_arr;
    }

    fftw_plan_type<dtype> plan;
    tracing::Event plan_ev = tracing::Event("fft2: plan");
    unsigned int planning_flag = get_fftw_planning_flag();
    set_fftw_timeout();
    if (parallel) {
        plan = fftw_plan_dft_2d_fn<dtype, true>(
            shape[0], shape[1],
            (fftw_complex_type<dtype>*)out_arr->buffers[0]->mutable_data(),
            (fftw_complex_type<dtype>*)out_arr->buffers[0]->mutable_data(),
            MPI_COMM_WORLD, FFTW_FORWARD, planning_flag);
    } else {
        plan = fftw_plan_dft_2d_fn<dtype, false>(
            shape[0], shape[1],
            (fftw_complex_type<dtype>*)out_arr->buffers[0]->mutable_data(),
            (fftw_complex_type<dtype>*)out_arr->buffers[0]->mutable_data(),
            MPI_COMM_WORLD, FFTW_FORWARD, planning_flag);
    }
    plan_ev.finalize();

    if (parallel) {
        redistribute_for_fftw<dtype>(arr, out_arr, rank, npes, shape,
                                     local_arr_start, local_arr_len);
    } else {
        memcpy(out_arr->buffers[0]->mutable_data(),
               arr->buffers[0]->mutable_data(),
               sizeof(fftw_complex_type<dtype>) * arr->length);
    }

    tracing::Event execute_ev = tracing::Event("fft2: execute");
    fftw_execute_fn<dtype>(plan);
    execute_ev.finalize();

    if (swapped_shape) {
        if (parallel) {
            shape[1] = shape[0];
            shape[0] = local_arr_len;
        } else {
            std::swap(shape[0], shape[1]);
        }
    } else if (parallel) {
        shape[0] = local_n0;
    }
    export_wisdom<dtype>(rank);
    return out_arr;
}

/**
 * @brief Calculates the new index of an element after an fftshift
 * @param global_idx index to be shifted, this is 1d idx taking all arrays
 * across all ranks into account
 * @param shape shape of the global array
 * @param shift how much to shift the idx by on each axis
 * @return The shifted idx in global, 1d space
 */
size_t calc_fftshift_new_global_idx(size_t global_idx, uint64_t shape[2],
                                    size_t shift[2]) {
    // Calculate the index for each axis
    size_t global_idx_0 = global_idx % shape[1];
    size_t global_idx_1 = global_idx / shape[1];
    // Shift each axis index
    size_t new_global_idx_0 = (global_idx_0 + shift[1]) % shape[1];
    size_t new_global_idx_1 = (global_idx_1 + shift[0]) % shape[0];
    // Calculate the 1d idx from the axes indices
    size_t new_global_idx = new_global_idx_0 + new_global_idx_1 * shape[1];
    return new_global_idx;
}
/**
 * @brief redistribute data to the appropriate rank for an fftshift
 * If a rank is reencountered for a non-contiguous index we need to
 * stop and distribute what we have so far
 * the calling function will then call this again with an updated
 *send_start_offset
 * @param arr Array to distribute
 * @param shape Global shape of array
 * @param shift How much to shift by in each dimension
 * @param send_start_offset What index to start sending from
 * @param global_idx_offset The offset of the first element in arr in the global
 *array
 * @param global_idx_offset_per_rank The offset of the first element in arr in
 *the global array for each rank, this is updated by this function to account
 *for send offsets
 * @param rank The rank of the current process
 * @param npes The number of processes
 * @return The redistributed array, the recv_offsets, and how many elements were
 *sent by this rank
 **/
template <Bodo_CTypes::CTypeEnum dtype>
    requires complex_dtype<dtype>
std::tuple<std::shared_ptr<array_info>, std::vector<int>, size_t>
fftshift_redistribute_across_ranks(std::shared_ptr<array_info> arr,
                                   uint64_t shape[2], size_t shift[2],
                                   size_t send_start_offset,
                                   size_t global_idx_offset,
                                   std::vector<int>& global_idx_offset_per_rank,
                                   int rank, int npes) {
    std::vector<int> send_counts = std::vector<int>(npes, 0);
    std::vector<int> recv_counts = std::vector<int>(npes);
    std::vector<int> send_offsets = std::vector<int>(npes, -1);
    std::vector<int> recv_offsets = std::vector<int>(npes + 1, 0);
    std::vector<bool> seen_rank = std::vector<bool>(npes, false);
    size_t global_arr_len = shape[0] * shape[1];

    // Calculate how many elements will be distributed to each rank
    // and the offset of the first element for each rank
    // If a rank is reencountered for a non-contiguous index we need to
    // stop and distribute what we have so far
    // the calling function will then call this again with the remaining
    // elements
    size_t prev_rank = 0;
    size_t elements_distributed = arr->length - send_start_offset;
    for (size_t i = send_start_offset; i < arr->length; ++i) {
        size_t global_idx = global_idx_offset + i;
        size_t new_global_idx =
            calc_fftshift_new_global_idx(global_idx, shape, shift);
        size_t new_rank = index_rank(global_arr_len / shape[1], npes,
                                     new_global_idx / shape[1]);
        if (seen_rank[new_rank] && new_rank != prev_rank) {
            elements_distributed = i;
            break;
        } else if (!seen_rank[new_rank]) {
            seen_rank[new_rank] = true;
        }
        prev_rank = new_rank;

        send_counts[new_rank] += 1;
        send_offsets[new_rank] =
            send_offsets[new_rank] == -1 ? i : send_offsets[new_rank];
    }
    //  Do this so leftover -1s don't mess up global_idx_offset_per_rank
    //  below
    std::replace(send_offsets.begin(), send_offsets.end(), -1, 0);
    recv_offsets[npes] = global_arr_len;
    // Distribute send_counts so we can calculate recv_counts/offsets
    HANDLE_MPI_ERROR(
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1,
                     MPI_INT, MPI_COMM_WORLD),
        "fftshift_redistribute_across_ranks: MPI error on MPI_Alltoall:");
    for (int i = 1; i < npes; ++i) {
        recv_offsets[i] = recv_offsets[i - 1] + recv_counts[i - 1];
    }
    uint64_t recv_arr_len =
        std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    // temp_arr is used as the recv buffer for alltoallv and arr is copied into
    // from temp in the shift code
    std::shared_ptr<array_info> temp_arr =
        alloc_array_top_level(recv_arr_len, 0, 0, arr->arr_type, dtype);

    // Each element of global_idx_offset_per_rank needs to be adjusted by
    // the corresponding ranks' send offset for this rank so we can properly
    // calculate each received element's starting global_idx
    std::vector<int> other_ranks_send_offset_for_this_rank =
        std::vector<int>(npes);
    HANDLE_MPI_ERROR(
        MPI_Alltoall(send_offsets.data(), 1, MPI_INT,
                     other_ranks_send_offset_for_this_rank.data(), 1, MPI_INT,
                     MPI_COMM_WORLD),
        "fftshift_redistribute_across_ranks: MPI error on MPI_Alltoall:");
    for (size_t i = 0; i < global_idx_offset_per_rank.size(); ++i) {
        global_idx_offset_per_rank[i] +=
            other_ranks_send_offset_for_this_rank[i];
    }
    // Since send_counts/offsets are different from recv_counts/offsets
    // this can't be done in place
    HANDLE_MPI_ERROR(
        MPI_Alltoallv(arr->buffers[0]->mutable_data(), send_counts.data(),
                      send_offsets.data(), fftw_mpi_type<dtype>,
                      temp_arr->buffers[0]->mutable_data(), recv_counts.data(),
                      recv_offsets.data(), fftw_mpi_type<dtype>,
                      MPI_COMM_WORLD),
        "fftshift_redistribute_across_ranks: MPI error on MPI_Alltoallv:");
    return {temp_arr, recv_offsets, elements_distributed};
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex_dtype<dtype>
void fftshift(std::shared_ptr<array_info> arr, uint64_t shape[2],
              const bool parallel) {
    tracing::Event ev = tracing::Event("fftshift");
    int npes, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // These are needed in the shifting code below but are only updated if
    // parallel
    std::vector<uint64_t> arr_lens;
    size_t global_idx_offset = 0;
    size_t global_arr_len = arr->length;
    std::vector<int> global_idx_offset_per_rank = std::vector<int>(npes, 0);

    if (parallel) {
        // Distribute each ranks length to every other rank
        arr_lens = std::vector<uint64_t>(npes, 0);
        HANDLE_MPI_ERROR(
            MPI_Allgather(&arr->length, 1, MPI_UINT64_T, arr_lens.data(), 1,
                          MPI_UINT64_T, MPI_COMM_WORLD),
            "fftshift: MPI error on MPI_Allgather:");
        // Calculate the size of arr across all ranks
        global_arr_len =
            std::accumulate<std::vector<uint64_t>::iterator, size_t>(
                arr_lens.begin(), arr_lens.end(), 0);
        // Calculate each ranks offset mapping from local index to global index
        std::partial_sum(arr_lens.begin(), arr_lens.end() - 1,
                         global_idx_offset_per_rank.begin() + 1);
        // Get this rank's global index offset
        global_idx_offset = global_idx_offset_per_rank[rank];
    }

    size_t shift[2];
    // This captures all received arrays and the data necessary
    // to calculate their original global idx
    std::vector<std::tuple<std::shared_ptr<array_info>, std::vector<int>,
                           std::vector<int>>>
        recv_data;
    if (parallel) {
        // Make shape global
        shape[0] = global_arr_len / shape[1];
        // Calculate the shift
        shift[0] = shape[0] / 2;
        shift[1] = shape[1] / 2;

        std::vector<int> global_idx_offset_per_rank_updated =
            global_idx_offset_per_rank;
        auto [recv_arr, recv_offsets, elements_distributed] =
            fftshift_redistribute_across_ranks<dtype>(
                arr, shape, shift, 0, global_idx_offset,
                global_idx_offset_per_rank_updated, rank, npes);

        // Global idx offset needs to be updated based on the updated data
        // distribution
        auto recv_arr_lens = std::vector<uint64_t>(npes);
        HANDLE_MPI_ERROR(MPI_Allgather(&recv_arr->length, 1, MPI_UINT64_T,
                                       recv_arr_lens.data(), 1, MPI_UINT64_T,
                                       MPI_COMM_WORLD),
                         "fftshift: MPI error on MPI_Allgather:");
        global_idx_offset = std::accumulate(recv_arr_lens.begin(),
                                            recv_arr_lens.begin() + rank, 0);
        // Capture all the related data from this exchange
        recv_data.push_back(
            {recv_arr, global_idx_offset_per_rank_updated, recv_offsets});

        // Figure out how many elements where distributed by all ranks
        // if all elements weren't distributed, distribute again
        size_t global_elements_distributed =
            std::accumulate(recv_arr_lens.begin(), recv_arr_lens.end(), 0);
        if (global_elements_distributed != global_arr_len) {
            // These will be updated by fftshift_redistribute_across_ranks
            std::vector<int> global_idx_offset_per_rank_updated_secondary =
                global_idx_offset_per_rank;

            auto [recv_arr_secondary, recv_offsets_secondary,
                  elements_distributed_secondary] =
                fftshift_redistribute_across_ranks<dtype>(
                    arr, shape, shift, elements_distributed, global_idx_offset,
                    global_idx_offset_per_rank_updated_secondary, rank, npes);
            // Global idx offset needs to be updated based on the updated data
            // distribution
            std::vector<uint64_t> recv_arr_lens_secondary =
                std::vector<uint64_t>(npes);
            HANDLE_MPI_ERROR(
                MPI_Allgather(&recv_arr_secondary->length, 1, MPI_UINT64_T,
                              recv_arr_lens_secondary.data(), 1, MPI_UINT64_T,
                              MPI_COMM_WORLD),
                "fftshift: MPI error on MPI_Allgather:");
            for (size_t i = 0; i < recv_arr_lens_secondary.size(); ++i) {
                recv_arr_lens_secondary[i] += recv_arr_lens[i];
            }
            global_idx_offset =
                std::accumulate(recv_arr_lens_secondary.begin(),
                                recv_arr_lens_secondary.begin() + rank, 0);
            // Capture all related data from this exchange
            recv_data.push_back({recv_arr_secondary,
                                 global_idx_offset_per_rank_updated_secondary,
                                 recv_offsets_secondary});
        }
        // Ensure arr has enough capacity to hold all received rows, it's copied
        // into below
        size_t total_recv_len = 0;
        for (auto [recv_arr, _1, _2] : recv_data) {
            total_recv_len += recv_arr->length;
        }
        CHECK_ARROW_MEM(arr->buffers[0]->Reserve(
                            total_recv_len * sizeof(fftw_complex_type<dtype>)),
                        "fftshift: failed to allocate memory for output");
    } else {
        std::shared_ptr<array_info> recv_arr =
            alloc_array_top_level(arr->length, 0, 0, arr->arr_type, dtype);
        shift[0] = shape[0] / 2;
        shift[1] = shape[1] / 2;
        // There might be a way to do the below shifting in place that would
        // remove the temp_buffer and this memcpy but I don't think it would
        // work in the parallel case because of the source_rank update
        memcpy(recv_arr->buffers[0]->mutable_data(),
               arr->buffers[0]->mutable_data(),
               sizeof(fftw_complex_type<dtype>) * arr->length);
        recv_data.push_back({recv_arr, global_idx_offset_per_rank,
                             std::vector<int>(npes + 1, 0)});
    }
    // This assumes each rank will always send one contiguous chunk to each
    // other rank Justification: each row is contiguous in memory and on the
    // same rank so we can shift each row safely. Once each row is on the right
    // rank we shift within the row to the final location. There is trouble if
    // one rank has too much data because then we are splitting rows
    // rows so our assumption is broken. In particular global_arr_len -
    // (global_arr_len / npes) rows because this means the rank the data at the
    // start is sent to is wrapping around to the end of the array so the data
    // isn't contiguous. This is why we need to redistribute if a new rank is
    // reencountered for a non-contiguous index.
    arr->length = 0;
    size_t source_rank = 0;
    for (auto [recv_arr, global_idx_offset_per_rank, recv_offsets] :
         recv_data) {
        for (size_t i = 0; i < recv_arr->length; ++i) {
            while (parallel && i >= (size_t)recv_offsets[source_rank + 1]) {
                source_rank++;
            }
            // Caclulate the element's original global idx
            size_t global_idx = global_idx_offset_per_rank[source_rank] + i -
                                recv_offsets[source_rank];
            size_t new_global_idx =
                calc_fftshift_new_global_idx(global_idx, shape, shift);
            size_t new_local_idx = new_global_idx - global_idx_offset;
            ((complex_type<dtype>*)arr->buffers[0]
                 ->mutable_data())[new_local_idx] =
                ((complex_type<dtype>*)recv_arr->buffers[0]->mutable_data())[i];
        }
        source_rank = 0;
        arr->length += recv_arr->length;
    };
    if (parallel) {
        // Update shape based on the new size of arr
        shape[0] = arr->length / shape[1];
    }
}

/**
 * @brief Performs a parallel FFT2 on the given array, of shape shape.
 * @param arr The array to perform the FFT on
 * @param shape The shape of the array
 * @param parallel Whether arr is distributed or not
 * @return  The result of the FFT
 */
array_info* fft2_py_entry(array_info* arr, uint64_t shape[2], bool parallel) {
    try {
        std::shared_ptr<array_info> arr_wrapped =
            std::shared_ptr<array_info>(arr);
        std::shared_ptr<array_info> out_arr;
        // FFTW has different functions and types for different precisions
        // so we need to wrap them in a template
        switch (arr->dtype) {
            case Bodo_CTypes::COMPLEX128: {
                out_arr =
                    fft2<Bodo_CTypes::COMPLEX128>(arr_wrapped, shape, parallel);
                break;
            }
            case Bodo_CTypes::COMPLEX64: {
                out_arr =
                    fft2<Bodo_CTypes::COMPLEX64>(arr_wrapped, shape, parallel);
                break;
            }
            default: {
                throw std::runtime_error("fft2: Unsupported dtype: " +
                                         GetDtype_as_string(arr->dtype));
            }
        }
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Performs fftshift on the given 2d array, shift the zero-frequency
 * component to the center of the spectrum.
 * @param arr The array to perform the fftshift on
 * @param shape The shape of the array, if arr's shape changes this is updated
 * @param parallel Whether arr is distributed
 * @return The result of the fftshift
 */
array_info* fftshift_py_entry(array_info* arr, uint64_t shape[2],
                              bool parallel) {
    try {
        std::shared_ptr<array_info> arr_wrapped =
            std::shared_ptr<array_info>(arr);
        switch (arr->dtype) {
            case Bodo_CTypes::COMPLEX128: {
                fftshift<Bodo_CTypes::COMPLEX128>(arr_wrapped, shape, parallel);
                break;
            }
            case Bodo_CTypes::COMPLEX64: {
                fftshift<Bodo_CTypes::COMPLEX64>(arr_wrapped, shape, parallel);
                break;
            }
            default: {
                throw std::runtime_error("fft2: Unsupported dtype: " +
                                         GetDtype_as_string(arr->dtype));
            }
        }
        return new array_info(*arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyMODINIT_FUNC PyInit_fft_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "fft_cpp", "No docs", NULL);
    if (m == NULL)
        return NULL;

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, fft2_py_entry);
    SetAttrStringFromVoidPtr(m, fftshift_py_entry);
    return m;
}
