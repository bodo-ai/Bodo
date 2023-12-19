#include "_fft.h"
#include <mpi.h>
#include <numeric>
#include "_array_utils.h"
#include "_distributed.h"
#include "tracing.h"

/**
 * @brief Redistributes the given array to match fftw's expected distribution
 * @param arr The array to redistribute
 * @param comp_buffer The buffer to redistribute into
 * @param rank The rank of the current process
 * @param npes The number of processes
 * @param shape The shape of the array
 * @param local_arr_start The start of the local array as calculated by fftw
 * @param local_arr_len The length of the local array as calculated by fftw
 */
template <Bodo_CTypes::CTypeEnum dtype>
    requires complex_dtype<dtype>
void redistribute_for_fftw(std::shared_ptr<array_info> arr, void* comp_buffer,
                           int rank, int npes, uint64_t shape[2],
                           ptrdiff_t local_arr_start, ptrdiff_t local_arr_len) {
    tracing::Event ev = tracing::Event("redistribute_for_fftw");
    int arr_len = arr->length;
    std::vector<int> arr_lens = std::vector<int>(npes, 0);
    std::vector<int> local_arr_starts = std::vector<int>(npes + 1, 0);
    local_arr_starts[npes] = shape[0] * shape[1];
    MPI_Allgather(&arr_len, 1, MPI_INT, arr_lens.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);
    MPI_Allgather(&local_arr_start, 1, MPI_INT, local_arr_starts.data(), 1,
                  MPI_INT, MPI_COMM_WORLD);
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
        while (global_idx >= local_arr_starts[target_rank + 1]) {
            target_rank++;
            sendoffsets[target_rank] = i;
        }
        sendcounts[target_rank]++;
    }

    std::vector<int> recvoffsets = std::vector<int>(npes, 0);
    std::vector<int> recvcounts = std::vector<int>(npes, 0);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);
    for (int i = 1; i < npes; ++i) {
        recvoffsets[i] = recvoffsets[i - 1] + recvcounts[i - 1];
    }

    MPI_Alltoallv(arr->buffers[0]->mutable_data(), sendcounts.data(),
                  sendoffsets.data(), fftw_mpi_type<dtype>, comp_buffer,
                  recvcounts.data(), recvoffsets.data(), fftw_mpi_type<dtype>,
                  MPI_COMM_WORLD);
    arr->length = local_arr_len;
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
void fft2(std::shared_ptr<array_info> arr, uint64_t shape[2], bool parallel) {
    tracing::Event ev = tracing::Event("fft2");
    fftw_init_fn<dtype>();

    int npes, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (parallel) {
        // Get the global shape
        MPI_Allreduce(MPI_IN_PLACE, &shape[0], 1, MPI_UINT64_T, MPI_SUM,
                      MPI_COMM_WORLD);
    }
    ptrdiff_t local_alloc, local_n0, local_0_start;

    // FFTW might need more memory than the array has for scratch space
    local_alloc = fftw_local_size_2d_fn<dtype>(
        shape[0], shape[1], MPI_COMM_WORLD, &local_n0, &local_0_start);
    // Our 1d distribution does not handle nitems / nranks != 0 the same way as
    // fftw so we always have to redistribute and return 1d_var
    uint64_t local_arr_len = local_n0 * shape[1];
    uint64_t local_arr_start = local_0_start * shape[1];
    CHECK_ARROW_MEM(arr->buffers[0]->Reserve(local_arr_len *
                                             sizeof(fftw_complex_type<dtype>)),
                    "fft2: Failed to allocate memory for fftw");
    fftw_complex_type<dtype>* comp_buffer = fftw_alloc_complex_fn<dtype>(
        std::max<uint64_t>(local_alloc, arr->length));

    fftw_plan_type<dtype> plan;
    tracing::Event plan_ev = tracing::Event("fft2: plan");
    if (parallel) {
        plan = fftw_plan_dft_2d_fn<dtype, true>(
            shape[0], shape[1], comp_buffer,
            ((fftw_complex_type<dtype>*)arr->buffers[0]->mutable_data()),
            MPI_COMM_WORLD, FFTW_FORWARD,
            /*Unfortunately we have to use FFTW_ESTIMATE otherwise it will
             * overwrite the array, the other option is to allocate, plan and
             * then copy*/
            FFTW_ESTIMATE);
    } else {
        plan = fftw_plan_dft_2d_fn<dtype, false>(
            shape[0], shape[1],
            ((fftw_complex_type<dtype>*)arr->buffers[0]->mutable_data()),
            ((fftw_complex_type<dtype>*)arr->buffers[0]->mutable_data()),
            MPI_COMM_WORLD, FFTW_FORWARD,
            /*Unfortunately we have to use FFTW_ESTIMATE otherwise it will
             * overwrite the array, the other option is to allocate, plan and
             * then copy*/
            FFTW_ESTIMATE);
    }
    plan_ev.finalize();

    if (parallel) {
        redistribute_for_fftw<dtype>(arr, comp_buffer, rank, npes, shape,
                                     local_arr_start, local_arr_len);
    }
    tracing::Event execute_ev = tracing::Event("fft2: execute");
    fftw_execute_fn<dtype>(plan);
    execute_ev.finalize();

    if (parallel) {
        shape[0] = local_n0;
    }
    fftw_free_fn<dtype>(comp_buffer);
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
        MPI_Allgather(&arr->length, 1, MPI_UINT64_T, arr_lens.data(), 1,
                      MPI_UINT64_T, MPI_COMM_WORLD);
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
    // This is needed in the shifting code below but is only updated if parallel
    std::vector<int> recv_offsets = std::vector<int>(npes + 1, 0);
    // Allocate the send buffer for alltoallv since we can't use in place
    std::shared_ptr<array_info> temp_arr =
        alloc_array_top_level(arr->length, 0, 0, arr->arr_type, dtype);
    // This is later updated if parallel
    uint64_t recv_arr_len = arr->length;

    if (parallel) {
        std::vector<int> send_counts = std::vector<int>(npes, 0);
        std::vector<int> recv_counts = std::vector<int>(npes);
        std::vector<int> send_offsets = std::vector<int>(npes, -1);
        //  Make shape global
        shape[0] = global_arr_len / shape[1];
        // Calculate the shift
        shift[0] = shape[0] / 2;
        shift[1] = shape[1] / 2;
        for (size_t i = 0; i < arr->length; ++i) {
            size_t global_idx = global_idx_offset + i;
            size_t new_global_idx =
                calc_fftshift_new_global_idx(global_idx, shape, shift);
            size_t new_rank = index_rank(global_arr_len / shape[1], npes,
                                         new_global_idx / shape[1]);
            send_counts[new_rank] += 1;
            send_offsets[new_rank] =
                send_offsets[new_rank] == -1 ? i : send_offsets[new_rank];
        }
        //  Do this so leftover -1s don't mess up global_idx_offset_per_rank
        //  below
        std::replace(send_offsets.begin(), send_offsets.end(), -1, 0);
        recv_offsets[npes] = global_arr_len;
        // Distribute send_counts so we can calculate recv_counts/offsets
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1,
                     MPI_INT, MPI_COMM_WORLD);
        for (int i = 1; i < npes; ++i) {
            recv_offsets[i] = recv_offsets[i - 1] + recv_counts[i - 1];
        }
        recv_arr_len =
            std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
        // Both of these will hold of this rank's data, temp_arr is used as
        // the recv buffer for alltoallv and arr is copied into from temp
        // in the shift code
        CHECK_ARROW_MEM(arr->buffers[0]->Reserve(
                            recv_arr_len * sizeof(fftw_complex_type<dtype>)),
                        "fftshift: failed to allocate memory for output");
        CHECK_ARROW_MEM(temp_arr->buffers[0]->Reserve(
                            recv_arr_len * sizeof(fftw_complex_type<dtype>)),
                        "fftshift: failed to allocate memory for temp buffer");
        // Each element of global_idx_offset_per_rank needs to be adjusted by
        // the corresponding ranks' send offset for this rank so we can properly
        // calculate each received element's starting global_idx
        std::vector<int> other_ranks_send_offset_for_this_rank =
            std::vector<int>(npes);
        MPI_Alltoall(send_offsets.data(), 1, MPI_INT,
                     other_ranks_send_offset_for_this_rank.data(), 1, MPI_INT,
                     MPI_COMM_WORLD);
        for (size_t i = 0; i < global_idx_offset_per_rank.size(); ++i) {
            global_idx_offset_per_rank[i] +=
                other_ranks_send_offset_for_this_rank[i];
        }
        // Global idx offset needs to be updated based on the updated data
        // distribution
        auto recv_arr_lens = std::vector<uint64_t>(npes);
        MPI_Allgather(&recv_arr_len, 1, MPI_UINT64_T, recv_arr_lens.data(), 1,
                      MPI_UINT64_T, MPI_COMM_WORLD);
        global_idx_offset = std::accumulate(recv_arr_lens.begin(),
                                            recv_arr_lens.begin() + rank, 0);
        // Since send_counts/offsets are different from recv_counts/offsets
        // this can't be done in place
        MPI_Alltoallv(arr->buffers[0]->mutable_data(), send_counts.data(),
                      send_offsets.data(), fftw_mpi_type<dtype>,
                      temp_arr->buffers[0]->mutable_data(), recv_counts.data(),
                      recv_offsets.data(), fftw_mpi_type<dtype>,
                      MPI_COMM_WORLD);
    } else {
        shift[0] = shape[0] / 2;
        shift[1] = shape[1] / 2;
        // There might be a way to do the below shifting in place that would
        // remove the temp_buffer and this memcpy but I don't think it would
        // work in the parallel case because of the source_rank update
        memcpy(temp_arr->buffers[0]->mutable_data(),
               arr->buffers[0]->mutable_data(),
               sizeof(fftw_complex_type<dtype>) * arr->length);
    }
    // This assumes each rank will always send one contiguous chunk to each
    // other rank Justification: each 1d index is shifted by a constant shift
    // which is shift[0] + shift[1] * shape[0]. This means that idx_old + shift
    // = idx_new and idx_old + 1 + shift = idx_new + 1 The only time this isn't
    // true is when idx_old + shift > global_arr_len In this case the new_index
    // will always go to rank n and new_index + 1 will go to rank 0 so it
    // doesn't matter if they're not contiguous
    size_t source_rank = 0;
    for (size_t i = 0; i < recv_arr_len; ++i) {
        while (parallel && i >= (size_t)recv_offsets[source_rank + 1]) {
            source_rank++;
        }
        // Caclulate the element's original global idx
        size_t global_idx = global_idx_offset_per_rank[source_rank] + i -
                            recv_offsets[source_rank];
        size_t new_global_idx =
            calc_fftshift_new_global_idx(global_idx, shape, shift);
        size_t new_local_idx = new_global_idx - global_idx_offset;
        ((complex_type<dtype>*)arr->buffers[0]->mutable_data())[new_local_idx] =
            ((complex_type<dtype>*)temp_arr->buffers[0]->mutable_data())[i];
    }
    arr->length = recv_arr_len;
    if (parallel) {
        // Update shape based on the new size of arr
        shape[0] = recv_arr_len / shape[1];
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
        // FFTW has different functions and types for different precisions
        // so we need to wrap them in a template
        switch (arr->dtype) {
            case Bodo_CTypes::COMPLEX128: {
                fft2<Bodo_CTypes::COMPLEX128>(arr_wrapped, shape, parallel);
                break;
            }
            case Bodo_CTypes::COMPLEX64: {
                fft2<Bodo_CTypes::COMPLEX64>(arr_wrapped, shape, parallel);
                break;
            }
            default: {
                throw std::runtime_error("fft2: Unsupported dtype: " +
                                         GetDtype_as_string(arr->dtype));
            }
        }
        return new array_info(*arr_wrapped);

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
