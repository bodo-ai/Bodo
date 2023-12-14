#include "_fft.h"
#include <mpi.h>
#include <iostream>
#include "_array_utils.h"
#include "_bodo_common.h"
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
    requires(dtype == Bodo_CTypes::COMPLEX128 ||
             dtype == Bodo_CTypes::COMPLEX64)
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
        return new array_info(*arr_wrapped.get());

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
    return m;
}
