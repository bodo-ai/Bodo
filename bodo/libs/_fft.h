#include <fftw3-mpi.h>
#include <fftw3.h>
#include <mpi.h>
#include "_array_utils.h"
#include "_bodo_common.h"

// All of the below code is to allow for templating on the dtype of the array.
// FFTW has two different sets of functions and types, one for single precision
// and one for double precision.
template <Bodo_CTypes::CTypeEnum dtype>
    requires complex_dtype<dtype>
using fftw_complex_type =
    typename std::conditional<dtype == Bodo_CTypes::COMPLEX128, fftw_complex,
                              fftwf_complex>::type;

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex_dtype<dtype>
using fftw_plan_type =
    typename std::conditional<dtype == Bodo_CTypes::COMPLEX128, fftw_plan,
                              fftwf_plan>::type;

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex_dtype<dtype>
auto fftw_mpi_type = dtype == Bodo_CTypes::COMPLEX128 ? MPI_C_DOUBLE_COMPLEX
                                                      : MPI_C_FLOAT_COMPLEX;

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex128_dtype<dtype>
ptrdiff_t fftw_local_size_2d_fn(ptrdiff_t n0, ptrdiff_t n1, MPI_Comm comm,
                                ptrdiff_t *local_n0, ptrdiff_t *local_0_start) {
    return fftw_mpi_local_size_2d(n0, n1, comm, local_n0, local_0_start);
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires(complex64_dtype<dtype>)
ptrdiff_t fftw_local_size_2d_fn(ptrdiff_t n0, ptrdiff_t n1, MPI_Comm comm,
                                ptrdiff_t *local_n0, ptrdiff_t *local_0_start) {
    return fftwf_mpi_local_size_2d(n0, n1, comm, local_n0, local_0_start);
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex128_dtype<dtype>
void fftw_free_fn(void *p) {
    fftw_free(p);
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires(complex64_dtype<dtype>)
void fftw_free_fn(void *p) {
    fftwf_free(p);
}

template <Bodo_CTypes::CTypeEnum dtype, bool parallel>
    requires(complex128_dtype<dtype> && parallel)
fftw_plan fftw_plan_dft_2d_fn(ptrdiff_t n0, ptrdiff_t n1, fftw_complex *in,
                              fftw_complex *out, MPI_Comm comm, int sign,
                              unsigned int flags) {
    return fftw_mpi_plan_dft_2d(n0, n1, in, out, comm, sign, flags);
}

template <Bodo_CTypes::CTypeEnum dtype, bool parallel>
    requires(complex64_dtype<dtype> && parallel)
fftwf_plan fftw_plan_dft_2d_fn(ptrdiff_t n0, ptrdiff_t n1, fftwf_complex *in,
                               fftwf_complex *out, MPI_Comm comm, int sign,
                               unsigned int flags) {
    return fftwf_mpi_plan_dft_2d(n0, n1, in, out, comm, sign, flags);
}

template <Bodo_CTypes::CTypeEnum dtype, bool parallel>
    requires(complex128_dtype<dtype> && !parallel)
fftw_plan fftw_plan_dft_2d_fn(ptrdiff_t n0, ptrdiff_t n1, fftw_complex *in,
                              fftw_complex *out, MPI_Comm comm, int sign,
                              unsigned int flags) {
    return fftw_plan_dft_2d(n0, n1, in, out, sign, flags);
}

template <Bodo_CTypes::CTypeEnum dtype, bool parallel>
    requires(complex64_dtype<dtype> && !parallel)
fftwf_plan fftw_plan_dft_2d_fn(ptrdiff_t n0, ptrdiff_t n1, fftwf_complex *in,
                               fftwf_complex *out, MPI_Comm comm, int sign,
                               unsigned int flags) {
    return fftwf_plan_dft_2d(n0, n1, in, out, sign, flags);
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex128_dtype<dtype>
void fftw_execute_fn(fftw_plan plan) {
    fftw_execute(plan);
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires(complex64_dtype<dtype>)
void fftw_execute_fn(fftwf_plan plan) {
    fftwf_execute(plan);
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex128_dtype<dtype>
void fftw_init_fn() {
    fftw_mpi_init();
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires(complex64_dtype<dtype>)
void fftw_init_fn() {
    fftwf_mpi_init();
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex128_dtype<dtype>
void fftw_import_wisdom_from_filename_fn() {
    fftw_import_wisdom_from_filename(".fftw_wisdom");
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires(complex64_dtype<dtype>)
void fftw_import_wisdom_from_filename_fn() {
    fftwf_import_wisdom_from_filename(".fftwf_wisdom");
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex128_dtype<dtype>
void fftw_export_wisdom_to_filename_fn() {
    fftw_export_wisdom_to_filename(".fftw_wisdom");
}
template <Bodo_CTypes::CTypeEnum dtype>
    requires(complex64_dtype<dtype>)
void fftw_export_wisdom_to_filename_fn() {
    fftwf_export_wisdom_to_filename(".fftwf_wisdom");
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex128_dtype<dtype>
void fftw_mpi_broadcast_wisdom_fn(MPI_Comm comm) {
    fftw_mpi_broadcast_wisdom(comm);
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires(complex64_dtype<dtype>)
void fftw_mpi_broadcast_wisdom_fn(MPI_Comm comm) {
    fftwf_mpi_broadcast_wisdom(comm);
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires complex128_dtype<dtype>
void fftw_mpi_gather_wisdom_fn(MPI_Comm comm) {
    fftw_mpi_gather_wisdom(comm);
}

template <Bodo_CTypes::CTypeEnum dtype>
    requires(complex64_dtype<dtype>)
void fftw_mpi_gather_wisdom_fn(MPI_Comm comm) {
    fftwf_mpi_gather_wisdom(comm);
}
