#pragma once

#ifdef USE_CUDF
#include <mpi.h>
#include <nccl.h>
#include <cudf/table/table.hpp>

/**
 * @brief Get the GPU device ID for the current process. All ranks must call
 * this function.
 * @return rmm::cuda_device_id, -1 if no GPU is assigned to this rank
 */
rmm::cuda_device_id get_gpu_id();

/**
 * @brief Get the MPI communicator for ranks with GPUs assigned
 * @return MPI_Comm
 */
MPI_Comm get_gpu_mpi_comm();

#endif
