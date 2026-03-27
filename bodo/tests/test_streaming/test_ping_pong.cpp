#include "../libs/gpu_utils.h"
#include "../test.hpp"
#include "cuda_runtime_api.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <iostream>
#include <numeric>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <vector>

// Should use zero copy
#define N (1 << 24)  // ~64MB

static bodo::tests::suite tests([] {
    bodo::tests::test("test_mpi_cuda_ping_pong", [] {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        rmm::cuda_device_id device_id = get_gpu_id();
        if (device_id.value() >= 0) {
            std::cout << "Rank " << rank << " using GPU " << device_id.value()
                      << std::endl;
            cudaSetDevice(device_id.value());
        }
        if (rank != 0 && rank != 1) {
            return;
        }
        cudaFree(nullptr);

        // Host buffer
        std::vector<int> h_buf(N);

        // Device buffer
        int* d_buf;
        cudaMalloc(&d_buf, N * sizeof(int));

        if (rank == 0) {
            // Fill on CPU
            for (int i = 0; i < N; i++) {
                h_buf[i] = i % 100;
            }

            // Copy to GPU
            cudaMemcpy(d_buf, h_buf.data(), N * sizeof(int),
                       cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            std::cout << "Rank " << rank << " sending GPU buffer..."
                      << std::endl;

            MPI_Send(d_buf, N, MPI_INT, RECV_RANK, 0, MPI_COMM_WORLD);
        }

        if (rank == 1) {
            std::cout << "Rank " << rank << " receiving GPU buffer..."
                      << std::endl;

            cudaDeviceSynchronize();
            MPI_Recv(d_buf, N, MPI_INT, SEND_RANK, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

            // Copy back to host
            cudaMemcpy(h_buf.data(), d_buf, N * sizeof(int),
                       cudaMemcpyDeviceToHost);

            // CPU checksum
            long long sum = std::accumulate(h_buf.begin(), h_buf.end(), 0LL);

            std::cout << "Checksum: " << sum << std::endl;
        }

        cudaFree(d_buf);
    });
});
