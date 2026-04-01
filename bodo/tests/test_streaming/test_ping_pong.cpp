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

#define N (1 << 20)  // ~64MB

static bodo::tests::suite tests([] {
    bodo::tests::test("test_mpi_cuda_ping_pong", [] {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        cudaSetDevice(0);
        if (rank != 0 && rank != 1) {
            return;
        }
        cudaFree(nullptr);

        std::vector<int> h_buf(N);

        int* d_buf;
        cudaMalloc(&d_buf, N * sizeof(int));
        cudaMemset(d_buf, 0, N * sizeof(int));

        if (rank == 0) {
            for (int i = 0; i < N; i++) {
                h_buf[i] = i % 100;
            }

            cudaMemcpy(d_buf, h_buf.data(), N * sizeof(int),
                       cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            std::cout << "Rank " << rank << " sending GPU buffer..."
                      << std::endl;

            CHECK_MPI(MPI_Send(d_buf, N, MPI_INT, 1, 0, MPI_COMM_WORLD),
                      "MPI_Send failed");
        }

        if (rank == 1) {
            std::cout << "Rank " << rank << " receiving GPU buffer..."
                      << std::endl;

            cudaDeviceSynchronize();
            CHECK_MPI(MPI_Recv(d_buf, N, MPI_INT, 0, 0, MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE),
                      "MPI_Recv failed");

            cudaMemcpy(h_buf.data(), d_buf, N * sizeof(int),
                       cudaMemcpyDeviceToHost);

            long long sum = std::accumulate(h_buf.begin(), h_buf.end(), 0LL);

            std::cout << "Checksum: " << sum << std::endl;
        }

        cudaFree(d_buf);
    });
});
