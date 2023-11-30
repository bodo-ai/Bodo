#include "_utils.h"

#include <mpi.h>

std::tuple<int, int> dist_get_ranks_on_node() {
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) {
        MPI_Init(NULL, NULL);
    }

    int npes_node;
    int rank_on_node;
    MPI_Comm shmcomm;

    // Split comm into comms that has same shared memory.
    // This is a collective operation and all ranks must call it.
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shmcomm);
    // Get number of ranks on this sub-communicator (i.e. node).
    // By definition, all ranks on the same node will get the same
    // output.
    MPI_Comm_size(shmcomm, &npes_node);
    MPI_Comm_rank(shmcomm, &rank_on_node);

    MPI_Comm_free(&shmcomm);
    return std::make_tuple(npes_node, rank_on_node);
}
