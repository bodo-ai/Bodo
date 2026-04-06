#include "_utils.h"

#ifdef __linux__
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <arrow/util/windows_compatibility.h>

#include "_distributed.h"
#include "_mpi.h"

int get_node_id() {
    static int cached_node_id;
    static bool cache_initialized = false;

    if (cache_initialized) {
        return cached_node_id;
    }

    int rank, n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int len;
    CHECK_MPI(MPI_Get_processor_name(hostname, &len),
              "dist_get_ranks_on_node: MPI error on MPI_Get_processor_name:");

    std::vector<char> all_hostnames;
    if (rank == 0) {
        all_hostnames.resize(n_pes * MPI_MAX_PROCESSOR_NAME);
    }

    CHECK_MPI(MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         all_hostnames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         0, MPI_COMM_WORLD),
              "dist_get_ranks_on_node: MPI error on MPI_Gather:");

    int node_id;
    if (rank == 0) {
        std::map<std::string, int> node_map;
        int next_id = 0;

        std::vector<int> node_ids(n_pes);

        for (int i = 0; i < n_pes; ++i) {
            std::string h(&all_hostnames[i * MPI_MAX_PROCESSOR_NAME]);

            if (node_map.count(h) == 0) {
                node_map[h] = next_id++;
            }
            node_ids[i] = node_map[h];
        }

        CHECK_MPI(MPI_Scatter(node_ids.data(), 1, MPI_INT, &node_id, 1, MPI_INT,
                              0, MPI_COMM_WORLD),
                  "dist_get_ranks_on_node: MPI error on MPI_Scatter:");
    } else {
        CHECK_MPI(MPI_Scatter(nullptr, 1, MPI_INT, &node_id, 1, MPI_INT, 0,
                              MPI_COMM_WORLD),
                  "dist_get_ranks_on_node: MPI error on MPI_Scatter:");
    }

    cached_node_id = node_id;
    cache_initialized = true;
    return cached_node_id;
}

#ifdef USE_CUDF

/**
 * @brief Get the number of ranks on this node and the current rank's position.
 *
 * Use MPI_Comm_split to create a communicator for each node based on node_id
 * (node id is a unique id based on output of MPI_Get_processor_name). This
 * works around issues with MPI_Comm_split_type for CUDA-Aware MPICH.
 * TODO: use hwloc approach for all platforms and make sure it's packaged
 * properly.
 */
std::tuple<int, int> dist_get_ranks_on_node() {
    static std::tuple<int, int> cached_result;
    static bool cache_initialized = false;

    if (cache_initialized) {
        return cached_result;
    }

    int node_id = get_node_id();
    MPI_Comm node_comm;
    CHECK_MPI(MPI_Comm_split(MPI_COMM_WORLD, node_id, 0, &node_comm),
              "dist_get_ranks_on_node: MPI error on MPI_Comm_split:");

    int npes_node, rank_on_node;
    MPI_Comm_size(node_comm, &npes_node);
    MPI_Comm_rank(node_comm, &rank_on_node);

    MPI_Comm_free(&node_comm);

    cached_result = std::make_tuple(npes_node, rank_on_node);
    cache_initialized = true;

    return cached_result;
}

#else

std::tuple<int, int> dist_get_ranks_on_node() {
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) {
        CHECK_MPI(MPI_Init(nullptr, nullptr),
                  "dist_get_ranks_on_node: MPI error on MPI_Init:");
    }

    int npes_node;
    int rank_on_node;
    MPI_Comm shmcomm;

    // Split comm into comms that has same shared memory.
    // This is a collective operation and all ranks must call it.
    CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &shmcomm),
              "dist_get_ranks_on_node: MPI error on MPI_Comm_split_type:");
    // Get number of ranks on this sub-communicator (i.e. node).
    // By definition, all ranks on the same node will get the same
    // output.
    MPI_Comm_size(shmcomm, &npes_node);
    MPI_Comm_rank(shmcomm, &rank_on_node);

    MPI_Comm_free(&shmcomm);
    return std::make_tuple(npes_node, rank_on_node);
}

#endif

std::string BytesToHumanReadableString(const size_t bytes) {
    auto kibibytes = bytes / 1024;
    auto mebibyte = kibibytes / 1024;
    kibibytes -= mebibyte * 1024;
    auto gibibyte = mebibyte / 1024;
    mebibyte -= gibibyte * 1024;
    auto tebibyte = gibibyte / 1024;
    gibibyte -= tebibyte * 1024;
    auto pebibyte = tebibyte / 1024;
    tebibyte -= pebibyte * 1024;
    if (pebibyte > 0) {
        return std::to_string(pebibyte) + "." +
               std::to_string((tebibyte * 100) / 1024) + "PiB";
    }
    if (tebibyte > 0) {
        return std::to_string(tebibyte) + "." +
               std::to_string((gibibyte * 100) / 1024) + "TiB";
    } else if (gibibyte > 0) {
        return std::to_string(gibibyte) + "." +
               std::to_string((mebibyte * 100) / 1024) + "GiB";
    } else if (mebibyte > 0) {
        return std::to_string(mebibyte) + "." +
               std::to_string((kibibytes * 100) / 1024) + "MiB";
    } else if (kibibytes > 0) {
        return std::to_string(kibibytes) + "KiB";
    } else {
        return std::to_string(bytes) + (bytes == 1 ? " byte" : " bytes");
    }
}

std::optional<std::chrono::steady_clock::time_point> start_now(bool get) {
    return get ? std::optional<std::chrono::steady_clock::time_point>(
                     std::chrono::steady_clock::now())
               : std::nullopt;
}

uint64_t get_physically_installed_memory() {
// Handle Linux
#ifdef __linux__
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages == -1 || page_size == -1) {
        throw std::runtime_error("Failed to get memory size");
    }
    return pages * page_size;
#endif

// Handle macOS
#ifdef __APPLE__
    uint64_t memory;
    size_t memorySize = sizeof(memory);
    if (sysctlbyname("hw.memsize", (void*)&memory, &memorySize, nullptr, 0) !=
        0) {
        throw std::runtime_error("Failed to get memory size");
    };
    return memory;
#endif

#ifdef _WIN32
    uint64_t memory;
    if (!GetPhysicallyInstalledSystemMemory(&memory)) {
        throw std::runtime_error("Failed to get memory size");
    }
    return memory * 1024;
#endif
}

std::shared_ptr<arrow::Buffer> SerializeTableToIPC(
    const std::shared_ptr<arrow::Table>& table) {
    auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer =
        arrow::ipc::MakeStreamWriter(sink, table->schema()).ValueOrDie();
    CHECK_ARROW_BASE(writer->WriteTable(*table),
                     "Failed to write Arrow Table to IPC stream");
    CHECK_ARROW_BASE(writer->Close(),
                     "Failed to close Arrow IPC stream writer");
    return sink->Finish().ValueOrDie();
}

// Deserialize an IPC buffer back to an Arrow Table
std::shared_ptr<arrow::Table> DeserializeIPC(
    std::shared_ptr<arrow::Buffer> buffer) {
    auto reader = arrow::ipc::RecordBatchStreamReader::Open(
                      std::make_shared<arrow::io::BufferReader>(buffer))
                      .ValueOrDie();
    return reader->ToTable().ValueOrDie();
}
