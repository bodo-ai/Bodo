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
