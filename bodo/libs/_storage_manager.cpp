#include "_storage_manager.h"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>

#include <arrow/status.h>
#include <fcntl.h>
#include <sys/errno.h>
#ifdef __linux__
// For fallocate
#include <linux/falloc.h>
#endif

#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/result.h>

#include <boost/json.hpp>
#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>          // streaming operators etc.

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include <mpi.h>

#include "_utils.h"

#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_MEM_AND_ASSIGN(expr, msg, lhs) \
    {                                              \
        auto res = expr;                           \
        CHECK_ARROW_MEM(res.status(), msg);        \
        lhs = std::move(res).ValueOrDie();         \
    }

#define CHECK_ARROW_AND_ASSIGN(expr, msg, lhs)  \
    {                                           \
        auto res = expr;                        \
        CHECK_ARROW_MEM_RET(res.status(), msg); \
        lhs = std::move(res).ValueOrDie();      \
    }

using namespace std::chrono;

namespace bodo {

void StorageManager::UpdateSpilledBytes(int64_t diff) {
    stats_.curr_spilled_bytes += diff;
    stats_.curr_num_blocks_spilled += diff > 0 ? 1 : -1;

    if (stats_.curr_spilled_bytes > stats_.max_spilled_bytes) {
        stats_.max_spilled_bytes = stats_.curr_spilled_bytes;
    }

    if (diff > 0) {
        this->stats_.total_num_blocks_spilled++;
        this->stats_.total_spilled_bytes += diff;
    }
}

std::shared_ptr<StorageOptions> StorageOptions::Defaults(uint8_t tier) {
    std::shared_ptr<StorageOptions> options = nullptr;

    // Get the comma-delimited list of drive locations for this storage option
    std::vector<std::string> locations;
    StorageType type = StorageType::Local;

    auto location_env_str =
        fmt::format("BODO_BUFFER_POOL_STORAGE_CONFIG_{}_DRIVES", tier);
    const char* location_env_ = std::getenv(location_env_str.c_str());
    if (location_env_ == nullptr) {
        // If no location env, assume that no storage option was
        // defined for this tier.
        return nullptr;
    }

    if (std::string_view(location_env_).starts_with("s3://")) {
        const char* disable_remote_spilling_env_ =
            std::getenv("BODO_BUFFER_POOL_DISABLE_REMOTE_SPILLING");
        if (disable_remote_spilling_env_ &&
            !std::strcmp(disable_remote_spilling_env_, "1")) {
            return nullptr;
        }
        type = StorageType::S3;

    } else if (std::string_view(location_env_).starts_with("abfs://")) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0) {
            std::cerr << "StorageOptions::Defaults: Spilling to Azure Blob "
                         "Storage is not supported yet."
                      << std::endl;
        }
        return nullptr;
    }

    std::stringstream ss(location_env_);
    while (ss.good()) {
        std::string substr;
        std::getline(ss, substr, ',');
        locations.push_back(substr);
    }

    // Compute the number of bytes available for this storage location
    auto gb_available_env_str = fmt::format(
        "BODO_BUFFER_POOL_STORAGE_CONFIG_{}_SPACE_PER_DRIVE_GiB", tier);

    if (const char* gb_available_env_ =
            std::getenv(gb_available_env_str.c_str())) {
        // Create Options
        options = std::make_shared<StorageOptions>();
        options->type = type;

        // Assign Location
        auto [num_ranks_on_node, rank_on_node] = dist_get_ranks_on_node();
        int loc_idx = rank_on_node % locations.size();
        options->location = locations[loc_idx];

        // Parse GB Storage Numbers
        int gb_in_storage = std::stoi(gb_available_env_);

        // If gb_in_storage < 0, assume unlimited storage
        // Indicated by a -1
        if (gb_in_storage < 0) {
            options->usable_size = -1;
            return options;
        }

        int64_t bytes_in_storage = gb_in_storage * 1024ll * 1024ll * 1024ll;

        // What percentage of the total available space in storage
        // should be used for the buffer pool. For now, the default
        // will be 90% unless specified by the environment variable
        double space_percent = 0.90;
        auto space_percent_env_str = fmt::format(
            "BODO_BUFFER_POOL_STORAGE_CONFIG_{}_USABLE_PERCENTAGE", tier);
        if (const char* space_percent_env_ =
                std::getenv(space_percent_env_str.c_str())) {
            // We expect this to be in percentages and not fraction,
            // i.e. it should be set to 45 if we want to use 45% (or 0.45)
            // of the total available space.
            space_percent = std::stod(space_percent_env_) / 100.0;
        }
        double bytes_available = bytes_in_storage * space_percent;

        // Determine the number of bytes each rank gets by evenly
        // distributing ranks to available locations in a round
        // robin fashion
        // Ex: If node has 5 ranks and 3 locations of 20GB each
        // - Rank 0 and 3 use Location 1 and max 10GB
        // - Rank 1 and 4 use Location 2 and max 10GB
        // - Rank 2 uses Location 3 and max 20GB

        // Compute Even Storage for Ranks Using Same Location
        int leftover_ranks = num_ranks_on_node % locations.size();
        int min_ranks_per_loc =
            (num_ranks_on_node - leftover_ranks) / locations.size();
        int num_ranks_on_loc =
            min_ranks_per_loc + (rank_on_node < leftover_ranks);
        int64_t storage_for_rank =
            static_cast<int64_t>(bytes_available / (double)num_ranks_on_loc);
        options->usable_size = storage_for_rank;
    }

    return options;
}

StorageManager::StorageManager(std::string storage_name,
                               std::shared_ptr<StorageOptions> options)
    : storage_name(storage_name), options(options) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Generate Unique UUID per Rank
    boost::uuids::uuid _uuid = boost::uuids::random_generator()();
    std::string uuid = boost::uuids::to_string(_uuid);

    this->uuid = std::to_string(rank) + "-" + uuid;
}

/// @brief Storage Manager for Filesystems with an Arrow Interface
template <typename Fs>
class ArrowStorageManager final : public StorageManager {
   public:
    explicit ArrowStorageManager(
        const std::shared_ptr<StorageOptions> options, std::string storage_name,
        std::shared_ptr<Fs> fs,
        const std::span<const uint64_t> size_class_bytes_,
        bool is_object_store_)
        : StorageManager(storage_name, options),
          is_object_store(is_object_store_),
          location(std::filesystem::path(options->location) / this->uuid),
          fs(fs) {
        this->size_class_bytes.assign(size_class_bytes_.begin(),
                                      size_class_bytes_.end());

        // Object stores don't require directories to be created
        // beforehand, since all objects are stored in a flat namespace
        // with tags for the directory structure
        if (this->is_object_store) {
            return;
        }

        // Create Top Directory and Subdirectories for Each Block Size
        CHECK_ARROW_MEM(this->fs->CreateDir(this->location.string()),
                        storage_name + ": Unable to Create Top Directory -");

        for (const auto& size_ : size_class_bytes_) {
            std::filesystem::path fpath =
                this->location / std::to_string(size_);
            CHECK_ARROW_MEM(
                this->fs->CreateDir(fpath.string()),
                storage_name + ": Unable to Create Sub-Block Directory -");
        }
    }

    bool CanSpillTo(uint8_t size_class_idx) override {
        return options->usable_size < 0 ||
               (stats_.curr_spilled_bytes + size_class_bytes[size_class_idx]) <=
                   static_cast<uint64_t>(options->usable_size);
    }

    void ReadBlock(int64_t block_id, uint8_t size_class_idx,
                   uint8_t* out_ptr) override {
        auto start = start_now(options->tracing_mode);
        uint64_t n_bytes = this->size_class_bytes[size_class_idx];

        // Construct File Path
        std::filesystem::path fname =
            this->location / std::to_string(n_bytes) / std::to_string(block_id);

        // Read File Contents to Frame
        std::shared_ptr<arrow::io::InputStream> in_stream;
        CHECK_ARROW_MEM_AND_ASSIGN(
            this->fs->OpenInputStream(fname.string()),
            storage_name + "::ReadBlock: Unable to Open FileReader -",
            in_stream);

        int64_t n_bytes_read;
        CHECK_ARROW_MEM_AND_ASSIGN(
            in_stream->Read(n_bytes, (void*)out_ptr),
            storage_name + "::ReadBlock: Unable to Read Block File -",
            n_bytes_read);

        CHECK_ARROW_MEM(
            in_stream->Close(),
            storage_name + "::ReadBlock: Unable to Close FileReader -");

        if (n_bytes_read != static_cast<int64_t>(n_bytes)) {
            throw std::runtime_error(fmt::format(
                "{}::ReadBlock: Read Fewer Bytes ({}) than expected ({})",
                storage_name, n_bytes_read, n_bytes));
        }

        stats_.total_read_bytes += n_bytes;
        stats_.total_num_blocks_read++;
        if (options->tracing_mode) {
            milli_double dur = steady_clock::now() - start.value();
            this->stats_.total_read_time += dur;
        }

        CHECK_ARROW_MEM(this->DeleteBlock(block_id, size_class_idx),
                        storage_name + "::ReadBlock: Unable to delete block -");
    }

    int64_t WriteBlock(uint8_t* in_ptr, uint8_t size_class_idx) override {
        if (!this->CanSpillTo(size_class_idx)) {
            return -1;
        }

        uint64_t n_bytes = this->size_class_bytes[size_class_idx];

        auto start = start_now(options->tracing_mode);
        uint64_t block_id = this->GetNewBlockID();

        // Construct File Path
        std::filesystem::path fname =
            this->location / std::to_string(n_bytes) / std::to_string(block_id);

        std::shared_ptr<arrow::io::OutputStream> out_stream;
        CHECK_ARROW_MEM_AND_ASSIGN(
            this->fs->OpenOutputStream(fname.string()),
            storage_name + "::WriteBlock: Unable to Open FileWriter -",
            out_stream);

        CHECK_ARROW_MEM(
            out_stream->Write(in_ptr, (int64_t)n_bytes),
            storage_name + "::WriteBlock: Unable to Write Block to File -");

        CHECK_ARROW_MEM(
            out_stream->Close(),
            storage_name + "::WriteBlock: Unable to Close Writer -");

        this->UpdateSpilledBytes(n_bytes);
        if (options->tracing_mode) {
            milli_double dur = steady_clock::now() - start.value();
            this->stats_.total_write_time += dur;
        }
        return block_id;
    }

    arrow::Status DeleteBlock(int64_t block_id,
                              uint8_t size_class_idx) override {
        auto start = start_now(options->tracing_mode);
        uint64_t n_bytes = this->size_class_bytes[size_class_idx];

        // Construct File Path
        std::filesystem::path fname =
            this->location / std::to_string(n_bytes) / std::to_string(block_id);

        CHECK_ARROW_MEM_RET(
            this->fs->DeleteFile(fname.string()),
            storage_name + "::DeleteBlock: Unable to Delete Block File -");

        this->UpdateSpilledBytes(-n_bytes);
        this->stats_.total_num_del_calls++;
        this->stats_.total_bytes_del += n_bytes;
        if (options->tracing_mode) {
            milli_double dur = steady_clock::now() - start.value();
            this->stats_.total_delete_time += dur;
        }
        return arrow::Status::OK();
    }

    void Cleanup() override {
        if (this->is_object_store && this->stats_.curr_spilled_bytes == 0) {
            return;
        }

        arrow::Status cleanup_status =
            this->fs->DeleteDir(this->location.string());
        if (this->is_object_store && !cleanup_status.ok()) {
            fmt::println(stderr,
                         "[Warning] {}::Cleanup: Failed to delete spill "
                         "directory - {}",
                         storage_name, cleanup_status.ToString());
        } else if (!this->is_object_store) {
            CHECK_ARROW_MEM(cleanup_status, storage_name +
                                                "::Cleanup: Unable to Delete "
                                                "Spill Directory -");
        }
    }

   private:
    /// @brief Check if storage is an object store
    /// If it is, we may need to ignore an error on cleanup if we never spill
    const bool is_object_store;

    /// @brief Location to write spill contents to
    std::filesystem::path location;

    /// @brief Number of bytes per block
    std::vector<uint64_t> size_class_bytes;

    /// @brief LocalFileSystem handler for reading and writing
    std::shared_ptr<Fs> fs;
};

/// Storage Manager for Local Disk-based Filesystems
struct SparseFileSizeInfo {
    int file_descriptor = -1;
    uint64_t block_size = 0;
    uint32_t blocks_used = 0;
    uint32_t block_capacity = 0;
    uint64_t leftover_threshold = 0;
    std::vector<uint32_t> free_block_list;
};

class SparseFileStorageManager final : public StorageManager {
   public:
    explicit SparseFileStorageManager(
        const std::shared_ptr<StorageOptions> options,
        std::span<const uint64_t> size_class_bytes)
        : StorageManager("SparseFileStorageManager", options),
          location(std::filesystem::path(options->location) / this->uuid) {
        // Set Threshold for Cleanup
        uint64_t total_threshold;
        if (const char* threshold_env =
                std::getenv("BODO_BUFFER_POOL_SPARSE_DELETE_THRESHOLD")) {
            total_threshold = std::stoul(threshold_env);
        } else {
            // Default to total 1MiB, 32-bit idxs
            total_threshold = 1024 * 1024 / sizeof(uint32_t);
        }
        // For the total to be 'total_threshold', we will start the smallest
        // block size's SizeClass with total_threshold/2.
        total_threshold /= 2;

        bool created = std::filesystem::create_directories(location);
        if (!created) {
            throw std::runtime_error(
                "SparseFileStorageManager(): Unable to create "
                "directory");
        }

        // Set O_DIRECT flag if requested
        // This is not default because right now, the Buffer Pool thinks Bodo
        // is using much more memory than what the OS sees, due to bad
        // utilization of memory requests. When O_DIRECT is not used, the OS
        // will cache writes and reads in any leftover memory, slightly
        // improving performance.
        int o_flags = 0;
        bool use_o_direct = false;
        if (const char* o_direct_env =
                std::getenv("BODO_BUFFER_POOL_USE_O_DIRECT")) {
            use_o_direct = !std::strcmp(o_direct_env, "1");
        }

#ifdef __linux__
        if (use_o_direct) {
            o_flags |= O_DIRECT;
        }
#endif

        // Initialize File Size Info
        for (auto& size : size_class_bytes) {
            SparseFileSizeInfo fi = SparseFileSizeInfo();
            fi.block_capacity = 1;
            fi.blocks_used = 0;
            fi.block_size = size;
            fi.leftover_threshold =
                std::max(total_threshold, static_cast<uint64_t>(1));
            total_threshold /= 2;

            std::string path =
                (location / std::to_string(fi.block_size)).string();
            fi.file_descriptor = open(path.c_str(), O_CREAT | O_RDWR | o_flags,
                                      S_IRUSR | S_IWUSR);

            if (fi.file_descriptor == -1) {
                this->Cleanup();
                throw std::runtime_error(
                    "SparseFileStorageManager(): Unable to open file");
            }

            // Equivalent to O_DIRECT on macOS
#ifdef __APPLE__
            if (use_o_direct) {
                fcntl(fi.file_descriptor, F_NOCACHE, 1);
            }
#endif

            // Construct 1 Frame per File
            int err = ftruncate(fi.file_descriptor, fi.block_size);
            if (err == -1) {
                this->Cleanup();
                throw std::runtime_error(
                    "SparseFileStorageManager(): Unable to size the "
                    "file at initialization");
            }

            fi.free_block_list.reserve(fi.leftover_threshold);
            size_class_to_file_info.push_back(fi);
        }
    }

    /**
     * @brief Cleanup any leftover bytes in the storage manager
     * one size class.
     * @param fi Info related to the size class of the block to clean
     * Used when the storage manager is either full or when
     * there are too many leftover blocks in a size class
     */
    arrow::Status cleanup_helper(SparseFileSizeInfo& fi) {
        auto start = start_now(this->options->tracing_mode);
        off_t n_bytes = fi.block_size;
        for (auto& block_id : fi.free_block_list) {
            off_t offset = block_id * fi.block_size;
#ifdef __linux__
            int err = fallocate(fi.file_descriptor,
                                FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                                offset, n_bytes);
#elif __APPLE__
            fpunchhole_t args = {0, 0, offset, n_bytes};
            int err = fcntl(fi.file_descriptor, F_PUNCHHOLE, &args);
#endif
            if (err != 0) {
                return arrow::Status::IOError(
                    "SparseFileStorageManager::cleanup_helper: Unable "
                    "to punch hole in file");
            }
        }
        fi.free_block_list.clear();
        // TODO: Add an additional metrics to track number of fallocate calls
        this->stats_.total_bytes_del +=
            (this->curr_occupied_bytes - this->stats_.curr_spilled_bytes);
        this->curr_occupied_bytes = this->stats_.curr_spilled_bytes;

        if (this->options->tracing_mode) {
            milli_double dur = steady_clock::now() - start.value();
            this->stats_.total_delete_time += dur;
        }
        return arrow::Status::OK();
    }

    /**
     * @brief Cleanup any leftover bytes in the storage manager for all
     * size classes.
     *
     * Used when the storage manager is either full or when
     * there are too many leftover blocks in a size class
     */
    arrow::Status cleanup_all_helper() {
        for (auto& fi : this->size_class_to_file_info) {
            CHECK_ARROW_MEM_RET(this->cleanup_helper(fi), "");
        }
        this->stats_.total_num_del_calls++;
        return arrow::Status::OK();
    }

    bool CanSpillTo(uint8_t size_class_idx) override {
        uint64_t amount =
            this->size_class_to_file_info[size_class_idx].block_size;
        if ((options->usable_size < 0) ||
            ((this->curr_occupied_bytes + amount) <=
             static_cast<uint64_t>(options->usable_size))) {
            return true;
        } else if ((this->stats_.curr_spilled_bytes + amount) >
                   static_cast<uint64_t>(options->usable_size)) {
            return false;
        } else if (this->size_class_to_file_info[size_class_idx]
                       .free_block_list.size() > 0) {
            // If there are leftover blocks in the size class, we can reuse one
            // so no need to clean up
            return true;
        }

        CHECK_ARROW_MEM(this->cleanup_all_helper(),
                        "SparseFileStorageManager::CanSpillTo(): Unable to "
                        "cleanup leftover bytes - ");
        return ((this->curr_occupied_bytes + amount) <=
                static_cast<uint64_t>(options->usable_size));
    }

    /**
     * @brief Helper to delete a block. Allows for some quick reuse
     * between ReadBlock and DeleteBlock
     *
     * @param fi Info related to the size class of the block to delete
     * @param block_id ID of block to delete
     * @return arrow::Status Did the delete succeed or error if not
     */
    arrow::Status delete_helper(SparseFileSizeInfo& fi, uint64_t block_id) {
        auto start = start_now(this->options->tracing_mode);
        fi.free_block_list.push_back(block_id);
        this->UpdateSpilledBytes(-fi.block_size);

        if (fi.free_block_list.size() >= fi.leftover_threshold) {
            CHECK_ARROW_MEM_RET(this->cleanup_helper(fi), "");
            this->stats_.total_num_del_calls++;
        }

        if (this->options->tracing_mode) {
            milli_double dur = steady_clock::now() - start.value();
            this->stats_.total_delete_time += dur;
        }
        return arrow::Status::OK();
    }

    void ReadBlock(int64_t block_id, uint8_t size_class_idx,
                   uint8_t* out_ptr) override {
        auto start = start_now(this->options->tracing_mode);
        SparseFileSizeInfo& fi = this->size_class_to_file_info[size_class_idx];
        int64_t n_bytes_read = pread(fi.file_descriptor, out_ptr, fi.block_size,
                                     (off_t)(block_id * fi.block_size));

        if (n_bytes_read == -1) {
            throw std::runtime_error(
                "SparseFileStorageManager::ReadBlock: Error when reading from "
                "file");
        } else if (n_bytes_read == 0) {
            throw std::runtime_error(
                "SparseFileStorageManager::ReadBlock: Unexpected reading at "
                "EOF");
        } else if (n_bytes_read != static_cast<int64_t>(fi.block_size)) {
            throw std::runtime_error(
                "SparseFileStorageManager::ReadBlock: Failed to read expected "
                "contents from file");
        }

        if (this->options->tracing_mode) {
            milli_double dur = steady_clock::now() - start.value();
            this->stats_.total_read_time += dur;
        }
        CHECK_ARROW_MEM(
            delete_helper(fi, block_id),
            "SparseFileStorageManager::ReadBlock: Unable to delete block -");
    }

    int64_t WriteBlock(uint8_t* in_ptr, uint8_t size_class_idx) override {
        if (!this->CanSpillTo(size_class_idx)) {
            return -1;
        }

        auto start = start_now(this->options->tracing_mode);
        SparseFileSizeInfo& fi = this->size_class_to_file_info[size_class_idx];

        int64_t block_id;
        if (fi.free_block_list.size() > 0) {
            block_id = static_cast<uint32_t>(fi.free_block_list.back());
            fi.free_block_list.pop_back();
        } else {
            if (fi.blocks_used == fi.block_capacity) {
                fi.block_capacity *= 2;
                int err = ftruncate(fi.file_descriptor,
                                    (off_t)(fi.block_capacity * fi.block_size));
                if (err == -1) {
                    throw std::runtime_error(
                        "SparseFileStorageManager::WriteBlock: Error when "
                        "expanding file");
                }
            }
            block_id = static_cast<int64_t>(fi.blocks_used);
            fi.blocks_used++;
            this->curr_occupied_bytes += fi.block_size;
        }

        int64_t n_bytes_written =
            pwrite(fi.file_descriptor, in_ptr, fi.block_size,
                   (off_t)(block_id * fi.block_size));

        if (n_bytes_written == -1) {
            throw std::runtime_error(
                "SparseFileStorageManager::WriteBlock: Error when writing to "
                "file");
        } else if (n_bytes_written != static_cast<int64_t>(fi.block_size)) {
            throw std::runtime_error(
                "SparseFileStorageManager::WriteBlock: Did not write full "
                "contents "
                "of block");
        }

        this->UpdateSpilledBytes(fi.block_size);
        if (this->options->tracing_mode) {
            milli_double dur = steady_clock::now() - start.value();
            this->stats_.total_write_time += dur;
        }
        return block_id;
    }

    arrow::Status DeleteBlock(int64_t block_id,
                              uint8_t size_class_idx) override {
        SparseFileSizeInfo& fi = size_class_to_file_info[size_class_idx];
        return delete_helper(fi, block_id);
    }

    void Cleanup() override {
        for (auto& fi : this->size_class_to_file_info) {
            if (close(fi.file_descriptor) != 0) {
                throw std::runtime_error(
                    "SparseFileStorageManager::Cleanup: Error when closing "
                    "file");
            }
        }

        std::filesystem::remove_all(location);
    }

   private:
    std::vector<SparseFileSizeInfo> size_class_to_file_info;

    /// @brief Base Location to write spill contents to
    std::filesystem::path location;

    /// @brief Current number of bytes occupied in storage
    /// The storage manager may not delete blocks immediately
    /// so we need to track any leftovers
    uint64_t curr_occupied_bytes = 0;
};

using LocalStorageManager = ArrowStorageManager<arrow::fs::LocalFileSystem>;
static std::unique_ptr<StorageManager> MakeLocal(
    const std::shared_ptr<StorageOptions> options,
    const std::span<const uint64_t> size_class_bytes) {
    // Sparse File Storage Manager is supported on
    // macOS Monterey and above (12.0+, oldest supported version in 2024)
    // Linux should support from 2013 onwards
    // - O_DIRECT: Linux 2.4.10 (2001), ignored in previous versions
    // - O_TMPFILE: Linux 3.11 (2013)
    //   - XFS in Linux 3.15 (2014)
    //   - Btrfs in Linux 3.16 (2014)
    //   - F2FS in Linux 3.16 (2014)
    //   - ubifs in Linux 4.9 (2016)
    // - fallocate: Linux 2.6.38 in glibc 2.10 (2011)
    // - FALLOC_FL_PUNCH_HOLE in glib 2.18 (2013)
    //   - XFS in Linux 2.6.38 (2011)
    //   - ext4 in Linux 3.0 (2011)
    //   - Btrfs in Linux 3.7 (2012)
    //   - tmpfs in Linux 3.5 (2012)
    //   - gfs2 in Linux 4.16 (2018)
    // - FALLOC_FL_KEEP_SIZE in glib 2.18 (2013)
    // - ftruncate: glibc 2.3.5 (2006)

    // Just in case, we test if SparseFileStorageManager works
    // and default to LocalStorageManager if it doesn't
    try {
        return std::make_unique<SparseFileStorageManager>(options,
                                                          size_class_bytes);
    } catch (const std::exception& e) {
        fmt::println(
            stderr,
            "MakeLocal: SparseFileStorageManager failed to initialize: {}",
            e.what());

        auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
        return std::make_unique<LocalStorageManager>(
            options, "LocalStorageManager", fs, size_class_bytes, false);
    }
}

using S3StorageManager = ArrowStorageManager<arrow::fs::S3FileSystem>;
static std::unique_ptr<S3StorageManager> MakeS3(
    const std::shared_ptr<StorageOptions> options,
    const std::span<const uint64_t> size_class_bytes) {
    auto s3_location = options->location;
    std::string rel_path;

    // Arrow 14 adds support for the AWS_ENDPOINT_URL env
    // If set, this will use it as the S3 endpoint
    // For example, it will spill to MinIO if it is exposed through the env
    if (!arrow::fs::IsS3Initialized()) {
        arrow::fs::S3GlobalOptions s3_global_options;
        s3_global_options.log_level = arrow::fs::S3LogLevel::Fatal;
        auto stat = arrow::fs::InitializeS3(s3_global_options);
        CHECK_ARROW_MEM(stat,
                        "S3StorageManager::Make: Failed to "
                        "initialize Arrow S3 -");
    }

    auto s3_opts_res = arrow::fs::S3Options::FromUri(s3_location, &rel_path);
    CHECK_ARROW_MEM(s3_opts_res.status(),
                    "S3StorageManager::Make: Failed to "
                    "parse S3 URI for Buffer Pool Spilling -");
    auto s3_opts = s3_opts_res.ValueOrDie();

    auto fs_res = arrow::fs::S3FileSystem::Make(s3_opts);
    CHECK_ARROW_MEM(fs_res.status(),
                    "S3StorageManager::Make: Failed to "
                    "create S3FileSystem for Buffer Pool "
                    "Spilling -");
    auto fs = fs_res.ValueOrDie();

    options->location = rel_path;
    return std::make_unique<S3StorageManager>(options, "S3StorageManager", fs,
                                              size_class_bytes, true);
}

std::unique_ptr<StorageManager> MakeStorageManager(
    const std::shared_ptr<StorageOptions>& options,
    const std::span<const uint64_t> size_class_bytes) {
    std::unique_ptr<StorageManager> manager;

    switch (options->type) {
        case StorageType::Local:
            manager = MakeLocal(options, size_class_bytes);
            break;
        case StorageType::S3:
            manager = MakeS3(options, size_class_bytes);
            break;
    }

    return manager;
}

constexpr size_t NumStatCols = 13;
const char* StatColNames[NumStatCols] = {
    "Storage Name",         "Current Spilled Bytes", "Current Blocks Spilled",

    "Total Blocks Spilled", "Total Blocks Read",     "Total Num Delete Calls",
    "Total Bytes Spilled",  "Total Bytes Read",      "Total Bytes Deleted",
    "Max Spilled Bytes",

    "Total Read Time",      "Total Write Time",      "Total Delete Time"};

void PrintStorageManagerStats(
    FILE* os, const std::span<std::string_view> m_names,
    const std::span<const StorageManagerStats> stats) {
    size_t row_name_width = 0;
    for (const auto& s : StatColNames) {
        row_name_width = std::max(row_name_width, std::strlen(s));
    }
    row_name_width++;

    size_t total_width = row_name_width + 6 * m_names.size();
    for (const auto& m : m_names) {
        total_width += m.size();
    }

    // Construct Row Names
    std::vector<std::string> out_lines(NumStatCols, "");
    for (size_t i = 0; i < NumStatCols; i++) {
        fmt::format_to(std::back_inserter(out_lines[i]), "{0:<{1}}",
                       StatColNames[i], row_name_width);
    }

    // Append Storage Manager Index and Name
    // Construct Row Values
    for (size_t i = 0; i < m_names.size(); i++) {
        fmt::format_to(std::back_inserter(out_lines[0]), "│ {}: {} ", i,
                       m_names[i]);
        size_t len = m_names[i].size();
        const auto& s = stats[i];

        fmt::format_to(std::back_inserter(out_lines[1]), "│    {0:>{1}} ",
                       BytesToHumanReadableString(s.curr_spilled_bytes), len);
        fmt::format_to(std::back_inserter(out_lines[2]), "│    {0:>{1}} ",
                       s.curr_num_blocks_spilled, len);
        fmt::format_to(std::back_inserter(out_lines[3]), "│    {0:>{1}} ",
                       s.total_num_blocks_spilled, len);
        fmt::format_to(std::back_inserter(out_lines[4]), "│    {0:>{1}} ",
                       s.total_num_blocks_read, len);
        fmt::format_to(std::back_inserter(out_lines[5]), "│    {0:>{1}} ",
                       s.total_num_del_calls, len);
        fmt::format_to(std::back_inserter(out_lines[6]), "│    {0:>{1}} ",
                       BytesToHumanReadableString(s.total_spilled_bytes), len);
        fmt::format_to(std::back_inserter(out_lines[7]), "│    {0:>{1}} ",
                       BytesToHumanReadableString(s.total_read_bytes), len);
        fmt::format_to(std::back_inserter(out_lines[8]), "│    {0:>{1}} ",
                       BytesToHumanReadableString(s.total_bytes_del), len);
        fmt::format_to(std::back_inserter(out_lines[9]), "│    {0:>{1}} ",
                       BytesToHumanReadableString(s.max_spilled_bytes), len);
        fmt::format_to(std::back_inserter(out_lines[10]), "│    {0:>{1}} ",
                       s.total_read_time, len);
        fmt::format_to(std::back_inserter(out_lines[11]), "│    {0:>{1}} ",
                       s.total_write_time, len);
        fmt::format_to(std::back_inserter(out_lines[12]), "│    {0:>{1}} ",
                       s.total_delete_time, len);
    }

    // Print Header
    fmt::println(os, "{0:─^{1}}", " Storage Manager Stats ", total_width);
    fmt::println(os, "{0:─^{1}}", "", total_width);
    fmt::print(os, "{0:<{1}}", out_lines[0], row_name_width);
    fmt::print(os, "\n{0:─^{1}}\n", "", total_width);
    for (size_t i = 1; i < out_lines.size(); i++) {
        fmt::print(os, "{}\n", out_lines[i]);
    }
    fmt::println(os, "{0:─^{1}}", "", total_width);
}

boost::json::object GetStorageManagerStats(
    const std::span<std::string_view> m_names,
    const std::span<const StorageManagerStats> stats) {
    boost::json::object out_stats;

    // Construct Row Values
    for (size_t i = 0; i < m_names.size(); i++) {
        boost::json::object stat_for_manager;
        stat_for_manager["Current Spilled Bytes"] = stats[i].curr_spilled_bytes;
        stat_for_manager["Current Blocks Spilled"] =
            stats[i].curr_num_blocks_spilled;
        stat_for_manager["Total Blocks Spilled"] =
            stats[i].total_num_blocks_spilled;
        stat_for_manager["Total Blocks Read"] = stats[i].total_num_blocks_read;
        stat_for_manager["Total Num Delete Calls"] =
            stats[i].total_num_del_calls;
        stat_for_manager["Total Bytes Spilled"] = stats[i].total_spilled_bytes;
        stat_for_manager["Total Bytes Read"] = stats[i].total_read_bytes;
        stat_for_manager["Total Bytes Deleted"] = stats[i].total_bytes_del;
        stat_for_manager["Max Spilled Bytes"] = stats[i].max_spilled_bytes;
        stat_for_manager["Total Read Time (ms)"] =
            stats[i].total_read_time.count();
        stat_for_manager["Total Write Time (ms)"] =
            stats[i].total_write_time.count();
        stat_for_manager["Total Delete Time (ms)"] =
            stats[i].total_delete_time.count();

        out_stats[std::string(m_names[i])] = stat_for_manager;
    }

    return out_stats;
}
}  // namespace bodo
