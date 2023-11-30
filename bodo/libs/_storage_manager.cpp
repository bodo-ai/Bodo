#include "_storage_manager.h"

#include <filesystem>
#include <iostream>
#include <sstream>

#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/s3fs.h>

#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>          // streaming operators etc.

#include <mpi.h>

#include "_utils.h"

#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(expr, msg, lhs)  \
    {                                           \
        auto res = expr;                        \
        CHECK_ARROW_MEM_RET(res.status(), msg); \
        lhs = std::move(res).ValueOrDie();      \
    }

namespace bodo {

std::shared_ptr<StorageOptions> StorageOptions::Defaults(uint8_t tier) {
    std::shared_ptr<StorageOptions> options = nullptr;
    // TODO: Use fmt::format when available

    // Get the comma-delimited list of drive locations for this storage option
    std::vector<std::string> locations;
    StorageType type = StorageType::Local;

    auto location_env_str = std::string("BODO_BUFFER_POOL_STORAGE_CONFIG_") +
                            std::to_string(tier) + std::string("_DRIVES");
    const char* location_env_ = std::getenv(location_env_str.c_str());
    if (location_env_ == nullptr) {
        // If no location env, assume that no storage option was
        // defined for this tier.
        return nullptr;
    }

    if (std::string_view(location_env_).starts_with("s3://")) {
        const char* enable_s3_spilling_env_ =
            std::getenv("BODO_BUFFER_POOL_ENABLE_REMOTE_SPILLING");
        if (enable_s3_spilling_env_ == nullptr) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            if (rank == 0) {
                std::cerr << "StorageOptions::Defaults: Spilling to S3 is "
                             "disabled by default. Please set "
                             "BODO_BUFFER_POOL_ENABLE_REMOTE_SPILLING to "
                             "enable."
                          << std::endl;
            }
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
    auto gb_available_env_str =
        std::string("BODO_BUFFER_POOL_STORAGE_CONFIG_") + std::to_string(tier) +
        std::string("_SPACE_PER_DRIVE_GiB");

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
        auto space_percent_env_str =
            std::string("BODO_BUFFER_POOL_STORAGE_CONFIG_") +
            std::to_string(tier) + std::string("_USABLE_PERCENTAGE");
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
    arrow::Status Initialize() override {
        return this->fs->CreateDir(this->location.string());
    }

    arrow::Status ReadBlock(uint64_t block_id, int64_t n_bytes,
                            uint8_t* out_ptr) override {
        // Construct File Path
        std::filesystem::path fname =
            this->location / std::to_string(n_bytes) / std::to_string(block_id);

        // Read File Contents to Frame
        std::shared_ptr<arrow::io::InputStream> in_stream;
        CHECK_ARROW_AND_ASSIGN(
            this->fs->OpenInputStream(fname.string()),
            storage_name + "::ReadBlock: Unable to Open FileReader -",
            in_stream);

        int64_t n_bytes_read;
        CHECK_ARROW_AND_ASSIGN(
            in_stream->Read(n_bytes, (void*)out_ptr),
            storage_name + "::ReadBlock: Unable to Read Block File -",
            n_bytes_read);

        CHECK_ARROW_MEM_RET(
            in_stream->Close(),
            storage_name + "::ReadBlock: Unable to Close FileReader -");

        if (n_bytes_read != n_bytes) {
            return arrow::Status::Invalid(
                storage_name + "::ReadBlock: Read Fewer Bytes (" +
                std::to_string(n_bytes_read) + ") than expected (" +
                std::to_string(n_bytes) + ")");
        }

        return this->DeleteBlock(block_id, n_bytes);
    }

    arrow::Result<uint64_t> WriteBlock(uint8_t* in_ptr,
                                       uint64_t n_bytes) override {
        if (!this->CanSpillTo(n_bytes)) {
            return arrow::Status::OutOfMemory(
                "Can not spill to storage manager. Not enough space available");
        }

        uint64_t block_id = this->GetNewBlockID();

        // Construct File Path
        std::filesystem::path fpath = this->location / std::to_string(n_bytes);
        std::filesystem::path fname = fpath / std::to_string(block_id);

        CHECK_ARROW_MEM_RET(
            this->fs->CreateDir(fpath.string(), true),
            storage_name + "::WriteBlock: Unable to Create Directory -");

        std::shared_ptr<arrow::io::OutputStream> out_stream;
        CHECK_ARROW_AND_ASSIGN(
            this->fs->OpenOutputStream(fname.string()),
            storage_name + "::WriteBlock: Unable to Open FileWriter -",
            out_stream);

        CHECK_ARROW_MEM_RET(
            out_stream->Write(in_ptr, (int64_t)n_bytes),
            storage_name + "::WriteBlock: Unable to Write Block to File -");

        CHECK_ARROW_MEM_RET(
            out_stream->Close(),
            storage_name + "::WriteBlock: Unable to Close Writer -");

        this->UpdateSpilledBytes(n_bytes);
        return block_id;
    }

    arrow::Status DeleteBlock(uint64_t block_id, int64_t n_bytes) override {
        // Construct File Path
        std::filesystem::path fname =
            this->location / std::to_string(n_bytes) / std::to_string(block_id);

        CHECK_ARROW_MEM_RET(
            this->fs->DeleteFile(fname.string()),
            storage_name + "::DeleteBlock: Unable to Delete Block File -");

        this->UpdateSpilledBytes(-n_bytes);
        return arrow::Status::OK();
    }

    void Cleanup() override {
        CHECK_ARROW_MEM(
            this->fs->DeleteDir(this->location.string()),
            storage_name + "::Cleanup: Failed to delete spill directory");
    }

    explicit ArrowStorageManager(const std::shared_ptr<StorageOptions> options,
                                 std::string storage_name,
                                 std::shared_ptr<Fs> fs)
        : StorageManager(storage_name, options),
          location(std::filesystem::path(options->location) / this->uuid),
          fs(fs) {}

    explicit ArrowStorageManager(const std::shared_ptr<StorageOptions> options,
                                 std::string storage_name,
                                 std::shared_ptr<Fs> fs, std::string location)
        : StorageManager(storage_name, options),
          location(std::filesystem::path(location) / this->uuid),
          fs(fs) {}

   private:
    /// @brief Location to write spill contents to
    std::filesystem::path location;

    /// @brief arrow::fs::FileSystem handler for reading and writing
    std::shared_ptr<Fs> fs;
};

using LocalStorageManager = ArrowStorageManager<arrow::fs::LocalFileSystem>;
static std::unique_ptr<LocalStorageManager> MakeLocal(
    const std::shared_ptr<StorageOptions> options) {
    auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
    return std::make_unique<LocalStorageManager>(options, "LocalStorageManager",
                                                 fs);
}

using S3StorageManager = ArrowStorageManager<arrow::fs::S3FileSystem>;
static std::unique_ptr<S3StorageManager> MakeS3(
    const std::shared_ptr<StorageOptions> options) {
    auto s3_location = options->location;
    std::string rel_path;

    // TODO(srilman): Arrow 14 adds support for the AWS_ENDPOINT_URL env
    // so it can potentially pick up a MinIO endpoint from here.
    // Double check if this still works after upgrading Arrow
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

    return std::make_unique<S3StorageManager>(options, "S3StorageManager", fs,
                                              rel_path);
}

std::unique_ptr<StorageManager> MakeStorageManager(
    const std::shared_ptr<StorageOptions>& options) {
    std::unique_ptr<StorageManager> manager;

    switch (options->type) {
        case StorageType::Local:
            manager = MakeLocal(options);
            break;
        case StorageType::S3:
            manager = MakeS3(options);
            break;
    }

    CHECK_ARROW_MEM(manager->Initialize(),
                    "Failed to Initialize Storage Manager:");

    return manager;
}

}  // namespace bodo
