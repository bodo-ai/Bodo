#include "_memory.h"
#include <arrow/util/bit_util.h>
#include <sys/mman.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <sstream>

#define MAX_NUM_STORAGE_MANAGERS 4

#define CHECK_ARROW_AND_ASSIGN(expr, msg, lhs)  \
    {                                           \
        auto res = expr;                        \
        CHECK_ARROW_MEM_RET(res.status(), msg); \
        lhs = std::move(res).ValueOrDie();      \
    }

namespace bodo {

//// Helper Functions
/**
 * @brief Determine if buffer pointer is unswizzled
 * @return True if ptr is unswizzled, false otherwise
 */
inline bool is_unswizzled(Swip ptr) {
    uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);
    return (ptr_val >> 63) == 1ull;
}

/**
 * @brief Extract encoded info from a swip pointer
 * that could have been unswizzled.
 *
 * @param ptr Swip pointer that could be unswizzled
 * @return std::optional<std::tuple<uint8_t, uint8_t, uint64_t>>
 * - std::nullopt if pointer is swizzled (not evicted)
 * - std::tuple   if pointer is unswizzled
 *   - uint8_t  for size class index
 *   - uint8_t  for storage class index
 *   - uint64_t for block id
 */
std::optional<std::tuple<uint8_t, uint8_t, uint64_t>> extract_swip_ptr(
    Swip ptr) {
    if (!is_unswizzled(ptr)) {
        return {};
    }

    uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);
    uint8_t size_class = (ptr_val >> 57) & 0b111111ull;
    uint8_t storage_class = (ptr_val >> 55) & 0b11ull;
    uint64_t block_id = ptr_val & (0x7F'FF'FF'FF'FF'FF'FFull);
    return std::make_tuple(size_class, storage_class, block_id);
}

/**
 * @brief Construct unswizzled swip pointer from components
 *
 * @param size_class_idx Index of size class block is from
 * @param storage_class_idx Storage class block will be written to
 * @param block_id Unique index of Block
 * @return Swip Unswizzled swip pointer
 */
Swip construct_unswizzled_swip(uint8_t size_class_idx,
                               uint8_t storage_class_idx, uint64_t block_id) {
    uint64_t size_class_enc = static_cast<uint64_t>(size_class_idx) << 57;
    uint64_t storage_class_enc = static_cast<uint64_t>(storage_class_idx) << 55;
    return (Swip)((1ull << 63) | size_class_enc | storage_class_enc | block_id);
}

//// StorageManager

arrow::Status LocalStorageManager::ReadBlock(uint64_t block_id, int64_t n_bytes,
                                             uint8_t* out_ptr) {
    // Construct File Path
    std::filesystem::path fname = this->location / this->uuid /
                                  std::to_string(n_bytes) /
                                  std::to_string(block_id);

    // Read File Contents to Frame
    std::shared_ptr<arrow::io::InputStream> in_stream;
    CHECK_ARROW_AND_ASSIGN(
        this->fs.OpenInputStream(fname.string()),
        "LocalStorageManager::ReadBlock: Unable to Open FileReader -",
        in_stream);

    int64_t n_bytes_read;
    CHECK_ARROW_AND_ASSIGN(
        in_stream->Read(n_bytes, (void*)out_ptr),
        "LocalStorageManager::ReadBlock: Unable to Read Block File -",
        n_bytes_read);

    CHECK_ARROW_MEM_RET(
        in_stream->Close(),
        "LocalStorageManager::ReadBlock: Unable to Close FileReader -");

    if (n_bytes_read != n_bytes) {
        return arrow::Status::Invalid(
            "LocalStorageManager::ReadBlock: Read Fewer Bytes than Expected");
    }

    return this->DeleteBlock(block_id, n_bytes);
}

arrow::Result<uint64_t> LocalStorageManager::WriteBlock(uint8_t* in_ptr,
                                                        uint64_t n_bytes) {
    if (!this->CanSpillTo(n_bytes)) {
        return arrow::Status::OutOfMemory(
            "Can not spill to storage manager. Not enough space available");
    }

    uint64_t block_id = this->GetNewBlockID();

    // Construct File Path
    std::filesystem::path fpath =
        this->location / this->uuid / std::to_string(n_bytes);
    std::filesystem::path fname = fpath / std::to_string(block_id);

    CHECK_ARROW_MEM_RET(
        this->fs.CreateDir(fpath.string(), true),
        "LocalStorageManager::WriteBlock: Unable to Create Directory -");

    std::shared_ptr<arrow::io::OutputStream> out_stream;
    CHECK_ARROW_AND_ASSIGN(
        this->fs.OpenOutputStream(fname.string()),
        "LocalStorageManager::WriteBlock: Unable to Open FileWriter -",
        out_stream);

    CHECK_ARROW_MEM_RET(
        out_stream->Write(in_ptr, (int64_t)n_bytes),
        "LocalStorageManager::WriteBlock: Unable to Write Block to File -");

    CHECK_ARROW_MEM_RET(
        out_stream->Close(),
        "LocalStorageManager::WriteBlock: Unable to Close Writer -");

    this->UpdateSpilledBytes(n_bytes);
    return block_id;
}

arrow::Status LocalStorageManager::DeleteBlock(uint64_t block_id,
                                               int64_t n_bytes) {
    // Construct File Path
    std::filesystem::path fname = this->location / this->uuid /
                                  std::to_string(n_bytes) /
                                  std::to_string(block_id);

    CHECK_ARROW_MEM_RET(
        this->fs.DeleteFile(fname.string()),
        "LocalStorageManager::DeleteBlock: Unable to Delete Block File -");

    this->UpdateSpilledBytes(-n_bytes);
    return arrow::Status::OK();
}

//// SizeClass

SizeClass::SizeClass(
    uint8_t idx,
    const std::span<std::unique_ptr<StorageManager>> storage_managers,
    size_t capacity, size_t block_size)
    : idx_(idx),
      capacity_(capacity),
      block_size_(block_size),
      byteSize_(capacity * block_size),
      bitmask_nbytes_((this->capacity_ + 7) >> 3),
      // Storage Managers View
      storage_managers_(storage_managers),
      // Initialize the bitmasks as all 0s
      mapped_bitmask_(bitmask_nbytes_),
      pinned_bitmask_(bitmask_nbytes_),
      // Start off priority hints as 1s
      priority_hint_(bitmask_nbytes_, 0xff),
      // Initialize vector of swips
      swips_(capacity, nullptr) {
    // Allocate the address range using mmap.
    // Create a private (i.e. only visible to this process) anonymous (i.e. not
    // backed by a physical file) mapping.
    // Ref: https://man7.org/linux/man-pages/man2/mmap.2.html
    // We use MAP_NORESERVE which doesn't reserve swap space up front.
    // It will reserve swap space lazily when it needs it.
    // This is fine for our use-case since we're mapping a large
    // address space up front. If we reserve swap space, it will
    // block other applications (e.g. Spark in our unit tests) from
    // being able to allocate memory.
    // Ref:
    // https://unix.stackexchange.com/questions/571043/what-is-lazy-swap-reservation
    // https://man7.org/linux/man-pages/man5/proc.5.html (see the
    // /proc/sys/vm/overcommit_memory section)
    void* ptr = mmap(/*addr*/ nullptr, this->byteSize_,
                     /*We need both read/write access*/ PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, /*fd*/ -1,
                     /*offset*/ 0);
    if (ptr == MAP_FAILED || ptr == nullptr) {
        throw std::runtime_error(
            std::string("Could not allocate memory for SizeClass ") +
            std::to_string(block_size) + std::string(". Failed with errno: ") +
            std::strerror(errno) + std::string("."));
    }
    this->address_ = static_cast<uint8_t*>(ptr);
}

SizeClass::~SizeClass() {
    // Unmap the allocated address range.
    munmap(this->address_, this->byteSize_);
}

bool SizeClass::isInRange(uint8_t* ptr) const {
    if ((ptr >= this->address_) && (ptr < (this->address_ + this->byteSize_))) {
        if ((ptr - this->address_) % this->block_size_ != 0) {
            throw std::runtime_error(
                "Pointer is in SizeClass but not at a frame boundary.");
        }
        return true;
    }
    return false;
}

uint8_t* SizeClass::getFrameAddress(uint64_t idx) const {
    if (idx >= this->capacity_) {
        throw std::runtime_error("Frame index is out of bounds.");
    }
    return this->address_ + (idx * this->block_size_);
}

inline void SizeClass::markFrameAsPinned(uint64_t idx) {
    if (!::arrow::bit_util::GetBit(this->mapped_bitmask_.data(), idx)) {
        throw std::runtime_error("Cannot pin an unmapped frame.");
    }
    ::arrow::bit_util::SetBitTo(this->pinned_bitmask_.data(), idx, true);
}

inline void SizeClass::markFrameAsUnpinned(uint64_t idx) {
    if (!::arrow::bit_util::GetBit(this->mapped_bitmask_.data(), idx)) {
        throw std::runtime_error("Cannot unpin an unmapped frame.");
    }
    ::arrow::bit_util::SetBitTo(this->pinned_bitmask_.data(), idx, false);
}

inline void SizeClass::markFrameAsMapped(uint64_t idx) {
    ::arrow::bit_util::SetBitTo(this->mapped_bitmask_.data(), idx, true);
}

inline void SizeClass::markFrameAsUnmapped(uint64_t idx) {
    ::arrow::bit_util::SetBitTo(this->mapped_bitmask_.data(), idx, false);
}

bool SizeClass::isFrameMapped(uint64_t idx) const {
    return ::arrow::bit_util::GetBit(this->mapped_bitmask_.data(), idx);
}

bool SizeClass::isFramePinned(uint64_t idx) const {
    return ::arrow::bit_util::GetBit(this->pinned_bitmask_.data(), idx);
}

uint8_t** SizeClass::getSwip(uint64_t idx) const { return this->swips_[idx]; }

uint64_t SizeClass::getFrameIndex(uint8_t* ptr) const {
    if (!this->isInRange(ptr)) {
        throw std::runtime_error("Pointer is not in size-class");
    }
    return (uint64_t)((ptr - this->address_) / this->block_size_);
}

void SizeClass::adviseAwayFrame(uint64_t idx) {
    // Ref: https://man7.org/linux/man-pages/man2/madvise.2.html
    int madvise_out =
        ::madvise(this->getFrameAddress(idx), this->block_size_, MADV_DONTNEED);
    if (madvise_out < 0) {
        throw std::runtime_error(std::string("madvise returned errno: ") +
                                 std::strerror(errno));
    }
}

int64_t SizeClass::findUnmappedFrame() noexcept {
    for (size_t i = 0; i < this->bitmask_nbytes_; i++) {
        if (this->mapped_bitmask_[i] != static_cast<uint8_t>(0xff)) {
            // There's a free bit in this byte.
            // If this is the last byte, this may be a false
            // positive.
            uint64_t frame_idx = (8 * i);
            for (size_t j = 0; j < 8; j++) {
                if (!::arrow::bit_util::GetBit(&this->mapped_bitmask_[i], j) &&
                    frame_idx < this->capacity_) {
                    return frame_idx;
                }
                frame_idx++;
            }
        }
    }
    // Return -1 if not found.
    return -1;
}

int64_t SizeClass::AllocateFrame(OwningSwip swip) {
    int64_t frame_idx = this->findUnmappedFrame();
    if (frame_idx == -1) {
        return -1;
    }
    // Mark the frame as mapped
    this->markFrameAsMapped(frame_idx);
    // Mark the frame as pinned (default behavior)
    this->markFrameAsPinned(frame_idx);
    // Assign the swip (if there is one)
    this->swips_[frame_idx] = swip;
    return frame_idx;
}

void SizeClass::FreeFrame(uint64_t idx) {
    if (idx >= this->capacity_) {
        throw std::runtime_error("FreeFrame: Frame Index is out of bounds!");
    }
    // Advise away the frame
    this->adviseAwayFrame(idx);
    // Mark the frame as unpinned (in case it was pinned)
    this->markFrameAsUnpinned(idx);
    // Mark the frame as unmapped
    this->markFrameAsUnmapped(idx);
    // Reset the swip (if there was one)
    this->swips_[idx] = nullptr;
}

void SizeClass::FreeFrame(uint8_t* ptr) {
    uint64_t frame_idx = this->getFrameIndex(ptr);
    this->FreeFrame(frame_idx);
}

void SizeClass::PinFrame(uint64_t idx) {
    if (idx >= this->capacity_) {
        throw std::runtime_error("PinFrame: Frame Index is out of bounds!");
    }

    this->markFrameAsPinned(idx);
}

void SizeClass::UnpinFrame(uint64_t idx) {
    if (idx >= this->capacity_) {
        throw std::runtime_error("UnpinFrame: Frame Index is out of bounds!");
    }

    this->markFrameAsUnpinned(idx);
}

arrow::Status SizeClass::EvictFrame(uint64_t idx) {
    if (idx >= this->capacity_) {
        throw std::runtime_error("EvictFrame: Frame Index is out of bounds!");
    }
    if (this->isFramePinned(idx)) {
        throw std::runtime_error("EvictFrame: Frame is not unpinned!");
    }

    auto ptr = this->getFrameAddress(idx);
    int64_t size = static_cast<int64_t>(this->getBlockSize());

    arrow::Result<uint64_t> block_id = arrow::Status::OutOfMemory(
        "No storage locations provided to evict to.");
    uint8_t manager_id;
    for (manager_id = 0; manager_id < this->storage_managers_.size();
         manager_id++) {
        block_id = this->storage_managers_[manager_id]->WriteBlock(ptr, size);
        if (block_id.ok()) {
            break;
        }
    }

    if (!block_id.ok()) {
        return block_id.status();
    }

    // Construct the Swip
    OwningSwip swip = this->swips_[idx];
    *swip = construct_unswizzled_swip(this->idx_, manager_id,
                                      block_id.ValueOrDie());
    this->swips_[idx] = nullptr;

    // Mark the frame as unmapped
    this->markFrameAsUnmapped(idx);
    // Advise away the frame
    this->adviseAwayFrame(idx);
    return arrow::Status::OK();
}

arrow::Status SizeClass::ReadbackToFrame(OwningSwip swip, uint64_t frame_idx,
                                         uint64_t block_idx,
                                         uint8_t manager_idx) {
    int64_t size = static_cast<int64_t>(this->getBlockSize());

    auto ptr = this->getFrameAddress(frame_idx);
    CHECK_ARROW_MEM_RET(
        this->storage_managers_[manager_idx]->ReadBlock(block_idx, size, ptr),
        "SizeClass::ReadbackToFrame: Failed to Read Spill Block from Storage "
        "-");

    // Mark the frame as mapped
    this->markFrameAsMapped(frame_idx);
    // Mark the frame as pinned (default behavior)
    this->markFrameAsPinned(frame_idx);
    // Assign the Swip
    *swip = ptr;
    this->swips_[frame_idx] = swip;

    return arrow::Status::OK();
}

//// StorageOptions and BufferPoolOptions

/**
 * @brief Get the number of ranks on this node and the
 * current rank's position.
 *
 * We do this by creating sub-communicators based
 * on shared-memory. This is a collective operation
 * and therefore all ranks must call it.
 */
static std::tuple<int, int> dist_get_ranks_on_node() {
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

std::shared_ptr<StorageOptions> StorageOptions::Defaults(uint8_t tier) {
    std::shared_ptr<StorageOptions> options = nullptr;
    // TODO: Use fmt::format when available

    // Get the comma-delimited list of drive locations for this storage option
    std::vector<std::string> locations;
    auto location_env_str = std::string("BODO_BUFFER_POOL_STORAGE_CONFIG_") +
                            std::to_string(tier) + std::string("_DRIVES");
    if (const char* location_env_ = std::getenv(location_env_str.c_str())) {
        std::stringstream ss(location_env_);
        while (ss.good()) {
            std::string substr;
            std::getline(ss, substr, ',');

            // TODO: Check for StorageType
            if (substr.starts_with("s3://") || substr.starts_with("abfs://")) {
                std::cerr << "Spilling to S3 or ABFS is not supported yet"
                          << std::endl;
                return nullptr;
            }

            locations.push_back(substr);
        }
    } else {
        // If no location env, assume that no storage option was
        // defined for this tier.
        return nullptr;
    }

    // Compute the number of bytes available for this storage location
    auto gb_available_env_str =
        std::string("BODO_BUFFER_POOL_STORAGE_CONFIG_") + std::to_string(tier) +
        std::string("_SPACE_PER_DRIVE_GiB");

    if (const char* gb_available_env_ =
            std::getenv(gb_available_env_str.c_str())) {
        // Create Options
        options = std::make_shared<StorageOptions>();

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

BufferPoolOptions BufferPoolOptions::Defaults() {
    BufferPoolOptions options;

    // Disable Storage Manager Parsing
    // Useful for debugging purposes
    const char* disable_spilling_env_ =
        std::getenv("BODO_BUFFER_POOL_DISABLE_SPILLING");
    if (!disable_spilling_env_ || std::strcmp(disable_spilling_env_, "1")) {
        // Parse Storage Managers
        for (uint8_t i = 1; i <= MAX_NUM_STORAGE_MANAGERS; i++) {
            auto storage_option = StorageOptions::Defaults(i);
            if (storage_option != nullptr) {
                options.storage_options.push_back(storage_option);
            } else {
                break;
            }
        }
    }

    // Read memory_size from env_var if provided.
    // If env var is not set, we will get the memory
    // information from the OS.
    if (char* memory_size_env_ =
            std::getenv("BODO_BUFFER_POOL_MEMORY_SIZE_MiB")) {
        options.memory_size = std::stoi(memory_size_env_);
    } else {
        // Fraction of total memory we should actually assign to the buffer
        // pool. We will read this from an environment variable if it's set, but
        // will default to 95% if spilling is available and enabled, and to 500%
        // otherwise.
        double mem_fraction = options.storage_options.empty() ? 5. : .95;

        // We expect this to be in percentages and not fraction,
        // i.e. it should be set to 45 if we want to use 45% (or 0.45)
        // of the total available space.
        if (char* mem_percent_env_ =
                std::getenv("BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT")) {
            mem_fraction =
                static_cast<double>(std::stoi(mem_percent_env_) / 100.0);
        }

        // Get number of ranks on this node.
        auto [num_ranks_on_node, _] = dist_get_ranks_on_node();
        // Get total memory size of the node.
        size_t mem_on_node =
            static_cast<size_t>(get_total_node_memory() / (1024.0 * 1024.0));

        // Equal allocation of memory for each rank on this node.
        size_t mem_per_rank =
            static_cast<size_t>(mem_on_node / (double)num_ranks_on_node);
        // Set memory_size as mem_fraction of mem_per_rank
        options.memory_size =
            static_cast<uint64_t>(mem_per_rank * mem_fraction);
    }

    if (char* min_size_class_env_ =
            std::getenv("BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB")) {
        options.min_size_class = std::stoi(min_size_class_env_);
    }

    if (char* max_num_size_classes_env_ =
            std::getenv("BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES")) {
        options.max_num_size_classes = std::stoi(max_num_size_classes_env_);
    }

    // BufferPool's equal allocation per rank
    // approach can cause issues for existing Bodo workloads
    // until we have full spill support. In particular,
    // we might preemptively disallow allocations on a rank
    // in case of skew. Thus, we want to be able to attempt
    // allocation even if it's beyond the allocated limit.
    if (const char* ignore_max_limit_env_ =
            std::getenv("BODO_BUFFER_POOL_IGNORE_MAX_ALLOCATION_LIMIT")) {
        options.ignore_max_limit_during_allocation =
            !std::strcmp(ignore_max_limit_env_, "1");
    }

    return options;
}

//// BufferPool

BufferPool::BufferPool(const BufferPoolOptions& options)
    : options_(std::move(options)),
      // Convert MiB to bytes
      memory_size_bytes_(options.memory_size * 1024 * 1024) {
    for (auto storage_option : this->options_.storage_options) {
        std::unique_ptr<StorageManager> manager;

        switch (storage_option->type) {
            case StorageType::Local:
                manager = std::make_unique<LocalStorageManager>(storage_option);
                break;
        }

        CHECK_ARROW_MEM(manager->Initialize(),
                        "Failed to Initialize Storage Manager:");
        // std::move is needed to push a unique_ptr
        this->storage_managers_.push_back(std::move(manager));
    }

    this->Initialize();
}

BufferPool::BufferPool() : BufferPool(BufferPoolOptions::Defaults()) {}

/**
 * @brief Find the highest power of 2 that is lesser than or equal
 * to N. Note that 0 returns 0.
 *
 * Ref:
 * https://www.geeksforgeeks.org/highest-power-2-less-equal-given-number/
 *
 * If we ever needed to find the lowest power of 2 that is greater than
 * or equal to N, look at:
 * https://www.geeksforgeeks.org/smallest-power-of-2-greater-than-or-equal-to-n/
 *
 */
static inline int64_t highest_power_of_2(int64_t N) {
    if (N <= 0) {
        throw std::runtime_error("highest_power_of_2: N must be >0");
    }
    // if N is a power of two simply return it
    if (!(N & (N - 1)))
        return N;
    // else set only the most significant bit
    return 0x8000000000000000UL >> (__builtin_clzll(N));
}

void BufferPool::Initialize() {
    // Verify that min-size-class is a power of 2
    // TODO Move to BufferPoolOptions constructor.
    if (((this->options_.min_size_class &
          (this->options_.min_size_class - 1)) != 0) ||
        (this->options_.min_size_class == 0)) {
        throw std::runtime_error("min_size_class must be a power of 2");
    }

    // Convert from KiB to bytes
    uint64_t min_size_class_bytes = this->options_.min_size_class * 1024;

    // Calculate max possible size-class based on available memory
    // (i.e. highest power of 2 lower or equal to this->memory_size_bytes_).
    uint64_t max_size_class_possible_bytes =
        static_cast<uint64_t>(highest_power_of_2(this->memory_size_bytes_));

    if (min_size_class_bytes > max_size_class_possible_bytes) {
        throw std::runtime_error(
            "min_size_class is larger than available memory!");
    }

    // Based on this, the max size classes possible is:
    uint64_t max_num_size_classes =
        static_cast<uint64_t>(std::log2(max_size_class_possible_bytes)) -
        static_cast<uint64_t>(std::log2(min_size_class_bytes)) + 1;

    // Take the minumum of this, and the number specified in options_
    // to get the actual number of size-classes.
    // The absolute max is 63 (since it needs to be encodable in 6 bits).
    uint8_t num_size_classes = static_cast<uint8_t>(std::min(
        std::min(this->options_.max_num_size_classes, max_num_size_classes),
        (uint64_t)63));

    // Create the SizeClass objects
    this->size_classes_.reserve(num_size_classes);
    this->size_class_bytes_.reserve(num_size_classes);
    uint64_t size_class_bytes_i = min_size_class_bytes;
    for (uint8_t i = 0; i < num_size_classes; i++) {
        uint64_t num_blocks = static_cast<uint64_t>(this->memory_size_bytes_ /
                                                    size_class_bytes_i);
        this->size_classes_.emplace_back(
            std::make_unique<SizeClass>(i, std::span(this->storage_managers_),
                                        num_blocks, size_class_bytes_i));
        this->size_class_bytes_.emplace_back(size_class_bytes_i);
        size_class_bytes_i *= 2;
    }

    this->malloc_threshold_ =
        static_cast<uint64_t>(MALLOC_THRESHOLD_RATIO * min_size_class_bytes);
}

size_t BufferPool::num_size_classes() const {
    return this->size_classes_.size();
}

int64_t BufferPool::size_align(int64_t size, int64_t alignment) const {
    const auto remainder = size % alignment;
    return (remainder == 0) ? size : (size + alignment - remainder);
}

int64_t BufferPool::max_memory() const { return this->stats_.max_memory(); }

int64_t BufferPool::bytes_allocated() const {
    return this->stats_.bytes_allocated();
}

uint64_t BufferPool::bytes_pinned() const { return this->bytes_pinned_.load(); }

inline void BufferPool::update_pinned_bytes(int64_t diff) {
    this->bytes_pinned_.fetch_add(diff);
}

std::string BufferPool::backend_name() const { return "bodo"; }

bool BufferPool::is_spilling_enabled() const {
    return !this->options_.storage_options.empty();
}

int64_t BufferPool::find_size_class_idx(int64_t size) const {
    if (static_cast<uint64_t>(size) > this->size_class_bytes_.back()) {
        return -1;
    }
    return std::distance(this->size_class_bytes_.begin(),
                         std::lower_bound(this->size_class_bytes_.begin(),
                                          this->size_class_bytes_.end(), size));
}

// static
inline void BufferPool::zero_padding(uint8_t* ptr, size_t size,
                                     size_t capacity) {
    memset(ptr + size, 0, static_cast<size_t>(capacity - size));
}

arrow::Status BufferPool::evict(uint64_t size_class_idx) {
    // Attempt to evict smaller size classes
    uint64_t bytes_rem = this->size_class_bytes_[size_class_idx];
    std::vector<uint64_t> evicting_frames_size_class;
    std::vector<uint64_t> evicting_frame_idxs;

    for (int64_t i = size_class_idx; i >= 0; i--) {
        auto& size_class = this->size_classes_[i];
        auto num_frames = size_class->getNumBlocks();
        auto size_bytes = this->size_class_bytes_[i];

        for (uint64_t j = 0; j < num_frames; j++) {
            if (bytes_rem <= 0) {
                break;
            }

            if (size_class->isFrameMapped(j) && !size_class->isFramePinned(j)) {
                bytes_rem -= size_bytes;
                evicting_frames_size_class.push_back(i);
                evicting_frame_idxs.push_back(j);
            }
        }

        if (bytes_rem <= 0) {
            break;
        }
    }

    if (bytes_rem <= 0) {
        for (uint32_t i = 0; i < evicting_frame_idxs.size(); i++) {
            auto evict_status =
                this->size_classes_[evicting_frames_size_class[i]]->EvictFrame(
                    evicting_frame_idxs[i]);
            if (!evict_status.ok()) {
                return evict_status;
            }

            this->stats_.UpdateAllocatedBytes(
                -this->size_class_bytes_[evicting_frames_size_class[i]]);
        }

        return arrow::Status::OK();
    }

    // There are not enough small frames to evict to free up enough
    // space. Thus, we need to find one larger frame to evict instead
    for (auto i = size_class_idx + 1; i < this->num_size_classes(); i++) {
        auto& size_class = this->size_classes_[i];
        auto num_frames = size_class->getNumBlocks();
        for (uint64_t j = 0; j < num_frames; j++) {
            if (size_class->isFrameMapped(j) && !size_class->isFramePinned(j)) {
                auto evict_status = size_class->EvictFrame(j);
                if (!evict_status.ok()) {
                    return evict_status;
                }

                this->stats_.UpdateAllocatedBytes(-this->size_class_bytes_[i]);
                return arrow::Status::OK();
            }
        }
    }

    // We couldn't find enough smaller frames
    // or 1 larger frame to evict
    return arrow::Status::OutOfMemory(
        "Unable to evict enough frames to free up the required space");
}

::arrow::Status BufferPool::Allocate(int64_t size, int64_t alignment,
                                     uint8_t** out) {
    if (size < 0) {
        return ::arrow::Status::Invalid("Negative allocation size requested.");
    }

    if ((alignment <= 0) || ((alignment & (alignment - 1)) != 0)) {
        return ::arrow::Status::Invalid(
            "Alignment must be a positive number and a power of 2.");
    }

    // Copied from Arrow (they are probably just being conservative for
    // compatibility with 32-bit architectures).
    if (static_cast<uint64_t>(size) >= std::numeric_limits<size_t>::max()) {
        return ::arrow::Status::OutOfMemory("malloc size overflows size_t");
    }

    // If size 0 allocation, point to a pre-defined area (same as Arrow)
    if (size == 0) {
        *out = kZeroSizeArea;
        return ::arrow::Status::OK();
    }

    const int64_t aligned_size = this->size_align(size, alignment);

    // Get a lock on the BufferPool state for the duration
    // of this function. 'scoped_lock' guarantees that the
    // lock will be released when the function ends (even if
    // there's an exception).
    std::scoped_lock lock(this->mtx_);

    if (aligned_size <= static_cast<int64_t>(this->malloc_threshold_)) {
        // Use malloc

        // If non-pinned memory is less than needed, immediately fail
        if (!this->options_.ignore_max_limit_during_allocation &&
            aligned_size > static_cast<int64_t>(this->memory_size_bytes_ -
                                                this->bytes_pinned())) {
            return ::arrow::Status::OutOfMemory(
                "Allocation failed. Not enough space in the buffer pool.");
        }

        // If available memory is less than needed, start spilling
        int64_t bytes_available_in_mem =
            static_cast<int64_t>(this->memory_size_bytes_) -
            this->bytes_allocated();
        if (!this->options_.ignore_max_limit_during_allocation &&
            aligned_size > bytes_available_in_mem) {
            auto evict_status = this->evict(0);
            if (!evict_status.ok()) {
                if (evict_status.IsOutOfMemory()) {
                    return evict_status.WithMessage("Allocation failed. " +
                                                    evict_status.message());
                } else {
                    return evict_status;
                }
            }
        }

        // There's essentially two options:
        // 1. posix_memalign/memalign: This is what Arrow uses
        //    (https://github.com/apache/arrow/blob/ea6875fd2a3ac66547a9a33c5506da94f3ff07f2/cpp/src/arrow/memory_pool.cc#L318)
        // 2. aligned_alloc: This is what Velox uses
        //    (https://github.com/facebookincubator/velox/blob/8324ac7f1839db009def00e7450f38c2591dd4bb/velox/common/memory/MmapAllocator.cpp#L371)
        //     and seems to be what is generally recommended. The only
        //     requirement is that the allocation size must be a multiple of the
        //     requested alignment, which we do in size_align already.
        // malloc does 16B alignment by default, so we can use it for
        // those cases.
        // All these allocations can be free-d using 'free'.
        void* result = alignment > kMinAlignment
                           ? ::aligned_alloc(alignment, aligned_size)
                           : ::malloc(aligned_size);
        if (result == nullptr) {
            // XXX This is an unlikely branch, so it would
            // be good to indicate that to the compiler
            // similar to how Velox does it using "folly".
            return ::arrow::Status::UnknownError(
                "Failed to allocate required bytes.");
        }
        *out = static_cast<uint8_t*>(result);

        // Update statistics
        this->update_pinned_bytes(aligned_size);
        this->stats_.UpdateAllocatedBytes(aligned_size);

        // Zero-pad to match Arrow
        // https://github.com/apache/arrow/blob/5b2fbade23eda9bc95b1e3854b19efff177cd0bd/cpp/src/arrow/memory_pool.cc#L932
        // https://github.com/apache/arrow/blob/5b2fbade23eda9bc95b1e3854b19efff177cd0bd/cpp/src/arrow/buffer.h#L125
        this->zero_padding(static_cast<uint8_t*>(result), size, aligned_size);

    } else {
        // Mmap-ed memory is always page (typically 4096B) aligned
        // (https://stackoverflow.com/questions/42259495/does-mmap-return-aligned-pointer-values).
        const static long page_size = sysconf(_SC_PAGE_SIZE);
        if (alignment > page_size) {
            return ::arrow::Status::Invalid(
                "Requested alignment higher than max supported alignment.");
        }
        // Use one of the mmap-ed buffer frames.
        int64_t size_class_idx = this->find_size_class_idx(aligned_size);

        if (size_class_idx == -1) {
            return ::arrow::Status::Invalid(
                "Request allocation size is larger than the largest block-size "
                "available!");
            // XXX If number of size classes was artificially set to too low
            // (through BufferPoolOptions), we also need to have ability to
            // allocate contiguous and non-contiguous blocks for cases where the
            // request is a valid size but we just don't have a size-class for
            // it. This would be similar to Velox.
        }

        uint64_t size_class_bytes = this->size_class_bytes_[size_class_idx];

        // If non-pinned memory is less than needed, immediately fail
        if (!this->options_.ignore_max_limit_during_allocation &&
            size_class_bytes >
                (this->memory_size_bytes_ - this->bytes_pinned())) {
            return ::arrow::Status::OutOfMemory(
                "Allocation failed. Not enough space in the buffer pool.");
        }

        // If available memory is less than needed, start spilling
        uint64_t bytes_available_in_mem =
            this->memory_size_bytes_ -
            static_cast<uint64_t>(this->bytes_allocated());
        if (!this->options_.ignore_max_limit_during_allocation &&
            size_class_bytes > bytes_available_in_mem) {
            int64_t rem_bytes = size_class_bytes - bytes_available_in_mem;
            int64_t rem_class_idx = this->find_size_class_idx(rem_bytes);
            auto evict_status = this->evict(rem_class_idx);
            if (!evict_status.ok()) {
                if (evict_status.IsOutOfMemory()) {
                    return evict_status.WithMessage("Allocation failed. " +
                                                    evict_status.message());
                } else {
                    return evict_status;
                }
            }
        }

        // Allocate in the identified size-class.
        // Due to the previous check, we're guaranteed to be able to find a
        // frame. Proof: We allocated memory_size // block_size many blocks. Say
        // all frames were taken, that would mean that block_size * num_blocks
        // many bytes are allocated. Allocating another block would mean that
        // total memory usage would be greater than memory_size, but we already
        // checked that there's sufficient memory available for this allocation.
        int64_t frame_idx =
            this->size_classes_[size_class_idx]->AllocateFrame(out);
        if (frame_idx == -1) {
            return ::arrow::Status::OutOfMemory(
                "Could not find an empty frame of required size!");
        }

        *out = this->size_classes_[size_class_idx]->getFrameAddress(frame_idx);

        // We don't need to zero-pad here. mmap zero-initializes its pages when
        // using anonymous mapping
        // (https://man7.org/linux/man-pages/man2/mmap.2.html). Even in the case
        // that the page was "advised away", it will "zero-fill-on-demand pages
        // for anonymous private mappings" when the page is accessed again.
        // (https://man7.org/linux/man-pages/man2/madvise.2.html).
        // The only time we need to zero pad ourselves is when we re-use a page
        // without advising it away.
        // TODO When adding eviction support and re-using a "mapped" page,
        // make sure to zero pad the buffer.

        // Update statistics
        this->update_pinned_bytes(size_class_bytes);
        this->stats_.UpdateAllocatedBytes(size_class_bytes);
    }

    // Add debug markers.
    // See notes here about why these memory markers are useful:
    // https://stackoverflow.com/questions/370195/when-and-why-will-a-compiler-initialise-memory-to-0xcd-0xdd-etc-on-malloc-fre
    // Only fill up a couple cachelines to minimize overhead.
    memset(*out, 0xCB, std::min(size, (int64_t)256));

    return ::arrow::Status::OK();
}

std::tuple<bool, int64_t, int64_t, int64_t> BufferPool::get_alloc_details(
    uint8_t* buffer, int64_t size, int64_t alignment) const {
    if (size == -1) {
        // If we don't know the size, we need to find the buffer frame
        // (if there is one).
        for (size_t i = 0; i < this->num_size_classes(); i++) {
            if (this->size_classes_[i]->isInRange(buffer)) {
                uint64_t frame_idx =
                    this->size_classes_[i]->getFrameIndex(buffer);
                return std::make_tuple(true, i, frame_idx,
                                       this->size_class_bytes_[i]);
            }
        }
        // If no matches, it must be from malloc
        return std::make_tuple(false, -1, -1, -1);
    } else {
        // To match Allocate, get aligned_size before checking
        // for the size class.
        const int64_t aligned_size = this->size_align(size, alignment);
        if (aligned_size <= static_cast<int64_t>(this->malloc_threshold_)) {
            return std::make_tuple(false, -1, -1, aligned_size);
        } else {
            int64_t size_class_idx = this->find_size_class_idx(aligned_size);
            if (size_class_idx == -1) {
                // TODO Add compiler hint that this branch
                // is unlikely.
                throw std::runtime_error(
                    "Provided size doesn't match any of the "
                    "size-classes!");
            }
            return std::make_tuple(
                true, size_class_idx,
                this->size_classes_[size_class_idx]->getFrameIndex(buffer),
                this->size_class_bytes_[size_class_idx]);
        }
    }
}

void BufferPool::free_helper(uint8_t* ptr, bool is_mmap_alloc,
                             int64_t size_class_idx, int64_t frame_idx,
                             int64_t size_aligned) {
    bool frame_pinned;
    if (is_mmap_alloc) {
        frame_pinned =
            this->size_classes_[size_class_idx]->isFramePinned(frame_idx);
        this->size_classes_[size_class_idx]->FreeFrame(frame_idx);
    } else {
        frame_pinned = true;
        ::free(ptr);
    }

    if (size_aligned != -1) {
        if (frame_pinned) {
            this->update_pinned_bytes(-size_aligned);
        }
        this->stats_.UpdateAllocatedBytes(-size_aligned);
    }
}

void BufferPool::Free(uint8_t* buffer, int64_t size, int64_t alignment) {
    // Handle zero case.
    if (buffer == kZeroSizeArea) {
        if (size > 0) {  // neither 0 nor -1 (unknown)
            // TODO: Add compiler hint that this path is not likely.
            std::cerr << "Expected size of allocation pointing to ZeroArea to "
                         "be 0 or unknown."
                      << std::endl;
        }
        return;
    }
    if (size == 0) {
        // Should never happen, but just in case.
        return;
    }

    // Get a lock on the BufferPool state for the duration
    // of this function. 'scoped_lock' guarantees that the
    // lock will be released when the function ends (even if
    // there's an exception).
    std::scoped_lock lock(this->mtx_);

    // Evicted frames should be deleted directly from disk
    auto swip_info = extract_swip_ptr(buffer);
    if (swip_info.has_value()) {
        auto [size_class_idx, storage_manager_idx, block_id] =
            swip_info.value();
        auto status = this->storage_managers_[storage_manager_idx]->DeleteBlock(
            block_id, this->size_class_bytes_[size_class_idx]);

        // For simplicity of free, we will print any failures as warnings for
        // now
        if (!status.ok()) {
            std::cerr << "Free Failed. " << status.ToString() << std::endl;
        }
        return;
    }

    // Add debug markers to indicate dead memory only for frames in memory.
    memset(buffer, 0xDE, std::min(size, (int64_t)256));

    auto [is_mmap_alloc, size_class_idx, frame_idx, size_freed] =
        this->get_alloc_details(buffer, size, alignment);

    this->free_helper(buffer, is_mmap_alloc, size_class_idx, frame_idx,
                      size_freed);
    // XXX In the case where we still don't know the size of the allocation and
    // it was through malloc, we can't update stats_. Should we just enforce
    // that size be provided?
}

bool BufferPool::IsPinned(uint8_t* buffer, int64_t size,
                          int64_t alignment) const {
    if (is_unswizzled(buffer)) {
        // If it's been evicted to disk, it must be unpinned.
        return false;
    }
    auto [is_mmap_alloc, size_class_idx, frame_idx, _] =
        this->get_alloc_details(buffer, size, alignment);
    if (is_mmap_alloc) {
        return this->size_classes_[size_class_idx]->isFramePinned(frame_idx);
    }
    // Malloc allocations are always pinned:
    return true;
}

uint64_t BufferPool::get_memory_size_bytes() const {
    return this->memory_size_bytes_;
}

::arrow::Status BufferPool::Reallocate(int64_t old_size, int64_t new_size,
                                       int64_t alignment, uint8_t** ptr) {
    if (new_size < 0) {
        return ::arrow::Status::Invalid(
            "Negative reallocation size requested.");
    }
    if (static_cast<uint64_t>(new_size) >= std::numeric_limits<size_t>::max()) {
        return ::arrow::Status::OutOfMemory("realloc overflows size_t");
    }

    uint8_t* old_memory_ptr = *ptr;

    if (old_memory_ptr == kZeroSizeArea) {
        if (old_size > 0) {  // neither 0 nor -1 (unknown)
            // TODO: Add compiler hint that this path is not likely.
            std::cerr << "Expected size of allocation pointing to ZeroArea to "
                         "be 0 or unknown."
                      << std::endl;
        }
        return this->Allocate(new_size, alignment, ptr);
    }

    if (new_size == 0) {
        this->Free(old_memory_ptr, old_size, alignment);
        *ptr = kZeroSizeArea;
        return ::arrow::Status::OK();
    }

    if (is_unswizzled(old_memory_ptr)) {
        auto status = this->Pin(ptr);
        if (!status.IsOutOfMemory()) {
            return arrow::Status::OutOfMemory(
                "Reallocate failed. Not enough space to pin old allocation.");
        } else if (!status.ok()) {
            return status;
        }
    }

    auto [is_mmap_alloc, size_class_idx, frame_idx, old_size_aligned] =
        this->get_alloc_details(old_memory_ptr, old_size, alignment);

    // In case of an mmap frame: pin it if it isn't already. We either
    // need it to be in memory for the memcpy or even if we end up re-using
    // the same frame, Reallocate needs to pin the frame.
    if (is_mmap_alloc) {
        if (!this->size_classes_[size_class_idx]->isFramePinned(frame_idx)) {
            this->size_classes_[size_class_idx]->PinFrame(frame_idx);
            this->update_pinned_bytes(this->size_class_bytes_[size_class_idx]);
        }
    }
    // In case of an mmap frame: if new_size still fits, it's a NOP.
    // Note that we only do this when new_size >= old_size, because
    // otherwise we should change the SizeClass (to make sure assumptions
    // in other places still hold).
    // In case of mmap-allocation, old_size_aligned is the size of the block.
    // TODO To handle the size reduction case, change logic to check that the
    // size-class remains the same (and that it doesn't drop into the malloc
    // bucket).
    if (is_mmap_alloc && (new_size >= old_size) &&
        (new_size <= old_size_aligned)) {
        return ::arrow::Status::OK();
    }

    // Allocate new_size
    // Point ptr to the new memory. We have a pointer to old
    // memory in old_memory_ptr that we can use for the memcpy.
    // Since we pinned the old frame, this allocate won't evict
    // the old block which is required for the memcpy.
    CHECK_ARROW_MEM(this->Allocate(new_size, alignment, ptr),
                    "Allocation failed!");

    // Get a lock on the BufferPool state for the duration
    // of this function. 'scoped_lock' guarantees that the
    // lock will be released when the function ends (even if
    // there's an exception).
    std::scoped_lock lock(this->mtx_);

    uint8_t* new_memory_ptr = *ptr;

    // Copy over the contents
    memcpy(new_memory_ptr, old_memory_ptr,
           static_cast<size_t>(std::min(new_size, old_size)));

    // Free original memory (re-use information from get_alloc_details output)
    this->free_helper(old_memory_ptr, is_mmap_alloc, size_class_idx, frame_idx,
                      old_size_aligned);

    return ::arrow::Status::OK();
}

::arrow::Status BufferPool::Pin(uint8_t** ptr, int64_t size,
                                int64_t alignment) {
    // Handle zero case.
    if (*ptr == kZeroSizeArea) {
        return arrow::Status::OK();
    }

    // Get a lock on the BufferPool state for the duration
    // of this function. 'scoped_lock' guarantees that the
    // lock will be released when the function ends (even if
    // there's an exception).
    std::scoped_lock lock(this->mtx_);

    auto swip_info = extract_swip_ptr(*ptr);

    if (swip_info.has_value()) {
        // If ptr is unswizzled, then we have to load the
        // associated block into memory, and then mark as
        // pinned
        auto [size_class_idx, storage_manager_idx, block_id] =
            swip_info.value();
        auto& size_class = this->size_classes_[size_class_idx];
        uint64_t block_bytes = this->size_class_bytes_[size_class_idx];

        // Determine if there is enough space in memory to
        // read-back evicted block
        // If non-pinned memory is less than needed, immediately fail
        if (block_bytes > (this->memory_size_bytes_ - this->bytes_pinned())) {
            return ::arrow::Status::OutOfMemory(
                "Pin failed. Not enough space in the buffer pool.");
        }

        if (block_bytes >
            (this->memory_size_bytes_ - this->bytes_allocated())) {
            int64_t rem_bytes =
                block_bytes - (static_cast<int64_t>(this->memory_size_bytes_) -
                               this->bytes_allocated());
            int64_t rem_class_idx = this->find_size_class_idx(rem_bytes);
            CHECK_ARROW_MEM_RET(this->evict(rem_class_idx),
                                "Pin failed. Error when evicting frame:");
        }

        // Find an available frame in the size class
        int64_t frame_idx = size_class->findUnmappedFrame();
        if (frame_idx == -1) {
            // TODO: Should be impossible at this point, fatal error
            return ::arrow::Status::OutOfMemory(
                "Pin failed. Unable to find available frame");
        }

        // Load Block from Storage into Frame
        CHECK_ARROW_MEM_RET(
            size_class->ReadbackToFrame(ptr, frame_idx, block_id,
                                        storage_manager_idx),
            "Pin failed. Error when unevicting pinned location.");

        this->update_pinned_bytes(block_bytes);
        this->stats_.UpdateAllocatedBytes(block_bytes);
        return arrow::Status::OK();

    } else {
        // If ptr is swizzled, then just need to mark the
        // associated frame as pinned
        auto [is_pool_alloc, size_class_idx, frame_idx, _] =
            this->get_alloc_details(*ptr, size, alignment);

        // If allocated through malloc, Pin is No-op
        if (!is_pool_alloc) {
            return arrow::Status::OK();
        }

        if (!this->size_classes_[size_class_idx]->isFramePinned(frame_idx)) {
            this->size_classes_[size_class_idx]->PinFrame(frame_idx);
            this->update_pinned_bytes(this->size_class_bytes_[size_class_idx]);
        }
        return arrow::Status::OK();
    }
}

void BufferPool::Unpin(uint8_t* ptr, int64_t size, int64_t alignment) {
    // Handle zero case.
    if (ptr == kZeroSizeArea) {
        return;
    }

    // Get a lock on the BufferPool state for the duration
    // of this function. 'scoped_lock' guarantees that the
    // lock will be released when the function ends (even if
    // there's an exception).
    std::scoped_lock lock(this->mtx_);

    // If ptr is unswizzled, then it should already be unpinned
    if (is_unswizzled(ptr)) {
        // TODO: Should we include a check?
        return;
    }

    auto [is_pool_alloc, size_class_idx, frame_idx, _] =
        this->get_alloc_details(ptr, size, alignment);

    // If allocated through malloc, Unpin is No-op
    if (!is_pool_alloc) {
        return;
    }

    // If pinned, unpin and update stats, else we don't need to do anything
    if (this->size_classes_[size_class_idx]->isFramePinned(frame_idx)) {
        this->size_classes_[size_class_idx]->UnpinFrame(frame_idx);
        this->update_pinned_bytes(-this->size_class_bytes_[size_class_idx]);
    }
}

SizeClass* BufferPool::GetSizeClass_Unsafe(uint64_t idx) const {
    if (idx > this->size_classes_.size()) {
        throw std::runtime_error("Requested SizeClass doesn't exist.");
    }
    return this->size_classes_[idx].get();
}

uint64_t BufferPool::GetSmallestSizeClassSize() const {
    if (this->size_class_bytes_.size() > 0) {
        return this->size_class_bytes_[0];
    }
    return 0;
}

/// Helper Functions for using BufferPool in Arrow

::arrow::compute::ExecContext* buffer_exec_context(bodo::IBufferPool* pool) {
    using arrow::compute::ExecContext;
    return new ExecContext(pool);
}

::arrow::compute::ExecContext* default_buffer_exec_context() {
    using arrow::compute::ExecContext;

    static auto ctx_ =
        std::make_shared<ExecContext>(bodo::BufferPool::DefaultPtr());
    return ctx_.get();
}

::arrow::io::IOContext buffer_io_context(bodo::IBufferPool* pool) {
    using arrow::io::IOContext;

    return IOContext(pool);
}

::arrow::io::IOContext default_buffer_io_context() {
    using arrow::io::IOContext;

    static IOContext ctx_(bodo::BufferPool::DefaultPtr());
    return ctx_;
}

std::shared_ptr<::arrow::MemoryManager> buffer_memory_manager(
    bodo::IBufferPool* pool) {
    return arrow::CPUDevice::memory_manager(pool);
}

std::shared_ptr<::arrow::MemoryManager> default_buffer_memory_manager() {
    static auto mm_ =
        arrow::CPUDevice::memory_manager(bodo::BufferPool::DefaultPtr());
    return mm_;
}

}  // namespace bodo
