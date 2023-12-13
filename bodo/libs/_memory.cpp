#include "_memory.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>

#ifdef __linux__
// Needed for 'malloc_trim'
#include <malloc.h>
#endif
#include <mpi.h>
#include <sys/mman.h>

#include <fmt/args.h>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include "_utils.h"

#define MAX_NUM_STORAGE_MANAGERS 4

#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(expr, msg, lhs)  \
    {                                           \
        auto res = expr;                        \
        CHECK_ARROW_MEM_RET(res.status(), msg); \
        lhs = std::move(res).ValueOrDie();      \
    }

using namespace std::chrono;

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

//// SizeClass

SizeClass::SizeClass(
    uint8_t idx,
    const std::span<std::unique_ptr<StorageManager>> storage_managers,
    size_t capacity, size_t block_size, bool spill_on_unpin, bool move_on_unpin,
    bool tracing_mode)
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
      swips_(capacity, nullptr),
      spill_on_unpin_(spill_on_unpin),
      move_on_unpin_(move_on_unpin),
      tracing_mode_(tracing_mode) {
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
        throw std::runtime_error(std::string("SizeClass::SizeClass: Could not "
                                             "allocate memory for SizeClass ") +
                                 std::to_string(block_size) +
                                 std::string(". Failed with errno: ") +
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
                "SizeClass::isInRange: Pointer is in SizeClass but not at a "
                "frame boundary.");
        }
        return true;
    }
    return false;
}

uint8_t* SizeClass::getFrameAddress(uint64_t idx) const {
    if (idx >= this->capacity_) {
        throw std::runtime_error("SizeClass::getFrameAddress: Frame index " +
                                 std::to_string(idx) + " is out of bounds.");
    }
    return this->address_ + (idx * this->block_size_);
}

inline void SizeClass::markFrameAsPinned(uint64_t idx) {
    if (!::arrow::bit_util::GetBit(this->mapped_bitmask_.data(), idx)) {
        throw std::runtime_error(
            "SizeClass::markFrameAsPinned: Cannot pin an unmapped frame.");
    }
    ::arrow::bit_util::SetBitTo(this->pinned_bitmask_.data(), idx, true);
}

inline void SizeClass::markFrameAsUnpinned(uint64_t idx) {
    if (!::arrow::bit_util::GetBit(this->mapped_bitmask_.data(), idx)) {
        throw std::runtime_error(
            "SizeClass::markFrameAsUnpinned: Cannot unpin an unmapped frame.");
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
        throw std::runtime_error(
            "SizeClass::getFrameIndex: Pointer is not in size-class");
    }
    return (uint64_t)((ptr - this->address_) / this->block_size_);
}

void SizeClass::adviseAwayFrame(uint64_t idx) {
    auto start = start_now(this->tracing_mode_);

    // Ref: https://man7.org/linux/man-pages/man2/madvise.2.html
    int madvise_out =
        ::madvise(this->getFrameAddress(idx), this->block_size_, MADV_DONTNEED);
    if (madvise_out < 0) {
        throw std::runtime_error(
            std::string(
                "SizeClass::adviseAwayFrame: madvise returned errno: ") +
            std::strerror(errno));
    }

    this->stats_.total_advise_away_calls++;
    if (this->tracing_mode_) {
        milli_double dur = steady_clock::now() - start.value();
        this->stats_.total_advise_away_time += dur;
    }
}

int64_t SizeClass::findUnmappedFrame() noexcept {
    auto start = start_now(this->tracing_mode_);

    for (size_t i = 0; i < this->bitmask_nbytes_; i++) {
        if (this->mapped_bitmask_[i] != static_cast<uint8_t>(0xff)) {
            // There's a free bit in this byte.
            // If this is the last byte, this may be a false
            // positive.
            uint64_t frame_idx = (8 * i);
            for (size_t j = 0; j < 8; j++) {
                if (!::arrow::bit_util::GetBit(&this->mapped_bitmask_[i], j) &&
                    (frame_idx < this->capacity_)) {
                    if (this->tracing_mode_) {
                        milli_double dur = steady_clock::now() - start.value();
                        this->stats_.total_find_unmapped_time += dur;
                    }
                    return frame_idx;
                }
                frame_idx++;
            }
        }
    }

    if (this->tracing_mode_) {
        milli_double dur = steady_clock::now() - start.value();
        this->stats_.total_find_unmapped_time += dur;
    }
    // Return -1 if not found.
    return -1;
}

int64_t SizeClass::findMappedUnpinnedFrame() const noexcept {
    for (size_t i = 0; i < this->bitmask_nbytes_; i++) {
        if (this->pinned_bitmask_[i] != static_cast<uint8_t>(0xff)) {
            // There's a free bit in this byte.
            // If this is the last byte, this may be a false
            // positive.
            uint64_t frame_idx = (8 * i);
            for (size_t j = 0; j < 8; j++) {
                // Check if the frame is mapped but unpinned
                if (::arrow::bit_util::GetBit(&this->mapped_bitmask_[i], j) &&
                    !::arrow::bit_util::GetBit(&this->pinned_bitmask_[i], j) &&
                    (frame_idx < this->capacity_)) {
                    return frame_idx;
                }
                frame_idx++;
            }
        }
    }

    // Return -1 if not found
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
        throw std::runtime_error("SizeClass::FreeFrame: Frame Index (" +
                                 std::to_string(idx) + ") is out of bounds!");
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
        throw std::runtime_error("SizeClass::PinFrame: Frame Index (" +
                                 std::to_string(idx) + ") is out of bounds!");
    }

    this->markFrameAsPinned(idx);
}

void SizeClass::swapFrames(uint64_t idx1, uint64_t idx2) {
    if (idx1 >= this->capacity_) {
        throw std::runtime_error("SizeClass::swap_frames: idx1 (" +
                                 std::to_string(idx1) + ") is out of bounds!");
    }
    if (idx2 >= this->capacity_) {
        throw std::runtime_error("SizeClass::swap_frames: idx2 (" +
                                 std::to_string(idx2) + ") is out of bounds!");
    }

    uint8_t* addr1 = this->getFrameAddress(idx1);
    uint8_t* addr2 = this->getFrameAddress(idx2);
    OwningSwip swip1 = this->swips_[idx1];
    OwningSwip swip2 = this->swips_[idx2];

    // Swap data
    std::swap_ranges(addr1, addr1 + this->block_size_, addr2);

    // Update swips
    this->swips_[idx1] = swip2;
    this->swips_[idx2] = swip1;

    // Update the swip owners to point to the correct frames
    *swip1 = addr2;
    *swip2 = addr1;
}

void SizeClass::moveFrame(uint64_t source_idx, uint64_t target_idx) {
    if (source_idx >= this->capacity_) {
        throw std::runtime_error("SizeClass::moveFrame: source_idx (" +
                                 std::to_string(source_idx) +
                                 ") is out of bounds!");
    }
    if (target_idx >= this->capacity_) {
        throw std::runtime_error("SizeClass::moveFrame: target_idx (" +
                                 std::to_string(target_idx) +
                                 ") is out of bounds!");
    }

    uint8_t* old_addr = this->getFrameAddress(source_idx);
    uint8_t* new_addr = this->getFrameAddress(target_idx);
    OwningSwip swip = this->swips_[source_idx];

    // Copy data to the new frame
    std::copy(old_addr, old_addr + this->block_size_, new_addr);

    // Update swips
    this->swips_[source_idx] = nullptr;
    this->swips_[target_idx] = swip;

    // Update the swip owner of to point to the new frame
    *swip = new_addr;
}

bool SizeClass::UnpinFrame(uint64_t idx) {
    if (idx >= this->capacity_) {
        throw std::runtime_error("SizeClass::UnpinFrame: Frame Index (" +
                                 std::to_string(idx) + ") is out of bounds!");
    }

    bool spilled = false;

    if (this->move_on_unpin_) {
        // Move the data to another frame in the same SizeClass.
        // 1. Check if an unmapped frame is available.
        int64_t new_frame_idx = this->findUnmappedFrame();
        if (new_frame_idx != -1) {
            // Mark old frame as unpinned
            this->markFrameAsUnpinned(idx);

            // Move the data and swip to the new frame.
            this->moveFrame(idx, new_frame_idx);

            // Advise away the old frame
            this->adviseAwayFrame(idx);

            // Mark the old frame as unmapped and the new one
            // as mapped.
            this->markFrameAsUnmapped(idx);
            this->markFrameAsMapped(new_frame_idx);

            // Mark new frame as unpinned.
            this->markFrameAsUnpinned(new_frame_idx);
        }
        // 2. If no unmapped frame is available, look for another unpinned
        // frame.
        else {
            // NOTE: This will find an unpinned frame other than 'idx' since idx
            // hasn't been marked as unpinned yet.
            new_frame_idx = this->findMappedUnpinnedFrame();
            // Mark the frame as unpinned:
            this->markFrameAsUnpinned(idx);
            if (new_frame_idx != -1) {
                // Swap the data between the two frames
                // and update the swips. No other metadata needs
                // to be updated since both frames are already
                // mapped and unpinned and should remain so.
                this->swapFrames(idx, new_frame_idx);
            }
            // 3. If no unpinned frame is available either, just spill this
            // frame.
            else {
                CHECK_ARROW_MEM(
                    this->EvictFrame(idx),
                    "SizeClass::UnpinFrame: Error during EvictFrame which was "
                    "used because move_on_unpin is set and no other frame to "
                    "move the data to could be found: ");
                spilled = true;
            }
        }
    } else if (this->spill_on_unpin_) {
        // Mark the frame as unpinned:
        this->markFrameAsUnpinned(idx);
        // Force spill the frame to disk:
        CHECK_ARROW_MEM(
            this->EvictFrame(idx),
            "SizeClass::UnpinFrame: Error during EvictFrame which was "
            "used because spill_on_unpin is set: ");
        spilled = true;
    } else {
        // In the regular case, simply mark the frame as unpinned.
        this->markFrameAsUnpinned(idx);
    }
    return spilled;
}

arrow::Status SizeClass::EvictFrame(uint64_t idx) {
    if (idx >= this->capacity_) {
        throw std::runtime_error("SizeClass::EvictFrame: Frame Index " +
                                 std::to_string(idx) + " is out of bounds!");
    }
    if (this->isFramePinned(idx)) {
        throw std::runtime_error(
            "SizeClass::EvictFrame: Frame is not unpinned!");
    }

    auto start = start_now(tracing_mode_);
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

    this->stats_.total_blocks_spilled++;
    if (this->tracing_mode_) {
        milli_double dur = steady_clock::now() - start.value();
        this->stats_.total_spilling_time += dur;
    }
    return arrow::Status::OK();
}

arrow::Status SizeClass::ReadbackToFrame(OwningSwip swip, uint64_t frame_idx,
                                         uint64_t block_idx,
                                         uint8_t manager_idx) {
    auto start = start_now(this->tracing_mode_);
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

    this->stats_.total_blocks_readback++;
    if (this->tracing_mode_) {
        milli_double dur = steady_clock::now() - start.value();
        this->stats_.total_readback_time += dur;
    }
    return arrow::Status::OK();
}

//// StorageOptions and BufferPoolOptions
BufferPoolOptions BufferPoolOptions::Defaults() {
    BufferPoolOptions options;

    if (const char* debug_mode_env_ =
            std::getenv("BODO_BUFFER_POOL_DEBUG_MODE")) {
        options.debug_mode = !std::strcmp(debug_mode_env_, "1");
    }

    if (const char* tracing_mode_env_ =
            std::getenv("BODO_BUFFER_POOL_TRACING_MODE")) {
        options.tracing_mode = !std::strcmp(tracing_mode_env_, "1");
    }

    // Disable Storage Manager Parsing
    // Useful for debugging purposes
    const char* disable_spilling_env_ =
        std::getenv("BODO_BUFFER_POOL_DISABLE_SPILLING");
    if (!disable_spilling_env_ || std::strcmp(disable_spilling_env_, "1")) {
        // Parse Storage Managers
        for (uint8_t i = 1; i <= MAX_NUM_STORAGE_MANAGERS; i++) {
            auto storage_option = StorageOptions::Defaults(i);
            if (storage_option != nullptr) {
                storage_option->tracing_mode = options.tracing_mode;
                options.storage_options.push_back(storage_option);
            } else {
                break;
            }
        }
    }

    if (char* spill_on_unpin_env_ =
            std::getenv("BODO_BUFFER_POOL_SPILL_ON_UNPIN")) {
        options.spill_on_unpin = !std::strcmp(spill_on_unpin_env_, "1");
    }
    if (char* move_on_unpin_env_ =
            std::getenv("BODO_BUFFER_POOL_MOVE_ON_UNPIN")) {
        options.move_on_unpin = !std::strcmp(move_on_unpin_env_, "1");
    }

    if (options.spill_on_unpin && options.move_on_unpin) {
        std::cerr << "WARNING: BODO_BUFFER_POOL_SPILL_ON_UNPIN will override "
                     "BODO_BUFFER_POOL_MOVE_ON_UNPIN"
                  << std::endl;
        options.move_on_unpin = false;
    }

    // Since we will spill every unpinned allocation, the user
    // must provide at least one valid storage location to spill to.
    if (options.spill_on_unpin && (options.storage_options.size() == 0)) {
        throw std::runtime_error(
            "BufferPoolOptions::Defaults: Must specify at least one storage "
            "location when setting spill_on_unpin");
    }

    if (options.spill_on_unpin) {
        std::cerr << "Warning: Using SPILL ON UNPIN." << std::endl;
    } else if (options.move_on_unpin) {
        std::cerr << "Warning: Using MOVE ON UNPIN." << std::endl;
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
        options.sys_mem_mib = mem_per_rank;
        // Set memory_size as mem_fraction of mem_per_rank
        options.memory_size =
            static_cast<uint64_t>(mem_per_rank * mem_fraction);

        // If we're not over-allocating memory (e.g. the case where spilling is
        // available), use 16KiB as the size of the smallest frame to allow
        // packing buffers more efficiently. Otherwise, use the default (64KiB).
        if (mem_fraction <= 1.0) {
            options.min_size_class = 16;
        }
    }

    // Override the default size of the smallest Size-Class if provided
    // by an env var.
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
    // However, for testing purposes, we can enable this
    // enforcement.
    if (const char* enforce_max_limit_env_ =
            std::getenv("BODO_BUFFER_POOL_ENFORCE_MAX_ALLOCATION_LIMIT")) {
        options.enforce_max_limit_during_allocation =
            !std::strcmp(enforce_max_limit_env_, "1");
    }

    if (const char* malloc_free_trim_threshold_env_ =
            std::getenv("BODO_BUFFER_POOL_MALLOC_FREE_TRIM_THRESHOLD_MiB")) {
        options.malloc_free_trim_threshold =
            std::stoi(malloc_free_trim_threshold_env_) * 1024 * 1024;
    }

    return options;
}

//// BufferPool

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
        throw std::runtime_error("highest_power_of_2: N ( " +
                                 std::to_string(N) + ") must be >0");
    }
    // if N is a power of two simply return it
    if (!(N & (N - 1)))
        return N;
    // else set only the most significant bit
    return 0x8000000000000000UL >> (__builtin_clzll(N));
}

BufferPool::BufferPool(const BufferPoolOptions& options)
    : options_(std::move(options)),
      // Convert MiB to bytes
      memory_size_bytes_(options.memory_size * 1024 * 1024) {
    // Verify that min-size-class is a power of 2
    if (((this->options_.min_size_class &
          (this->options_.min_size_class - 1)) != 0) ||
        (this->options_.min_size_class == 0)) {
        throw std::runtime_error("BufferPool(): min_size_class (" +
                                 std::to_string(this->options_.min_size_class) +
                                 ") must be a power of 2");
    }

    // Convert from KiB to bytes
    uint64_t min_size_class_bytes = this->options_.min_size_class * 1024;

    // Calculate max possible size-class based on available memory
    // (i.e. highest power of 2 lower or equal to this->memory_size_bytes_).
    uint64_t max_size_class_possible_bytes =
        static_cast<uint64_t>(highest_power_of_2(this->memory_size_bytes_));

    if (min_size_class_bytes > max_size_class_possible_bytes) {
        throw std::runtime_error("BufferPool(): min_size_class " +
                                 std::to_string(min_size_class_bytes) +
                                 " is larger than available "
                                 "memory!");
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

    this->malloc_threshold_ =
        static_cast<uint64_t>(MALLOC_THRESHOLD_RATIO * min_size_class_bytes);

    // Construct Size Class Sizes in Bytes
    for (uint8_t i = 0; i < num_size_classes; i++) {
        this->size_class_bytes_.push_back(min_size_class_bytes);
        min_size_class_bytes *= 2;
    }

    // Construct Storage Managers
    for (auto& storage_option : this->options_.storage_options) {
        auto manager = MakeStorageManager(storage_option, size_class_bytes_);
        // std::move is needed to push a unique_ptr
        this->storage_managers_.push_back(std::move(manager));
    }

    // Create the SizeClass objects
    this->size_classes_.reserve(num_size_classes);
    for (uint8_t i = 0; i < num_size_classes; i++) {
        uint64_t num_blocks = static_cast<uint64_t>(this->memory_size_bytes_ /
                                                    size_class_bytes_[i]);
        this->size_classes_.emplace_back(std::make_unique<SizeClass>(
            i, std::span(this->storage_managers_), num_blocks,
            size_class_bytes_[i], this->options_.spill_on_unpin,
            this->options_.move_on_unpin, this->options_.tracing_mode));
    }
}

BufferPool::BufferPool() : BufferPool(BufferPoolOptions::Defaults()) {}

size_t BufferPool::num_size_classes() const {
    return this->size_classes_.size();
}

inline int64_t BufferPool::size_align(int64_t size, int64_t alignment) const {
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

inline int64_t BufferPool::find_size_class_idx(int64_t size) const {
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
    memset(ptr + size, 0, capacity - size);
}

arrow::Result<bool> BufferPool::best_effort_evict_helper(const uint64_t bytes) {
    if (!this->is_spilling_enabled()) {
        throw std::runtime_error(
            "BufferPool::best_effort_evict_helper: Cannot evict blocks when "
            "spilling is not enabled/available!");
    }

    // Attempt to evict smaller size classes
    int64_t bytes_rem = static_cast<int64_t>(bytes);
    int64_t ideal_size_class_idx = this->find_size_class_idx(bytes);
    // If number of bytes to spill is larger than the largest SizeClass,
    // we will start with the largest SizeClass and work our way down.
    if (ideal_size_class_idx == -1) {
        ideal_size_class_idx = this->num_size_classes() - 1;
    }
    std::vector<uint64_t> evicting_frames_size_class;
    std::vector<uint64_t> evicting_frame_idxs;

    for (int64_t i = ideal_size_class_idx; i >= 0; i--) {
        auto& size_class = this->size_classes_[i];
        uint64_t num_frames = size_class->getNumBlocks();
        int64_t size_bytes = this->size_class_bytes_[i];

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

    // If we've found sufficient frames, evict them and return. In case we
    // haven't, we'll come back to these after looking through larger
    // SizeClass-es.
    if (bytes_rem <= 0) {
        // Evict the identified frames. These shouldn't fail due to spill
        // locations not found errors since we already checked that spilling is
        // enabled.
        for (uint32_t i = 0; i < evicting_frame_idxs.size(); i++) {
            auto evict_status =
                this->size_classes_[evicting_frames_size_class[i]]->EvictFrame(
                    evicting_frame_idxs[i]);
            if (!evict_status.ok()) {
                return evict_status;
            }
            this->stats_.UpdateAllocatedBytes(
                -this->size_class_bytes_[evicting_frames_size_class[i]],
                /*is_free*/ true);
        }
        return true;
    }

    // There are not enough small frames to evict to free up enough
    // space. Thus, we need to find one larger frame to evict instead.
    // In the case that ideal_size_class_idx is already the largest SizeClass,
    // this loop will never run.
    for (size_t i = static_cast<size_t>(ideal_size_class_idx) + 1;
         i < this->num_size_classes(); i++) {
        auto& size_class = this->size_classes_[i];
        auto num_frames = size_class->getNumBlocks();
        for (uint64_t j = 0; j < num_frames; j++) {
            if (size_class->isFrameMapped(j) && !size_class->isFramePinned(j)) {
                auto evict_status = size_class->EvictFrame(j);
                if (!evict_status.ok()) {
                    return evict_status;
                }

                this->stats_.UpdateAllocatedBytes(-this->size_class_bytes_[i],
                                                  /*is_free*/ true);
                return true;
            }
        }
    }

    // If we couldn't find a larger frame, evict the smaller frames that
    // we'd identified earlier to reduce memory pressure at least a little
    for (uint32_t i = 0; i < evicting_frame_idxs.size(); i++) {
        auto evict_status =
            this->size_classes_[evicting_frames_size_class[i]]->EvictFrame(
                evicting_frame_idxs[i]);
        if (!evict_status.ok()) {
            return evict_status;
        }
        this->stats_.UpdateAllocatedBytes(
            -this->size_class_bytes_[evicting_frames_size_class[i]],
            /*is_free*/ true);
    }

    // Even though we might have reduced memory pressure by spilling some
    // frames, since we couldn't find enough smaller frames or 1 larger frame to
    // evict, we will return false.
    return false;
}

::arrow::Status BufferPool::evict_handler(uint64_t bytes,
                                          const std::string& caller) {
    // If spilling is available, start spilling
    if (this->is_spilling_enabled()) {
        ::arrow::Result<bool> evict_res = this->best_effort_evict_helper(bytes);
        ::arrow::Status evict_status = evict_res.status();
        if (!evict_status.ok()) {
            return evict_status.WithMessage("Error during eviction: " +
                                            evict_status.message());
        } else if (bool evicted_sufficient_bytes = evict_res.ValueOrDie();
                   !evicted_sufficient_bytes) {
            if (this->options_.enforce_max_limit_during_allocation) {
                return ::arrow::Status::OutOfMemory(
                    "Unable to evict enough frames to free up the "
                    "required space (" +
                    std::to_string(bytes) + ")");
            } else if (this->options_.debug_mode) {
                // If we weren't able to spill enough bytes, but max limit
                // enforcement is off, display a warning if debug mode
                // is enabled.
                std::cerr
                    << "[WARNING] BufferPool::" << caller
                    << ": Could not spill sufficient bytes. We will try to "
                       "allocate anyway. This may invoke the OOM killer."
                    << std::endl;
            }
        }
    } else {
        if (this->options_.enforce_max_limit_during_allocation) {
            return ::arrow::Status::OutOfMemory(
                "Spilling is not available to free up sufficient space in "
                "memory!");
        } else if (this->options_.debug_mode) {
            // Raise a warning if debug mode is enabled.
            std::cerr
                << "[WARNING] BufferPool::" << caller
                << ": Spilling is not available and available memory is less "
                   "than required amount ("
                << bytes
                << "). We will try to allocate anyway. This may invoke the OOM "
                   "killer."
                << std::endl;
        }
    }
    return ::arrow::Status::OK();
}

::arrow::Status BufferPool::Allocate(int64_t size, int64_t alignment,
                                     uint8_t** out) {
    if (size < 0) {
        return ::arrow::Status::Invalid("Negative allocation size (" +
                                        std::to_string(size) + ") requested.");
    }

    if ((alignment <= 0) || ((alignment & (alignment - 1)) != 0)) {
        return ::arrow::Status::Invalid(
            "Alignment (" + std::to_string(alignment) +
            ") must be a positive number and a power of 2.");
    }

    // Copied from Arrow (they are probably just being conservative for
    // compatibility with 32-bit architectures).
    if (static_cast<uint64_t>(size) >= std::numeric_limits<size_t>::max()) {
        return ::arrow::Status::OutOfMemory(
            "malloc size (" + std::to_string(size) + ") overflows size_t");
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
        // if enforce_max_limit_during_allocation is set.
        if (this->options_.enforce_max_limit_during_allocation &&
            aligned_size > (static_cast<int64_t>(this->memory_size_bytes_) -
                            static_cast<int64_t>(this->bytes_pinned()))) {
            return ::arrow::Status::OutOfMemory(
                "Allocation failed. Not enough space in the buffer pool to "
                "allocate (" +
                std::to_string(size) + ").");
        }

        // Note that this can be negative if max limit isn't
        // being enforced.
        int64_t bytes_available_in_mem =
            static_cast<int64_t>(this->memory_size_bytes_) -
            this->bytes_allocated();

        // If available memory is less than needed, handle it based
        // on spilling config and buffer pool options.
        if (aligned_size > bytes_available_in_mem) {
            CHECK_ARROW_MEM_RET(this->evict_handler(/*bytes*/ aligned_size -
                                                        bytes_available_in_mem,
                                                    /*caller*/ "Allocate"),
                                "BufferPool::Allocate failed: ");
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
                "Failed to allocate required bytes (" +
                std::to_string(aligned_size) + ").");
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
                "Requested alignment (" + std::to_string(alignment) +
                ") higher than max supported alignment (" +
                std::to_string(page_size) + ").");
        }
        // Use one of the mmap-ed buffer frames.
        int64_t size_class_idx = this->find_size_class_idx(aligned_size);

        if (size_class_idx == -1) {
            return ::arrow::Status::Invalid(
                "Request allocation size (" + std::to_string(size) +
                ") is larger than the largest block-size available!");
            // XXX If number of size classes was artificially set to too low
            // (through BufferPoolOptions), we also need to have ability to
            // allocate contiguous and non-contiguous blocks for cases where the
            // request is a valid size but we just don't have a size-class for
            // it. This would be similar to Velox.
        }

        const int64_t size_class_bytes =
            this->size_class_bytes_[size_class_idx];

        // If non-pinned memory is less than needed, immediately fail
        // if enforce_max_limit_during_allocation is set.
        if (this->options_.enforce_max_limit_during_allocation &&
            size_class_bytes > (static_cast<int64_t>(this->memory_size_bytes_) -
                                static_cast<int64_t>(this->bytes_pinned()))) {
            return ::arrow::Status::OutOfMemory(
                "Allocation failed. Not enough space in the buffer pool to "
                "allocate (" +
                std::to_string(size) + ").");
        }

        // Note that this can be negative if max limit isn't
        // being enforced.
        int64_t bytes_available_in_mem =
            static_cast<int64_t>(this->memory_size_bytes_) -
            this->bytes_allocated();

        // If available memory is less than needed, handle it based
        // on spilling config and buffer pool options.
        if (size_class_bytes > bytes_available_in_mem) {
            int64_t rem_bytes = size_class_bytes - bytes_available_in_mem;
            CHECK_ARROW_MEM_RET(this->evict_handler(/*bytes*/ rem_bytes,
                                                    /*caller*/ "Allocate"),
                                "BufferPool::Allocate failed: ");
        }

        // Allocate in the identified size-class.
        // In the case where max limit enforcement is enabled, due to the
        // previous check, we're guaranteed to be able to find a frame.
        // Proof:
        // We allocated memory_size // block_size many blocks. Say all frames
        // were taken, that would mean that block_size * num_blocks many bytes
        // are allocated. Allocating another block would mean that total memory
        // usage would be greater than memory_size, but we already checked that
        // there's sufficient memory available for this allocation.
        // However, in the case where max limit enforcement is not enabled,
        // we may actually have simply run out of frames.
        // XXX TODO Might need to allocate more frames (2x) when setting
        // enforce_max_limit_during_allocation as false, so that we don't run
        // out of frames.
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
    // Only fill up a couple cache lines to minimize overhead.
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
                    "BufferPool::get_alloc_details: Provided size (" +
                    std::to_string(size) +
                    ") doesn't match any of the size-classes!");
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
#ifdef __linux__
        // malloc doesn't always return memory back to the OS after
        // a 'free'. It may keep the freed memory for use by future
        // 'malloc' calls. This can have side-effects in certain
        // situations (e.g. when many small allocations worth GiBs of memory are
        // free-d without any malloc calls to re-use them). Calling malloc_trim
        // forces malloc to return unused memory to the OS (assuming it can). We
        // only call it once every malloc_free_trim_threshold bytes (default:
        // 100MiB) since calling it after 'free' call can be expensive. See
        // https://bodo.atlassian.net/browse/BSE-1768 for more context.
        this->bytes_freed_through_malloc_since_last_trim_ +=
            (size_aligned >= 0) ? size_aligned : this->malloc_threshold_;
        if (this->bytes_freed_through_malloc_since_last_trim_ >=
            this->options_.malloc_free_trim_threshold) {
            ::malloc_trim(0);
            this->bytes_freed_through_malloc_since_last_trim_ = 0;
        }
#endif
    }

    if (size_aligned != -1) {
        if (frame_pinned) {
            this->update_pinned_bytes(-size_aligned);
        }
        this->stats_.UpdateAllocatedBytes(-size_aligned, /*is_free*/ true);
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

int64_t BufferPool::get_sys_memory_bytes() const {
    if (this->options_.sys_mem_mib != -1) {
        return this->options_.sys_mem_mib * 1024 * 1024;
    } else {
        return -1;
    }
}

int64_t BufferPool::get_bytes_freed_through_malloc_since_last_trim() const {
    return this->bytes_freed_through_malloc_since_last_trim_;
}

::arrow::Status BufferPool::Reallocate(int64_t old_size, int64_t new_size,
                                       int64_t alignment, uint8_t** ptr) {
    if (new_size < 0) {
        return ::arrow::Status::Invalid("Negative reallocation size (" +
                                        std::to_string(new_size) +
                                        ") requested.");
    }
    if (static_cast<uint64_t>(new_size) >= std::numeric_limits<size_t>::max()) {
        return ::arrow::Status::OutOfMemory(
            "realloc (" + std::to_string(new_size) + ") overflows size_t");
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

    // Record if the allocation is pinned. If it isn't, we will pin it in the
    // next step. However, if there's a failure later while allocating the new
    // memory space, we will unpin the original allocation to restore the
    // original state.
    bool was_pinned = this->IsPinned(old_memory_ptr, old_size, alignment);

    // Pin it if it isn't already. We either need it to be in memory for the
    // memcpy or even if we end up re-using the same frame, Reallocate needs to
    // pin the frame.
    if (!was_pinned) {
        auto status = this->Pin(ptr, old_size, alignment);
        if (!status.ok()) {
            return status.WithMessage(
                "BufferPool::Reallocate: Failed while trying to pin old "
                "allocation (" +
                std::to_string(old_size) + "): " + status.ToString());
        }
        // Update old_memory_ptr in case the original was unswizzled and the
        // Pin operation brought it back into memory.
        old_memory_ptr = *ptr;
    }

    // We can only get these details once we've pinned it back.
    auto [is_mmap_alloc, size_class_idx, frame_idx, old_size_aligned] =
        this->get_alloc_details(old_memory_ptr, old_size, alignment);

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
    arrow::Status alloc_status = this->Allocate(new_size, alignment, ptr);
    if (!alloc_status.ok()) {
        // Undo pinning if it wasn't originally pinned (to restore
        // original state). Note that if it was read from disk, we
        // won't spill it back to disk at this point. Future allocations
        // can spill it if required.
        if (!was_pinned) {
            this->Unpin(old_memory_ptr, old_size, alignment);
        }
        return alloc_status.WithMessage(
            "BufferPool::Reallocate: Allocation of new memory failed: " +
            alloc_status.ToString());
    }

    // Get a lock on the BufferPool state for the rest
    // of this function. 'scoped_lock' guarantees that the
    // lock will be released when the function ends (even if
    // there's an exception).
    std::scoped_lock lock(this->mtx_);

    uint8_t* new_memory_ptr = *ptr;

    // Copy over the contents
    std::memcpy(new_memory_ptr, old_memory_ptr,
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
        // if enforce_max_limit_during_allocation is set.
        if (this->options_.enforce_max_limit_during_allocation &&
            (static_cast<int64_t>(block_bytes) >
             (static_cast<int64_t>(this->memory_size_bytes_) -
              static_cast<int64_t>(this->bytes_pinned())))) {
            return ::arrow::Status::OutOfMemory(
                "Pin failed. Not enough space in the buffer pool to pin " +
                std::to_string(size) + " bytes.");
        }

        // Note that this can be negative if max limit isn't
        // being enforced.
        int64_t bytes_available_in_mem =
            static_cast<int64_t>(this->memory_size_bytes_) -
            this->bytes_allocated();

        // If available memory is less than needed, handle it based
        // on spilling config and buffer pool options.
        if (static_cast<int64_t>(block_bytes) > bytes_available_in_mem) {
            int64_t rem_bytes = block_bytes - bytes_available_in_mem;
            CHECK_ARROW_MEM_RET(this->evict_handler(/*bytes*/ rem_bytes,
                                                    /*caller*/ "Pin"),
                                "BufferPool::Pin failed: ");
        }

        // Find an available frame in the size class
        int64_t frame_idx = size_class->findUnmappedFrame();
        if (frame_idx == -1) {
            // Should be impossible at this point unless max limit enforcement
            // is disabled.
            return ::arrow::Status::OutOfMemory(
                "Pin failed. Unable to find available frame");
        }

        // Load Block from Storage into Frame
        CHECK_ARROW_MEM_RET(
            size_class->ReadbackToFrame(ptr, frame_idx, block_id,
                                        storage_manager_idx),
            "Pin failed. Error when reading back block from storage:");

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
        bool was_spilled =
            this->size_classes_[size_class_idx]->UnpinFrame(frame_idx);
        this->update_pinned_bytes(-this->size_class_bytes_[size_class_idx]);

        // In the spill_on_unpin case, we will spill any unpinned frames.
        // In the move_on_unpin case, we might've spilled the frame if
        // we weren't able to find another frame to move the data to.
        // We must update the statistics to match this action.
        // 'was_spilled' is guaranteed to be false in the case that neither
        // of these macros are defined, so we can skip this check entirely
        // in the regular case.
        if (was_spilled) {
            this->stats_.UpdateAllocatedBytes(
                -this->size_class_bytes_[size_class_idx], /*is_free*/ true);
        }
    }
}

SizeClass* BufferPool::GetSizeClass_Unsafe(uint64_t idx) const {
    if (idx > this->size_classes_.size()) {
        throw std::runtime_error(
            "BufferPool::GetSizeClass_Unsafe: Requested SizeClass (" +
            std::to_string(idx) + ") doesn't exist.");
    }
    return this->size_classes_[idx].get();
}

uint64_t BufferPool::GetSmallestSizeClassSize() const {
    if (this->size_class_bytes_.size() > 0) {
        return this->size_class_bytes_[0];
    }
    return 0;
}

void BufferPool::Cleanup() {
    for (auto& manager : this->storage_managers_) {
        manager->Cleanup();
    }

    if (this->options_.tracing_mode) {
        const char* COL_NAMES[8] = {
            "Size Class",       "Blocks Spilled",     "Time Spilling",
            "Blocks Read Back", "Time Reading Back",  "Num Advise Away",
            "Time Advise Away", "Time Find Unmapped",

        };
        std::vector<size_t> col_widths(8);
        std::transform(std::begin(COL_NAMES), std::end(COL_NAMES),
                       col_widths.begin(), strlen);

        fmt::dynamic_format_arg_store<fmt::format_context> store;
        store.push_back("");
        for (const auto& x : col_widths)
            store.push_back(x);

        fmt::println(stderr, "{0:─^{1}}", " Size Class Metrics ",
                     col_widths.size() * 3 +
                         std::reduce(col_widths.begin(), col_widths.end()));
        fmt::vprint(stderr,
                    "{0:─^{1}}─┬─{0:─^{2}}─┬─{0:─^{3}}─┬─{0:─^{4}}─┬─{0:─^{5}}"
                    "─┬─{0:─^{6}}─┬─{0:─^{7}}─┬─{0:─^{8}}─\n",
                    store);
        fmt::println(stderr, "{} ", fmt::join(COL_NAMES, " │ "));
        fmt::vprint(stderr,
                    "{0:─^{1}}─┼─{0:─^{2}}─┼─{0:─^{3}}─┼─{0:─^{4}}─┼─{0:─^{5}}"
                    "─┼─{0:─^{6}}─┼─{0:─^{7}}─┼─{0:─^{8}}─\n",
                    store);

        for (const auto& s : size_classes_) {
            fmt::println(
                stderr,
                "{0:<{1}} │ {2:>{3}} │ {4:>{5}} │ {6:>{7}} │ {8:>{9}} "
                "│ {10:>{11}} │ {12:>{13}} │ {14:>{15}} ",
                BytesToHumanReadableString(s->getBlockSize()),
                strlen(COL_NAMES[0]), s->stats_.total_blocks_spilled,
                strlen(COL_NAMES[1]), s->stats_.total_spilling_time,
                strlen(COL_NAMES[2]), s->stats_.total_blocks_readback,
                strlen(COL_NAMES[3]), s->stats_.total_readback_time,
                strlen(COL_NAMES[4]), s->stats_.total_advise_away_calls,
                strlen(COL_NAMES[5]), s->stats_.total_advise_away_time,
                strlen(COL_NAMES[6]), s->stats_.total_find_unmapped_time,
                strlen(COL_NAMES[7]));
        }
        fmt::vprint(stderr,
                    "{0:─^{1}}─┴─{0:─^{2}}─┴─{0:─^{3}}─┴─{0:─^{4}}─┴─{0:─^{5}}"
                    "─┴─{0:─^{6}}─┴─{0:─^{7}}─┴─{0:─^{8}}─\n",
                    store);
        fmt::println(stderr, "");
    }

    if (this->options_.tracing_mode && this->storage_managers_.size() > 0) {
        std::vector<std::string_view> manager_names;
        for (auto& manager : this->storage_managers_) {
            manager_names.push_back(manager->storage_name);
        }

        std::vector<StorageManagerStats> stats;
        for (auto& manager : this->storage_managers_) {
            stats.push_back(manager->stats_);
        }

        PrintStorageManagerStats(stderr, manager_names, stats);
    }
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
