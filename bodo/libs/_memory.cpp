#include "_memory.h"
#include <arrow/util/bit_util.h>
#include <mpi.h>
#include <sys/mman.h>
#include <algorithm>
#include <cerrno>
#include <cmath>

namespace bodo {

//// SizeClass

SizeClass::SizeClass(size_t capacity, size_t block_size)
    : capacity_(capacity),
      block_size_(block_size),
      byteSize_(capacity * block_size),
      bitmask_nbytes_((this->capacity_ + 7) >> 3),
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

inline void SizeClass::pinFrame(uint64_t idx) {
    if (!::arrow::bit_util::GetBit(this->mapped_bitmask_.data(), idx)) {
        throw std::runtime_error("Cannot pin an unmapped frame.");
    }
    ::arrow::bit_util::SetBitTo(this->pinned_bitmask_.data(), idx, true);
}

inline void SizeClass::unpinFrame(uint64_t idx) {
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
    // Pin the frame (default behavior)
    this->pinFrame(frame_idx);
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
    // Unpin the frame (in case it was)
    this->unpinFrame(idx);
    // Mark the frame as unmapped
    this->markFrameAsUnmapped(idx);
    // Reset the swip (if there was one)
    this->swips_[idx] = nullptr;
}

void SizeClass::FreeFrame(uint8_t* ptr) {
    uint64_t frame_idx = this->getFrameIndex(ptr);
    this->FreeFrame(frame_idx);
}

//// BufferPoolOptions

/**
 * @brief Get the number of ranks on this node.
 * We do this by creating sub-communicators based
 * on shared-memory. This is a collective operation
 * and therefore all ranks must call it.
 *
 */
static int dist_get_ranks_on_node() {
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) {
        MPI_Init(NULL, NULL);
    }

    int npes_node;
    MPI_Comm shmcomm;

    // Split comm, into comms that has same shared memory.
    // This is a collective operation and all ranks must call it.
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shmcomm);
    // Get number of ranks on this sub-communicator (i.e. node).
    // By definition, all ranks on the same node will get the same
    // output.
    MPI_Comm_size(shmcomm, &npes_node);

    MPI_Comm_free(&shmcomm);
    return npes_node;
}

BufferPoolOptions BufferPoolOptions::Defaults() {
    BufferPoolOptions options;

    // Read memory_size from env_var if provided.
    // This will be the common case on the platform.
    // If env var is not set, we will get the memory
    // information from the OS.
    char* memory_size_env_ = std::getenv("BODO_BUFFER_POOL_MEMORY_SIZE_MiB");

    if (memory_size_env_) {
        options.memory_size = std::stoi(memory_size_env_);
    } else {
        // Fraction of total memory we should actually assign to the
        // buffer pool. We will read this from an environment variable
        // if it's set, but will default to 95% if it isn't.
        double mem_fraction = 0.95;

        // We expect this to be in percentages and not fraction,
        // i.e. it should be set to 45 if we want to use 45% (or 0.45)
        // of the total available space.
        char* mem_percent_env_ =
            std::getenv("BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT");
        if (mem_percent_env_) {
            mem_fraction =
                static_cast<double>(std::stoi(mem_percent_env_) / 100.0);
        }

        // Get number of ranks on this node.
        int num_ranks_on_node = dist_get_ranks_on_node();
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

    char* min_size_class_env_ =
        std::getenv("BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB");
    if (min_size_class_env_) {
        options.min_size_class = std::stoi(min_size_class_env_);
    }

    char* max_num_size_classes_env_ =
        std::getenv("BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES");
    if (max_num_size_classes_env_) {
        options.max_num_size_classes = std::stoi(max_num_size_classes_env_);
    }

    // BufferPool's equal allocation per rank
    // approach can cause issues for existing Bodo workloads
    // until we have spill support. In particular,
    // we might preemptively disallow allocations on a rank
    // in case of skew. Until we have spill support,
    // we want to be able to attempt allocation
    // even if it's beyond the allocated limit.
    options.ignore_max_limit_during_allocation = true;

    char* ignore_max_limit_env_ =
        std::getenv("BODO_BUFFER_POOL_IGNORE_MAX_ALLOCATION_LIMIT");
    if (ignore_max_limit_env_) {
        if (std::strcmp(ignore_max_limit_env_, "1") == 0) {
            options.ignore_max_limit_during_allocation = true;
        } else {
            options.ignore_max_limit_during_allocation = false;
        }
    }

    return options;
}

//// BufferPool

BufferPool::BufferPool(const BufferPoolOptions& options)
    : options_(std::move(options)),
      // Convert MiB to bytes
      memory_size_bytes_(options.memory_size * 1024 * 1024) {
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
    uint64_t num_size_classes = static_cast<uint8_t>(std::min(
        std::min(this->options_.max_num_size_classes, max_num_size_classes),
        (uint64_t)63));

    // Create the SizeClass objects
    this->size_classes_.reserve(num_size_classes);
    this->size_class_bytes_.reserve(num_size_classes);
    uint64_t size_class_bytes_i = min_size_class_bytes;
    for (size_t i = 0; i < num_size_classes; i++) {
        uint64_t num_blocks = static_cast<uint64_t>(this->memory_size_bytes_ /
                                                    size_class_bytes_i);
        this->size_classes_.emplace_back(
            std::make_unique<SizeClass>(num_blocks, size_class_bytes_i));
        this->size_class_bytes_.emplace_back(size_class_bytes_i);
        size_class_bytes_i *= 2;
    }

    this->malloc_threshold_ =
        static_cast<uint64_t>(MALLOC_THRESHOLD_RATIO * min_size_class_bytes);
}

size_t BufferPool::num_size_classes() const {
    return this->size_classes_.size();
}

int64_t BufferPool::size_align(int64_t size) {
    const auto remainder = size % this->alignment_;
    return (remainder == 0) ? size : (size + this->alignment_ - remainder);
}

int64_t BufferPool::max_memory() const { return stats_.max_memory(); }

int64_t BufferPool::bytes_allocated() const { return stats_.bytes_allocated(); }

std::string BufferPool::backend_name() const { return "bodo"; }

uint16_t BufferPool::alignment() const { return this->alignment_; }

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

::arrow::Status BufferPool::Allocate(int64_t size, uint8_t** out) {
    if (size < 0) {
        return ::arrow::Status::Invalid("Negative allocation size requested.");
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

    const int64_t aligned_size = this->size_align(size);

    if (aligned_size <= static_cast<int64_t>(this->malloc_threshold_)) {
        // Use malloc

        // Only check and throw an error if ignore_max_limit_during_allocation
        // is false.
        if (!this->options_.ignore_max_limit_during_allocation &&
            (aligned_size > (static_cast<int64_t>(this->memory_size_bytes_) -
                             this->bytes_allocated()))) {
            return ::arrow::Status::OutOfMemory(
                "Allocation failed. Not enough space in the buffer pool.");
        }

        // There's essentially two options:
        // 1. posix_memalign/memalign: This is what Arrow uses
        //    (https://github.com/apache/arrow/blob/ea6875fd2a3ac66547a9a33c5506da94f3ff07f2/cpp/src/arrow/memory_pool.cc#L318)
        // 2. aligned_alloc: This is what Velox uses
        //    (https://github.com/facebookincubator/velox/blob/8324ac7f1839db009def00e7450f38c2591dd4bb/velox/common/memory/MmapAllocator.cpp#L371)
        //     and seems to be what is generally recommended. The only
        //     requirement is that the allocation size must be a multiple of the
        //     requested alignment, which we do in size_align already.
        // malloc does 16-bit alignment by default, so we can use it for
        // those cases.
        // All these allocations can be free-d using 'free'.
        void* result = this->alignment_ > kMinAlignment
                           ? ::aligned_alloc(this->alignment_, aligned_size)
                           : ::malloc(aligned_size);
        if (result == nullptr) {
            // XXX This is an unlikely branch, so it would
            // be good to indicate that to the compiler
            // similar to how Velox does it using "foggy".
            return ::arrow::Status::UnknownError(
                "Failed to allocate required bytes.");
        }
        *out = static_cast<uint8_t*>(result);
        this->stats_.UpdateAllocatedBytes(aligned_size);

        // Zero-pad to match Arrow
        // https://github.com/apache/arrow/blob/5b2fbade23eda9bc95b1e3854b19efff177cd0bd/cpp/src/arrow/memory_pool.cc#L932
        // https://github.com/apache/arrow/blob/5b2fbade23eda9bc95b1e3854b19efff177cd0bd/cpp/src/arrow/buffer.h#L125
        this->zero_padding(static_cast<uint8_t*>(result), size, aligned_size);

    } else {
        // Use one of the mmap-ed buffer frames.
        // Mmap-ed memory is always (at-least) 64-byte aligned
        // (https://stackoverflow.com/questions/42259495/does-mmap-return-aligned-pointer-values)
        // so we don't need to worry about that.
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

        // Only check and throw an error if ignore_max_limit_during_allocation
        // is false.
        if (!this->options_.ignore_max_limit_during_allocation &&
            (size_class_bytes >
             (this->memory_size_bytes_ -
              static_cast<uint64_t>(this->bytes_allocated())))) {
            return ::arrow::Status::OutOfMemory(
                "Allocation failed. Not enough space in the buffer pool.");
        }

        // Allocate in the identified size-class.
        // We're guaranteed to be able to find a frame.
        // Proof: We allocated memory_size // block_size many blocks. Say all
        // frames were taken, that would mean that block_size * num_blocks many
        // bytes are allocated. Allocating another block would mean that total
        // memory usage would be greater than memory_size, but we already
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

        this->stats_.UpdateAllocatedBytes(size_class_bytes);
    }
    return ::arrow::Status::OK();
}

std::tuple<bool, int64_t, int64_t, int64_t> BufferPool::get_alloc_details(
    uint8_t* buffer, int64_t size) {
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
        const int64_t aligned_size = this->size_align(size);
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
    if (is_mmap_alloc) {
        this->size_classes_[size_class_idx]->FreeFrame(frame_idx);
    } else {
        ::free(ptr);
    }

    if (size_aligned != -1) {
        this->stats_.UpdateAllocatedBytes(-size_aligned);
    }
}

void BufferPool::Free(uint8_t* buffer, int64_t size) {
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

    auto [is_mmap_alloc, size_class_idx, frame_idx, size_freed] =
        this->get_alloc_details(buffer, size);

    this->free_helper(buffer, is_mmap_alloc, size_class_idx, frame_idx,
                      size_freed);
    // XXX In the case where we still don't know the size of the allocation and
    // it was through malloc, we can't update stats_. Should we just enforce
    // that size be provided?
}

::arrow::Status BufferPool::Reallocate(int64_t old_size, int64_t new_size,
                                       uint8_t** ptr) {
    if (new_size < 0) {
        return ::arrow::Status::Invalid(
            "Negative reallocation size requested.");
    }
    if (static_cast<uint64_t>(new_size) >= std::numeric_limits<size_t>::max()) {
        return ::arrow::Status::OutOfMemory("realloc overflows size_t");
    }

    uint8_t* previous_ptr = *ptr;

    if (previous_ptr == kZeroSizeArea) {
        if (old_size > 0) {  // neither 0 nor -1 (unknown)
            // TODO: Add compiler hint that this path is not likely.
            std::cerr << "Expected size of allocation pointing to ZeroArea to "
                         "be 0 or unknown."
                      << std::endl;
        }
        return this->Allocate(new_size, ptr);
    }

    if (new_size == 0) {
        this->Free(previous_ptr, old_size);
        *ptr = kZeroSizeArea;
        return ::arrow::Status::OK();
    }

    auto [is_mmap_alloc, size_class_idx, frame_idx, old_size_aligned] =
        this->get_alloc_details(previous_ptr, old_size);

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

    uint8_t* out = nullptr;
    // Allocate new_size
    CHECK_ARROW_MEM(this->Allocate(new_size, &out), "Allocation failed!");
    // Copy over the contents
    memcpy(out, *ptr, static_cast<size_t>(std::min(new_size, old_size)));
    // Free original memory (re-use information from get_alloc_details output)
    this->free_helper(*ptr, is_mmap_alloc, size_class_idx, frame_idx,
                      old_size_aligned);
    // Point ptr to the new memory
    *ptr = out;
    return ::arrow::Status::OK();
}

SizeClass* BufferPool::GetSizeClass_Unsafe(uint64_t idx) const {
    if (idx > this->size_classes_.size()) {
        throw std::runtime_error("Requested SizeClass doesn't exist.");
    }
    return this->size_classes_[idx].get();
}

}  // namespace bodo
