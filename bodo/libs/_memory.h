#pragma once

#include <memory>
#include <mutex>
#include <span>

#include <arrow/compute/exec.h>
#include <arrow/io/interfaces.h>
#include <arrow/memory_pool.h>
#include <arrow/util/windows_compatibility.h>

#include "_stl_allocator.h"
#include "_storage_manager.h"

// Forward declare boost::json::object to avoid including the entire header and
// increasing compile times
namespace boost::json {
class object;
}

// Fraction of the smallest mmap-ed SizeClass
// that should be used as threshold for allocation
// through malloc.
#define MALLOC_THRESHOLD_RATIO 0.75

// Call malloc_trim after every 100MiB have been freed through malloc:
#define MALLOC_FREE_TRIM_DEFAULT_THRESHOLD 100 * 1024 * 1024

namespace bodo {

class BufferPool;

// Buffer pool pointer that points to the central buffer pool from the
// memory_cpp module. Initialized in bodo_common_init() which should be called
// by all extension modules.
inline BufferPool* global_memory_pool = nullptr;
inline std::shared_ptr<BufferPool> global_memory_pool_shared = nullptr;

/**
 * @brief Initialize global BufferPool pointer. Only called in memory_cpp which
 * creates the global pool.
 *
 * @param ptr Pointer to the BufferPool
 */
void init_buffer_pool_ptr(int64_t ptr);

// Copied from Velox
// (https://github.com/facebookincubator/velox/blob/8324ac7f1839db009def00e7450f38c2591dd4bb/velox/common/memory/MemoryAllocator.h#L192)
static constexpr uint16_t kMinAlignment = alignof(max_align_t);
static constexpr uint16_t kMaxAlignment = 64;

// A static piece of memory for 0-size allocations, so as to return
// an aligned non-null pointer. (Copied from Arrow)
static int64_t zero_size_area[1];
static uint8_t* const kZeroSizeArea =
    reinterpret_cast<uint8_t*>(&zero_size_area);

typedef uint8_t* Swip;
typedef Swip* OwningSwip;

/**
 * @brief Abstract class for Buffer Pool implementations.
 * ::arrow::MemoryPool already defines Allocate, Free, Reallocate,
 * bytes_allocated, max_memory, etc., therefore this
 * extends it to define the additional functions that are required for
 * a Buffer Pool.
 *
 */
class IBufferPool : public ::arrow::MemoryPool {
   public:
    /// @brief The number of bytes currently pinned by this
    /// pool. Pinned bytes can never be spilled to disk.
    virtual uint64_t bytes_pinned() const = 0;

    /**
     * @brief Get the SizeClass and frame index for a given
     * memory allocation, if it was allocated by the BufferPool.
     *
     * This is essentially a safe wrapper over
     * BufferPool::get_alloc_details_unsafe
     *
     * @param ptr Pointer to allocated buffer
     * @param size Optional size of allocation if known
     * @param alignment Optional Alignment of allocation if known
     * @return SizeClass and frame index if allocated by the buffer pool
     *   and none otherwise.
     */
    virtual std::optional<std::tuple<uint64_t, uint64_t>> alloc_loc(
        uint8_t* ptr, int64_t size = -1,
        int64_t alignment = arrow::kDefaultBufferAlignment) const = 0;

    /**
     * @brief Pin an allocation to memory.
     * This will ensure that the allocation will never be
     * evicted to storage
     *
     * @param[in, out] ptr Location of swip pointer containing
     *   allocation to pin
     * @param size Size of the allocation (original requested size)
     * @param alignment Alignment used for the allocation
     * @return ::arrow::Status If the pin succeeded, if it
     *   failed due to OOM, or failed for another reason
     */
    virtual ::arrow::Status Pin(
        uint8_t** ptr, int64_t size = -1,
        int64_t alignment = arrow::kDefaultBufferAlignment) = 0;

    /**
     * @brief Unpin an allocation.
     * This allows future allocations to evict this
     * allocation from memory to storage.
     *
     * @param[in, out] ptr Swip pointer to allocation to unpin
     * @param size Size of the allocation (original requested size)
     * @param alignment Alignment used for the allocation
     */
    virtual void Unpin(uint8_t* ptr, int64_t size = -1,
                       int64_t alignment = arrow::kDefaultBufferAlignment) = 0;
};

/// @brief Options for the Buffer Pool implementation
struct BufferPoolOptions {
    /// @brief Total memory available to this rank (in MiB)
    /// for the buffer pool.
    uint64_t memory_size = 200;

    /// @brief Real system memory (in MiB) available to this rank.
    /// -1 indicates that this value is not known.
    int64_t sys_mem_mib = -1;

    /// @brief Size of the smallest size class (in KiB)
    /// Must be a power of 2.
    uint64_t min_size_class = 64;

    /// @brief Maximum number of size classes allowed.
    /// The actual number of size classes will be the minimum
    /// of this and the max size classes possible based
    /// on memory_size and min_size_class.
    /// Since this needs to be encodable in 6 bits,
    /// the max allowed value is 63.
    uint64_t max_num_size_classes = 23;

    /// @brief Whether or not to enforce the specified
    /// memory limit during allocations. This is useful
    /// for debugging purposes.
    bool enforce_max_limit_during_allocation = false;

    /// @brief Config for Storage Managers
    std::vector<std::shared_ptr<StorageOptions>> storage_options;

    /// @brief When spill-on-unpin is set, all unpinned allocations
    /// (assuming they are from a SizeClass and not allocated
    /// through malloc) are forcefully spilled to disk. This
    /// is useful for testing correctness and is not intended
    /// for production use cases.
    bool spill_on_unpin = false;

    /// @brief When move-on-unpin is set, all unpinned allocations
    /// (assuming they are from a SizeClass and not allocated
    /// through malloc) are forcefully moved to a different
    /// frame in the same SizeClass. We first try to move it to
    /// an unused frame. If one cannot be found, we swap it with
    /// another unpinned frame. If we cannot find another unpinned
    /// frame either, we just spill the block.
    /// This is useful for testing correctness and is not
    /// intended for production use cases.
    bool move_on_unpin = false;

    /// @brief Run the buffer pool in debug mode. Currently,
    /// this will print some warnings in evict_handler when
    /// enforce_max_limit_during_allocation is false.
    bool debug_mode = false;

    /// @brief Track various statistics about the buffer pool.
    /// Primarily used to enable timers for various operations
    /// for benchmarking purposes.
    /// trace_level = 0: No tracing
    /// trace_level = 1: Trace time, print for rank 0
    /// trace_level = 2: Trace time, print for all ranks
    uint8_t trace_level = 0;

    /// @brief Number of bytes free-d through malloc on this rank
    /// after which we call malloc_trim.
    /// NOTE: This is only applicable on Linux since malloc_trim
    /// is only available on Linux.
    /// This is part of BufferOptions purely for unit-testing
    /// purposes. The 100MiB default should work well for
    /// practical use cases.
    int64_t malloc_free_trim_threshold = MALLOC_FREE_TRIM_DEFAULT_THRESHOLD;

    /// @brief Whether we should allocate extra frames in the larger
    /// Size-Classes to account for potential under-utilization. In particular,
    /// we should only do this when we're not already allocating extra memory
    /// (we allocate 5x when spilling is not available).
    /// Note that the actual value that the BufferPool uses to decide whether or
    /// not to allocate extra frames comes from the 'allocate_extra_frames'
    /// function which only returns true if the max-limit is not being
    /// enforced.
    bool _allocate_extra_frames = true;

    /// @brief Flag indicating if Bodo is running on a remote system
    /// such as a cloud instance or the platform. Indicates that the pool
    /// can perform more dangerous actions without the risk of swap or other
    /// OS actions being a concern.
    bool remote_mode = false;

    static BufferPoolOptions Defaults();

    /// @brief Is tracing mode enabled?
    bool tracing_mode() const noexcept { return trace_level > 0; }

    bool allocate_extra_frames() const noexcept {
        // Extra frames are only useful if max limit enforcement is disabled.
        return this->_allocate_extra_frames &&
               !this->enforce_max_limit_during_allocation;
    }
};

/// @brief Statistics for a SizeClass
struct SizeClassStats {
    /// @brief Total Number of Blocks Ever Spilled
    uint64_t total_blocks_spilled = 0;
    /// @brief Total Number of Blocks Ever Read Back
    uint64_t total_blocks_readback = 0;
    /// @brief Total Number of Times Advise Away is Called
    uint64_t total_advise_away_calls = 0;

    /// @brief Total Time Spent Spilling Blocks
    milli_double total_spilling_time{};
    /// @brief Total Time Spent Reading Back Blocks
    milli_double total_readback_time{};
    /// @brief Total Time Spent Finding Unmapped Blocks
    milli_double total_find_unmapped_time{};
    /// @brief Total Time Spent on madvise
    milli_double total_advise_away_time{};
};

/// @brief Represents a range of virtual addresses used for allocating
/// buffer frames of a fixed size. Very similar to Velox's SizeClass
/// (https://github.com/facebookincubator/velox/blob/8324ac7f1839db009def00e7450f38c2591dd4bb/velox/common/memory/MmapAllocator.h#L141).
class SizeClass final {
   public:
    /**
     * @brief Create a new mmap-ed SizeClass for the Buffer Pool.
     *
     * @param idx Relative index / ordering of SizeClass
     * @param storage_managers Fixed-view span of Storage Managers
     *  to spill frames to. Owned by the Buffer Pool
     * @param capacity Number of frames to allocate
     * @param block_size Size of each frame (in bytes).
     * @param spill_on_unpin Whether the unpin behavior should be to spill the
     * frame. See BufferPoolOptions::spill_on_unpin for the full description.
     * @param move_on_unpin Whether the unpin behavior should be to move the
     * frame (or spill if no other frame could be found). See
     * BufferPoolOptions::move_on_unpin for the full description.
     * @param tracing_mode Whether to trace time spent in various functions.
     */
    SizeClass(uint8_t idx,
              const std::span<std::unique_ptr<StorageManager>> storage_managers,
              size_t capacity, size_t block_size, bool spill_on_unpin,
              bool move_on_unpin, bool tracing_mode);

    /**
     * @brief Delete the SizeClass and unmap the allocated
     * virtual address space.
     */
    ~SizeClass();

    /**
     * @brief True if 'ptr' is in the address range of this SizeClass. Checks
     * that ptr is at a size class page boundary.
     *
     * NOTE: This function is deterministic and therefore doesn't
     * need to be thread safe.
     *
     * @param ptr Pointer to check
     */
    bool isInRange(uint8_t* ptr) const;

    /**
     * @brief Get the address for the frame at index idx.
     *
     * NOTE: This function is deterministic and therefore doesn't
     * need to be thread safe.
     *
     * @param idx Frame Index
     * @return uint8_t* Address of the frame.
     */
    uint8_t* getFrameAddress(uint64_t idx) const;

    /**
     * @brief Get the index of the frame at which ptr starts.
     *
     * NOTE: This function is deterministic and therefore doesn't
     * need to be thread safe.
     *
     * @param ptr Pointer to the start of the frame
     * @return uint64_t Frame Index
     */
    uint64_t getFrameIndex(uint8_t* ptr) const;

    /**
     * @brief Get the Swip at frame idx. This is a helper
     * function for testing.
     *
     * @param idx The frame to get the current swip for
     * @return uint8_t** The current swip pointer
     */
    uint8_t** getSwip(uint64_t idx) const;

    /**
     * @brief Find and allocate an unmapped frame.
     * The frame will be marked as mapped and pinned.
     * If provided, the 'swip' will be stored as part
     * of the frame's metadata.
     *
     * This will return -1 in case an empty frame
     * wasn't found. The caller must handle this.
     *
     * NOTE: This function is thread safe.
     *
     * @param swip Swip to store for this frame.
     * @return int64_t Index of the allocated frame.
     */
    int64_t AllocateFrame(OwningSwip swip = nullptr);

    /**
     * @brief Free the frame at index idx.
     * This will "advise-away" the address space
     * associated with the frame, unpin and unmap the
     * frame and clear the swip metadata.
     *
     * NOTE: This function is thread safe.
     *
     * @param idx Frame index to free.
     */
    void FreeFrame(uint64_t idx);

    /**
     * @brief Same as FreeFrame(uint64_t idx), but
     * the frame is specified using the start of the
     * frame.
     *
     * NOTE: This function is thread safe.
     *
     * @param ptr Pointer to the start of the frame.
     */
    void FreeFrame(uint8_t* ptr);

    /**
     * @brief Pin the frame at index idx.
     * This is essentially a public wrapper for
     * markFrameAsPinned.
     *
     * @param idx Frame index to pin.
     */
    void PinFrame(uint64_t idx);

    /**
     * @brief Unpin the frame at index idx.
     * This is essentially a public wrapper for
     * markFrameAsUnpinned.
     *
     * @param idx Frame index to unpin.
     * @return bool Was the frame spilled to disk. This is only applicable in
     * the spill_on_unpin or move_on_unpin cases. In the regular case, this will
     * always be false. In the spill_on_unpin case, this will always be true.
     * In the move_on_unpin case, this will be true only if we weren't able to
     * find an alternative frame to move the data to and had to spill the block
     * instead.
     */
    bool UnpinFrame(uint64_t idx);

    /**
     * @brief Evict the frame at index idx from
     * memory and write contents to first available
     * storage location.
     *
     * Will throw an error if the frame is pinned
     *
     * @param idx Frame index to evict
     * @return
     *   arrow::Status::Ok If the frame was successfully evicted
     *   arrow::Status::OutOfMemory If eviction failed because
     *     no storage space is available
     *   arrow::Status If eviction failed for an unexpected reason
     */
    arrow::Status EvictFrame(uint64_t idx);

    /**
     * @brief Read block block_idx from storage manager
     * manager_idx into an available frame in the size class
     *
     * @param[in, out] swip Swip pointer to update with memory
     *   location of the block's data
     * @param frame_idx Index of frame to write block to
     * @param block_idx Index of block to read into memory
     * @param manager_idx Index of manager that is handling the block
     * @throw A potential filesystem error from the storage manager
     */
    void ReadbackToFrame(OwningSwip swip, uint64_t frame_idx, int64_t block_id,
                         uint8_t manager_idx);

    /// @brief Get size of each frame (in bytes)
    inline uint64_t getBlockSize() const { return this->block_size_; }

    /// @brief Get number of frames in this SizeClass
    inline uint64_t getNumBlocks() const { return this->capacity_; }

    /// @brief Check if frame at index 'idx' is mapped.
    /// Note: Not thread safe.
    bool isFrameMapped(uint64_t idx) const;

    /// @brief Check if frame at index 'idx' is pinned.
    /// Note: Not thread safe.
    bool isFramePinned(uint64_t idx) const;

    /**
     * @brief Find the index of the first unmapped
     * frame in this SizeClass. Note that this only
     * finds the frame but doesn't mark it as
     * mapped.
     *
     * @return int64_t Index of the first unmapped frame. -1 if no unmapped
     * frame was found.
     */
    int64_t findUnmappedFrame() noexcept;

    /// @brief Statistics for this SizeClass
    SizeClassStats stats_;

   private:
    /// @brief Size Class Index
    const uint8_t idx_;

    /// @brief Number of buffer frames.
    const uint64_t capacity_;

    /// @brief Size of each frame (in bytes).
    const uint64_t block_size_;

    /// @brief Size in bytes of the address range.
    const size_t byteSize_;

    /// @brief Number of bytes for mapped_bitmask / pinned_bitmask /
    /// priority_hint
    const uint64_t bitmask_nbytes_;

    /// @brief Const Pointer to Storage Managers
    /// Buffer Pool owns and is responsible for them, but SizeClass
    /// needs to call on them for eviction
    const std::span<std::unique_ptr<StorageManager>> storage_managers_;

    /// @brief Bitmask for whether the frame is currently mapped to
    /// real memory.
    std::vector<uint8_t> mapped_bitmask_;

    /// @brief Bitmask for whether the frame is pinned.
    std::vector<uint8_t> pinned_bitmask_;

    /// @brief Priority hint for the bitmask.
    /// Currently, this is 1 bit per frame. 0 implies
    /// high priority (e.g. hashmap) and 1 implies
    /// low priority.
    std::vector<uint8_t> priority_hint_;

    /// @brief Pointers to the swips for the frames.
    /// 'nullptr' for inactive frames, or frames
    /// without swip information (e.g. allocation
    /// done by STL containers).
    std::vector<OwningSwip> swips_;

    /// @brief See full description in BufferPoolOptions::spill_on_unpin.
    const bool spill_on_unpin_;

    /// @brief See full description in BufferPoolOptions::move_on_unpin.
    const bool move_on_unpin_;

    /// @brief Trace time spent in various functions
    const bool tracing_mode_;

    /// @brief Start of address range for this size class.
    uint8_t* const address_;

    /// @brief Helper function to mark the frame at
    /// index idx as "mapped", i.e. taken.
    inline void markFrameAsMapped(uint64_t idx);

    /// @brief Helper function to mark the frame at
    /// index idx as unmapped, i.e. not taken.
    inline void markFrameAsUnmapped(uint64_t idx);

    /// @brief Helper function to mark the frame at
    /// index idx as pinned.
    inline void markFrameAsPinned(uint64_t idx);

    /// @brief Helper function to mark the frame at
    /// index idx as unpinned.
    inline void markFrameAsUnpinned(uint64_t idx);

    /**
     * @brief Inform OS that the frame at index idx is
     * not required. We will do this using madvise
     * on the frame with the MADV_DONTNEED flag.
     */
    void adviseAwayFrame(uint64_t idx);

    /**
     * @brief Find the index of the first mapped unpinned
     * frame in this SizeClass. Note that this only finds
     * the frame but doesn't mark it as pinned.
     * This is only used in the move_on_unpin_ case at this point.
     *
     * @return int64_t Index of the first mapped unpinned frame. -1 if no such
     * frame was found.
     */
    int64_t findMappedUnpinnedFrame() const noexcept;

    /**
     * @brief Move data from the idx1'th frame to the idx2'th frame and update
     * the swips, including the swip owners pointing to these frames. Currently
     * this is only used in the move_on_unpin_ case and is only practical for
     * swapping two unpinned frames (swapping pinned frames is not unsafe).
     *
     * @param idx1
     * @param idx2
     */
    void swapFrames(uint64_t idx1, uint64_t idx2);

    /**
     * @brief Move data and swip (including swip owner pointing to the frame)
     * from the source_idx'th frame to the target_idx'th frame.
     * Currently this is only used in the move_on_unpin_ case. Note that
     * this is only really practical when the source_idx'th frame is
     * unpinned since moving pinned frames is not safe.
     * Also note that this function just moves the data and swip and doesn't
     * actually update the pinned/mapped bitmasks or advise away the frame.
     *
     * @param source_idx Mapped frame to move the data from.
     * @param target_idx The unmapped frame to move the data to.
     */
    void moveFrame(uint64_t source_idx, uint64_t target_idx);
};

/// @brief Top Level Buffer Pool Statistics
struct BufferPoolStats {
    /// @brief Current number of bytes allocated in the pool
    int64_t curr_bytes_allocated = 0;
    /// @brief Current number of bytes in memory owned by the pool
    /// curr_bytes_allocated - curr_bytes_in_memory = curr_bytes_spilled
    int64_t curr_bytes_in_memory = 0;
    /// @brief Current number of bytes allocated through malloc
    /// curr_bytes_allocated - curr_bytes_malloced = curr_bytes_mmapped
    int64_t curr_bytes_malloced = 0;
    /// @brief Current number of bytes pinned in the pool
    /// curr_bytes_in_memory - curr_bytes_pinned = curr_bytes_unpinned
    int64_t curr_bytes_pinned = 0;

    /// @brief Current number of allocations in the pool
    /// curr_bytes_allocated / curr_num_allocations = avg_allocation_size
    int64_t curr_num_allocations = 0;

    /// @brief Total number of bytes allocated in the pool
    uint64_t total_bytes_allocated = 0;
    /// @brief Total number of bytes requested through the pool
    /// total_bytes_requested / total_bytes_allocated = memory utilization
    uint64_t total_bytes_requested = 0;
    /// @brief Total number of bytes allocated through malloc
    /// total_bytes_allocated - total_bytes_malloced = total_bytes_mmapped
    uint64_t total_bytes_malloced = 0;
    /// @brief Total number of bytes pinned in all pin() calls by the pool
    /// No-op calls are ignored
    uint64_t total_bytes_pinned = 0;
    /// @brief Total number of bytes unpinned in all unpin() calls the pool
    /// No-op calls are ignored
    uint64_t total_bytes_unpinned = 0;
    /// @brief Total number of bytes that are reused through realloc
    uint64_t total_bytes_reallocs_reused = 0;

    /// @brief Total number of allocations performed by pool
    /// Includes allocations performed for realloc
    /// total_bytes_allocated / total_num_allocations = avg_allocation_size
    uint64_t total_num_allocations = 0;
    /// @brief Total number of reallocations performed by pool
    uint64_t total_num_reallocations = 0;
    /// @brief Total number of times allocations were pinned
    /// Note allocation are auto-pinned so
    /// total_num_pins - total_num_allocations = # BufferPool::Pin was called
    uint64_t total_num_pins = 0;
    /// @brief Total number of times allocations were unpinned
    /// Equivalent to # of times BufferPool::Unpin was called
    uint64_t total_num_unpins = 0;
    /// @brief Total number of times a spilled allocation was free'd
    uint64_t total_num_frees_from_spill = 0;
    /// @brief Total number of reallocations that are reused
    uint64_t total_num_reallocs_reused = 0;

    /// @brief Peak number of bytes allocated in the pool
    int64_t max_bytes_allocated = 0;
    /// @brief Peak number of bytes in memory owned by the pool
    int64_t max_bytes_in_memory = 0;
    /// @brief Peak number of bytes malloc-ed
    int64_t max_bytes_malloced = 0;
    /// @brief Peak number of bytes pinned in the pool
    int64_t max_bytes_pinned = 0;

    /// @brief Total time spent for allocations
    milli_double total_allocation_time{};
    /// @brief Total time spent for allocations through malloc
    /// Included in total_allocation_time
    milli_double total_malloc_time{};
    /// @brief Total time spent for reallocations
    milli_double total_realloc_time{};
    /// @brief Total time spent for frees
    milli_double total_free_time{};
    /// @brief Total time spent for pins
    milli_double total_pin_time{};
    /// @brief Total time spent for finding frames to evict
    milli_double total_find_evict_time{};

    void AddAllocatedBytes(uint64_t diff);

    void AddInMemoryBytes(uint64_t diff);

    void AddPinnedBytes(uint64_t diff);

    void AddMallocedBytes(uint64_t diff);

    /// @brief Print the current stats to file 'out'
    void Print(FILE* out) const;
};

class BufferPool final : public IBufferPool {
   public:
    /* ------ Functions from IBufferPool that we implement ------ */

    /// @brief Default constructor which will use the default
    /// options.
    explicit BufferPool();

    // Default is sufficient since we don't allocate
    // anything dynamically. We hold unique pointers
    // to the SizeClass instances, and so those will
    // get destroyed when this object does.
    ~BufferPool() override = default;

    using IBufferPool::Allocate;
    using IBufferPool::Free;
    using IBufferPool::Reallocate;

    /**
     * @brief Allocate a new memory region of at least 'size' bytes.
     * The allocated region will be 'alignment' byte aligned (64 by
     * default).
     *
     * @param size Number of bytes to allocate.
     * @param alignment Alignment that needs to be guaranteed for the
     * allocation.
     * @param[in, out] out Pointer to pointer which should store the address of
     * the allocated memory.
     */
    ::arrow::Status Allocate(int64_t size, int64_t alignment,
                             uint8_t** out) override;

    /**
     * @brief Resize an already allocated memory section.
     *
     * @param old_size Use -1 if previous size is not known.
     * @param new_size Number of bytes to allocate.
     * @param alignment Alignment that needs to be guaranteed for the
     * allocation.
     * @param[in, out] ptr Pointer to pointer which stores the address of the
     * previously allocated memory and should be modified to now store
     * the address of the new allocated memory region.
     */
    ::arrow::Status Reallocate(int64_t old_size, int64_t new_size,
                               int64_t alignment, uint8_t** ptr) override;

    /**
     * @brief Free an allocated region.
     *
     * @param buffer Pointer to the start of the allocated memory region
     * @param size Allocated size located at buffer. Pass -1 if size is not
     * known. If -1 is passed and if the memory was originally allocated using
     * malloc, we won't know the memory size and hence the stats won't be
     * updated.
     * @param alignment The alignment of the allocation. Defaults to 64 bytes.
     */
    void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;

    /**
     * @brief Pin an allocation/block to memory.
     *
     * @param[in, out] ptr Location of swip pointer containing
     *   allocation to pin.
     * @param size Size of the allocation (original requested size)
     * @param alignment Alignment used for the allocation
     * @return ::arrow::Status
     */
    ::arrow::Status Pin(
        uint8_t** ptr, int64_t size = -1,
        int64_t alignment = arrow::kDefaultBufferAlignment) override;

    /**
     * @brief Unpin an allocation/block. This makes the block eligible
     * for eviction when the BufferPool needs to free up
     * space in memory.
     *
     * @param[in, out] ptr Swip pointer to allocation to unpin
     * @param size Size of the allocation (original requested size)
     * @param alignment Alignment used for the allocation
     */
    void Unpin(uint8_t* ptr, int64_t size = -1,
               int64_t alignment = arrow::kDefaultBufferAlignment) override;

    /// @brief The number of bytes currently available in memory
    /// in the pool.
    uint64_t bytes_in_memory() const {
        return this->stats_.curr_bytes_in_memory;
    }

    /// @brief The number of bytes currently allocated through
    /// this allocator.
    int64_t bytes_allocated() const override;

    /// @brief Number of bytes allocated by the pool for its entire history
    int64_t total_bytes_allocated() const override {
        return this->stats_.total_bytes_allocated;
    }

    /// @brief Number of allocation performed by the pool for its entire history
    int64_t num_allocations() const override {
        return this->stats_.total_num_allocations;
    }

    /// @brief The number of bytes currently pinned.
    /// TODO: Get inline to work correctly
    uint64_t bytes_pinned() const override;

    /// @brief Get peak memory allocation in this memory pool
    int64_t max_memory() const override;

    /// @brief Get the SizeClass and frame index for a given memory allocation,
    /// if it was allocated by the BufferPool.
    std::optional<std::tuple<uint64_t, uint64_t>> alloc_loc(
        uint8_t* ptr, int64_t size = -1,
        int64_t alignment = arrow::kDefaultBufferAlignment) const override;

    /// @brief The name of the backend used by this memory pool.
    /// Always returns 'bodo'.
    std::string backend_name() const override;

    /// @brief If spilling is enabled for this memory pool.
    bool is_spilling_enabled() const;

    // ------------------ New functions ------------------ //

    /// @brief Construct a BufferPool using the provided options.
    /// @param options Options for the BufferPool.
    explicit BufferPool(const BufferPoolOptions& options);

    /// @brief Default Singleton Pool Object
    /// This is what will be used everywhere. The dynamically created pools
    /// (using make_shared) are strictly for unit-testing purposes.
    /// Ref:
    /// https://stackoverflow.com/a/40337728/15000847
    /// https://betterprogramming.pub/3-tips-for-using-singletons-in-c-c6822dc42649
    /// @return Singleton instance of the BufferPool
    static std::shared_ptr<BufferPool> Default() {
        return global_memory_pool_shared;
    }

    /// @brief Simple wrapper for getting a pointer to the BufferPool singleton
    /// @return Pointer to Singleton BufferPool
    static BufferPool* DefaultPtr() { return global_memory_pool; }

    /// Override the copy constructor and = operator as per the
    /// singleton pattern.
    BufferPool(BufferPool const&) = delete;
    BufferPool& operator=(BufferPool const&) = delete;

    /// @brief Cleanup any external resources
    /// Needs to be outside of the destructor to allow
    /// for exception propagation to work
    void Cleanup();

    // XXX Add Reserve/Release functions, or re-use planned
    // functions RegisterOffPoolUsage and DeregisterOffPoolUsage
    // for that (or rename those to be Reserve and Release).
    // (See Velox's reserve and release functions for reference).

    /// @brief Get the number of mmap-ed size-classes allocated by the Buffer
    /// Pool
    virtual size_t num_size_classes() const;

    /**
     * @brief Get a raw pointer to a SizeClass at index 'idx. This
     * can be unsafe since we're returning the underlying pointer
     * from a unique_ptr, i.e. there's no ownership transfer.
     * However, it should be safe to use this pointer until
     * the pool itself is deallocated.
     * This is meant to be used for testing purposes only.
     *
     * @param idx Index of the SizeClass to get a pointer to.
     * @return SizeClass* Raw pointer to the SizeClass instance.
     */
    virtual SizeClass* GetSizeClass_Unsafe(uint64_t idx) const;

    /**
     * @brief Get the size of the smallest SizeClass in bytes.
     * Will return 0 when there are no SizeClass-es.
     *
     * @return uint64_t
     */
    uint64_t GetSmallestSizeClassSize() const;

    /**
     * @brief Check if the buffer/swip is pinned.
     * If the swip is unswizzled, this will return false.
     * If the swip is swizzled, i.e. the allocation is still
     * in memory:
     * - If it's a frame in a SizeClass, we check there.
     * - If it was allocated through malloc, we return true
     *   since malloc allocations always stay pinned.
     *
     * @param buffer Pointer/Swip to the allocated buffer
     * @param size Size of the allocation (original requested size)
     * @param alignment Alignment used for the allocation
     */
    bool IsPinned(uint8_t* buffer, int64_t size = -1,
                  int64_t alignment = arrow::kDefaultBufferAlignment) const;

    /**
     * @brief Getter for memory_size_bytes_.
     *
     * @return uint64_t
     */
    uint64_t get_memory_size_bytes() const;

    /**
     * @brief Getter for options_.sys_mem_mib (converted from MiB to bytes).
     * Returns -1 if not known.
     *
     * @return int64_t
     */
    int64_t get_sys_memory_bytes() const;

    /**
     * @brief Getter for bytes_freed_through_malloc_since_last_trim_.
     *
     * @return int64_t
     */
    int64_t get_bytes_freed_through_malloc_since_last_trim() const;

    /// @brief Get the current allocation stats for the BufferPool
    boost::json::object get_stats() const;

    /// @brief Helper function for printing the stats to std::cerr.
    void print_stats();

   protected:
    /// @brief Options that were used for building the BufferPool.
    BufferPoolOptions options_;

    /// @brief Vector of unique-pointers to the SizeClass-es.
    /// Ordered by the frame size (ascending).
    std::vector<std::unique_ptr<SizeClass>> size_classes_;

    /// @brief Storage Managers to Spill Block To
    /// In order of priority. Only use next manager if all previous
    /// ones are out of space.
    /// Maximum number of storage managers allowed is 4
    std::vector<std::unique_ptr<StorageManager>> storage_managers_;

   private:
    /// @brief Current allocation stats. For now, tracks the
    /// - Current and Overall Total Number of Bytes Allocated, Pinned, etc
    /// - Current and Overall Number of Allocations, Pins, etc
    /// - Peak Number of Bytes Allocated, Pinned, etc at any point
    BufferPoolStats stats_;

    /// @brief Threshold for allocation through malloc. Allocations of this size
    /// and lower will go through malloc
    uint64_t malloc_threshold_;

    /// @brief Total memory size in bytes.
    uint64_t memory_size_bytes_;

    /// @brief Vector of block sizes of the allocated SizeClass-es for easy
    /// access.
    std::vector<uint64_t> size_class_bytes_;

    /// @brief Simple mutex to make the BufferPool thread-safe.
    /// Bodo itself doesn't use threading, but Arrow/PyArrow do use
    /// threading during IO, etc., so the BufferPool needs to be thread safe.
    /// TODO: Expand the mutex to allow for more flexible threading
    std::mutex mtx_;

    /// @brief Number of bytes that have been freed through malloc
    /// since the last time malloc_trim was called.
    /// NOTE: This is only used on Linux.
    int64_t bytes_freed_through_malloc_since_last_trim_ = 0;

    /**
     * @brief Returns a size that is 'alignment' aligned, essentially
     * rounding up to the closest multiple of 'alignment'.
     *
     * @param size Original size
     * @param alignment Alignment to use. Defaults to 64.
     * @return int64_t Aligned size
     */
    inline int64_t size_align(
        int64_t size, int64_t alignment = arrow::kDefaultBufferAlignment) const;

    /**
     * @brief Get the index of the size-class that would be most
     * appropriate for an allocation of 'size' bytes. This will
     * essentially find the smallest block-size that is greater than
     * or equal to size.
     *
     * @param size Allocation size
     * @return int64_t Size Class index (in size_classes_)
     */
    inline int64_t find_size_class_idx(int64_t size) const;

    /**
     * @brief Get details about allocation based on the memory address and
     * original allocation size.
     * In case the size is not provided (i.e. -1), we
     * will try to deduce the size. This can only be done in case the memory
     * was allocated through a SizeClass and not malloc.
     *
     * @param buffer Pointer to the memory region.
     * @param size Size of the allocation (from the caller perspective since we
     * might allocate and return a memory region that's greater than the
     * original request but the caller wouldn't know that).
     * @param alignment Alignment used for the allocation.
     * @return std::tuple<bool, int64_t, int64_t, int64_t>
     *  bool: Whether the allocation was made through mmap.
     *  int64_t: If the allocation was made through mmap, this
     *      will be the index of the size-class. Else -1
     *      is returned.
     *  int64_t: If the allocation was made through mmap, this
     *      will be the index of the frame inside the size-class.
     *      Else -1 will be returned.
     *  int64_t: Original allocation size, if it can be inferred. Else -1 will
     *      be returned. It it can be inferred, this will always be the
     *      "aligned" size.
     */
    std::tuple<bool, int64_t, int64_t, int64_t> get_alloc_details(
        uint8_t* buffer, int64_t size,
        int64_t alignment = arrow::kDefaultBufferAlignment) const;

    /**
     * @brief Helper for free-ing a memory allocation.
     * In the typical case we will provide the output of
     * 'get_alloc_details' to this function.
     *
     * @param ptr Pointer to the memory region to free.
     * @param is_mmap_alloc Whether the original allocation was done mmap (i.e.
     * a SizeClass).
     * @param size_class_idx Index of the Size-Class used for the allocation if
     * the allocation was done through mmap. If allocation was done through
     * malloc, set this to -1.
     * @param frame_idx Index of the frame in the SizeClass used for the
     * allocation (assuming the allocation was done through mmap). If allocation
     * was done through malloc, set this to -1.
     * @param size_aligned Aligned size of the allocation (from the BufferPool
     * perspective). In case of mmap allocation, this will be the frame size.
     * @param alignment The expected alignment of the memory. Used on Windows
     * to determine whether to use _aligned_free.
     */
    void free_helper(uint8_t* ptr, bool is_mmap_alloc, int64_t size_class_idx,
                     int64_t frame_idx, int64_t size_aligned,
                     int64_t alignment);

    /**
     * @brief Zero pad the extra bytes allocated during an allocation.
     *
     * @param ptr Pointer to the start of the allocated memory region.
     * @param size Size of the original allocation request.
     * @param capacity Capacity actually allocated.
     */
    inline void zero_padding(uint8_t* ptr, size_t size, size_t capacity);

    /**
     * @brief Try to evict 'bytes' many bytes.
     *
     * In case spilling is not enabled/available, it will throw a
     * runtime_error.
     *
     * Note that this function is best-effort. It will spill as many
     * bytes as possible, and return a boolean specifying whether or not
     * it was able to spill sufficient bytes. In case there are issues
     * while trying to spill, it will return a non-OK status.
     *
     * The ideal size-class is one that is >= bytes.
     * It will attempt to evict unpinned frames from size classes of
     * the ideal size and below if that would be enough. If
     * not, it will then try to evict frame of a larger size class.
     * If it cannot find a frame of larger size-class, it will
     * spill the frames identified earlier (from size classes of the
     * ideal size class and below) to reduce memory pressure as much as
     * possible.
     * This ensures that we strike a good balance between only evicting
     * the required amount of frames and relieving memory pressure.
     *
     * @param bytes Number of bytes to attempt to evict
     *
     * @return arrow::Result<bool> Whether we were able to spill sufficient
     * bytes.
     */
    arrow::Result<bool> best_effort_evict_helper(uint64_t bytes);

    /**
     * @brief Handle eviction of 'bytes' many bytes. This is handled based on
     * 'options_' (enforce_max_limit_during_allocation in particular) and
     * spilling config.
     *
     * Broadly, there are a few cases:
     * 1. Spilling is available: We will do a best effort spill to make enough
     *    memory available in the buffer pool. If there's an issue during this,
     *    we will forward the error status.
     *    If we are able to free up sufficient bytes, we will return an OK
     *    Status. If we aren't, we will handle this based on
     *    enforce_max_limit_during_allocation. If enforcement is on, we will
     *    return an OutOfMemory status. Otherwise, we will simply print
     *    a warning to stderr and return an OK status.
     * 2. If spilling isn't available, we will raise an OutOfMemory status
     *    if enforce_max_limit_during_allocation is true, else we will
     *    simply print a warning to stderr and return an OK status.
     *
     * @param bytes Number of bytes to evict
     * @param caller Name of the caller function. This will be used in the
     * warnings printed to stderr. e.g. 'Allocate' or 'Pin'.
     * @return ::arrow::Status
     */
    ::arrow::Status evict_handler(uint64_t bytes, const std::string& caller);

    /// @brief Construct a user-facing error message for OOMs.
    std::string oom_err_msg(uint64_t requested_bytes) const;
};

// Default STL allocator that plugs in the central buffer pool (used in bodo.ext
// module)
template <class T>
class DefaultSTLBufferPoolAllocator : public STLBufferPoolAllocator<T> {
   public:
    template <class U>
    struct rebind {
        using other = DefaultSTLBufferPoolAllocator<U>;
    };

    template <class U>
    DefaultSTLBufferPoolAllocator(
        const DefaultSTLBufferPoolAllocator<U>& other) noexcept
        : DefaultSTLBufferPoolAllocator(other.pool(), other.size()) {}

    template <class U>
    DefaultSTLBufferPoolAllocator(DefaultSTLBufferPoolAllocator<U>&& other)
        : DefaultSTLBufferPoolAllocator(other.pool(), other.size()) {}

    template <class U>
    DefaultSTLBufferPoolAllocator& operator=(
        const DefaultSTLBufferPoolAllocator<U>& other) {
        this->pool_ = other.pool();
        this->size_ = other.size();
    }

    template <class U>
    DefaultSTLBufferPoolAllocator& operator=(
        DefaultSTLBufferPoolAllocator<U>&& other) {
        this->pool_ = other.pool();
        this->size_ = other.size();
    }

    DefaultSTLBufferPoolAllocator(arrow::MemoryPool* pool, size_t size) noexcept
        : STLBufferPoolAllocator<T>(pool, size) {}

    DefaultSTLBufferPoolAllocator(arrow::MemoryPool* pool) noexcept
        : DefaultSTLBufferPoolAllocator(pool, 0) {}

    DefaultSTLBufferPoolAllocator() noexcept
        : STLBufferPoolAllocator<T>(BufferPool::DefaultPtr()) {}
};

template <typename T, class Allocator = DefaultSTLBufferPoolAllocator<T>>
using vector = std::vector<T, Allocator>;

/**
 * @brief Create a shared_ptr<T[]> whose underlying memory
 * is allocated through an IBufferPool instance.
 *
 * @tparam T Data type for the array.
 * @param size Number of elements of type T to allocate.
 * @param pool The pool to use to allocate the required memory.
 * @return std::shared_ptr<T[]> New allocation.
 */
template <typename T>
std::shared_ptr<T[]> make_shared_arr(
    size_t size, IBufferPool* const pool = BufferPool::DefaultPtr()) {
    T* ptr;  // By it's nature, this allocation will not be unpinnable, so this
             // is fine.
    CHECK_ARROW_MEM(pool->Allocate(sizeof(T) * size, (uint8_t**)(&ptr)),
                    "bodo::make_shared_arr: Allocation failed!");
    return std::shared_ptr<T[]>(
        ptr, [=](T* pt) { pool->Free((uint8_t*)(pt), sizeof(T) * size); });
}

/// Helper Functions for using BufferPool in Arrow

/**
 * @brief Construct an Arrow ExecContext for Compute Functions using the
 * specified 'pool'.
 *
 * @param pool Pool to create an ExecContext for.
 * @return ::arrow::compute::ExecContext*
 */
::arrow::compute::ExecContext* buffer_exec_context(bodo::IBufferPool* pool);

/**
 * @brief Construct an Arrow ExecContext for Compute Functions using the
 * default Bodo BufferPool.
 */
::arrow::compute::ExecContext* default_buffer_exec_context();

/**
 * @brief Construct an Arrow IOContext for IO Operations using the
 * specified 'pool'.
 *
 * @param pool Pool to create the IOContext for.
 * @return ::arrow::io::IOContext
 */
::arrow::io::IOContext buffer_io_context(bodo::IBufferPool* pool);

/**
 * @brief Construct an Arrow IOContext for IO Operations using the
 * default Bodo BufferPool.
 */
::arrow::io::IOContext default_buffer_io_context();

/**
 * @brief Construct an Arrow MemoryManager that allocates using the
 * specified 'pool'.
 *
 * @param pool Pool to create the MemoryManager for.
 * @return std::shared_ptr<::arrow::MemoryManager>
 */
std::shared_ptr<::arrow::MemoryManager> buffer_memory_manager(
    bodo::IBufferPool* pool);

/**
 * @brief Construct an Arrow MemoryManager that allocates using the
 * default Bodo BufferPool.
 */
std::shared_ptr<::arrow::MemoryManager> default_buffer_memory_manager();

template <typename Spillable, typename... Args>
class pin_guard;

template <typename Spillable, typename... Args>
pin_guard<Spillable, Args...> pin(Spillable& s, Args&&... args);

/**
 * @brief The pin() methods of a pinnable type can return a pointer-like value
 * that is used to provide smart pointer-like semantics to the pin_guard class.
 * This helper struct handles storing the pointed-to value, or, if the pin()
 * method returns void, nothing.
 */
template <typename X>
struct pinned_storage {
    using type = X;
};

/**
 * @brief When a pin() method returns void, nothing is stored
 */
template <>
struct pinned_storage<void> {
    struct type {};
};

/**
 * @brief This is an annoying template specialization so that we don't return
 * void()
 *
 * The template is specialized on the element_type of pin_guard<Spillable>.
 *
 * It takes a function that pins the spillable and simply calls it, and if not
 * void, returns the value.
 *
 * The function cannot take the spillable because it is not a friend of the
 * Spillable class and thus may not be able to call pin().
 *
 * on the other hand, pin_guard is allowed to create a lambda to access pin()
 * within its definition since pin_guard<Spillable> is a friend of Spillable.
 */
template <typename ElTy>
inline typename pinned_storage<ElTy>::type do_pin(std::function<ElTy()> pin) {
    return pin();
}

template <>
inline typename pinned_storage<void>::type do_pin<void>(
    std::function<void()> pin) {
    pin();
    return {};
}

/**
 * @brief Several classes support pinning or unpinning memory via pin() and
 * unpin() methods. This is brittle as c++ never guarantees that unpin() will
 * actually be called. It's better to use destructors to automatically handle
 * this.
 *
 * This class lets you automatically manage pinning for any class that has a
 * pin() method.
 */
template <typename Spillable, typename... Args>
class pin_guard {
   public:
    ~pin_guard() { release(); }

    using element_type =
        std::invoke_result<decltype(&Spillable::pin), Spillable*>::type;
    using reference_type = typename std::add_lvalue_reference<
        typename std::remove_pointer<element_type>::type>::type;
    using const_reference_type = typename std::add_lvalue_reference<
        typename std::add_const<reference_type>::type>::type;

    element_type operator->() const { return val_; }
    element_type get() const { return val_; }

    reference_type operator*() { return *val_; }
    element_type operator*() const { return *val_; }

    template <typename Sz>
    reference_type operator[](Sz sz) const {
        return get()[sz];
    }

    /// @brief Release the pinned value early. Any further attempts to
    /// de-reference or access this pinned (other than destruction) is undefined
    /// behavior.
    void release() {
        if (!released_) {
            released_ = true;
            std::apply(
                &Spillable::unpin,
                std::tuple_cat(std::make_tuple(&underlying_), pin_args_));
        }
    }

    pin_guard(const pin_guard& p) = delete;

    inline pin_guard(Spillable& s, Args&&... args)
        : underlying_(s),
          val_(do_pin<element_type>(
              std::bind(&Spillable::pin, &s, std::forward<Args>(args)...))),
          released_(false),
          pin_args_(std::forward<Args>(args)...) {}

   private:
    Spillable& underlying_;
    typename pinned_storage<element_type>::type val_;
    bool released_;

    std::tuple<Args&&...> pin_args_;

    friend pin_guard<Spillable, Args...> pin<Spillable>(Spillable& s);
};

/**
 * @brief Automatically manage pinning for any class  that has pin() and unpin()
 * methods
 *
 * That is, instead of doing:
 *   x.pin()
 *   ...
 *   x.unpin()
 *
 * Instead, do
 *
 *  {
 *   auto xPin(bodo::pin(x));
 *  }
 *
 * For maximum safety, do not expose a pin() or unpin() method on your class.
 * Instead make the methods private and add bodo::pin_guard<YourClass> as a
 * friend.
 *
 * @tparam Spillable The class whose pinning to manage. Inferred automatically.
 * Do not provide.
 * @param s The thing you want to pin
 * @return pin_guard<Spillable>
 */
template <typename Spillable, typename... Args>
pin_guard<Spillable, Args...> pin(Spillable& s, Args&&... args) {
    return pin_guard<Spillable, Args...>(s, std::forward<Args>(args)...);
}

}  // namespace bodo

#ifdef MS_WINDOWS

/**
 * @brief Get the total memory available on this node.
 * Note that this is the total memory size and not just
 * the unused memory.
 *
 */
inline size_t get_total_node_memory() {
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    auto memory_size = static_cast<size_t>(status.ullTotalPhys);
    return memory_size;
}

#else

#include <unistd.h>

/**
 * @brief Get the total memory available on this node.
 * Note that this is the total memory size and not just
 * the unused memory.
 *
 */
inline size_t get_total_node_memory() {
    unsigned long long memory_size;
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    memory_size = pages * page_size;
    return static_cast<size_t>(memory_size);
}

#endif
