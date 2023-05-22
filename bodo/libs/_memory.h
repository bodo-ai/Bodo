// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef BODO_MEMORY_INCLUDED
#define BODO_MEMORY_INCLUDED

#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/memory_pool.h>
#include <arrow/stl_allocator.h>
#include <iostream>
#include <mutex>

// TODO Tell the compiler that the branch is unlikely.
#define CHECK_ARROW_MEM(expr, msg)                                      \
    if (!(expr.ok())) {                                                 \
        std::string err_msg = std::string(msg) + " " + expr.ToString(); \
        throw std::runtime_error(err_msg);                              \
    }

// Fraction of the smallest mmap-ed SizeClass
// that should be used as threshold for allocation
// through malloc.
#define MALLOC_THRESHOLD_RATIO 0.75

namespace bodo {

// Copied from Velox
// (https://github.com/facebookincubator/velox/blob/8324ac7f1839db009def00e7450f38c2591dd4bb/velox/common/memory/MemoryAllocator.h#L192)
static constexpr uint16_t kMinAlignment = alignof(max_align_t);
static constexpr uint16_t kMaxAlignment = 64;

// A static piece of memory for 0-size allocations, so as to return
// an aligned non-null pointer. (Copied from Arrow)
static int64_t zero_size_area[1];
static uint8_t* const kZeroSizeArea =
    reinterpret_cast<uint8_t*>(&zero_size_area);

/// @brief Options for the Buffer Pool implementation
struct BufferPoolOptions {
    /// @brief Total available memory (in MiB) for the buffer pool.
    uint64_t memory_size = 200;

    /// @brief Size of the smallest size class (in KiB)
    /// Must be a power of 2.
    uint64_t min_size_class = 64;

    /// @brief Maximum number of size classes allowed.
    /// The actual number of size classes will be the minimum
    /// of this and the max size classes possible based
    /// on memory_size and min_size_class.
    /// Since this needs to be encodable in 6 bits,
    /// the max allowed value is 63.
    uint64_t max_num_size_classes = 21;

    /// @brief Whether or not to enforce the specified
    /// memory limit during allocations. This is useful
    /// until we have spill support.
    bool ignore_max_limit_during_allocation = false;

    // XXX BlockStorageConfig will be added later when
    // spilling support is added.

    static BufferPoolOptions Defaults();
};

typedef uint8_t* Swip;
typedef Swip* OwningSwip;

/// @brief Represents a range of virtual addresses used for allocating
/// buffer frames of a fixed size. Very similar to Velox's SizeClass
/// (https://github.com/facebookincubator/velox/blob/8324ac7f1839db009def00e7450f38c2591dd4bb/velox/common/memory/MmapAllocator.h#L141).
class SizeClass {
   public:
    /**
     * @brief Create a new mmap-ed SizeClass for the Buffer Pool.
     *
     * @param capacity Number of frames to allocate
     * @param block_size Size of each frame (in bytes).
     */
    SizeClass(size_t capacity, size_t block_size);

    /**
     * @brief Delete the SizeClass and unmap the allocated
     * virtual address space.
     *
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

   private:
    /// NOTE: The private functions are not thread safe
    /// on their own. They assume that the thread entered
    /// through some public function and obtained a
    /// lock to read/modify the state.

    /// @brief Number of buffer frames.
    const uint64_t capacity_;

    /// @brief Size of each frame (in bytes).
    const uint64_t block_size_;

    /// @brief Size in bytes of the address range.
    const size_t byteSize_;

    /// @brief Number of bytes for mapped_bitmask / pinned_bitmask /
    /// priority_hint
    const uint64_t bitmask_nbytes_;

    /// @brief Start of address range for this size class.
    uint8_t* address_;

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

    /**
     * @brief Helper function to mark the frame at
     * index idx as "mapped", i.e. taken.
     *
     */
    inline void markFrameAsMapped(uint64_t idx);

    /**
     * @brief Helper function to mark the frame at
     * index idx as unmapped, i.e. not taken.
     *
     */
    inline void markFrameAsUnmapped(uint64_t idx);

    /**
     * @brief Pin the frame at index idx.
     *
     */
    inline void pinFrame(uint64_t idx);

    /**
     * @brief Unpin the frame at index idx.
     *
     */
    inline void unpinFrame(uint64_t idx);

    /**
     * @brief Inform OS that the frame at index idx is
     * not required. We will do this using madvise
     * on the frame with the MADV_DONTNEED flag.
     *
     */
    void adviseAwayFrame(uint64_t idx);

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

    // Simple mutex to make the SizeClass state (various bitmaps, etc.)
    // thread-safe. Bodo itself doesn't use threading, but Arrow/PyArrow do use
    // threading during IO, etc., so the BufferPool needs to be thread safe.
    // Having the mutex at the SizeClass level reduces overall overheads
    // since allocations/frees in different SizeClass-es
    // won't slow down each other. This is similar to how Velox manages
    // thread safety in its Memory Allocator.
    std::mutex mtx_;
};

class BufferPool final : public ::arrow::MemoryPool {
   public:
    /* ------ Functions arrow::MemoryPool that we override ------ */

    /// @brief Default constructor which will use the default
    /// options.
    explicit BufferPool();

    // Default is sufficient since we don't allocate
    // anything dynamically. We hold unique pointers
    // to the SizeClass instances, and so those will
    // get destroyed when this object does.
    ~BufferPool() override = default;

    using ::arrow::MemoryPool::Allocate;
    using ::arrow::MemoryPool::Free;
    using ::arrow::MemoryPool::Reallocate;

    /**
     * @brief Allocate a new memory region of at least 'size' bytes.
     * The allocated region will be 'alignment' byte aligned (64 by
     * default).
     *
     * @param size Number of bytes to allocated
     * @param alignment Alignment that needs to be guaranteed for the
     * allocation.
     * @param[in, out] out Pointer to pointer which should store the address of
     * the allocated memory.
     */
    virtual ::arrow::Status Allocate(int64_t size, int64_t alignment,
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
    virtual ::arrow::Status Reallocate(int64_t old_size, int64_t new_size,
                                       int64_t alignment,
                                       uint8_t** ptr) override;

    /**
     * @brief Free an allocated region.
     *
     * NOTE: The signature has changed in Arrow-11, it will
     * need to be updated when we upgrade. The change is that now there is a
     * field to specify the alignment.
     *
     * @param buffer Pointer to the start of the allocated memory region
     * @param size Allocated size located at buffer. Pass -1 if size is not
     * known. If -1 is passed and if the memory was originally allocated using
     * malloc, we won't know the memory size and hence the stats won't be
     * updated.
     * @param alignment The alignment of the allocation. Defaults to 64 bytes.
     */
    virtual void Free(uint8_t* buffer, int64_t size,
                      int64_t alignment) override;

    /// @brief The number of bytes currently allocated through
    /// this allocator.
    virtual int64_t bytes_allocated() const override;

    /// @brief Get peak memory allocation in this memory pool
    virtual int64_t max_memory() const override;

    /// @brief The name of the backend used by this MemoryPool.
    /// Always returns 'bodo'.
    virtual std::string backend_name() const override;

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
        static std::shared_ptr<BufferPool> pool_(new BufferPool());
        return pool_;
    }

    /// @brief Simple wrapper for getting a pointer to the BufferPool singleton
    /// @return Pointer to Singleton BufferPool
    static BufferPool* DefaultPtr() {
        static BufferPool* pool = BufferPool::Default().get();
        return pool;
    }

    /// Override the copy constructor and = operator as per the
    /// singleton pattern.
    BufferPool(BufferPool const&) = delete;
    BufferPool& operator=(BufferPool const&) = delete;

    // XXX Add functions for Pin/Unpin when we add eviction/swizzling support.

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

   protected:
    /// @brief Options that were used for building the BufferPool.
    BufferPoolOptions options_;

    // Re-using Arrow's MemoryPoolStats for now.
    // This will be replaced with our own stats class (that extends this)
    // when we have custom stats (like number of evictions, number of
    // deallocations, etc.) that we want to track.
    ::arrow::internal::MemoryPoolStats stats_;

    /// @brief Vector of unique-pointers to the SizeClass-es.
    /// Ordered by the frame size (ascending).
    std::vector<std::unique_ptr<SizeClass>> size_classes_;

   private:
    /// @brief Threshold for allocation through malloc. Allocations of this size
    /// and lower will go through malloc
    uint64_t malloc_threshold_;

    /// @brief Total memory size in bytes.
    uint64_t memory_size_bytes_;

    /// @brief Vector of block sizes of the allocated SizeClass-es for easy
    /// access.
    std::vector<uint64_t> size_class_bytes_;

    /**
     * @brief Helper function for initializing
     * the BufferPool. This is called from the constructor.
     */
    void Initialize();

    /**
     * @brief Returns a size that is 'alignment' aligned, essentially
     * rounding up to the closest multiple of 'alignment'.
     *
     * @param size Original size
     * @param alignment Alignment to use. Defaults to 64.
     * @return int64_t Aligned size
     */
    int64_t size_align(int64_t size,
                       int64_t alignment = arrow::kDefaultBufferAlignment);

    /**
     * @brief Get the index of the size-class that would be most
     * appropriate for an allocation of 'size' bytes. This will
     * essentially find the smallest block-size that is greater than
     * or equal to size.
     *
     * @param size Allocation size
     * @return int64_t Size Class index (in size_classes_)
     */
    int64_t find_size_class_idx(int64_t size) const;

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
        int64_t alignment = arrow::kDefaultBufferAlignment);

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
     */
    void free_helper(uint8_t* ptr, bool is_mmap_alloc, int64_t size_class_idx,
                     int64_t frame_idx, int64_t size_aligned);

    /**
     * @brief Zero pad the extra bytes allocated during an allocation.
     *
     * @param ptr Pointer to the start of the allocated memory region.
     * @param size Size of the original allocation request.
     * @param capacity Capacity actually allocated.
     */
    inline void zero_padding(uint8_t* ptr, size_t size, size_t capacity);
};

/// Helper Tools for using BufferPool in STL Containers
/// Based on ::arrow::stl::allocator class. We did not extend from that
/// class due to buggy behavior when the destructor was being called
/// too early.

template <class T>
class STLBufferPoolAllocator {
   public:
    using value_type = T;
    using is_always_equal = std::true_type;

    template <class U>
    struct rebind {
        using other = STLBufferPoolAllocator<U>;
    };

    explicit STLBufferPoolAllocator() noexcept = default;

    template <class U>
    STLBufferPoolAllocator(const STLBufferPoolAllocator<U>& rhs) noexcept {
        (void)rhs;
    }

    ~STLBufferPoolAllocator() = default;

    STLBufferPoolAllocator(const STLBufferPoolAllocator& other) = default;
    STLBufferPoolAllocator(STLBufferPoolAllocator&& other) = default;
    STLBufferPoolAllocator& operator=(const STLBufferPoolAllocator& other) =
        default;
    STLBufferPoolAllocator& operator=(STLBufferPoolAllocator&& other) = default;

    T* address(T& r) const noexcept { return std::addressof(r); }

    const T* address(const T& r) const noexcept { return std::addressof(r); }

    template <class U>
    inline bool operator==(const STLBufferPoolAllocator<U>& rhs) {
        return true;
    }

    template <class U>
    inline bool operator!=(const STLBufferPoolAllocator<U>& rhs) {
        return false;
    }

    T* allocate(size_t n) {
        uint8_t* data;
        ::arrow::Status s =
            bodo::BufferPool::DefaultPtr()->Allocate(n * sizeof(T), &data);
        if (!s.ok()) {
            throw std::bad_alloc();
        }
        return reinterpret_cast<T*>(data);
    }

    void deallocate(T* p, size_t n) {
        bodo::BufferPool::DefaultPtr()->Free(reinterpret_cast<uint8_t*>(p),
                                             n * sizeof(T));
    }

    size_t size_max() const noexcept { return size_t(-1) / sizeof(T); }

    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        new (reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <class U>
    void destroy(U* p) {
        p->~U();
    }
};

template <typename T>
using vector = std::vector<T, STLBufferPoolAllocator<T>>;

/// Helper Functions for using BufferPool in Arrow

/**
 * @brief Construct an Arrow ExecContext for Compute Functions using the
 * underlying Bodo BufferPool.
 */
::arrow::compute::ExecContext* buffer_exec_context();

/**
 * @brief Construct an Arrow IOContext for IO Operations using the
 * underlying Bodo BufferPool.
 */
::arrow::io::IOContext buffer_io_context();

/**
 * @brief Construct an Arrow MemoryManager that allocates using the
 * underlying Bodo BufferPool.
 */
std::shared_ptr<::arrow::MemoryManager> buffer_memory_manager();

}  // namespace bodo

#ifdef MS_WINDOWS

#define NOMINMAX
#include <windows.h>

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

#endif  // #ifndef BODO_MEMORY_INCLUDED
