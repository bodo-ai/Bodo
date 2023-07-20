// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef BODO_MEMORY_INCLUDED
#define BODO_MEMORY_INCLUDED

#include <atomic>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <span>

#include <arrow/compute/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/io/api.h>
#include <arrow/memory_pool.h>
#include <arrow/stl_allocator.h>

#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>          // streaming operators etc.

#include <mpi.h>

// TODO Tell the compiler that the branch is unlikely.
#define CHECK_ARROW_MEM(expr, msg)                                      \
    if (!(expr.ok())) {                                                 \
        std::string err_msg = std::string(msg) + " " + expr.ToString(); \
        throw std::runtime_error(err_msg);                              \
    }

#define CHECK_ARROW_MEM_RET(expr, msg)                                      \
    {                                                                       \
        auto stat = expr;                                                   \
        if (!stat.ok()) {                                                   \
            std::string err_msg = std::string(msg) + " " + stat.ToString(); \
            return stat.WithMessage(err_msg);                               \
        }                                                                   \
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

/// @brief Enum to Indicate which Manager to Use
enum StorageType : uint8_t { Local = 0 };

/// @brief Options for Storage Manager Implementations
struct StorageOptions {
    /// @brief Amount of bytes allowed to be spilled to
    /// storage location
    int64_t usable_size = 1024ll * 1024ll * 1024ll;

    /// @brief Location / folder to write block spill files
    std::string location;

    /// @brief Type of StorageManager to use
    StorageType type = StorageType::Local;

    static std::shared_ptr<StorageOptions> Defaults(uint8_t tier);
};

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
    /// for debugging purposes.
    bool ignore_max_limit_during_allocation = false;

    /// @brief Config for Storage Managers
    std::vector<std::shared_ptr<StorageOptions>> storage_options;

    static BufferPoolOptions Defaults();
};

/// Abstract Class / Interface for Storage Managers
/// Storage Managers manage the reading + writing of blocks
/// from a storage location as well as size limitations
class StorageManager {
   public:
    StorageManager(const std::shared_ptr<StorageOptions> options)
        : options(options) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Generate Unique UUID per Rank
        boost::uuids::uuid _uuid = boost::uuids::random_generator()();
        std::string uuid = boost::uuids::to_string(_uuid);

        this->uuid = std::to_string(rank) + "-" + uuid;
    }

    // Required otherwise won't compile
    // All cleanup operations occur within the Cleanup virtual function
    // so that exceptions can be safely raised
    // Cleanup runs at program exit. However, if we never want cleanup
    // to raise errors, we should move it back to the destructor
    // TODO: Revisit in the future
    virtual ~StorageManager() = default;

    /// @brief Initialize the Manager by Setting Up any
    /// Background Resources
    virtual arrow::Status Initialize() = 0;

    /// @brief How many bytes are available to be stored in this
    /// storage location. -1 indicates unlimited
    inline int64_t usable_size() const { return options->usable_size; }

    /**
     * @brief Read a block with id block_id and size of n_bytes
     * from storage, write its contents to out_ptr, and delete
     * block from storage.
     *
     * @param[in] block_id Index of block to read from storage
     * @param[in] n_bytes Size of block in bytes
     * @param[out] out_ptr Location to write contents to
     * @return arrow::Status Ok if success, otherwise a potential
     * filesystem error raised during read.
     */
    virtual arrow::Status ReadBlock(uint64_t block_id, int64_t n_bytes,
                                    uint8_t* out_ptr) = 0;

    /**
     * @brief Write the contents of a frame located at in_ptr
     * and size frame_size to storage, and return the new block id.
     *
     * @param in_ptr Location of frame to write
     * @param frame_size Size of frame in bytes
     * @return arrow::Result<uint64_t> Block ID of contents if success.
     * Otherwise a potential filesystem error raised during write
     */
    virtual arrow::Result<uint64_t> WriteBlock(uint8_t* in_ptr,
                                               uint64_t frame_size) = 0;

    /**
     * @brief Delete a block with id block_id and size of n_bytes from
     * storage.
     *
     * @param block_id Index of block to delete from storage
     * @param n_bytes Size of block in bytes
     * @return arrow::Status Ok if success, otherwise a potential
     * filesystem error raised during delete.
     */
    virtual arrow::Status DeleteBlock(uint64_t block_id, int64_t n_bytes) = 0;

    /**
     * @brief Is there space available in this storage location for
     * allocations to be spilled to it
     *
     * @param amount Bytes to potentially be spilled
     * @return true If allocation can be spilled to it
     * @return false If allocation can not be spilled here
     */
    bool CanSpillTo(uint64_t amount) {
        return options->usable_size < 0 ||
               (curr_spilled_bytes + amount) <=
                   static_cast<uint64_t>(options->usable_size);
    }

    /**
     * @brief Update the current number of spilled bytes to this
     * storage location by diff
     *
     * @param diff The number of bytes added or removed from this
     * storage location.
     */
    void UpdateSpilledBytes(int64_t diff) { curr_spilled_bytes += diff; }

    /// @brief Get the next available block id for this storage location
    uint64_t GetNewBlockID() { return this->block_id_counter++; }

    /// @brief Cleanup any leftover spill files
    /// Expected to run during program exit and can throw
    /// an error on fail
    virtual void Cleanup() = 0;

   protected:
    /// @brief Rank and process-unique identifier for
    /// spilling. Can be used for location handling
    std::string uuid;

    /// @brief Configuration Options
    std::shared_ptr<StorageOptions> options;

   private:
    /// @brief Increment every time we write a block to disk
    uint64_t block_id_counter = 0;

    /// @brief Current # of bytes spilled to storage
    uint64_t curr_spilled_bytes = 0;
};

/// Storage Manager for Local Disk-based Filesystems
class LocalStorageManager final : public StorageManager {
   public:
    explicit LocalStorageManager(const std::shared_ptr<StorageOptions> options)
        : StorageManager(options), location(options->location) {}

    virtual arrow::Status Initialize() override {
        return this->fs.CreateDir((this->location / this->uuid).string());
    }

    // Required otherwise won't compile
    virtual ~LocalStorageManager() override = default;

    virtual arrow::Status ReadBlock(uint64_t block_id, int64_t n_bytes,
                                    uint8_t* out_ptr) override;

    virtual arrow::Result<uint64_t> WriteBlock(uint8_t* in_ptr,
                                               uint64_t frame_size) override;

    virtual arrow::Status DeleteBlock(uint64_t block_id,
                                      int64_t n_bytes) override;

    virtual void Cleanup() override {
        CHECK_ARROW_MEM(
            this->fs.DeleteDir((this->location / this->uuid).string()),
            "LocalStorageManager::Cleanup: Failed to delete spill directory");
    }

   private:
    /// @brief Location to write spill contents to
    std::filesystem::path location;

    /// @brief LocalFileSystem handler for reading and writing
    ::arrow::fs::LocalFileSystem fs;
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
     */
    SizeClass(uint8_t idx,
              const std::span<std::unique_ptr<StorageManager>> storage_managers,
              size_t capacity, size_t block_size);

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
     */
    void UnpinFrame(uint64_t idx);

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
     * @return OK or potentially raised filesystem error
     */
    arrow::Status ReadbackToFrame(OwningSwip swip, uint64_t frame_idx,
                                  uint64_t block_idx, uint8_t manager_idx);

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
     * @param buffer Pointer to the start of the allocated memory region
     * @param size Allocated size located at buffer. Pass -1 if size is not
     * known. If -1 is passed and if the memory was originally allocated using
     * malloc, we won't know the memory size and hence the stats won't be
     * updated.
     * @param alignment The alignment of the allocation. Defaults to 64 bytes.
     */
    virtual void Free(uint8_t* buffer, int64_t size,
                      int64_t alignment) override;

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

    /// @brief The number of bytes currently allocated through
    /// this allocator.
    virtual int64_t bytes_allocated() const override;

    /// @brief The number of bytes currently pinned.
    /// TODO: Get inline to work correctly
    uint64_t bytes_pinned() const override;

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

    /// @brief Cleanup any external resources
    /// Needs to be outside of the destructor to allow
    /// for exception propagation to work
    void Cleanup() {
        for (auto& manager : this->storage_managers_) {
            manager->Cleanup();
        }
    }

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

    /// @brief Storage Managers to Spill Block To
    /// In order of priority. Only use next manager if all previous
    /// ones are out of space.
    /// Maximum number of storage managers allowed is 4
    std::vector<std::unique_ptr<StorageManager>> storage_managers_;

   private:
    /// @brief Threshold for allocation through malloc. Allocations of this size
    /// and lower will go through malloc
    uint64_t malloc_threshold_;

    /// @brief Total memory size in bytes.
    uint64_t memory_size_bytes_;

    /// @brief Number of bytes currently pinned
    /// TODO: Integrate into stats_ class at some point?
    std::atomic<uint64_t> bytes_pinned_;

    /// @brief Vector of block sizes of the allocated SizeClass-es for easy
    /// access.
    std::vector<uint64_t> size_class_bytes_;

    /// @brief Simple mutex to make the BufferPool thread-safe.
    /// Bodo itself doesn't use threading, but Arrow/PyArrow do use
    /// threading during IO, etc., so the BufferPool needs to be thread safe.
    /// TODO: Expand the mutex to allow for more flexible threading
    std::mutex mtx_;

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
    int64_t size_align(
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

    /**
     * @brief Evict enough memory for a frame from size class
     * size_class_idx.
     *
     * This function assumes that there is not enough available
     * memory to allocate a frame of the specified size class. It
     * will attempt to evict unpinned frames from size classes of
     * the specified size and below if that would be enough. If
     * not, it will then try to evict frames of larger size classes.
     *
     * @param size_class_idx Size Class of Frame to make space
     * available for
     */
    arrow::Status evict(uint64_t size_class_idx);

    /**
     * @brief Atomically update the number of pinned bytes in the
     * BufferPool by the diff.
     *
     * @param diff The number of bytes to add to the current pinned
     * byte count.
     */
    inline void update_pinned_bytes(int64_t diff);
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

    STLBufferPoolAllocator(IBufferPool* pool) noexcept : pool_(pool) {}

    STLBufferPoolAllocator() noexcept
        : STLBufferPoolAllocator(BufferPool::DefaultPtr()) {}

    ~STLBufferPoolAllocator() {}

    template <class U>
    STLBufferPoolAllocator(const STLBufferPoolAllocator<U>& other) noexcept
        : STLBufferPoolAllocator(other.pool()) {}

    template <class U>
    STLBufferPoolAllocator(STLBufferPoolAllocator<U>&& other)
        : STLBufferPoolAllocator(other.pool()) {}

    template <class U>
    STLBufferPoolAllocator& operator=(const STLBufferPoolAllocator<U>& other) {
        this->pool_ = other.pool();
    }

    template <class U>
    STLBufferPoolAllocator& operator=(STLBufferPoolAllocator<U>&& other) {
        this->pool_ = other.pool();
    }

    T* address(T& r) const noexcept { return std::addressof(r); }

    const T* address(const T& r) const noexcept { return std::addressof(r); }

    template <class U>
    inline bool operator==(const STLBufferPoolAllocator<U>& rhs) {
        return this->pool_ == rhs.pool();
    }

    template <class U>
    inline bool operator!=(const STLBufferPoolAllocator<U>& rhs) {
        return this->pool_ != rhs.pool();
    }

    T* allocate(size_t n) {
        uint8_t* data;
        ::arrow::Status s = this->pool_->Allocate(n * sizeof(T), &data);
        if (!s.ok()) {
            throw std::bad_alloc();
        }
        return reinterpret_cast<T*>(data);
    }

    void deallocate(T* p, size_t n) {
        this->pool_->Free(reinterpret_cast<uint8_t*>(p), n * sizeof(T));
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

    IBufferPool* pool() const noexcept { return this->pool_; }

   private:
    IBufferPool* const pool_;
};

template <typename T>
using vector = std::vector<T, STLBufferPoolAllocator<T>>;

template <typename K, typename V, typename HASH_FCT = std::hash<K>,
          typename EQ_FCT = std::equal_to<K>>
using unordered_multimap =
    std::unordered_multimap<K, V, HASH_FCT, EQ_FCT,
                            STLBufferPoolAllocator<std::pair<const K, V>>>;

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
