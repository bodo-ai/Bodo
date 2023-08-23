#pragma once
#include "_memory.h"

namespace bodo {

/**
 * @brief BufferPool for tracking pinned memory at the
 * operator level. See design doc here:
 * https://bodo.atlassian.net/l/cp/VmYX8g3A
 *
 * NOTE: We only track the requested sizes at the OperatorBufferPool level.
 * The BufferPool will most likely allocate more memory than requested.
 * We could technically modify BufferPool to provide this information,
 * but we aren't doing that at this point to keep the implementation
 * simple. We might change this in the future if it becomes important
 * to track actual allocation size.
 *
 */
class OperatorBufferPool final : public IBufferPool {
   public:
    /**
     * @brief Constructor for OperatorBufferPool.
     *
     * @param max_pinned_size_bytes Max amount of bytes that
     * can be pinned at any point in time by this OperatorBufferPool.
     * @param parent_pool The parent pool that this OperatorBufferPool
     * is going to allocate/pin/reallocate/unpin/free through.
     * @param error_threshold The fraction of the available pinned
     * bytes limit at which OperatorPoolThresholdExceededError will
     * be raised when threshold enforcement is enabled.
     */
    explicit OperatorBufferPool(
        uint64_t max_pinned_size_bytes,
        std::shared_ptr<BufferPool> parent_pool = BufferPool::Default(),
        double error_threshold = 0.5)
        : parent_pool_(std::move(parent_pool)),
          max_pinned_size_bytes_(max_pinned_size_bytes),
          memory_error_threshold_(
              static_cast<uint64_t>(error_threshold * max_pinned_size_bytes)) {}

    /**
     * @brief Allocates a buffer of 'size' bytes. The allocated buffer will be
     * 'alignment' byte aligned (64 by default).
     * This will raise the OperatorPoolThresholdExceededError if
     * size + currently pinned bytes > memory_error_threshold (assuming
     * threshold enforcement is enabled).
     * It will similarly return an Arrow OutOfMemory Status if
     * size + currently pinned bytes > max_pinned_size_bytes.
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
     * @brief Free the specified 'buffer'.
     *
     * @param buffer Pointer to the start of the allocated memory region.
     * @param size Allocated size located at buffer.
     * @param alignment The alignment of the allocation. Defaults to 64 bytes.
     */
    virtual void Free(uint8_t* buffer, int64_t size,
                      int64_t alignment) override;

    /**
     * @brief Resize an already allocated buffer.
     *
     * NOTE: If pinning both the new buffer (new_size) and
     * old buffer (old_size) together would take the amount of
     * pinned bytes over the threshold and threshold enforcement
     * is enabled, it will raise the
     * OperatorPoolThresholdExceededError error. Similarly for
     * total pinned bytes limit where an Arrow OutOfMemory Status
     * will be returned.
     *
     * @param old_size Previous allocation size.
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

    /// @brief The number of bytes currently allocated through
    /// this allocator.
    virtual int64_t bytes_allocated() const override;

    /// @brief The number of bytes currently pinned.
    uint64_t bytes_pinned() const override;

    /// @brief Get peak memory allocation in this memory pool
    virtual int64_t max_memory() const override;

    /// @brief The name of the backend used by the parent pool.
    virtual std::string backend_name() const override;

    /// @brief If spilling is supported by the parent pool.
    bool is_spilling_enabled() const;

    /**
     * @brief Getter for max_pinned_size_bytes
     * attribute.
     *
     * @return uint64_t
     */
    uint64_t get_max_pinned_size_bytes() const;

    /**
     * @brief Getter for memory_error_threshold
     * attribute.
     *
     * @return uint64_t
     */
    uint64_t get_memory_error_threshold() const;

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
     * for eviction when the parent pool needs to free up
     * space in memory.
     *
     * @param[in, out] ptr Swip pointer to allocation to unpin
     * @param size Size of the allocation (original requested size)
     * @param alignment Alignment used for the allocation
     */
    void Unpin(uint8_t* ptr, int64_t size = -1,
               int64_t alignment = arrow::kDefaultBufferAlignment) override;

    /**
     * @brief Check if threshold enforcement enabled.
     *
     * @return true
     * @return false
     */
    bool ThresholdEnforcementEnabled() const;

    /**
     * @brief Disable threshold enforcement.
     *
     */
    void DisableThresholdEnforcement() noexcept;

    /**
     * @brief Enable threshold enforcement.
     *
     */
    void EnableThresholdEnforcement();

    /**
     * @brief Get the parent pool.
     *
     * @return std::shared_ptr<BufferPool>
     */
    std::shared_ptr<BufferPool> get_parent_pool() const;

    /**
     * @brief Custom runtime error to throw when the
     * pinned amount would exceed the threshold.
     *
     */
    class OperatorPoolThresholdExceededError : public std::runtime_error {
       public:
        OperatorPoolThresholdExceededError()
            : std::runtime_error(
                  "OperatorPoolThresholdExceededError: Tried allocating more "
                  "space than what's allowed to be pinned!") {}
    };

   protected:
    /// @brief Allocation stats
    ::arrow::internal::MemoryPoolStats stats_;

   private:
    /// @brief Parent pool through which all the allocations, etc. will go
    /// through.
    const std::shared_ptr<BufferPool> parent_pool_;
    /// @brief Total memory size in bytes that's allowed to
    /// be pinned at any point in time.
    const uint64_t max_pinned_size_bytes_;
    /// @brief The memory threshold (in bytes) to enforce when threshold
    /// enforcement is enabled.
    const uint64_t memory_error_threshold_;
    /// @brief Flag for threshold enforcement.
    bool threshold_enforcement_enabled_ = true;
    /// @brief Number of bytes currently pinned
    /// TODO: Integrate into stats_ class at some point?
    std::atomic<uint64_t> bytes_pinned_;

    /**
     * @brief Atomically update the number of pinned bytes in the
     * BufferPool by the diff.
     *
     * @param diff The number of bytes to add to the current pinned
     * byte count.
     */
    inline void update_pinned_bytes(int64_t diff);

    /**
     * @brief Check if 'size' bytes can be pinned without
     * overflowing any limits.
     * If threshold enforcement is enabled, we first check
     * if currently pinned bytes + size would be within the
     * memory threshold ('memory_error_threshold_'). If it
     * wouldn't be, we throw the custom
     * OperatorPoolThresholdExceededError error. Even if
     * threshold enforcement is disabled, we verify that
     * currently pinned bytes + size is within total pinned
     * bytes limit ('max_pinned_size_bytes_').
     *
     * @param size Number of bytes to pin.
     * @return ::arrow::Status OK if 'size' can be pinned
     * while being within limits. If it would cross
     * 'max_pinned_size_bytes_', we return an
     * 'OutOfMemory' status.
     */
    inline ::arrow::Status check_limits(int64_t size) const;
};

}  // namespace bodo
