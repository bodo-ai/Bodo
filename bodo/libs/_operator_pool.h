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
     * @param operator_id_ Operator ID of the operator this pool is associated
     * with.
     * @param operator_budget_bytes The budget assigned to this operator.
     * The absolute threshold values will be based on this budget.
     * @param parent_pool The parent pool that this OperatorBufferPool
     * is going to allocate/pin/reallocate/unpin/free through.
     * @param error_threshold The fraction of the operator budget
     * at which OperatorPoolThresholdExceededError will be raised
     * when threshold enforcement is enabled.
     */
    explicit OperatorBufferPool(
        int64_t operator_id_, uint64_t operator_budget_bytes,
        std::shared_ptr<BufferPool> parent_pool = BufferPool::Default(),
        double error_threshold = 0.5)
        : operator_id(operator_id_),
          parent_pool_(std::move(parent_pool)),
          operator_budget_bytes_(operator_budget_bytes) {
        this->SetErrorThreshold(error_threshold);
    }

    /// @brief Operator ID of the operator this pool is associated with. This
    /// will primarily be used to interface with the OperatorComptroller to get
    /// and update budgets for this operator.
    const int64_t operator_id = -1;

    /**
     * @brief Change the error threshold. The absolute error
     * threshold (in bytes) will be error_threshold * operator-budget.
     * Note that if threshold enforcement is enabled and the
     * number of bytes pinned is greater than what the new
     * threshold would be, we will raise a
     * OperatorPoolThresholdExceededError. The threshold
     * won't be updated in this case.
     * NOTE: We won't try to get an increased budget from the
     * OperatorComptroller in this case.
     *
     * @param error_threshold New threshold. Must be <= 1.0
     */
    void SetErrorThreshold(double error_threshold);

    /**
     * @brief Update the budget for this operator pool.
     * At this point, only decreasing the budget is supported, i.e.
     * new_operator_budget must be lesser than the existing budget.
     * Note that if threshold enforcement is enabled and the
     * number of bytes pinned is greater than what the new
     * threshold would be, we will raise a
     * OperatorPoolThresholdExceededError. The budget
     * won't be updated in this case.
     * NOTE: We won't try to get an increased budget from the
     * OperatorComptroller in this case.
     *
     * If the budget update is successful, this will also inform
     * the OperatorComptroller about this updated budget so that
     * this budget can be made available to another operator
     * that might need it.
     *
     * @param new_operator_budget New budget for this operator.
     */
    void SetBudget(uint64_t new_operator_budget);

    /**
     * @brief Allocates a buffer of 'size' bytes (from the main mem portion).
     * The allocated buffer will be 'alignment' byte aligned (64 by default).
     * This will raise the OperatorPoolThresholdExceededError if
     * size + currently main mem pinned bytes > memory_error_threshold (assuming
     * threshold enforcement is enabled).
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
     * @brief Allocates a buffer of 'size' bytes (from the scratch mem portion).
     * The allocated buffer will be 'alignment' byte aligned (64 by default).
     * This will raise the OperatorPoolThresholdExceededError if
     * size + currently pinned bytes > budget (assuming
     * threshold enforcement is enabled).
     *
     * @param size Number of bytes to allocate.
     * @param alignment Alignment that needs to be guaranteed for the
     * allocation.
     * @param[in, out] out Pointer to pointer which should store the address of
     * the allocated memory.
     */
    ::arrow::Status AllocateScratch(int64_t size, int64_t alignment,
                                    uint8_t** out);

    /**
     * @brief Free the specified 'buffer'. This assumes that the allocation
     * came from the main mem portion.
     *
     * @param buffer Pointer to the start of the allocated memory region.
     * @param size Allocated size located at buffer.
     * @param alignment The alignment of the allocation. Defaults to 64 bytes.
     */
    void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;

    /**
     * @brief Free the specified 'buffer'. This assumes that the allocation
     * came from the scratch mem portion.
     *
     * @param buffer Pointer to the start of the allocated memory region.
     * @param size Allocated size located at buffer.
     * @param alignment The alignment of the allocation. Defaults to 64 bytes.
     */
    void FreeScratch(uint8_t* buffer, int64_t size, int64_t alignment);

    /**
     * @brief Resize an already allocated buffer, assuming the original
     * was from the main mem portion. The new one will come from the
     * main mem portion as well.
     *
     * NOTE: If pinning both the new buffer (new_size) and
     * old buffer (old_size) together would take the amount of
     * main mem pinned bytes over the threshold and threshold enforcement
     * is enabled, it will raise the
     * OperatorPoolThresholdExceededError error.
     *
     * @param old_size Previous allocation size.
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
     * @brief Resize an already allocated buffer, assuming the original
     * was from the scratch mem portion. The new one will come from the
     * scratch mem portion as well.
     *
     * NOTE: If pinning both the new buffer (new_size) and
     * old buffer (old_size) together would take the amount of
     * total pinned bytes over the budget and threshold enforcement
     * is enabled, it will raise the
     * OperatorPoolThresholdExceededError error.
     *
     * @param old_size Previous allocation size.
     * @param new_size Number of bytes to allocate.
     * @param alignment Alignment that needs to be guaranteed for the
     * allocation.
     * @param[in, out] ptr Pointer to pointer which stores the address of the
     * previously allocated memory and should be modified to now store
     * the address of the new allocated memory region.
     */
    ::arrow::Status ReallocateScratch(int64_t old_size, int64_t new_size,
                                      int64_t alignment, uint8_t** ptr);

    /// @brief The number of bytes currently allocated through
    /// this allocator.
    int64_t bytes_allocated() const override;

    /// @brief The number of bytes currently allocated through
    /// the main mem portion this allocator.
    int64_t main_mem_bytes_allocated() const;

    /// @brief The number of bytes currently allocated through
    /// the main scratch portion this allocator.
    int64_t scratch_mem_bytes_allocated() const;

    /// @brief The number of bytes currently pinned.
    uint64_t bytes_pinned() const override;

    /// @brief Get the SizeClass and frame index for a given memory allocation,
    /// if it was allocated by the BufferPool.
    std::optional<std::tuple<uint64_t, uint64_t>> alloc_loc(
        uint8_t* ptr, int64_t size = -1,
        int64_t alignment = arrow::kDefaultBufferAlignment) const override;

    /// @brief The number of bytes currently pinned in the
    /// main mem portion.
    uint64_t main_mem_bytes_pinned() const;

    /// @brief The number of bytes currently pinned in the
    /// scratch mem portion.
    uint64_t scratch_mem_bytes_pinned() const;

    int64_t total_bytes_allocated() const override { return 0; }

    int64_t main_mem_total_bytes_allocated() const { return 0; }

    int64_t scratch_mem_total_bytes_allocated() const { return 0; }

    int64_t num_allocations() const override { return 0; }

    int64_t main_mem_num_allocations() const { return 0; }

    int64_t scratch_mem_num_allocations() const { return 0; }

    /// @brief Get peak memory allocation in this memory pool
    int64_t max_memory() const override;

    /// @brief Get peak memory allocation in the main mem portion
    /// of this memory pool
    int64_t main_mem_max_memory() const;

    /// @brief The name of the backend used by the parent pool.
    std::string backend_name() const override;

    /// @brief If spilling is supported by the parent pool.
    bool is_spilling_enabled() const;

    /**
     * @brief Getter for the operator_budget_bytes
     * attribute.
     *
     * @return uint64_t
     */
    uint64_t get_operator_budget_bytes() const;

    /**
     * @brief Getter for memory_error_threshold
     * attribute.
     *
     * @return uint64_t
     */
    uint64_t get_memory_error_threshold() const;

    /**
     * @brief Getter for the error_threshold attribute.
     *
     * @return double
     */
    double get_error_threshold() const;

    /**
     * @brief Pin an allocation/block to memory. This assumes that
     * the allocation was made in the main mem portion.
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
     * @brief Pin an allocation/block to memory. This assumes that
     * the allocation was made in the scratch mem portion.
     *
     * @param[in, out] ptr Location of swip pointer containing
     *   allocation to pin.
     * @param size Size of the allocation (original requested size)
     * @param alignment Alignment used for the allocation
     * @return ::arrow::Status
     */
    ::arrow::Status PinScratch(
        uint8_t** ptr, int64_t size = -1,
        int64_t alignment = arrow::kDefaultBufferAlignment);

    /**
     * @brief Unpin an allocation/block. This makes the block eligible
     * for eviction when the parent pool needs to free up
     * space in memory. This assumes that the allocation was made
     * in the main mem portion.
     *
     * @param[in, out] ptr Swip pointer to allocation to unpin
     * @param size Size of the allocation (original requested size)
     * @param alignment Alignment used for the allocation
     */
    void Unpin(uint8_t* ptr, int64_t size = -1,
               int64_t alignment = arrow::kDefaultBufferAlignment) override;

    /**
     * @brief Unpin an allocation/block. This makes the block eligible
     * for eviction when the parent pool needs to free up
     * space in memory. This assumes that the allocation was made
     * in the scratch mem portion.
     *
     * @param[in, out] ptr Swip pointer to allocation to unpin
     * @param size Size of the allocation (original requested size)
     * @param alignment Alignment used for the allocation
     */
    void UnpinScratch(uint8_t* ptr, int64_t size = -1,
                      int64_t alignment = arrow::kDefaultBufferAlignment);

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
     * If the number of pinned bytes is larger than
     * memory_error_threshold_, an OperatorPoolThresholdExceededError will be
     * raised and threshold enforcement will not be enabled.
     * NOTE: We won't try to get an increased budget from the
     * OperatorComptroller in this case.
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
    /// @brief Allocation stats for "main" allocations
    /// (i.e. made through Allocate API).
    ::arrow::internal::MemoryPoolStats main_mem_stats_;
    /// @brief Allocation stats (combined)
    ::arrow::internal::MemoryPoolStats stats_;

    // NOTE: We don't maintain a similar struct for the scratch
    // portion since they can be inferred from the other two. The
    // only stat we don't get is the max memory usage in the scratch
    // portion. However, that's fine for now since there's no use case
    // where we need that information.

   private:
    /// @brief Parent pool through which all the allocations, etc. will go
    /// through.
    const std::shared_ptr<BufferPool> parent_pool_;

    /// @brief Memory budget for this operator (in bytes).
    uint64_t operator_budget_bytes_;
    /// @brief The current error threshold. This is relative to
    /// the operator budget. This must be <= 1.0.
    double error_threshold_;
    /// @brief The memory threshold (in bytes) to enforce when threshold
    /// enforcement is enabled. This is essentially
    /// (error_threshold * operator_budget_bytes_)
    uint64_t memory_error_threshold_;
    /// @brief Flag for threshold enforcement.
    bool threshold_enforcement_enabled_ = true;
    /// @brief Number of bytes currently pinned (main + scratch)
    /// TODO: Integrate into stats_ class at some point?
    std::atomic<uint64_t> bytes_pinned_;
    /// @brief Number of bytes currently pinned in the main
    /// mem portion.
    std::atomic<uint64_t> main_mem_bytes_pinned_;

    /**
     * @brief Helper function to try to get additional budget from
     * the OperatorComptroller. If additional budget is allocated,
     * it will update the pool state to reflect the new budget.
     *
     */
    inline void try_increase_budget();

    /**
     * @brief Helper function to update the pool budget
     * (and the memory_error_threshold).
     * Note that if threshold enforcement is enabled and the
     * number of bytes pinned is greater than what the new
     * threshold would be, we will raise a
     * OperatorPoolThresholdExceededError. The budget
     * won't be updated in this case.
     *
     * @param diff Delta between the new and old budget.
     */
    inline void update_budget(int64_t diff);

    /**
     * @brief Atomically update the number of pinned bytes in the
     * BufferPool by the diff.
     *
     * @param diff The number of bytes to add to the current pinned
     * byte count.
     */
    inline void update_pinned_bytes(int64_t diff);

    /**
     * @brief Atomically update the number of pinned bytes in the
     * main mem portion of the BufferPool by the diff.
     *
     * @param diff The number of bytes to add to the current pinned
     * byte count.
     */
    inline void update_main_mem_pinned_bytes(int64_t diff);

    /**
     * @brief Utility function for throwing the
     * OperatorPoolThresholdExceededError if threshold
     * enforcement is enabled and if
     * currently pinned bytes + size would be over the
     * memory threshold ('memory_error_threshold_').
     * Before raising the error, we will attempt to increase
     * our available budget. If we're still short, we will
     * raise the error.
     *
     * @param size Number of additional bytes to pin/allocate.
     */
    template <bool is_scratch>
    inline void enforce_threshold(int64_t size);

    /// Templated "inner" functions for Allocate, Free, Reallocate, Pin &
    /// Unpin. These contain most of the actual implementation. There are two
    /// main differences in the way that main mem and scratch mem functions
    /// work:
    ///  1. They call 'enforce_threshold' with the correct template parameter
    ///     for 'is_scratch'
    ///  2. They need to update the right set of statistics.
    /// The user facing functions just call these with the correct template
    /// parameter value.
    /// When 'is_scratch' is true, we make allocations / free from from the
    /// scratch mem portion of the pool. When it's false, we make allocations /
    /// free from the main mem portion of the pool.

    template <bool is_scratch>
    inline ::arrow::Status allocate_inner_(int64_t size, int64_t alignment,
                                           uint8_t** out);

    template <bool is_scratch>
    inline void free_inner_(uint8_t* buffer, int64_t size, int64_t alignment);

    template <bool is_scratch>
    inline ::arrow::Status reallocate_inner_(int64_t old_size, int64_t new_size,
                                             int64_t alignment, uint8_t** ptr);

    template <bool is_scratch>
    inline ::arrow::Status pin_inner_(uint8_t** ptr, int64_t size,
                                      int64_t alignment);

    template <bool is_scratch>
    inline void unpin_inner_(uint8_t* ptr, int64_t size, int64_t alignment);
};

/**
 * @brief This is an indirection layer on top of OperatorBufferPool
 * that will always allocate from the scratch mem portion of its
 * parent OperatorBufferPool while following the IBufferPool
 * interface.
 * This allows us to use this as a regular pool while allocating
 * from the scratch mem portion of the OperatorBufferPool.
 * e.g. We can pass an instance of this to the STLAllocator
 * for the hash tables in Groupby/Join. The memory of the hash
 * table would get allocated through the scratch mem portion of
 * the OperatorBufferPool without changing any other implementation.
 * The OperatorScratchPool is essentially "stateless", i.e. it doesn't
 * track its memory usage, etc. It simply calls the relevant
 * functions in the parent pool.
 *
 */
class OperatorScratchPool final : public IBufferPool {
   public:
    explicit OperatorScratchPool(OperatorBufferPool* const parent_pool)
        : parent_pool_(parent_pool) {}

    ::arrow::Status Allocate(int64_t size, int64_t alignment,
                             uint8_t** out) override;

    void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;

    ::arrow::Status Reallocate(int64_t old_size, int64_t new_size,
                               int64_t alignment, uint8_t** ptr) override;

    int64_t bytes_allocated() const override;

    uint64_t bytes_pinned() const override;

    std::optional<std::tuple<uint64_t, uint64_t>> alloc_loc(
        uint8_t* ptr, int64_t size = -1,
        int64_t alignment = arrow::kDefaultBufferAlignment) const override;

    int64_t total_bytes_allocated() const override;

    int64_t num_allocations() const override;

    /// @brief There's currently no use case for this, so we haven't
    /// implemented it.
    /// Unlike the other stats, we cannot calculate this based
    /// on the total stats and main mem stats. We'd need to
    /// maintain this separately, which would add unnecessary
    /// overhead.
    /// @return Always returns 0 at this point.
    int64_t max_memory() const override { return 0; }

    std::string backend_name() const override;

    ::arrow::Status Pin(
        uint8_t** ptr, int64_t size = -1,
        int64_t alignment = arrow::kDefaultBufferAlignment) override;

    void Unpin(uint8_t* ptr, int64_t size = -1,
               int64_t alignment = arrow::kDefaultBufferAlignment) override;

    /// @brief Get a pointer to its parent OperatorBufferPool.
    OperatorBufferPool* get_parent_pool() const;

   private:
    /// @brief Parent operator pool through which all the allocations, etc. will
    /// go through.
    OperatorBufferPool* const parent_pool_;
};

}  // namespace bodo
