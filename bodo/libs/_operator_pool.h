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
     * @brief Allocates a buffer of 'size' bytes. The allocated buffer will be
     * 'alignment' byte aligned (64 by default).
     * This will raise the OperatorPoolThresholdExceededError if
     * size + currently pinned bytes > memory_error_threshold (assuming
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
     * @brief Free the specified 'buffer'.
     *
     * @param buffer Pointer to the start of the allocated memory region.
     * @param size Allocated size located at buffer.
     * @param alignment The alignment of the allocation. Defaults to 64 bytes.
     */
    void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;

    /**
     * @brief Resize an already allocated buffer.
     *
     * NOTE: If pinning both the new buffer (new_size) and
     * old buffer (old_size) together would take the amount of
     * pinned bytes over the threshold and threshold enforcement
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

    /// @brief The number of bytes currently allocated through
    /// this allocator.
    int64_t bytes_allocated() const override;

    /// @brief The number of bytes currently pinned.
    uint64_t bytes_pinned() const override;

    int64_t total_bytes_allocated() const override { return 0; }

    int64_t num_allocations() const override { return 0; }

    /// @brief Get peak memory allocation in this memory pool
    int64_t max_memory() const override;

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
    /// @brief Allocation stats
    ::arrow::internal::MemoryPoolStats stats_;

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
    /// @brief Number of bytes currently pinned
    /// TODO: Integrate into stats_ class at some point?
    std::atomic<uint64_t> bytes_pinned_;

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
    inline void enforce_threshold(int64_t size);
};

}  // namespace bodo
