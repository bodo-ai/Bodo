#include "_operator_pool.h"
#include "_memory_budget.h"

namespace bodo {

void OperatorBufferPool::SetErrorThreshold(double error_threshold) {
    assert(error_threshold <= 1.0);

    uint64_t new_memory_error_threshold_ =
        static_cast<uint64_t>(error_threshold * this->operator_budget_bytes_);

    // If new threshold is smaller than before and threshold enforcement is
    // enabled, then check that number of bytes that are currently pinned is
    // lesser than the new threshold.
    if (error_threshold < this->error_threshold_) {
        if (this->threshold_enforcement_enabled_ &&
            (this->bytes_pinned() > new_memory_error_threshold_)) {
            throw OperatorPoolThresholdExceededError();
        }
    }
    this->error_threshold_ = error_threshold;
    this->memory_error_threshold_ = new_memory_error_threshold_;
}

void OperatorBufferPool::SetBudget(uint64_t new_operator_budget) {
    int64_t delta = static_cast<int64_t>(new_operator_budget) -
                    static_cast<int64_t>(this->operator_budget_bytes_);
    if (delta > 0) {
        throw std::runtime_error(
            "OperatorBufferPool::SetBudget: Increasing the budget is not "
            "supported through this API.");
    }
    // Update budget at the pool level
    this->update_budget(delta);
    // If the update was successful, notify the OperatorComptroller about
    // the new budget. This will allow other operators to receive the
    // additional budget should they need it.
    OperatorComptroller::Default()->ReduceOperatorBudget(this->operator_id,
                                                         new_operator_budget);
}

inline void OperatorBufferPool::update_budget(int64_t diff) {
    if (diff < ((-1L) * static_cast<int64_t>(this->operator_budget_bytes_))) {
        throw std::runtime_error("OperatorBufferPool::update_budget: diff (" +
                                 std::to_string(diff) +
                                 ") would make the budget negative!");
    }

    uint64_t new_operator_budget_bytes_ = this->operator_budget_bytes_ + diff;
    uint64_t new_memory_error_threshold_ = static_cast<uint64_t>(
        this->error_threshold_ * new_operator_budget_bytes_);

    // If the new budget is smaller than before and threshold enforcement is
    // enabled, then check that number of bytes that are currently pinned is
    // lesser than the new threshold.
    if ((diff < 0) && this->threshold_enforcement_enabled_ &&
        (this->bytes_pinned() > new_memory_error_threshold_)) {
        throw OperatorPoolThresholdExceededError();
    }

    // Update state:
    this->operator_budget_bytes_ = new_operator_budget_bytes_;
    this->memory_error_threshold_ = new_memory_error_threshold_;
}

inline void OperatorBufferPool::try_increase_budget() {
    // Request all available budget from the comptroller.
    size_t addln_budget =
        OperatorComptroller::Default()->RequestAdditionalBudget(
            this->operator_id, -1);
    // If additional budget was allocated, update the pool state:
    if (addln_budget > 0) {
        this->update_budget(addln_budget);
    }
}

inline void OperatorBufferPool::enforce_threshold(int64_t size) {
    // If threshold enforcement is turned on, then check if this
    // allocation would cross the threshold. If it does, throw
    // a custom error.
    if (this->threshold_enforcement_enabled_ &&
        ((size + this->bytes_pinned()) > this->memory_error_threshold_)) {
        // Try to get additional budget from OperatorComptroller.
        this->try_increase_budget();
        // If we still don't have sufficient budget, raise
        // OperatorPoolThresholdExceededError.
        if ((size + this->bytes_pinned()) > this->memory_error_threshold_) {
            throw OperatorPoolThresholdExceededError();
        }
    }
}

::arrow::Status OperatorBufferPool::Allocate(int64_t size, int64_t alignment,
                                             uint8_t** out) {
    if (size < 0) {
        return ::arrow::Status::Invalid("Negative allocation size requested.");
    }

    // Copied from Arrow (they are probably just being conservative for
    // compatibility with 32-bit architectures).
    if (static_cast<uint64_t>(size) >= std::numeric_limits<size_t>::max()) {
        return ::arrow::Status::OutOfMemory("malloc size overflows size_t");
    }

    this->enforce_threshold(size);

    // Allocate through the parent pool
    ::arrow::Status alloc_status =
        this->parent_pool_->Allocate(size, alignment, out);
    // If allocation went through successfully, update stats
    if (alloc_status.ok()) {
        this->update_pinned_bytes(size);
        this->stats_.UpdateAllocatedBytes(size);
    }
    return alloc_status;
}

void OperatorBufferPool::Free(uint8_t* buffer, int64_t size,
                              int64_t alignment) {
    // Check if the buffer was pinned:
    bool is_pinned = this->parent_pool_->IsPinned(buffer, size, alignment);
    // Free through the parent pool:
    this->parent_pool_->Free(buffer, size, alignment);
    // Update stats:
    if (is_pinned) {
        this->update_pinned_bytes(-size);
    }
    this->stats_.UpdateAllocatedBytes(-size);
}

::arrow::Status OperatorBufferPool::Reallocate(int64_t old_size,
                                               int64_t new_size,
                                               int64_t alignment,
                                               uint8_t** ptr) {
    if (new_size < 0) {
        return ::arrow::Status::Invalid(
            "Negative reallocation size requested.");
    }
    if (static_cast<uint64_t>(new_size) >= std::numeric_limits<size_t>::max()) {
        return ::arrow::Status::OutOfMemory("realloc overflows size_t");
    }

    uint8_t* old_mem_ptr = *ptr;

    // Check if the buffer is already pinned.
    bool was_pinned =
        this->parent_pool_->IsPinned(old_mem_ptr, old_size, alignment);

    // Note that there's the possibility that both the old and the new
    // buffers are in memory and pinned at the same time.
    // To be conservative, we'll add it to pinned bytes and check
    // that we're still under the threshold.
    int64_t combined_pinned_bytes = new_size;
    if (!was_pinned) {
        combined_pinned_bytes += old_size;
    }

    this->enforce_threshold(combined_pinned_bytes);

    // Re-allocate will pin this in memory if it wasn't already, so we must
    // update our stats:
    if (!was_pinned) {
        this->update_pinned_bytes(old_size);
    }

    // Re-alloc through the parent pool:
    ::arrow::Status realloc_status =
        this->parent_pool_->Reallocate(old_size, new_size, alignment, ptr);
    if (realloc_status.ok()) {
        // Update stats if realloc was successful.
        // We don't need limit checks since we've already made sure that
        // (new_size + old_size) can be pinned, so this can only be lesser than
        // that.
        this->update_pinned_bytes(new_size - old_size);
        this->stats_.UpdateAllocatedBytes(new_size - old_size);
    } else {
        // If it failed, undo the stats updates if old buffer was never actually
        // pinned:
        bool is_pinned_now =
            this->parent_pool_->IsPinned(old_mem_ptr, old_size, alignment);
        if (!was_pinned && !is_pinned_now) {
            this->update_pinned_bytes(-old_size);
        }
    }
    return realloc_status;
}

int64_t OperatorBufferPool::bytes_allocated() const {
    return this->stats_.bytes_allocated();
}

uint64_t OperatorBufferPool::bytes_pinned() const {
    return this->bytes_pinned_.load();
}

int64_t OperatorBufferPool::max_memory() const {
    return this->stats_.max_memory();
}

std::string OperatorBufferPool::backend_name() const {
    return this->parent_pool_->backend_name();
}

bool OperatorBufferPool::is_spilling_enabled() const {
    return this->parent_pool_->is_spilling_enabled();
}

uint64_t OperatorBufferPool::get_operator_budget_bytes() const {
    return this->operator_budget_bytes_;
}

uint64_t OperatorBufferPool::get_memory_error_threshold() const {
    return this->memory_error_threshold_;
}

double OperatorBufferPool::get_error_threshold() const {
    return this->error_threshold_;
}

std::shared_ptr<BufferPool> OperatorBufferPool::get_parent_pool() const {
    return this->parent_pool_;
}

::arrow::Status OperatorBufferPool::Pin(uint8_t** ptr, int64_t size,
                                        int64_t alignment) {
    // Check if the buffer was pinned:
    bool is_pinned = this->parent_pool_->IsPinned(*ptr, size, alignment);
    if (!is_pinned) {
        // Verify that there's enough space to pin this allocation:
        this->enforce_threshold(size);

        // Call Pin on parent pool
        ::arrow::Status pin_status =
            this->parent_pool_->Pin(ptr, size, alignment);
        if (pin_status.ok()) {
            // If pin was successful, update the stats
            this->update_pinned_bytes(size);
        }
        return pin_status;
    }
    // If it was already pinned, this is a NOP
    return ::arrow::Status::OK();
}

void OperatorBufferPool::Unpin(uint8_t* ptr, int64_t size, int64_t alignment) {
    // Check if the buffer is pinned:
    bool is_pinned = this->parent_pool_->IsPinned(ptr, size, alignment);
    if (is_pinned) {
        // Call Unpin on parent pool
        this->parent_pool_->Unpin(ptr, size, alignment);
        // Unpin might be a NOP if the initial allocation was
        // through malloc:
        is_pinned = this->parent_pool_->IsPinned(ptr, size, alignment);
        // Update stats if it was actually unpinned
        if (!is_pinned) {
            this->update_pinned_bytes(-size);
        }
    }
    // If it was already unpinned, this is a NOP
}

bool OperatorBufferPool::ThresholdEnforcementEnabled() const {
    return this->threshold_enforcement_enabled_;
}

void OperatorBufferPool::DisableThresholdEnforcement() noexcept {
    this->threshold_enforcement_enabled_ = false;
}

void OperatorBufferPool::EnableThresholdEnforcement() {
    if (this->bytes_pinned() > this->memory_error_threshold_) {
        throw OperatorPoolThresholdExceededError();
    }
    this->threshold_enforcement_enabled_ = true;
}

inline void OperatorBufferPool::update_pinned_bytes(int64_t diff) {
    this->bytes_pinned_.fetch_add(diff);
}

}  // namespace bodo
