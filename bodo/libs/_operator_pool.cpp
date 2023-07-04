#include "_operator_pool.h"

namespace bodo {

inline ::arrow::Status OperatorBufferPool::check_limits(int64_t size) const {
    // If threshold enforcement is turned on, then check if this
    // allocation would cross the threshold. If it does, throw
    // a custom error.
    if (this->threshold_enforcement_enabled_ &&
        ((size + this->bytes_pinned()) > this->memory_error_threshold_)) {
        throw OperatorPoolThresholdExceededError();
    }
    // Even if threshold enforcement is not turned on, we need
    // to ensure that the number of pinned bytes at any time
    // is within the alloted limit.
    if ((size + this->bytes_pinned()) > this->max_pinned_size_bytes_) {
        return ::arrow::Status::OutOfMemory(
            "Allocation failed. This allocation would lead to more pinned "
            "memory than what is allowed for this operator.");
    }
    return ::arrow::Status::OK();
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

    ::arrow::Status limits_status = this->check_limits(size);
    if (!limits_status.ok()) {
        return limits_status;
    }

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
    ::arrow::Status limits_status = this->check_limits(combined_pinned_bytes);
    if (!limits_status.ok()) {
        return limits_status;
    }
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

uint64_t OperatorBufferPool::get_max_pinned_size_bytes() const {
    return this->max_pinned_size_bytes_;
}

uint64_t OperatorBufferPool::get_memory_error_threshold() const {
    return this->memory_error_threshold_;
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
        ::arrow::Status limits_status = this->check_limits(size);
        if (!limits_status.ok()) {
            return limits_status;
        }

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
