#pragma once

#include <arrow/memory_pool.h>

namespace bodo {

/// Helper Tools for using BufferPool in STL Containers
/// Based on ::arrow::stl::allocator class. We did not extend from that
/// class due to buggy behavior when the destructor was being called
/// too early.

// Minimal base class for STL allocators that use our buffer pool. The child
// clasess plug in the buffer pool pointer manually, which is needed in modules
// other than bodo.ext.
template <class T>
class STLBufferPoolAllocator {
   public:
    using value_type = T;
    using is_always_equal = std::true_type;

    template <class U>
    struct rebind {
        using other = STLBufferPoolAllocator<U>;
    };

    STLBufferPoolAllocator(arrow::MemoryPool* pool, size_t size) noexcept
        : pool_(pool), size_(size) {}

    STLBufferPoolAllocator(arrow::MemoryPool* pool) noexcept
        : STLBufferPoolAllocator(pool, 0) {}

    ~STLBufferPoolAllocator() {}

    template <class U>
    STLBufferPoolAllocator(const STLBufferPoolAllocator<U>& other) noexcept
        : STLBufferPoolAllocator(other.pool(), other.size()) {}

    template <class U>
    STLBufferPoolAllocator(STLBufferPoolAllocator<U>&& other)
        : STLBufferPoolAllocator(other.pool(), other.size()) {}

    template <class U>
    STLBufferPoolAllocator& operator=(const STLBufferPoolAllocator<U>& other) {
        this->pool_ = other.pool();
        this->size_ = other.size();
    }

    template <class U>
    STLBufferPoolAllocator& operator=(STLBufferPoolAllocator<U>&& other) {
        this->pool_ = other.pool();
        this->size_ = other.size();
    }

    T* address(T& r) const noexcept { return std::addressof(r); }

    const T* address(const T& r) const noexcept { return std::addressof(r); }

    template <class U>
    inline bool operator==(const STLBufferPoolAllocator<U>& rhs) {
        return this->pool_ == rhs.pool() && this->size_ == rhs.size();
    }

    template <class U>
    inline bool operator!=(const STLBufferPoolAllocator<U>& rhs) {
        return this->pool_ != rhs.pool() || this->size_ != rhs.size();
    }

    T* allocate(size_t n) {
        uint8_t* data;
        size_t bytes = n * sizeof(T);
        ::arrow::Status s = this->pool_->Allocate(bytes, &data);
        if (!s.ok()) {
            throw std::bad_alloc();
        }
        this->size_ += bytes;
        return reinterpret_cast<T*>(data);
    }

    void deallocate(T* p, size_t n) {
        size_t bytes = n * sizeof(T);
        this->pool_->Free(reinterpret_cast<uint8_t*>(p), bytes);
        this->size_ -= bytes;
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

    arrow::MemoryPool* pool() const noexcept { return this->pool_; }
    size_t size() const noexcept { return this->size_; }

   private:
    arrow::MemoryPool* const pool_;
    // Size of all allocations made through this allocator in bytes.
    size_t size_;
};

}  // namespace bodo
