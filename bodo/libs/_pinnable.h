#pragma once

#include <cstdint>

#include "_memory.h"
#include "_stl.h"

// This file provides several utilities for managing pinnable memory
// Background:
// https://bodo.atlassian.net/wiki/spaces/B/pages/1407221763/Spilling+support+for+bodo+vector+unordered+map
//
// Pinnable memory is memory allocated for data structures that may exist
// in memory *or* on disk.
//
// In this header:
//   - PinnableAllocator - This is a c++ allocator that uses a 'smart pointer'
//   type that
//      allows all allocations to be 'unpinned'. Internally, the allocator keeps
//      all allocated pointers in a doubly linked list. Upon a call to the
//      'unpin()' method, the allocator walks through all allocations and unpins
//      them in the buffer pool
//   - pinnable - This is a template class that wraps containers using the
//   PinnableAllocator.
//      It uses RAII to ensure that no access can be made to the container
//      unless the container has been pinned back into memory.
//   -  pinned - Utility class used by pinnable<T>::pin() that allows access to
//   the wrapped class
//
// Note that only classes that have an appropriate specialization for
// `bodo::pinning_traits` will work with `bodo::pinnable.`

namespace bodo {
template <typename T>
class PinnableAllocator;

/// @brief Every pinnable pointer that derives from a single allocation shares
/// a reference to a 'pinnable_ptr_base'. For example, if you make one
/// allocation and assign it to the variable 'x', and then copy 'x' into 'y',
/// both 'x' and 'y' will share a reference to the same pinnable_ptr_base.
///
/// This is a necessary abstraction because most containers keep multiple
/// pointers to various 'parts' of their interior data.
///
/// Our buffer pool mandates that each pointer to an allocation not be
/// duplicated. Thus, we use this structure to share the underlying pointer so
/// there's only one copy

struct pinnable_ptr_base {
    /// @brief Create a pinnable_ptr
    /// @param pool The pool to allocate from
    /// @param size  The size of the allocation (saves a lookup in the
    /// allocator)
    pinnable_ptr_base(IBufferPool *pool, std::size_t size)
        : prev_(nullptr),
          next_(nullptr),
          value_(nullptr),
          size_(size),
          refcnt_(1) {
        CHECK_ARROW_MEM(pool->Allocate(size, &value_),
                        "Could not allocate pinnable pointer");
    }

    /// @brief Pins the pointer in the buffer pool. Pins are reference counted,
    /// so it's fine to call this recursively.
    /// @param pool The buffer pool instance to pin in. Should be the same one
    /// that allocated this pointer.
    void pin(IBufferPool *pool) {
        if (refcnt_ == 0) {
            CHECK_ARROW_MEM(pool->Pin(&value_, size_),
                            "Could not pin pinnable_ptr_base");
        }
        refcnt_++;
    }

    /// @brief Unpin the pointer in the buffer pool.
    /// @param pool The pool that allocated this pointer.
    void unpin(IBufferPool *pool) {
        if (refcnt_ == 1) {
            pool->Unpin(value_, size_);
        }
        refcnt_--;
    }

    /// @brief  Every bodo::pinnable_ptr_base belongs to a bodo::pinnable via
    /// the PinnableAllocator. The bodo::pinnable keeps all pointers allocated
    /// by its inner structure in a doubly linked list so that it can pin or
    /// unpin all allocated buffers on demand.
    pinnable_ptr_base *prev_, *next_;

    std::uint8_t *value_;
    std::size_t size_;

    int refcnt_;
};

/// @brief This is the pointer type for any bodo::PinnableAllocator. The STL
/// containers will store these internally. Users should not use this directly
/// or risk invalid memory accesses.
/// @tparam T The thing pointed to
template <typename T>
struct pinnable_ptr {
    // Standard STL container definitions for smart pointers / iterators
    using element_type = std::remove_extent_t<T>;
    using pointer = T *;
    using reference = T &;
    using const_reference = const T &;
    using difference_type = typename std::iterator_traits<T *>::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    /// @brief  Construct a null pinnable_ptr
    pinnable_ptr() : base_(nullptr), offset_(0) {}

    /// @brief Copy a pinnable_ptr
    pinnable_ptr(const pinnable_ptr<T> &other)
        : base_(other.base_), offset_(other.offset_) {}

    /// @brief Produce a pinnable_ptr using nullptr
    pinnable_ptr(std::nullptr_t np) : pinnable_ptr() {}

    /// @brief Cast a pinnable pointer of one type into this type with proper
    /// casting behavior.
    /// @tparam U Any type
    /// @param other The pinnable pointer to copy.
    template <typename U>
    pinnable_ptr(const pinnable_ptr<U> &other)
        : base_(other.base_), offset_(0) {
        T *p = static_cast<T *>(other.get());
        offset_ = reinterpret_cast<std::intptr_t>(p) -
                  reinterpret_cast<std::intptr_t>(base());
    }

    // This function must only be used when the corresponding memory is pinned
    pointer operator->() const {
        return reinterpret_cast<T *>(base() + offset_);
    }

    pointer get() const { return operator->(); }

    // Get references to the inner value
    reference operator*() const { return *get(); }

    operator bool() const { return get(); }
    operator T *() const { return get(); }

    pinnable_ptr<T> &operator=(const pinnable_ptr<T> &other) {
        base_ = other.base_;
        offset_ = other.offset_;
        return *this;
    }

    template <typename U>
    bool operator==(const pinnable_ptr<U> &other) const {
        return get() == other.get();
    }

    bool operator==(std::nullptr_t np) const { return get() == np; }

    bool operator!=(const pinnable_ptr<T> &other) const {
        bool ret(!(*this == other));
        return ret;
    }

    bool operator<(const pinnable_ptr<T> &other) const {
        return get() < other.get();
    }

    bool operator>(const pinnable_ptr<T> &other) const {
        return get() > other.get();
    }

    bool operator<=(const pinnable_ptr<T> &other) const {
        return get() <= other.get();
    }

    bool operator>=(const pinnable_ptr<T> &other) const {
        return get() >= other.get();
    }

    template <typename Sz>
    T &operator[](Sz sz) const {
        return get()[sz];
    }

    // Arithmetic operators
    pinnable_ptr<T> &operator--() {
        (*this) -= 1;
        return *this;
    }
    pinnable_ptr<T> operator--(int i) {
        pinnable_ptr<T> old(*this);
        (*this) -= 1;
        return old;
    }
    pinnable_ptr<T> &operator++() {
        (*this) += 1;
        return *this;
    }
    pinnable_ptr<T> operator++(int i) {
        pinnable_ptr<T> old(*this);
        (*this) += 1;
        return old;
    }

    template <typename N>
    pinnable_ptr<T> &operator+=(N n) {
        T *p(get() + n);
        offset_ = reinterpret_cast<std::intptr_t>(p) - base();
        return *this;
    }

    template <typename N>
    pinnable_ptr<T> &operator-=(N n) {
        T *p(get() - n);
        offset_ = reinterpret_cast<std::intptr_t>(p) - base();

        return *this;
    }

    template <typename N>
    pinnable_ptr<T> operator+(N n) const {
        pinnable_ptr<T> o(*this);
        o += n;
        return o;
    }

    template <typename N>
    pinnable_ptr<T> operator-(N n) const {
        pinnable_ptr<T> o(*this);
        o -= n;
        return o;
    }

    template <typename U>
    difference_type operator-(const bodo::pinnable_ptr<U> &p) const {
        return get() - p.get();
    }

   private:
    /// @brief Create a pinnable_ptr from a new allocator. Only called by
    /// PinnableAllocator
    /// @param p
    pinnable_ptr(pinnable_ptr_base *p) : base_(p), offset_(0) {}

    /// @brief A pinnable_ptr_base identifying the allocation from which this
    /// pointer derives
    pinnable_ptr_base *base_;

    /// @brief bodo::pinnable_ptr supports pointer arithmetic as normal but it
    /// must still keep around the base pointer.
    std::ptrdiff_t offset_;

    /// @brief Get the raw allocated pointer, or nullptr if we're empty
    std::intptr_t base() const {
        return reinterpret_cast<std::intptr_t>(base_ ? base_->value_ : nullptr);
    }

    friend class PinnableAllocator<T>;
    friend struct pinned;

    template <typename U>
    friend struct pinnable_ptr;
};

/// Shared state to support copying of PinnableAllocator
struct PinnableAllocatorState {
    inline PinnableAllocatorState(IBufferPool *pool) : pool_(pool) {};

    /// @brief The pool we will allocate from
    IBufferPool *const pool_;

    /// @brief A doubly linked list of pointers allocated from this pool
    pinnable_ptr_base *ptrs_{nullptr};
};

// C++-compatible allocator that keeps track of allocations from the given
// buffer pool
//
// Deviations from standard:
//    - address() not implemented (no longer required in c++20 anyway)
template <typename T>
class PinnableAllocator {
   public:
    using value_type = T;
    using pointer = pinnable_ptr<T>;
    using const_pointer = pinnable_ptr<const T>;

    template <class U>
    struct rebind {
        using other = PinnableAllocator<U>;
    };

    /// @brief Make a pinnable allocator that will use the given buffer pool
    /// @param pool  The pool to allocate from
    PinnableAllocator(IBufferPool *pool) noexcept
        : state_(std::make_shared<PinnableAllocatorState>(pool)) {}

    /// Make a pinnable allocator using the default buffer pool
    PinnableAllocator() noexcept
        : PinnableAllocator(BufferPool::DefaultPtr()) {}

    /// @brief Support rebinding the allocator to a new type. Mandated by C++
    /// standard
    /// @tparam U The old allocator type
    /// @param other The old allocator
    template <class U>
    PinnableAllocator(const PinnableAllocator<U> &other)
        : state_(other.state_) {}

    template <class U>
    PinnableAllocator &operator=(PinnableAllocator<U> &&other) {
        std::swap(other.state_, state_);
    }

    template <class U>
    inline bool operator==(const PinnableAllocator<U> &rhs) const {
        return state_ == rhs.state_;
    }

    template <class U>
    inline bool operator!=(const PinnableAllocator<U> &other) const {
        return !(*this == other);
    }

    pinnable_ptr<T> allocate(size_t n) {
        std::size_t sz = n * sizeof(T);

        auto base(new pinnable_ptr_base(this->state_->pool_, sz));
        base->next_ = this->state_->ptrs_;
        if (this->state_->ptrs_) {
            this->state_->ptrs_->prev_ = base;
        }
        this->state_->ptrs_ = base;

        return pinnable_ptr<T>(base);
    }

    void deallocate(const pinnable_ptr<T> &p, size_t n) {
        std::size_t sz = n * sizeof(T);
        if (!p.base_) {
            return;
        }
        if (!p.base_->value_) {
            return;
        }

        this->state_->pool_->Free(p.base_->value_, sz);
        p.base_->value_ = nullptr;

        if (p.base_->prev_) {
            p.base_->prev_->next_ = p.base_->next_;
        } else {
            this->state_->ptrs_ = p.base_->next_;
        }

        if (p.base_->next_) {
            p.base_->next_->prev_ = p.base_->prev_;
        }

        p.base_->next_ = p.base_->prev_ = nullptr;
    }

    void destroy(const pinnable_ptr<T> &p) {
        // Destructor may access memory, thus this needs to be pinned
        auto g(bodo::pin(p.base_, this->state_->pool_));
        p->~T();
    }

    // Ensure all pointers are in memory and dereferenceable by an actual
    // memory address
    void pin() {
        for (auto cur(this->state_->ptrs_); cur; cur = cur->next_) {
            cur->pin(this->state_->pool_);
        }
    }

    // Unpin all memory locations and store the swizzled pointer
    void unpin() {
        for (auto cur(this->state_->ptrs_); cur; cur = cur->next_) {
            cur->unpin(this->state_->pool_);
        }
    }

#ifdef IS_TESTING
    void spillAndMove() {
        //  copy all pointers in the spilling state and replace.
        pinnable_ptr_base *cur = state_->ptrs_;
        while (cur) {
            std::uint8_t *new_buffer;
            CHECK_ARROW_MEM(
                this->state_->pool_->Allocate(cur->size_, &new_buffer),
                "Could not allocate pseudo-spill buffer");
            std::copy(cur->value_, cur->value_ + cur->size_, new_buffer);
            std::swap(cur->value_, new_buffer);

            this->state_->pool_->Free(new_buffer, cur->size_);
            cur = cur->next_;
        }
    }
#endif

   private:
    /// @brief Allocators must be copyable and re-bindable in order to comply
    /// with the C++ standard.
    ///
    /// However, each allocator derived from the same data structure is part of
    /// the same 'pinning' group. Thus we need a separate structure to share
    /// state.
    ///
    /// In particular, the linked list of allocated pointers needs to be shared
    /// so that calling 'pin()' on bodo::pinnable pins *all* the buffers in the
    /// container.
    std::shared_ptr<PinnableAllocatorState> state_;

    template <typename U>
    friend class PinnableAllocator;
};

/// Any container class that wants to support pinning/unpinning should
/// specialize this class.
template <typename T>
struct pinning_traits {
    /// @brief  If the pinnable type is different than T, then note that here.
    ///
    /// For example, bodo::vector<T, Allocator>'s 'pinnable_type' is
    /// bodo::vector<T, Allocator1> where Allocator1 is the pinning allocator
    /// corresponding to Allocator (see bodo::pinning_allocator_traits)
    using pinnable_type = std::enable_if_t<std::is_arithmetic_v<T>, T>;

    /// @brief The allocator type that  this container should use. For STL
    /// containers, this is just 'typename pinnable_type::allocator_type', but
    /// primitive types don't have that, so we have it here.
    using allocator_type =
        std::enable_if<std::is_arithmetic_v<T>, PinnableAllocator<T>>;

    /// @brief Bool value indicating whether the type is actually pinnable. By
    /// default all C++ primitive ints and floats (not pointers) are pinnable.
    static constexpr bool is_pinnable = std::is_arithmetic_v<T>;
};

template <typename K, typename V>
struct pinning_traits<std::pair<K, V>> {
    static constexpr bool is_pinnable = bodo::pinning_traits<K>::is_pinnable &&
                                        bodo::pinning_traits<V>::is_pinnable;
    using pinnable_type = std::enable_if_t<
        is_pinnable,
        std::pair<typename bodo::pinning_traits<K>::pinnable_type,
                  typename bodo::pinning_traits<V>::pinnable_type>>;
    using allocator_type =
        std::enable_if<is_pinnable, PinnableAllocator<pinnable_type>>;
};

/// @brief Some allocators can be automatically turned into pinning allocators.
/// For example, any bodo buffer pool allocator can be made a pinning allocator
////
/// To make a new allocator adaptable with the pinning support,you should
/// specialize this template, and define is_pinnable=true and set
/// pinnable_type to the type of the pinning allocator.
///
/// @tparam T The allocator type
template <typename T>
struct pinning_allocator_traits {
    static constexpr bool is_pinnable = false;
};

template <typename T>
struct pinning_allocator_traits<STLBufferPoolAllocator<T>> {
    static constexpr bool is_pinnable = true;

    using pinnable_type =
        PinnableAllocator<typename pinning_traits<T>::pinnable_type>;
};

template <typename T>
struct pinning_allocator_traits<PinnableAllocator<T>> {
    static constexpr bool is_pinnable = true;
    using pinnable_type = PinnableAllocator<T>;
};

template <typename T, typename Allocator>
struct pinning_traits<bodo::vector<T, Allocator>> {
    static constexpr bool is_pinnable =
        pinning_traits<T>::is_pinnable &&
        pinning_allocator_traits<Allocator>::is_pinnable;
    static constexpr bool size_without_pin = true;
    using allocator_type =
        typename pinning_allocator_traits<Allocator>::pinnable_type;
    using pinnable_type =
        bodo::vector<typename pinning_traits<T>::pinnable_type, allocator_type>;
};

template <typename K, typename V, typename HASH_FCT, typename EQ_FCT,
          typename Allocator>
struct pinning_traits<
    bodo::unord_map_container<K, V, HASH_FCT, EQ_FCT, Allocator>> {
    static constexpr bool is_pinnable =
        pinning_traits<K>::is_pinnable && pinning_traits<V>::is_pinnable &&
        pinning_allocator_traits<Allocator>::is_pinnable;
    static constexpr bool size_without_pin = false;
    using allocator_type =
        typename pinning_allocator_traits<Allocator>::pinnable_type;
    using pinnable_type =
        bodo::unord_map_container<typename pinning_traits<K>::pinnable_type,
                                  typename pinning_traits<V>::pinnable_type,
                                  HASH_FCT, EQ_FCT, allocator_type>;
};

template <typename T,
          typename = std::enable_if_t<pinning_allocator_traits<
              typename pinning_traits<T>::allocator_type>::is_pinnable>,
          typename = std::enable_if_t<pinning_traits<T>::is_pinnable>>

struct pinnable {
   public:
    using allocator_type = typename pinning_traits<T>::allocator_type;
    using element_type = typename pinning_traits<T>::pinnable_type;
    /// @brief Convenience function so that you don't have to pin() std::vector
    /// to get the size()
    /// @return The size of the underlying std::vector
    std::enable_if<pinning_traits<T>::size_without_pin, typename T::size_type>
    size() const {
        return underlying_.size();
    }

    int32_t pin_count() const { return pincnt_; }

    allocator_type allocator() { return underlying_.get_allocator(); }

    pinnable() : pincnt_(0) { allocator().unpin(); }

    pinnable(element_type &&v) : pincnt_(0), underlying_(std::move(v)) {
        allocator().unpin();
    }

    // Forwards arguments to the inner container's constructor
    template <typename... Args>
    pinnable(Args... args) : underlying_(std::forward<Args>(args)...) {}

#ifdef IS_TESTING
    /// @brief When the data is unpinned and testing is enabled, this function
    /// reallocates new buffers in the allocator and copies the data over. This
    /// simulates a spill and reload in a new position.
    void spillAndMove() { allocator().spillAndMove(); }
#endif

   private:
    int32_t pincnt_ = 0;

    /// Pin the data into memory and get a pinned guard that automatically
    /// handles unpinning.
    element_type *pin() {
        if (pincnt_ == 0) {
            allocator().pin();
        }
        pincnt_++;
        return &underlying_;
    }

    void unpin() {
        pincnt_--;

        if (pincnt_ == 0) {
            allocator().unpin();
        }
    }

    /// @brief  The underlying container. You must use 'pinned' to get access to
    /// this.
    element_type underlying_;

    template <typename X, typename... Args>
    friend class ::bodo::pin_guard;
};
}  // namespace bodo

namespace std {}
