# distutils: language = c++

# Declare raw C++ library APIs from _memory.h for usage in Cython.
# Here the C++ classes and methods are declared as they are
# so that in memory.pyx file, they can be used to implement Python classes,
# functions and helpers.


from pyarrow.includes.libarrow cimport CMemoryPool

from bodo.libs.common cimport *


cdef extern from "../libs/_memory.h" namespace "bodo" nogil:

    cdef cppclass CSizeClass" bodo::SizeClass":
        CSizeClass(size_t capacity, size_t block_size) except +
        c_bool isInRange(uint8_t* ptr) const
        uint8_t* getFrameAddress(uint64_t idx) const
        uint64_t getFrameIndex(uint8_t* ptr) const
        uint8_t** getSwip(uint64_t idx) const
        int64_t AllocateFrame(uint8_t** swip)
        void FreeFrame(uint64_t idx)
        void FreeFrame(uint8_t* ptr)
        uint64_t getBlockSize() const
        uint64_t getNumBlocks() const
        c_bool isFrameMapped(uint64_t idx) const
        c_bool isFramePinned(uint64_t idx) const
    

    cdef enum CStorageType" bodo::StorageType":
        Local = 0
        S3 = 1

    cdef cppclass CStorageOptions" bodo::StorageOptions":
        int64_t usable_size
        c_string location
        CStorageType type
        c_bool tracing_mode

        CStorageOptions()
        
        @staticmethod
        shared_ptr[CStorageOptions] Defaults(uint8_t tier)


    cdef cppclass CBufferPoolOptions" bodo::BufferPoolOptions":
        uint64_t memory_size
        uint64_t min_size_class
        uint64_t max_num_size_classes
        c_bool enforce_max_limit_during_allocation
        vector[shared_ptr[CStorageOptions]] storage_options
        c_bool spill_on_unpin
        c_bool move_on_unpin
        c_bool debug_mode
        uint8_t trace_level
        int64_t malloc_free_trim_threshold

        CBufferPoolOptions()
        
        @staticmethod
        CBufferPoolOptions Defaults() except +


    cdef cppclass CIBufferPool" bodo::IBufferPool"(CMemoryPool):
        # put overloads under a different name to avoid cython bug with multiple
        # layers of inheritance
        uint64_t i_get_bytes_pinned" bytes_pinned"()
        CStatus i_Pin" Pin"(uint8_t** ptr, int64_t size, int64_t alignment)
        void i_Unpin" Unpin"(uint8_t* ptr, int64_t size, int64_t alignment)


    cdef cppclass CBufferPool" bodo::BufferPool"(CIBufferPool):
        void Cleanup()

        CStatus Allocate(int64_t size, int64_t alignment, uint8_t** out) except +
        CStatus Reallocate(int64_t old_size, int64_t new_size, int64_t alignment, uint8_t** ptr) except +
        void Free(uint8_t* buffer, int64_t size, int64_t alignment) except +
        CStatus Pin(uint8_t** ptr, int64_t size, int64_t alignment) except +
        void Unpin(uint8_t* ptr, int64_t size, int64_t alignment) except +
        c_bool IsPinned(uint8_t* buffer, int64_t size, int64_t alignment) const

        # put overloads under a different name to avoid cython bug with multiple
        # layers of inheritance
        uint64_t get_bytes_pinned" bytes_pinned"()
        uint64_t get_bytes_in_memory" bytes_in_memory"()
        int64_t get_bytes_allocated" bytes_allocated"()
        int64_t get_max_memory" max_memory"()
        c_string get_backend_name" backend_name"()
        size_t num_size_classes()
        c_bool is_spilling_enabled()
        CSizeClass* GetSizeClass_Unsafe(uint64_t idx)
        uint64_t GetSmallestSizeClassSize()
        int64_t get_bytes_freed_through_malloc_since_last_trim() const

        @staticmethod
        shared_ptr[CBufferPool] Default()

    cdef void init_buffer_pool_ptr(int64_t ptr)


cdef extern from "../libs/_operator_pool.h" namespace "bodo" nogil:

    cdef cppclass COperatorBufferPool" bodo::OperatorBufferPool"(CIBufferPool):

        COperatorBufferPool(int64_t operator_id, uint64_t operator_budget_bytes, shared_ptr[CBufferPool] parent_pool, double error_threshold)
        void SetErrorThreshold(double error_threshold) except +
        void SetBudget(uint64_t new_operator_budget) except +
        CStatus Allocate(int64_t size, int64_t alignment, uint8_t** out) except +
        CStatus AllocateScratch(int64_t size, int64_t alignment, uint8_t** out) except +
        CStatus Reallocate(int64_t old_size, int64_t new_size, int64_t alignment, uint8_t** ptr) except +
        CStatus ReallocateScratch(int64_t old_size, int64_t new_size, int64_t alignment, uint8_t** ptr) except +
        void Free(uint8_t* buffer, int64_t size, int64_t alignment) except +
        void FreeScratch(uint8_t* buffer, int64_t size, int64_t alignment) except +
        CStatus Pin(uint8_t** ptr, int64_t size, int64_t alignment) except +
        CStatus PinScratch(uint8_t** ptr, int64_t size, int64_t alignment) except +
        void Unpin(uint8_t* ptr, int64_t size, int64_t alignment) except +
        void UnpinScratch(uint8_t* ptr, int64_t size, int64_t alignment) except +
        c_bool ThresholdEnforcementEnabled() const
        void DisableThresholdEnforcement()
        void EnableThresholdEnforcement() except +
        shared_ptr[CBufferPool] get_parent_pool() const
        uint64_t get_operator_budget_bytes() const
        uint64_t get_memory_error_threshold() const
        double get_error_threshold() const

        # put overloads under a different name to avoid cython bug with multiple
        # layers of inheritance
        uint64_t get_bytes_pinned" bytes_pinned"() const
        int64_t get_bytes_allocated" bytes_allocated"() const
        int64_t get_max_memory" max_memory"() const
        c_string get_backend_name" backend_name"() const

        # Main/Scratch stats
        int64_t main_mem_bytes_allocated() const
        int64_t scratch_mem_bytes_allocated() const
        uint64_t main_mem_bytes_pinned() const
        uint64_t scratch_mem_bytes_pinned() const
        int64_t main_mem_max_memory() const

    cdef cppclass COperatorScratchPool" bodo::OperatorScratchPool"(CIBufferPool):

        COperatorScratchPool(COperatorBufferPool* parent_pool)
        CStatus Allocate(int64_t size, int64_t alignment, uint8_t** out) except +
        CStatus Reallocate(int64_t old_size, int64_t new_size, int64_t alignment, uint8_t** ptr) except +
        void Free(uint8_t* buffer, int64_t size, int64_t alignment) except +
        CStatus Pin(uint8_t** ptr, int64_t size, int64_t alignment) except +
        void Unpin(uint8_t* ptr, int64_t size, int64_t alignment) except +

        # put overloads under a different name to avoid cython bug with multiple
        # layers of inheritance
        uint64_t get_bytes_pinned" bytes_pinned"() const
        int64_t get_bytes_allocated" bytes_allocated"() const
        int64_t get_max_memory" max_memory"() const
        c_string get_backend_name" backend_name"() const
