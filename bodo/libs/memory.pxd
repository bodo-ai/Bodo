# Copyright (C) 2023 Bodo Inc. All rights reserved.
# distutils: language = c++

# Declare raw C++ library APIs from _memory.h for usage in Cython.
# Here the C++ classes and methods are declared as they are
# so that in memory.pyx file, they can be used to implement Python classes,
# functions and helpers.

from pyarrow.includes.libarrow cimport CMemoryPool

from bodo.libs.common cimport *


cdef extern from "_memory.h" namespace "bodo" nogil:

    cdef cppclass CSizeClass" bodo::SizeClass":
        CSizeClass(size_t capacity, size_t block_size) except +
        c_bool isInRange(uint8_t* ptr) const
        uint8_t* getFrameAddress(uint64_t idx) const
        uint64_t getFrameIndex(uint8_t* ptr) const
        int64_t AllocateFrame(uint8_t** swip)
        void FreeFrame(uint64_t idx)
        void FreeFrame(uint8_t* ptr)
        uint64_t getBlockSize() const
        uint64_t getNumBlocks() const
        c_bool isFrameMapped(uint64_t idx) const
        c_bool isFramePinned(uint64_t idx) const
    
    
    cdef cppclass CBufferPoolOptions" bodo::BufferPoolOptions":
        uint64_t memory_size
        uint64_t min_size_class
        uint64_t max_num_size_classes
        c_bool ignore_max_limit_during_allocation

        CBufferPoolOptions()
        
        @staticmethod
        CBufferPoolOptions Defaults()


    cdef cppclass CBufferPool" bodo::BufferPool"(CMemoryPool):
        CStatus Allocate(int64_t size, uint8_t** out)
        CStatus Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr)
        void Free(uint8_t* buffer, int64_t size)
        # put overloads under a different name to avoid cython bug with multiple
        # layers of inheritance
        int64_t get_bytes_allocated" bytes_allocated"()
        int64_t get_max_memory" max_memory"()
        c_string get_backend_name" backend_name"()
        size_t num_size_classes()
        uint16_t alignment()
        CSizeClass* GetSizeClass_Unsafe(uint64_t idx)

        @staticmethod
        shared_ptr[CBufferPool] Default()
