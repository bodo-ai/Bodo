# Copyright (C) 2023 Bodo Inc. All rights reserved.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True

# Here we will create the glue code which will put together
# the C++ capabilities and turn it into Python classes
# and methods.
# For the most part, this will be used for integrating
# our BufferPool with PyArrow and for unit testing purposes.

from cython.operator cimport dereference as deref
from pyarrow.lib cimport MemoryPool, _Weakrefable, check_status

import pyarrow as pa
from pyarrow.lib import frombytes


cdef class BufferPoolAllocation():
    """
    A class for storing information about allocations
    made through the BufferPool. In particular, it
    stores a pointer to the allocated memory and the
    allocation size (from the user perspective, not
    the actual allocated memory from the BufferPool's
    perspective).

    NOTE: This is for unit testing purposes only at this
    point.
    """

    cdef:
        # Pointer to the allocated memory region
        uint8_t* ptr
        # Size of the allocation
        int size
        # Alignment for the allocation
        int alignment
    
    def __init__(self):
        """
        Empty constructor.
        """
        pass
    
    cdef void set_ptr(self, uint8_t* ptr):
        """
        Set the pointer to the memory region.
        """
        self.ptr = ptr
    
    cdef void set_size(self, int size):
        """
        Set allocation size.
        """
        self.size = size

    cdef void set_alignment(self, int alignment):
        """
        Set allocation alignment.
        """
        self.alignment = alignment

    @staticmethod
    cdef BufferPoolAllocation ccopy(BufferPoolAllocation other):
        """
        Copy over the fields from another instance.
        """
        new_object = BufferPoolAllocation()
        new_object.set_ptr(other.ptr)
        new_object.set_size(other.size)
        new_object.set_alignment(other.alignment)
        return new_object
    
    @staticmethod
    def copy(other: BufferPoolAllocation) -> BufferPoolAllocation:
        """
        Wrapper around ccopy.
        Returns a deepcopy of a BufferPoolAllocation.
        """
        cdef:
            new_object = BufferPoolAllocation.ccopy(other)
        return new_object
    
    @property
    def alignment(self):
        return self.alignment

    cdef int64_t c_ptr_as_int(self):
        """
        Simply casts the 'ptr' as 'int64_t'
        """
        return <int64_t>self.ptr


    def get_ptr_as_int(self) -> int:
        """
        Get the pointer as an integer. This is
        useful in verifying that the alignment
        is correct.
        """
        return int(self.c_ptr_as_int())

    
    def __eq__(self, other: BufferPoolAllocation) -> bool:
        """
        Check if two BufferPoolAllocations instances are
        equivalent, i.e. they both point to the same
        memory region and have the same allocation size.
        """
        if isinstance(other, BufferPoolAllocation):
            return (
                (self.ptr == other.ptr) and
                (self.size == other.size) and
                (self.alignment == other.alignment)
            )
        else:
            return False

    def __ne__(self, other: BufferPoolAllocation) -> bool:
        """
        Inverse of __eq__.
        """
        return not self.__eq__(other)
    
    def has_same_ptr(self, other: BufferPoolAllocation) -> bool:
        """
        Check if this and 'other' and point to the same memory region.
        """
        return self.ptr == other.ptr


cdef class SizeClass(_Weakrefable):
    """
    Python class around the SizeClass class defined in C++.
    It defines wrappers for most of the public APIs and
    properties exposed by the C++ class.
    NOTE: This is only for unit-testing purposes at this point.
    """

    cdef:
        # Store a pointer to the C++ instance.
        # All APIs will use this to build wrappers
        # around the C++ class.
        CSizeClass* c_size_class
    
    cdef void cinit(self, CSizeClass* c_size_class):
        """
        Cython helper to set the pointer to the
        C++ class.
        """
        self.c_size_class = c_size_class
    
    def __init__(self, capacity, block_size):
        """
        Since this is for unit-testing purposes only,
        we don't need a way to create a new instance,
        we will always get an existing one from a
        BufferPool instance.
        """
        raise NotImplementedError
    
    cdef bint c_is_in_range(self, uint8_t* ptr):
        return self.c_size_class.isInRange(ptr)

    def is_in_range(self, allocation: BufferPoolAllocation) -> bool:
        """
        Check if the memory region that 'allocation' points
        to is in the address space (and at frame boundary)
        of this SizeClass.
        This is a wrapper around c_is_in_range.
        """
        cdef:
            uint8_t* ptr = allocation.ptr
        return self.c_is_in_range(ptr)

    cdef BufferPoolAllocation c_get_alloc_for_frame(self, idx: int):
        allocation = BufferPoolAllocation()
        allocation.set_ptr(self.c_size_class.getFrameAddress(idx))
        allocation.set_size(self.c_size_class.getBlockSize())
        return allocation
    
    def get_alloc_for_frame(self, idx: int) -> BufferPoolAllocation:
        """
        Get a BufferPoolAllocation instance that has the
        information corresponding to the frame at index 'idx'
        of this SizeClass. In particular, the 'ptr' will point
        to the start of the frame and 'size' will be the block
        size of this SizeClass.
        This is a wrapper around c_get_alloc_for_frame.
        """
        return self.c_get_alloc_for_frame(idx)
    
    def get_frame_index(self, allocation: BufferPoolAllocation) -> int:
        """
        Wrapper around 'getFrameIndex' function in the C++ class.
        Gets the index of the frame that the 'ptr' in a
        BufferPoolAllocation points to.
        """
        cdef:
            int idx = self.c_size_class.getFrameIndex(allocation.ptr)
        return idx

    def get_block_size(self) -> int:
        """
        Get the size in bytes of each block/frame in this SizeClass.
        """
        cdef:
            int size = self.c_size_class.getBlockSize()
        return size
    
    def get_num_blocks(self) -> int:
        """
        Get number of blocks/frames in this SizeClass.
        """
        cdef:
            int num_blocks = self.c_size_class.getNumBlocks()
        return num_blocks
    
    def is_frame_mapped(self, idx: int) -> bool:
        """
        Check if frame at index 'idx' is mapped.
        """
        cdef:
            bint is_mapped = self.c_size_class.isFrameMapped(idx)
        return is_mapped
    
    def is_frame_pinned(self, idx: int) -> bool:
        """
        Check if frame at index 'idx' is pined.
        """
        cdef:
            bint is_pinned = self.c_size_class.isFramePinned(idx)
        return is_pinned
        

cdef class BufferPoolOptions(_Weakrefable):
    """
    Simple Python class around the BufferPoolOptions class in C++.
    NOTE: This is only for unit-testing purposes at this point.
    """

    cdef:
        # Underlying C++ object for storing the information.
        CBufferPoolOptions options

    def __init__(self, *,
                 memory_size=None, 
                 min_size_class=None,
                 max_num_size_classes=None,
                 ignore_max_limit_during_allocation=None):
        """
        Constructor for BufferPoolOptions.
        If the attributes are not provided, they default
        to the default values set in C++.
        """
        self.options = CBufferPoolOptions()
        if memory_size is not None:
            self.options.memory_size = memory_size
        if min_size_class is not None:
            self.options.min_size_class = min_size_class
        if max_num_size_classes is not None:
            self.options.max_num_size_classes = max_num_size_classes
        if ignore_max_limit_during_allocation is not None:
            self.options.ignore_max_limit_during_allocation = ignore_max_limit_during_allocation

    cdef void cinit(self, CBufferPoolOptions options):
        self.options = options
    
    @property
    def memory_size(self):
        return self.options.memory_size
    
    @property
    def min_size_class(self):
        return self.options.min_size_class
    
    @property
    def max_num_size_classes(self):
        return self.options.max_num_size_classes
    
    @property
    def ignore_max_limit_during_allocation(self):
        return self.options.ignore_max_limit_during_allocation

    @staticmethod
    def defaults():
        cdef:
            BufferPoolOptions options = BufferPoolOptions.__new__(BufferPoolOptions)
            CBufferPoolOptions default_options = CBufferPoolOptions.Defaults()
        options.cinit(default_options)
        return options


cdef class BufferPool(MemoryPool):
    """
    Python interface to BufferPool class defined in C++.
    This will be primarily used for unit-testing and
    PyArrow integration (we need a way to provide a
    pointer to our BufferPool to PyArrow to set it as
    the default memory pool for all PyArrow allocations).
    """

    # Each instance will store a shared_ptr to the
    # C++ instance of the BufferPool.
    cdef shared_ptr[CBufferPool] c_pool

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, "
                        "use default_buffer_pool(), "
                        "BufferPool.default() or "
                        "BufferPool.from_options() instead."
                        .format(self.__class__.__name__))

    cdef void cinit(self, shared_ptr[CBufferPool] c_pool):
        """
        Set the underlying shared_ptr for this BufferPool instance.
        """
        self.c_pool = c_pool

    @staticmethod
    def default():
        """
        Get Python wrapper around the default BufferPool instance.
        This will be used for all real world usage of BufferPool
        such as the PyArrow integration.
        """
        cdef:
            # Create an uninitialized instance.
            BufferPool pool = BufferPool.__new__(BufferPool)
            # Get shared_ptr to the Default instance
            shared_ptr[CBufferPool] c_pool = CBufferPool.Default()
        # Set the shared_ptr for the instance.
        pool.cinit(c_pool)
        return pool

    @staticmethod
    def from_options(options: BufferPoolOptions) -> BufferPool:
        """
        Create a BufferPool from the provided options.
        This will be used to create small pools for unit-testing
        purposes.
        """
        cdef:
            # Create an uninitialized instance.
            BufferPool pool = BufferPool.__new__(BufferPool)
            # Create a BufferPool in C++ from the provided options
            # and get shared_ptr to it.
            shared_ptr[CBufferPool] c_pool = make_shared[CBufferPool](options.options)
        # Set the shared_ptr for the instance.
        pool.cinit(c_pool)
        return pool

    def bytes_allocated(self) -> int:
        """
        Get number of bytes currently allocated by the
        BufferPool.
        """
        return deref(self.c_pool).get_bytes_allocated()
    
    def num_size_classes(self) -> int:
        """
        Get the number of SizeClass-es created by the
        BufferPool.
        """
        return deref(self.c_pool).num_size_classes()
    
    def max_memory(self) -> int:
        """
        Get the max-memory usage of the BufferPool
        at any point in its lifetime.
        """
        return deref(self.c_pool).get_max_memory()
    
    def release_unused(self):
        """
        NOP in this case
        """
        pass
    
    @property
    def backend_name(self) -> str:
        """
        Get the memory backend. Returns 'bodo'.
        """
        return frombytes(deref(self.c_pool).get_backend_name())
    
    cdef CBufferPool* get_pool_ptr(self):
        return self.c_pool.get()
    
    ## The functions below are only for unit-testing purposes
    ## and might not be safe for regular usage.

    def get_size_class(self, idx: int) -> SizeClass:
        """
        Get the SizeClass at index 'idx'. This will
        create a SizeClass instance using a raw pointer
        from an underlying unique_ptr, which can be unsafe
        if used after the BufferPool goes out of scope.
        """
        cdef:
            # Get a raw pointer to the C++ SizeClass instance.
            CSizeClass* c_size_class = deref(self.c_pool).GetSizeClass_Unsafe(idx)
            # Create an uninitialized instance of SizeClass.
            SizeClass size_class = SizeClass.__new__(SizeClass)
        # Set the raw pointer in the SizeClass instance.
        size_class.cinit(c_size_class)
        return size_class

    cdef BufferPoolAllocation c_allocate(self, int size, int alignment):
        # Create an empty BufferPoolAllocation instance.
        allocation = BufferPoolAllocation()
        # Allocate memory through the BufferPool.
        # Pass the pointer in the BufferPoolAllocation instance
        # to be used as the "swip".
        check_status(deref(self.c_pool).Allocate(size, alignment, &(allocation.ptr)))
        # Set the allocation size. This is required during
        # free and reallocate.
        allocation.size = size
        allocation.alignment = alignment
        return allocation
    
    cdef void c_free(self, BufferPoolAllocation allocation):
        deref(self.c_pool).Free(allocation.ptr, allocation.size, allocation.alignment)
    
    cdef void c_reallocate(self, int new_size, BufferPoolAllocation allocation):
        check_status(deref(self.c_pool).Reallocate(allocation.size, new_size, allocation.alignment, &(allocation.ptr)))
        allocation.size = new_size

    def allocate(self, size, alignment=64) -> BufferPoolAllocation:
        """
        Wrapper around c_allocate. We encode the information
        from the allocation in a BufferPoolAllocation object. 
        """
        return self.c_allocate(size, alignment)

    def free(self, allocation: BufferPoolAllocation):
        """
        Free memory allocated through BuferrPool.
        We use the information from a BufferPoolAllocation
        instance to get the information about the memory region
        and allocated size.
        """
        self.c_free(allocation)
    
    def reallocate(self, new_size: int, allocation: BufferPoolAllocation):
        """
        Resize a previous allocation to 'new_size' bytes.
        The BufferPoolAllocation instance is updated in place.
        """
        self.c_reallocate(new_size, allocation)


def default_buffer_pool():
    """
    Get the default BufferPool instance.
    """
    return BufferPool.default()


def get_arrow_memory_pool_wrapper_for_buffer_pool(buffer_pool: BufferPool) -> MemoryPool:
    """
    Get an Arrow MemoryPool instance that actually points to
    and uses a BufferPool instance.
    This is required since `pa.set_memory_pool` requires a
    MemoryPool instance. Doing this is safe since our
    Python and C++ BufferPool classes inherit from
    Arrow's MemoryPool class.
    """
    cdef:
        MemoryPool pool = MemoryPool.__new__(MemoryPool)
    pool.init(buffer_pool.get_pool_ptr())
    return pool


def get_arrow_memory_pool_wrapper_for_default_buffer_pool() -> MemoryPool:
    """
    Specialization of get_arrow_memory_pool_wrapper_for_buffer_pool
    to get an Arrow MemoryPool wrapper around the default
    BufferPool instance.
    """
    cdef:
        BufferPool buffer_pool = BufferPool.default()
    return get_arrow_memory_pool_wrapper_for_buffer_pool(buffer_pool)


def set_default_buffer_pool_as_arrow_memory_pool():
    """
    Helper function to set our default BufferPool instance
    as the default MemoryPool for all PyArrow allocations.
    """
    pa.set_memory_pool(get_arrow_memory_pool_wrapper_for_default_buffer_pool())
