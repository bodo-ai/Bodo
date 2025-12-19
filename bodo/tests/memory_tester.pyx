# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True

# Here we will create the glue code which will put together
# the C++ capabilities and turn it into Python classes
# and methods.
# For the most part, this will be used for integrating
# our BufferPool with PyArrow and for unit testing purposes.

from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc
from libc.string cimport strcpy
from pyarrow.lib cimport MemoryPool, _Weakrefable, check_status

from pyarrow.lib import frombytes

import bodo
import bodo.memory_cpp


# Initialize the buffer pool pointer to be the one from the main module and
# make sure we have a single buffer pool. Necessary since memory_tester module
# doesn't call bodo_common_init().
init_buffer_pool_ptr(bodo.memory_cpp.default_buffer_pool_ptr())


cdef class BufferPoolAllocation:
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

    # Pointer to the allocated memory region
    cdef uint8_t* ptr
    # Size of the allocation
    cdef int size
    # Alignment for the allocation
    cdef int alignment

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
        cdef BufferPoolAllocation new_object
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

    def is_nullptr(self) -> bool:
        """
        Does the allocation point to nullptr
        """
        return self.get_ptr_as_int() == 0

    def is_in_memory(self) -> bool:
        """
        Check if the frame is currently
        in memory (by checking the swip ptr)
        """
        return self.get_ptr_as_int() > 0

    cdef int64_t c_swip_as_int(self):
        return <int64_t> &self.ptr

    def get_swip_as_int(self) -> int:
        return int(self.c_swip_as_int())

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

    cdef bytes read(self, int n_bytes):
        cdef char* contents = <char *> malloc((n_bytes + 1) * sizeof(char))
        # TODO: Malloc fails?
        strcpy(contents, <char*> self.ptr)
        return contents

    cdef void write(self, char* contents):
        strcpy(<char*> self.ptr, contents)

    def read_bytes(self, n_bytes: int) -> bytes:
        """
        Read n bytes of the allocation
        """
        contents = self.read(n_bytes)
        return contents

    def write_bytes(self, contents: bytes) -> None:
        """
        Write some contents to the allocation
        """
        self.write(contents)

    def __repr__(self):
        cdef size_t ptr
        ptr = <size_t> self.ptr
        return f"BufferPoolAllocation({ptr}, size={self.size}, alignment={self.alignment})"


cdef class SizeClass(_Weakrefable):
    """
    Python class around the SizeClass class defined in C++.
    It defines wrappers for most of the public APIs and
    properties exposed by the C++ class.
    NOTE: This is only for unit-testing purposes at this point.
    """

    # Store a pointer to the C++ instance.
    # All APIs will use this to build wrappers
    # around the C++ class.
    cdef CSizeClass* c_size_class

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
        cdef uint8_t* ptr = allocation.ptr
        return self.c_is_in_range(ptr)

    cdef BufferPoolAllocation c_build_alloc_for_frame(self, int idx):
        cdef BufferPoolAllocation allocation
        allocation = BufferPoolAllocation()
        allocation.set_ptr(self.c_size_class.getFrameAddress(idx))
        allocation.set_size(self.c_size_class.getBlockSize())
        return allocation

    def build_alloc_for_frame(self, idx: int) -> BufferPoolAllocation:
        """
        Create a BufferPoolAllocation instance that has the
        information corresponding to the frame at index 'idx'
        of this SizeClass. In particular, the 'ptr' will point
        to the start of the frame and 'size' will be the block
        size of this SizeClass.
        This is a wrapper around c_build_alloc_for_frame.
        """
        return self.c_build_alloc_for_frame(idx)

    cdef int64_t c_get_swip_at_frame(self, int idx):
        cdef uint8_t** ptr = self.c_size_class.getSwip(idx)
        if ptr == nullptr:
            return -1

        return <int64_t> ptr

    def get_swip_at_frame(self, idx: int) -> int | None:
        """
        Get the BufferPoolAllocation instance that is currently
        allocated in the frame at index 'idx' of this SizeClass.
        """
        res = int(self.c_get_swip_at_frame(idx))
        if res == -1:
            return None
        return res

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


cdef class StorageOptions(_Weakrefable):
    """
    Simple Python class around the StorageOptions class in C++.
    NOTE: Only for unit-testing purposes
    """

    cdef:
        # Underlying C++ object for storing the information.
        shared_ptr[CStorageOptions] options

    def __init__(self, location, storage_type=None, usable_size=None, tracing_mode=None):
        """
        Constructor for StorageOptions.
        If the attributes are not provided, they default
        to values set in C++.
        """
        self.options = make_shared[CStorageOptions]()
        deref(self.options).location = location
        if usable_size is not None:
            deref(self.options).usable_size = usable_size
        if storage_type is not None:
            deref(self.options).type = storage_type
        if tracing_mode is not None:
            deref(self.options).tracing_mode = tracing_mode

    cdef void cinit(self, shared_ptr[CStorageOptions] options):
        self.options = options

    @property
    def location(self):
        return deref(self.options).location

    @property
    def usable_size(self):
        return deref(self.options).usable_size

    @staticmethod
    def defaults(rank):
        cdef:
            StorageOptions options = StorageOptions.__new__(StorageOptions)
            shared_ptr[CStorageOptions] default_options = CStorageOptions.Defaults(rank)
        if default_options == nullptr:
            return None
        options.cinit(default_options)
        return options


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
                 enforce_max_limit_during_allocation=None,
                 storage_options=None,
                 spill_on_unpin=None,
                 move_on_unpin=None,
                 debug_mode=None,
                 trace_level=None,
                 malloc_free_trim_threshold=None):
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
        if enforce_max_limit_during_allocation is not None:
            self.options.enforce_max_limit_during_allocation = enforce_max_limit_during_allocation
        if storage_options is not None:
            for option in storage_options:
                self.c_add_storage(option)
        if spill_on_unpin is not None:
            self.options.spill_on_unpin = spill_on_unpin
        if move_on_unpin is not None:
            self.options.move_on_unpin = move_on_unpin
        if debug_mode is not None:
            self.options.debug_mode = debug_mode
        if trace_level is not None:
            self.options.trace_level = trace_level
        if malloc_free_trim_threshold is not None:
            self.options.malloc_free_trim_threshold = malloc_free_trim_threshold

    cdef void c_add_storage(self, StorageOptions option):
        self.options.storage_options.push_back(option.options)

    def c_storage_option(self, i):
        cdef:
            shared_ptr[CStorageOptions] c_opt = self.options.storage_options[i]
            StorageOptions options = StorageOptions.__new__(StorageOptions)
        options.cinit(c_opt)
        return options

    cdef int c_storage_options_len(self):
       return self.options.storage_options.size()

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
    def enforce_max_limit_during_allocation(self):
        return self.options.enforce_max_limit_during_allocation

    @property
    def storage_options(self):
        res = []
        for i in range(self.c_storage_options_len()):
            res.append(self.c_storage_option(i))
        return res
    
    @property
    def spill_on_unpin(self):
        return self.options.spill_on_unpin
    
    @property
    def move_on_unpin(self):
        return self.options.move_on_unpin
    
    @property
    def debug_mode(self):
        return self.options.debug_mode

    @property
    def trace_level(self):
        return self.options.trace_level

    @property
    def malloc_free_trim_threshold(self):
        return self.options.malloc_free_trim_threshold

    @staticmethod
    def defaults():
        cdef:
            BufferPoolOptions options = BufferPoolOptions.__new__(BufferPoolOptions)
            CBufferPoolOptions default_options = CBufferPoolOptions.Defaults()
        options.cinit(default_options)
        return options


cdef class IBufferPool(MemoryPool):
    """
    Python interface to IBufferPool class defined in C++.
    This is just an abstract class defined for consistency
    between the Python and C++ class hierarchy.
    """

    def __init__(self):
        raise TypeError("{} is an abstract class. Use "
                        "default_buffer_pool(), "
                        "BufferPool.default() or "
                        "BufferPool.from_options() instead."
                        .format(self.__class__.__name__))


cdef class BufferPool(IBufferPool):
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

    def cleanup(self):
        return deref(self.c_pool).Cleanup()

    def bytes_pinned(self) -> int:
        """
        Get the number of bytes currently pinned in the
        BufferPool.
        """
        return deref(self.c_pool).get_bytes_pinned()

    def bytes_in_memory(self) -> int:
        """
        Get number of bytes currently in memory by the
        BufferPool.
        """
        return deref(self.c_pool).get_bytes_in_memory()

    def bytes_allocated(self) -> int:
        """
        Get number of bytes currently allocated by the
        BufferPool including spilled memory.
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
        at any point in its lifetime, including spilled memory.
        """
        return deref(self.c_pool).get_max_memory()

    def is_spilling_enabled(self) -> c_bool:
        return deref(self.c_pool).is_spilling_enabled()

    def release_unused(self):
        """
        NOP in this case
        """
        pass

    def bytes_freed_through_malloc_since_last_trim(self) -> int:
        """
        Get the number of bytes freed through malloc since
        the last time malloc_trim was called.
        NOTE: Only applicable on Linux.
        """
        return deref(self.c_pool).get_bytes_freed_through_malloc_since_last_trim()

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

    cdef void c_free(self, BufferPoolAllocation allocation) noexcept:
        deref(self.c_pool).Free(allocation.ptr, allocation.size, allocation.alignment)

    cdef void c_reallocate(self, int new_size, BufferPoolAllocation allocation) except *:
        check_status(deref(self.c_pool).Reallocate(allocation.size, new_size, allocation.alignment, &(allocation.ptr)))
        allocation.size = new_size

    cdef void c_pin(self, BufferPoolAllocation allocation) except *:
        check_status(deref(self.c_pool).Pin(&(allocation.ptr), allocation.size, allocation.alignment))

    cdef void c_unpin(self, BufferPoolAllocation allocation) noexcept:
        deref(self.c_pool).Unpin(allocation.ptr, allocation.size, allocation.alignment)

    cdef c_bool c_is_pinned(self, BufferPoolAllocation allocation) except *:
        return deref(self.c_pool).IsPinned(allocation.ptr, allocation.size, allocation.alignment)

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

    def pin(self, allocation: BufferPoolAllocation):
        """
        Pin a previous allocation to memory
        """
        self.c_pin(allocation)

    def unpin(self, allocation: BufferPoolAllocation):
        """
        Unpin a previous allocation to memory, allowing
        it to be spilled if storage managers are provided
        """
        self.c_unpin(allocation)

    def is_pinned(self, allocation: BufferPoolAllocation) -> bool:
        """
        Check if an allocation is pinned.
        """
        return self.c_is_pinned(allocation)


    cdef uint64_t c_get_smallest_size_class_size(self):
        return deref(self.c_pool).GetSmallestSizeClassSize()

    def get_smallest_size_class_size(self) -> int:
        """
        Get the size of the smallest SizeClass.
        Returns 0 when there are no SizeClass-es.
        """
        return self.c_get_smallest_size_class_size()

    def __eq__(self, other: BufferPool) -> bool:
        """
        Check if two BufferPool instances are
        equivalent, i.e. they both point to the same
        underlying pool in C++.
        """
        if isinstance(other, BufferPool):
            return self.c_pool.get() == other.c_pool.get()
        else:
            return False

    def __ne__(self, other: BufferPool) -> bool:
        """
        Inverse of __eq__.
        """
        return not self.__eq__(other)


cdef class OperatorBufferPool(IBufferPool):
    """
    Python interface to OperatorBufferPool class defined in C++.
    This will be used for unit-testing purposes only at this point.
    """

    # Each instance will store a shared_ptr to the
    # C++ instance of the OperatorBufferPool
    cdef shared_ptr[COperatorBufferPool] c_pool

    def __init__(self,
                 int operator_budget_bytes,
                 parent_pool: BufferPool = None,
                 double error_threshold = 0.5):
        """
        Create a new OperatorBufferPool for an operator with
        a budget of 'operator_budget_bytes' bytes. 
        It will use the provided 'parent_pool'.
        If a parent_pool is not specified, the default
        BufferPool will be used.
        The 'error_threshold' defines the fraction of
        'operator_budget_bytes' at which the OperatorBufferPool
        will raise a 'OperatorPoolThresholdExceededError' error.
        """
        if parent_pool is None:
            parent_pool = BufferPool.default()

        # We pass operator Id as -1 since this is only used for unit
        # testing purposes and the operator ID is non-consequential at
        # this point.
        self.c_pool = make_shared[COperatorBufferPool](-1, operator_budget_bytes,
                                                       parent_pool.c_pool,
                                                       error_threshold)

    @property
    def operator_budget_bytes(self) -> int:
        """
        Getter for the 'operator_budget_bytes' attribute.
        """
        return (deref(self.c_pool)).get_operator_budget_bytes()
    
    @property
    def memory_error_threshold(self) -> int:
        """
        Getter for the 'memory_error_threshold' attribute.
        """
        return (deref(self.c_pool)).get_memory_error_threshold()
    
    @property
    def error_threshold(self) -> double:
        """
        Getter for the 'error_threshold' attribute
        """
        return (deref(self.c_pool)).get_error_threshold()

    @property
    def threshold_enforcement_enabled(self) -> bool:
        """
        Getter for the 'threshold_enforcement_enabled' attribute.
        """
        return (deref(self.c_pool)).ThresholdEnforcementEnabled()

    @property
    def parent_pool(self) -> BufferPool:
        """
        Getter for the parent pool. We wrap it in a new
        BufferPool instance.
        """
        cdef:
            # Create an uninitialized instance.
            BufferPool parent_pool = BufferPool.__new__(BufferPool)
            # Get shared_ptr to the parent pool
            shared_ptr[CBufferPool] parent_pool_shared_ptr = (deref(self.c_pool)).get_parent_pool()
        # Set the shared_ptr for the instance.
        parent_pool.cinit(parent_pool_shared_ptr)
        return parent_pool

    cdef void c_set_error_threshold(self, double error_threshold) except *:
        (deref(self.c_pool)).SetErrorThreshold(error_threshold)
    
    def set_error_threshold(self, error_threshold: float):
        """
        Set the error threshold ratio.
        """
        self.c_set_error_threshold(error_threshold)
    
    cdef void c_set_budget(self, int new_operator_budget) except *:
        (deref(self.c_pool)).SetBudget(new_operator_budget)
    
    def set_budget(self, new_operator_budget: int):
        """
        Update the budget. The new budget must be lower
        than the existing budget.
        """
        self.c_set_budget(new_operator_budget)

    cdef void c_enable_threshold_enforcement(self) except *:
        (deref(self.c_pool)).EnableThresholdEnforcement()

    def enable_threshold_enforcement(self):
        """
        Enable threshold enforcement for this OperatorBufferPool.
        """
        self.c_enable_threshold_enforcement()

    cdef void c_disable_threshold_enforcement(self) except *:
        (deref(self.c_pool)).DisableThresholdEnforcement()

    def disable_threshold_enforcement(self):
        """
        Disable threshold enforcement for this OperatorBufferPool.
        """
        self.c_disable_threshold_enforcement()

    def main_mem_bytes_allocated(self) -> int:
        """
        Get number of bytes currently allocated by the
        OperatorBufferPool in the main mem section.
        """
        return deref(self.c_pool).main_mem_bytes_allocated()
    
    def scratch_mem_bytes_allocated(self) -> int:
        """
        Get number of bytes currently allocated by the
        OperatorBufferPool in the scratch mem section.
        """
        return deref(self.c_pool).scratch_mem_bytes_allocated()
    
    def main_mem_bytes_pinned(self) -> int:
        """
        Get number of bytes currently pinned by the
        OperatorBufferPool in the main mem section.
        """
        return deref(self.c_pool).main_mem_bytes_pinned()
    
    def scratch_mem_bytes_pinned(self) -> int:
        """
        Get number of bytes currently pinned by the
        OperatorBufferPool in the scratch mem section.
        """
        return deref(self.c_pool).scratch_mem_bytes_pinned()
    
    def main_mem_max_memory(self) -> int:
        """
        Get the max-memory usage of the main mem section
        of the OperatorBufferPool at any point in its lifetime.
        """
        return deref(self.c_pool).main_mem_max_memory()

    ## NOTE: The functions below are mostly copied over from BufferPool implementation.
    ## Ideally we'd be able to move them to IBufferPool and reuse them in both
    ## BufferPool and OperatorBufferPool, but Cython's support for inheritance
    ## is not great and there were a lot of conflicts/errors when trying to do
    ## that.

    def bytes_pinned(self) -> int:
        """
        Get the number of bytes currently pinned by the
        OperatorBufferPool.
        """
        return deref(self.c_pool).get_bytes_pinned()

    def bytes_allocated(self) -> int:
        """
        Get number of bytes currently allocated by the
        OperatorBufferPool.
        """
        return deref(self.c_pool).get_bytes_allocated()

    def max_memory(self) -> int:
        """
        Get the max-memory usage of the OperatorBufferPool
        at any point in its lifetime.
        """
        return deref(self.c_pool).get_max_memory()

    @property
    def backend_name(self) -> str:
        """
        Get the memory backend. Returns memory backend
        of the parent pool.
        """
        return frombytes(deref(self.c_pool).get_backend_name())

    cdef BufferPoolAllocation c_allocate(self, int size, int alignment):
        # Create an empty BufferPoolAllocation instance.
        allocation = BufferPoolAllocation()
        # Allocate memory through the OperatorBufferPool.
        # Pass the pointer in the BufferPoolAllocation instance
        # to be used as the "swip".
        check_status(deref(self.c_pool).Allocate(size, alignment, &(allocation.ptr)))
        # Set the allocation size. This is required during
        # free and reallocate.
        allocation.size = size
        allocation.alignment = alignment
        return allocation

    cdef void c_free(self, BufferPoolAllocation allocation) except *:
        deref(self.c_pool).Free(allocation.ptr, allocation.size, allocation.alignment)

    cdef void c_reallocate(self, int new_size, BufferPoolAllocation allocation) except *:
        check_status(deref(self.c_pool).Reallocate(allocation.size, new_size, allocation.alignment, &(allocation.ptr)))
        allocation.size = new_size

    cdef void c_pin(self, BufferPoolAllocation allocation) except *:
        check_status(deref(self.c_pool).Pin(&(allocation.ptr), allocation.size, allocation.alignment))

    cdef void c_unpin(self, BufferPoolAllocation allocation) except *:
        deref(self.c_pool).Unpin(allocation.ptr, allocation.size, allocation.alignment)

    def allocate(self, size, alignment=64) -> BufferPoolAllocation:
        """
        Wrapper around c_allocate. We encode the information
        from the allocation in a BufferPoolAllocation object.
        """
        return self.c_allocate(size, alignment)

    def free(self, allocation: BufferPoolAllocation):
        """
        Free memory allocated through OperatorBufferPool.
        We use the information from a BufferPoolAllocation
        instance to get the information about the memory region
        and allocated size.
        """
        self.c_free(allocation)

    def reallocate(self, new_size: int, allocation: BufferPoolAllocation):
        """
        Resize an allocation to 'new_size' bytes.
        The BufferPoolAllocation instance is updated in place.
        """
        self.c_reallocate(new_size, allocation)

    def pin(self, allocation: BufferPoolAllocation):
        """
        Pin an allocation to memory.
        """
        self.c_pin(allocation)

    def unpin(self, allocation: BufferPoolAllocation):
        """
        Unpin an allocation from memory.
        """
        self.c_unpin(allocation)

    def __eq__(self, other: OperatorBufferPool) -> bool:
        """
        Check if two OperatorBufferPool instances are
        equivalent, i.e. they both point to the same
        underlying pool in C++.
        """
        if isinstance(other, OperatorBufferPool):
            return self.c_pool.get() == other.c_pool.get()
        else:
            return False

    def __ne__(self, other: OperatorBufferPool) -> bool:
        """
        Inverse of __eq__.
        """
        return not self.__eq__(other)

    ## Equivalent functions for scratch allocations

    cdef BufferPoolAllocation c_allocate_scratch(self, int size, int alignment):
        # Create an empty BufferPoolAllocation instance.
        allocation = BufferPoolAllocation()
        # Allocate memory through the OperatorBufferPool's AllocateScratch API.
        # Pass the pointer in the BufferPoolAllocation instance
        # to be used as the "swip".
        check_status(deref(self.c_pool).AllocateScratch(size, alignment, &(allocation.ptr)))
        # Set the allocation size. This is required during
        # free and reallocate.
        allocation.size = size
        allocation.alignment = alignment
        return allocation

    cdef void c_free_scratch(self, BufferPoolAllocation allocation) except *:
        deref(self.c_pool).FreeScratch(allocation.ptr, allocation.size, allocation.alignment)

    cdef void c_reallocate_scratch(self, int new_size, BufferPoolAllocation allocation) except *:
        check_status(deref(self.c_pool).ReallocateScratch(allocation.size, new_size, allocation.alignment, &(allocation.ptr)))
        allocation.size = new_size

    cdef void c_pin_scratch(self, BufferPoolAllocation allocation) except *:
        check_status(deref(self.c_pool).PinScratch(&(allocation.ptr), allocation.size, allocation.alignment))

    cdef void c_unpin_scratch(self, BufferPoolAllocation allocation) except *:
        deref(self.c_pool).UnpinScratch(allocation.ptr, allocation.size, allocation.alignment)

    def allocate_scratch(self, size, alignment=64) -> BufferPoolAllocation:
        """
        Wrapper around c_allocate_scratch. We encode the information
        from the allocation in a BufferPoolAllocation object.
        """
        return self.c_allocate_scratch(size, alignment)

    def free_scratch(self, allocation: BufferPoolAllocation):
        """
        Free scratch memory allocated through OperatorBufferPool.
        We use the information from a BufferPoolAllocation
        instance to get the information about the memory region
        and allocated size.
        """
        self.c_free_scratch(allocation)

    def reallocate_scratch(self, new_size: int, allocation: BufferPoolAllocation):
        """
        Resize a scratch allocation to 'new_size' bytes.
        The BufferPoolAllocation instance is updated in place.
        """
        self.c_reallocate_scratch(new_size, allocation)

    def pin_scratch(self, allocation: BufferPoolAllocation):
        """
        Pin a scratch allocation to memory.
        """
        self.c_pin_scratch(allocation)

    def unpin_scratch(self, allocation: BufferPoolAllocation):
        """
        Unpin a scratch allocation from memory.
        """
        self.c_unpin_scratch(allocation)


cdef class OperatorScratchPool(IBufferPool):
    """
    Python interface to OperatorScratchPool defined in C++.
    This will be used for unit-testing purposes only at this point.
    """

    # Each instance will store a shared_ptr to the
    # C++ instance of the OperatorScratchPool
    cdef shared_ptr[COperatorScratchPool] c_pool

    def __init__(self, parent_pool: OperatorBufferPool):
        """
        Create a new OperatorScratchPool for a given
        OperatorBufferPool.
        """
        self.c_pool = make_shared[COperatorScratchPool](parent_pool.c_pool.get())
    
    ## NOTE: The functions below are mostly copied over from BufferPool implementation.
    ## Ideally we'd be able to move them to IBufferPool and reuse them in both
    ## BufferPool and OperatorScratchPool, but Cython's support for inheritance
    ## is not great and there were a lot of conflicts/errors when trying to do
    ## that.

    def bytes_pinned(self) -> int:
        """
        Get the number of scratch bytes currently pinned by the
        OperatorScratchPool.
        """
        return deref(self.c_pool).get_bytes_pinned()

    def bytes_allocated(self) -> int:
        """
        Get number of bytes currently allocated by the
        OperatorScratchPool.
        """
        return deref(self.c_pool).get_bytes_allocated()

    def max_memory(self) -> int:
        """
        Get the max-memory usage of the OperatorScratchPool
        at any point in its lifetime.
        This will always return 0 at this point since we don't
        track this.
        """
        return deref(self.c_pool).get_max_memory()

    @property
    def backend_name(self) -> str:
        """
        Get the memory backend. Returns memory backend
        of the parent pool.
        """
        return frombytes(deref(self.c_pool).get_backend_name())

    cdef BufferPoolAllocation c_allocate(self, int size, int alignment):
        # Create an empty BufferPoolAllocation instance.
        allocation = BufferPoolAllocation()
        # Allocate memory through the OperatorScratchPool.
        # Pass the pointer in the BufferPoolAllocation instance
        # to be used as the "swip".
        check_status(deref(self.c_pool).Allocate(size, alignment, &(allocation.ptr)))
        # Set the allocation size. This is required during
        # free and reallocate.
        allocation.size = size
        allocation.alignment = alignment
        return allocation

    cdef void c_free(self, BufferPoolAllocation allocation) except *:
        deref(self.c_pool).Free(allocation.ptr, allocation.size, allocation.alignment)

    cdef void c_reallocate(self, int new_size, BufferPoolAllocation allocation) except *:
        check_status(deref(self.c_pool).Reallocate(allocation.size, new_size, allocation.alignment, &(allocation.ptr)))
        allocation.size = new_size

    cdef void c_pin(self, BufferPoolAllocation allocation) except *:
        check_status(deref(self.c_pool).Pin(&(allocation.ptr), allocation.size, allocation.alignment))

    cdef void c_unpin(self, BufferPoolAllocation allocation) except *:
        deref(self.c_pool).Unpin(allocation.ptr, allocation.size, allocation.alignment)

    def allocate(self, size, alignment=64) -> BufferPoolAllocation:
        """
        Wrapper around c_allocate. We encode the information
        from the allocation in a BufferPoolAllocation object.
        """
        return self.c_allocate(size, alignment)

    def free(self, allocation: BufferPoolAllocation):
        """
        Free memory allocated through OperatorScratchPool.
        We use the information from a BufferPoolAllocation
        instance to get the information about the memory region
        and allocated size.
        """
        self.c_free(allocation)

    def reallocate(self, new_size: int, allocation: BufferPoolAllocation):
        """
        Resize an allocation to 'new_size' bytes.
        The BufferPoolAllocation instance is updated in place.
        """
        self.c_reallocate(new_size, allocation)

    def pin(self, allocation: BufferPoolAllocation):
        """
        Pin an allocation to memory.
        """
        self.c_pin(allocation)

    def unpin(self, allocation: BufferPoolAllocation):
        """
        Unpin an allocation from memory.
        """
        self.c_unpin(allocation)

    def __eq__(self, other: OperatorScratchPool) -> bool:
        """
        Check if two OperatorScratchPool instances are
        equivalent, i.e. they both point to the same
        underlying pool in C++.
        """
        if isinstance(other, OperatorScratchPool):
            return self.c_pool.get() == other.c_pool.get()
        else:
            return False

    def __ne__(self, other: OperatorScratchPool) -> bool:
        """
        Inverse of __eq__.
        """
        return not self.__eq__(other)


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
