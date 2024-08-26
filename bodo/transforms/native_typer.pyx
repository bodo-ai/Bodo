# Copyright (C) 2024 Bodo Inc. All rights reserved.
# distutils: language = c++

from numba.core import types
from numba.core.typing.templates import signature

from libcpp cimport bool as c_bool
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
from libcpp.string cimport string as c_string


cdef extern from "fast_typer.h" namespace "bodo" nogil:
    cdef cppclass CType" bodo::Type":
        CType()
        object to_py()

    cdef cppclass CNoneType" bodo::NoneType"(CType):
        CNoneType()
        object to_py()

    cdef cppclass CInteger" bodo::Integer"(CType):
        CInteger(int bitwidth, c_bool is_signed)
        CInteger()
        object to_py()
        @staticmethod
        shared_ptr[CInteger] getInstance(int bitwidth, c_bool is_signed)

    cdef cppclass CEnumMember" bodo::EnumMember"(CType):
        # passing PyObject* as void* to avoid Cython issues
        CEnumMember(void* instance_class, shared_ptr[CType] dtype)
        CEnumMember()
        object to_py()

    cdef cppclass CSignature" bodo::Signature"(CType):
        shared_ptr[CType] return_type
        vector[shared_ptr[CType]] args
        CSignature(shared_ptr[CType] return_type, vector[shared_ptr[CType]] args)
        CSignature()

    shared_ptr[CSignature] resolve_call_type(char* func_path, object args)


cdef class CTypeWrapper:
    """Wrapper around native CType value to pass to Python (cache in Numba type)
    """
    cdef shared_ptr[CType] c_type

    @staticmethod
    cdef from_c_type(shared_ptr[CType] ct):
        cdef CTypeWrapper wrapper = CTypeWrapper.__new__(CTypeWrapper)
        wrapper.c_type = ct
        return wrapper

    cdef shared_ptr[CType] get_c_type(self):
        return self.c_type


cdef shared_ptr[CType] unbox_type(object t):
    """Unbox Type object to equivalent native CType value
    """
    cdef CTypeWrapper wrapped_type

    # Use wrapped CType in the object if available
    if hasattr(t, "c_type"):
        wrapped_type = t.c_type
        return wrapped_type.get_c_type()

    if isinstance(t, types.Integer):
        ret = <shared_ptr[CType]> CInteger.getInstance(<int>t.bitwidth, <c_bool>t.signed)
        # Wrap and cache the CType in Numba type object for reuse
        t.c_type = CTypeWrapper.from_c_type(ret)
        return ret

    if isinstance(t, types.EnumMember):
        ret = <shared_ptr[CType]> make_shared[CEnumMember](<void*>t.instance_class, <shared_ptr[CType]>unbox_type(t.dtype))
        t.c_type = CTypeWrapper.from_c_type(ret)
        return ret

    # NOTE: avoiding exception here to allow more optimized codegen potentially
    return make_shared[CType]()


cdef public vector[shared_ptr[CType]] unbox_args(pos_args):
    """Unbox tuple of Type objects into native vector of CType values
    """
    cdef vector[shared_ptr[CType]] ctypes
    for t in pos_args:
        ctypes.push_back(unbox_type(t))

    return ctypes


cdef box_sig(shared_ptr[CSignature] csig):
    """Box CSignature value into Signature objects
    """
    return signature(csig.get().return_type.get().to_py(), *tuple(t.get().to_py() for t in csig.get().args))


def bodo_resolve_call(func_key, pos_args):
    """Resolve call type using native type inference in C++
    """
    return box_sig(resolve_call_type(func_key, pos_args))
