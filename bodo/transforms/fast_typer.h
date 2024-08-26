// Copyright (C) 2024 Bodo Inc. All rights reserved.
#pragma once

#include <Python.h>
#include <fmt/format.h>
#include <string>
#include <vector>

namespace bodo {

/**
 * @brief Singleton class that wraps numba.types object to allow reuse (avoiding
 * import overheads) and proper decref.
 *
 */
class NumbaTypesModWrapper {
   public:
    PyObject* types_mod;
    NumbaTypesModWrapper() { types_mod = PyImport_ImportModule("numba.types"); }
    // TODO: add decref to a Python atexit since destructor is called too late,
    // causing segfault ~NumbaTypesModWrapper() { Py_DECREF(types_mod); }
    static NumbaTypesModWrapper& getInstance() {
        static NumbaTypesModWrapper instance;
        return instance;
    }
    NumbaTypesModWrapper(NumbaTypesModWrapper const&) = delete;
    void operator=(NumbaTypesModWrapper const&) = delete;
};

// Equivalent to Numba's types.Type
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/abstract.py#L81
class Type {
   public:
    virtual std::string ToString() const { return "Type()"; }
    virtual PyObject* to_py() const {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* t_obj = PyObject_CallMethod(types_mod, "Type", "s", "Type()");
        return t_obj;
    }
};

// Equivalent to Numba's types.NoneType
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/misc.py#L272
class NoneType : public Type {
   public:
    NoneType() {}
    std::string ToString() const override { return "NoneType()"; }
    PyObject* to_py() const override {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* t_obj = PyObject_GetAttrString(types_mod, "none");
        return t_obj;
    }
    static std::shared_ptr<NoneType> getInstance() {
        static std::shared_ptr<NoneType> instance{new NoneType};
        return instance;
    }
};

// Equivalent to Numba's types.Integer
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/scalars.py#L31
class Integer : public Type {
   private:
    int bitwidth;
    bool is_signed;

   public:
    Integer(int _bitwidth, bool _is_signed)
        : bitwidth(_bitwidth), is_signed(_is_signed) {}
    // Default constructor is necessary for Cython but shouldn't be used
    Integer() : bitwidth(64), is_signed(true) {}

    std::string ToString() const override {
        return fmt::format("Integer({}, {})", this->bitwidth, this->is_signed);
    }
    PyObject* to_py() const override {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* integer_obj = PyObject_GetAttrString(types_mod, "Integer");
        PyObject* t_obj =
            PyObject_CallMethod(integer_obj, "from_bitwidth", "ii",
                                this->bitwidth, this->is_signed);
        Py_DECREF(integer_obj);
        return t_obj;
    }
    static std::shared_ptr<Integer> getInstance(int _bitwidth,
                                                bool _is_signed) {
        static std::shared_ptr<Integer> int64_instance{new Integer(64, true)};
        if (_bitwidth == 64 && _is_signed) {
            return int64_instance;
        }
        return std::make_shared<Integer>(_bitwidth, _is_signed);
    }
};

// Equivalent to Numba's types.EnumMember
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/scalars.py#L230
class EnumMember : public Type {
   private:
    PyObject* instance_class;
    std::shared_ptr<Type> dtype;

   public:
    EnumMember(void* _instance_class, std::shared_ptr<Type> _dtype)
        : instance_class(static_cast<PyObject*>(_instance_class)),
          dtype(_dtype) {}
    // Default constructor is necessary for Cython but shouldn't be used
    EnumMember() : instance_class(nullptr), dtype(nullptr) {}

    std::string ToString() const override {
        PyObject* class_str_obj = PyObject_Str(this->instance_class);
        std::string class_str = PyUnicode_AsUTF8(class_str_obj);
        Py_DECREF(class_str_obj);
        return fmt::format("EnumMember({}, {})", class_str,
                           this->dtype->ToString());
    }
    PyObject* to_py() const override {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* t_obj =
            PyObject_CallMethod(types_mod, "EnumMember", "OO",
                                this->instance_class, this->dtype->to_py());
        return t_obj;
    }
};

// Equivalent to Numba's Signature
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typing/templates.py#L27
class Signature {
   public:
    std::shared_ptr<Type> return_type;
    std::vector<std::shared_ptr<Type>> args;

    Signature(std::shared_ptr<Type> _return_type,
              std::vector<std::shared_ptr<Type>> _args)
        : return_type(_return_type), args(_args) {}
    // Default constructor is necessary for Cython but shouldn't be used
    Signature()
        : return_type(nullptr), args(std::vector<std::shared_ptr<Type>>()) {}
};

/**
 * @brief call type resolver called from Cython
 *
 * @param func_path full module path of the function to type
 * @param args argument types (tuple object of Type objects)
 * @return std::shared_ptr<Signature> call type signature
 */
std::shared_ptr<Signature> resolve_call_type(char* func_path, PyObject* args);

}  // namespace bodo
