// Copyright (C) 2024 Bodo Inc. All rights reserved.
#pragma once

#include <boost/container_hash/hash.hpp>
#include <functional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Python.h>
#include <fmt/format.h>

namespace bodo {

/// A conversion kind from one type to the other. The enum members are ordered
/// from stricter to looser.
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typeconv/castgraph.py#L6
enum Conversion {
    /// The two types are identical
    exact = 1,
    /// The two types are of the same kind, the destination type has more
    /// extension or precision than the source type (e.g. float32 -> float64,
    /// or int32 -> int64)
    promote = 2,
    /// The source type can be converted to the destination type without loss
    /// of information (e.g. int32 -> int64).  Note that the conversion may
    /// still fail explicitly at runtime (e.g. Optional(int32) -> int32)
    safe = 3,
    /// The conversion may appear to succeed at runtime while losing information
    /// or precision (e.g. int32 -> uint32, float64 -> float32, int64 -> int32,
    /// etc.)
    unsafe = 4,
    /// This value is only used internally
    nil = 99,
};

class Type;
class Signature;
typedef std::function<std::shared_ptr<Signature>(std::vector<Type*>&)>
    InferFunc;

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typing/context.py#L561
std::optional<Conversion> can_convert(Type* fromty, Type* toty);

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

/**
 * @brief Singleton class that wraps
 * bodo.transforms.type_inference.typeinfer.bodo_registry.function_map object to
 * allow reuse (avoiding import overheads) and proper decref.
 *
 */
class FunctionMapWrapper {
   public:
    PyObject* function_map;
    FunctionMapWrapper() {
        PyObject* infer_mod =
            PyImport_ImportModule("bodo.transforms.type_inference.typeinfer");
        PyObject* bodo_registry_obj =
            PyObject_GetAttrString(infer_mod, "bodo_registry");
        function_map =
            PyObject_GetAttrString(bodo_registry_obj, "function_map");
        Py_DECREF(infer_mod);
        Py_DECREF(bodo_registry_obj);
    }
    // TODO: add decref to a Python atexit since destructor is called too late,
    // causing segfault ~FunctionMapWrapper() { Py_DECREF(function_map); }
    static FunctionMapWrapper& getInstance() {
        static FunctionMapWrapper instance;
        return instance;
    }
    FunctionMapWrapper(FunctionMapWrapper const&) = delete;
    void operator=(FunctionMapWrapper const&) = delete;
};

// Equivalent to Numba's types.Type
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/abstract.py#L81

// This follows LLVM's approach of immutable Type instances that are created
// only once and can be compared with raw pointer comparison.
// TODO: LLVM never frees Type instances but we may allow clearing states when
// "safe" (when no compilation is in progress and no pointer values are around).
// This may be necessary for Bodo compiler "servers" since complex types like
// dataframes are less likely to be reused. We may need a context object similar
// to LLVM.
// https://github.com/llvm/llvm-project/blob/ea9204505cf1099b98b1fdcb898f0bd35e463984/llvm/include/llvm/IR/Type.h#L37
// https://llvm.org/docs/ProgrammersManual.html
class Type {
   public:
    virtual std::string ToString() const = 0;
    virtual PyObject* to_py() const = 0;
    virtual bool is_precise() const { return true; }

    /**
     * @brief Try to unify this type with the *other*.  A third type must
        be returned, or nullptr if unification is not possible.
        Only override this if the coercion logic cannot be expressed
        as simple casting rules.
     *
     */
    // NOTE: returns nullptr when unification not possible
    virtual Type* unify(Type* other) { return nullptr; }

    /**
     * @brief Check whether this type can be converted to the *other*.
        If successful, must return a the conversion description, e.g.
        "exact", "promote", "unsafe", "safe"; otherwise nullopt is returned.
     */
    virtual std::optional<Conversion> can_convert_to(Type* other) {
        return std::nullopt;
    }

    /**
     * @brief Similar to *can_convert_to*, but in reverse. Only needed if
        the type provides conversion from other types.
     *
     * @param other
     */
    virtual std::optional<Conversion> can_convert_from(Type* other) {
        return std::nullopt;
    }
};

// Equivalent to Numba's types.NoneType
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/misc.py#L272
class NoneType : public Type {
   private:
    NoneType() {}

   public:
    std::string ToString() const override { return "NoneType()"; }
    PyObject* to_py() const override {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* t_obj = PyObject_GetAttrString(types_mod, "none");
        // No need to incref/decref since returning the new reference directly
        // to caller as expected
        return t_obj;
    }
    static NoneType* get() {
        static NoneType instance;
        return &instance;
    }
};

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/abstract.py#L248
class Number : virtual public Type {
   private:
    int bitwidth;

   protected:
    Number(int bitwidth) : bitwidth(bitwidth) {}

   public:
    int getBitwidth() const { return bitwidth; }
    /**
     * @brief Unify this Number type with another type if possible. Return
     * nullptr if not possible.
     *
     */
    Type* unify(Type* other) override {
        // Unify if other is Number
        Number* other_number = dynamic_cast<Number*>(other);
        if (other_number) {
            // Numba calls np.promote_types but we handle common cases manually
            // for simplicity and performance
            // TODO[BSE-3967]: match Numpy's type promotion table
            // https://github.com/numpy/numpy/blob/7e6e48ca7aacae9994d18a3dadbabd2b91c32151/numpy/_core/src/multiarray/convert_datatype.c#L976
            // https://github.com/numpy/numpy/blob/7e6e48ca7aacae9994d18a3dadbabd2b91c32151/numpy/_core/src/multiarray/scalartypes.c.src#L4256
            return this->unify_impl(other_number);
        }
        return nullptr;
    }
    /**
     * @brief Implement type unification for another Number. Must return result
     * matching np.promote_types or throw exception so we fall back to Numba
     * properly.
     *
     */
    virtual Type* unify_impl(Number* other) {
        throw std::runtime_error("Number unify not implemented yet");
    }
};

// Equivalent to Numba's types.Integer
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/scalars.py#L31
class Integer : public Number {
   private:
    bool is_signed;

   protected:
    Integer(int _bitwidth, bool _is_signed)
        : Number(_bitwidth), is_signed(_is_signed) {}

   public:
    std::string ToString() const override {
        return fmt::format("Integer({}, {})", this->getBitwidth(),
                           this->is_signed);
    }
    PyObject* to_py() const override {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* integer_obj = PyObject_GetAttrString(types_mod, "Integer");
        PyObject* t_obj =
            PyObject_CallMethod(integer_obj, "from_bitwidth", "ii",
                                this->getBitwidth(), this->is_signed);
        Py_DECREF(integer_obj);
        return t_obj;
    }
    static Integer* get(int _bitwidth, bool _is_signed) {
        static Integer int8(8, true);
        static Integer int16(16, true);
        static Integer int32(32, true);
        static Integer int64(64, true);
        static Integer int128(128, true);

        static Integer uint8(8, false);
        static Integer uint16(16, false);
        static Integer uint32(32, false);
        static Integer uint64(64, false);
        static Integer uint128(128, false);

        if (_is_signed) {
            switch (_bitwidth) {
                case 8:
                    return &int8;
                case 16:
                    return &int16;
                case 32:
                    return &int32;
                case 64:
                    return &int64;
                case 128:
                    return &int128;
                default:
                    break;
            }
        } else {
            switch (_bitwidth) {
                case 8:
                    return &uint8;
                case 16:
                    return &uint16;
                case 32:
                    return &uint32;
                case 64:
                    return &uint64;
                case 128:
                    return &uint128;
                default:
                    break;
            }
        }
        throw std::runtime_error(
            fmt::format("Invalid Integer type {} {}", _bitwidth, _is_signed));
    }
    Type* unify_impl(Number* other) override {
        Integer* other_int = dynamic_cast<Integer*>(other);
        if (other_int) {
            if (is_signed == other_int->is_signed) {
                return Integer::get(
                    std::max(this->getBitwidth(), other_int->getBitwidth()),
                    is_signed);
            }
            // TODO[BSE-3967]: upcast to float when signs don't match
        }
        throw std::runtime_error("Number unify not implemented yet");
    }
};

class Literal : virtual public Type {
   public:
    virtual Type* unliteral() = 0;
};

class IntegerLiteral : public Integer, Literal {
   private:
    int64_t value;
    // TODO[BSE-3921]: handle non-int64 cases properly
    // Numba resolves integer constants as int64 if value fits, otherwise uint64
    // There is also 32-bit case which needs investigation (call Windows cases
    // or just 32-bit ones?)
    // https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typing/typeof.py#L131
    IntegerLiteral(int64_t _value) : value(_value), Integer(64, true) {}

   public:
    static IntegerLiteral* get(int64_t _value) {
        static std::unordered_map<int64_t, IntegerLiteral*> int_literal_types;
        IntegerLiteral*& type = int_literal_types[_value];
        if (!type) {
            type = new IntegerLiteral(_value);
        }
        return type;
    }
    Type* unliteral() override { return Integer::get(64, true); }
    PyObject* to_py() const override {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* t_obj =
            PyObject_CallMethod(types_mod, "IntegerLiteral", "i", this->value);
        return t_obj;
    }
    std::string ToString() const override {
        return fmt::format("IntegerLiteral({})", this->value);
    }
    virtual std::optional<Conversion> can_convert_to(Type* other) override {
        std::optional<Conversion> conv = can_convert(this->unliteral(), other);
        if (conv.has_value()) {
            return std::max(conv.value(), Conversion::promote);
        }
        return std::nullopt;
    }
};

// Equivalent to Numba's types.boolean
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/scalars.py#L12
class Boolean : public Type {
   private:
    Boolean() {}

   public:
    std::string ToString() const override { return "bool"; }
    PyObject* to_py() const override {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* t_obj = PyObject_GetAttrString(types_mod, "boolean");
        return t_obj;
    }
    static Boolean* get() {
        static Boolean instance;
        return &instance;
    }
};

// Equivalent to Numba's types.EnumMember
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/scalars.py#L230
class EnumMember : public Type {
   private:
    // instance_class is an owned reference which needs cleaned.
    PyObject* instance_class;
    Type* dtype;

    EnumMember(void* _instance_class, Type* _dtype)
        : instance_class(static_cast<PyObject*>(_instance_class)),
          dtype(_dtype) {
        Py_INCREF(instance_class);
    }
    ~EnumMember() { Py_DECREF(instance_class); }

   public:
    static EnumMember* get(void* _instance_class, Type* _dtype);
    std::string ToString() const override {
        PyObject* class_str_obj = PyObject_Str(this->instance_class);
        std::string class_str = PyUnicode_AsUTF8(class_str_obj);
        Py_DECREF(class_str_obj);
        return fmt::format("EnumMember({}, {})", class_str,
                           this->dtype->ToString());
    }
    PyObject* to_py() const override {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* dtype_py = this->dtype->to_py();
        PyObject* t_obj = PyObject_CallMethod(types_mod, "EnumMember", "OO",
                                              this->instance_class, dtype_py);
        Py_DECREF(dtype_py);
        return t_obj;
    }
    virtual std::optional<Conversion> can_convert_to(Type* other) override {
        throw std::runtime_error(
            "EnumMember can_convert_to not implemented yet");
    }
};

// Wrapper around Numba's Function type object to pass back to Python eventually
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/functions.py#L342
class Function : public Type {
   private:
    // function_type is an owned reference which needs cleaned.
    PyObject* function_type;

    Function(void* _function_type) : function_type((PyObject*)_function_type) {
        Py_INCREF(function_type);
    }
    ~Function() { Py_DECREF(function_type); }

   public:
    static Function* get(void* function_type);
    PyObject* to_py() const override {
        // Return a new reference (need to give ownership to caller per Python
        // convention)
        Py_INCREF(function_type);
        return function_type;
    }
    std::string ToString() const override {
        PyObject* t_str_obj = PyObject_Str(this->function_type);
        std::string t_str = PyUnicode_AsUTF8(t_str_obj);
        Py_DECREF(t_str_obj);
        return t_str;
    }
};

/**
 * @brief Type for functions that can be resolved here in native type inference.
 *
 */
class BodoFunction : public Type {
   private:
    const std::string path;
    const InferFunc& infer_func;

    BodoFunction(std::string path, const InferFunc& infer_func)
        : path(path), infer_func(infer_func) {}

   public:
    static BodoFunction* get(std::string path, const InferFunc& infer_func);
    PyObject* to_py() const override {
        // function_map[path]
        PyObject* function_map = FunctionMapWrapper::getInstance().function_map;
        PyObject* func_type_obj =
            PyDict_GetItemString(function_map, this->path.c_str());
        if (!func_type_obj) {
            throw std::runtime_error(fmt::format(
                "Function {} not found in function_map", this->path));
        }
        // PyDict_GetItemString returns borrowed reference so need to incref to
        // return a new reference
        Py_INCREF(func_type_obj);
        return func_type_obj;
    }
    std::string ToString() const override {
        return fmt::format("BodoFunction(\"{}\")", path);
    }

    std::shared_ptr<Signature> get_call_type(std::vector<Type*> args) {
        return infer_func(args);
    }
};

// Equivalent of Numba's Module type:
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/misc.py#L133
class Module : public Type {
   private:
    // NOTE: pymod is a new reference and needs decrefed (since an ephemeral
    // module object is passed in resolve_getattr).
    PyObject* pymod;

    Module(void* _pymod) : pymod((PyObject*)_pymod) { Py_INCREF(pymod); }

   public:
    PyObject* getPyModule() { return pymod; }
    static Module* get(void* _pymod);
    ~Module() { Py_DECREF(pymod); }
    PyObject* to_py() const override {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* t_obj =
            PyObject_CallMethod(types_mod, "Module", "O", this->pymod);
        return t_obj;
    }
    std::string ToString() const override {
        PyObject* t_str_obj = PyObject_Str(this->pymod);
        std::string t_str = PyUnicode_AsUTF8(t_str_obj);
        Py_DECREF(t_str_obj);
        return fmt::format("Module({})", t_str);
    }
};

// Equivalent to Numba's NumberClass:
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/functions.py#L675
class NumberClass : public Type {
   private:
    const Type* instance_type;
    NumberClass(Type* instance_type) : instance_type(instance_type) {}

   public:
    static NumberClass* get(Type* instance_type);
    PyObject* to_py() const override {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* dtype_obj = instance_type->to_py();
        PyObject* t_obj =
            PyObject_CallMethod(types_mod, "NumberClass", "O", dtype_obj);
        Py_DECREF(dtype_obj);
        return t_obj;
    }
    std::string ToString() const override {
        return fmt::format("class({})", this->instance_type->ToString());
    }
};

// Equivalent to Numba's UndefVar:
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/misc.py#L36
class UndefVar : public Type {
   private:
    UndefVar() {}

   public:
    std::optional<Conversion> can_convert_to(Type* other) override {
        return Conversion::promote;
    }

    std::string ToString() const override { return "_undef_var"; }
    PyObject* to_py() const override {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* t_obj = PyObject_GetAttrString(types_mod, "_undef_var");
        // No need to incref/decref since returning the new reference directly
        // to caller as expected
        return t_obj;
    }
    static UndefVar* get() {
        static UndefVar instance;
        return &instance;
    }
};

// Equivalent to Numba's unknown:
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/__init__.py#L29
class Unknown : public Type {
   private:
    Unknown() {}

   public:
    std::string ToString() const override { return "unknown"; }
    PyObject* to_py() const override {
        PyObject* types_mod = NumbaTypesModWrapper::getInstance().types_mod;
        PyObject* t_obj = PyObject_GetAttrString(types_mod, "unknown");
        // No need to incref/decref since returning the new reference directly
        // to caller as expected
        return t_obj;
    }
    static Unknown* get() {
        static Unknown instance;
        return &instance;
    }
};

// Equivalent to Numba's Signature
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typing/templates.py#L27
class Signature {
   public:
    Type* return_type;
    std::vector<Type*> args;

    Signature(Type* _return_type, std::vector<Type*> _args)
        : return_type(_return_type), args(_args) {}
    // Default constructor is necessary for Cython but shouldn't be used
    Signature() : return_type(nullptr), args(std::vector<Type*>()) {}
    std::string ToString() const {
        auto get_str = [](Type* t) { return t->ToString(); };
        return fmt::format(
            "{} -> {}",
            fmt::join(this->args | std::views::transform(get_str), ", "),
            this->return_type->ToString());
    }
};

// Key hasher for TypeManager's cast rules map
struct TypeManagerKeyHasher {
    std::size_t operator()(const std::pair<Type*, Type*>& k) const {
        size_t seed = 0;
        // TODO[BSE-3966]: implement proper hashing for Type classes
        boost::hash_combine(seed,
                            std::hash<std::string>()(k.first->ToString()));
        boost::hash_combine(seed,
                            std::hash<std::string>()(k.second->ToString()));
        return seed;
    }
};

// Key equality for TypeManager's cast rules map
struct TypeManagerKeyEqual {
    bool operator()(const std::pair<Type*, Type*>& k1,
                    const std::pair<Type*, Type*>& k2) const {
        return ((k1.first) == (k2.first)) && ((k1.second) == (k2.second));
    }
};

/**
 * @brief Singleton that provides type casting rules equivalent to Numba's
 TypeManager:
 https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typeconv/typeconv.hpp#L63
 *
 */
class TypeManager {
   private:
    std::unordered_map<std::pair<Type*, Type*>, int, TypeManagerKeyHasher,
                       TypeManagerKeyEqual>
        tccmap;

   public:
    TypeManager() {}

    static TypeManager& getInstance() {
        static TypeManager instance;
        return instance;
    }
    TypeManager(TypeManager const&) = delete;
    void operator=(TypeManager const&) = delete;

    static void setCompatibleTypes(Type* t1, Type* t2, int conv) {
        TypeManager& instance = getInstance();
        instance.tccmap.insert_or_assign(std::make_pair(t1, t2), conv);
    }

    static std::optional<Conversion> checkCompatibleTypes(Type* fromty,
                                                          Type* toty) {
        if (fromty == toty) {
            return Conversion::exact;
        }

        std::pair<Type*, Type*> map_key = std::make_pair(fromty, toty);
        TypeManager& instance = getInstance();
        auto it = instance.tccmap.find(map_key);
        if (it != instance.tccmap.end()) {
            return Conversion(it->second);
        }
        return {};
    }
};

/**
 * @brief type inference function for bodo.libs.memory_budget.register_operator
 *
 * @param args argument types (tuple object of Type objects)
 * @return std::shared_ptr<Signature> call type signature
 */
std::shared_ptr<Signature> register_operator_infer(std::vector<Type*>& args);

std::shared_ptr<Signature> get_rank_infer(std::vector<Type*>& args);

/**
 * @brief Singleton registry of type inference functions.
 NOTE: should match
 bodo.transforms.type_inference.typeinfer.bodo_registry.function_map since we
 use it for boxing function types.
 */
class CallTyperRegistry {
   public:
    const std::unordered_map<std::string, InferFunc> callTypers = {
        {"bodo.libs.memory_budget.register_operator", register_operator_infer},
        {"bodo.get_rank", get_rank_infer},
        {"bodo.libs.distributed_api.get_rank", get_rank_infer},
    };
    static CallTyperRegistry& getInstance() {
        static CallTyperRegistry instance;
        return instance;
    }
    CallTyperRegistry(CallTyperRegistry const&) = delete;
    void operator=(CallTyperRegistry const&) = delete;

   private:
    CallTyperRegistry() {}
};

// Same as types.unliteral
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/misc.py#L63
Type* unliteral_type(Type* t);

// https://github.com/Bodo-inc/Bodo/blob/4d3e1bbb323c56c5067aeb6310e915e1a96def4c/bodo/numba_compat.py#L5361
// NOTE: returns nullptr if unification failed
Type* unify_pairs(Type* first, Type* second);

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typing/context.py#L683
Type* unify_types(std::vector<Type*>& types);

/**
 * @brief call type resolver called from Cython
 *
 * @param func_path full module path of the function to type
 * @param args argument types (tuple object of Type objects)
 * @return std::shared_ptr<Signature> call type signature
 */
std::shared_ptr<Signature> resolve_call_type(char* func_path,
                                             std::vector<Type*>& args);

}  // namespace bodo
