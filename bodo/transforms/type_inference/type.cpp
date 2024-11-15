// Copyright (C) 2024 Bodo Inc. All rights reserved.
#include "type.h"

#include <algorithm>
#include <unordered_map>

namespace bodo {

/**
 * @brief call type resolver called from Cython
 *
 * @param func_path full module path of the function to type
 * @param args argument types (tuple object of Type objects)
 * @return std::shared_ptr<Signature> call type signature
 */
std::shared_ptr<Signature> resolve_call_type(char* func_path,
                                             std::vector<Type*>& args) {
    return CallTyperRegistry::getInstance().callTypers.at(func_path)(args);
}

// Same as types.unliteral
// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/types/misc.py#L63
Type* unliteral_type(Type* t) {
    if (Literal* lit_t = dynamic_cast<Literal*>(t)) {
        return lit_t->unliteral();
    }
    return t;
}

// https://github.com/bodo-ai/Bodo/blob/4d3e1bbb323c56c5067aeb6310e915e1a96def4c/bodo/numba_compat.py#L5361
// NOTE: returns nullptr if unification failed
Type* unify_pairs(Type* first, Type* second) {
    if (first == second) {
        return first;
    }

    // Check undefined types
    if (first == UndefVar::get()) {
        return second;
    }
    if (second == UndefVar::get()) {
        return first;
    }

    // Check unknown types
    Type* unknown = Unknown::get();
    if ((first == unknown) || (second == unknown)) {
        return unknown;
    }

    Type* unified = first->unify(second);
    if (unified) {
        return unified;
    }

    unified = second->unify(first);
    if (unified) {
        return unified;
    }

    // Other types with simple conversion rules
    std::optional<Conversion> conv = can_convert(first, second);
    if (conv.has_value() && conv.value() <= Conversion::safe) {
        // Can convert from first to second
        return second;
    }

    conv = can_convert(second, first);
    if (conv.has_value() && conv.value() <= Conversion::safe) {
        // Can convert from second to first
        return first;
    }

    // handle literals
    if (dynamic_cast<Literal*>(first) || dynamic_cast<Literal*>(second)) {
        Type* first_unlit = unliteral_type(first);
        Type* second_unlit = unliteral_type(second);
        // Avoid recursion if literal type is the same (e.g. function literals)
        // See test_groupby.py::test_groupby_agg_const_dict
        if ((first_unlit != first) || (second_unlit != second)) {
            return unify_pairs(first_unlit, second_unlit);
        }
    }

    // Cannot unify
    return nullptr;
}

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typing/context.py#L683
Type* unify_types(std::vector<Type*>& types) {
    // Sort numbers based on bitwidth before pairwise unification
    auto get_bitwidth = [](Type* t) {
        if (Number* number_type = dynamic_cast<Number*>(t)) {
            return number_type->getBitwidth();
        }
        return 0;
    };
    std::ranges::sort(types, {}, get_bitwidth);

    Type* unified = types[0];
    for (Type* t : types | std::views::drop(1)) {
        unified = unify_pairs(unified, t);
        if (!unified) {
            break;
        }
    }
    return unified;
}

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typing/context.py#L561
std::optional<Conversion> can_convert(Type* fromty, Type* toty) {
    if (fromty == toty) {
        return Conversion::exact;
    }

    // First check with the type manager (some rules are registered
    // at startup there, see numba.typeconv.rules)
    std::optional<Conversion> conv =
        TypeManager::checkCompatibleTypes(fromty, toty);
    if (conv.has_value()) {
        return conv.value();
    }

    // Fall back on type-specific rules
    std::optional<Conversion> forward = fromty->can_convert_to(toty);
    std::optional<Conversion> backward = toty->can_convert_from(fromty);
    if (!backward.has_value()) {
        return forward;
    } else if (!forward.has_value()) {
        return backward;
    } else {
        return std::min(forward.value(), backward.value());
    }
}

std::shared_ptr<Signature> register_operator_infer(std::vector<Type*>& args) {
    static std::shared_ptr<Signature> sig =
        std::make_shared<Signature>(NoneType::get(), args);
    return sig;
}

std::shared_ptr<Signature> get_rank_infer(std::vector<Type*>& args) {
    static std::shared_ptr<Signature> sig =
        std::make_shared<Signature>(Integer::get(32, true), args);
    return sig;
}

// Key hasher for EnumMember type cache
struct EnumMapKeyHasher {
    std::size_t operator()(const std::pair<PyObject*, Type*>& k) const {
        size_t seed = 0;
        // TODO[BSE-3966]: implement proper hashing for Type classes
        boost::hash_combine(seed, PyObject_Hash(k.first));
        boost::hash_combine(seed,
                            std::hash<std::string>()(k.second->ToString()));
        return seed;
    }
};

// Key equality for EnumMember type cache
struct EnumMapKeyEqual {
    bool operator()(const std::pair<PyObject*, Type*>& k1,
                    const std::pair<PyObject*, Type*>& k2) const {
        return ((k1.second) == (k2.second)) &&
               PyObject_RichCompareBool(k1.first, k2.first, Py_EQ);
    }
};

EnumMember* EnumMember::get(void* _instance_class, Type* _dtype) {
    static std::unordered_map<std::pair<PyObject*, Type*>, EnumMember*,
                              EnumMapKeyHasher, EnumMapKeyEqual>
        enum_member_types;

    EnumMember*& type =
        enum_member_types[std::make_pair((PyObject*)_instance_class, _dtype)];
    if (!type) {
        type = new EnumMember(_instance_class, _dtype);
    }
    return type;
}

// Key hasher for Function type cache
struct ObjectMapKeyHasher {
    std::size_t operator()(PyObject* const& k) const {
        return PyObject_Hash(k);
    }
};

// Key equality for Function type cache
struct ObjectMapKeyEqual {
    bool operator()(PyObject* const& k1, PyObject* const& k2) const {
        return PyObject_RichCompareBool(k1, k2, Py_EQ);
    }
};

Function* Function::get(void* function_type) {
    static std::unordered_map<PyObject*, Function*, ObjectMapKeyHasher,
                              ObjectMapKeyEqual>
        function_types;

    Function*& type = function_types[((PyObject*)function_type)];
    if (!type) {
        type = new Function(function_type);
    }
    return type;
}

BodoFunction* BodoFunction::get(std::string path, const InferFunc& infer_func) {
    static std::unordered_map<std::string, BodoFunction*> function_types;

    BodoFunction*& type = function_types[path];
    if (!type) {
        type = new BodoFunction(path, infer_func);
    }
    return type;
}

Module* Module::get(void* _pymod) {
    static std::unordered_map<PyObject*, Module*, ObjectMapKeyHasher,
                              ObjectMapKeyEqual>
        module_types;

    Module*& type = module_types[((PyObject*)_pymod)];
    if (!type) {
        type = new Module(_pymod);
    }
    return type;
}

// Key hasher for NumberClass type cache
struct NumberClassMapKeyHasher {
    // TODO[BSE-3966]: implement proper hashing for Type classes
    std::size_t operator()(Type* const& k) const {
        return std::hash<std::string>()(k->ToString());
    }
};

NumberClass* NumberClass::get(Type* instance_type) {
    static std::unordered_map<Type*, NumberClass*, NumberClassMapKeyHasher>
        number_class_types;

    NumberClass*& type = number_class_types[instance_type];
    if (!type) {
        type = new NumberClass(instance_type);
    }
    return type;
}

}  // namespace bodo
