#include "typeinfer.h"

#include <object.h>
#include <pytypedefs.h>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "ir.h"
#include "native_typer.h"
#include "type.h"

namespace bodo {

void TypeInferer::build_constraint() {
    BuildConstraintVisitor build_constraint_visitor(*this);
    for (const std::shared_ptr<Block>& block :
         std::views::values(this->func_ir->blocks)) {
        for (const std::shared_ptr<Stmt>& inst : block->body) {
            inst->accept(build_constraint_visitor);
        }
    }
}

void ArgConstraint::apply(TypeInferer& inferer) {
    TypeVar& src = inferer.typevars.get(this->src);

    if (!src.defined()) {
        return;
    }

    Type* ty = src.getone();
    // TODO[BSE-3931]: handle Omitted
    if (!ty->is_precise()) {
        throw std::runtime_error(
            fmt::format("non-precise type {}", ty->ToString()));
    }
    inferer.add_type(this->dst, ty, this->loc);
};

void Propagate::apply(TypeInferer& inferer) {
    inferer.copy_type(this->src, this->dst, this->loc);
    // If `dst` is refined, notify us
    inferer.refine_map[dst] = this;
};

void Propagate::refine(TypeInferer& inferer, Type* target_type) {
    if (!target_type->is_precise()) {
        throw std::runtime_error(fmt::format(
            "Propagate::refine: non-precise type {}", target_type->ToString()));
    }
    // Do not back-propagate to locked variables (e.g. constants)
    inferer.add_type(this->src, target_type, this->loc, true);
};

void CallConstraint::apply(TypeInferer& inferer) {
    Type* fnty = inferer.typevars.get(this->func_varname).getone();
    this->resolve(inferer, inferer.typevars, fnty);
};

// Get Numba Type objects for arguments for calling back to Numba
PyObject* box_arg_types(const std::vector<Type*>& args) {
    PyObject* tuple = PyTuple_New(args.size());
    for (size_t i = 0; i < args.size(); i++) {
        PyObject* typ = args[i]->to_py();
        // NOTE: PyTuple_SetItem steals the reference
        PyTuple_SetItem(tuple, i, typ);
    }
    return tuple;
}

/**
 * @brief Calls typing_context.resolve_function_type() to get signature of
 * function with specified argument types. Has faster shortcut for common
 * functions.
 *
 * @param fnty input function type
 * @param args call argument types
 * @return std::shared_ptr<Signature> call signature
 */
std::shared_ptr<Signature> resolve_function_type(Type* fnty,
                                                 std::vector<Type*>& args) {
    // Shortcut for functions handled in native type inference
    BodoFunction* bodo_func_type = dynamic_cast<BodoFunction*>(fnty);
    if (bodo_func_type) {
        return bodo_func_type->get_call_type(args);
    }

    // Call back to Numba to get function type
    // sig = typing_context.resolve_function_type(fnty, pos_args, kw_args)
    PyObject* args_obj = box_arg_types(args);
    PyObject* typing_context =
        TypingContextWrapper::getInstance().typing_context;
    PyObject* py_fn_type = fnty->to_py();
    PyObject* empty_dict = PyDict_New();
    PyObject* sig_obj =
        PyObject_CallMethod(typing_context, "resolve_function_type", "OOO",
                            py_fn_type, args_obj, empty_dict);
    Py_DECREF(py_fn_type);
    Py_DECREF(empty_dict);
    Py_DECREF(args_obj);
    // Catch typing failure
    if (!sig_obj) {
        throw std::runtime_error(
            "typing_context.resolve_function_type() failed");
    }

    std::shared_ptr<Signature> sig = unbox_sig(sig_obj);
    if (PyErr_Occurred()) {
        throw std::runtime_error("unbox_sig failed");
    }
    return sig;
}

// https://github.com/Bodo-inc/Bodo/blob/18094bb44a243e9f2160e2500de0e4ce69db8afb/bodo/numba_compat.py#L2952
void CallConstraint::resolve(TypeInferer& inferer, TypeVarMap typevars,
                             Type* fnty) {
    // TODO[BSE-3931]: handle argument folding
    if (this->kws.size() != 0) {
        throw std::runtime_error(
            "CallConstraint::resolve: kws not supported yet");
    }

    for (std::shared_ptr<Var>& v : this->args) {
        // Cannot resolve call type until all argument types are known
        TypeVar& tv = inferer.typevars.get(v->name);
        if (!tv.defined()) {
            return;
        }

        // Check argument to be precise
        Type* ty = tv.getone();
        // TODO[BSE-3931]: handle array types
        if (!ty->is_precise()) {
            return;
        }
    }

    // TODO[BSE-3931]: handle TypeRef/ExternalFunction/BodoFunction
    // TODO[BSE-3931]: FunctionType/RecursiveCall

    std::vector<Type*> arg_types;
    for (std::shared_ptr<Var>& v : args) {
        arg_types.push_back(inferer.typevars.get(v->name).getone());
    }
    std::shared_ptr<Signature> sig = resolve_function_type(fnty, arg_types);

    inferer.add_type(this->target_varname, sig->return_type, this->loc);

    // TODO[BSE-3931]: handle BoundFunction

    // If the return type is imprecise but can be unified with the
    // target variable's inferred type, use the latter.
    // Useful for code such as::
    //    s = set()
    //    s.add(1)
    // (the set() call must be typed as int64(), not undefined())
    if (!sig->return_type->is_precise()) {
        TypeVar& target = inferer.typevars.get(this->target_varname);
        if (target.defined()) {
            Type* targetty = target.getone();
            Type* unified = unify_pairs(targetty, sig->return_type);
            if (unified && (unified == targetty)) {
                sig->return_type = targetty;
            }
        }
    }

    this->signature = sig;
    // TODO[BSE-3931]: handle refine map
}

/**
 * @brief Resolves getattr type by calling typing_context.resolve_getattr().
 *
 * @param value_type type of value in getattr
 * @param attr attribute of value
 * @return Type* attribute type
 */
Type* resolve_getattr(Type* value_type, const std::string attr) {
    // Shortcut for module values and functions handled in native type inference
    Module* mod_ty = dynamic_cast<Module*>(value_type);
    if (mod_ty) {
        // Shortcut for module values
        PyObject* mod_val =
            PyObject_GetAttrString(mod_ty->getPyModule(), attr.c_str());
        if (PyModule_Check(mod_val)) {
            Type* out_typ = Module::get(mod_val);
            Py_DECREF(mod_val);
            return out_typ;
        }

        PyObject* mod_name =
            PyObject_GetAttrString(mod_ty->getPyModule(), "__name__");
        std::string func_path =
            fmt::format("{}.{}", PyUnicode_AsUTF8(mod_name), attr);
        CallTyperRegistry& registry = CallTyperRegistry::getInstance();
        auto it = registry.callTypers.find(func_path);
        if (it != registry.callTypers.end()) {
            return BodoFunction::get(func_path, it->second);
        }
    }

    // attrty = typing_context.resolve_getattr(ty, attr)
    PyObject* typing_context =
        TypingContextWrapper::getInstance().typing_context;
    PyObject* ty_obj = value_type->to_py();
    PyObject* attrty_obj = PyObject_CallMethod(
        typing_context, "resolve_getattr", "Os", ty_obj, attr.c_str());
    Py_DECREF(ty_obj);
    // Catch typing failure
    if (!attrty_obj) {
        throw std::runtime_error("typing_context.resolve_getattr() failed");
    }

    Type* attrty = unbox_type(attrty_obj);
    Py_DECREF(attrty_obj);
    if (PyErr_Occurred()) {
        throw std::runtime_error("unbox_type failed");
    }
    return attrty;
}

void GetAttrConstraint::apply(TypeInferer& inferer) {
    std::optional<Type*> ty = inferer.typevars.get(this->value->name).get();
    if (ty.has_value()) {
        Type* attrty = resolve_getattr(ty.value(), this->attr);

        if (!attrty->is_precise()) {
            throw std::runtime_error(
                "GetAttrConstraint::apply(): imprecise type found");
        }

        inferer.add_type(this->target_varname, attrty, this->loc);
    }

    inferer.refine_map[this->target_varname] = this;
}

void GetAttrConstraint::refine(TypeInferer& inferer, Type* target_type) {
    // TODO[BSE-3931]: handle BoundFunction
}

/**
 * @brief Calls typing_context.resolve_value_type() to get type for object. Has
 * faster shortcuts for common types like module.
 *
 * @param value input object to type
 * @return Type* type of object
 */
Type* resolve_value_type(PyObject* value) {
    // Shortcut for Module type since common
    if (PyModule_Check(value)) {
        return Module::get(value);
    }

    // typ = typing_context.resolve_value_type(value)
    PyObject* typing_context =
        TypingContextWrapper::getInstance().typing_context;
    PyObject* t_obj =
        PyObject_CallMethod(typing_context, "resolve_value_type", "O", value);
    // Catch typing failure
    if (!t_obj) {
        throw std::runtime_error("typing_context.resolve_value_type() failed");
    }

    Type* typ = unbox_type(t_obj);
    if (PyErr_Occurred()) {
        throw std::runtime_error("unbox_type failed");
    }
    Py_DECREF(t_obj);
    return typ;
}

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typeinfer.py#L1579
void BuildConstraintVisitor::visit(const Global* global,
                                   const std::string& target_varname,
                                   std::shared_ptr<Loc> loc) const {
    Type* typ = resolve_value_type(global->value);

    // TODO[BSE-3931]: handle recursive calls
    // TODO[BSE-3931]: handle setting Array type to readonly
    // TODO[BSE-3931]: handle tuples
    // TODO[BSE-3931]: handle invalid global modification
    // (sentry_modified_builtin)
    // TODO[BSE-3931]: handle literal values

    TypeVar& tv = inferer.typevars.get(target_varname);
    if (tv.locked) {
        tv.add_type(typ, loc);
    } else {
        inferer.lock_type(target_varname, typ, loc);
    }

    // Bodo change: removed assumed_immutables since unused
}

void BuildConstraintVisitor::visit(const CallExpr* callexpr,
                                   const std::string& target_varname,
                                   std::shared_ptr<Loc> loc) const {
    assert(callexpr->varkwarg == std::nullopt);
    std::unique_ptr<Constraint> constraint = std::make_unique<CallConstraint>(
        target_varname, callexpr->func->name, callexpr->args, callexpr->kws,
        callexpr->vararg, loc);
    inferer.calls.push_back(std::make_pair((Expr*)callexpr, constraint.get()));
    inferer.constraints.append(std::move(constraint));
}

};  // namespace bodo
