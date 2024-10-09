// Copyright (C) 2024 Bodo Inc. All rights reserved.
#pragma once
#include <cassert>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ir.h"
#include "ir_utils.h"
#include "type.h"

namespace bodo {

/**
 * @brief Singleton class that wraps
 * numba.core.registry.cpu_target.typing_context object to allow reuse (avoiding
 * import overheads).
 *
 */
class TypingContextWrapper {
   public:
    PyObject* typing_context;
    TypingContextWrapper() {
        // get numba.core.registry.cpu_target.typing_context
        PyObject* registry_mod = PyImport_ImportModule("numba.core.registry");
        PyObject* cpu_target_obj =
            PyObject_GetAttrString(registry_mod, "cpu_target");
        typing_context =
            PyObject_GetAttrString(cpu_target_obj, "typing_context");
        Py_DECREF(registry_mod);
        Py_DECREF(cpu_target_obj);
    }
    // TODO: add decref to a Python atexit since destructor is called too late,
    // causing segfault ~TypingContextWrapper() { Py_DECREF(typing_context); }
    static TypingContextWrapper& getInstance() {
        static TypingContextWrapper instance;
        return instance;
    }
    TypingContextWrapper(TypingContextWrapper const&) = delete;
    void operator=(TypingContextWrapper const&) = delete;
};

/**
 * @brief Struct for returning infer.unify() results to Cython
 *
 */
class UnifyResult {
   public:
    // TODO[BSE-3930]: use UniqueDict
    std::unordered_map<std::string, Type*> typemap;
    Type* return_type;
    // TODO[BSE-3930]: use UniqueDict
    // calltypes stores Expr pointers to pass to Cython but does not "own" them
    // (no need to dealloc). This is fine since IR nodes stay alive and
    // unchanged during type inference.
    std::unordered_map<Expr*, std::shared_ptr<Signature>> calltypes;
};

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typeinfer.py#L43
class TypeVar {
   private:
    // Bodo change: removed context
    // TODO[BSE-3927]: support define_loc attribute
    // TODO[BSE-3926]: support literal_value attribute
    std::string varname;
    std::optional<Type*> type;

   public:
    bool locked = false;

    TypeVar(std::string _varname) : varname(_varname), type(std::nullopt) {}

    TypeVar() = default;

    void lock(Type* tp, std::shared_ptr<Loc> loc) {
        // TODO[BSE-3926]: support literal_value argument
        if (this->locked) {
            assert(this->type.has_value());
            // TODO[BSE-3927]: raise compiler error with loc
            throw std::runtime_error(fmt::format(
                "Invalid reassignment of a type-variable detected, type "
                "variables are locked according to the user provided "
                "function signature or from an ir.Const node. This is a "
                "bug! Type={}. {}",
                this->type.value()->ToString(), tp->ToString()));
        }

        // If there is already a type, ensure we can convert it to the
        // locked type.
        if (this->type.has_value() &&
            !can_convert(this->type.value(), tp).has_value()) {
            throw std::runtime_error(fmt::format(
                "No conversion from {} to {} for '{}'", tp->ToString(),
                this->type.value()->ToString(), this->varname));
        }

        // TODO[BSE-3927]: support define_loc attribute
        // TODO[BSE-3926]: support literal_value attribute
        this->type = tp;
        this->locked = true;
    }

    Type* getone() {
        // TODO[BSE-3927]: Better error message
        if (!this->type.has_value()) {
            // TODO[BSE-3927]: raise compiler error
            throw std::runtime_error("TypeVar not defined");
        }
        return this->type.value();
    }

    Type* add_type(Type* tp, std::shared_ptr<Loc> loc) {
        // Special case for _undef_var.
        // If the typevar is the _undef_var, use the incoming type directly.
        if (this->type.has_value() && (this->type.value() == UndefVar::get())) {
            this->type = tp;
            return this->type.value();
        }

        if (this->locked) {
            if (this->type.has_value() && (this->type.value() != tp)) {
                if (!can_convert(tp, this->type.value()).has_value()) {
                    throw std::runtime_error(fmt::format(
                        "No conversion from {} to {} for '{}'", tp->ToString(),
                        this->type.value()->ToString(), this->varname));
                }
            }
        } else {
            Type* unified;
            if (this->type.has_value()) {
                unified = unify_pairs(this->type.value(), tp);
                if (!unified) {
                    throw std::runtime_error(fmt::format(
                        "Cannot unify {} and {} for '{}'", tp->ToString(),
                        this->type.value()->ToString(), this->varname));
                }
            } else {
                unified = tp;
            }
            this->type = unified;
        }

        return this->type.value();
    }

    void _union(TypeVar& other, std::shared_ptr<Loc> loc) {
        if (other.type.has_value()) {
            this->add_type(other.type.value(), loc);
        }
        // Bodo change: not returning type since not needed
    }

    // Bodo change: return scalar instead of tuple
    std::optional<Type*> get() { return this->type; }

    bool defined() { return this->type.has_value(); }
};

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typeinfer.py#L912
class TypeVarMap {
   public:
    std::unordered_map<std::string, TypeVar> map;

    TypeVar& get(const std::string& varname) {
        // Insert a new TypeVar for variable if not exists
        auto it = this->map.try_emplace(varname);
        return (*it.first).second;
    }
};

class TypeInferer;

class Constraint {
   public:
    virtual void apply(TypeInferer& inferer) = 0;
    virtual ~Constraint() = default;
    virtual std::shared_ptr<Signature> get_call_signature() {
        throw std::runtime_error(
            "Constraint doesn't support get_call_signature()");
    }
};

class RefinableConstraint : public Constraint {
   public:
    virtual void refine(TypeInferer& inferer, Type* target_type) = 0;
};

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typeinfer.py#L191
class Propagate : public RefinableConstraint {
    std::string dst;
    std::string src;
    std::shared_ptr<Loc> loc;

   public:
    Propagate(std::string dst, std::string src, std::shared_ptr<Loc> loc)
        : dst(dst), src(src), loc(loc) {}

    void apply(TypeInferer& inferer) override;
    void refine(TypeInferer& inferer, Type* target_type) override;
};

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typeinfer.py#L215
class ArgConstraint : public Constraint {
    std::string dst;
    std::string src;
    std::shared_ptr<Loc> loc;

   public:
    ArgConstraint(std::string dst, std::string src, std::shared_ptr<Loc> loc)
        : dst(dst), src(src), loc(loc) {}

    void apply(TypeInferer& inferer) override;
};

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typeinfer.py#L561
class CallConstraint : public Constraint {
   protected:
    std::string target_varname;
    std::string func_varname;
    std::vector<std::shared_ptr<Var>> args;
    std::vector<std::pair<std::string, std::shared_ptr<Var>>> kws;
    std::optional<std::shared_ptr<Var>> vararg;
    std::shared_ptr<Loc> loc;
    std::shared_ptr<Signature> signature;

    void resolve(TypeInferer& inferer, TypeVarMap typevars, Type* fnty);

   public:
    CallConstraint(
        std::string target_varname, std::string func_varname,
        std::vector<std::shared_ptr<Var>> args,
        std::vector<std::pair<std::string, std::shared_ptr<Var>>> kws,
        std::optional<std::shared_ptr<Var>> vararg, std::shared_ptr<Loc> loc)
        : target_varname(target_varname),
          func_varname(func_varname),
          args(args),
          kws(kws),
          vararg(vararg),
          loc(loc),
          signature(nullptr) {}

    void apply(TypeInferer& inferer) override;
    std::shared_ptr<Signature> get_call_signature() override {
        return signature;
    }
};

class GetAttrConstraint : public RefinableConstraint {
    const std::string target_varname;
    const std::string attr;
    const std::shared_ptr<Var> value;
    const std::shared_ptr<Loc> loc;

   public:
    GetAttrConstraint(std::string target_varname, std::string attr,
                      std::shared_ptr<Var> value, std::shared_ptr<Loc> loc)
        : target_varname(target_varname), attr(attr), value(value), loc(loc) {}

    void apply(TypeInferer& inferer) override;
    void refine(TypeInferer& inferer, Type* target_type) override;
};

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typeinfer.py#L136
class ConstraintNetwork {
    std::vector<std::unique_ptr<Constraint>> constraints;

   public:
    void append(std::unique_ptr<Constraint> constraint) {
        this->constraints.push_back(std::move(constraint));
    }
    void propagate(TypeInferer& inferer) {
        for (auto& constraint : this->constraints) {
            constraint->apply(inferer);
            // TODO[BSE-3931]: add all of the Error handling
        }
    }
};

// https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/typeinfer.py#L960
class TypeInferer {
   private:
    std::shared_ptr<FunctionIR> func_ir;

   public:
    // TODO[BSE-3930]: add other attributes as necessary
    TypeVarMap typevars;
    ConstraintNetwork constraints;
    // calls store Expr pointers to pass to calltypes eventually, but do not
    // "own" them (no need to dealloc). This is fine since IR nodes stay alive
    // and unchanged during type inference.
    std::vector<std::pair<Expr*, Constraint*>> calls;
    // Target var -> constraint with refine hook
    // NOTE: doesn't own RefinableConstraint pointers so no need to delete.
    // Also, Constraints stay alive during type inference.
    std::unordered_map<std::string, RefinableConstraint*> refine_map;

    TypeInferer(std::shared_ptr<FunctionIR> _func_ir) : func_ir(_func_ir) {}
    // Default constructor is necessary for Cython but shouldn't be used
    TypeInferer() : func_ir(nullptr) {}

    // Disambiguise argument name
    std::string _mangle_arg_name(std::string name) { return "arg." + name; }

    void seed_argument(std::string name, uint64_t index, Type* typ) {
        name = this->_mangle_arg_name(name);
        this->seed_type(name, typ);
        // Bodo change: avoiding arg_names class variable since only used for
        // generator handling
    }

    void seed_type(std::string var, Type* tp) {
        this->lock_type(var, tp, nullptr);
    }

    void lock_type(std::string var, Type* tp, std::shared_ptr<Loc> loc) {
        // TODO[BSE-3926]: support literal_value argument
        TypeVar& tv = this->typevars.get(var);
        tv.lock(tp, loc);
    }

    void add_type(std::string var, Type* tp, std::shared_ptr<Loc> loc,
                  bool unless_locked = false) {
        TypeVar& tv = this->typevars.get(var);
        if (unless_locked && tv.locked) {
            return;
        }
        std::optional<Type*> oldty = tv.get();
        Type* unified = tv.add_type(tp, loc);
        if (oldty.has_value() && (unified != oldty.value())) {
            this->propagate_refined_type(var, unified);
        }
    }

    void propagate_refined_type(const std::string& updated_var,
                                Type* updated_type) {
        if (this->refine_map.contains(updated_var)) {
            RefinableConstraint* source_constraint =
                this->refine_map.at(updated_var);
            source_constraint->refine(*this, updated_type);
        }
    }

    void copy_type(std::string src_var, std::string dst_var,
                   std::shared_ptr<Loc> loc) {
        this->typevars.get(dst_var)._union(this->typevars.get(src_var), loc);
    }

    /// The algorithm is monotonic.  It can only grow or "refine" the
    /// typevar map.
    std::vector<std::optional<Type*>> get_state_token() {
        std::vector<std::optional<Type*>> v;
        v.reserve(this->typevars.map.size());
        // TODO[BSE-3930]: sort by key
        for (TypeVar& t : std::views::values(this->typevars.map)) {
            v.push_back(t.get());
        }
        return v;
    }

    void build_constraint();

    void propagate(bool raise_errors = true) {
        // TODO[BSE-3930]: avoid vector value copy overhead?
        std::vector<std::optional<Type*>> newtoken = this->get_state_token();
        std::vector<std::optional<Type*>> oldtoken;
        // Since the number of types are finite, the typesets will eventually
        // stop growing.

        while (newtoken != oldtoken) {
            oldtoken = newtoken;
            this->constraints.propagate(*this);
            newtoken = this->get_state_token();
        }
        // TODO[BSE-3927]: handle returning errors and ForceLiteralArg handling
    }

    std::vector<std::shared_ptr<Var>> _get_return_vars() {
        std::vector<std::shared_ptr<Var>> ret_vars;
        for (const std::shared_ptr<Block>& block :
             std::views::values(this->func_ir->blocks)) {
            assert(block->body.size() > 0);
            std::shared_ptr<Terminator> const& terminator =
                block->get_terminator();
            if (std::shared_ptr<Return> ret =
                    std::dynamic_pointer_cast<Return>(terminator)) {
                ret_vars.push_back(ret->value);
            }
        }

        return ret_vars;
    }

    Type* _unify_return_types(std::vector<Type*>& ret_types) {
        // Function without a successful return path
        if (ret_types.size() == 0) {
            return NoneType::get();
        }

        Type* unified = unify_types(ret_types);
        // TODO[BSE-3930]: check FunctionType
        if (!unified || !unified->is_precise()) {
            // TODO[BSE-3927]: throw better error like Numba
            throw std::runtime_error("Type unification failed");
        }

        return unified;
    }

    Type* get_return_type(std::unordered_map<std::string, Type*> typemap) {
        // Get types of all return values
        std::vector<Type*> ret_types;
        for (const std::shared_ptr<Var>& var : this->_get_return_vars()) {
            ret_types.push_back(typemap[var->name]);
        }

        Type* retty = this->_unify_return_types(ret_types);

        // Check return value is not undefined
        if (retty == UndefVar::get()) {
            throw std::runtime_error("return value is undefined");
        }

        return retty;
    }

    std::unordered_map<Expr*, std::shared_ptr<Signature>> get_function_types() {
        std::unordered_map<Expr*, std::shared_ptr<Signature>> calltypes;
        for (auto& p : this->calls) {
            calltypes[p.first] = p.second->get_call_signature();
        }
        return calltypes;
    }

    std::shared_ptr<UnifyResult> unify(bool raise_errors = true) {
        std::shared_ptr<UnifyResult> unify_res =
            std::make_shared<UnifyResult>();

        for (auto& [varname, tv] : this->typevars.map) {
            if (!tv.defined()) {
                if (raise_errors) {
                    throw std::runtime_error(fmt::format(
                        "Type of variable '{}' cannot be determined", varname));
                } else {
                    unify_res->typemap[varname] = Unknown::get();
                    continue;
                }
            }
            Type* tp = tv.getone();

            // TODO[BSE-3930]: check UndefinedFunctionType

            if (!tp->is_precise()) {
                if (raise_errors) {
                    throw std::runtime_error(
                        fmt::format("Cannot infer the type of variable {}, "
                                    "have imprecise type: {}",
                                    varname, tp->ToString()));
                } else {
                    unify_res->typemap[varname] = Unknown::get();
                    continue;
                }
            }

            unify_res->typemap[varname] = tp;
        }
        unify_res->return_type = this->get_return_type(unify_res->typemap);

        // set return type for return variables
        for (const std::shared_ptr<Var>& var : this->_get_return_vars()) {
            unify_res->typemap[var->name] = unify_res->return_type;
        }

        unify_res->calltypes = this->get_function_types();

        // Check for undefined variables in the call arguments.
        for (std::shared_ptr<Signature>& sig :
             std::views::values(unify_res->calltypes)) {
            if (sig != nullptr) {
                for (Type* arg_type : sig->args) {
                    if (arg_type == UndefVar::get()) {
                        throw std::runtime_error(
                            "undefined variable used in call argument");
                    }
                }
            }
        }

        return unify_res;
    }
};

#define TODO                                                               \
    throw std::runtime_error("BuildConstraintVisitor: TODO" __FILE__ ":" + \
                             std::to_string(__LINE__));

class BuildConstraintVisitor : public InstVisitor {
   private:
    TypeInferer& inferer;

   public:
    BuildConstraintVisitor(TypeInferer& _inferer) : inferer(_inferer) {}

    // TODO[BSE-3931]: handle all IR types properly
    // NOTE: visitor methods take raw pointers since "this" pointer is passed by
    // IR objects. This is fine since visitor doesn't manipulate, pass or store
    // the pointers (just reads attributes) and the IR stays alive during the
    // visit.
    void visit(const Assign* assign) const override {}
    void visit(const Return* ret) const override {}
    void visit(const Branch* ret) const override {}
    void visit(const Jump* ret) const override { TODO }
    void visit(const Raise* ret) const override { TODO }
    void visit(const StaticRaise* ret) const override { TODO }
    void visit(const SetItem* ret) const override { TODO }
    void visit(const StaticSetItem* ret) const override { TODO }
    void visit(const SetAttr* ret) const override { TODO }
    void visit(const Del* ret) const override { TODO }
    void visit(const Print* ret) const override { TODO }
    void visit(const EnterWith* ret) const override { TODO }
    void visit(const PopBlock* ret) const override { TODO }

    void visit(const FreeVar* freevar, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }

    void visit(const BinOpExpr* binop, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }
    void visit(const InPlaceBinOpExpr* binop, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }
    void visit(const UnaryExpr* unary, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }
    void visit(const BuildTupleExpr* tuple, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }
    void visit(const BuildListExpr* list, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }
    void visit(const PairFirstExpr* pairfirst,
               const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }
    void visit(const PairSecondExpr* pairsecond,
               const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }
    void visit(const GetIterExpr* getiter, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }
    void visit(const IterNextExpr* iternext, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }
    void visit(const ExhaustIterExpr* exhaustiter,
               const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }
    void visit(const GetItemExpr* getitem, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }
    void visit(const StaticGetItemExpr* getitem,
               const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }
    void visit(const PhiExpr* phi, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        TODO
    }

    void visit(const Arg* arg, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        inferer.constraints.append(std::move(std::make_unique<ArgConstraint>(
            target_varname, inferer._mangle_arg_name(arg->name), loc)));
    }
    void visit(const Var* var, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        inferer.constraints.append(std::move(
            std::make_unique<Propagate>(target_varname, var->name, loc)));
    }
    void visit(const Const* constant, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        Type* ty = constant->getType();
        inferer.add_type(target_varname, ty, loc);
    }

    void visit(const Global* global, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override;

    void visit(const CallExpr* global, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override;

    void visit(const CastExpr* cast, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        inferer.constraints.append(std::move(std::make_unique<Propagate>(
            target_varname, cast->value->name, loc)));
    }
    void visit(const GetAttrExpr* getattr, const std::string& target_varname,
               std::shared_ptr<Loc> loc) const override {
        inferer.constraints.append(
            std::move(std::make_unique<GetAttrConstraint>(
                target_varname, getattr->attr, getattr->value, loc)));
    }
};

}  // namespace bodo
