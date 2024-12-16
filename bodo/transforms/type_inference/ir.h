// Missing IR Nodes. All classes are defined but not filled out.
//  Expressions:
//  - build_set
//  - build_map
//  - typed_getitem
//  - make_function
//  - null
//  - undefined
//  - dummy
//  Statements:
//  - DelItem
//  - DelAttr
//  - StoreMap
//  - DynamicRaise
//  - Yield
//  - Parfor

#pragma once
#include <compare>
#include <cstdint>
#include <map>
#include <memory>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "type.h"

namespace bodo {

// TODO[BSE-3921]: support all Numba IR nodes

// Forward declaration from ir_utils.h
class InstVisitor;

/// Source Location in IR
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L25
class Loc final {
   public:
    /// Name of the file
    std::string filename;
    // Shortened filename from Python
    // Stored for caching & printing purposes
    std::string short_fname;
    /// Line in file
    int64_t line;
    /// Column
    std::optional<uint64_t> col;
    /// Set to True if location is likely a jit decorator
    bool maybe_decorator;
    std::optional<std::string> lines;

    explicit Loc() = default;
    explicit Loc(std::string filename, std::string short_fname, int64_t line,
                 std::optional<uint64_t> col, bool maybe_decorator)
        : filename(filename),
          short_fname(short_fname),
          line(line),
          col(col),
          maybe_decorator(maybe_decorator),
          lines(std::nullopt) {}

    bool operator==(const Loc& other) const {
        return filename == other.filename && line == other.line &&
               col == other.col;
    }

    std::string ToRepr() const {
        return fmt::format(
            "Loc(filename={}, line={}, col={})", filename, line,
            col.has_value() ? std::to_string(col.value()) : "None");
    }

    std::string ToString() const {
        if (col.has_value()) {
            return fmt::format("{} ({}:{})", filename, line, col.value());
        }
        return fmt::format("{} ({})", filename, line);
    }

    std::string _short() const {
        return fmt::format("{}:{}", short_fname, line);
    }

    Loc with_lineno(int64_t line,
                    std::optional<uint64_t> col = std::nullopt) const {
        return Loc(filename, short_fname, line, col, maybe_decorator);
    }
};

// Abstract base class for anything that can be the RHS of an assignment.
/// This class **does not** define any methods.
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L301
class AbstractRHS {
   public:
    /// All potential operations for the RHS of an assignment
    enum RHSType {
        VAR,
        ARG,
        CONST,
        GLOBAL,
        FREEVAR,
        EXPR,
    };

    explicit AbstractRHS(RHSType op) : OP(op) {}
    ~AbstractRHS() = default;
    // Copy Constructor
    AbstractRHS(const AbstractRHS&) = default;

    RHSType get_rhs_op() const { return OP; }

    // Virtual Methods
    virtual void accept(InstVisitor& visitor, const std::string& target_varname,
                        std::shared_ptr<Loc> loc) const = 0;
    virtual std::string ToString() const = 0;

   private:
    const RHSType OP;
};

/// IR Variable
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L1088
class Var final : public AbstractRHS {
   public:
    // Variable name. Starts with $ means temp
    std::string name;
    std::shared_ptr<Loc> loc;

    explicit Var() : AbstractRHS(RHSType::VAR) {}
    explicit Var(std::string _name, std::shared_ptr<Loc> _loc)
        : AbstractRHS(RHSType::VAR), name(_name), loc(_loc) {}

    // Copy Constructor
    // TODO: Add memoization logic from Numba
    // https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L1141
    Var(const Var& other)
        : AbstractRHS(RHSType::VAR), name(other.name), loc(other.loc) {}

    // Equal to EqualityCheckinMixin for Python Var class
    // https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L236
    bool operator==(const Var& other) const { return name == other.name; }
    // Note: C++20 <=> is equivalent to Python @total_ordering
    auto operator<=>(const Var& other) const { return name <=> other.name; }

    bool is_temp() const { return name[0] == '$'; }
    // Note: __repr__ is used for testing purposes
    std::string ToRepr() const {
        return fmt::format("Var({}, {})", name, loc->_short());
    }

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override { return name; }
};

/// Helper to format variables in fmt::format without calling ToString() every
/// time
/// TODO: Can apply to other classes
inline auto format_as(std::shared_ptr<Var> v) { return v->ToString(); }

/// A function argument
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L1000
class Arg final : public AbstractRHS {
   public:
    int index;
    std::string name;
    std::shared_ptr<Loc> loc;

    explicit Arg(int _index, std::string _name, std::shared_ptr<Loc> _loc)
        : AbstractRHS(RHSType::ARG), index(_index), name(_name), loc(_loc) {}
    explicit Arg() : AbstractRHS(RHSType::ARG) {}

    // Equal to EqualityCheckinMixin for Python Arg class
    // https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L236
    bool operator==(const Arg& other) const {
        return index == other.index && name == other.name;
    }
    std::strong_ordering operator<=>(const Arg& other) const {
        if (auto cmp = index <=> other.index; cmp != 0)
            return cmp;
        return name <=> other.name;
    }

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override {
        return fmt::format("arg({}, name={})", this->index, this->name);
    };
    // TODO: Implement infer_constant
};

/// A constant value, as loaded by LOAD_CONST.
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L1016
class Const final : public AbstractRHS {
   public:
    PyObject* value;
    std::shared_ptr<Loc> loc;
    bool use_literal_type;

    explicit Const() : AbstractRHS(RHSType::CONST) {}
    explicit Const(void* value, std::shared_ptr<Loc> _loc,
                   bool _use_literal_type)
        : AbstractRHS(RHSType::CONST),
          value((PyObject*)value),
          loc(_loc),
          use_literal_type(_use_literal_type) {}

    // TODO: Implement operator== and operator<=>
    // How to handle PyObjects?

    Type* getType() const;

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override;
    // TODO: Implement infer_constant
};

/// A global variable, as loaded by LOAD_GLOBAL.
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L1038
class Global final : public AbstractRHS {
   public:
    std::string name;
    // value is a borrowed reference (no incref/decref needed).
    // Owner is outside of JIT and stays alive during compilation.
    // See https://docs.python.org/3/extending/extending.html#reference-counts
    PyObject* value;
    std::shared_ptr<Loc> loc;

    explicit Global() : AbstractRHS(RHSType::GLOBAL) {}
    explicit Global(std::string name, void* value, std::shared_ptr<Loc> loc)
        : AbstractRHS(RHSType::GLOBAL),
          name(name),
          value((PyObject*)value),
          loc(loc) {}

    // Copy Constructor
    Global(const Global& other)
        : AbstractRHS(RHSType::GLOBAL),
          name(other.name),
          value(other.value),
          loc(other.loc) {}

    bool operator==(const Global& other) const {
        bool is_equal = PyObject_RichCompareBool(value, other.value, Py_EQ);
        return name == other.name && is_equal;
    }
    // TODO: Missing operator <=> method

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override;
    // TODO: Implement infer_constant
};

/// A freevar (i.e. a variable defined in an enclosing non-global scope)
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L1057
class FreeVar final : public AbstractRHS {
   public:
    int64_t index;
    std::string name;
    PyObject* value;
    std::shared_ptr<Loc> loc;

    explicit FreeVar() : AbstractRHS(RHSType::FREEVAR) {}
    explicit FreeVar(int64_t _index, std::string _name, void* value,
                     std::shared_ptr<Loc> _loc)
        : AbstractRHS(RHSType::FREEVAR),
          index(_index),
          name(_name),
          value((PyObject*)value),
          loc(_loc) {}

    // Copy Constructor
    FreeVar(const FreeVar& other)
        : AbstractRHS(RHSType::FREEVAR),
          index(other.index),
          name(other.name),
          value(other.value),
          loc(other.loc) {}

    bool operator==(const FreeVar& other) const {
        bool is_equal = PyObject_RichCompareBool(value, other.value, Py_EQ);
        return index == other.index && name == other.name && is_equal;
    }
    // TODO: Missing operator <=> method

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override;
    // TODO: Implement infer_constant
};

/// Base class for all IR instructions
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L307
class Inst {
   public:
    // Bodo change: we explicitly don't extend from AbstractRHS
    explicit Inst() = default;

    // Virtual Methods
    virtual void accept(InstVisitor& visitor) const = 0;
    virtual std::string ToString() const = 0;
    virtual std::vector<std::shared_ptr<Var>> list_vars() = 0;
};

/// Base class for IR statements (instructions which can appear on their own in
/// a Block).
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L340
class Stmt : public Inst {
   public:
    /// All potential types of statements currently implemented in C++
    enum StmtType {
        ASSIGN,
        PRINT,
        SETITEM,
        STATIC_SETITEM,
        DELITEM,
        SETATTR,
        DELATTR,
        STOREMAP,
        TRY_RAISE,
        STATIC_TRY_RAISE,
        DYNAMIC_TRY_RAISE,
        DEL,
        YIELD,
        ENTER_WITH,
        POP_BLOCK,
        PARFOR,
        TERMINATOR,
    };

    explicit Stmt(StmtType op) : OP(op) {}
    StmtType get_stmt_op() const { return OP; }

    // Virtual Methods
    virtual bool is_terminator() const { return false; }
    virtual bool is_exit() const { return false; }

   private:
    const StmtType OP;
};

/// IR statements that are terminators: the last statement in a block.
/// A terminator must either:
/// - exit the function
/// - jump to a block
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L355
class Terminator : public Stmt {
   public:
    enum TerminatorType {
        RAISE,
        STATIC_RAISE,
        DYNAMIC_RAISE,
        RETURN,
        JUMP,
        BRANCH,
    };

    std::shared_ptr<Loc> loc;

    explicit Terminator(TerminatorType op)
        : Stmt(StmtType::TERMINATOR), OP(op) {}
    explicit Terminator(TerminatorType op, std::shared_ptr<Loc> _loc)
        : Stmt(StmtType::TERMINATOR), OP(op), loc(_loc) {}

    TerminatorType get_terminator_op() const { return OP; }

    // Virtual Implementations
    bool is_terminator() const override final { return true; }
    bool is_exit() const override { return false; }

    // Virtual Methods
    virtual std::vector<int64_t> get_targets() = 0;

   private:
    const TerminatorType OP;
};

/// Throw an exception.
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L726
class Raise final : public Terminator {
   public:
    const bool IS_EXIT = true;
    std::optional<std::shared_ptr<Var>> exception;

    explicit Raise() : Terminator(TerminatorType::RAISE) {}
    explicit Raise(std::optional<std::shared_ptr<Var>> _exception,
                   std::shared_ptr<Loc> _loc)
        : Terminator(TerminatorType::RAISE, _loc), exception(_exception) {}

    void accept(InstVisitor& visitor) const override;
    std::string ToString() const override {
        if (!exception.has_value()) {
            return "raise None";
        }
        return fmt::format("raise {}", exception.value());
    }
    std::vector<int64_t> get_targets() override { return {}; }
    std::vector<std::shared_ptr<Var>> list_vars() override {
        using out = std::vector<std::shared_ptr<Var>>;
        return exception.has_value() ? out{exception.value()} : out{};
    }
};

/// Raise an exception class and arguments known at compile-time.
/// Note that if *exc_class* is None, a bare "raise" statement is implied
/// (i.e. re-raise the current exception).
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L742
class StaticRaise final : public Terminator {
   public:
    const bool IS_EXIT = true;
    std::optional<PyObject*> exc_class;
    // TODO: Allow for non-string arguments
    // When implemented, remove check from native_typer.pyx
    std::optional<std::vector<std::string>> exc_args;

    explicit StaticRaise() : Terminator(TerminatorType::STATIC_RAISE) {}
    explicit StaticRaise(void* _exc_class,
                         std::optional<std::vector<std::string>> _exc_args,
                         std::shared_ptr<Loc> _loc)
        : Terminator(TerminatorType::STATIC_RAISE, _loc),
          exc_class((PyObject*)_exc_class),
          exc_args(_exc_args) {}

    void accept(InstVisitor& visitor) const override;
    std::string ToString() const override;
    std::vector<int64_t> get_targets() override { return {}; }
    std::vector<std::shared_ptr<Var>> list_vars() override { return {}; }
};

// TODO: Implement DynamicRaise
class DynamicRaise : public Terminator {
   public:
    explicit DynamicRaise() : Terminator(TerminatorType::DYNAMIC_RAISE) {}
};

/// Return to caller.
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L858
class Return final : public Terminator {
   public:
    std::shared_ptr<Var> value;

    explicit Return() : Terminator(TerminatorType::RETURN) {}
    explicit Return(std::shared_ptr<Var> _value, std::shared_ptr<Loc> _loc)
        : Terminator(TerminatorType::RETURN, _loc), value(_value) {}

    std::string ToString() const override {
        return fmt::format("return {}", this->value);
    }
    void accept(InstVisitor& visitor) const override;
    bool is_exit() const override final { return true; }
    std::vector<int64_t> get_targets() override { return {}; }
    std::vector<std::shared_ptr<Var>> list_vars() override { return {value}; }
};

/// Unconditional branch.
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L877
class Jump final : public Terminator {
   public:
    int64_t target;

    explicit Jump() : Terminator(TerminatorType::JUMP) {}
    explicit Jump(int64_t _target, std::shared_ptr<Loc> _loc)
        : Terminator(TerminatorType::JUMP, _loc), target(_target) {}

    std::string ToString() const override {
        return fmt::format("jump {}", target);
    }
    void accept(InstVisitor& visitor) const override;
    std::vector<int64_t> get_targets() override { return {target}; }
    std::vector<std::shared_ptr<Var>> list_vars() override { return {}; }
};

/// Conditional branch.
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L894
class Branch final : public Terminator {
   public:
    std::shared_ptr<Var> cond;
    int64_t truebr;
    int64_t falsebr;

    explicit Branch() : Terminator(TerminatorType::BRANCH) {}
    explicit Branch(std::shared_ptr<Var> _cond, int64_t _truebr,
                    int64_t _falsebr, std::shared_ptr<Loc> _loc)
        : Terminator(TerminatorType::BRANCH, _loc),
          cond(_cond),
          truebr(_truebr),
          falsebr(_falsebr) {}

    void accept(InstVisitor& visitor) const override;
    std::string ToString() const override {
        return fmt::format("branch {}, {}, {}", cond, truebr, falsebr);
    }
    std::vector<std::shared_ptr<Var>> list_vars() override { return {cond}; }
    std::vector<int64_t> get_targets() override { return {truebr, falsebr}; }
};

/// Statements like `target[index] = value`
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L615
class SetItem final : public Stmt {
   public:
    // Target value that will be modified
    std::shared_ptr<Var> target;
    // Index / attribute in the target that will be set
    std::shared_ptr<Var> index;
    // New value to set at the index / attribute
    std::shared_ptr<Var> value;
    std::shared_ptr<Loc> loc;

    explicit SetItem() : Stmt(StmtType::SETITEM) {}
    explicit SetItem(std::shared_ptr<Var> _target, std::shared_ptr<Var> _index,
                     std::shared_ptr<Var> _value, std::shared_ptr<Loc> _loc)
        : Stmt(StmtType::SETITEM),
          target(_target),
          index(_index),
          value(_value),
          loc(_loc) {}

    void accept(InstVisitor& visitor) const override;
    std::string ToString() const override {
        return fmt::format("{}[{}] = {}", target, index, value);
    }
    std::vector<std::shared_ptr<Var>> list_vars() override {
        return {target, index, value};
    }
};

// Helper types for Static___Item operations
// TODO: Extend to support slices with any constant arguments
// Strings, bools, technically anything can be allowed
using Slice = std::tuple<std::optional<int64_t>, std::optional<int64_t>,
                         std::optional<int64_t>>;
using StaticItemIndex = std::variant<int64_t, Slice>;

/// Statements like `target[constant index] = value`
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L634
class StaticSetItem final : public Stmt {
   public:
    std::shared_ptr<Var> target;
    StaticItemIndex index;
    std::shared_ptr<Var> index_var;
    std::shared_ptr<Var> value;
    std::shared_ptr<Loc> loc;

    explicit StaticSetItem() : Stmt(StmtType::STATIC_SETITEM) {}
    explicit StaticSetItem(std::shared_ptr<Var> target, int64_t index,
                           std::shared_ptr<Var> index_var,
                           std::shared_ptr<Var> value, std::shared_ptr<Loc> loc)
        : Stmt(StmtType::STATIC_SETITEM),
          target(target),
          index(index),
          index_var(index_var),
          value(value),
          loc(loc) {}

    void accept(InstVisitor& visitor) const override;
    std::string ToString() const override;
    std::vector<std::shared_ptr<Var>> list_vars() override {
        return {target, index_var, value};
    }
};

// TODO: Implement DelItem
class DelItem : public Stmt {
   public:
    explicit DelItem() : Stmt(StmtType::DELITEM) {}
};

/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L672
class SetAttr final : public Stmt {
   public:
    std::shared_ptr<Var> target;
    std::string attr;
    std::shared_ptr<Var> value;
    std::shared_ptr<Loc> loc;

    explicit SetAttr() : Stmt(StmtType::SETATTR) {}
    explicit SetAttr(std::shared_ptr<Var> target, std::string attr,
                     std::shared_ptr<Var> value, std::shared_ptr<Loc> loc)
        : Stmt(StmtType::SETATTR),
          target(target),
          attr(attr),
          value(value),
          loc(loc) {}

    void accept(InstVisitor& visitor) const override;
    std::string ToString() const override {
        return fmt::format("({}).{} = {}", target, attr, value);
    }
    std::vector<std::shared_ptr<Var>> list_vars() override {
        return {target, value};
    }
};

// TODO: Implement DelAttr
class DelAttr : public Stmt {
   public:
    explicit DelAttr() : Stmt(StmtType::DELATTR) {}
};
// TODO: Implement StoreMap
class StoreMap : public Stmt {
   public:
    explicit StoreMap() : Stmt(StmtType::STOREMAP) {}
};

/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L715
class Del final : public Stmt {
   public:
    std::string value;
    std::shared_ptr<Loc> loc;

    explicit Del() : Stmt(StmtType::DEL) {}
    explicit Del(std::string value, std::shared_ptr<Loc> loc)
        : Stmt(StmtType::DEL), value(value), loc(loc) {}

    void accept(InstVisitor& visitor) const override;
    std::string ToString() const override {
        return fmt::format("del {}", value);
    }
    std::vector<std::shared_ptr<Var>> list_vars() override { return {}; }
};

// TODO: Implement TryRaise and related classes
class TryRaise : public Stmt {
   public:
    explicit TryRaise() : Stmt(StmtType::TRY_RAISE) {}
};
class StaticTryRaise : public Stmt {
   public:
    explicit StaticTryRaise() : Stmt(StmtType::STATIC_TRY_RAISE) {}
};
class DynamicTryRaise : public Stmt {
   public:
    explicit DynamicTryRaise() : Stmt(StmtType::DYNAMIC_TRY_RAISE) {}
};

/// Assign to a variable.
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L914
class Assign final : public Stmt {
   public:
    std::shared_ptr<Var> target;
    std::shared_ptr<AbstractRHS> value;
    std::shared_ptr<Loc> loc;

    explicit Assign() : Stmt(StmtType::ASSIGN) {}
    explicit Assign(std::shared_ptr<Var> _target,
                    std::shared_ptr<AbstractRHS> _value,
                    std::shared_ptr<Loc> _loc)
        : Stmt(StmtType::ASSIGN), target(_target), value(_value), loc(_loc) {}

    void accept(InstVisitor& visitor) const override;
    std::string ToString() const override {
        return fmt::format("{} = {}", this->target, this->value->ToString());
    }
    std::vector<std::shared_ptr<Var>> list_vars() override;
};

/// Print some values
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L930
class Print final : public Stmt {
   public:
    std::vector<std::shared_ptr<Var>> args;
    std::optional<std::shared_ptr<Var>> vararg;
    std::shared_ptr<Loc> loc;
    // TODO: std::unordered_map<?, ?> consts;

    explicit Print() : Stmt(StmtType::PRINT) {}
    explicit Print(std::vector<std::shared_ptr<Var>> _args,
                   std::optional<std::shared_ptr<Var>> _vararg,
                   std::shared_ptr<Loc> _loc)
        : Stmt(StmtType::PRINT), args(_args), vararg(_vararg), loc(_loc) {}

    void accept(InstVisitor& visitor) const override;
    std::string ToString() const override {
        return fmt::format("print({})", fmt::join(args, ", "));
    }
    std::vector<std::shared_ptr<Var>> list_vars() override {
        std::vector<std::shared_ptr<Var>> res = args;
        if (vararg.has_value()) {
            res.push_back(vararg.value());
        }
        return res;
    }
};

// TODO: Implement Yield
class Yield : public Stmt {
   public:
    explicit Yield() : Stmt(StmtType::YIELD) {}
};

/// Enter a "with" context
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L963
class EnterWith final : public Stmt {
   public:
    std::shared_ptr<Var> contextmanager;
    int64_t begin;
    int64_t end;
    std::shared_ptr<Loc> loc;

    explicit EnterWith() : Stmt(StmtType::ENTER_WITH) {}
    explicit EnterWith(std::shared_ptr<Var> _contextmanager, int64_t _begin,
                       int64_t _end, std::shared_ptr<Loc> _loc)
        : Stmt(StmtType::ENTER_WITH),
          contextmanager(_contextmanager),
          begin(_begin),
          end(_end),
          loc(_loc) {}

    void accept(InstVisitor& visitor) const override;
    std::string ToString() const override {
        return fmt::format("enter_with {}", contextmanager);
    }
    std::vector<std::shared_ptr<Var>> list_vars() override {
        return {contextmanager};
    }
};

/// Marker statement for a pop block op code
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L990
class PopBlock final : public Stmt {
   public:
    std::shared_ptr<Loc> loc;

    explicit PopBlock() : Stmt(StmtType::POP_BLOCK) {}
    explicit PopBlock(std::shared_ptr<Loc> _loc)
        : Stmt(StmtType::POP_BLOCK), loc(_loc) {}

    void accept(InstVisitor& visitor) const override;
    std::string ToString() const override { return std::string("pop_block"); }
    std::vector<std::shared_ptr<Var>> list_vars() override { return {}; }
};

// TODO: Implement Parfor
class Parfor : public Stmt {
   public:
    explicit Parfor() : Stmt(StmtType::PARFOR) {}
};

/// An IR expression (an instruction which can only be part of a larger
/// statement).
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L371
class Expr : public AbstractRHS {
   public:
    std::shared_ptr<Loc> loc;

    /// All potential types of expressions currently implemented in C++
    enum ExprType {
        BINOP,
        INPLACE_BINOP,
        UNARY,
        CALL,
        BUILD_TUPLE,
        BUILD_LIST,
        BUILD_SET,
        BUILD_MAP,
        PAIR_FIRST,
        PAIR_SECOND,
        GETITER,
        ITERNEXT,
        EXHAUST_ITER,
        GETATTR,
        GETITEM,
        TYPED_GETITEM,
        STATIC_GETITEM,
        CAST,
        PHI,
        MAKE_FUNCTION,
        NULL_OP,
        UNDEF,
        // TODO: Is dummy necessary
        DUMMY,
    };

    explicit Expr(const ExprType op) : AbstractRHS(RHSType::EXPR), OP(op) {}
    explicit Expr(const ExprType op, std::shared_ptr<Loc> _loc)
        : AbstractRHS(RHSType::EXPR), OP(op), loc(_loc) {}

    ExprType get_expr_op() const { return OP; }
    bool operator==(const Expr&) const {
        // TODO[BSE-3921]: replace dummy
        return true;
    }

    virtual PyObject* to_py() {
        throw std::runtime_error("Expr doesn't support to_py() yet");
    }

    // Virtual Methods
    virtual std::vector<std::shared_ptr<Var>> list_vars() = 0;

   private:
    const ExprType OP;
};

/// Binary Expression (e.g. a + b)
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L396
class BinOpExpr final : public Expr {
   public:
    std::shared_ptr<Var> lhs;
    std::shared_ptr<Var> rhs;
    // A PyObject* to a built-in Python function
    PyObject* fn;
    // Bodo Added: String representation of the operator
    // from numba.core.utils.OPERATORS_TO_BUILTINS
    std::optional<std::string> op_str;

    explicit BinOpExpr() : Expr(Expr::ExprType::BINOP) {}
    // void* is used for `fn` for Cython compatibility
    explicit BinOpExpr(std::shared_ptr<Var> _lhs, std::shared_ptr<Var> _rhs,
                       void* fn, std::shared_ptr<Loc> _loc,
                       std::optional<std::string> _op_str)
        : Expr(Expr::ExprType::BINOP, _loc),
          lhs(_lhs),
          rhs(_rhs),
          fn((PyObject*)fn),
          op_str(_op_str) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override;
    std::vector<std::shared_ptr<Var>> list_vars() override {
        return {lhs, rhs};
    }
};

/// In-Place Binary Expression (e.g. a += b)
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L406
class InPlaceBinOpExpr final : public Expr {
   public:
    std::shared_ptr<Var> lhs;
    std::shared_ptr<Var> rhs;
    // A PyObject* to a built-in Python function
    PyObject* fn;
    // TODO: What is the purpose of immutable_fn?
    PyObject* immutable_fn;

    explicit InPlaceBinOpExpr() : Expr(Expr::ExprType::INPLACE_BINOP) {}
    // void* is used for `fn` and `immutable_fn` for Cython compatibility
    explicit InPlaceBinOpExpr(std::shared_ptr<Var> _lhs,
                              std::shared_ptr<Var> _rhs, void* fn,
                              void* immutable_fn, std::shared_ptr<Loc> _loc)
        : Expr(Expr::ExprType::INPLACE_BINOP, _loc),
          lhs(_lhs),
          rhs(_rhs),
          fn((PyObject*)fn),
          immutable_fn((PyObject*)immutable_fn) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override;
    std::vector<std::shared_ptr<Var>> list_vars() override {
        return {lhs, rhs};
    }
};

/// Unary Expression (e.g. -a)
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L418
class UnaryExpr final : public Expr {
   public:
    // A PyObject* to a built-in Python function
    PyObject* fn;
    // Input variable
    std::shared_ptr<Var> value;

    explicit UnaryExpr() : Expr(Expr::ExprType::UNARY) {}
    explicit UnaryExpr(void* fn, std::shared_ptr<Var> _value,
                       std::shared_ptr<Loc> _loc)
        : Expr(Expr::ExprType::UNARY, _loc), value(_value), fn((PyObject*)fn) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override;
    std::vector<std::shared_ptr<Var>> list_vars() override { return {value}; }
};

class CallExpr final : public Expr {
   public:
    std::shared_ptr<Var> func;
    std::vector<std::shared_ptr<Var>> args;
    std::vector<std::pair<std::string, std::shared_ptr<Var>>> kws;
    std::optional<std::shared_ptr<Var>> vararg;
    std::optional<std::shared_ptr<Var>> varkwarg;
    std::shared_ptr<Loc> loc;
    // py_expr is a borrowed reference (no incref/decref needed).
    // It comes from the original IR during unboxing, which stays
    // alive during native type inference.
    PyObject* py_expr;

    explicit CallExpr() : Expr(ExprType::CALL) {}
    explicit CallExpr(
        std::shared_ptr<Var> func, std::vector<std::shared_ptr<Var>> args,
        std::vector<std::pair<std::string, std::shared_ptr<Var>>> kws,
        std::optional<std::shared_ptr<Var>> vararg,
        std::optional<std::shared_ptr<Var>> varkwarg, std::shared_ptr<Loc> loc,
        void* py_expr)
        : Expr(ExprType::CALL, loc),
          func(func),
          args(args),
          kws(kws),
          vararg(vararg),
          varkwarg(varkwarg),
          py_expr((PyObject*)py_expr) {}

    PyObject* to_py() override {
        // Return a new reference (need to give ownership to caller per Python
        // convention)
        Py_INCREF(py_expr);
        return py_expr;
    }

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override;
    std::vector<std::shared_ptr<Var>> list_vars() override;
};

/// Construct a tuple expression (e.g. (a, b, c))
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L434
class BuildTupleExpr final : public Expr {
   public:
    // The items in the new tuple
    std::vector<std::shared_ptr<Var>> items;

    explicit BuildTupleExpr() : Expr(Expr::ExprType::BUILD_TUPLE) {}
    explicit BuildTupleExpr(std::vector<std::shared_ptr<Var>> items,
                            std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::BUILD_TUPLE, loc), items(items) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override {
        // TODO: Segregate between Numba and intuitive formatting
        return fmt::format(
            "build_tuple(items=[{}])",
            fmt::join(items | std::views::transform(
                                  [](const auto& v) { return v->ToRepr(); }),
                      ", "));
    }
    std::vector<std::shared_ptr<Var>> list_vars() override { return items; }
};

/// Construct a list expression (e.g. [a, b, c])
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L440
class BuildListExpr final : public Expr {
   public:
    // The items in the new list
    std::vector<std::shared_ptr<Var>> items;

    explicit BuildListExpr() : Expr(Expr::ExprType::BUILD_LIST) {}
    explicit BuildListExpr(std::vector<std::shared_ptr<Var>> items,
                           std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::BUILD_LIST, loc), items(items) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override {
        // TODO: Segregate between Numba and intuitive formatting
        return fmt::format(
            "build_list(items=[{}])",
            fmt::join(items | std::views::transform(
                                  [](const auto& v) { return v->ToRepr(); }),
                      ", "));
    }
    std::vector<std::shared_ptr<Var>> list_vars() override { return items; }
};

// TODO: Implement BuildSetExpr and BuildMapExpr
class BuildSetExpr : public Expr {};
class BuildMapExpr : public Expr {};

/// Get the first element of a pair (e.g. first(a, b))
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L459
class PairFirstExpr final : public Expr {
   public:
    // Variable containing the pair to extract first value from
    std::shared_ptr<Var> value;

    explicit PairFirstExpr() : Expr(Expr::ExprType::PAIR_FIRST) {}
    explicit PairFirstExpr(std::shared_ptr<Var> value, std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::PAIR_FIRST, loc), value(value) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override {
        return fmt::format("first(value={})", value);
    }
    std::vector<std::shared_ptr<Var>> list_vars() override { return {value}; }
};

/// Get the second element of a pair (e.g. second(a, b))
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L465
class PairSecondExpr : public Expr {
   public:
    // Variable containing the pair to extract second value from
    std::shared_ptr<Var> value;

    explicit PairSecondExpr() : Expr(Expr::ExprType::PAIR_SECOND) {}
    explicit PairSecondExpr(std::shared_ptr<Var> value,
                            std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::PAIR_SECOND, loc), value(value) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override {
        return fmt::format("second(value={})", value);
    }
    std::vector<std::shared_ptr<Var>> list_vars() override { return {value}; }
};

/// Get an iterator from an object (e.g. iter(a))
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L472
class GetIterExpr : public Expr {
   public:
    // The object to get an iterator from
    std::shared_ptr<Var> value;

    explicit GetIterExpr() : Expr(Expr::ExprType::GETITER) {}
    explicit GetIterExpr(std::shared_ptr<Var> value, std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::GETITER, loc), value(value) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override {
        return fmt::format("iter(value={})", value);
    }
    std::vector<std::shared_ptr<Var>> list_vars() override { return {value}; }
};

/// Get the next element from an iterator (e.g. next(a))
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L479
class IterNextExpr : public Expr {
   public:
    std::shared_ptr<Var> value;

    explicit IterNextExpr() : Expr(Expr::ExprType::ITERNEXT) {}
    explicit IterNextExpr(std::shared_ptr<Var> value, std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::ITERNEXT, loc), value(value) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override {
        return fmt::format("next({})", value);
    }
    std::vector<std::shared_ptr<Var>> list_vars() override { return {value}; }
};

/// Exhaust an iterator (e.g. a, b, c = iter)
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L486
class ExhaustIterExpr final : public Expr {
   public:
    std::shared_ptr<Var> value;
    int64_t count;
    std::shared_ptr<Loc> loc;

    explicit ExhaustIterExpr() : Expr(Expr::ExprType::EXHAUST_ITER) {}
    explicit ExhaustIterExpr(std::shared_ptr<Var> value, int64_t count,
                             std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::EXHAUST_ITER, loc), value(value), count(count) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override {
        return fmt::format("exhaust_iter({}, {})", value, count);
    }
    std::vector<std::shared_ptr<Var>> list_vars() override { return {value}; }
};

/// Get an attribute from an object (e.g. a.b)
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L494
class GetAttrExpr final : public Expr {
   public:
    std::shared_ptr<Var> value;
    std::string attr;

    explicit GetAttrExpr() : Expr(Expr::ExprType::GETATTR) {}
    explicit GetAttrExpr(std::shared_ptr<Var> value, std::string attr,
                         std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::GETATTR, loc), value(value), attr(attr) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override {
        return fmt::format("getattr(value={}, attr={})", value, attr);
    }
    std::vector<std::shared_ptr<Var>> list_vars() override { return {value}; }
};

/// Get an item from an object (e.g. a[b])
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L502
class GetItemExpr final : public Expr {
   public:
    std::shared_ptr<Var> value;
    std::shared_ptr<Var> index;

    explicit GetItemExpr() : Expr(Expr::ExprType::GETITEM) {}
    explicit GetItemExpr(std::shared_ptr<Var> value, std::shared_ptr<Var> index,
                         std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::GETITEM, loc), value(value), index(index) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override;
    std::vector<std::shared_ptr<Var>> list_vars() override {
        return {value, index};
    }
};

// TODO: Implement TypedGetItemExpr
class TypedGetItemExpr : public Expr {};

/// Get an item from an object with a fixed / static index (e.g. a[1])
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L519
class StaticGetItemExpr final : public Expr {
   public:
    std::shared_ptr<Var> value;
    // TODO: Extend index types, see TODO in StaticItemIndex
    StaticItemIndex index;
    std::optional<std::shared_ptr<Var>> index_var;

    explicit StaticGetItemExpr() : Expr(Expr::ExprType::STATIC_GETITEM) {}
    explicit StaticGetItemExpr(std::shared_ptr<Var> value, int64_t index,
                               std::optional<std::shared_ptr<Var>> index_var,
                               std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::STATIC_GETITEM, loc),
          value(value),
          index(index),
          index_var(index_var) {}
    StaticGetItemExpr(std::shared_ptr<Var> value,
                      std::optional<int64_t> start_index,
                      std::optional<int64_t> stop_index,
                      std::optional<int64_t> step_index,
                      std::optional<std::shared_ptr<Var>> index_var,
                      std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::STATIC_GETITEM, loc),
          value(value),
          index(std::make_tuple(start_index, stop_index, step_index)),
          index_var(index_var) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override;
    std::vector<std::shared_ptr<Var>> list_vars() override {
        if (index_var.has_value()) {
            return {value, index_var.value()};
        }
        return {value};
    }
};

class CastExpr final : public Expr {
   public:
    std::shared_ptr<Var> value;
    std::shared_ptr<Loc> loc;

    explicit CastExpr() : Expr(ExprType::CAST) {}
    explicit CastExpr(std::shared_ptr<Var> _value, std::shared_ptr<Loc> _loc)
        : Expr(ExprType::CAST, _loc), value(_value), loc(_loc) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override {
        return fmt::format("cast(value={})", this->value);
    };
    std::vector<std::shared_ptr<Var>> list_vars() override { return {value}; }
};

/// Phi Node
/// Intended to be modified in-place
/// https://github.com/numba/numba/blob/a344e8f55440c91d40c5221e93a38ce0c149b803/numba/core/ir.py#L539
class PhiExpr final : public Expr {
   public:
    // Incoming values from different blocks
    std::vector<std::shared_ptr<Var>> incoming_values;
    std::vector<uint64_t> incoming_blocks;

    explicit PhiExpr(std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::PHI, loc),
          incoming_values(),
          incoming_blocks() {}
    explicit PhiExpr(std::vector<std::shared_ptr<Var>> incoming_values,
                     std::vector<uint64_t> incoming_blocks,
                     std::shared_ptr<Loc> loc)
        : Expr(Expr::ExprType::PHI, loc),
          incoming_values(incoming_values),
          incoming_blocks(incoming_blocks) {}

    void accept(InstVisitor& visitor, const std::string& target_varname,
                std::shared_ptr<Loc> loc) const override;
    std::string ToString() const override {
        return fmt::format("phi(incoming_blocks={}, incoming_values={})",
                           fmt::join(incoming_blocks, ", "),
                           fmt::join(incoming_values, ", "));
    }
    std::vector<std::shared_ptr<Var>> list_vars() override {
        return incoming_values;
    }
};

// TODO: Implement MakeFunctionExpr, NullExpr, UndefExpr, DummyExpr
class MakeFunctionExpr : public Expr {
   public:
    explicit MakeFunctionExpr() : Expr(ExprType::MAKE_FUNCTION) {}
};
class NullExpr : public Expr {
   public:
    explicit NullExpr() : Expr(ExprType::NULL_OP) {}
};
class UndefExpr : public Expr {
   public:
    explicit UndefExpr() : Expr(ExprType::UNDEF) {}
};
class DummyExpr : public Expr {
   public:
    explicit DummyExpr() : Expr(ExprType::DUMMY) {}
};

/// An IR Block (stmts with no branching)
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L1262
class Block final {
   public:
    std::vector<std::shared_ptr<Stmt>> body;
    std::shared_ptr<Loc> loc;

    explicit Block() = default;
    explicit Block(std::shared_ptr<Loc> _loc) : loc(_loc) {}

    bool operator==(const Block& other) const { return body == other.body; }

    // Terminator detection
    // TODO: Restructure Block class to have explicit terminator field?
    bool is_terminated() const {
        return !body.empty() && body.back()->is_terminator();
    }

    std::shared_ptr<Terminator> get_terminator() const {
        if (body.empty()) {
            return nullptr;
        }
        if (body.back()->is_terminator()) {
            return std::static_pointer_cast<Terminator>(body.back());
        }
        return nullptr;
    }

    void verify() const {
        if (!is_terminated()) {
            throw std::runtime_error("Missing block terminator");
        }

        // Only last instruction can be a terminator
        for (size_t i = 0; i < body.size() - 1; i++) {
            if (body[i]->is_terminator()) {
                throw std::runtime_error(
                    "Terminator before the last instruction");
            }
        }
    }

    std::string ToString() const;

    // Missing methods
    // - find_exprs
    // - find_insts
    // - find_variable_assignments
    // - prepend
    // - append
    // - remove
    // - clear
    // - insert_after
    // - insert_before_terminator
    // - __repr__
    // - <= operator
};

// TODO: Implement Loop and With
class Loop;
class With;

/// C++ Function IR Representation
/// https://github.com/numba/numba/blob/d72f8949bcddee7ebe48cd93970b5599daa8c248/numba/core/ir.py#L1394
class FunctionIR final {
   public:
    // Using map to keep sorted by labels (needed in some algorithms and
    // printing)
    std::map<int64_t, std::shared_ptr<Block>> blocks;

    explicit FunctionIR() = default;
    explicit FunctionIR(std::map<int64_t, std::shared_ptr<Block>> _blocks)
        : blocks(_blocks) {}

    std::string ToString() const {
        std::ostringstream out;
        for (const auto& [label, block] : this->blocks) {
            out << fmt::format("label {}:\n", label);
            out << block->ToString();
        }
        return out.str();
    }

    // Missing methods
    // - equal_ir
    // - diff_str
    // - _reset_analysis_variables
    // - derive
    // - copy
    // - get_block_entry_vars
    // - infer_constant
    // - get_definition
    // - get_assignee
    // - dump
    // - dump_generator_info
    // - render_dot
};
};  // namespace bodo

template <>
struct std::hash<bodo::Var> {
    std::size_t operator()(const bodo::Var& s) const noexcept {
        // TODO[BSE-3921]: proper hash function for Var
        return std::hash<std::string>()(s.ToString());
    }
};

template <>
struct std::hash<bodo::Expr> {
    std::size_t operator()(const bodo::Expr& s) const noexcept {
        // TODO[BSE-3921]: proper hash function
        return std::hash<std::string>()(s.ToString());
    }
};
