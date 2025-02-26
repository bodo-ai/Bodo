#include "ir.h"

#include <algorithm>

#include <longobject.h>
#include "ir_utils.h"

/// Helper function to get the string representation of a Python object
std::string pyobject_as_str(PyObject* pyo) {
    PyObject* pystr = PyObject_Str(pyo);
    if (pystr == nullptr) {
        throw std::runtime_error(
            "Failed to get string representation for PyObject");
    }

    Py_ssize_t size;
    const char* cstr = PyUnicode_AsUTF8AndSize(pystr, &size);
    if (cstr == nullptr) {
        throw std::runtime_error("Failed to convert PyObject to std::string");
    }

    auto out = std::string(cstr, size);
    Py_DECREF(pystr);
    return out;
}

namespace bodo {

// ------------------------------ AbstractRHS ------------------------------ //
void Var::accept(InstVisitor& visitor, const std::string& target_varname,
                 std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

void Arg::accept(InstVisitor& visitor, const std::string& target_varname,
                 std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

Type* Const::getType() const {
    // TODO[BSE-3921]: handle non-int64 cases properly
    if (!PyLong_Check(value)) {
        throw std::runtime_error(
            "Only int constants are supported in type inference");
    }

    if (this->use_literal_type) {
        return IntegerLiteral::get(PyLong_AsLongLong(value));
    }
    return Integer::get(64, true);
}

void Const::accept(InstVisitor& visitor, const std::string& target_varname,
                   std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

std::string Const::ToString() const {
    PyObject* py_type = PyObject_Type(value);
    if (py_type == nullptr) {
        throw std::runtime_error(
            "Const::ToString: Failed to get type of PyObject");
    }

    PyObject* type_name = PyObject_GetAttrString(py_type, "__name__");
    if (type_name == nullptr) {
        throw std::runtime_error(
            "Const::ToString: Failed to get __name__ of PyObject");
    }
    Py_DECREF(py_type);
    std::string type_str = pyobject_as_str(type_name);
    Py_DECREF(type_name);

    return fmt::format("const({}, {})", type_str, pyobject_as_str(value));
}

void Global::accept(InstVisitor& visitor, const std::string& target_varname,
                    std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

std::string Global::ToString() const {
    return fmt::format("global({}: {})", name, pyobject_as_str(value));
}

void FreeVar::accept(InstVisitor& visitor, const std::string& target_varname,
                     std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

std::string FreeVar::ToString() const {
    return fmt::format("freevar({}: {})", name, pyobject_as_str(value));
}

// ------------------------------- Terminator ------------------------------ //
void Raise::accept(InstVisitor& visitor) const { visitor.visit(this); }

void StaticRaise::accept(InstVisitor& visitor) const { visitor.visit(this); }

std::string StaticRaise::ToString() const {
    if (!exc_class.has_value()) {
        return "<static> raise";
    }

    std::string cls_str = pyobject_as_str(exc_class.value());
    if (!exc_args.has_value()) {
        return fmt::format("<static> raise {}", cls_str);
    } else {
        return fmt::format("<static> raise {}({})", cls_str,
                           fmt::join(exc_args.value(), ", "));
    }
}

void Return::accept(InstVisitor& visitor) const { visitor.visit(this); }

void Jump::accept(InstVisitor& visitor) const { visitor.visit(this); }

void Branch::accept(InstVisitor& visitor) const { visitor.visit(this); }

// ---------------------------------- Stmt --------------------------------- //
void SetItem::accept(InstVisitor& visitor) const { visitor.visit(this); }

void StaticSetItem::accept(InstVisitor& visitor) const { visitor.visit(this); }

std::string StaticSetItem::ToString() const {
    if (index.index() == 0) {
        return fmt::format("{}[{}] = {}", value, target,
                           std::get<int64_t>(index), value);
    }

    auto slice = std::get<Slice>(index);
    auto index_str = fmt::format("({}, {}, {})", std::get<0>(slice).value_or(0),
                                 std::get<1>(slice).value_or(0),
                                 std::get<2>(slice).value_or(0));

    return fmt::format("{}[{}] = {}", target, index_str, value);
}

void SetAttr::accept(InstVisitor& visitor) const { visitor.visit(this); }

void Del::accept(InstVisitor& visitor) const { visitor.visit(this); }

void Assign::accept(InstVisitor& visitor) const {
    this->value->accept(visitor, target->name, this->loc);
    visitor.visit(this);
}

std::vector<std::shared_ptr<Var>> Assign::list_vars() {
    std::vector<std::shared_ptr<Var>> res = {target};
    if (value->get_rhs_op() == AbstractRHS::RHSType::EXPR) {
        auto inst = std::static_pointer_cast<Expr>(value);
        auto inst_vars = inst->list_vars();
        res.insert(res.end(), inst_vars.begin(), inst_vars.end());
    } else if (value->get_rhs_op() == AbstractRHS::RHSType::VAR) {
        res.push_back(std::static_pointer_cast<Var>(value));
    }

    return res;
}

void Print::accept(InstVisitor& visitor) const { visitor.visit(this); }

void EnterWith::accept(InstVisitor& visitor) const { visitor.visit(this); }

void PopBlock::accept(InstVisitor& visitor) const { visitor.visit(this); }

// ---------------------------------- Expr --------------------------------- //
void BinOpExpr::accept(InstVisitor& visitor, const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

std::string BinOpExpr::ToString() const {
    std::string op_str =
        this->op_str.has_value() ? this->op_str.value() : pyobject_as_str(fn);
    return fmt::format("{} {} {}", lhs, op_str, rhs);
}

void InPlaceBinOpExpr::accept(InstVisitor& visitor,
                              const std::string& target_varname,
                              std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

std::string InPlaceBinOpExpr::ToString() const {
    return fmt::format(
        "inplace_binop(fn={}, immutable_fn={}, lhs={}, rhs={}, "
        "static_lhs=Undefined, static_rhs=Undefined)",
        pyobject_as_str(fn), pyobject_as_str(immutable_fn), lhs, rhs);
}

void UnaryExpr::accept(InstVisitor& visitor, const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

std::string UnaryExpr::ToString() const {
    return fmt::format("unary(fn={}, value={})", pyobject_as_str(fn), value);
}

void CallExpr::accept(InstVisitor& visitor, const std::string& target_varname,
                      std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

std::string CallExpr::ToString() const {
    std::vector<std::string> contents;

    if (args.size() != 0) {
        contents.push_back(fmt::format("{}", fmt::join(args, ", ")));
    }
    if (vararg.has_value()) {
        contents.push_back("*" + vararg.value()->ToString());
    }
    if (kws.size() != 0) {
        contents.push_back(fmt::format(
            "{}", fmt::join(kws | std::views::transform([](const auto& pair) {
                                return fmt::format("{}={}", pair.first,
                                                   pair.second);
                            }),
                            ", ")));
    }
    if (varkwarg.has_value()) {
        contents.push_back("**" + varkwarg.value()->ToString());
    }

    contents.push_back(fmt::format("func={}", func));

    contents.push_back(
        fmt::format("args=[{}]",
                    fmt::join(args | std::views::transform([](const auto& arg) {
                                  return arg->ToRepr();
                              }),
                              ", ")));

    contents.push_back(fmt::format(
        "kws=({})", fmt::join(kws | std::views::transform([](const auto& pair) {
                                  return fmt::format("({}, {})", pair.first,
                                                     pair.second);
                              }),
                              ", ")));
    contents.push_back(fmt::format(
        "vararg={}", vararg.has_value() ? vararg.value()->ToString() : "None"));
    contents.push_back(fmt::format(
        "varkwarg={}",
        varkwarg.has_value() ? varkwarg.value()->ToString() : "None"));
    contents.push_back("target=None");

    return fmt::format("call {}({})", func, fmt::join(contents, ", "));
}

std::vector<std::shared_ptr<Var>> CallExpr::list_vars() {
    std::vector<std::shared_ptr<Var>> res = {func};
    res.insert(res.end(), args.begin(), args.end());
    for (const auto& [_, kw] : kws) {
        res.push_back(kw);
    }
    if (vararg.has_value()) {
        res.push_back(vararg.value());
    }
    if (varkwarg.has_value()) {
        res.push_back(varkwarg.value());
    }
    return res;
}

void BuildTupleExpr::accept(InstVisitor& visitor,
                            const std::string& target_varname,
                            std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

void BuildListExpr::accept(InstVisitor& visitor,
                           const std::string& target_varname,
                           std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

void PairFirstExpr::accept(InstVisitor& visitor,
                           const std::string& target_varname,
                           std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

void PairSecondExpr::accept(InstVisitor& visitor,
                            const std::string& target_varname,
                            std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

void GetIterExpr::accept(InstVisitor& visitor,
                         const std::string& target_varname,
                         std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

void IterNextExpr::accept(InstVisitor& visitor,
                          const std::string& target_varname,
                          std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

void ExhaustIterExpr::accept(InstVisitor& visitor,
                             const std::string& target_varname,
                             std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

void GetAttrExpr::accept(InstVisitor& visitor,
                         const std::string& target_varname,
                         std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

void GetItemExpr::accept(InstVisitor& visitor,
                         const std::string& target_varname,
                         std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

std::string GetItemExpr::ToString() const {
    return fmt::format(
        "getitem(value={}, index={}, fn=<built-in function getitem>)", value,
        index);
}

void StaticGetItemExpr::accept(InstVisitor& visitor,
                               const std::string& target_varname,
                               std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

std::string StaticGetItemExpr::ToString() const {
    std::string index_var_str =
        index_var.has_value() ? index_var.value()->ToString() : "None";
    if (index.index() == 0) {
        return fmt::format(
            "static_getitem(fn='getitem', value={}, index={}, index)", value,
            std::get<int64_t>(index), index_var_str);
    }

    auto slice = std::get<Slice>(index);
    auto index_str = fmt::format("({}, {}, {})", std::get<0>(slice).value_or(0),
                                 std::get<1>(slice).value_or(0),
                                 std::get<2>(slice).value_or(0));
    return fmt::format(
        "static_getitem(fn='getitem', value={}, index={}, index_var={})", value,
        index_str, index_var_str);
}

void CastExpr::accept(InstVisitor& visitor, const std::string& target_varname,
                      std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

void PhiExpr::accept(InstVisitor& visitor, const std::string& target_varname,
                     std::shared_ptr<Loc> loc) const {
    visitor.visit(this, target_varname, loc);
}

// --------------------------------- Block --------------------------------- //
std::string Block::ToString() const {
    fmt::memory_buffer ss;
    for (const auto& stmt : body) {
        auto vars = stmt->list_vars();
        std::vector<std::string> var_str;
        for (const auto& var : vars) {
            var_str.push_back(var->ToString());
        }
        std::sort(var_str.begin(), var_str.end(), std::less<>());
        fmt::format_to(std::back_inserter(ss), "    {:<40} [{}]\n",
                       stmt->ToString(),
                       fmt::join(var_str | std::views::transform([](auto& x) {
                                     return fmt::format("'{}'", x);
                                 }),
                                 ", "));
    }
    return fmt::to_string(ss);
}

};  // namespace bodo
