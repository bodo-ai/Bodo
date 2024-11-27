#include "ir.h"

namespace bodo {
// Similar to LLVM's InstVisitor
// https://github.com/llvm/llvm-project/blob/ea9204505cf1099b98b1fdcb898f0bd35e463984/llvm/include/llvm/IR/InstVisitor.h#L32
class InstVisitor {
   public:
    // Statements
    virtual void visit(const SetItem* setitem) const = 0;
    virtual void visit(const StaticSetItem* setitem) const = 0;
    virtual void visit(const SetAttr* setattr) const = 0;
    virtual void visit(const Del* del) const = 0;
    virtual void visit(const Assign* assign) const = 0;
    virtual void visit(const Print* print) const = 0;
    virtual void visit(const EnterWith* enterwith) const = 0;
    virtual void visit(const PopBlock* popblock) const = 0;

    // Terminators
    virtual void visit(const Raise* raise) const = 0;
    virtual void visit(const StaticRaise* raise) const = 0;
    virtual void visit(const Return* ret) const = 0;
    virtual void visit(const Jump* jump) const = 0;
    virtual void visit(const Branch* branch) const = 0;

    // RHS visits take target variable name and loc of their assignment
    // since needed for later processing
    virtual void visit(const Arg* arg, const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const Var* constant, const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const Const* constant, const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const Global* global, const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const FreeVar* freevar,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;

    // Expressions
    virtual void visit(const BinOpExpr* binop,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const InPlaceBinOpExpr* binop,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const UnaryExpr* unary,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const CallExpr* global,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const BuildTupleExpr* tuple,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const BuildListExpr* list,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const PairFirstExpr* pairfirst,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const PairSecondExpr* pairsecond,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const GetIterExpr* getiter,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const IterNextExpr* iternext,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const ExhaustIterExpr* exhaustiter,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const GetAttrExpr* getattr,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const GetItemExpr* getitem,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const StaticGetItemExpr* staticgetitem,
                       const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const CastExpr* cast, const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
    virtual void visit(const PhiExpr* phi, const std::string& target_varname,
                       std::shared_ptr<Loc> loc) const = 0;
};

}  // namespace bodo
