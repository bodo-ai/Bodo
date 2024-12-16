# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

import os

from numba.core import types, ir, utils
from numba.core.typing.templates import signature
from numba.core.typed_passes import _TypingResults
from numba.core.typeconv.castgraph import Conversion
from numba.parfors.parfor import Parfor

from libc.stdint cimport int64_t, uint64_t
from libcpp cimport bool as c_bool
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map
from libcpp.string cimport string as c_string
from libcpp.pair cimport pair
from libcpp.optional cimport optional, nullopt


cdef extern from "type.h" namespace "bodo" nogil:
    cdef cppclass CType" bodo::Type":
        object to_py()

    cdef cppclass CNoneType" bodo::NoneType"(CType):
        @staticmethod
        CNoneType* get()
        object to_py()

    cdef cppclass CNumber" bodo::Number"(CType):
        pass

    cdef cppclass CInteger" bodo::Integer"(CNumber):
        object to_py()
        @staticmethod
        CInteger* get(int bitwidth, c_bool is_signed)

    cdef cppclass CBoolean" bodo::Boolean"(CType):
        object to_py()
        @staticmethod
        CBoolean* get()

    cdef cppclass CEnumMember" bodo::EnumMember"(CType):
        # passing PyObject* as void* to avoid Cython issues
        @staticmethod
        CEnumMember* get(void* instance_class, CType* dtype)
        object to_py()

    cdef cppclass CFunction" bodo::Function"(CType):
        @staticmethod
        CFunction* get(void* t)
        object to_py()

    cdef cppclass CModule" bodo::Module"(CType):
        @staticmethod
        CModule* get(void* t)
        object to_py()

    cdef cppclass CNumberClass" bodo::NumberClass"(CType):
        @staticmethod
        CNumberClass* get(CType* instance_type)
        object to_py()

    cdef cppclass CSignature" bodo::Signature"(CType):
        CType* return_type
        vector[CType*] args
        CSignature(CType* return_type, vector[CType*] args)
        CSignature()

    cdef cppclass CConversion" bodo::Conversion":
        pass

    cdef cppclass CTypeManager" bodo::TypeManager":
        @staticmethod
        void setCompatibleTypes(CType* t1, CType* t2, int conv)
        @staticmethod
        optional[CConversion] checkCompatibleTypes(CType* t1, CType* t2)

    shared_ptr[CSignature] resolve_call_type(char* func_path, vector[CType*]& args)


cdef extern from "ir.h" namespace "bodo" nogil:
    # TODO[BSE-3921]: support all Numba IR nodes

    cdef cppclass CLoc" bodo::Loc":
        CLoc()
        CLoc(c_string filename, c_string short_fname, uint64_t line, optional[uint64_t] col, c_bool maybe_decorator)

    cdef cppclass CAbstractRHS" bodo::AbstractRHS":
        CAbstractRHS()

    cdef cppclass CInst" bodo::Inst":
        CInst()
    
    cdef cppclass CStmt" bodo::Stmt"(CInst):
        CStmt()

    cdef cppclass CVar" bodo::Var"(CAbstractRHS):
        CVar()
        CVar(c_string name, shared_ptr[CLoc] loc)

    cdef cppclass CArg" bodo::Arg"(CAbstractRHS):
        CArg()
        CArg(int index, c_string name, shared_ptr[CLoc] loc)

    cdef cppclass CConst" bodo::Const"(CAbstractRHS):
        CConst()
        CConst(void* value, shared_ptr[CLoc] loc, c_bool use_literal_type)

    cdef cppclass CGlobal" bodo::Global"(CAbstractRHS):
        CGlobal()
        CGlobal(c_string name, void* value, shared_ptr[CLoc] loc)

    cdef cppclass CFreeVar" bodo::FreeVar"(CAbstractRHS):
        CFreeVar()
        CFreeVar(int64_t index, c_string name, void* value, shared_ptr[CLoc] loc)

    cdef cppclass CRaise" bodo::Raise"(CStmt):
        CRaise(shared_ptr[CVar] exception, shared_ptr[CLoc] loc)

    cdef cppclass CStaticRaise" bodo::StaticRaise"(CStmt):
        CStaticRaise(void* exc_class, optional[vector[c_string]] exc_args, shared_ptr[CLoc] loc)

    cdef cppclass CReturn" bodo::Return"(CStmt):
        CReturn()
        CReturn(shared_ptr[CVar] value, shared_ptr[CLoc] loc)

    cdef cppclass CJump" bodo::Jump"(CStmt):
        CJump()
        CJump(uint64_t target, shared_ptr[CLoc] loc)

    cdef cppclass CBranch" bodo::Branch"(CStmt):
        CBranch()
        CBranch(shared_ptr[CVar] cond, int64_t truebr, int64_t falsebr, shared_ptr[CLoc] loc)

    cdef cppclass CSetItem" bodo::SetItem"(CStmt):
        CSetItem()
        CSetItem(shared_ptr[CVar] target, shared_ptr[CVar] index, shared_ptr[CVar] value, shared_ptr[CLoc] loc)

    cdef cppclass CStaticSetItem" bodo::StaticSetItem"(CStmt):
        CStaticSetItem()
        CStaticSetItem(shared_ptr[CVar] target, int64_t index, shared_ptr[CVar] index_var, shared_ptr[CVar] value, shared_ptr[CLoc] loc)

    cdef cppclass CSetAttr" bodo::SetAttr"(CStmt):
        CSetAttr()
        CSetAttr(shared_ptr[CVar] target, c_string attr, shared_ptr[CVar] value, shared_ptr[CLoc] loc)

    cdef cppclass CPrint" bodo::Print"(CStmt):
        CPrint()
        CPrint(vector[shared_ptr[CVar]] values, optional[shared_ptr[CVar]] vararg, shared_ptr[CLoc] loc)

    cdef cppclass CDel" bodo::Del"(CStmt):
        CDel()
        CDel(c_string value, shared_ptr[CLoc] loc)

    cdef cppclass CEnterWith" bodo::EnterWith"(CStmt):
        CEnterWith(shared_ptr[CVar] contextmanager, int64_t begin, int64_t end, shared_ptr[CLoc] loc)

    cdef cppclass CPopBlock" bodo::PopBlock"(CStmt):
        CPopBlock(shared_ptr[CLoc] loc)

    cdef cppclass CAssign" bodo::Assign"(CStmt):
        CAssign()
        CAssign(shared_ptr[CVar] target, shared_ptr[CAbstractRHS] value, shared_ptr[CLoc] loc)

    cdef cppclass CExpr" bodo::Expr"(CAbstractRHS):
        CExpr()
        object to_py()

    cdef cppclass CBinOpExpr" bodo::BinOpExpr"(CExpr):
        CBinOpExpr(shared_ptr[CVar] lhs, shared_ptr[CVar] rhs, void* op, shared_ptr[CLoc] loc, optional[c_string] op_str)

    cdef cppclass CInPlaceBinOpExpr" bodo::InPlaceBinOpExpr"(CExpr):
        CInPlaceBinOpExpr(shared_ptr[CVar] lhs, shared_ptr[CVar] rhs, void* op, void* immutable_op, shared_ptr[CLoc] loc)

    cdef cppclass CUnaryExpr" bodo::UnaryExpr"(CExpr):
        CUnaryExpr(void* op, shared_ptr[CVar] value, shared_ptr[CLoc] loc)

    cdef cppclass CCallExpr" bodo::CallExpr"(CExpr):
        CCallExpr(shared_ptr[CVar] func, vector[shared_ptr[CVar]] args, vector[pair[c_string, shared_ptr[CVar]]] kws,
            optional[shared_ptr[CVar]] vararg, optional[shared_ptr[CVar]] varkwarg, shared_ptr[CLoc] loc, void* py_expr)

    cdef cppclass CBuildTupleExpr" bodo::BuildTupleExpr"(CExpr):
        CBuildTupleExpr(vector[shared_ptr[CVar]] items, shared_ptr[CLoc] loc)

    cdef cppclass CBuildListExpr" bodo::BuildListExpr"(CExpr):
        CBuildListExpr(vector[shared_ptr[CVar]] items, shared_ptr[CLoc] loc)

    cdef cppclass CPairFirstExpr" bodo::PairFirstExpr"(CExpr):
        CPairFirstExpr(shared_ptr[CVar] value, shared_ptr[CLoc] loc)

    cdef cppclass CPairSecondExpr" bodo::PairSecondExpr"(CExpr):
        CPairSecondExpr(shared_ptr[CVar] value, shared_ptr[CLoc] loc)

    cdef cppclass CGetAttrExpr" bodo::GetAttrExpr"(CExpr):
        CGetAttrExpr(shared_ptr[CVar] value, c_string attr, shared_ptr[CLoc] loc)

    cdef cppclass CGetItemExpr" bodo::GetItemExpr"(CExpr):
        CGetItemExpr(shared_ptr[CVar] value, shared_ptr[CVar] index, shared_ptr[CLoc] loc)

    cdef cppclass CStaticGetItemExpr" bodo::StaticGetItemExpr"(CExpr):
        CStaticGetItemExpr(shared_ptr[CVar] value, int64_t index, optional[shared_ptr[CVar]] index_var, shared_ptr[CLoc] loc)
        CStaticGetItemExpr(shared_ptr[CVar] value, optional[int64_t] start_index, optional[int64_t] stop_index, optional[int64_t] step_index, optional[shared_ptr[CVar]] index_var, shared_ptr[CLoc] loc)

    cdef cppclass CExhaustIterExpr" bodo::ExhaustIterExpr"(CExpr):
        CExhaustIterExpr(shared_ptr[CVar] value, int64_t count, shared_ptr[CLoc] loc)

    cdef cppclass CGetIterExpr" bodo::GetIterExpr"(CExpr):
        CGetIterExpr(shared_ptr[CVar] value, shared_ptr[CLoc] loc)

    cdef cppclass CIterNextExpr" bodo::IterNextExpr"(CExpr):
        CIterNextExpr(shared_ptr[CVar] value, shared_ptr[CLoc] loc)

    cdef cppclass CCast" bodo::CastExpr"(CExpr):
        CCast(shared_ptr[CVar] value, shared_ptr[CLoc] loc)

    cdef cppclass CPhiExpr" bodo::PhiExpr"(CExpr):
        CPhiExpr(vector[shared_ptr[CVar]] incoming_values, vector[uint64_t] incoming_blocks, shared_ptr[CLoc] loc)

    cdef cppclass CBlock" bodo::Block":
        vector[shared_ptr[CStmt]] body
        CBlock()
        CBlock(shared_ptr[CLoc] loc)

    cdef cppclass CFunctionIR" bodo::FunctionIR":
        CFunctionIR()
        CFunctionIR(map[int64_t, shared_ptr[CBlock]] blocks)
        c_string ToString()


ctypedef CExpr* CExprPtr


cdef extern from "typeinfer.h" namespace "bodo" nogil:
    cdef cppclass CUnifyResult" bodo::UnifyResult":
        unordered_map[c_string, CType*] typemap
        CType* return_type
        unordered_map[CExprPtr, shared_ptr[CSignature]] calltypes

    cdef cppclass CTypeInferer" bodo::TypeInferer":
        CTypeInferer(shared_ptr[CFunctionIR] f_ir)
        CTypeInferer()
        void seed_argument(c_string name, int index, CType* typ)
        void build_constraint() except +
        void propagate(c_bool raise_errors) except +
        shared_ptr[CUnifyResult] unify(c_bool raise_errors) except +


cdef class CTypeWrapper:
    """Wrapper around native CType value to pass to Python (cache in Numba type)
    """
    cdef CType* c_type

    @staticmethod
    cdef from_c_type(CType* ct):
        cdef CTypeWrapper wrapper = CTypeWrapper.__new__(CTypeWrapper)
        wrapper.c_type = ct
        return wrapper

    cdef CType* get_c_type(self):
        return self.c_type


cdef public CType* unbox_type(object t):
    """Unbox Type object to equivalent native CType value
    """
    cdef CTypeWrapper wrapped_type

    # Use wrapped CType in the object if available
    if hasattr(t, "c_type"):
        wrapped_type = t.c_type
        return wrapped_type.get_c_type()

    # Not using isinstance() since we want to match exact types (not subclasses)
    if type(t) is types.Integer:
        ret = <CType*> CInteger.get(<int>t.bitwidth, <c_bool>t.signed)
        # Wrap and cache the CType in Numba type object for reuse
        t.c_type = CTypeWrapper.from_c_type(ret)
        return ret

    if type(t) is types.EnumMember:
        ret = <CType*> CEnumMember.get(<void*>t.instance_class, <CType*>unbox_type(t.dtype))
        t.c_type = CTypeWrapper.from_c_type(ret)
        return ret

    if type(t) is types.Function:
        return <CType*> CFunction.get(<void*>t)

    if type(t) is types.Module:
        return <CType*> CModule.get(<void*>(t.pymod))

    if type(t) is types.NumberClass:
        return <CType*> CNumberClass.get(<CType*>unbox_type(t.instance_type))

    if t == types.boolean:
        ret = <CType*> CBoolean.get()
        t.c_type = CTypeWrapper.from_c_type(ret)
        return ret

    raise TypeError(f"unbox_type: type {type(t)} not supported yet")


def set_compatible_types(fromty, toty, by):
    """Add cast rule to native typer's TypeManager
    """
    CTypeManager.setCompatibleTypes(unbox_type(fromty), unbox_type(toty), by)


def check_compatible_types(fromty, toty):
    """Get type cast rule from native typer's TypeManager. Used for testing only.
    """
    cdef optional[CConversion] conv = CTypeManager.checkCompatibleTypes(unbox_type(fromty), unbox_type(toty))
    if conv == nullopt:
        return None
    return Conversion(<int>conv.value())

cdef public vector[CType*] unbox_args(pos_args):
    """Unbox tuple of Type objects into native vector of CType values
    """
    cdef vector[CType*] ctypes
    for t in pos_args:
        ctypes.push_back(unbox_type(t))

    return ctypes


cdef box_sig(shared_ptr[CSignature] csig):
    """Box CSignature value into Signature objects
    """
    return signature(csig.get().return_type.to_py(), *tuple(t.to_py() for t in csig.get().args))


def bodo_resolve_call(func_key, pos_args):
    """Resolve call type using native type inference in C++
    """
    cdef vector[CType*] c_args = unbox_args(pos_args)
    return box_sig(resolve_call_type(func_key, c_args))


cdef public shared_ptr[CSignature] unbox_sig(object sig):
    """Unbox Signature object to native version
    """
    cdef vector[CType*] cargs
    for t in sig.args:
        cargs.push_back(unbox_type(t))
    return make_shared[CSignature](unbox_type(sig.return_type), cargs)


cdef shared_ptr[CLoc] unbox_loc(object loc, c_bool empty_loc=False):
    """Unbox location from Python to native"""
    if empty_loc:
        return make_shared[CLoc]()

    cdef c_string short_fname = <c_string>(os.path.basename(loc.filename))
    cdef optional[uint64_t] col
    if loc.col is not None:
        col = <uint64_t>(loc.col)
    return make_shared[CLoc](
        <c_string>(loc.filename), short_fname, <int64_t>(loc.line), col,
        <c_bool>(loc.maybe_decorator)
    )


cdef shared_ptr[CVar] unbox_var(object var, c_bool empty_loc=False):
    """Unbox ir.Var from Python to native"""
    return make_shared[CVar](<c_string>var.name.encode(), unbox_loc(var.loc, empty_loc))


cdef shared_ptr[CExpr] unbox_expr(expr, c_bool empty_loc=False):
    """Unbox ir.Expr from Python to native"""

    # Multiple
    cdef vector[shared_ptr[CVar]] cargs
    # CallExpr
    cdef vector[pair[c_string, shared_ptr[CVar]]] ckws
    cdef optional[shared_ptr[CVar]] cvararg = nullopt
    cdef optional[shared_ptr[CVar]] cvarkwarg = nullopt
    # StaticGetItem
    cdef optional[shared_ptr[CVar]] index_var
    cdef optional[int64_t] start_index
    cdef optional[int64_t] stop_index
    cdef optional[int64_t] step_index
    # Phi
    cdef vector[shared_ptr[CVar]] incoming_values
    cdef vector[uint64_t] incoming_blocks

    # Operators
    if expr.op == "binop":
        assert expr.static_lhs == ir.UNDEFINED, f"Unsupported static_lhs type: {type(expr.static_lhs)}"
        assert expr.static_rhs == ir.UNDEFINED, f"Unsupported static_rhs type: {type(expr.static_rhs)}"
        op_str = utils.OPERATORS_TO_BUILTINS.get(expr.fn, None)
        return <shared_ptr[CExpr]>make_shared[CBinOpExpr](unbox_var(expr.lhs, empty_loc), unbox_var(expr.rhs, empty_loc), <void*>(expr.fn), unbox_loc(expr.loc, empty_loc), <c_string>op_str)
    if expr.op == "inplace_binop":
        assert expr.static_lhs == ir.UNDEFINED, f"Unsupported static_lhs type: {type(expr.static_lhs)}"
        assert expr.static_rhs == ir.UNDEFINED, f"Unsupported static_rhs type: {type(expr.static_rhs)}"
        return <shared_ptr[CExpr]>make_shared[CInPlaceBinOpExpr](unbox_var(expr.lhs, empty_loc), unbox_var(expr.rhs, empty_loc), <void*>(expr.fn), <void*>(expr.immutable_fn), unbox_loc(expr.loc, empty_loc))
    if expr.op == "unary":
        assert isinstance(expr.value, ir.Var), f"Unsupported value type: {type(expr.value)}"
        return <shared_ptr[CExpr]>make_shared[CUnaryExpr](<void*>(expr.fn), unbox_var(expr.value, empty_loc), unbox_loc(expr.loc, empty_loc))

    if expr.op == "call":
        assert expr.target is None, f"Call target not supported yet: {expr.target}"
        for arg in expr.args:
            cargs.push_back(unbox_var(arg))
        for kwname, kw in expr.kws:
            ckws.push_back(pair[c_string, shared_ptr[CVar]](<c_string>kwname.encode(), unbox_var(kw)))
        if expr.vararg is not None:
            cvararg = unbox_var(expr.vararg)
        if expr.varkwarg is not None:
            cvarkwarg = unbox_var(expr.varkwarg)
        return <shared_ptr[CExpr]>make_shared[CCallExpr](unbox_var(expr.func, empty_loc), cargs, ckws, cvararg, cvarkwarg, unbox_loc(expr.loc, empty_loc), <void*>expr)

    # Container Building
    if expr.op == "build_tuple":
        for item in expr.items:
            cargs.push_back(unbox_var(item, empty_loc))
        return <shared_ptr[CExpr]>make_shared[CBuildTupleExpr](cargs, unbox_loc(expr.loc, empty_loc))
    if expr.op == "build_list":
        for item in expr.items:
            cargs.push_back(unbox_var(item, empty_loc))
        return <shared_ptr[CExpr]>make_shared[CBuildListExpr](cargs, unbox_loc(expr.loc, empty_loc))

    # Item Accessing
    if expr.op == "pair_first":
        return <shared_ptr[CExpr]>make_shared[CPairFirstExpr](unbox_var(expr.value, empty_loc), unbox_loc(expr.loc, empty_loc))
    if expr.op == "pair_second":
        return <shared_ptr[CExpr]>make_shared[CPairSecondExpr](unbox_var(expr.value, empty_loc), unbox_loc(expr.loc, empty_loc))
    if expr.op == "getattr":
        return <shared_ptr[CExpr]>make_shared[CGetAttrExpr](unbox_var(expr.value, empty_loc), <c_string>(expr.attr), unbox_loc(expr.loc, empty_loc))
    if expr.op == "getitem":
        return <shared_ptr[CExpr]>make_shared[CGetItemExpr](unbox_var(expr.value, empty_loc), unbox_var(expr.index, empty_loc), unbox_loc(expr.loc, empty_loc))
    elif expr.op == "static_getitem":
        if expr.index_var is not None:
            index_var = unbox_var(expr.index_var)
        if isinstance(expr.index, int):
            return <shared_ptr[CExpr]>make_shared[CStaticGetItemExpr](unbox_var(expr.value, empty_loc), <int64_t>(expr.index), index_var, unbox_loc(expr.loc, empty_loc))
        elif isinstance(expr.index, slice):
            if expr.index.start is not None:
                start_index = <int64_t>(expr.index.start)
            if expr.index.stop is not None:
                stop_index = <int64_t>(expr.index.stop)
            if expr.index.step is not None:
                step_index = <int64_t>(expr.index.step)
            return <shared_ptr[CExpr]>make_shared[CStaticGetItemExpr](
                unbox_var(expr.value, empty_loc),
                start_index, stop_index, step_index, index_var,
                unbox_loc(expr.loc, empty_loc)
            )
        else:
            raise ValueError(f"Unsupported index type in static_getitem expr: {type(expr.index)}")

    # Iterators
    if expr.op == "exhaust_iter":
        return <shared_ptr[CExpr]>make_shared[CExhaustIterExpr](unbox_var(expr.value, empty_loc), <int64_t>(expr.count), unbox_loc(expr.loc, empty_loc))
    if expr.op == "getiter":
        return <shared_ptr[CExpr]>make_shared[CGetIterExpr](unbox_var(expr.value, empty_loc), unbox_loc(expr.loc, empty_loc))
    if expr.op == "iternext":
        return <shared_ptr[CExpr]>make_shared[CIterNextExpr](unbox_var(expr.value, empty_loc), unbox_loc(expr.loc, empty_loc))

    if expr.op == "cast":
        return <shared_ptr[CExpr]>make_shared[CCast](unbox_var(expr.value, empty_loc), unbox_loc(expr.loc, empty_loc))

    if expr.op == "phi":
        for v in expr.incoming_values:
            incoming_values.push_back(unbox_var(v, empty_loc))
        for b in expr.incoming_blocks:
            incoming_blocks.push_back(b)
        return <shared_ptr[CExpr]>make_shared[CPhiExpr](incoming_values, incoming_blocks, unbox_loc(expr.loc, empty_loc))

    raise TypeError(f"Unsupported Expr node {expr.op} {expr}")


cdef shared_ptr[CAbstractRHS] unbox_rhs(object rhs, c_bool empty_loc=False):
    """Unbox ir.AbstractRHS child object into equivalent C++ version"""

    if isinstance(rhs, ir.Arg):
        return <shared_ptr[CAbstractRHS]>make_shared[CArg](<int>rhs.index, <c_string>rhs.name, unbox_loc(rhs.loc, empty_loc))
    elif isinstance(rhs, ir.Var):
        return <shared_ptr[CAbstractRHS]>unbox_var(rhs, empty_loc)
    elif isinstance(rhs, ir.Const):
        return <shared_ptr[CAbstractRHS]>make_shared[CConst](<void*>(rhs.value), unbox_loc(rhs.loc, empty_loc), <c_bool>rhs.use_literal_type)
    elif isinstance(rhs, ir.Global):
        return <shared_ptr[CAbstractRHS]>make_shared[CGlobal](<c_string>rhs.name, <void*>(rhs.value), unbox_loc(rhs.loc, empty_loc))
    elif isinstance(rhs, ir.FreeVar):
        return <shared_ptr[CAbstractRHS]>make_shared[CFreeVar](<int64_t>(rhs.index), <c_string>(rhs.name), <void*>(rhs.value), unbox_loc(rhs.loc, empty_loc))
    elif isinstance(rhs, ir.Expr):
        return <shared_ptr[CAbstractRHS]>(unbox_expr(rhs))
    else:
        raise TypeError(f"Unsupported RHS node {type(rhs)}")


cdef shared_ptr[CStmt] unbox_stmt(stmt, c_bool empty_loc=False):
    """Unbox statement from Python to native"""

    # Print
    cdef vector[shared_ptr[CVar]] values
    cdef optional[shared_ptr[CVar]] vararg
    # StaticRaise
    cdef vector[c_string] exc_args_inner
    cdef optional[vector[c_string]] exc_args

    if isinstance(stmt, ir.Assign):
        return <shared_ptr[CStmt]>make_shared[CAssign](unbox_var(stmt.target, empty_loc), unbox_rhs(stmt.value, empty_loc), unbox_loc(stmt.loc, empty_loc))
    elif isinstance(stmt, ir.SetItem):
        return <shared_ptr[CStmt]>make_shared[CSetItem](unbox_var(stmt.target, empty_loc), unbox_var(stmt.index, empty_loc), unbox_var(stmt.value, empty_loc), unbox_loc(stmt.loc, empty_loc))
    elif isinstance(stmt, ir.StaticSetItem):
        if not isinstance(stmt.index, int):
            raise ValueError(f"Unsupported index type: {type(stmt.index)}")
        return <shared_ptr[CStmt]>make_shared[CStaticSetItem](unbox_var(stmt.target, empty_loc), <int64_t>(stmt.index), unbox_var(stmt.index_var, empty_loc), unbox_var(stmt.value, empty_loc), unbox_loc(stmt.loc, empty_loc))
    elif isinstance(stmt, ir.SetAttr):
        return <shared_ptr[CStmt]>make_shared[CSetAttr](unbox_var(stmt.target, empty_loc), <c_string>(stmt.attr), unbox_var(stmt.value, empty_loc), unbox_loc(stmt.loc, empty_loc))
    elif isinstance(stmt, ir.Print):
        for v in stmt.args:
            values.push_back(unbox_var(v, empty_loc))
        if stmt.vararg is not None:
            vararg = unbox_var(stmt.vararg, empty_loc)
        return <shared_ptr[CStmt]>make_shared[CPrint](values, vararg, unbox_loc(stmt.loc, empty_loc))
    elif isinstance(stmt, ir.Del):
        return <shared_ptr[CStmt]>make_shared[CDel](<c_string>(stmt.value), unbox_loc(stmt.loc, empty_loc))
    elif isinstance(stmt, ir.EnterWith):
        return <shared_ptr[CStmt]>make_shared[CEnterWith](unbox_var(stmt.contextmanager, empty_loc), <int64_t>(stmt.begin), <int64_t>(stmt.end), unbox_loc(stmt.loc, empty_loc))
    elif isinstance(stmt, ir.PopBlock):
        return <shared_ptr[CStmt]>make_shared[CPopBlock](unbox_loc(stmt.loc, empty_loc))
    elif isinstance(stmt, Parfor):
        raise NotImplementedError("Not implementation for statement type: Parfor")

    # Terminators
    elif isinstance(stmt, ir.Return):
        return <shared_ptr[CStmt]>make_shared[CReturn](unbox_var(stmt.value, empty_loc), unbox_loc(stmt.loc, empty_loc))
    elif isinstance(stmt, ir.Jump):
        return <shared_ptr[CStmt]>make_shared[CJump](<int64_t>(stmt.target), unbox_loc(stmt.loc, empty_loc))
    elif isinstance(stmt, ir.Branch):
        return <shared_ptr[CStmt]>make_shared[CBranch](unbox_var(stmt.cond, empty_loc), <int64_t>stmt.truebr, <int64_t>stmt.falsebr, unbox_loc(stmt.loc, empty_loc))
    elif isinstance(stmt, ir.Raise):
        return <shared_ptr[CStmt]>make_shared[CRaise](unbox_var(stmt.exception, empty_loc), unbox_loc(stmt.loc, empty_loc))
    elif isinstance(stmt, ir.StaticRaise):
        if stmt.exc_args is not None:
            for arg in stmt.exc_args:
                assert isinstance(arg, str), f"Unsupported exc_args type: {type(arg)}"
                exc_args_inner.push_back(arg)
            exc_args = exc_args_inner
        return <shared_ptr[CStmt]>make_shared[CStaticRaise](<void*>(stmt.exc_class), exc_args, unbox_loc(stmt.loc, empty_loc))
    
    raise TypeError(f"Unsupported Stmt node {type(stmt)}")


cdef shared_ptr[CBlock] unbox_block(object block, c_bool empty_loc=False):
    """Unbox ir.Block object into equivalent C++ version"""
    cdef shared_ptr[CBlock] c_block = make_shared[CBlock](unbox_loc(block.loc, empty_loc))

    for inst in block.body:
        c_block.get().body.push_back(unbox_stmt(inst, empty_loc))
    return c_block


cdef map[int64_t, shared_ptr[CBlock]] unbox_blocks(blocks, c_bool empty_loc=False):
    """Unbox blocks from native to Python"""
    cdef map[int64_t, shared_ptr[CBlock]] block_map

    for k, v in blocks.items():
        block_map[k] = unbox_block(v, empty_loc)
    return block_map


cdef shared_ptr[CFunctionIR] unbox_ir(object f_ir, c_bool empty_loc=False):
    """Unbox ir.FunctionIR object into equivalent C++ version"""
    return make_shared[CFunctionIR](
        unbox_blocks(f_ir.blocks, empty_loc),
    )


cdef public c_string cpp_ir_native_to_string(object func_ir):
    cdef shared_ptr[CFunctionIR] fir = unbox_ir(func_ir, empty_loc=True)
    return fir.get().ToString()


def ir_native_to_string(func_ir):
    """
    Convert an IR to string representation using native code
    This is intended for internal or testing use
    """
    return cpp_ir_native_to_string(func_ir)


cdef box_typemap(unordered_map[c_string, CType*] typemap):
    return {p.first: p.second.to_py() for p in typemap}


cdef box_calltypes(unordered_map[CExprPtr, shared_ptr[CSignature]] calltypes):
    return {p.first.to_py(): box_sig(p.second) for p in calltypes}


def bodo_type_inference(interp, args, return_type,
                         locals=None, raise_errors=True):
    """Call native type inferer (boxes arguments and unboxes results).
    Throws error if native type inferer was unsuccessful due to gaps (needs fallback to Numba).
    """
    if locals is None:
        locals = {}
    if len(args) != interp.arg_count:
        raise TypeError("Mismatch number of argument types")

    # TODO[BSE-3924]: support return type and locals in native type inference
    assert return_type is None, "bodo_type_inference: return_type not supported yet"
    assert locals == {}, "bodo_type_inference: locals not supported yet"

    infer = CTypeInferer(unbox_ir(interp))

    # Bodo change: removed callstack_ctx and warnings contexts

    # Seed argument types
    for index, (name, ty) in enumerate(zip(interp.arg_names, args)):
        infer.seed_argument(name.encode(), index, unbox_type(ty))

    # TODO[BSE-3924]: support return type
    # Seed return type
    # if return_type is not None:
    #     infer.seed_return(return_type)

    # TODO[BSE-3924]: support locals
    # Seed local types
    # for k, v in locals.items():
    #     infer.seed_type(k, v)

    infer.build_constraint()
    # return errors in case of partial typing
    # TODO[BSE-3925]: return errors from propagate
    infer.propagate(raise_errors=raise_errors)
    errs = None
    unify_res = infer.unify(raise_errors=raise_errors)
    typemap = box_typemap(unify_res.get().typemap)
    restype = unify_res.get().return_type.to_py()
    calltypes = box_calltypes(unify_res.get().calltypes)
    # NOTE: native IR should stay alive until box_calltypes is done
    # since native calltypes has Expr pointers that need to stay alive.

    return _TypingResults(typemap, restype, calltypes, errs)
