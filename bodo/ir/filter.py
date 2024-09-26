"""
This file contains dictionaries mapping BodoSQL kernel name to
corresponding SQL functions. This file also contains
supported_arrow_funcs_map, which is a dictionary that maps
BodoSQL kernel name to an equivalent PyArrow compute function.

Dictionaries are separated by category
(string functions, datetime functions, etc.) and
number of arguments.

The keys are the name of the BodoSQL kernel.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from numba.core import cgutils, ir, types
from numba.core.ir_utils import find_callname, get_definition, guard
from numba.core.typing.templates import (
    AbstractTemplate,
    infer_global,
    signature,
)
from numba.extending import (
    lower_builtin,
)

from bodo.utils.transform import get_const_value_inner
from bodo.utils.typing import is_overload_constant_str
from bodo.utils.utils import BodoError, is_call

string_funcs_no_arg_map = {
    "lower": "LOWER",
    "upper": "UPPER",
    "length": "LENGTH",
    "reverse": "REVERSE",
}

numeric_funcs_no_arg_map = {
    "abs": "ABS",
    "sign": "SIGN",
}

date_funcs_no_arg_map = {
    "get_hour": "HOUR",
    "get_minute": "MINUTE",
    "get_second": "SECOND",
    # TODO (srilman): YEAROFWEEK seems to map to get_year, but I think thats wrong (no account for weeks that start in previous year)
    "get_year": "YEAR",
    "yearofweek": "YEAROFWEEK",
    "yearofweekiso": "YEAROFWEEKISO",
    "dayofmonth": "DAY",
    "dayofweek": "DAYOFWEEK",
    "dayofweekiso": "DAYOFWEEKISO",
    "dayofyear": "DAYOFYEAR",
    # TODO (srilman): Why are there 2 different ones?
    "week": "WEEK",
    "weekofyear": "WEEKOFYEAR",
    # TODO (srilman): WEEKISO seems to map to get_weekofyear, but I think thats wrong (non ISO version)
    "get_month": "MONTH",
    "get_quarter": "QUARTER",
}

string_funcs_map = {
    "ltrim": "LTRIM",
    "rtrim": "RTRIM",
    "lpad": "LPAD",
    "rpad": "RPAD",
    "trim": "TRIM",
    "split": "SPLIT_PART",
    "contains": "CONTAINS",
    "coalesce": "COALESCE",
    "repeat": "REPEAT",
    "translate": "TRANSLATE",
    "strtok": "STRTOK",
    "initcap": "INITCAP",
    "concat_ws": "CONCAT",
    "left": "LEFT",
    "right": "RIGHT",
    "position": "POSITION",
    "replace": "REPLACE",
    "substring": "SUBSTRING",
    "charindex": "POSITION",
    "editdistance_no_max": "EDITDISTANCE",
    "editdistance_with_max": "EDITDISTANCE",
    "regexp_substr": "REGEXP_SUBSTR",
    "regexp_instr": "REGEXP_INSTR",
    "regexp_replace": "REGEXP_REPLACE",
    "regexp_count": "REGEXP_COUNT",
    "startswith": "STARTSWITH",
    "endswith": "ENDSWITH",
}

numeric_funcs_map = {
    "mod": "MOD",
    "round": "ROUND",
    "trunc": "TRUNC",
    "truncate": "TRUNCATE",
    "ceil": "CEIL",
    "floor": "FLOOR",
}

cond_funcs_map = {
    "least": "LEAST",
    "greatest": "GREATEST",
}

# TODO(njriasan): Add remaining cast functions.
# only to_char and try_to_char have 1 argument.
cast_funcs_map = {
    "to_char": "TO_CHAR",
    "try_to_char": "TRY_TO_CHAR",
}

supported_funcs_no_arg_map = (
    string_funcs_no_arg_map | numeric_funcs_no_arg_map | date_funcs_no_arg_map
)

supported_funcs_map = (
    supported_funcs_no_arg_map
    | numeric_funcs_map
    | string_funcs_map
    | cond_funcs_map
    | cast_funcs_map
)

supported_arrow_funcs_map = {
    "lower": "utf8_lower",
    "upper": "utf8_upper",
    "length": "utf8_length",
    "reverse": "utf8_reverse",
    "startswith": "starts_with",
    "endswith": "ends_with",
    "contains": "match_substring",
    "coalesce": "coalesce",
    "case_insensitive_startswith": "starts_with",
    "case_insensitive_endswith": "ends_with",
    "case_insensitive_contains": "match_substring",
    "initcap": "utf8_capitalize",
}


# ----------------- Bodo IR Filter Expression Data Structure -----------------
class Filter(ABC):
    pass


@dataclass(repr=True, frozen=True)
class Scalar(Filter):
    val: ir.Var

    def __str__(self) -> str:
        return f"scalar({self.val})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Scalar):
            return False

        return self.val.name == other.val.name


@dataclass(repr=True, eq=True, frozen=True)
class Ref(Filter):
    val: str

    def __str__(self) -> str:
        return f"ref({self.val})"


class Op(Filter):
    op: str
    args: tuple[Filter, ...]

    def __init__(self, op: str, *args: Filter) -> None:
        self.op = op
        self.args = tuple(args)

    def __str__(self) -> str:
        return f"{self.op}({', '.join(str(arg) for arg in self.args)})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Op):
            return False

        return self.op == other.op and self.args == other.args

    def __hash__(self) -> int:
        return hash((self.op, self.args))


# -------------- Dummy Functions for Numba Typing and Lowering --------------
# -------- Necessary to pass filter expressions from BodoSQL to Bodo --------


def make_scalar(val: Any) -> Scalar:
    raise NotImplementedError("bodo.ir.filter.make_scalar is not implemented in Python")


def make_ref(val: str) -> Ref:
    raise NotImplementedError("bodo.ir.filter.make_ref is not implemented in Python")


def make_op(op: str, *args: Filter) -> Op:
    raise NotImplementedError("bodo.ir.filter.make_op is not implemented in Python")


@infer_global(make_scalar)
class ScalarTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1, "bodo.ir.filter.Scalar takes exactly 1 argument"
        (val_arg,) = args
        # First Arg can be any type in the IR
        return signature(types.bool_, val_arg)


@infer_global(make_ref)
class RefTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1, "bodo.ir.filter.Ref takes exactly 1 argument"
        (val_arg,) = args
        assert (
            val_arg == types.unicode_type
        ), "Argumnt to bodo.ir.filter.Ref must be type string"
        return signature(types.bool_, val_arg)


@infer_global(make_op)
class OpTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.bool_, *args)


@lower_builtin(make_scalar, types.VarArg(types.Any))
@lower_builtin(make_ref, types.VarArg(types.Any))
@lower_builtin(make_op, types.VarArg(types.Any))
def lower_join_dummy(context, builder, sig, args):
    return cgutils.true_bit


# ----------------------------------------------------------------------------


T = TypeVar("T")


class FilterVisitor(Generic[T], ABC):
    """
    Visitor Pattern for Bodo IR Filter Expression
    Can be used to traverse and transform a filter object

    Used mainly in Distributed Pass of Connector Nodes to construct
    filters for different backends (SQL, Arrow, Iceberg, etc)

    Entrance function is visit(filter: Filter) -> T
    Child classes should implement visit_* methods
    """

    def visit(self, filter: Filter) -> T:
        if isinstance(filter, Scalar):
            return self.visit_scalar(filter)
        elif isinstance(filter, Ref):
            return self.visit_ref(filter)
        elif isinstance(filter, Op):
            return self.visit_op(filter)
        else:
            raise BodoError(f"FilterVisitor: Unknown Filter Type: {type(filter)}")

    @abstractmethod
    def visit_scalar(self, filter: Scalar) -> T:
        raise NotImplementedError()

    @abstractmethod
    def visit_ref(self, filter: Ref) -> T:
        raise NotImplementedError()

    @abstractmethod
    def visit_op(self, filter: Op) -> T:
        raise NotImplementedError()


class SimplifyFilterVisitor(FilterVisitor[Filter]):
    """
    Visitor to Simplify Filter Expressions. Applies the following rules:
    - Flatten nested AND and OR expressions
    - Remove redundant terms in AND and OR expressions
    """

    def visit_scalar(self, scalar: Scalar) -> Filter:
        return scalar

    def visit_ref(self, ref: Ref) -> Filter:
        return ref

    def visit_op(self, op: Op) -> Filter:
        op_name = op.op
        args = op.args

        if op_name in ("AND", "OR"):
            # Flatten nested AND and OR expressions
            new_args = []
            for arg in args:
                arg = self.visit(arg)
                if isinstance(arg, Op) and arg.op == op_name:
                    new_args.extend(arg.args)
                else:
                    new_args.append(arg)

            # Remove duplicate terms
            # Note, using dict.fromkeys instead of set to preserve order
            # Dicts are ordered in Python 3.7+
            new_args = tuple(dict.fromkeys(new_args))
            return Op(op_name, *new_args) if len(new_args) > 1 else new_args[0]

        return op


def build_filter_from_ir(filter_var: ir.Var, fir: ir.FunctionIR, typemap) -> Filter:
    """
    Constructs a Bodo IR Filter Expression for Use in Connector Nodes
    when the filter is explicitly defined as a function argument, constructed
    via the bodo.ir.filter.make_* functions.

    Args:
        filter_var: The IR Variable representing the filter arg
        fir: The Functions IR
        typemap: Mapping from IR variable name to Numba type

    Returns:
        The Bodo IR Filter expression representing the filter expression
    """

    filter_def = get_definition(fir, filter_var)
    if not is_call(filter_def):
        raise BodoError(
            "Building Filter from IR Failed, Filter is incorrectly constructed"
        )

    fdef = guard(find_callname, fir, filter_def)
    match fdef:
        case ("make_scalar", "bodo.ir.filter"):
            if len(filter_def.args) != 1:
                raise BodoError(
                    "Building Filter from IR Failed, Scalar filter has more than 1 argument"
                )
            return Scalar(filter_def.args[0])

        case ("make_ref", "bodo.ir.filter"):
            if len(filter_def.args) != 1:
                raise BodoError(
                    "Building Filter from IR Failed, Ref filter has more than 1 argument"
                )
            if not is_overload_constant_str(typemap[filter_def.args[0].name]):
                raise BodoError(
                    "Building Filter from IR Failed, Ref filter arg is not constant str"
                )
            arg_val: str = get_const_value_inner(
                fir, filter_def.args[0], typemap=typemap
            )
            return Ref(arg_val)

        case ("make_op", "bodo.ir.filter"):
            if not is_overload_constant_str(typemap[filter_def.args[0].name]):
                raise BodoError(
                    "Building Filter from IR Failed, first arg of Op filter is not constant str"
                )
            op_name: str = get_const_value_inner(
                fir, filter_def.args[0], typemap=typemap
            )

            args = (
                build_filter_from_ir(arg, fir, typemap) for arg in filter_def.args[1:]
            )
            return Op(op_name, *args)

        case (name, path):
            raise BodoError(
                f"Building Filter from IR Failed, Unknown Filter Func: {path}.{name}"
            )
        case None:
            raise ValueError(f"Building Filter from IR Failed, Undefined Filter Def")


def get_filter_predicate_compute_func(col_val) -> str:
    """
    Verifies that the input filter (col_val) is a valid
    type based on the Bodo compiler internals.

    Returns the compute function name as a string literal.
    """
    assert isinstance(
        col_val, Op
    ), f"Filter must of type bodo.ir.filter.Op. Invalid filter: {col_val}"

    compute_func = col_val.op
    assert (
        compute_func in supported_funcs_map
    ), f"Unsupported compute function for column in filter predicate: {compute_func}"
    return compute_func
