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

import re
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
        assert val_arg == types.unicode_type, (
            "Argumnt to bodo.ir.filter.Ref must be type string"
        )
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
            raise ValueError("Building Filter from IR Failed, Undefined Filter Def")


def get_filter_predicate_compute_func(col_val) -> str:
    """
    Verifies that the input filter (col_val) is a valid
    type based on the Bodo compiler internals.

    Returns the compute function name as a string literal.
    """
    assert isinstance(col_val, Op), (
        f"Filter must of type bodo.ir.filter.Op. Invalid filter: {col_val}"
    )

    compute_func = col_val.op
    assert compute_func in supported_funcs_map, (
        f"Unsupported compute function for column in filter predicate: {compute_func}"
    )
    return compute_func


def convert_sql_pattern_to_python_compile_time(
    pattern: str,
    escape: str,
    make_output_lowercase: bool,
) -> tuple[str, bool, bool, bool, bool]:
    """
    Converts a SQL pattern to its Python equivalent. This is used for like/ilike
    paths where the pattern is a constant string literal. One challenge that arises
    is that Python has additional special characters that SQL doesn't have. As a result,
    we have to be careful and escape patterns that may arise in Python.

    In some cases it may also be possible to avoid regex entirely. In those cases
    it may be possible to replace a regular expression check with `==`, `startswith`,
    `endswith` or `in`. In those cases we pass back a few extra boolean values:

        - requires_regex: Can we avoid regex entirely
        - must_match_start: Must the pattern be found at the start of the string?
        - must_match_end: Must the pattern be found at the end of the string?
        - match_anything: Will the pattern match any non-null string?

    If requires_regex == False then the string returned will be returned without escaping
    the contents.

    Note there may be additional optimizations possible based on the _ escape
    character, but at this time we only consider % for avoiding regex.

    Here are a couple example calls to explain this more clearly

    ("%s", "", False) -> ("s", False, False, True, False)

        This first pattern matches any string that ends with "s". This does not
        require a regex so we keep the pattern as a regular string and set
        requires_regex=False. Then we set must_match_end=True so we know we must do endswith.

    ("Ha^%", "^", False) -> ("Ha%", False, True, True, False)

        This second pattern matches the literal string "Ha%". The escape value of ^
        tells us that "^%" matches a literal "%" rather than a wild card. As a result
        we do not need a regex (there are no wildcard), but we must match the string exactly.

    ("Ha_d", "", True) -> ("^ha.d$", True, True, True, False)

        This third pattern requires an actual regex. Since both the start and end must match
        we append the required "^" and "$" to the Python regex. In addition, since we specified
        "make_output_lowercase" the output pattern will be converted to all lowercase characters.
        This is used for case insensitive comparison.

    Args:
        pattern (str): A SQL pattern passed to like. This pattern can contain values
        that are interpreted literally and SQL wildcards (_ and %).

        escape (str): Character used to escape SQL wildcards. If this character is followed
        by either a _ or a % then that matches the literal character _ or %. For example,
        if escape == ^, then ^% -> %. If there is no escape character this will be the empty
        string, but we do not have an optimized path.

        make_output_lowercase(bool): Should the output pattern be converted to lowercase.
        For case insensitive matching we convert everything to lowercase. However, we cannot
        just convert the whole pattern because the escape character must remain case sensitive.


    Returns:
        Tuple[str, bool, bool, bool, bool]: The modified string and a series of variables
        used for deciding which operator needs to be used to compute the result.
    """
    # At high level we iterate through the string one character
    # at a time checking its value. In this process we create several
    # "groups", which are just batches of characters that we process all
    # at once (escape or convert to lowercase). When iterating can have 1 of 3 cases.
    #
    # Case 1: Escape character match
    #
    #   If we match the escape character then we check if the next character
    #   is a valid SQL wildcard. If so we append the current group
    #   and append the literal wildcard character.
    #
    # Case 2: Wildcard match
    #   If the character is a wildcard then we append the current group.
    #   In addition we add the Python regex equivalent for the wildcard. % maps to ".*"
    #   and _ maps to "."
    #
    # Case 3: Regular Character match
    #   Do nothing except advance the state/update metadata.
    #
    #
    # Once we have iterated through the string we have the groups necessary to construct a
    # pattern. We combine the groups together and depending on metadata either output a regular
    # string or a Python regex (as well as other booleans used for optimizations). Since we don't
    # know if a string is a regex until we finish, we keep two copies of the groups, one for if
    # we create a regex and one for if we keep it a regular string. Here is an example of splitting
    # a string into groups:
    #
    # pattern="St.ar_er", escape=""
    #   This has one wildcard, so the string effectively has 3 groups: "St.ar", "_", "er". Our two
    #   lists looks like:
    #
    #   escaped_lst (regex): ["St\\.ar", ".", "er"] - Note we must escape the "." to a literal.
    #
    #   unescaped_lst (regular string): ["St.ar", "er"] - We omit the wildcard because this
    #           list can't be chosen in this case.

    def append_group(unescaped_lst, escaped_lst, pattern, group_start, group_end):
        """Append a group of characters from the pattern to the two lists.
        For escaped_lst the pattern should be escaped to avoid conflicts with
        Python regex special characters.

        Args:
            unescaped_lst (List[str]): List of unescaped groups. Used by the regular
                String path.
            escaped_lst (List[str]): List of unescaped groups. Used by the regular
                String path.
            pattern (str): The whole pattern
            group_start (int): Index for the start of the group.
            group_end (int): Index for the end of the group (non-inclusive)
        """
        if group_start != group_end:
            group = pattern[group_start:group_end]
            # Make the group lowercase before escape so escape characters remain
            # accurate.
            if make_output_lowercase:
                group = group.lower()
            unescaped_lst.append(group)
            # Escape any possible Python wildcards
            escaped_lst.append(re.escape(group))

    # List of groups that have been escaped.
    escaped_lst = []
    # List of groups that haven't been escaped
    unescaped_lst = []
    # Wildcards to check for SQL
    sql_wildcards = ("%", "_")

    # Metadata we store to enable optimizations. These can tell us
    # if a string requires a regex for example.

    # Track the first non percent location
    # for tracking if we have all % or leading %
    first_non_percent = -1
    # Track info about the size of a percent group
    # for determining trailing %
    group_is_percent = False
    percent_group_size = 0
    # Get information about the first non starting percent
    # index. This is used to determine if we need a regex
    # (we can skip regex if the only % groups are at start
    # and end and there is no _ character).
    first_non_start_percent = -1

    # End of metadata

    # Index starting the current group
    group_start = 0
    # Should we output a regex or regular String
    requires_regex = False
    pattern_size = len(pattern)
    i = 0
    while i < pattern_size:
        current_char = pattern[i]
        # Case 1: Escape character match
        # If we match escape followed by a wildcard then this is the literal wildcard.
        if (
            current_char == escape
            and (i < (pattern_size - 1))
            and pattern[i + 1] in sql_wildcards
        ):
            # Add the previous group if it exists
            append_group(unescaped_lst, escaped_lst, pattern, group_start, i)
            # Update our metadata to indicate that the current group
            # is not a %. In addition, we indicate the first non-percent
            # character has been reached.
            group_is_percent = False
            if first_non_percent == -1:
                first_non_percent = i
            # Append the wildcard. To future proof against new
            # re special character changes in later Python versions
            # we escape even though its not necessary now.
            wildcard = pattern[i + 1]
            unescaped_lst.append(wildcard)
            escaped_lst.append(re.escape(wildcard))
            # Skip the character and the wildcard for the next group
            group_start = i + 2
            i += 2
        else:
            # Case 2: Wildcard Match
            if current_char in sql_wildcards:
                # Add the previous group if it exists
                append_group(unescaped_lst, escaped_lst, pattern, group_start, i)
                # Next group will start after this section
                group_start = i + 1
                if current_char == "%":
                    if not group_is_percent:
                        percent_group_size = 0
                    group_is_percent = True
                    percent_group_size += 1
                    # We can omit any leading % as an
                    # optimization.
                    if first_non_percent != -1:
                        # Replace the wildcards. We can avoid
                        # unescaped_lst because it can't be chosen
                        # if there any kept wild cards.
                        escaped_lst.append(".*")
                        # If first_non_percent != -1 and first_non_start_percent == -1
                        # then we know we have reached our first % that isn't at the start
                        # of the string. As a result we update our metadata.
                        if first_non_start_percent == -1:
                            first_non_start_percent = i
                else:
                    # We are not optimized for _ yet, so
                    # we always require a regex.
                    requires_regex = True
                    # Update our metadata to indicate that the current group
                    # is not a %. In addition, we indicate the first non-percent
                    # character has been reached.
                    group_is_percent = False
                    if first_non_percent == -1:
                        first_non_percent = i

                    # Replace the wildcards. We can avoid
                    # unescaped_lst because it can't be chosen
                    # if there any kept wild cards.
                    escaped_lst.append(".")
            # Case 3: Regular Character match
            else:
                # Update our metadata to indicate that the current group
                # is not a %. In addition, we indicate the first non-percent
                # character has been reached.
                group_is_percent = False
                if first_non_percent == -1:
                    first_non_percent = i
            i += 1

    # If we didn't trail with a wildcard append the final group.
    append_group(unescaped_lst, escaped_lst, pattern, group_start, len(pattern))

    # Set the metadata for the output flags. We are basically checking for the following
    # information:
    #
    # requires_regex - Did the regex contain a % in the middle or a _ anywhere (not escaped).
    #   For example "b%t" or "_t" -> True, but "%s" -> False.
    #
    # must_match_start: Was the first character a non % character (or do we have an empty string)
    #
    # must_match_end: Was the last character a % that's not escaped.
    #
    # match_anything: Is the string entirely %

    # Determine if we have an internal %. This mean
    # we have a percent after the start and it is
    # not the trailing group.
    has_internal_percent = first_non_start_percent != -1 and (
        (not group_is_percent)
        or ((first_non_start_percent + percent_group_size) != len(pattern))
    )

    # Determine if we need a regex.
    requires_regex = requires_regex or has_internal_percent

    # If we have the empty string for pattern or don't
    # start with %s then we must match the start.
    must_match_start = (first_non_percent == 0) or (len(pattern) == 0)

    # If we don't end in a percent we must match the end
    must_match_end = not group_is_percent

    # If we have all % we are always True
    match_anything = first_non_percent == -1 and len(pattern) > 0

    # Create the final pattern depending on if we need a regex.
    if requires_regex:
        # Update the regex to include ^ and $ if necessary.

        # We can remove any trailing percents if we don't match the end
        if must_match_end:
            # If we must match the end append a $
            escaped_lst.append("$")
        else:
            # We can omit any trailing % as an optimization
            escaped_lst = escaped_lst[:-percent_group_size]

        if must_match_start:
            # If we must match the start insert a ^
            escaped_lst = ["^"] + escaped_lst

        # Regex uses the escaped list
        target_list = escaped_lst
    else:
        # Non-regex must use the unescaped list
        target_list = unescaped_lst

    final_pattern = "".join(target_list)

    return (
        final_pattern,
        requires_regex,
        must_match_start,
        must_match_end,
        match_anything,
    )
