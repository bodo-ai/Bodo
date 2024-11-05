import typing as pt  # Any, Tuple, Dict
from abc import ABCMeta, abstractmethod
from types import NoneType

from numba.core import types

from bodo.hiframes.pd_series_ext import (
    SeriesType,
    is_dt64_series_typ,
    is_timedelta64_series_typ,
)
from bodo.utils.typing import (
    BodoError,
    get_literal_value,
    get_overload_const_str_len,
    is_iterable_type,
    is_literal_type,
    is_overload_bool,
    is_overload_constant_str,
    is_overload_float,
    is_overload_int,
    is_overload_none,
    is_overload_numeric_scalar,
    is_overload_str,
    is_str_arr_type,
)

_types_to_str: dict[type, str] = {
    int: "Integer",
    str: "String",
    bool: "Boolean",
    NoneType: "None",
    tuple: "Tuple",
    dict: "Dict",
    list: "List",
}


def format_requirements_list(
    to_string_elem: pt.Callable[[pt.Any], str], elems: list, usetick: bool
) -> str:
    """Format a list of requirements `elems` as a comma separated list where
    the last element is separated by an "or".

    Args:
        to_string_elem: Function mapping requirements in elems to an
            equivalent string representation.
        elems: The list of requirements.
        usetick: Whether to wrap requirements with `(for documentation-style
            formatting).

    Returns:
        The list of requirements separated by commas.
    """

    def to_string(elem: pt.Any) -> str:
        tick = "`" if usetick else ""
        elem_as_str = to_string_elem(elem)
        return f"{tick}{elem_as_str}{tick}"

    if len(elems) == 1:
        return to_string(elems[0])

    elems_as_strs = [to_string(elem) for elem in elems]

    return ", ".join(elems_as_strs[:-1]) + " or " + elems_as_strs[-1]


class ArgumentTypeChecker(metaclass=ABCMeta):
    @abstractmethod
    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        """Checks that `arg_type` is a valid argument given `context`.

        Checks that `arg_type` is a valid argument given `context` and
        returns the argument after some transformation (e.g. getting value
        from Literal type) along with an error string describing an "error
        message fragment" (or it can be None if the argument is valid.) An
        error message fragment should make sense either as a stand alone
        statement or as a list if the restrictions are intended to be composed
        for example "must be an Integer" is a valid since it can be used as a
        standalone such as `'arg' must be an Integer. Got: float` or it can be
        used combined with another requirement such as:
        ```
            'arg1' should satisfy one of the following requirements:
                * 'arg1' must be an Integer
                * 'arg1' must be a Tuple of 2 Integers
        ```

        Args:
            context: A context containing all arguments and their types or
                values that have been processed before the current arg.
            arg_type: The argument type or value to check.

        Returns:
            The argument after some transformation plus an error message if
                the argument was invalid.
        """

    @abstractmethod
    def explain_arg(self, context: dict[str, pt.Any]) -> str:
        """Generate documentation for type restrictions on arg.

        Generates a docstring fragment for the argument. Similar to the error
        message fragment (see `check_arg`), it should be general enough so
        that it can be used as a standalone statement or combined with
        multiple requirements.

        Args:
            context: A context containing all previously processed arguments.

        Returns:
            A brief, composable description of the argument requirement.
        """


class NDistinctValueArgumentChecker(ArgumentTypeChecker):
    """
    Checks that the argument is a compile time constant and is in a specific
    set of values.
    """

    def __init__(self, arg_name: str, values: pt.Iterable[pt.Any]):
        self.arg_name = arg_name
        self.values = values

    def _get_values_str(self, val):
        return f'"{val}"' if isinstance(val, str) else str(val)

    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        if is_literal_type(arg_type):
            val = get_literal_value(arg_type)
            if val in self.values:
                return val, None
        elif arg_type in self.values:
            # check default argument case
            return arg_type, None

        values_str = format_requirements_list(self._get_values_str, self.values, False)
        return (
            arg_type,
            f"must be a compile time constant and must be {values_str}",
        )

    def explain_arg(self, context: dict[str, pt.Any]) -> str:  # pragma: no cover
        values_str = format_requirements_list(self._get_values_str, self.values, True)
        return f"must be a compile time constant and must be {values_str}"


class ConstantArgumentChecker(ArgumentTypeChecker):
    """
    Checks that the argument is a compile time constant and has the correct
    type.
    """

    def __init__(self, arg_name: str, types: pt.Iterable[type]):
        self.arg_name = arg_name
        self.types = tuple(types)

    def _get_types_str(self, typ: pt.Any) -> str:
        return _types_to_str[typ] if typ in _types_to_str else str(typ)

    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        if is_literal_type(arg_type):
            val = get_literal_value(arg_type)
            if isinstance(val, self.types):
                return val, None
        elif isinstance(arg_type, self.types):
            # check default argument case
            return arg_type, None

        types_str = format_requirements_list(
            self._get_types_str, self.types, usetick=False
        )
        return arg_type, f"must be a constant {types_str}"

    def explain_arg(self, context: dict[str, pt.Any]) -> str:  # pragma: no cover
        types_str = format_requirements_list(
            self._get_types_str, self.types, usetick=True
        )
        return f"must be a compile time constant and must be type {types_str}"


class PrimitiveTypeArgumentChecker(ArgumentTypeChecker):
    """
    Base class for checkers that enforce an argument is a Primitive type.
    """

    def __init__(
        self,
        arg_name: str,
        type_name: str,
        is_overload_typ: pt.Callable[[pt.Any], bool],
    ):
        """Initialize PrimitiveTypeArgumentChecker with the type information to
        check.

        Args:
            arg_name: The name of the argument as it will appear to users.
            type_name: The name of the type the argument must be.
            is_overload_typ: A function that takes in a value or type and
                returns True if the argument is valid.
        """
        self.arg_name = arg_name
        self.type_name = type_name
        self.is_overload_typ = is_overload_typ

    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        if self.is_overload_typ(arg_type):
            return arg_type, None
        return arg_type, f"must be a {self.type_name}"

    def explain_arg(self, context: dict[str, pt.Any]) -> str:  # pragma: no cover
        return f"must be type `{self.type_name}`"


class IntegerScalarArgumentChecker(PrimitiveTypeArgumentChecker):
    def __init__(self, arg_name: str):
        super().__init__(arg_name, "Integer", is_overload_int)


class BooleanScalarArgumentChecker(PrimitiveTypeArgumentChecker):
    def __init__(self, arg_name: str):
        super().__init__(arg_name, "Boolean", is_overload_bool)


class FloatScalarArgumentChecker(PrimitiveTypeArgumentChecker):
    def __init__(self, arg_name: str):
        super().__init__(arg_name, "Float", is_overload_float)


class StringScalarArgumentChecker(PrimitiveTypeArgumentChecker):
    def __init__(self, arg_name: str):
        super().__init__(arg_name, "String", is_overload_str)


class CharScalarArgumentChecker(PrimitiveTypeArgumentChecker):
    """Checks that a value is a String and checks length is 1 if it can be
    determined at compile time.

    Note that if it the length cannot be determined at compile time, the check
    will pass assuming there is a check at runtime.
    """

    def __init__(self, arg_name: str):
        def is_overload_const_char_or_str(t: pt.Any) -> bool:
            return isinstance(t, types.UnicodeType) or (
                is_overload_constant_str(t) and get_overload_const_str_len(t) == 1
            )

        super().__init__(arg_name, "Character", is_overload_const_char_or_str)


class NumericScalarArgumentChecker(ArgumentTypeChecker):
    """
    Checker for arguments that can either be float or integer or None.
    """

    def __init__(self, arg_name: str):
        self.arg_name = arg_name

    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        if is_overload_numeric_scalar(arg_type):
            return arg_type, None

        return arg_type, "must be a Float, Integer or Boolean"

    def explain_arg(self, context: dict[str, pt.Any]) -> str:  # pragma: no cover
        return "must be `Integer`, `Float` or `Boolean`"


class NumericSeriesBinOpChecker(ArgumentTypeChecker):
    """Checker for arguments that can be float or integer scalar or iterable
    with 1-d numeric data such as list, tuple, Series, Index, etc. Intended for
    for Series Binop methods such as Series.sub.

    Args:
        arg_name: The name of the argument as it will appear to users.
    """

    def __init__(self, arg_name: str):
        self.arg_name = arg_name

    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        """
        Can either be numeric Scalar, or iterable with numeric data.
        """
        is_numeric_scalar = is_overload_numeric_scalar(arg_type)
        is_numeric_iterable = is_iterable_type(arg_type) and (
            isinstance(arg_type.dtype, types.Number) or arg_type.dtype == types.bool_
        )
        if is_numeric_scalar or is_numeric_iterable:
            return arg_type, None
        return (
            arg_type,
            "must be a numeric scalar or Series, Index, Array, List or Tuple with numeric data",
        )

    def explain_arg(self, context: dict[str, pt.Any]) -> str:  # pragma: no cover
        return "must be a numeric scalar or Series, Index, Array, List, or Tuple with numeric data"


class AnySeriesArgumentChecker(ArgumentTypeChecker):
    """
    Argument checker for explicitly stating/documenting Series with any data
    are supported.
    """

    def __init__(self, arg_name: str, is_self: bool | None = False):
        """Initialize checker and set argument name.

        Args:
            arg_name: The name of the argument.
            is_self: True if the argument should appear as "self" in user
                    facing error messages. Defaults to False.
        """
        self.is_self = is_self
        self.arg_name = "self" if self.is_self else arg_name

    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        if not isinstance(arg_type, SeriesType):
            return arg_type, "must be a Series"
        return arg_type, None

    def explain_arg(self, context: dict[str, pt.Any]) -> str:  # pragma: no cover
        return "all Series types supported"


class DatetimeLikeSeriesArgumentChecker(AnySeriesArgumentChecker):
    """
    Checker for documenting methods/attributes found in Series.dt.
    """

    def __init__(
        self,
        arg_name: str,
        is_self: bool | None = False,
        type: str | None = "any",
    ):
        """Initialize DatetimeLikeSeriesArgumentChecker with the subset of
        dt64/td64 types to check.

        Args:
            arg_name: The name of the argument.
            is_self: True if the argument should appear as "self" in user
                    facing error messages. Defaults to False.
            type: The subset of datetimelike types to accept. Options are
                "datetime", "timedelta" or "any" for both datetime and
                timedelta. Defaults to "any".
        """
        super().__init__(arg_name, is_self)
        self.type = type

        # any: datetime or timedelta types accepted
        assert self.type in ["any", "datetime", "timedelta"]

    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        """
        Check that arg_type is a Series of valid datetimelike data.
        """
        # Access underlying Series type for XMethodType (using getattr to avoid the circular import)
        series_type = getattr(arg_type, "stype", arg_type)

        if (
            self.type in ["any", "timedelta"] and is_timedelta64_series_typ(series_type)
        ) or (self.type in ["any", "datetime"] and is_dt64_series_typ(series_type)):
            return series_type, None

        if self.type == "any":
            supported_types = "datetime64 or timedelta64"
        else:
            supported_types = f"{self.type}64"

        return series_type, f"must be a Series of {supported_types} data"

    def explain_arg(self, context: dict[str, pt.Any]) -> str:  # pragma: no cover
        supported_types = (
            "`datetime64` or `timedelta64`"
            if self.type == "any"
            else f"`{self.type}64`"
        )
        return f"must be a Series of {supported_types} data"


class NumericSeriesArgumentChecker(AnySeriesArgumentChecker):
    """
    For Series Arguments that require numeric data (Float or Integer).
    """

    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        if not isinstance(arg_type, SeriesType) or not isinstance(
            arg_type.dtype, types.Number
        ):
            return arg_type, "must be a Series of Float or Integer data"
        return arg_type, None

    def explain_arg(self, context: dict[str, pt.Any]) -> str:  # pragma: no cover
        return "must be a Series of `Integer` or `Float` data"


class StringSeriesArgumentChecker(AnySeriesArgumentChecker):
    """
    For Series arguments that require String data.
    """

    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        """
        Check that the underlying data of Seires is a valid string type.
        """
        # Access underlying Series type for XMethodType (using getattr to avoid the circular import)
        series_type = getattr(arg_type, "stype", arg_type)
        if not (
            isinstance(series_type, SeriesType) and is_str_arr_type(series_type.data)
        ):
            return series_type, "must be a Series of String data"
        return series_type, None

    def explain_arg(self, context: dict[str, pt.Any]) -> str:
        return "must be a Series of `String` data"


class AnyArgumentChecker(ArgumentTypeChecker):
    """
    Dummy class for overload attribute that allows all types.
    """

    def __init__(self, arg_name: str, is_self: bool | None = False):
        self.is_self = is_self
        self.arg_name = "self" if self.is_self else arg_name

    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        return arg_type, None

    def explain_arg(self, context: dict[str, pt.Any]) -> str:
        return "supported on all datatypes"


class OptionalArgumentChecker(ArgumentTypeChecker):
    """
    Checks that arguments either are None or are valid according to another
    argument checker.
    """

    def __init__(self, arg_checker: ArgumentTypeChecker):
        self.arg_checker = arg_checker

    @property
    def arg_name(self):
        return self.arg_checker.arg_name

    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        if is_overload_none(arg_type):
            return None, None

        arg_type, err_str = self.arg_checker.check_arg(context, arg_type)
        if err_str is None:
            return arg_type, err_str

        return arg_type, f"{err_str}, or it can be None"

    def explain_arg(self, context: dict[str, pt.Any]) -> str:
        arg_description = self.arg_checker.explain_arg(context)
        return f"(optional, defaults to `None`) {arg_description}"


class GenericArgumentChecker(ArgumentTypeChecker):
    """
    Generic Argument type checker that accepts custom logic for check and
    explain.
    """

    def __init__(
        self,
        arg_name: str,
        check_fn: pt.Callable[[dict[str, pt.Any], pt.Any], tuple[pt.Any, str | None]],
        explain_fn: pt.Callable[[dict[str, pt.Any]], str],
        is_self: bool | None = False,
    ):
        """Initialize a GenericArgumentChecker with custom check and explain
        logic.

        Args:
            arg_name: The name of the argument.
            check_fn: a lambda which accepts a context containing all
                previously processed arguments and a type and returns a tuple
                where the first value contains the argument type either
                unchanged or after some transformation. The second value is a
                string description of any errors that occured (or None if
                there were no errors).
            explain_fn: a lambda that accepts a context containing all
                previously processed arguments and returns a string
                description of the typing rules for the argument.
            is_self: True if the argument should appear as "self" in user
                facing error messages. Defaults to False.
        """
        self.is_self = is_self
        self.arg_name = "self" if self.is_self else arg_name
        self.check_fn = check_fn
        self.explain_fn = explain_fn

    def check_arg(
        self, context: dict[str, pt.Any], arg_type: pt.Any
    ) -> tuple[pt.Any, str | None]:
        arg_type, err_str = self.check_fn(context, arg_type)
        return arg_type, err_str

    def explain_arg(self, context: dict[str, pt.Any]) -> str:  # pragma: no cover
        return self.explain_fn(context)


class OverloadArgumentsChecker:
    """
    Class for handling the orchestration for multiple argument checkers for
    methods/functions.
    """

    def __init__(self, argument_checkers: list[ArgumentTypeChecker]):
        self.argument_checkers = {
            arg_checker.arg_name: arg_checker for arg_checker in argument_checkers
        }
        self.context = {}

    def set_context(self, key: str, value: pt.Any):
        """
        Updates the type information of *key* in the Checker's internal
        context.
        """
        self.context.update({key: value})

    def check_args(self, path: str, arg_types: dict[str, pt.Any]):
        """Checks that an object satisfies the requirements to get an.

        Args:
            path: The path for the function or method as it should be
                displayed to users in error messages e.g. "pd.Series.pow()".
            arg_types: A dictionary mapping argument's names to their types.

        Raises:
            BodoError: If the type of the object trying to access this
                attribute is invalid.
        """
        for arg_name, typ in arg_types.items():
            if arg_name in self.argument_checkers:
                arg_checker = self.argument_checkers[arg_name]
                new_arg_type, err_str = arg_checker.check_arg(self.context, typ)
                if err_str is not None:
                    raise BodoError(
                        f"{path}: '{arg_name}' {err_str}. Got: {new_arg_type}"
                    )
                self.set_context(arg_name, new_arg_type)

    def explain_args(self) -> dict[str, str]:
        """
        Creates a dictionary mapping argument names to their description.
        """
        return {
            arg_name: arg_checker.explain_arg(self.context)
            for arg_name, arg_checker in self.argument_checkers.items()
        }


class OverloadAttributeChecker(OverloadArgumentsChecker):
    """
    Class for managing the argument check and documentation for attributes.
    """

    def __init__(self, argument_checker: ArgumentTypeChecker):
        self.argument_checker = argument_checker
        self.context = {}

    def check_args(self, path: str, arg_type: pt.Any):
        """Checks that an object satisfies the requirements to get an
        attribute.

        Args:
            path: The path for the attribute as it should be displayed to
                users in error messages e.g. "pd.Series.str".
            arg_type: The type of the object that we are trying to get this
                attribute of.

        Raises:
            BodoError: If the type of the object trying to access this
                attribute is invalid.
        """
        new_arg_type, err_str = self.argument_checker.check_arg(self.context, arg_type)
        if err_str is not None:
            raise BodoError(f"{path}: input {err_str}. Got: {new_arg_type}")
        self.set_context(self.argument_checker.arg_name, new_arg_type)

    def explain_args(self) -> str:
        """
        Creates a descriptions of the requirements on the type of the object in
        order for this attribute to have a valid implementation.
        """
        return self.argument_checker.explain_arg(self.context)
