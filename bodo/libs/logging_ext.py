"""
JIT support for Python's logging module
"""
import logging

import numba
from numba.core import types
from numba.core.typing.templates import (
    AttributeTemplate,
    bound_function,
    infer_getattr,
    signature,
)
from numba.extending import (
    NativeValue,
    box,
    models,
    overload_attribute,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)

from bodo.utils.typing import create_unsupported_overload


class LoggingRootLoggerType(types.Type):
    """JIT type for logging.RootLogger"""

    def __init__(self):
        super(LoggingRootLoggerType, self).__init__(name="LoggingRootLoggerType()")


logging_rootlogger_type = LoggingRootLoggerType()


@typeof_impl.register(logging.RootLogger)
def typeof_logging(val, c):
    return logging_rootlogger_type


register_model(LoggingRootLoggerType)(models.OpaqueModel)
types.logging_rootlogger_type = logging_rootlogger_type


@box(LoggingRootLoggerType)
def box_logging_logger(typ, val, c):
    c.pyapi.incref(val)
    return val


@unbox(LoggingRootLoggerType)
def unbox_logging_logger(typ, obj, c):
    c.pyapi.incref(obj)
    return NativeValue(obj)


@infer_getattr
class LoggingRootLoggerAttribute(AttributeTemplate):
    """
    Template used for typing logging.RootLogger attributes. This is used for functions that cannot use a traditional overload due to *args and **kwargs limitations in overloads.
    """

    key = LoggingRootLoggerType

    @bound_function("logging.RootLogger.info")
    def resolve_info(self, string_typ, args, kws):
        kws = dict(kws)
        # add dummy default value for kws to avoid errors
        arg_names = ", ".join("e{}".format(i) for i in range(len(args)))
        if arg_names:
            arg_names += ", "
        kw_names = ", ".join("{} = ''".format(a) for a in kws.keys())
        func_text = f"def format_stub(string, {arg_names} {kw_names}):\n"
        func_text += "    pass\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        format_stub = loc_vars["format_stub"]
        pysig = numba.core.utils.pysignature(format_stub)
        arg_types = (string_typ,) + args + tuple(kws.values())
        return signature(string_typ, arg_types).replace(pysig=pysig)


logging_rootlogger_unsupported_attrs = {
    "disabled",
    "filters",
    "handlers",
    "level",
    "manager",
    "name",
    "parent",
    "propagate",
    "root",
}


logging_rootlogger_unsupported_methods = {
    "addHandler",
    "callHandlers",
    "critical",
    "debug",
    "error",
    "exception",
    "fatal",
    "findCaller",
    "getChild",
    "getEffectiveLevel",
    "handle",
    "hasHandlers",
    "info",
    "isEnabledFor",
    "log",
    "makeRecord",
    "removeHandler",
    "setLevel",
    "warn",
    "warning",
}


def _install_logging_rootlogger_unsupported_objects():
    """install overload that raises BodoError for unsupported logger.RootLogger methods"""

    for attr_name in logging_rootlogger_unsupported_attrs:
        full_name = "logging.RootLogger." + attr_name
        overload_attribute(LoggingRootLoggerType, attr_name)(
            create_unsupported_overload(full_name)
        )

    for fname in logging_rootlogger_unsupported_methods:
        full_name = "logging.Rootlogger." + fname
        overload_method(LoggingRootLoggerType, fname)(
            create_unsupported_overload(full_name)
        )


_install_logging_rootlogger_unsupported_objects()
