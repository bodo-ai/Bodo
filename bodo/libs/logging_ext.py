"""
JIT support for Python's logging module
"""
import logging

import numba
from numba.core import types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import (
    AttributeTemplate,
    bound_function,  # noqa
    infer_getattr,
    signature,
)
from numba.extending import (
    NativeValue,
    box,
    models,
    register_model,
    typeof_impl,
    unbox,
)

import bodo
from bodo.utils.typing import (
    gen_objmode_attr_overload,
)


class LoggingLoggerType(types.Type):
    """JIT type for logging.Logger and logging.RootLogger"""

    def __init__(self, is_root=False):
        # TODO: flag is unused, remove?
        self.is_root = is_root
        super(LoggingLoggerType, self).__init__(
            name=f"LoggingLoggerType(is_root={is_root})"
        )


@typeof_impl.register(logging.RootLogger)
@typeof_impl.register(logging.Logger)
def typeof_logging(val, c):
    # logging.RootLogger is the child of logging.Logger
    if isinstance(val, logging.RootLogger):
        return LoggingLoggerType(is_root=True)
    else:
        return LoggingLoggerType(is_root=False)


register_model(LoggingLoggerType)(models.OpaqueModel)


@box(LoggingLoggerType)
def box_logging_logger(typ, val, c):
    c.pyapi.incref(val)
    return val


@unbox(LoggingLoggerType)
def unbox_logging_logger(typ, obj, c):
    c.pyapi.incref(obj)
    return NativeValue(obj)


@lower_constant(LoggingLoggerType)
def lower_constant_logger(context, builder, ty, pyval):
    pyapi = context.get_python_api(builder)
    return pyapi.unserialize(pyapi.serialize_object(pyval))


# Generate attribute overloads for LoggerLoggingType
gen_objmode_attr_overload(LoggingLoggerType, "level", None, types.int64)
gen_objmode_attr_overload(LoggingLoggerType, "name", None, "unicode_type")
gen_objmode_attr_overload(LoggingLoggerType, "propagate", None, types.boolean)
gen_objmode_attr_overload(LoggingLoggerType, "disabled", None, types.boolean)
gen_objmode_attr_overload(LoggingLoggerType, "parent", None, LoggingLoggerType())
gen_objmode_attr_overload(
    LoggingLoggerType, "root", None, LoggingLoggerType(is_root=True)
)


@infer_getattr
class LoggingLoggerAttribute(AttributeTemplate):
    """
    Template used for typing logging.Logger and logging.RootLogger attributes.
    This is used for functions that cannot use a traditional overload due to *args and **kwargs limitations in overloads.
    """

    key = LoggingLoggerType

    def _resolve_helper(self, logger_typ, args, kws):
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
        arg_types = (logger_typ,) + args + tuple(kws.values())
        return signature(logger_typ, arg_types).replace(pysig=pysig)

    func_names = (
        "debug",
        "warning",
        "warn",
        "info",
        "error",
        "exception",
        "critical",
        "log",
        "setLevel",
    )

    for logger in ("logging.Logger", "logging.RootLogger"):
        for func_name in func_names:
            resolve_text = f"""@bound_function("{logger}.{func_name}")\n"""
            resolve_text += f"def resolve_{func_name}(self, logger_typ, args, kws):\n"
            resolve_text += "    return self._resolve_helper(logger_typ, args, kws)"
            exec(resolve_text)


logging_logger_unsupported_attrs = {
    "filters",
    "handlers",
    "manager",
}


logging_logger_unsupported_methods = {
    "addHandler",
    "callHandlers",
    "fatal",
    "findCaller",
    "getChild",
    "getEffectiveLevel",
    "handle",
    "hasHandlers",
    "isEnabledFor",
    "makeRecord",
    "removeHandler",
}


def _install_logging_logger_unsupported_objects():
    """install overload that raises BodoError for unsupported logger.Logger methods"""

    # Installs overload for logger.Logger
    for attr_name in logging_logger_unsupported_attrs:
        full_name = "logging.Logger." + attr_name
        bodo.overload_unsupported_attribute(LoggingLoggerType, attr_name, full_name)

    for fname in logging_logger_unsupported_methods:
        full_name = "logging.Logger." + fname
        bodo.overload_unsupported_method(LoggingLoggerType, fname, full_name)


_install_logging_logger_unsupported_objects()
