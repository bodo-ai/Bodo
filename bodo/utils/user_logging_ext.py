from numba.extending import overload

from bodo.ir.object_mode import no_warning_objmode
from bodo.user_logging import get_verbose_level, log_message


@overload(get_verbose_level)
def overload_get_verbose_level():
    """
    Implementation of get_verbose_level that can be called from JIT.
    """

    def impl():  # pragma: no cover
        with no_warning_objmode(verbose_level="int64"):
            verbose_level = get_verbose_level()
        return verbose_level

    return impl


@overload(log_message)
def overload_log_message(header, msg):
    """
    Implementation of log_message that can be called from JIT.
    This implementation doesn't support additional arguments.
    """

    def impl(header, msg):  # pragma: no cover
        with no_warning_objmode():
            log_message(header, msg)

    return impl
