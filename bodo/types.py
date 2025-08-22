import numba
from numba.core.types import *  # noqa

datetime64ns = numba.core.types.NPDatetime("ns")
timedelta64ns = numba.core.types.NPTimedelta("ns")
