"""Support for Spark parity functions in objmode"""

import math
import zlib

from bodo.utils.typing import gen_objmode_func_overload

#### zlib.crc32 support ####
gen_objmode_func_overload(zlib.crc32, "uint32")


#### math.factorial support ####
# Python factorial is a fast divide and conquer algorithm written in C.
# Use of factorial is probably uncommon, so we will use object mode due to
# the complexity needed. Can convert to a native implementation if requested.
# https://hg.python.org/cpython/file/d42f264f291e/Modules/mathmodule.c#l1218
gen_objmode_func_overload(math.factorial, "int64")

try:
    import scipy.special

    gen_objmode_func_overload(scipy.special.factorial, "int64")
except ImportError:
    pass
