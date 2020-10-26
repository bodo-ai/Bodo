""" Support for Spark parity functions in objmode """
import zlib

from bodo.utils.typing import gen_objmode_func_overload

#### zlib.crc32 support ####
gen_objmode_func_overload(zlib.crc32, "uint32")
