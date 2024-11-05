import numpy as np
import pandas as pd
from llvmlite import binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic, models, overload, register_model

import bodo
import bodosql
from bodo.libs.array import (
    array_info_type,
    array_to_info,
    delete_info,
    info_to_array,
    py_table_to_cpp_table,
)
from bodo.utils.typing import (
    MetaType,
    is_tuple_like_type,
    raise_bodo_error,
    unwrap_typeref,
)
from bodo.utils.utils import numba_to_c_array_types, numba_to_c_types

# javascript_udf_cpp is only built when BUILD_WITH_V8 is set so
# we need to check if the module is available
javascript_udf_enabled = True
try:
    from bodo.ext import javascript_udf_cpp
except ImportError:
    javascript_udf_enabled = False

if javascript_udf_enabled:  # pragma: no cover
    ll.add_symbol(
        "create_javascript_udf_py_entry",
        javascript_udf_cpp.create_javascript_udf_py_entry,
    )
    ll.add_symbol(
        "delete_javascript_udf_py_entry",
        javascript_udf_cpp.delete_javascript_udf_py_entry,
    )
    ll.add_symbol(
        "execute_javascript_udf_py_entry",
        javascript_udf_cpp.execute_javascript_udf_py_entry,
    )


class JavaScriptFunctionType(types.Type):
    """Type for C++ JavaScript UDF function pointer"""

    def __init__(self, return_type):  # pragma: no cover
        self.return_type = return_type
        super().__init__(name=f"JavaScriptFunctionType(return_type={return_type})")


register_model(JavaScriptFunctionType)(models.OpaqueModel)


@intrinsic
def _create_javascript_udf(
    typing_context,
    output_type_ref,
    body,
    body_len,
    argnames,
    return_arr_arr_type,
    return_arr_c_type,
    size_return_array_type,
):  # pragma: no cover
    """Wrapper to call the C++ function create_javascript_udf_py_entry"""
    output_type = unwrap_typeref(output_type_ref)

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # body
                lir.IntType(32),  # body_len
                lir.IntType(8).as_pointer(),  # argnames
                lir.IntType(8).as_pointer(),  # return_arr_arr_type
                lir.IntType(8).as_pointer(),  # return_arr_c_type
                lir.IntType(32),  # size_return_array_type
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="create_javascript_udf_py_entry"
        )
        # Call the C++ function with all args except the first one (output_type_ref)
        ret = builder.call(fn_tp, args[1:])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = output_type(
        output_type_ref,
        types.voidptr,
        types.int32,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.int32,
    )
    return sig, codegen


def create_javascript_udf(body, argnames, return_type):  # pragma: no cover
    pass


@overload(create_javascript_udf)
def overload_create_javascript_udf(body, argnames, return_type):  # pragma: no cover
    """Overload for create_javascript_udf, ensures we have the cpp extension, checks the input types and calls the intrinsic"""
    if not javascript_udf_enabled:
        raise_bodo_error(
            "JavaScript UDF support is only available on Bodo Platform. https://docs.bodo.ai/latest/quick_start/quick_start_platform/"
        )

    body_type = unwrap_typeref(body)
    if not (isinstance(body_type, MetaType) and isinstance(body_type.meta, str)):
        raise_bodo_error(f"Expected body to be type MetaType(string), got: {body_type}")

    argnames_type = unwrap_typeref(argnames)
    if not isinstance(argnames_type, MetaType):
        raise_bodo_error(
            f"Expected argnames to be type MetaType(), got: {argnames_type}"
        )
    for i, arg in enumerate(argnames_type.meta):
        arg_type = unwrap_typeref(types.unliteral(arg))
        if not isinstance(arg_type, str):
            raise_bodo_error(
                f"Expected elements of argnames to be type string, got: {arg_type} at idx {i}"
            )

    return_type_type = unwrap_typeref(return_type)
    if not isinstance(return_type_type, types.ArrayCompatible):
        raise_bodo_error(
            f"Expected return_type to be an array type, got {return_type_type}"
        )

    body_str = np.frombuffer(body_type.meta.encode("utf-8"), dtype=np.uint8)
    argnames_series = pd.array(list(argnames_type.meta), dtype=pd.StringDtype())

    return_arr_arr_type = numba_to_c_array_types([return_type_type])
    return_arr_c_type = numba_to_c_types([return_type_type])

    output_type = JavaScriptFunctionType(return_type_type)

    def impl(body, argnames, return_type):  # pragma: no cover
        argnames_arr = array_to_info(argnames_series)
        return _create_javascript_udf(
            output_type,
            body_str.ctypes,
            len(body_str),
            argnames_arr,
            return_arr_arr_type.ctypes,
            return_arr_c_type.ctypes,
            len(return_arr_arr_type),
        )

    return impl


@intrinsic
def _delete_javascript_udf(typing_context, js_func):  # pragma: no cover
    """Wrapper to call the C++ function delete_javascript_udf_py_entry"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="delete_javascript_udf_py_entry"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(js_func)
    return sig, codegen


def delete_javascript_udf(js_func):  # pragma: no cover
    pass


@overload(delete_javascript_udf)
def overload_delete_javascript_udf(js_func):  # pragma: no cover
    """Overload for delete_javascript_udf, ensures we have the cpp extension, checks the input type and calls the intrinsic"""
    if not javascript_udf_enabled:
        raise_bodo_error(
            "JavaScript UDF support is only available on Bodo Platform. https://docs.bodo.ai/latest/quick_start/quick_start_platform/"
        )
    js_func_type = unwrap_typeref(js_func)
    if not isinstance(js_func_type, JavaScriptFunctionType):
        raise_bodo_error(
            f"Expected js_func to be type JavaScriptFunctionType, got {js_func_type}"
        )

    def impl(js_func):  # pragma: no cover
        _delete_javascript_udf(js_func)

    return impl


@intrinsic
def _execute_javascript_udf(typing_context, js_func, js_func_args):  # pragma: no cover
    """Wrapper to call the C++ function execute_javascript_udf_py_entry"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # js_func
                lir.IntType(8).as_pointer(),  # js_func_args
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="execute_javascript_udf_py_entry"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = array_info_type(js_func, js_func_args)
    return sig, codegen


def arrays_to_cpp_table(arrs_tup):  # pragma: no cover
    pass


@overload(arrays_to_cpp_table)
def overload_arrays_to_cpp_table(arrs_tup):
    """
    Converts a collection of arrays into a C++ table
    """
    n_cols = len(arrs_tup)
    kept_cols = MetaType(tuple(range(len(arrs_tup))))
    table_type = bodo.hiframes.table.TableType(tuple(arrs_tup.types))
    arrs_text = [f"arrs_tup[{i}]" for i in range(n_cols)]
    func_text = "def impl(arrs_tup):\n"
    func_text += f"  T = bodo.hiframes.table.logical_table_to_table((), ({', '.join(arrs_text)},), kept_cols, 0)\n"
    func_text += "  return py_table_to_cpp_table(T, table_type)"
    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "bodosql": bodosql,
            "py_table_to_cpp_table": py_table_to_cpp_table,
            "table_type": table_type,
            "kept_cols": kept_cols,
        },
        loc_vars,
    )
    return loc_vars["impl"]


def execute_javascript_udf(js_func, args):  # pragma: no cover
    pass


@overload(execute_javascript_udf)
def overload_execute_javascript_udf(js_func, args):  # pragma: no cover
    """Overload for execute_javascript_udf, ensures we have the cpp extension,
    checks the input types and calls the intrinsic. Converts ouput to scalars
    if no args are provided or all arguments are scalars. If any of the arguments
    are arrays, all scalars are broadcasted to arrays of that size and the result
    is an arary of the same length. It is assumed that all array arguments passed
    in are of the same length.
    """

    if not javascript_udf_enabled:
        raise_bodo_error(
            "JavaScript UDF support is only available on Bodo Platform. https://docs.bodo.ai/latest/quick_start/quick_start_platform/"
        )

    js_func_type = unwrap_typeref(js_func)
    if not isinstance(js_func, JavaScriptFunctionType):
        raise_bodo_error(
            f"Expected js_func to be type JavaScriptFunctionType, got {js_func_type}"
        )

    args_type = unwrap_typeref(args)
    if not is_tuple_like_type(args_type):
        raise_bodo_error(f"Expected args to be type tuple, got {args_type}")

    # Used to keep track of which argument index is an array so we can broadcast scalars
    # to use that length
    first_array_arg = -1
    # Boolean to track whether all of the inputs (and therefore the output) are scalar
    is_scalar = True
    # Booleans indicating whether each individual argument is scalar
    scalar_args = []
    # Strings specifying how to get each argument as an array
    args_text = []
    for i, typ in enumerate(args_type):
        if bodo.utils.utils.is_array_typ(typ):
            first_array_arg = i
            is_scalar = False
            scalar_args.append(False)
            args_text.append(f"args[{i}]")
        else:
            scalar_args.append(True)
            # length_arg will be defined to be the length of the first array argument
            args_text.append(f"coerce_scalar_to_array(args[{i}], length_arg, unknown)")

    ret_type = js_func_type.return_type

    func_text = "def impl(js_func, args):\n"
    if len(args_type) == 0:
        # If there are no arguments, pass in a nullptr for the args
        func_text += "  args_table = None\n"
    else:
        # Otherwise, place all of them in a combined buffer. If any of them were scalars,
        # add a statement to indicate what length to use
        if any(scalar_args):
            if all(scalar_args):
                func_text += "  length_arg = 1\n"
            else:
                func_text += f"  length_arg = len(args[{first_array_arg}])\n"
        func_text += f"  args_table = arrays_to_cpp_table(({','.join(args_text)},))\n"
    # Run the UDF and extract the result array
    func_text += "  ret_info = _execute_javascript_udf(js_func, args_table)\n"
    func_text += "  ret_array = info_to_array(ret_info, ret_type)\n"
    func_text += "  delete_info(ret_info)\n"
    # If all of the inputs were scalars, return the singleton output. Otherwise, return
    # the entire array output.
    if is_scalar:
        func_text += "  return None if bodo.libs.array_kernels.isna(ret_array, 0) else ret_array[0]"
    else:
        func_text += "  return ret_array"

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "bodosql": bodosql,
            "ret_type": ret_type,
            "_execute_javascript_udf": _execute_javascript_udf,
            "unknown": types.unknown,
            "coerce_scalar_to_array": bodo.utils.conversion.coerce_scalar_to_array,
            "info_to_array": info_to_array,
            "array_to_info": array_to_info,
            "delete_info": delete_info,
            "np": np,
            "arrays_to_cpp_table": arrays_to_cpp_table,
        },
        loc_vars,
    )
    return loc_vars["impl"]
