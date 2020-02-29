# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Analysis and transformation for HDF5 support.
"""
import types as pytypes  # avoid confusion with numba.types

import numba
from numba import ir, analysis, types, config, numpy_support
from numba.ir_utils import (
    mk_unique_var,
    replace_vars_inner,
    find_topo_order,
    find_callname,
    guard,
    require,
    get_definition,
    build_definitions,
    find_const,
    compile_to_numba_ir,
    replace_arg_nodes,
)

import numpy as np

import bodo
import bodo.io
from bodo.io import h5_api
from bodo.utils.utils import debug_prints
from bodo.utils.transform import find_str_const


class H5_IO:
    """analyze and transform hdf5 calls"""

    def __init__(self, func_ir, _locals, flags, arg_types, reverse_copies):
        self.func_ir = func_ir
        self.locals = _locals
        self.flags = flags
        self.arg_types = arg_types
        self.reverse_copies = reverse_copies

    def handle_possible_h5_read(self, assign, lhs, rhs):
        tp = self._get_h5_type(lhs, rhs)
        if tp is not None:
            dtype_str = str(tp.dtype)
            func_text = "def _h5_read_impl(dset, index):\n"
            # TODO: index arg?
            func_text += "  arr = bodo.io.h5_api.h5_read_dummy(dset, {}, '{}', index)\n".format(
                tp.ndim, dtype_str
            )
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            _h5_read_impl = loc_vars["_h5_read_impl"]
            f_block = compile_to_numba_ir(
                _h5_read_impl, {"bodo": bodo}
            ).blocks.popitem()[1]
            index_var = rhs.index if rhs.op == "getitem" else rhs.index_var
            replace_arg_nodes(f_block, [rhs.value, index_var])
            nodes = f_block.body[:-3]  # remove none return
            nodes[-1].target = assign.target
            return nodes
        return None

    def _get_h5_type(self, lhs, rhs):
        tp = self._get_h5_type_locals(lhs)
        if tp is not None:
            return tp
        return guard(self._infer_h5_typ, rhs)

    def _infer_h5_typ(self, rhs):
        # infer the type if it is of the from f['A']['B'][:] or f['A'][b,:]
        # with constant filename
        # TODO: static_getitem has index_var for sure?
        # make sure it's slice, TODO: support non-slice like integer
        require(rhs.op in ("getitem", "static_getitem"))
        # XXX can't know the type of index here especially if it is bool arr
        # make sure it is not string (we're not in the middle a select chain)
        index_var = rhs.index if rhs.op == "getitem" else rhs.index_var
        index_val = guard(find_const, self.func_ir, index_var)
        require(not isinstance(index_val, str))
        # index_def = get_definition(self.func_ir, index_var)
        # require(isinstance(index_def, ir.Expr) and index_def.op == 'call')
        # require(find_callname(self.func_ir, index_def) == ('slice', 'builtins'))
        # collect object names until the call
        val_def = rhs
        obj_name_list = []
        while True:
            val_def = get_definition(self.func_ir, val_def.value)
            require(isinstance(val_def, ir.Expr))
            if val_def.op == "call":
                return self._get_h5_type_file(val_def, obj_name_list)

            # object_name should be constant str
            require(val_def.op in ("getitem", "static_getitem"))
            val_index_var = (
                val_def.index if val_def.op == "getitem" else val_def.index_var
            )
            obj_name = find_str_const(
                self.func_ir, val_index_var, arg_types=self.arg_types
            )
            obj_name_list.append(obj_name)

    def _get_h5_type_file(self, val_def, obj_name_list):
        require(len(obj_name_list) > 0)
        require(find_callname(self.func_ir, val_def) == ("File", "h5py"))
        require(len(val_def.args) > 0)
        f_name = find_str_const(self.func_ir, val_def.args[0], arg_types=self.arg_types)
        obj_name_list.reverse()

        import h5py

        f = h5py.File(f_name, "r")
        obj = f
        for obj_name in obj_name_list:
            obj = obj[obj_name]
        require(isinstance(obj, h5py.Dataset))
        ndims = len(obj.shape)
        numba_dtype = numba.numpy_support.from_dtype(obj.dtype)
        f.close()
        return types.Array(numba_dtype, ndims, "C")

    def _get_h5_type_locals(self, varname):
        # TODO: can we do this without reverse_copies?
        # TODO: if copy propagation is done, varname itself should be checked
        new_name = self.reverse_copies.get(varname, None)
        typ = self.locals.pop(new_name, None)
        if typ is None and new_name is not None:
            typ = self.flags.h5_types.get(new_name, None)
        return typ
