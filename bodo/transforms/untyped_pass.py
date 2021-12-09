# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
transforms the IR to remove features that Numba's type inference cannot support
such as non-uniform dictionary input of `pd.DataFrame({})`.
"""
import types as pytypes
import warnings
import itertools
import datetime
import pandas as pd
import numpy as np
from urllib.parse import urlparse

import numba
from numba.core import ir, ir_utils, types

from numba.core.ir_utils import (
    mk_unique_var,
    find_topo_order,
    dprint_func_ir,
    find_const,
    GuardException,
    compile_to_numba_ir,
    replace_arg_nodes,
    find_callname,
    guard,
    get_definition,
    build_definitions,
    compute_cfg_from_blocks,
)


import bodo
import bodo.io
from bodo.io import h5
from bodo.utils.utils import is_assign, is_call, is_expr
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.hiframes.pd_index_ext import RangeIndexType
import bodo.ir
import bodo.ir.aggregate
import bodo.ir.join
import bodo.ir.sort
from bodo.ir import csv_ext
from bodo.ir import sql_ext
from bodo.ir import json_ext
from bodo.io.parquet_pio import ParquetHandler

from bodo.hiframes.pd_categorical_ext import PDCategoricalDtype, CategoricalArrayType
import bodo.hiframes.pd_dataframe_ext
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.utils.transform import (
    get_const_value,
    get_const_value_inner,
    update_node_list_definitions,
    compile_func_single_block,
    get_call_expr_arg,
    gen_const_tup,
    fix_struct_return,
    set_call_expr_arg,
)
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    raise_bodo_error,
    to_nullable_type,
    FileInfo,
)
from bodo.utils.utils import check_java_installation


# dummy sentinel singleton to designate constant value not found for variable
class ConstNotFound:
    pass


CONST_NOT_FOUND = ConstNotFound()


class UntypedPass:
    """
    Transformations before typing to enable type inference.
    This pass transforms the IR to remove operations that cannot be handled in Numba's
    type inference due to complexity such as pd.read_csv().
    """

    def __init__(self, func_ir, typingctx, args, _locals, metadata, flags):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.args = args
        self.locals = _locals
        self.metadata = metadata
        self.flags = flags
        # TODO: remove this? update _the_max_label just in case?
        ir_utils._the_max_label.update(max(func_ir.blocks.keys()))

        self.arrow_tables = {}
        self.pq_handler = ParquetHandler(func_ir, typingctx, args, _locals)
        self.h5_handler = h5.H5_IO(self.func_ir, _locals, flags, args)
        # save names of arguments and return values to catch invalid dist annotation
        self._arg_names = set()
        self._return_varnames = set()
        self._has_h5py = bodo.utils.utils.has_supported_h5py()

    def run(self):
        """run untyped pass transform"""
        dprint_func_ir(self.func_ir, "starting untyped pass")
        self._handle_metadata()
        blocks = self.func_ir.blocks
        # call build definition since rewrite pass doesn't update definitions
        # e.g. getitem to static_getitem in test_column_list_select2
        self.func_ir._definitions = build_definitions(blocks)
        # remove dead branches to avoid unnecessary typing issues
        remove_dead_branches(self.func_ir)
        # topo_order necessary since df vars need to be found before use
        topo_order = find_topo_order(blocks)
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        while work_list:
            label, block = work_list.pop()
            self._working_body = []
            for inst in block.body:
                out_nodes = [inst]

                if isinstance(inst, ir.Assign):
                    self.func_ir._definitions[inst.target.name].remove(inst.value)
                    out_nodes = self._run_assign(inst, label)
                elif isinstance(inst, ir.Return):
                    out_nodes = self._run_return(inst)

                assert isinstance(out_nodes, list)
                # TODO: fix scope/loc
                self._working_body.extend(out_nodes)
                update_node_list_definitions(out_nodes, self.func_ir)

            blocks[label].body = self._working_body

        self.func_ir.blocks = ir_utils.simplify_CFG(self.func_ir.blocks)
        # self.func_ir._definitions = build_definitions(blocks)
        # XXX: remove dead here fixes h5 slice issue
        # iterative remove dead to make sure all extra code (e.g. df vars) is removed
        # while remove_dead(blocks, self.func_ir.arg_names, self.func_ir):
        #     pass
        self.func_ir._definitions = build_definitions(blocks)
        # return {"A": 1, "B": 2.3} -> return struct((1, 2.3), ("A", "B"))
        fix_struct_return(self.func_ir)
        dprint_func_ir(self.func_ir, "after untyped pass")

        # raise a warning if a variable that is not an argument or return value has a
        # "distributed" annotation
        extra_vars = (
            self.metadata["distributed"] - self._return_varnames - self._arg_names
        )
        if extra_vars and bodo.get_rank() == 0:
            warnings.warn(
                BodoWarning(
                    "Only function arguments and return values can be specified as "
                    "distributed. Ignoring the flag for variables: {}.".format(
                        extra_vars
                    )
                )
            )
        return

    def _run_assign(self, assign, label):
        lhs = assign.target.name
        rhs = assign.value

        # pass pivot values to df.pivot_table() calls using a meta
        # variable passed as argument. The meta variable's type
        # is set to MetaType with pivot values baked in.
        if lhs in self.flags.pivots:
            pivot_values = self.flags.pivots[lhs]
            # put back the definition removed earlier
            self.func_ir._definitions[lhs].append(rhs)
            pivot_call = guard(get_definition, self.func_ir, lhs)
            assert pivot_call is not None
            meta_var = ir.Var(assign.target.scope, mk_unique_var("pivot_meta"), rhs.loc)
            meta_assign = ir.Assign(ir.Const(0, rhs.loc), meta_var, rhs.loc)
            self._working_body.insert(0, meta_assign)
            pivot_call.kws = list(pivot_call.kws)
            pivot_call.kws.append(("_pivot_values", meta_var))
            self.locals[meta_var.name] = bodo.utils.typing.MetaType(pivot_values)

        # save arg name to catch invalid dist annotations
        if isinstance(rhs, ir.Arg):
            self._arg_names.add(rhs.name)

        # Throw proper error if the user has installed an unsupported HDF5 version
        # see [BE-1382]
        # TODO(ehsan): the code may not use "h5py" directly ("from h5py import File")
        # but that's rare and not high priority at this time
        if (
            not self._has_h5py
            and isinstance(rhs, (ir.Const, ir.Global, ir.FreeVar))
            and isinstance(rhs.value, pytypes.ModuleType)
            and rhs.value.__name__ == "h5py"
        ):  # pragma: no cover
            raise BodoError("Bodo requires HDF5 1.10 for h5py support", rhs.loc)

        if isinstance(rhs, ir.Expr):
            if rhs.op == "call":
                return self._run_call(assign, label)

            if rhs.op in ("getitem", "static_getitem"):
                return self._run_getitem(assign, rhs, label)

            if rhs.op == "getattr":
                return self._run_getattr(assign, rhs)

            if rhs.op == "make_function":
                # HACK make globals available for typing in series.map()
                rhs.globals = self.func_ir.func_id.func.__globals__

        # handle copies lhs = f
        if isinstance(rhs, ir.Var) and rhs.name in self.arrow_tables:
            self.arrow_tables[lhs] = self.arrow_tables[rhs.name]
            # enables function matching without node in IR
            self.func_ir._definitions[lhs].append(rhs)
            return []
        return [assign]

    def _run_getattr(self, assign, rhs):
        """transform ir.Expr.getattr nodes if necessary"""
        lhs = assign.target.name
        val_def = guard(get_definition, self.func_ir, rhs.value)

        # HACK: delete pd.DataFrame({}) nodes to avoid typing errors
        # TODO: remove when dictionaries are implemented and typing works
        if (
            isinstance(val_def, ir.Global)
            and isinstance(val_def.value, pytypes.ModuleType)
            and val_def.value == pd
            and rhs.attr in ("read_csv", "read_parquet", "read_json")
        ):
            # put back the definition removed earlier but remove node
            # enables function matching without node in IR
            self.func_ir._definitions[lhs].append(rhs)
            return []

        if (
            isinstance(val_def, ir.Global)
            and isinstance(val_def.value, pytypes.ModuleType)
            and val_def.value == np
            and rhs.attr == "fromfile"
        ):
            # put back the definition removed earlier but remove node
            self.func_ir._definitions[lhs].append(rhs)
            return []

        # HACK: delete pyarrow.parquet.read_table() to avoid typing errors
        if rhs.attr == "read_table":
            import pyarrow.parquet as pq

            val_def = guard(get_definition, self.func_ir, rhs.value)
            if isinstance(val_def, ir.Global) and val_def.value == pq:
                # put back the definition removed earlier but remove node
                self.func_ir._definitions[lhs].append(rhs)
                return []

        if rhs.value.name in self.arrow_tables and rhs.attr == "to_pandas":
            # put back the definition removed earlier but remove node
            self.func_ir._definitions[lhs].append(rhs)
            return []

        # replace datetime.date.today and datetime.datetime.today with an internal function since class methods
        # are not supported in Numba's typing
        if rhs.attr == "today":
            is_datetime_date_today = False
            is_datetime_datetime_today = False
            if is_expr(val_def, "getattr"):
                if val_def.attr == "date":
                    # Handle global import via getattr
                    mod_def = guard(get_definition, self.func_ir, val_def.value)
                    is_datetime_date_today = (
                        isinstance(mod_def, (ir.Global, ir.FreeVar))
                        and mod_def.value == datetime
                    )
                elif val_def.attr == "datetime":
                    mod_def = guard(get_definition, self.func_ir, val_def.value)
                    is_datetime_datetime_today = (
                        isinstance(mod_def, (ir.Global, ir.FreeVar))
                        and mod_def.value == datetime
                    )
            else:
                # Handle relative imports by checking if the value matches importing from Python
                is_datetime_date_today = (
                    isinstance(val_def, (ir.Global, ir.FreeVar))
                    and val_def.value == datetime.date
                )
                is_datetime_datetime_today = (
                    isinstance(val_def, (ir.Global, ir.FreeVar))
                    and val_def.value == datetime.datetime
                )
            if is_datetime_date_today:
                return compile_func_single_block(
                    eval("lambda: bodo.hiframes.datetime_date_ext.today_impl"),
                    (),
                    assign.target,
                )
            elif is_datetime_datetime_today:
                return compile_func_single_block(
                    eval("lambda: bodo.hiframes.datetime_datetime_ext.today_impl"),
                    (),
                    assign.target,
                )

        if (
            rhs.attr == "from_product"
            and is_expr(val_def, "getattr")
            and val_def.attr == "MultiIndex"
        ):
            val_def.attr = "Index"
            mod_def = guard(get_definition, self.func_ir, val_def.value)
            if isinstance(mod_def, ir.Global) and mod_def.value == pd:
                return compile_func_single_block(
                    eval("lambda: bodo.hiframes.pd_multi_index_ext.from_product"),
                    (),
                    assign.target,
                )

        # replace datetime.date.fromordinal with an internal function since class methods
        # are not supported in Numba's typing
        if (
            rhs.attr == "fromordinal"
            and is_expr(val_def, "getattr")
            and val_def.attr == "date"
        ):
            mod_def = guard(get_definition, self.func_ir, val_def.value)
            if isinstance(mod_def, ir.Global) and mod_def.value == datetime:
                return compile_func_single_block(
                    eval("lambda: bodo.hiframes.datetime_date_ext.fromordinal_impl"),
                    (),
                    assign.target,
                )

        # replace datetime.datedatetime.now with an internal function since class methods
        # are not supported in Numba's typing
        if (
            rhs.attr == "now"
            and is_expr(val_def, "getattr")
            and val_def.attr == "datetime"
        ):
            mod_def = guard(get_definition, self.func_ir, val_def.value)
            if isinstance(mod_def, ir.Global) and mod_def.value == datetime:
                return compile_func_single_block(
                    eval("lambda: bodo.hiframes.datetime_datetime_ext.now_impl"),
                    (),
                    assign.target,
                )

        # replace pd.Timestamp.now with an internal function since class methods
        # are not supported in Numba's typing
        if (
            rhs.attr == "now"
            and is_expr(val_def, "getattr")
            and val_def.attr == "Timestamp"
        ):
            mod_def = guard(get_definition, self.func_ir, val_def.value)
            if isinstance(mod_def, ir.Global) and mod_def.value == pd:
                return compile_func_single_block(
                    eval("lambda: bodo.hiframes.pd_timestamp_ext.now_impl"),
                    (),
                    assign.target,
                )

        # replace datetime.datedatetime.strptime with an internal function since class methods
        # are not supported in Numba's typing
        if (
            rhs.attr == "strptime"
            and is_expr(val_def, "getattr")
            and val_def.attr == "datetime"
        ):
            mod_def = guard(get_definition, self.func_ir, val_def.value)
            if isinstance(mod_def, ir.Global) and mod_def.value == datetime:
                return compile_func_single_block(
                    eval("lambda: bodo.hiframes.datetime_datetime_ext.strptime_impl"),
                    (),
                    assign.target,
                )

        # replace itertools.chain.from_iterable with an internal function since
        #  class methods are not supported in Numba's typing
        if (
            rhs.attr == "from_iterable"
            and is_expr(val_def, "getattr")
            and val_def.attr == "chain"
        ):
            mod_def = guard(get_definition, self.func_ir, val_def.value)
            if isinstance(mod_def, ir.Global) and mod_def.value == itertools:
                return compile_func_single_block(
                    eval("lambda: bodo.utils.typing.from_iterable_impl"),
                    (),
                    assign.target,
                )

        # replace SparkSession.builder since class attributes are not supported in Numba
        if (
            bodo.compiler._pyspark_installed
            and rhs.attr == "builder"
            and isinstance(val_def, ir.Global)
        ):
            from pyspark.sql import SparkSession

            if val_def.value == SparkSession:
                # replace SparkSession global to avoid typing errors
                val_def.value = "dummy"
                return compile_func_single_block(
                    eval("lambda: bodo.libs.pyspark_ext.init_session_builder()"),
                    (),
                    assign.target,
                )

        # replace bytes.fromhex() since class attributes are not supported in Numba
        if (
            rhs.attr == "fromhex"
            and isinstance(val_def, ir.Global)
            and val_def.value == bytes
        ):
            return compile_func_single_block(
                eval("lambda: bodo.libs.binary_arr_ext.bytes_fromhex"),
                (),
                assign.target,
            )

        return [assign]

    def _run_getitem(self, assign, rhs, label):
        # fix type for f['A'][:] dset reads
        if self._has_h5py:
            lhs = assign.target.name
            h5_nodes = self.h5_handler.handle_possible_h5_read(assign, lhs, rhs)
            if h5_nodes is not None:
                return h5_nodes

        return [assign]

    def _run_call(self, assign, label):
        """handle calls and return new nodes if needed"""
        lhs = assign.target
        rhs = assign.value

        # add output type checking/handling to objmode output variables
        func_var_def = guard(get_definition, self.func_ir, rhs.func)
        if isinstance(func_var_def, ir.Const) and isinstance(
            func_var_def.value, numba.core.dispatcher.ObjModeLiftedWith
        ):
            return self._handle_objmode(assign, func_var_def.value)

        func_name = ""
        func_mod = ""
        fdef = guard(find_callname, self.func_ir, rhs)
        if fdef is None:
            # could be make_function from list comprehension which is ok
            func_def = guard(get_definition, self.func_ir, rhs.func)
            if isinstance(func_def, ir.Expr) and func_def.op == "make_function":
                return [assign]
            # since typemap is not available in untyped pass, var.func() is not
            # recognized if var has multiple definitions (e.g. intraday
            # example). find_callname() assumes var could be a module, which
            # isn't an issue since we only match and transform 'drop' and
            # 'sort_values' here for variable in a safe way (TODO test more).
            if is_expr(func_def, "getattr"):
                func_mod = func_def.value
                func_name = func_def.attr
            # ignore objmode block calls
            elif isinstance(func_def, ir.Const) and isinstance(
                func_def.value, numba.core.dispatcher.ObjModeLiftedWith
            ):
                return [assign]
            # input to _bodo_groupby_apply_impl() is a UDF dispatcher
            elif isinstance(func_def, ir.Arg) and isinstance(
                self.args[func_def.index], types.Dispatcher
            ):
                return [assign]
            else:
                warnings.warn("function call couldn't be found for initial analysis")
                return [assign]
        else:
            func_name, func_mod = fdef

        # handling pd.DataFrame() here since input can be constant dictionary
        if fdef == ("DataFrame", "pandas"):
            return self._handle_pd_DataFrame(assign, lhs, rhs, label)

        # handling pd.read_csv() here since input can have constants
        # like dictionaries for typing
        if fdef == ("read_csv", "pandas"):
            return self._handle_pd_read_csv(assign, lhs, rhs, label)

        # handling pd.read_sql() here since input can have constants
        # like dictionaries for typing
        if fdef == ("read_sql", "pandas"):
            return self._handle_pd_read_sql(assign, lhs, rhs, label)

        # handling pd.read_json() here since input can have constants
        # like dictionaries for typing
        if fdef == ("read_json", "pandas"):
            return self._handle_pd_read_json(assign, lhs, rhs, label)

        # handling pd.read_excel() here since typing info needs to be extracted
        if fdef == ("read_excel", "pandas"):
            return self._handle_pd_read_excel(assign, lhs, rhs, label)

        # match flatmap pd.Series(list(itertools.chain(*A))) and flatten
        if fdef == ("Series", "pandas"):
            return self._handle_pd_Series(assign, lhs, rhs)

        # replace pd.NamedAgg() with equivalent tuple to be handled in groupby typing
        if fdef == ("NamedAgg", "pandas"):
            return self._handle_pd_named_agg(assign, lhs, rhs)

        if fdef == ("read_table", "pyarrow.parquet"):
            return self._handle_pq_read_table(assign, lhs, rhs)

        if (
            func_name == "to_pandas"
            and isinstance(func_mod, ir.Var)
            and func_mod.name in self.arrow_tables
        ):
            return self._handle_pq_to_pandas(assign, lhs, rhs, func_mod)

        if fdef == ("read_parquet", "pandas"):
            return self._handle_pd_read_parquet(assign, lhs, rhs)

        if fdef == ("fromfile", "numpy"):
            return self._handle_np_fromfile(assign, lhs, rhs)

        if fdef == ("where", "numpy") and len(rhs.args) == 3:
            return self._handle_np_where(assign, lhs, rhs)

        if fdef == ("where", "numpy") and len(rhs.args) == 1:
            return self._handle_np_where_one_arg(assign, lhs, rhs)

        if fdef == ("BodoSQLContext", "bodosql"):  # pragma: no cover
            return self._handle_bodosql_BodoSQLContext(assign, lhs, rhs, label)

        # add distributed flag input to SparkDataFrame.toPandas() if specified by user
        if (
            bodo.compiler._pyspark_installed
            and func_name == "toPandas"
            and isinstance(func_mod, ir.Var)
            and lhs.name in self.metadata["distributed"]
        ):
            # avoid raising warning since flag is valid
            self._return_varnames.add(lhs.name)
            true_var = ir.Var(lhs.scope, mk_unique_var("true"), lhs.loc)
            rhs.args.append(true_var)
            return [ir.Assign(ir.Const(True, lhs.loc), true_var, lhs.loc), assign]
        return [assign]

    def _handle_objmode(self, assign, objmode_val):
        """
        Add output type checking/handling to objmode output variables.
        Generates check_objmode_output_type() on output variable inside the objmode
        function.
        """
        loc = assign.loc
        scope = assign.target.scope

        # to pass the data type to check_objmode_output_type(), create a variable
        # outside objmode and pass as argument to the objmode call.
        # This allows caching since the type value will be serialized in the binary.

        # unique variable name for type to avoid potential conflicts
        type_name = f"objmode_type{ir_utils.next_label()}"

        # add a new ir.Arg assignment in first block
        first_blk = find_topo_order(objmode_val.func_ir.blocks)[0]
        first_body = objmode_val.func_ir.blocks[first_blk].body
        type_var_in = ir.Var(
            objmode_val.func_ir.blocks[first_blk].scope, type_name, loc
        )
        n_args = objmode_val.func_ir.arg_count
        # assuming the first nodes are ir.Arg
        for i in range(n_args):
            assert is_assign(first_body[i]) and isinstance(
                first_body[i].value, ir.Arg
            ), "invalid objmode format"

        first_body.insert(
            n_args, ir.Assign(ir.Arg(type_name, n_args, loc), type_var_in, loc)
        )

        # generate check_objmode_output_type() call on the return variables
        for block in objmode_val.func_ir.blocks.values():
            last_node = block.terminator
            if isinstance(last_node, ir.Return):
                new_var = ir.Var(
                    block.scope, mk_unique_var("objmode_return"), last_node.loc
                )
                block.body = (
                    block.body[:-1]
                    + compile_func_single_block(
                        eval(
                            f"lambda A, t: bodo.utils.typing.check_objmode_output_type(A, t)"
                        ),
                        [last_node.value, type_var_in],
                        new_var,
                    )
                    + [last_node]
                )
                last_node.value = new_var

        # add new argument to objmode function IR
        new_arg_count = objmode_val.func_ir.arg_count + 1
        new_arg_names = objmode_val.func_ir.arg_names + (type_name,)
        objmode_val.func_ir = objmode_val.func_ir.derive(
            objmode_val.func_ir.blocks, new_arg_count, new_arg_names
        )

        # create type variable outside objmode and pass to the call
        type_var = ir.Var(scope, type_name, loc)
        glb_assign = ir.Assign(
            ir.Global(type_name, objmode_val.output_types, loc), type_var, loc
        )
        assign.value.args = list(assign.value.args) + [type_var]

        return [glb_assign, assign]

    def _handle_np_where(self, assign, lhs, rhs):
        """replace np.where() calls with Bodo's version since Numba's typer assumes
        non-Array types like Series are scalars and produces wrong output type.
        """
        return compile_func_single_block(
            eval("lambda c, x, y: bodo.hiframes.series_impl.where_impl(c, x, y)"),
            rhs.args,
            lhs,
        )

    def _handle_np_where_one_arg(self, assign, lhs, rhs):
        """replace np.where() calls with 1 arg with Bodo's version since
        Numba's typer cannot handle our array types.
        """
        return compile_func_single_block(
            eval("lambda c: bodo.hiframes.series_impl.where_impl_one_arg(c)"),
            rhs.args,
            lhs,
        )

    def _handle_pd_DataFrame(self, assign, lhs, rhs, label):
        """
        Enable typing for dictionary data arg to pd.DataFrame({'A': A}) call.
        Converts constant dictionary to tuple with sentinel if present.
        """
        nodes = [assign]
        kws = dict(rhs.kws)
        data_arg = get_call_expr_arg("pd.DataFrame", rhs.args, kws, 0, "data", "")
        index_arg = get_call_expr_arg("pd.DataFrame", rhs.args, kws, 1, "index", "")

        arg_def = guard(get_definition, self.func_ir, data_arg)

        if isinstance(arg_def, ir.Expr) and arg_def.op == "build_map":
            msg = "DataFrame column names should be constant strings or ints"
            (tup_vars, new_nodes,) = bodo.utils.transform._convert_const_key_dict(
                self.args,
                self.func_ir,
                arg_def,
                msg,
                lhs.scope,
                lhs.loc,
                output_sentinel_tuple=True,
            )
            tup_var = tup_vars[0]
            # replace data arg with dict tuple
            if "data" in kws:
                kws["data"] = tup_var
                rhs.kws = list(kws.items())
            else:
                rhs.args[0] = tup_var

            nodes = new_nodes + nodes
            # HACK replace build_map to avoid inference errors
            # TODO: don't replace if used in other places
            arg_def.op = "build_list"
            arg_def.items = [v[0] for v in arg_def.items]

        # replace range() with pd.RangeIndex() for index argument
        arg_def = guard(get_definition, self.func_ir, index_arg)
        if is_call(arg_def) and guard(find_callname, self.func_ir, arg_def) == (
            "range",
            "builtins",
        ):
            # gen pd.RangeIndex() call
            func_text = "" "def _call_range_index():\n" "    return pd.RangeIndex()\n"

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            f_block = compile_to_numba_ir(
                loc_vars["_call_range_index"], {"pd": pd}
            ).blocks.popitem()[1]
            new_nodes = f_block.body[:-2]
            new_nodes[-1].value.args = arg_def.args
            new_index = new_nodes[-1].target
            # replace index arg
            if "index" in kws:
                kws["index"] = new_index
                rhs.kws = list(kws.items())
            else:
                rhs.args[1] = new_index
            nodes = new_nodes + nodes

        return nodes

    def _handle_bodosql_BodoSQLContext(
        self, assign, lhs, rhs, label
    ):  # pragma: no cover
        """
        Enable typing for dictionary data arg to bodosql.BodoSQLContext({'table1': df}).
        Converts constant dictionary to tuple with sentinel.
        """
        kws = dict(rhs.kws)
        data_arg = get_call_expr_arg(
            "bodosql.BodoSQLContext", rhs.args, kws, 0, "tables"
        )
        arg_def = guard(get_definition, self.func_ir, data_arg)
        msg = "bodosql.BodoSQLContext(): 'tables' argument should be a dictionary with constant string keys"
        if not is_expr(arg_def, "build_map"):
            raise BodoError(msg)

        (tup_vars, new_nodes,) = bodo.utils.transform._convert_const_key_dict(
            self.args,
            self.func_ir,
            arg_def,
            msg,
            lhs.scope,
            lhs.loc,
            output_sentinel_tuple=True,
        )
        tup_var = tup_vars[0]
        set_call_expr_arg(tup_var, rhs.args, kws, 0, "tables")
        new_nodes.append(assign)
        return new_nodes

    def _handle_pd_read_sql(self, assign, lhs, rhs, label):
        """transform pd.read_sql calls"""
        # schema: pd.read_sql(sql, con, index_col=None,
        # coerce_float=True, params=None, parse_dates=None,
        # columns=None, chunksize=None
        kws = dict(rhs.kws)
        sql_var = get_call_expr_arg("read_sql", rhs.args, kws, 0, "sql")
        # The sql request has to be constant
        msg = (
            "pd.read_sql() requires 'sql' argument to be a constant string or an "
            "argument to the JIT function currently"
        )
        sql_const = get_const_value(sql_var, self.func_ir, msg, arg_types=self.args)
        con_var = get_call_expr_arg("read_sql", rhs.args, kws, 1, "con", "")
        msg = (
            "pd.read_sql() requires 'con' argument to be a constant string or an "
            "argument to the JIT function currently"
        )
        # the connection string has to be constant
        con_const = get_const_value(con_var, self.func_ir, msg, arg_types=self.args)
        index_col = self._get_const_arg(
            "read_sql", rhs.args, kws, 2, "index_col", rhs.loc, default=-1
        )
        # coerce_float = self._get_const_arg(
        #     "read_sql", rhs.args, kws, 3, "coerce_float", default=True
        # )
        # params = self._get_const_arg("read_sql", rhs.args, kws, 4, "params", default=-1)
        # parse_dates = self._get_const_arg(
        #     "read_sql", rhs.args, kws, 5, "parse_dates", default=-1
        # )
        columns = self._get_const_arg(
            "read_sql", rhs.args, kws, 6, "columns", rhs.loc, default=""
        )
        # chunksize = get_call_expr_arg("read_sql", rhs.args, kws, 7, "chunksize", -1)

        # SUPPORTED:
        # sql is supported since it is fundamental
        # con is supported since it is fundamental but only as a string
        # index_col is supported since setting the index is something useful.
        # UNSUPPORTED:
        # chunksize is unsupported because it is a different API and it makes for a different usage of pd.read_sql
        # columns   is unsupported because selecting columns could actually be done in SQL.
        # coerce_float is currently unsupported but it could be useful to support it.
        # params is currently unsupported because not needed for mysql but surely will be needed later.
        supported_args = ("sql", "con", "index_col", "parse_dates")

        unsupported_args = set(kws.keys()) - set(supported_args)
        if unsupported_args:
            raise BodoError(
                "read_sql() arguments {} not supported yet".format(unsupported_args)
            )

        # find db type
        db_type = urlparse(con_const).scheme
        # find df type
        df_type, converted_colnames = _get_sql_df_type_from_db(
            sql_const, con_const, db_type
        )
        dtypes = df_type.data
        dtype_map = {c: dtypes[i] for i, c in enumerate(df_type.columns)}
        col_names = [c for c in df_type.columns]

        # date columns
        date_cols = []

        columns, data_arrs, out_types = self._get_read_file_col_info(
            dtype_map, date_cols, col_names, lhs
        )

        nodes = [
            sql_ext.SqlReader(
                sql_const,
                con_const,
                lhs.name,
                columns,
                data_arrs,
                out_types,
                converted_colnames,
                db_type,
                lhs.loc,
            )
        ]

        columns = columns.copy()  # copy since modified below
        n_cols = len(columns)
        args = ["data{}".format(i) for i in range(n_cols)]
        data_args = args.copy()

        # one column is index
        if index_col != -1 and index_col != False:
            # convert column number to column name
            if isinstance(index_col, int):
                index_col = columns[index_col]
            index_ind = columns.index(index_col)
            index_arg = "bodo.utils.conversion.convert_to_index({})".format(
                data_args[index_ind]
            )
            columns.remove(index_col)
            data_args.remove(data_args[index_ind])
        else:
            # generate RangeIndex as default index
            assert len(data_args) > 0
            index_arg = "bodo.hiframes.pd_index_ext.init_range_index(0, len({}), 1, None)".format(
                data_args[0]
            )

        # Below we assume that the columns are strings
        col_var = gen_const_tup(columns)
        func_text = "def _init_df({}):\n".format(", ".join(args))
        func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), {}, {})\n".format(
            ", ".join(data_args), index_arg, col_var
        )
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _init_df = loc_vars["_init_df"]

        nodes += compile_func_single_block(_init_df, data_arrs, lhs)
        return nodes

    def _handle_pd_read_excel(self, assign, lhs, rhs, label):
        """add typing info to pd.read_excel() using extra argument '_bodo_df_type'
        This enables the overload implementation to just call Pandas
        """
        # schema (Pandas 1.0.3): io, sheet_name=0, header=0, names=None, index_col=None,
        # usecols=None, squeeze=False, dtype=None, engine=None, converters=None,
        # true_values=None, false_values=None, skiprows=None, nrows=None,
        # na_values=None, keep_default_na=True, verbose=False, parse_dates=False,
        # date_parser=None, thousands=None, comment=None, skipfooter=0,
        # convert_float=True, mangle_dupe_cols=True,
        kws = dict(rhs.kws)
        fname_var = get_call_expr_arg("read_excel", rhs.args, kws, 0, "io")
        sheet_name = self._get_const_arg(
            "read_excel",
            rhs.args,
            kws,
            1,
            "sheet_name",
            rhs.loc,
            default=0,
            typ="str or int",
        )
        header = self._get_const_arg(
            "read_excel",
            rhs.args,
            kws,
            2,
            "header",
            rhs.loc,
            default=0,
            typ="int",
        )
        col_names = self._get_const_arg(
            "read_excel", rhs.args, kws, 3, "names", rhs.loc, default=0
        )
        # index_col = self._get_const_arg("read_excel", rhs.args, kws, 4, "index_col", -1)
        comment = self._get_const_arg(
            "read_excel", rhs.args, kws, 20, "comment", rhs.loc, default=""
        )
        date_cols = self._get_const_arg(
            "pd.read_excel",
            rhs.args,
            kws,
            17,
            "parse_dates",
            rhs.loc,
            default=[],
            typ="int or str",
        )
        dtype_var = get_call_expr_arg("read_excel", rhs.args, kws, 7, "dtype", "")
        skiprows = self._get_const_arg(
            "read_excel",
            rhs.args,
            kws,
            12,
            "skiprows",
            rhs.loc,
            default=0,
            typ="int",
        )

        # replace "" placeholder with default None (can't use None in _get_const_arg)
        if comment == "":
            comment = None

        # TODO: support index_col
        # if index_col == -1:
        #     index_col = None

        # check unsupported arguments
        supported_args = (
            "io",
            "sheet_name",
            "header",
            "names",
            # "index_col",
            "comment",
            "dtype",
            "skiprows",
            "parse_dates",
        )
        unsupported_args = set(kws.keys()) - set(supported_args)
        if unsupported_args:
            raise BodoError(
                "read_excel() arguments {} not supported yet".format(unsupported_args)
            )

        if dtype_var != "" and col_names == 0 or dtype_var == "" and col_names != 0:
            raise BodoError(
                "pd.read_excel(): both 'dtype' and 'names' should be provided if either is provided"
            )

        # inference is necessary
        # TODO: refactor with read_csv
        if dtype_var == "":
            # infer column names and types from constant filename
            msg = (
                "pd.read_excel() requires explicit type annotation using "
                "the 'names' and 'dtype' arguments if the filename is not constant. For more information, see: https://docs.bodo.ai/latest/source/programming_with_bodo/file_io.html#non-constant-filepaths"
            )
            fname_const = get_const_value(
                fname_var, self.func_ir, msg, arg_types=self.args
            )

            df_type = _get_excel_df_type_from_file(
                fname_const, sheet_name, skiprows, header, comment, date_cols
            )

        else:
            dtype_map_const = get_const_value(
                dtype_var,
                self.func_ir,
                "pd.read_excel(): 'dtype' argument should be a constant value",
                arg_types=self.args,
            )
            if isinstance(dtype_map_const, dict):
                self._fix_dict_typing(dtype_var)
                dtype_map = {
                    c: _dtype_val_to_arr_type(t, "pd.read_excel", rhs.loc)
                    for c, t in dtype_map_const.items()
                }
            else:
                dtype_map = _dtype_val_to_arr_type(
                    dtype_map_const, "pd.read_excel", rhs.loc
                )

            index = RangeIndexType(types.none)
            # TODO: support index_col
            # if index_col is not None:
            #     index_name = col_names[index_col]
            #     index = bodo.hiframes.pd_index_ext.array_type_to_index(dtype_map[index_name], types.StringLiteral(index_name))
            #     col_names.remove(index_name)
            data_arrs = tuple(dtype_map[c] for c in col_names)
            df_type = DataFrameType(data_arrs, index, tuple(col_names))

        tp_var = ir.Var(lhs.scope, mk_unique_var("df_type_var"), rhs.loc)
        typ_assign = ir.Assign(ir.Const(df_type, rhs.loc), tp_var, rhs.loc)
        kws["_bodo_df_type"] = tp_var
        rhs.kws = list(kws.items())
        return [typ_assign, assign]

    def _handle_pd_read_csv(self, assign, lhs, rhs, label):
        """transform pd.read_csv(names=[A], dtype={'A': np.int32}) call"""
        # schema: pd.read_csv(filepath_or_buffer, sep=',', delimiter=None,
        # header='infer', names=None, index_col=None, usecols=None,
        # squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None,
        # engine=None, converters=None, true_values=None, false_values=None,
        # skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None,
        # na_values=None, keep_default_na=True, na_filter=True, verbose=False,
        # skip_blank_lines=True, parse_dates=False, infer_datetime_format=False,
        # keep_date_col=False, date_parser=None, dayfirst=False, cache_dates=True,
        # iterator=False, chunksize=None, compression='infer', thousands=None,
        # decimal=b'.', lineterminator=None, quotechar='"', quoting=0,
        # doublequote=True, escapechar=None, comment=None, encoding=None,
        # encoding_errors='strict', dialect=None,
        # error_bad_lines=True, warn_bad_lines=True, on_bad_lines='error',
        # delim_whitespace=False, low_memory=True, memory_map=False,
        # float_precision=None, storage_options=None)
        kws = dict(rhs.kws)

        # TODO: Can we use fold the arguments even though this untyped pass?

        fname = get_call_expr_arg("pd.read_csv", rhs.args, kws, 0, "filepath_or_buffer")
        # fname's type is checked at typing pass or when it is forced to be a constant.

        # Users can only provide either sep or delim. Use a dummy default value to track
        # this behavior.
        sep_val = self._get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            1,
            "sep",
            rhs.loc,
            default=None,
            use_default=True,
        )
        delim_val = self._get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            2,
            "delimiter",
            rhs.loc,
            default=None,
            use_default=True,
        )
        sep_arg_name = "sep"
        if sep_val is None and delim_val is None:
            # If both arguments are the default, use ","
            sep = ","
        elif sep_val is None:
            sep = delim_val
            sep_arg_name = "delimiter"
        elif delim_val is None:
            sep = sep_val
        else:
            raise BodoError(
                "pd.read_csv() Specified a 'sep' and a 'delimiter'; you can only specify one.",
                loc=rhs.loc,
            )
        # Pandas doesn't catch this error and produces a stack trace, but we need to.
        if not isinstance(sep, str):
            raise BodoError(
                f"pd.read_csv() '{sep_arg_name}' must be a constant string.",
                loc=rhs.loc,
            )

        # [BE-869] Bodo can't handle len(sep) > 1 except '\\s+' because the Pandas
        # C engine can't handle it. Pandas won't use the Python engine because we
        # set low_memory=True
        if len(sep) > 1 and sep != "\\s+":
            raise BodoError(
                f"pd.read_csv() '{sep_arg_name}' is an invalid separator. Bodo only supports single character separators and '\\s+'.",
                loc=rhs.loc,
            )

        header = self._get_const_arg(
            "pd.read_csv", rhs.args, kws, 3, "header", rhs.loc, default="infer"
        )
        # Per Pandas documentation (header: int, list of int, default ‘infer’)
        if header not in ("infer", 0, None):
            raise BodoError(
                f"pd.read_csv() 'header' should be one of 'infer', 0, or None",
                loc=rhs.loc,
            )

        col_names = self._get_const_arg(
            "pd.read_csv", rhs.args, kws, 4, "names", rhs.loc, default=0
        )
        # Per Pandas documentation (names: array-like). Since columns don't need string types,
        # we only check that this is a list or tuple (since constant arrays aren't supported).
        if col_names != 0 and not isinstance(col_names, (list, tuple)):
            raise BodoError(
                "pd.read_csv() 'names' should be a constant list if provided",
                loc=rhs.loc,
            )

        index_col = self._get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            5,
            "index_col",
            rhs.loc,
            default=None,
            use_default=True,
        )
        # Per Pandas documentation (index_col: int, str, sequence of int / str, or False, default None).
        # We don't support sequences yet
        if (
            index_col is not None
            and not isinstance(index_col, (int, str))
            # isinstance(True, int) == True, so check True is unsupported.
            or index_col is True
        ):
            raise BodoError(
                "pd.read_csv() 'index_col' must be a constant integer, constant string that matches a column name, or False",
                loc=rhs.loc,
            )

        usecols = self._get_const_arg(
            "pd.read_csv()",
            rhs.args,
            kws,
            6,
            "usecols",
            rhs.loc,
            default=None,
            use_default=True,
        )
        # Per Pandas documentation (usecols: list-like or callable).
        # We don't support callables yet.
        if usecols is not None and (not isinstance(usecols, (tuple, list))):
            raise BodoError(
                "pd.read_csv() 'usecols' must be a constant list of columns names or column indices if provided",
                loc=rhs.loc,
            )

        dtype_var = get_call_expr_arg(
            "pd.read_csv", rhs.args, kws, 10, "dtype", None, use_default=True
        )

        _skiprows = get_call_expr_arg(
            "pd.read_csv", rhs.args, kws, 16, "skiprows", default=None, use_default=True
        )
        # Initialize skiprows_val = 0 since it's needed for CSVFileInfo
        skiprows_val = 0

        # Per Pandas documentation (skiprows: list-like, int or callable)
        # skiprows must be constant known at compile time or variable with column names provided by the user
        # Reason: Bodo needs a constant value for skiprows as it uses read_csv to get file information.
        # When skiprows is used, column name changes.
        # To allow variables, we set skiprows to 0 and this means that we don't get same column names as Pandas
        # Solution: let user specify column names.
        if _skiprows is None:
            # Skiprows isn't provided so we don't need to check for constant requirement.
            _skiprows = ir.Const(0, rhs.loc)
        else:
            try:
                if isinstance(_skiprows, ir.Const):
                    skiprows_val = _skiprows.value
                else:
                    skiprows_val = get_const_value_inner(
                        self.func_ir, _skiprows, arg_types=self.args
                    )
            except GuardException:
                # raise error if skiprows is used but not constant without column names
                if col_names == 0:
                    raise BodoError(
                        "pd.read_csv() column names must be provided if 'skiprows' is not constant known at compile-time",
                        loc=_skiprows.loc,
                    )

        # This checks for constant list at compile time.
        is_skiprows_list = _check_int_list(skiprows_val)
        if not isinstance(skiprows_val, int) and not is_skiprows_list:
            raise BodoError(
                "pd.read_csv() 'skiprows' must be an integer or list of integers.",
                loc=_skiprows.loc,
            )
        # Sort list and remove duplicates
        skiprows_val = sorted(set(skiprows_val)) if is_skiprows_list else skiprows_val
        # Since list is sorted, test first value only in the list
        if (isinstance(skiprows_val, int) and skiprows_val < 0) or (
            is_skiprows_list and skiprows_val[0] < 0
        ):
            # If skiprows integer is already a constant, check the size at compile time
            raise BodoError(
                "pd.read_csv() skiprows must be integer >= 0.", loc=_skiprows.loc
            )

        _nrows = get_call_expr_arg(
            "pd.read_csv", rhs.args, kws, 18, "nrows", default=ir.Const(-1, rhs.loc)
        )

        date_cols = self._get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            24,
            "parse_dates",
            rhs.loc,
            default=False,
            typ="int or str",
        )
        # Per Pandas documentation (parse_dates: bool or list of int or names or list of lists or dict, default false)
        # Check for False if the user provides the default value
        if date_cols == False:
            date_cols = []
        if not isinstance(date_cols, (tuple, list)):
            raise BodoError(
                "pd.read_csv() 'parse_dates' must be a constant list of column names or column indices if provided",
                loc=rhs.loc,
            )

        chunksize = self._get_const_arg(
            "pandas.read_csv",
            rhs.args,
            kws,
            31,
            "chunksize",
            rhs.loc,
            default=None,
            use_default=True,
            typ="int",
        )
        if chunksize is not None and (not isinstance(chunksize, int) or chunksize < 1):
            raise BodoError(
                "pd.read_csv() 'chunksize' must be a constant integer >= 1 if provided."
            )

        compression = self._get_const_arg(
            "pd.read_csv", rhs.args, kws, 32, "compression", rhs.loc, default="infer"
        )
        # Per Pandas documentation (compression: {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default ‘infer’)
        supported_compression_options = ["infer", "gzip", "bz2", "zip", "xz", None]
        if compression not in supported_compression_options:
            raise BodoError(
                f"pd.read_csv() 'compression' must be one of {supported_compression_options}",
                loc=rhs.loc,
            )

        # Pandas default is True but Bodo is False
        pd_low_memory = self._get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            48,
            "low_memory",
            rhs.loc,
            default=False,
            use_default=True,
        )

        # Bodo specific arguments. To avoid constantly needing to update Pandas we
        # make these kwargs only.

        # _bodo_upcast_to_float64 updates types if inference may not be fully accurate.
        # This upcasts all integer/float values to float64. This runs before
        # dtype dictionary and won't impact those columns.
        _bodo_upcast_to_float64 = self._get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            -1,
            "_bodo_upcast_to_float64",
            rhs.loc,
            default=False,
            use_default=True,
        )

        # List of all possible args and a support default value. This should match the header above.
        # If a default value is not supported, use None. We provide the default value to enable passing
        # an argument so long as it matches the default value. For example, if someone provides engine=None
        # this is logically equivalent to excluding it and we want to minimize code rewrites.
        total_args = (
            ("filepath_or_buffer", None),
            ("sep", ","),
            ("delimiter", None),
            ("header", "infer"),
            ("names", None),
            ("index_col", None),
            ("usecols", None),
            ("squeeze", False),
            ("prefix", None),
            ("mangle_dupe_cols", True),
            ("dtype", None),
            ("engine", None),
            ("converters", None),
            ("true_values", None),
            ("false_values", None),
            ("skipinitialspace", False),
            ("skiprows", None),
            ("skipfooter", 0),
            ("nrows", None),
            ("na_values", None),
            ("keep_default_na", True),
            ("na_filter", True),
            ("verbose", False),
            ("skip_blank_lines", True),
            ("parse_dates", False),
            ("infer_datetime_format", False),
            ("keep_date_col", False),
            ("date_parser", None),
            ("dayfirst", False),
            ("cache_dates", True),
            ("iterator", False),
            ("chunksize", None),
            ("compression", "infer"),
            ("thousands", None),
            ("decimal", b"."),
            ("lineterminator", None),
            ("quotechar", '"'),
            ("quoting", 0),
            ("doublequote", True),
            ("escapechar", None),
            ("comment", None),
            ("encoding", None),
            ("encoding_errors", "strict"),
            ("dialect", None),
            ("error_bad_lines", True),
            ("warn_bad_lines", True),
            ("on_bad_lines", "error"),
            ("delim_whitespace", False),
            ("low_memory", False),
            ("memory_map", False),
            ("float_precision", None),
            ("storage_options", None),
            # TODO: Specify this is kwonly in error checks
            ("_bodo_upcast_to_float64", False),
        )
        # Arguments that are supported
        supported_args = set(
            (
                "filepath_or_buffer",
                "sep",
                "delimiter",
                "header",
                "names",
                "index_col",
                "usecols",
                "dtype",
                "skiprows",
                "nrows",
                "parse_dates",
                "chunksize",
                "compression",
                "low_memory",
                "_bodo_upcast_to_float64",
            )
        )
        # Iterate through the provided args. If an argument is in the supported_args,
        # skip it. Otherwise we check that the value matches the default value.
        unsupported_args = []
        for i, arg_pair in enumerate(total_args):
            name, default = arg_pair
            if name not in supported_args:
                try:
                    # Catch the exceptions because don't want the constant value exception
                    # Instead we want to indicate the argument isn't supported.
                    provided_val = self._get_const_arg(
                        "pd.read_csv",
                        rhs.args,
                        kws,
                        i,
                        name,
                        rhs.loc,
                        default=default,
                        use_default=True,
                    )
                    if provided_val != default:
                        unsupported_args.append(name)
                except BodoError:
                    # If the value is not a constant then the user tried to use an unsupported argument.
                    unsupported_args.append(name)
            # TODO: Replace with folding?
            # If i < len(args), then the value was passed as an argument (since its in location i).
            # If we also find it in kws this is an error.
            if i < len(rhs.args) and name in kws:
                raise BodoError(
                    f"pd.read_csv() got multiple values for argument '{name}'.",
                    loc=rhs.loc,
                )
            kws.pop(name, 0)

        if unsupported_args:
            raise BodoError(
                f"pd.read_csv() arguments {unsupported_args} not supported yet",
                loc=rhs.loc,
            )

        if len(rhs.args) > len(total_args):
            raise BodoError(
                f"pd.read_csv() {len(rhs.args)} arguments provided, but this function only accepts {len(total_args)} arguments",
                loc=rhs.loc,
            )

        if kws:
            extra_kws = list(kws.keys())
            raise BodoError(
                f"pd.read_csv() Unknown argument(s) {extra_kws} provided.",
                loc=rhs.loc,
            )

        # infer the column names: if no names
        # are passed the behavior is identical to ``header=0`` and column
        # names are inferred from the first line of the file, if column
        # names are passed explicitly then the behavior is identical to
        # ``header=None``
        if header == "infer":
            header = 0 if col_names == 0 else None

        # if inference is required
        dtype_map = {}
        if dtype_var is None or col_names == 0:
            # infer column names and types from constant filename
            msg = (
                "pd.read_csv() requires explicit type "
                "annotation using the 'names' and 'dtype' arguments if the filename is not constant. For more information, see: https://docs.bodo.ai/latest/source/programming_with_bodo/file_io.html#non-constant-filepaths"
            )
            fname_const = get_const_value(
                fname,
                self.func_ir,
                msg,
                arg_types=self.args,
                file_info=CSVFileInfo(sep, skiprows_val, header, compression),
            )
            if not isinstance(fname_const, str):
                raise BodoError(
                    "pd.read_csv() 'filepath_or_buffer' must be a string.", loc=rhs.loc
                )

            got_schema = False
            # get_const_value forces variable to be literal which should convert it to
            # FilenameType. If so, the schema will be part of the type
            var_def = guard(get_definition, self.func_ir, fname)
            if isinstance(var_def, ir.Arg):
                typ = self.args[var_def.index]
                if isinstance(typ, types.FilenameType):
                    df_type = typ.schema
                    got_schema = True
            if not got_schema:
                df_type = _get_csv_df_type_from_file(
                    fname_const, sep, skiprows_val, header, compression
                )
            dtypes = df_type.data
            if usecols is None:
                usecols = list(range(len(dtypes)))
            else:
                # make sure usecols has column indices (not names)
                usecols = [
                    _get_col_ind_from_name_or_ind(
                        c, col_names if col_names else df_type.columns
                    )
                    for c in usecols
                ]
            # convert Pandas generated integer names if any
            cols = [str(df_type.columns[i]) for i in usecols]
            # overwrite column names like Pandas if explicitly provided
            if col_names != 0:
                cols[-len(col_names) :] = col_names
            col_names = cols
            dtype_map = {c: dtypes[usecols[i]] for i, c in enumerate(col_names)}
        else:
            if usecols is None:
                usecols = list(range(len(col_names)))
            else:
                # make sure usecols has column indices (not names)
                usecols = [_get_col_ind_from_name_or_ind(c, col_names) for c in usecols]

        if _bodo_upcast_to_float64:
            dtype_map_cpy = dtype_map.copy()
            for c, t in dtype_map_cpy.items():
                if isinstance(t, (types.Array, IntegerArrayType)) and isinstance(
                    t.dtype, (types.Integer, types.Float)
                ):
                    dtype_map[c] = types.Array(types.float64, 1, "C")

        # handle dtype arg if provided
        if dtype_var is not None:
            # NOTE: the user may provide dtype for only a subset of columns

            dtype_map_const = get_const_value(
                dtype_var,
                self.func_ir,
                "pd.read_csv() 'dtype' argument should be a constant value",
                arg_types=self.args,
            )
            if isinstance(dtype_map_const, dict):
                self._fix_dict_typing(dtype_var)
                dtype_update_map = {}
                colname_set = set(col_names)
                missing_columns = []
                for c, t in dtype_map_const.items():
                    # Check int to avoid cases where the key is the column index.
                    # i.e. {0: str}. See _get_col_ind_from_name_or_ind.
                    if c not in colname_set and not isinstance(c, int):
                        missing_columns.append(c)
                    else:
                        dtype_update_map[
                            col_names[_get_col_ind_from_name_or_ind(c, col_names)]
                        ] = _dtype_val_to_arr_type(t, "pd.read_csv", rhs.loc)
                dtype_map.update(dtype_update_map)
                if missing_columns:
                    warnings.warn(
                        BodoWarning(
                            f"pd.read_csv(): Columns {missing_columns} included in dtype dictionary but not found in output DataFrame. These entries have been ignored."
                        )
                    )
            else:
                dtype_map = _dtype_val_to_arr_type(
                    dtype_map_const, "pd.read_csv", rhs.loc
                )

        columns, _, out_types = self._get_read_file_col_info(
            dtype_map, date_cols, col_names, lhs
        )

        orig_columns = columns.copy()  # copy since modified below

        data_args = ["table_val", "idx_arr_val"]

        # one column is index
        if index_col is not None and not (
            isinstance(index_col, bool) and index_col == False
        ):
            # convert column number to column name
            if isinstance(index_col, int):
                index_col = columns[index_col]

            index_ind = columns.index(index_col)

            index_arr_typ = out_types.pop(index_ind)

            index_elem_dtype = index_arr_typ.dtype
            index_name = index_col
            index_typ = bodo.utils.typing.index_typ_from_dtype_name(
                index_elem_dtype, index_name
            )

            columns.remove(index_col)
            orig_columns.remove(index_col)
            if index_ind in usecols:
                usecols.remove(index_ind)

            index_arg = f'bodo.utils.conversion.convert_to_index({data_args[1]}, name = "{index_name}")'

        else:
            # generate RangeIndex as default index
            index_ind = None
            index_name = None
            index_arr_typ = types.none
            index_arg = "bodo.hiframes.pd_index_ext.init_range_index(0, len({}), 1, None)".format(
                data_args[0]
            )
            index_typ = RangeIndexType(types.none)

        # I'm not certain if this is possible, but I'll add a check just in case
        if isinstance(index_typ, bodo.hiframes.pd_multi_index_ext.MultiIndexType):
            raise_bodo_error("Read_csv(): Index column cannot be a multindex")

        df_type = DataFrameType(
            tuple(out_types), index_typ, tuple(columns), is_table_format=True
        )

        # If we have a chunksize we need to create an iterator, so we determine
        # the yield DataFrame type directly.
        if chunksize is not None:
            out_types = [
                bodo.io.csv_iterator_ext.CSVIteratorType(
                    df_type,
                    orig_columns,
                    out_types,
                    usecols,
                    sep,
                    index_ind,
                    index_arr_typ,
                    index_name,
                )
            ]
            # Create a new temp var so this is always exactly one variable.
            data_arrs = [ir.Var(lhs.scope, mk_unique_var("csv_iterator"), lhs.loc)]
        else:
            data_arrs = [
                ir.Var(lhs.scope, mk_unique_var("csv_table"), lhs.loc),
                ir.Var(lhs.scope, mk_unique_var("index_col"), lhs.loc),
            ]
        nodes = [
            csv_ext.CsvReader(
                fname,
                lhs.name,
                sep,
                orig_columns,
                data_arrs,
                out_types,
                usecols,
                lhs.loc,
                header,
                compression,
                _nrows,
                _skiprows,
                chunksize,
                is_skiprows_list,
                pd_low_memory,
                index_ind,
                # CsvReader expects the type of the read column
                # not the index type itself
                bodo.utils.typing.get_index_data_arr_types(index_typ)[0],
            )
        ]

        # Below we assume that the columns are strings
        if chunksize is not None:
            # Generate an assign because init_csv_iterator will happen inside read_csv
            nodes += [ir.Assign(data_arrs[0], lhs, lhs.loc)]
        else:
            func_text = f"def _type_func({data_args[0]}, {data_args[1]}):\n"
            func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe((table_val,), {index_arg}, df_type)\n"

            loc_vars = {}

            exec(func_text, {}, loc_vars)
            _type_func = loc_vars["_type_func"]

            nodes += compile_func_single_block(
                _type_func, data_arrs, lhs, extra_globals={"df_type": df_type}
            )
        return nodes

    def _handle_pd_read_json(self, assign, lhs, rhs, label):
        """transform pd.read_json() call,
        where default orient = 'records'

        schema: pandas.read_json(path_or_buf=None, orient=None, typ='frame',
        dtype=None, convert_axes=None, convert_dates=True,
        keep_default_dates=True, numpy=False, precise_float=False,
        date_unit=None, encoding=None, lines=False, chunksize=None,
        compression='infer')
        """
        # convert_dates required for date cols
        kws = dict(rhs.kws)
        fname = get_call_expr_arg("read_json", rhs.args, kws, 0, "path_or_buf")
        orient = self._get_const_arg(
            "read_json", rhs.args, kws, 1, "orient", rhs.loc, default="records"
        )
        frame_or_series = get_call_expr_arg(
            "read_json", rhs.args, kws, 3, "typ", "frame"
        )
        dtype_var = get_call_expr_arg("read_json", rhs.args, kws, 10, "dtype", "")
        # default value is True
        convert_dates = self._get_const_arg(
            "read_json",
            rhs.args,
            kws,
            5,
            "convert_dates",
            rhs.loc,
            default=True,
            typ="int or str",
        )
        date_cols = [] if convert_dates is True else convert_dates
        precise_float = self._get_const_arg(
            "read_json", rhs.args, kws, 8, "precise_float", rhs.loc, default=False
        )
        lines = self._get_const_arg(
            "read_json", rhs.args, kws, 11, "lines", rhs.loc, default=True
        )
        compression = self._get_const_arg(
            "read_json", rhs.args, kws, 13, "compression", rhs.loc, default="infer"
        )

        # check unsupported arguments
        unsupported_args = {
            "convert_axes",
            "keep_default_dates",
            "numpy",
            "date_unit",
            "encoding",
            "chunksize",
        }

        passed_unsupported = unsupported_args.intersection(kws.keys())
        if len(passed_unsupported) > 0:
            if unsupported_args:
                raise BodoError(
                    "read_json() arguments {} not supported yet".format(
                        passed_unsupported
                    )
                )

        supported_compression_options = {"infer", "gzip", "bz2", None}
        if compression not in supported_compression_options:
            raise BodoError(
                "pd.read_json() compression = {} is not supported."
                " Supported options are {}".format(
                    compression, supported_compression_options
                )
            )

        if frame_or_series != "frame":
            raise BodoError(
                "pd.read_json() typ = {} is not supported."
                "Currently only supports orient = 'frame'".format(frame_or_series)
            )

        if orient != "records":
            raise BodoError(
                "pd.read_json() orient = {} is not supported."
                "Currently only supports orient = 'records'".format(orient)
            )

        if type(lines) != bool:
            raise BodoError(
                "pd.read_json() lines = {} is not supported."
                "lines must be of type bool.".format(lines)
            )

        col_names = []

        # infer column names and types from constant filenames if:
        # not explicitly passed with dtype
        # not reading from s3 & hdfs
        # not reading from directory
        msg = "pd.read_json() requires the filename to be a compile time constant. For more information, see: https://docs.bodo.ai/latest/source/programming_with_bodo/file_io.html#non-constant-filepaths"
        fname_const = get_const_value(
            fname,
            self.func_ir,
            msg,
            arg_types=self.args,
            file_info=JSONFileInfo(
                orient, convert_dates, precise_float, lines, compression
            ),
        )

        if dtype_var == "":
            # can only read partial of the json file
            # when orient == 'records' && lines == True
            # TODO: check this
            if not lines:
                raise BodoError(
                    "pd.read_json() requires explicit type annotation using 'dtype',"
                    " when lines != True"
                )
            # TODO: more error checking needed

            got_schema = False
            # get_const_value forces variable to be literal which should convert it to
            # FilenameType. If so, the schema will be part of the type
            var_def = guard(get_definition, self.func_ir, fname)
            if isinstance(var_def, ir.Arg):
                typ = self.args[var_def.index]
                if isinstance(typ, types.FilenameType):
                    df_type = typ.schema
                    got_schema = True
            if not got_schema:
                df_type = _get_json_df_type_from_file(
                    fname_const,
                    orient,
                    convert_dates,
                    precise_float,
                    lines,
                    compression,
                )
            dtypes = df_type.data
            # convert Pandas generated integer names if any
            col_names = [str(df_type.columns[i]) for i in range(len(dtypes))]
            dtype_map = {c: dtypes[i] for i, c in enumerate(col_names)}
        else:  # handle dtype arg if provided
            dtype_map_const = get_const_value(
                dtype_var,
                self.func_ir,
                "pd.read_json(): 'dtype' argument should be a constant value",
                arg_types=self.args,
            )
            if isinstance(dtype_map_const, dict):
                self._fix_dict_typing(dtype_var)
                dtype_map = {
                    c: _dtype_val_to_arr_type(t, "pd.read_json", rhs.loc)
                    for c, t in dtype_map_const.items()
                }
            else:
                dtype_map = _dtype_val_to_arr_type(
                    dtype_map_const, "pd.read_json", rhs.loc
                )

        columns, data_arrs, out_types = self._get_read_file_col_info(
            dtype_map, date_cols, col_names, lhs
        )

        nodes = [
            json_ext.JsonReader(
                lhs.name,
                lhs.loc,
                data_arrs,
                out_types,
                fname,
                columns,
                orient,
                convert_dates,
                precise_float,
                lines,
                compression,
            )
        ]

        columns = columns.copy()  # copy since modified below
        n_cols = len(columns)
        args = ["data{}".format(i) for i in range(n_cols)]
        data_args = args.copy()

        # initialize range index
        assert len(data_args) > 0
        index_arg = (
            "bodo.hiframes.pd_index_ext.init_range_index(0, len({}), 1, None)".format(
                data_args[0]
            )
        )

        # Below we assume that the columns are strings
        col_var = gen_const_tup(columns)
        func_text = "def _init_df({}):\n".format(", ".join(args))
        func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), {}, {})\n".format(
            ", ".join(data_args), index_arg, col_var
        )
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _init_df = loc_vars["_init_df"]

        nodes += compile_func_single_block(_init_df, data_arrs, lhs)
        return nodes

    def _get_read_file_col_info(self, dtype_map, date_cols, col_names, lhs):
        """get column names, ir.Var objects, and data types for file read (csv/json)"""
        # single dtype is provided instead of dictionary
        if isinstance(dtype_map, types.Type):
            typ = dtype_map
            data_arrs = [
                ir.Var(lhs.scope, mk_unique_var(cname), lhs.loc) for cname in col_names
            ]
            return col_names, data_arrs, [typ] * len(col_names)

        columns = []
        data_arrs = []
        out_types = []
        for i, (col_name, typ) in enumerate(dtype_map.items()):
            columns.append(col_name)
            # get array dtype
            if i in date_cols or col_name in date_cols:
                typ = types.Array(types.NPDatetime("ns"), 1, "C")
            out_types.append(typ)
            # output array variable
            data_arrs.append(ir.Var(lhs.scope, mk_unique_var(col_name), lhs.loc))

        return columns, data_arrs, out_types

    def _handle_pd_Series(self, assign, lhs, rhs):
        """transform pd.Series(A) call for flatmap case"""
        kws = dict(rhs.kws)
        data = get_call_expr_arg("pd.Series", rhs.args, kws, 0, "data", "")
        if data == "":
            return [assign]

        # match flatmap pd.Series(list(itertools.chain(*A))) and flatten
        data_def = guard(get_definition, self.func_ir, data)
        if (
            is_call(data_def)
            and guard(find_callname, self.func_ir, data_def) == ("list", "builtins")
            and len(data_def.args) == 1
        ):
            data_def = guard(get_definition, self.func_ir, data_def.args[0])

        fdef = guard(find_callname, self.func_ir, data_def)
        if is_call(data_def) and fdef in (
            ("chain", "itertools"),
            ("from_iterable_impl", "bodo.utils.typing"),
        ):
            if fdef == ("chain", "itertools"):
                in_data = data_def.vararg
                data_def.vararg = None  # avoid typing error
            else:
                in_data = data_def.args[0]
            new_arr = ir.Var(in_data.scope, mk_unique_var("flat_arr"), in_data.loc)
            nodes = compile_func_single_block(
                eval(
                    "lambda A: bodo.utils.conversion.flatten_array(bodo.utils.conversion.coerce_to_array(A))"
                ),
                (in_data,),
                new_arr,
            )
            # put the new array back to pd.Series call
            if len(rhs.args) > 0:
                rhs.args[0] = new_arr
            else:  # kw case
                # TODO: test
                kws["data"] = new_arr
                rhs.kws = list(kws.items())
            nodes.append(assign)
            return nodes

        # pd.Series() is handled in typed pass now
        return [assign]

    def _handle_pd_named_agg(self, assign, lhs, rhs):
        """replace pd.NamedAgg() with equivalent tuple to be handled in groupby typing.
        For example, df.groupby("A").agg(C=pd.NamedAgg(column="B", aggfunc="sum")) ->
        df.groupby("A").agg(C=("B", "sum"))
        Tuple is the same as NamedAgg in Pandas groupby. Tuple enables typing since it
        preserves constants while NamedAgg which is a namedtuple doesn't (Numba
        limitation).
        """
        kws = dict(rhs.kws)
        column_var = get_call_expr_arg("pd.NamedAgg", rhs.args, kws, 0, "column")
        aggfunc_var = get_call_expr_arg("pd.NamedAgg", rhs.args, kws, 1, "aggfunc")
        assign.value = ir.Expr.build_tuple([column_var, aggfunc_var], rhs.loc)
        return [assign]

    def _handle_pq_read_table(self, assign, lhs, rhs):
        if len(rhs.args) != 1:  # pragma: no cover
            raise BodoError("Invalid read_table() arguments")
        # put back the definition removed earlier but remove node
        self.func_ir._definitions[lhs.name].append(rhs)
        self.arrow_tables[lhs.name] = rhs.args[0]
        return []

    def _handle_pq_to_pandas(self, assign, lhs, rhs, t_var):
        return self._gen_parquet_read(self.arrow_tables[t_var.name], lhs)

    def _gen_parquet_read(self, fname, lhs, columns=None, storage_options=None):
        # make sure pyarrow is available
        if not bodo.utils.utils.has_pyarrow():
            raise RuntimeError("pyarrow is required for Parquet support")

        (
            columns,
            data_arrs,
            index_col,
            nodes,
            col_types,
            index_col_type,
        ) = self.pq_handler.gen_parquet_read(
            fname, lhs, columns, storage_options=storage_options
        )
        n_cols = len(columns)

        index_colname = (
            None if (isinstance(index_col, dict) or index_col is None) else index_col
        )
        if index_col is None:
            assert n_cols > 0
            index_arg = (
                "bodo.hiframes.pd_index_ext.init_range_index(0, len(T), 1, None)"
            )
            index_typ = RangeIndexType(types.none)
        elif isinstance(index_col, dict):
            if index_col["name"] is None:
                index_col_name = None
            else:
                index_col_name = "'{}'".format(index_col["name"])
            # In case there is filtering we take a min between the stop and the filtered length.
            # This won't match Pandas, which instead should have a Numeric Index if there is any filtering.
            # TODO: Match Pandas
            min_str = f"min({index_col['stop']}, (len(T) * {index_col['step']}) + {index_col['start']})"
            index_arg = f"bodo.hiframes.pd_index_ext.init_range_index({index_col['start']}, {min_str}, {index_col['step']}, {index_col_name})"
            index_typ = RangeIndexType(
                types.none
                if index_col["name"] is None
                else types.literal(index_col["name"])
            )
        else:
            # if the index_col is __index_level_0_, it means it has no name.
            # Thus we do not write the name instead of writing '__index_level_0_' as the name
            index_name = None if "__index_level_" in index_col else index_col
            index_arg = (
                f"bodo.utils.conversion.convert_to_index(index_arr, {index_name!r})"
            )
            index_typ = bodo.hiframes.pd_index_ext.array_type_to_index(
                index_col_type,
                types.none if index_name is None else types.literal(index_name),
            )

        col_args = tuple(
            c for c in columns if index_colname is None or c != index_colname
        )
        df_col_types = tuple(
            t
            for (c, t) in zip(columns, col_types)
            if index_colname is None or c != index_colname
        )

        out_df_type = DataFrameType(
            df_col_types, index_typ, col_args, is_table_format=True
        )
        func_text = "def _init_df(T, index_arr):\n"
        func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe((T,), {index_arg}, out_df_type)\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _init_df = loc_vars["_init_df"]
        nodes += compile_func_single_block(
            _init_df, data_arrs, lhs, extra_globals={"out_df_type": out_df_type}
        )
        return nodes

    def _handle_pd_read_parquet(self, assign, lhs, rhs):
        # get args and check values
        kws = dict(rhs.kws)
        fname = get_call_expr_arg("read_parquet", rhs.args, kws, 0, "path")
        engine = get_call_expr_arg("read_parquet", rhs.args, kws, 1, "engine", "auto")
        columns = self._get_const_arg(
            "read_parquet", rhs.args, kws, 2, "columns", rhs.loc, default=-1
        )
        storage_options = self._get_const_arg(
            "read_parquet", rhs.args, kws, 10e4, "storage_options", rhs.loc, default={}
        )

        # check unsupported arguments
        supported_args = (
            "path",
            "engine",
            "columns",
            "storage_options",
        )
        unsupported_args = set(kws.keys()) - set(supported_args)
        if unsupported_args:
            raise BodoError(
                "read_parquet() arguments {} not supported yet".format(unsupported_args)
            )

        if isinstance(storage_options, dict):
            supported_storage_options = ("anon",)
            unsupported_storage_options = set(storage_options.keys()) - set(
                supported_storage_options
            )
            if unsupported_storage_options:
                raise BodoError(
                    "read_parquet() arguments {} for 'storage_options' not supported yet".format(
                        unsupported_storage_options
                    )
                )

            if "anon" in storage_options:
                if not isinstance(storage_options["anon"], bool):
                    raise BodoError(
                        "read_parquet: 'anon' in 'storage_options' must be a constant boolean value"
                    )
        else:
            raise BodoError(
                "read_parquet: 'storage_options' must be a constant dictionary"
            )

        if engine not in ("auto", "pyarrow"):
            raise BodoError("read_parquet: only pyarrow engine supported")

        if columns == -1:
            columns = None

        return self._gen_parquet_read(fname, lhs, columns, storage_options)

    def _handle_np_fromfile(self, assign, lhs, rhs):
        """translate np.fromfile() to native
        file and dtype are required arguments. sep is supported for only the
        default value.
        """
        kws = dict(rhs.kws)
        if len(rhs.args) + len(kws) > 5:  # pragma: no cover
            raise bodo.utils.typing.BodoError(
                f"np.fromfile(): at most 5 arguments expected"
                f" ({len(rhs.args) + len(kws)} given)"
            )
        valid_kws = {"file", "dtype", "count", "sep", "offset"}
        for kw in set(kws) - valid_kws:  # pragma: no cover
            raise bodo.utils.typing.BodoError(
                f"np.fromfile(): unexpected keyword argument {kw}"
            )
        np_fromfile = "np.fromfile"
        _fname = get_call_expr_arg(np_fromfile, rhs.args, kws, 0, "file")
        _dtype = get_call_expr_arg(np_fromfile, rhs.args, kws, 1, "dtype")
        _count = get_call_expr_arg(
            np_fromfile, rhs.args, kws, 2, "count", default=ir.Const(-1, lhs.loc)
        )
        sep_err_msg = err_msg = f"{np_fromfile}(): sep argument is not supported"
        _sep = self._get_const_arg(
            np_fromfile,
            rhs.args,
            kws,
            3,
            "sep",
            rhs.loc,
            default="",
            err_msg=sep_err_msg,
        )
        if _sep != "":
            raise bodo.utils.typing.BodoError(sep_err_msg)
        _offset = get_call_expr_arg(
            np_fromfile,
            rhs.args,
            kws,
            4,
            "offset",
            default=ir.Const(0, lhs.loc),
        )

        func_text = (
            ""
            "def fromfile_impl(fname, dtype, count, offset):\n"
            # check_java_installation is a check for hdfs that java is installed
            "    check_java_installation(fname)\n"
            "    dtype_size = get_dtype_size(dtype)\n"
            "    size = get_file_size(fname, count, offset, dtype_size)\n"
            "    A = np.empty(size // dtype_size, dtype=dtype)\n"
            "    file_read(fname, A, size, offset)\n"
            "    read_arr = A\n"
        )

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        f_block = compile_to_numba_ir(
            loc_vars["fromfile_impl"],
            {
                "np": np,
                "get_file_size": bodo.io.np_io.get_file_size,
                "file_read": bodo.io.np_io.file_read,
                "get_dtype_size": bodo.io.np_io.get_dtype_size,
                "check_java_installation": check_java_installation,
            },
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [_fname, _dtype, _count, _offset])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = lhs
        return nodes

    def _get_const_arg(
        self,
        f_name,
        args,
        kws,
        arg_no,
        arg_name,
        loc,
        *,
        default=None,
        err_msg=None,
        typ=None,
        use_default=False,
    ):
        """Get constant value for a function call argument. Raise error if the value is
        not constant.
        """
        typ = "str" if typ is None else typ
        arg = CONST_NOT_FOUND
        if err_msg is None:
            err_msg = ("{} requires '{}' argument as a constant {}").format(
                f_name, arg_name, typ
            )

        arg_var = get_call_expr_arg(f_name, args, kws, arg_no, arg_name, "")

        try:
            arg = get_const_value_inner(self.func_ir, arg_var, arg_types=self.args)
        except GuardException:
            # raise error if argument specified but not constant
            if arg_var != "":
                raise BodoError(err_msg, loc=loc)

        if arg is CONST_NOT_FOUND:
            # Provide use_default to allow letting None be the default value
            if use_default or default is not None:
                return default
            raise BodoError(err_msg, loc=loc)
        return arg

    def _handle_metadata(self):
        """remove distributed input annotation from locals and add to metadata"""
        if "distributed" not in self.metadata:
            # TODO: keep updated in variable renaming?
            self.metadata["distributed"] = self.flags.distributed.copy()

        if "distributed_block" not in self.metadata:
            self.metadata["distributed_block"] = self.flags.distributed_block.copy()

        if "threaded" not in self.metadata:
            self.metadata["threaded"] = self.flags.threaded.copy()

        if "is_return_distributed" not in self.metadata:
            self.metadata["is_return_distributed"] = False

        # store args_maybe_distributed to be used in distributed_analysis of a potential
        # JIT caller
        if "args_maybe_distributed" not in self.metadata:
            self.metadata["args_maybe_distributed"] = self.flags.args_maybe_distributed

        # handle old input flags
        # e.g. {"A:input": "distributed"} -> "A"
        dist_inputs = {
            var_name.split(":")[0]
            for (var_name, flag) in self.locals.items()
            if var_name.endswith(":input") and flag == "distributed"
        }

        thread_inputs = {
            var_name.split(":")[0]
            for (var_name, flag) in self.locals.items()
            if var_name.endswith(":input") and flag == "threaded"
        }

        # check inputs to be in actuall args
        for arg_name in dist_inputs | thread_inputs:
            if arg_name not in self.func_ir.arg_names:
                raise BodoError(
                    "distributed input {} not found in arguments".format(arg_name)
                )
            self.locals.pop(arg_name + ":input")

        self.metadata["distributed"] |= dist_inputs
        self.metadata["threaded"] |= thread_inputs

        # handle old return flags
        # e.g. {"A:return":"distributed"} -> "A"
        flagged_returns = {
            var_name.split(":")[0]: flag
            for (var_name, flag) in self.locals.items()
            if var_name.endswith(":return")
        }

        for v, flag in flagged_returns.items():
            if flag == "distributed":
                self.metadata["distributed"].add(v)
            elif flag == "threaded":
                self.metadata["threaded"].add(v)

            self.locals.pop(v + ":return")

    def _run_return(self, ret_node):
        # TODO: handle distributed analysis, requires handling variable name
        # change in simplify() and replace_var_names()
        flagged_vars = (
            self.metadata["distributed"]
            | self.metadata["distributed_block"]
            | self.metadata["threaded"]
        )
        all_returns_distributed = self.flags.all_returns_distributed
        nodes = [ret_node]
        cast = guard(get_definition, self.func_ir, ret_node.value)
        assert cast is not None, "return cast not found"
        assert isinstance(cast, ir.Expr) and cast.op == "cast"
        scope = cast.value.scope
        loc = cast.loc
        # XXX: using split('.') since the variable might be renamed (e.g. A.2)
        ret_name = cast.value.name.split(".")[0]
        # save return name to catch invalid dist annotations
        self._return_varnames.add(ret_name)

        if ret_name in flagged_vars or all_returns_distributed:
            flag = (
                "distributed"
                if (
                    ret_name in self.metadata["distributed"]
                    or ret_name in self.metadata["distributed_block"]
                    or all_returns_distributed
                )
                else "threaded"
            )
            # save in metadata that the return value is distributed
            # TODO(ehsan): support other flags like distributed_block?
            if flag == "distributed":
                self.metadata["is_return_distributed"] = True
            nodes = self._gen_replace_dist_return(cast.value, flag)
            new_arr = nodes[-1].target
            new_cast = ir.Expr.cast(new_arr, loc)
            new_out = ir.Var(scope, mk_unique_var(flag + "_return"), loc)
            nodes.append(ir.Assign(new_cast, new_out, loc))
            ret_node.value = new_out
            nodes.append(ret_node)
            return nodes

        cast_def = guard(get_definition, self.func_ir, cast.value)
        if (
            cast_def is not None
            and isinstance(cast_def, ir.Expr)
            and cast_def.op == "build_tuple"
        ):
            nodes = []
            new_var_list = []
            tup_varnames = []
            for v in cast_def.items:
                vname = v.name.split(".")[0]
                self._return_varnames.add(vname)
                tup_varnames.append(vname)
                if vname in flagged_vars or all_returns_distributed:
                    flag = (
                        "distributed"
                        if (
                            vname in self.metadata["distributed"]
                            or all_returns_distributed
                        )
                        else "threaded"
                    )
                    nodes += self._gen_replace_dist_return(v, flag)
                    new_var_list.append(nodes[-1].target)
                else:
                    new_var_list.append(v)
            # store a list of distributions for tuple return case
            self.metadata["is_return_distributed"] = [
                v in self.metadata["distributed"] for v in tup_varnames
            ]
            new_tuple_node = ir.Expr.build_tuple(new_var_list, loc)
            new_tuple_var = ir.Var(scope, mk_unique_var("dist_return_tp"), loc)
            nodes.append(ir.Assign(new_tuple_node, new_tuple_var, loc))
            new_cast = ir.Expr.cast(new_tuple_var, loc)
            new_out = ir.Var(scope, mk_unique_var("dist_return"), loc)
            nodes.append(ir.Assign(new_cast, new_out, loc))
            ret_node.value = new_out
            nodes.append(ret_node)

        return nodes

    def _gen_replace_dist_return(self, var, flag):
        if flag == "distributed":

            func_text = (
                ""
                "def f(_dist_arr):\n"
                "    dist_return = bodo.libs.distributed_api.dist_return(_dist_arr)\n"
            )

        elif flag == "threaded":

            func_text = (
                ""
                "def f(_threaded_arr):\n"
                "    _th_arr = bodo.libs.distributed_api.threaded_return(_threaded_arr)\n"
            )

        else:
            raise BodoError("Invalid return flag {}".format(flag))
        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        f_block = compile_to_numba_ir(loc_vars["f"], {"bodo": bodo}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [var])
        return f_block.body[:-3]  # remove none return

    def _fix_dict_typing(self, var):
        """replace dict variable's definition to be non-dict to avoid Numba's typing
        issues for heterogenous dictionaries. E.g. {"A": int, "B": "str"}
        TODO(ehsan): fix in Numba and avoid this workaround
        """
        var_def = guard(get_definition, self.func_ir, var)
        if is_expr(var_def, "build_map"):
            var_def.op = "build_list"
            var_def.items = [v[0] for v in var_def.items]
        elif isinstance(var_def, (ir.Global, ir.FreeVar, ir.Const)):
            var_def.value = 11  # arbitrary value that can be typed


def remove_dead_branches(func_ir):
    """
    Remove branches that have a compile-time constant as condition.
    Similar to dead_branch_prune() of Numba, but dead_branch_prune() only focuses on
    binary expressions in conditions, not simple constants like global values.
    """
    for block in func_ir.blocks.values():
        assert len(block.body) > 0
        last_stmt = block.body[-1]
        if isinstance(last_stmt, ir.Branch):
            # handle const bool() calls like bool(False)
            cond = last_stmt.cond
            cond_def = guard(get_definition, func_ir, cond)
            if (
                guard(find_callname, func_ir, cond_def)
                in (("bool", "numpy"), ("bool", "builtins"))
                and guard(find_const, func_ir, cond_def.args[0]) is not None
            ):
                cond = cond_def.args[0]
            try:
                cond_val = find_const(func_ir, cond)
                target_label = last_stmt.truebr if cond_val else last_stmt.falsebr
                block.body[-1] = ir.Jump(target_label, last_stmt.loc)
            except GuardException:
                pass

    # Remove dead blocks using CFG
    cfg = compute_cfg_from_blocks(func_ir.blocks)
    for dead in cfg.dead_nodes():
        del func_ir.blocks[dead]


def _dtype_val_to_arr_type(t, func_name, loc):
    """get array type from type value 't' specified in calls like read_csv()
    e.g. "str" -> string_array_type
    """
    if t == object:
        # TODO: Add a link to IO dtype documentation when available
        raise BodoError(
            f"{func_name}() 'dtype' does not support object dtype.", loc=loc
        )

    if t in ("str", str, "unicode"):
        return string_array_type

    if isinstance(t, str):
        if t.startswith("Int") or t.startswith("UInt"):
            dtype = bodo.libs.int_arr_ext.typeof_pd_int_dtype(
                pd.api.types.pandas_dtype(t), None
            )
            return IntegerArrayType(dtype.dtype)

        # datetime64 case
        if t == "datetime64[ns]":
            return types.Array(types.NPDatetime("ns"), 1, "C")

        t = "int64" if t == "int" else t
        t = "float64" if t == "float" else t
        t = "bool_" if t == "bool" else t
        # XXX: bool with NA needs to be object, TODO: fix somehow? doc.
        t = "bool_" if t == "O" else t

        if t == "bool_":
            return boolean_array

        typ = getattr(types, t)
        typ = types.Array(typ, 1, "C")
        return typ

    if t == int:
        return types.Array(types.int64, 1, "C")

    if t == float:
        return types.Array(types.float64, 1, "C")

    # categorical type
    if isinstance(t, pd.CategoricalDtype):
        cats = tuple(t.categories)
        elem_typ = bodo.string_type if len(cats) == 0 else bodo.typeof(cats[0])
        typ = PDCategoricalDtype(cats, elem_typ, t.ordered)
        return CategoricalArrayType(typ)

    # nullable int types
    if isinstance(t, pd.core.arrays.integer._IntegerDtype):
        dtype = bodo.libs.int_arr_ext.typeof_pd_int_dtype(t, None)
        return IntegerArrayType(dtype.dtype)

    # try numpy dtypes
    try:
        dtype = numba.np.numpy_support.from_dtype(t)
        return types.Array(dtype, 1, "C")
    except:
        pass

    raise BodoError(f"{func_name}() 'dtype' does not support {t}", loc=loc)


def _get_col_ind_from_name_or_ind(c, col_names):
    """get column index from column name or index"""
    # TODO(ehsan): error checking
    if isinstance(c, int) and c not in col_names:
        return c
    return col_names.index(c)


class JSONFileInfo(FileInfo):
    """FileInfo object passed to ForceLiteralArg for
    file name arguments that refer to a JSON dataset"""

    def __init__(self, orient, convert_dates, precise_float, lines, compression):
        self.orient = orient
        self.convert_dates = convert_dates
        self.precise_float = precise_float
        self.lines = lines
        self.compression = compression
        super().__init__()

    def _get_schema(self, fname):
        return _get_json_df_type_from_file(
            fname,
            self.orient,
            self.convert_dates,
            self.precise_float,
            self.lines,
            self.compression,
        )


def _get_json_df_type_from_file(
    fname_const, orient, convert_dates, precise_float, lines, compression
):
    """get dataframe type for read_json() using file path constant or raise error if
    path is invalid.
    Only rank 0 looks at the file to infer df type, then broadcasts.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    # dataframe type or Exception raised trying to find the type
    df_type_or_e = None
    if bodo.get_rank() == 0:
        from bodo.io.fs_io import find_file_name_or_handler

        is_handler = None
        try:
            rows_to_read = 20  # TODO: tune this
            is_handler, file_name_or_handler, f_size, _ = find_file_name_or_handler(
                fname_const, "json"
            )
            if is_handler and compression == "infer":
                # pandas can't infer compression without filename, we need to do it
                if fname_const.endswith(".gz"):
                    compression = "gzip"
                elif fname_const.endswith(".bz2"):
                    compression = "bz2"
                else:
                    compression = None

            # Ideally, we should use chunksize to read the number of rows desired
            # instead of manually reading the number of rows into a buffer.
            # There is a issue for it in pandas:
            # https://github.com/pandas-dev/pandas/issues/27135
            # read() is used instead of readline() because
            # Pyarrow's hdfs open() returns stream (NativeFile) that does not
            # readline() implemented, but seems like they are working on it:
            # https://issues.apache.org/jira/browse/ARROW-7584

            file_name_or_buff = file_name_or_handler

            # TODO read only `rows_to_read` of compressed files
            if is_handler and compression is None:
                read_chunk_size = 500  # max number of bytes read at a time
                rows_read = 0  # rows seen
                size_read = 0  # number of bytes seen
                buff = ""  # bytes read
                # keep reading until we read at least rows_to_read number of rows
                while size_read < f_size and rows_read < rows_to_read:
                    read_size = min(read_chunk_size, f_size - size_read)
                    tmp_buff = file_name_or_handler.read(read_size).decode("utf-8")
                    rows_read += tmp_buff.count("\n")
                    buff += tmp_buff
                    size_read += read_size
                file_name_or_buff = buff

            df = pd.read_json(
                file_name_or_buff,
                orient=orient,
                convert_dates=convert_dates,
                precise_float=precise_float,
                lines=lines,
                compression=compression,
                # chunksize=rows_to_read,
            )

            # TODO: categorical, etc.
            df_type_or_e = numba.typeof(df)
            # always convert to nullable type since initial rows of a column could be all
            # int for example, but later rows could have NAs
            df_type_or_e = to_nullable_type(df_type_or_e)
        except Exception as e:
            df_type_or_e = e
        finally:
            if is_handler:
                file_name_or_handler.close()

    df_type_or_e = comm.bcast(df_type_or_e)

    # raise error on all processors if found (not just rank 0 which would cause hangs)
    if isinstance(df_type_or_e, Exception):
        raise BodoError(
            f"error from: {type(df_type_or_e).__name__}: {str(df_type_or_e)}\n"
        )

    return df_type_or_e


def _get_excel_df_type_from_file(
    fname_const, sheet_name, skiprows, header, comment, date_cols
):
    """get dataframe type for read_excel() using file path constant.
    Only rank 0 looks at the file to infer df type, then broadcasts.
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    df_type_or_e = None
    if bodo.get_rank() == 0:
        try:
            rows_to_read = 100  # TODO: tune this
            df = pd.read_excel(
                fname_const,
                sheet_name=sheet_name,
                nrows=rows_to_read,
                skiprows=skiprows,
                header=header,
                # index_col=index_col,
                comment=comment,
                parse_dates=date_cols,
            )
            df_type_or_e = numba.typeof(df)
            # always convert to nullable type since initial rows of a column could be all
            # int for example, but later rows could have NAs
            df_type_or_e = to_nullable_type(df_type_or_e)
        except Exception as e:
            df_type_or_e = e

    df_type_or_e = comm.bcast(df_type_or_e)
    # raise error on all processors if found (not just rank 0 which would cause hangs)
    if isinstance(df_type_or_e, Exception):
        raise BodoError(df_type_or_e)

    return df_type_or_e


def _get_sql_df_type_from_db(sql_const, con_const, db_type):
    """access the database to find df type for read_sql() output.
    Only rank zero accesses the database, then broadcasts.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    df_info = None
    if bodo.get_rank() == 0:
        # Any columns that had their name converted. These need to be reverted
        # in any dead column elimination
        converted_colnames = set()
        rows_to_read = 100  # TODO: tune this
        sql_call = f"select * from ({sql_const}) x LIMIT {rows_to_read}"
        if db_type == "snowflake":
            import snowflake.connector
            from bodo.io.snowflake import get_connection_params

            conn_params = get_connection_params(con_const)
            conn = snowflake.connector.connect(**conn_params)
            df = conn.cursor().execute(sql_call).fetch_pandas_all()

            # Ensure column name case matches Pandas/sqlalchemy. See:
            # https://github.com/snowflakedb/snowflake-sqlalchemy#object-name-case-handling
            # If a name is returned as all uppercase by the Snowflake connector
            # it means it is case insensitive or it was inserted as all
            # uppercase with double quotes. In both of these situations
            # pd.read_sql() returns the name with all lower case
            new_colnames = []
            for x in df.columns:
                if x.isupper():
                    converted_colnames.add(x.lower())
                    new_colnames.append(x.lower())
                else:
                    new_colnames.append(x)
            df.columns = new_colnames
        else:
            try:
                import sqlalchemy  # noqa
            except ImportError:  # pragma: no cover
                message = (
                    "Using URI string without sqlalchemy installed."
                    " sqlalchemy can be installed by calling"
                    " 'conda install -c conda-forge sqlalchemy'."
                )
                raise BodoError(message)
            df = pd.read_sql(sql_call, con_const)
        df_type = numba.typeof(df)
        # always convert to nullable type since initial rows of a column could be all
        # int for example, but later rows could have NAs
        df_type = to_nullable_type(df_type)
        df_info = (df_type, converted_colnames)

    df_type, converted_colnames = comm.bcast(df_info)
    return df_type, converted_colnames


class CSVFileInfo(FileInfo):
    """FileInfo object passed to ForceLiteralArg for
    file name arguments that refer to a CSV dataset"""

    def __init__(self, sep, skiprows, header, compression):
        self.sep = sep
        self.skiprows = skiprows
        self.header = header
        self.compression = compression
        super().__init__()

    def _get_schema(self, fname):
        return _get_csv_df_type_from_file(
            fname, self.sep, self.skiprows, self.header, self.compression
        )


def _get_csv_df_type_from_file(fname_const, sep, skiprows, header, compression):
    """get dataframe type for read_csv() using file path constant or raise error if not
    possible (e.g. file doesn't exist).
    If fname_const points to a directory, find a non-empty csv file from
    the directory.
    For posix, pass the file name directly to pandas. For s3 & hdfs, open the
    file reader, and pass it to pandas.
    Only rank 0 looks at the file to infer df type, then broadcasts.
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    # dataframe type or Exception raised trying to find the type
    df_type_or_e = None
    if bodo.get_rank() == 0:
        from bodo.io.fs_io import find_file_name_or_handler

        is_handler = None
        try:
            is_handler, file_name_or_handler, _, _ = find_file_name_or_handler(
                fname_const, "csv"
            )

            if is_handler and compression == "infer":
                # pandas can't infer compression without filename, we need to do it
                if fname_const.endswith(".gz"):
                    compression = "gzip"
                elif fname_const.endswith(".bz2"):
                    compression = "bz2"
                elif fname_const.endswith(".zip"):
                    compression = "zip"
                elif fname_const.endswith(".xz"):
                    compression = "xz"
                else:
                    compression = None

            rows_to_read = 100  # TODO: tune this
            df = pd.read_csv(
                file_name_or_handler,
                sep=sep,
                nrows=rows_to_read,
                skiprows=skiprows,
                header=header,
                compression=compression,
            )

            # TODO: categorical, etc.
            df_type_or_e = numba.typeof(df)
            # always convert to nullable type since initial rows of a column could be all
            # int for example, but later rows could have NAs
            df_type_or_e = to_nullable_type(df_type_or_e)
        except Exception as e:
            df_type_or_e = e
        finally:
            if is_handler:
                file_name_or_handler.close()

    df_type_or_e = comm.bcast(df_type_or_e)

    # raise error on all processors if found (not just rank 0 which would cause hangs)
    if isinstance(df_type_or_e, Exception):
        raise BodoError(
            f"error from: {type(df_type_or_e).__name__}: {str(df_type_or_e)}\n"
        )

    return df_type_or_e


def _check_type(val, typ):
    """check whether "val" is of type "typ", or any type in "typ" if "typ" is a list"""
    if isinstance(typ, list):
        return any(isinstance(val, t) for t in typ)
    return isinstance(val, typ)


def _check_int_list(list_val):
    """ check whether list_val is list/tuple and its elements are of type int"""
    return isinstance(list_val, (list, tuple)) and all(
        isinstance(val, int) for val in list_val
    )
