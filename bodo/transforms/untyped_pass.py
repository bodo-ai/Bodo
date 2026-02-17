"""
transforms the IR to remove features that Numba's type inference cannot support
such as non-uniform dictionary input of `pd.DataFrame({})`.
"""

from __future__ import annotations

import datetime
import itertools
import sys
import types as pytypes
import warnings
from typing import TYPE_CHECKING

import numba
import numpy as np
import pandas as pd
from numba.core import ir, ir_utils, types
from numba.core.ir_utils import (
    GuardException,
    build_definitions,
    compile_to_numba_ir,
    compute_cfg_from_blocks,
    dprint_func_ir,
    find_callname,
    find_const,
    find_topo_order,
    get_definition,
    guard,
    mk_unique_var,
    replace_arg_nodes,
)
from numba.core.registry import CPUDispatcher

import bodo
import bodo.hiframes.pd_dataframe_ext
import bodo.io
import bodo.io.utils
import bodo.ir
import bodo.ir.aggregate
import bodo.ir.join
import bodo.ir.sort
import bodo.pandas as bd
from bodo.hiframes.pd_categorical_ext import CategoricalArrayType, PDCategoricalDtype
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_index_ext import RangeIndexType
from bodo.io import h5
from bodo.ir import csv_ext, json_ext, parquet_ext, sql_ext
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.numba_compat import mini_dce
from bodo.utils.transform import (
    compile_func_single_block,
    fix_struct_return,
    get_call_expr_arg,
    get_const_arg,
    get_const_value,
    get_const_value_inner,
    get_runtime_join_filter_terms,
    set_call_expr_arg,
    update_node_list_definitions,
)
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    ColNamesMetaType,
    FileInfo,
    raise_bodo_error,
    to_nullable_type,
    to_str_arr_if_dict_array,
)
from bodo.utils.utils import (
    check_java_installation,
    is_assign,
    is_call,
    is_expr,
)

# Imports for typechecking
if TYPE_CHECKING:  # pragma: no cover
    from snowflake.connector import SnowflakeConnection


class UntypedPass:
    """
    Transformations before typing to enable type inference.
    This pass transforms the IR to remove operations that cannot be handled in Numba's
    type inference due to complexity such as pd.read_csv().
    """

    def __init__(
        self,
        func_ir,
        typingctx,
        args,
        _locals,
        metadata,
        flags,
        is_independent: bool = False,
    ):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.args = args
        self.locals = _locals
        self.metadata = metadata
        self.flags = flags
        # TODO: remove this? update _the_max_label just in case?
        ir_utils._the_max_label.update(max(func_ir.blocks.keys()))

        self.arrow_tables = {}
        self.pq_handler = parquet_ext.ParquetHandler(func_ir, typingctx, args, _locals)
        self.h5_handler = h5.H5_IO(self.func_ir, _locals, flags, args)
        # save names of arguments and return values to catch invalid dist annotation
        self._arg_names = set()
        self._return_varnames = set()
        # Is code executing independently on each rank?
        self._is_independent = is_independent
        self._snowflake_conn_cache = {}

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

        for label in topo_order:
            block = blocks[label]
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
        # remove variables that are now dead due to transformations that could cause typing issues
        # (see test_empty_dataframe_creation)
        mini_dce(self.func_ir)
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
                    f"distributed. Ignoring the flag for variables: {extra_vars}."
                )
            )
        # same for "replicated" flag
        extra_vars = (
            self.metadata["replicated"] - self._return_varnames - self._arg_names
        )
        if extra_vars and bodo.get_rank() == 0:
            warnings.warn(
                BodoWarning(
                    "Only function arguments and return values can be specified as "
                    f"replicated. Ignoring the flag for variables: {extra_vars}."
                )
            )

        # clear connection cache to close connections
        self._snowflake_conn_cache.clear()

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
            self.locals[meta_var.name] = bodo.utils.typing.MetaType(tuple(pivot_values))

        # save arg name to catch invalid dist annotations
        if isinstance(rhs, ir.Arg):
            self._arg_names.add(rhs.name)

        # Throw proper error if the user has installed an unsupported HDF5 version
        # see [BE-1382]
        # TODO(ehsan): the code may not use "h5py" directly ("from h5py import File")
        # but that's rare and not high priority at this time
        if (
            isinstance(rhs, (ir.Const, ir.Global, ir.FreeVar))
            and isinstance(rhs.value, pytypes.ModuleType)
            and rhs.value.__name__ == "h5py"
            and not bodo.utils.utils.has_supported_h5py()
        ):  # pragma: no cover
            raise BodoError("Bodo requires HDF5 >=1.10 for h5py support", rhs.loc)

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
            and val_def.value in (pd, bd)
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
            if isinstance(mod_def, ir.Global) and mod_def.value in (pd, bd):
                return compile_func_single_block(
                    eval("lambda: bodo.hiframes.pd_multi_index_ext.from_product"),
                    (),
                    assign.target,
                )

        # multiIndex unsupported class methods
        if (
            (
                rhs.attr == "from_arrays"
                or rhs.attr == "from_tuples"
                or rhs.attr == "from_frame"
            )
            and is_expr(val_def, "getattr")
            and val_def.attr == "MultiIndex"
        ):  # pragma: no cover
            mod_def = guard(get_definition, self.func_ir, val_def.value)
            if isinstance(mod_def, ir.Global) and mod_def.value in (pd, bd):
                raise bodo.utils.typing.BodoError(
                    f"pandas.MultiIndex.{rhs.attr}() is not yet supported"
                )

        # IntervalIndex unsupported class methods
        if (
            (
                rhs.attr == "from_arrays"
                or rhs.attr == "from_tuples"
                or rhs.attr == "from_breaks"
            )
            and is_expr(val_def, "getattr")
            and val_def.attr == "IntervalIndex"
        ):  # pragma: no cover
            mod_def = guard(get_definition, self.func_ir, val_def.value)
            if isinstance(mod_def, ir.Global) and mod_def.value in (pd, bd):
                raise bodo.utils.typing.BodoError(
                    f"pandas.IntervalIndex.{rhs.attr}() is not yet supported"
                )

        # RangeIndex unsupported class methods
        if (
            rhs.attr == "from_range"
            and is_expr(val_def, "getattr")
            and val_def.attr == "RangeIndex"
        ):  # pragma: no cover
            mod_def = guard(get_definition, self.func_ir, val_def.value)
            if isinstance(mod_def, ir.Global) and mod_def.value in (pd, bd):
                raise bodo.utils.typing.BodoError(
                    f"pandas.RangeIndex.{rhs.attr}() is not yet supported"
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
            if isinstance(mod_def, ir.Global) and mod_def.value in (pd, bd):
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

        # Handle timedelta unsupported class methods/attrs
        if rhs.attr in ["max", "min", "resolution"]:
            if is_expr(val_def, "getattr") and val_def.attr == "Timedelta":
                mod_def = guard(get_definition, self.func_ir, val_def.value)
                is_pd_Timedelta = isinstance(mod_def, ir.Global) and mod_def.value in (
                    pd,
                    bd,
                )
            else:
                # Handle relative imports by checking if the value matches importing from Python
                is_pd_Timedelta = isinstance(
                    val_def, (ir.Global, ir.FreeVar)
                ) and val_def.value in (pd.Timedelta, bd.Timedelta)
            if is_pd_Timedelta:
                raise BodoError(f"pandas.Timedelta.{rhs.attr} not yet supported.")

        # Unsupported pd.Timestamp class methods
        if rhs.attr in [
            "combine",
            "fromisocalendar",
            "fromisoformat",
            "fromordinal",
            "fromtimestamp",
            "today",
            "utcfromtimestamp",
            "utcnow",
            "max",
            "min",
            "resolution",
        ]:
            is_timestamp_unsupported = False
            if is_expr(val_def, "getattr") and val_def.attr == "Timestamp":
                mod_def = guard(get_definition, self.func_ir, val_def.value)
                is_timestamp_unsupported = isinstance(
                    mod_def, ir.Global
                ) and mod_def.value in (pd, bd)
            else:
                # Handle relative imports by checking if the value matches importing from Python
                is_timestamp_unsupported = isinstance(
                    val_def, (ir.Global, ir.FreeVar)
                ) and val_def.value in (pd.Timestamp, bd.Timestamp)
            if is_timestamp_unsupported:
                raise BodoError("pandas.Timestamp." + rhs.attr + " not supported yet")

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
            "pyspark" in sys.modules
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
            and val_def.value is bytes
        ):
            return compile_func_single_block(
                eval("lambda: bodo.libs.binary_arr_ext.bytes_fromhex"),
                (),
                assign.target,
            )

        return [assign]

    def _run_getitem(self, assign, rhs, label):
        # fix type for f['A'][:] dset reads
        if "h5py" in sys.modules and bodo.utils.utils.has_supported_h5py():
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

        if fdef == ("dict", "builtins"):
            return self._handle_dict(assign, lhs, rhs)

        # Erroring for SnowflakeCatalog.from_conn_str inside of JIT
        if fdef == ("from_conn_str", "bodosql.SnowflakeCatalog"):
            raise BodoError(
                "SnowflakeCatalog.from_conn_str: This constructor can not be called from inside of a Bodo-JIT function. Please construct the SnowflakeCatalog in regular Python."
            )

        # handling pd.DataFrame() here since input can be constant dictionary
        if fdef in (("DataFrame", "pandas"), ("DataFrame", "bodo.pandas")):
            return self._handle_pd_DataFrame(assign, lhs, rhs, label)

        # handling pd.read_csv() here since input can have constants
        # like dictionaries for typing
        if fdef in (("read_csv", "pandas"), ("read_csv", "bodo.pandas")):
            return self._handle_pd_read_csv(assign, lhs, rhs, label)

        # handling pd.read_sql() here since input can have constants
        # like dictionaries for typing
        if fdef in (("read_sql", "pandas"), ("read_sql", "bodo.pandas")):
            return self._handle_pd_read_sql(assign, lhs, rhs, label)

        # handling pd.read_json() here since input can have constants
        # like dictionaries for typing
        if fdef in (("read_json", "pandas"), ("read_json", "bodo.pandas")):
            return self._handle_pd_read_json(assign, lhs, rhs, label)

        # handling pd.read_excel() here since typing info needs to be extracted
        if fdef in (("read_excel", "pandas"), ("read_excel", "bodo.pandas")):
            return self._handle_pd_read_excel(assign, lhs, rhs, label)

        # match flatmap pd.Series(list(itertools.chain(*A))) and flatten
        if fdef in (("Series", "pandas"), ("Series", "bodo.pandas")):
            return self._handle_pd_Series(assign, lhs, rhs)

        # replace pd.NamedAgg() with equivalent tuple to be handled in groupby typing
        if fdef in (("NamedAgg", "pandas"), ("NamedAgg", "bodo.pandas")):
            return self._handle_pd_named_agg(assign, lhs, rhs)

        # replace bodo.ExtendedNamedAgg() with equivalent tuple to be handled in groupby typing
        if fdef == ("ExtendedNamedAgg", "bodo.utils.utils"):
            return self._handle_extended_named_agg(assign, lhs, rhs)

        if fdef == ("read_table", "pyarrow.parquet"):
            return self._handle_pq_read_table(assign, lhs, rhs)

        if (
            func_name == "to_pandas"
            and isinstance(func_mod, ir.Var)
            and func_mod.name in self.arrow_tables
        ):
            return self._handle_pq_to_pandas(assign, lhs, rhs, func_mod)

        if fdef in (("read_parquet", "pandas"), ("read_parquet", "bodo.pandas")):
            return self._handle_pd_read_parquet(assign, lhs, rhs)

        if fdef == ("fromfile", "numpy"):
            return self._handle_np_fromfile(assign, lhs, rhs)

        if fdef == ("where", "numpy") and len(rhs.args) == 3:
            return self._handle_np_where(assign, lhs, rhs)

        if fdef == ("where", "numpy") and len(rhs.args) == 1:
            return self._handle_np_where_one_arg(assign, lhs, rhs)

        # If there is a relative import it uses ('BodoSQLContext', 'bodosql.context')
        if fdef in (
            ("BodoSQLContext", "bodosql.context"),
            ("BodoSQLContext", "bodosql"),
        ):  # pragma: no cover
            return self._handle_bodosql_BodoSQLContext(assign, lhs, rhs, label)

        # add distributed flag input to SparkDataFrame.toPandas() if specified by user
        if (
            "pyspark" in sys.modules
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
                            "lambda A, t: bodo.utils.typing.check_objmode_output_type(A, t)"
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

        # convert_code_obj_to_function() replaces nested functions/lambdas with jitted
        # calls to enable jit compilation. However, functions inside objmode shouldn't
        # be jitted so we revert them back to regular functions here
        # see test_heterogeneous_series_box
        for block in objmode_val.func_ir.blocks.values():
            for stmt in block.body:
                if (
                    is_assign(stmt)
                    and isinstance(stmt.value, (ir.Global, ir.FreeVar, ir.Const))
                    and isinstance(stmt.value.value, CPUDispatcher)
                    and getattr(stmt.value.value, "is_nested_func", False)
                ):
                    stmt.value.value = stmt.value.value.py_func

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

    def _handle_dict(
        self, assign: ir.Assign, lhs: ir.Var, rhs: ir.Expr
    ) -> list[ir.Assign]:
        """Optimization step to convert function calls for dict()
        into Python dictionary literals whenever possible. Literal dictionaries
        in Python use the BUILD_MAP instruction, which is more efficient and used
        by Bodo/Numba for analysis of constant dictionaries. This cannot be done in
        regular Python because any Python programmer can override the dict() function,
        but this is not possible in Bodo/Numba, making this safe.
        https://madebyme.today/blog/python-dict-vs-curly-brackets/

        Right now we only support the case where the dict() function is called without
        any arguments. This is a common pattern if people prefer the readability of dict()
        over {}. We can easily support more cases in the future if needed. If the call to
        dict() is not safe to convert to a dictionary literal, we will just return the
        original assign node.

        Note: The caller is expected to validate that the input is a call to dict().

        Args:
            assign (ir.Assign): The original assignment. We return this value if we cannot
                convert the dict() call to a dictionary literal.
            lhs (ir.Var): The target variable of the assignment. This will be our new target
                variable if we can convert the dict() call to a dictionary literal.
            rhs (ir.Expr): The right-hand side of the assignment. This is the dict() call
                that we will analyze to see if we can convert it to a dictionary literal.

        Returns:
            list[ir.Assign]: An 1 element list containing either a new `build_map` call or the original
                `dict()` call as an assignment.
        """
        has_args = (
            len(rhs.args) > 0
            or len(rhs.kws) > 0
            or rhs.vararg is not None
            or rhs.varkwarg is not None
        )
        # TODO: Support more complex cases like converting
        # dict(x=1, y=2, z=3) to {"x": 1, "y": 2, "z": 3}
        if has_args:
            return [assign]
        else:
            # Create a new dictionary literal
            new_dict = ir.Expr.build_map([], 0, {}, {}, rhs.loc)
            return [ir.Assign(new_dict, lhs, lhs.loc)]

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
            # TODO[BSE-4021]: Verify the build_map is not modified in the IR.
            (
                tup_vars,
                new_nodes,
            ) = bodo.utils.transform._convert_const_key_dict(
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
            # arg_def will be removed if not used anywhere else

        # replace range() with pd.RangeIndex() for index argument
        arg_def = guard(get_definition, self.func_ir, index_arg)
        if is_call(arg_def) and guard(find_callname, self.func_ir, arg_def) == (
            "range",
            "builtins",
        ):
            # gen pd.RangeIndex() call
            func_text = "def _call_range_index():\n    return pd.RangeIndex()\n"

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
        import bodosql.compiler  # isort:skip # noqa

        kws = dict(rhs.kws)
        data_arg = get_call_expr_arg(
            "bodosql.BodoSQLContext", rhs.args, kws, 0, "tables"
        )
        arg_def = guard(get_definition, self.func_ir, data_arg)
        msg = "bodosql.BodoSQLContext(): 'tables' argument should be a dictionary with constant string keys"
        if not is_expr(arg_def, "build_map"):
            raise BodoError(msg)

        # TODO[BSE-4021]: Verify the build_map is not modified in the IR.
        (
            tup_vars,
            new_nodes,
        ) = bodo.utils.transform._convert_const_key_dict(
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

    def _handle_pd_read_sql(self, assign, lhs: ir.Var, rhs: ir.Expr, label):
        """transform pd.read_sql calls"""
        # schema: pd.read_sql(sql, con, index_col=None,
        # coerce_float=True, params=None, parse_dates=None,
        # columns=None, chunksize=None, _bodo_read_as_dict,
        # _bodo_is_table_input, _bodo_downcast_decimal_to_double,
        # _bodo_read_as_table, _bodo_orig_table_name,
        # _bodo_orig_table_indices, _bodo_sql_op_id,
        # _bodo_runtime_join_filters)
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
        index_col = get_const_arg(
            "read_sql",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            2,
            "index_col",
            rhs.loc,
            default=-1,
        )
        # If defined, perform batched reads on the table
        # Only supported for Snowflake tables
        # Note that we don't want to use Pandas chunksize, since it returns an iterator
        chunksize: int | None = get_const_arg(
            "read_sql",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e4,
            "_bodo_chunksize",
            rhs.loc,
            use_default=True,
            default=None,
            typ="int",
        )
        if chunksize is not None and (
            not isinstance(chunksize, int) or chunksize < 1
        ):  # pragma: no cover
            raise BodoError(
                "pd.read_sql() '_bodo_chunksize' must be a constant integer >= 1."
            )

        # Users can use this to specify what columns should be read in as
        # dictionary-encoded string arrays. This is in addition
        # to whatever columns bodo determines should be read in
        # with dictionary encoding. This is only supported when
        # reading from Snowflake.
        _bodo_read_as_dict = get_const_arg(
            "read_sql",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e4,
            "_bodo_read_as_dict",
            rhs.loc,
            use_default=True,
            default=None,
        )
        # TODO[BE-4362]: detect table input automatically
        _bodo_is_table_input: bool = get_const_arg(
            "read_sql",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e5,
            "_bodo_is_table_input",
            rhs.loc,
            default=False,
        )
        # Allow Unsafe Downcasting to Double when we dont support decimal computation
        _bodo_downcast_decimal_to_double: bool = get_const_arg(
            "read_sql",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e5,
            "_bodo_downcast_decimal_to_double",
            rhs.loc,
            default=False,
        )
        _bodo_read_as_table: bool = get_const_arg(
            "read_sql",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e5,
            "_bodo_read_as_table",
            rhs.loc,
            default=False,
        )

        _bodo_orig_table_name_const: str | None = get_const_arg(
            "read_sql",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e5,
            "_bodo_orig_table_name",
            rhs.loc,
            default=None,
            use_default=True,
        )

        _bodo_orig_table_indices_const: tuple[int] | None = get_const_arg(
            "read_sql",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e5,
            "_bodo_orig_table_indices",
            rhs.loc,
            default=None,
            use_default=True,
        )

        # Operator ID assigned by the planner for query profile purposes.
        # Only applicable in the streaming case.
        _bodo_sql_op_id_const: int = get_const_arg(
            "read_sql",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e5,
            "_bodo_sql_op_id",
            rhs.loc,
            default=-1,
            use_default=True,
            typ="int",
        )
        # Retrieve the tuple of runtime join filters in the form
        # ((state_1, indices_1), (state_2, indices_2)...) where each
        # state is a join state object and each indices is a tuple of
        # column indices.
        _bodo_runtime_join_filters_arg = get_call_expr_arg(
            "read_sql",
            rhs.args,
            kws,
            -1,
            "_bodo_runtime_join_filters",
            default=None,
            use_default=True,
        )
        rtjf_terms = get_runtime_join_filter_terms(
            self.func_ir, _bodo_runtime_join_filters_arg
        )
        if rtjf_terms is not None and len(rtjf_terms):
            assert chunksize is not None, (
                "Cannot provide rtjf_terms in a non-streaming read"
            )

        # coerce_float = self._get_const_arg(
        #     "read_sql", rhs.args, kws, 3, "coerce_float", default=True
        # )
        # params = self._get_const_arg("read_sql", rhs.args, kws, 4, "params", default=-1)
        # parse_dates = self._get_const_arg(
        #     "read_sql", rhs.args, kws, 5, "parse_dates", default=-1
        # )
        # columns = self._get_const_arg(
        #     "read_sql", rhs.args, kws, 6, "columns", rhs.loc, default=""
        # )

        # SUPPORTED:
        # sql is supported since it is fundamental
        # con is supported since it is fundamental but only as a string
        # index_col is supported since setting the index is something useful.
        # _bodo_chunksize is supported to enable batched reads for Snowflake
        # _bodo_sql_op_id is supported to enable query profile for batched Snowflake reads.
        # UNSUPPORTED:
        # chunksize is unsupported but can easily be extended from _bodo_chunksize
        # columns   is unsupported because selecting columns could actually be done in SQL.
        # parse_dates is unsupported because it requires remapping which subset of loaded columns should be updated.
        #        and needs to be implemented with snowflake.
        # coerce_float is currently unsupported but it could be useful to support it.
        # params is currently unsupported because not needed for mysql but surely will be needed later.
        supported_args = (
            "sql",
            "con",
            "index_col",
            "_bodo_chunksize",
            "_bodo_read_as_dict",
            "_bodo_is_table_input",
            "_bodo_downcast_decimal_to_double",
            "_bodo_read_as_table",
            "_bodo_orig_table_name",
            "_bodo_orig_table_indices",
            "_bodo_sql_op_id",
            "_bodo_runtime_join_filters",
        )

        unsupported_args = set(kws.keys()) - set(supported_args)
        if unsupported_args:
            raise BodoError(
                f"read_sql() arguments {unsupported_args} not supported yet"
            )

        # Generate the type info.
        (
            db_type,
            col_names,
            data_arrs,
            out_types,
            converted_colnames,
            unsupported_columns,
            unsupported_arrow_types,
            is_select_query,
            has_side_effects,
            pyarrow_table_schema,
        ) = _get_sql_types_arr_colnames(
            sql_const,
            con_const,
            _bodo_read_as_dict,
            lhs,
            rhs.loc,
            _bodo_is_table_input,
            self._is_independent,
            _bodo_downcast_decimal_to_double,
            _bodo_orig_table_name_const,
            _bodo_orig_table_indices_const,
            snowflake_conn_cache=self._snowflake_conn_cache,
        )

        if chunksize is not None and db_type != "snowflake":  # pragma: no cover
            raise BodoError(
                "pd.read_sql(): The `chunksize` argument is only supported for Snowflake table reads"
            )

        if (_bodo_sql_op_id_const != -1) and (db_type != "snowflake"):
            raise BodoError(
                "pd.read_sql(): The `_bodo_sql_op_id` argument is only supported for Snowflake table reads"
            )

        index_ind = None
        index_col_name = None
        index_arr_typ = types.none
        if index_col != -1 and index_col != False:
            # convert column number to column name
            if isinstance(index_col, int):
                index_ind = index_col
                index_col = col_names[index_col]
            else:
                index_ind = col_names.index(index_col)
            # Set the information for the index type
            index_col_name = index_col
            index_arr_typ = out_types[index_ind]
            # Remove the index column from the table.
            col_names.remove(index_col_name)
            out_types.remove(index_arr_typ)

        data_args = ["table_val", "idx_arr_val"]

        if index_ind is not None:
            # Convert the output array to index
            index_arg = f"bodo.utils.conversion.convert_to_index({data_args[1]}, {index_col_name!r})"
            index_type = bodo.utils.typing.index_typ_from_dtype_name_arr(
                index_arr_typ.dtype, index_col_name, index_arr_typ
            )
        else:
            # generate RangeIndex as default index
            index_arg = f"bodo.hiframes.pd_index_ext.init_range_index(0, len({data_args[0]}), 1, None)"
            index_type = RangeIndexType(None)

        # Create the output DataFrameType. This will be lowered as a Typeref for use with TableFormat
        df_type = DataFrameType(
            tuple(out_types),
            index_type,
            tuple(col_names),
            is_table_format=True,
        )
        col_meta = ColNamesMetaType(df_type.columns)

        if chunksize is not None:  # pragma: no cover
            # Create a new temp var so this is always exactly one variable
            data_arrs = [ir.Var(lhs.scope, mk_unique_var("arrow_iterator"), lhs.loc)]

        nodes = [
            sql_ext.SqlReader(
                sql_const,
                con_const,
                lhs.name,
                col_names,
                out_types,
                data_arrs,
                converted_colnames,
                db_type,
                lhs.loc,
                unsupported_columns,
                unsupported_arrow_types,
                is_select_query,
                has_side_effects,
                index_col_name,
                index_arr_typ,
                None,  # database_schema
                pyarrow_table_schema,
                _bodo_downcast_decimal_to_double,
                chunksize,
                _bodo_sql_op_id_const,
                rtjf_terms,
            )
        ]

        # _bodo_read_as_table = do not wrap the table in a DataFrame
        if chunksize is not None or _bodo_read_as_table:  # pragma: no cover
            nodes += [ir.Assign(data_arrs[0], lhs, lhs.loc)]
        else:
            # TODO: Pull out to helper function for most IO functions (except Iceberg)
            func_text = (
                f"def _init_df({data_args[0]}, {data_args[1]}):\n"
                f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(\n"
                f"    ({data_args[0]},), {index_arg}, __col_name_meta_value_pd_read_sql\n"
                f"  )\n"
            )
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            _init_df = loc_vars["_init_df"]

            nodes += compile_func_single_block(
                _init_df,
                data_arrs,
                lhs,
                extra_globals={"__col_name_meta_value_pd_read_sql": col_meta},
            )
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
        sheet_name = get_const_arg(
            "read_excel",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            1,
            "sheet_name",
            rhs.loc,
            default=0,
            typ="str or int",
        )
        header = get_const_arg(
            "read_excel",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            2,
            "header",
            rhs.loc,
            default=0,
            typ="int",
        )
        col_names = get_const_arg(
            "read_excel",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            3,
            "names",
            rhs.loc,
            default=0,
        )
        # index_col = self._get_const_arg("read_excel", rhs.args, kws, 4, "index_col", -1)
        comment = get_const_arg(
            "read_excel",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            20,
            "comment",
            rhs.loc,
            default=None,
            use_default=True,
        )
        date_cols = get_const_arg(
            "pd.read_excel",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            17,
            "parse_dates",
            rhs.loc,
            default=[],
            typ="int or str",
        )
        dtype_var = get_call_expr_arg("read_excel", rhs.args, kws, 7, "dtype", "")
        skiprows = get_const_arg(
            "read_excel",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            12,
            "skiprows",
            rhs.loc,
            default=0,
            typ="int",
        )

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
                f"read_excel() arguments {unsupported_args} not supported yet"
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
                "the 'names' and 'dtype' arguments if the filename is not constant. "
                "For more information, see: https://docs.bodo.ai/latest/file_io/#io_workflow."
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
        # schema: pd.read_csv(filepath_or_buffer, *, sep=_NoDefault.no_default,
        # delimiter=None, header='infer', names=_NoDefault.no_default, index_col=None,
        # usecols=None, dtype=None, engine=None, converters=None, true_values=None,
        # false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0,
        # nrows=None, na_values=None, keep_default_na=True, na_filter=True,
        # verbose=_NoDefault.no_default, skip_blank_lines=True, parse_dates=None,
        # keep_date_col=_NoDefault.no_default, date_parser=_NoDefault.no_default,
        # date_format=None, dayfirst=False, cache_dates=True, iterator=False,
        # chunksize=None, compression='infer', thousands=None, decimal='.',
        # lineterminator=None, quotechar='"', quoting=0, doublequote=True,
        # escapechar=None, comment=None, encoding=None, encoding_errors='strict',
        # dialect=None, on_bad_lines='error', delim_whitespace=_NoDefault.no_default,
        # low_memory=True, memory_map=False, float_precision=None, storage_options=None,
        # dtype_backend=_NoDefault.no_default)
        kws = dict(rhs.kws)

        # TODO: Can we use fold the arguments even though this untyped pass?

        fname = get_call_expr_arg("pd.read_csv", rhs.args, kws, 0, "filepath_or_buffer")
        # fname's type is checked at typing pass or when it is forced to be a constant.

        # Users can only provide either sep or delim. Use a dummy default value to track
        # this behavior.
        sep_val = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            1,
            "sep",
            rhs.loc,
            default=None,
            use_default=True,
        )
        delim_val = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
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

        header = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            3,
            "header",
            rhs.loc,
            default="infer",
        )
        # Per Pandas documentation (header: int, list of int, default ‘infer’)
        if header not in ("infer", 0, None):
            raise BodoError(
                "pd.read_csv() 'header' should be one of 'infer', 0, or None",
                loc=rhs.loc,
            )

        col_names = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            4,
            "names",
            rhs.loc,
            default=0,
        )
        # Per Pandas documentation (names: array-like). Since columns don't need string types,
        # we only check that this is a list or tuple (since constant arrays aren't supported).
        if col_names != 0 and not isinstance(col_names, (list, tuple)):
            raise BodoError(
                "pd.read_csv() 'names' should be a constant list if provided",
                loc=rhs.loc,
            )

        index_col = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
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

        usecols = get_const_arg(
            "pd.read_csv()",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
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
        # In case col_names < requested used columns.
        if (
            col_names != 0
            and usecols is not None
            and len(set(usecols)) > len(col_names)
        ):
            raise BodoError(
                "pd.read_csv() number of used columns exceeds the number of passed names  "
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

        date_cols = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
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

        chunksize = get_const_arg(
            "pandas.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
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

        compression = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            32,
            "compression",
            rhs.loc,
            default="infer",
        )
        # Per Pandas documentation (compression: {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default ‘infer’)
        supported_compression_options = ["infer", "gzip", "bz2", "zip", "xz", None]
        if compression not in supported_compression_options:
            raise BodoError(
                f"pd.read_csv() 'compression' must be one of {supported_compression_options}",
                loc=rhs.loc,
            )

        # TODO: [BE-2146] support passing as a variable.
        escapechar = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            39,
            "escapechar",
            rhs.loc,
            default=None,
            use_default=True,
            typ="str",
        )
        if escapechar is not None and (
            not isinstance(escapechar, str) or len(escapechar) != 1
        ):
            raise BodoError(
                "pd.read_csv(): 'escapechar' must be a one-character string.",
                loc=rhs.loc,
            )
        if escapechar == sep:
            raise BodoError(
                f"pd.read_csv(): 'escapechar'={escapechar} must not be equal to 'sep'={sep}.",
                loc=rhs.loc,
            )
        if escapechar == "\n":
            raise BodoError(
                "pd.read_csv(): newline as 'escapechar' is not supported.", loc=rhs.loc
            )

        # Pandas default is True but Bodo is False
        pd_low_memory = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            48,
            "low_memory",
            rhs.loc,
            default=False,
            use_default=True,
        )
        # storage_options
        csv_storage_options = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            51,
            "storage_options",
            rhs.loc,
            default={},
            use_default=True,
        )
        # sample_nrows
        csv_sample_nrows = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            52,
            "sample_nrows",
            rhs.loc,
            default=100,
            use_default=True,
            typ="int",
        )
        if not isinstance(csv_sample_nrows, int) or csv_sample_nrows < 1:
            raise BodoError(
                "pd.read_csv() 'sample_nrows' must be a constant integer >= 1 if provided."
            )
        _check_storage_options(csv_storage_options, "read_csv", rhs)

        # Bodo specific arguments. To avoid constantly needing to update Pandas we
        # make these kwargs only.

        # _bodo_upcast_to_float64 updates types if inference may not be fully accurate.
        # This upcasts all integer/float values to float64. This runs before
        # dtype dictionary and won't impact those columns.
        _bodo_upcast_to_float64 = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            -1,
            "_bodo_upcast_to_float64",
            rhs.loc,
            default=False,
            use_default=True,
        )
        # Allows users specify what columns should be read in as dictionary-encoded
        # string arrays manually.
        _bodo_read_as_dict = get_const_arg(
            "pd.read_csv",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            -1,
            "_bodo_read_as_dict",
            rhs.loc,
            use_default=True,
            default=[],
        )
        if not isinstance(_bodo_read_as_dict, list):
            raise BodoError(
                "pandas.read_csv(): '_bodo_read_as_dict', if provided, must be a constant list of column names."
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
            ("keep_date_col", False),
            ("date_parser", None),
            ("date_format", None),
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
            ("on_bad_lines", "error"),
            ("delim_whitespace", False),
            ("low_memory", False),
            ("memory_map", False),
            ("float_precision", None),
            ("storage_options", None),
            ("dtype_backend", "numpy_nullable"),
            # TODO: Specify this is kwonly in error checks
            ("_bodo_upcast_to_float64", False),
            ("sample_nrows", 100),
            ("_bodo_read_as_dict", None),
        )
        # Arguments that are supported
        supported_args = {
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
            "escapechar",
            "storage_options",
            "sample_nrows",
            "_bodo_read_as_dict",
            "dtype_backend",
        }
        # Iterate through the provided args. If an argument is in the supported_args,
        # skip it. Otherwise we check that the value matches the default value.
        unsupported_args = []
        for i, arg_pair in enumerate(total_args):
            name, default = arg_pair
            if name not in supported_args:
                try:
                    # Catch the exceptions because don't want the constant value exception
                    # Instead we want to indicate the argument isn't supported.
                    provided_val = get_const_arg(
                        "pd.read_csv",
                        rhs.args,
                        kws,
                        self.func_ir,
                        self.args,
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

        # Holds type of each used column. { col_idx : col_type }
        dtype_map = {}
        # inference is required
        # when either dtype or column names are not known yet.
        # i.e. weren't passed by the user `dtype={col:coltype,..}, names=['col', ...]`.
        # NOTE: If dtype_var is not None but user passed column names, we still generate dtype_map in this if-stmt
        # and update the dictionary in another if-stmt (if dtype_var is not None:)
        if dtype_var is None or col_names == 0:
            # infer column names and types from constant filename
            msg = (
                "pd.read_csv() requires explicit type "
                "annotation using the 'names' and 'dtype' arguments if the filename is "
                "not constant. For more information, "
                "see: https://docs.bodo.ai/latest/file_io/#io_workflow."
            )
            fname_const = get_const_value(
                fname,
                self.func_ir,
                msg,
                arg_types=self.args,
                file_info=CSVFileInfo(
                    sep,
                    skiprows_val,
                    header,
                    compression,
                    pd_low_memory,
                    escapechar,
                    csv_storage_options,
                    csv_sample_nrows,
                ),
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
                # TODO: BE-2596 investigate passing `usecols` and `names` here
                df_type = _get_csv_df_type_from_file(
                    fname_const,
                    sep,
                    skiprows_val,
                    header,
                    compression,
                    pd_low_memory,
                    escapechar,
                    csv_storage_options,
                    csv_sample_nrows,
                )
            dtypes = df_type.data
            # Generate usecols indices
            col_name_src = col_names if col_names else df_type.columns
            usecols, _ = _get_usecols_as_indices(col_name_src, usecols, df_type.columns)
            # convert Pandas generated integer names if any
            cols = [str(df_type.columns[i]) for i in usecols]
            # overwrite column names like Pandas if explicitly provided
            if col_names != 0:
                cols = _replace_col_names(col_names, usecols)
            col_names = cols
            # Date types are handled through a separate argument and omitted now.
            dtype_map = {
                c: dtypes[usecols[i]]
                for i, c in enumerate(col_names)
                if i not in date_cols and c not in date_cols
            }
        # Update usecols and col_names
        else:
            # Get usecols as indices.
            usecols, all_cols = _get_usecols_as_indices(col_names, usecols, col_names)
            # Update col_names with usecols
            if not all_cols:
                cols = _replace_col_names(col_names, usecols)
                col_names = cols

        if _bodo_upcast_to_float64:
            dtype_map_cpy = dtype_map.copy()
            for c, t in dtype_map_cpy.items():
                if isinstance(
                    t, (types.Array, IntegerArrayType, FloatingArrayType)
                ) and isinstance(
                    t.dtype, (types.Integer, types.Float)
                ):  # pragma: no cover
                    dtype_map[c] = types.Array(types.float64, 1, "C")

        # handle dtype arg if provided
        # This could be a single type for the whole data
        # or a dictionary specifying type for each/some of the columns.
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
                col_names_map = {name: i for i, name in enumerate(col_names)}
                missing_columns = []
                for c, t in dtype_map_const.items():
                    # Check int to avoid cases where the key is the column index.
                    # i.e. {0: str}. See _get_col_ind_from_name_or_ind.
                    if c not in colname_set and not isinstance(c, int):
                        missing_columns.append(c)
                    else:
                        dtype_update_map[
                            col_names[_get_col_ind_from_name_or_ind(c, col_names_map)]
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

        # error check _bodo_read_as_dict values and update string arrays to dict-encoded
        if _bodo_read_as_dict:
            for c in _bodo_read_as_dict:
                if c not in col_names:
                    raise BodoError(
                        f"pandas.read_csv(): column name '{c}' in _bodo_read_as_dict is not in data columns {col_names}"
                    )
            dtype_map_cpy = dtype_map.copy()
            for c, t in dtype_map_cpy.items():
                if c in _bodo_read_as_dict:
                    if dtype_map[c] != bodo.types.string_array_type:
                        raise BodoError(
                            f"pandas.read_csv(): column name '{c}' in _bodo_read_as_dict is not a string column"
                        )
                    dtype_map[c] = bodo.types.dict_str_arr_type

        columns, _, out_types = _get_read_file_col_info(
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
            index_typ = bodo.utils.typing.index_typ_from_dtype_name_arr(
                index_elem_dtype, index_name, index_arr_typ
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
            index_arg = f"bodo.hiframes.pd_index_ext.init_range_index(0, len({data_args[0]}), 1, None)"
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
            chunk_iterator = bodo.io.csv_iterator_ext.CSVIteratorType(
                df_type,
                orig_columns,
                out_types,
                usecols,
                sep,
                index_ind,
                index_arr_typ,
                index_name,
                escapechar,
                csv_storage_options,
            )

            # Create a new temp var so this is always exactly one variable.
            data_arrs = [ir.Var(lhs.scope, mk_unique_var("csv_iterator"), lhs.loc)]
        else:
            chunk_iterator = None
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
                chunk_iterator,
                is_skiprows_list,
                pd_low_memory,
                escapechar,
                csv_storage_options,
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
            func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe((table_val,), {index_arg}, __col_name_meta_value_pd_read_csv)\n"

            loc_vars = {}

            exec(
                func_text,
                {},
                loc_vars,
            )
            _type_func = loc_vars["_type_func"]

            nodes += compile_func_single_block(
                _type_func,
                data_arrs,
                lhs,
                extra_globals={
                    "__col_name_meta_value_pd_read_csv": ColNamesMetaType(
                        df_type.columns
                    )
                },
            )
        return nodes

    def _handle_pd_read_json(self, assign, lhs, rhs, label):
        """transform pd.read_json() call,
        where default orient = 'records'

        schema: pandas.read_json(path_or_buf=None, orient=None, typ='frame',
        dtype=None, convert_axes=None, convert_dates=True,
        keep_default_dates=True, numpy=False, precise_float=False,
        date_unit=None, encoding=None, encoding_errors='strict',
        lines=False, chunksize=None, compression='infer'
        # NOTE: sample_nrows is Bodo specific argument
        nrows=None, storage_options=None, sample_nrows=100,
        )
        """
        # convert_dates required for date cols
        kws = dict(rhs.kws)
        fname = get_call_expr_arg("read_json", rhs.args, kws, 0, "path_or_buf")
        orient = get_const_arg(
            "read_json",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            1,
            "orient",
            rhs.loc,
            default="records",
        )
        frame_or_series = get_call_expr_arg(
            "read_json", rhs.args, kws, 2, "typ", "frame"
        )
        dtype_var = get_call_expr_arg("read_json", rhs.args, kws, 3, "dtype", "")
        # default value is True
        convert_dates = get_const_arg(
            "read_json",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            5,
            "convert_dates",
            rhs.loc,
            default=True,
            typ="int or str",
        )
        # Q: Why set date_cols = False not empty list as well?
        date_cols = [] if convert_dates is True else convert_dates
        precise_float = get_const_arg(
            "read_json",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            8,
            "precise_float",
            rhs.loc,
            default=False,
        )
        lines = get_const_arg(
            "read_json",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            12,
            "lines",
            rhs.loc,
            default=True,
        )
        compression = get_const_arg(
            "read_json",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            14,
            "compression",
            rhs.loc,
            default="infer",
        )
        # storage_options
        json_storage_options = get_const_arg(
            "pd.read_json",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            16,
            "storage_options",
            rhs.loc,
            default={},
            use_default=True,
        )
        _check_storage_options(json_storage_options, "read_json", rhs)

        # sample_nrows
        json_sample_nrows = get_const_arg(
            "pd.read_json",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            17,
            "sample_nrows",
            rhs.loc,
            default=100,
            use_default=True,
            typ="int",
        )
        if not isinstance(json_sample_nrows, int) or json_sample_nrows < 1:
            raise BodoError(
                "pd.read_json() 'sample_nrows' must be a constant integer >= 1 if provided."
            )

        # check unsupported arguments
        unsupported_args = {
            "convert_axes",
            "keep_default_dates",
            "numpy",
            "date_unit",
            "encoding",
            "encoding_errors",
            "chunksize",
            "nrows",
        }

        passed_unsupported = unsupported_args.intersection(kws.keys())
        if len(passed_unsupported) > 0:
            if unsupported_args:
                raise BodoError(
                    f"read_json() arguments {passed_unsupported} not supported yet"
                )

        supported_compression_options = {"infer", "gzip", "bz2", None}
        if compression not in supported_compression_options:
            raise BodoError(
                f"pd.read_json() compression = {compression} is not supported."
                f" Supported options are {supported_compression_options}"
            )

        if frame_or_series != "frame":
            raise BodoError(
                f"pd.read_json() typ = {frame_or_series} is not supported."
                "Currently only supports orient = 'frame'"
            )

        if orient != "records":
            raise BodoError(
                f"pd.read_json() orient = {orient} is not supported."
                "Currently only supports orient = 'records'"
            )

        if type(lines) is not bool:
            raise BodoError(
                f"pd.read_json() lines = {lines} is not supported."
                "lines must be of type bool."
            )

        col_names = []

        # infer column names and types from constant filenames if:
        # not explicitly passed with dtype
        # not reading from s3 & hdfs
        # not reading from directory
        msg = (
            "pd.read_json() requires the filename to be a compile time constant. "
            "For more information, "
            "see: https://docs.bodo.ai/latest/file_io/#json-section."
        )
        fname_const = get_const_value(
            fname,
            self.func_ir,
            msg,
            arg_types=self.args,
            file_info=JSONFileInfo(
                orient,
                convert_dates,
                precise_float,
                lines,
                compression,
                json_storage_options,
                json_sample_nrows,
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
                    json_storage_options,
                    json_sample_nrows,
                )
            df_type = df_type.copy(
                tuple(to_str_arr_if_dict_array(t) for t in df_type.data)
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
            # NOTE: read_json's behavior is different from read_csv since it doesn't
            # have the "names" argument for specifying column names. Therefore, we need
            # to infer column names from dtype to pass to _get_read_file_col_info below.
            col_names = list(dtype_map.keys())

        columns, data_arrs, out_types = _get_read_file_col_info(
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
                json_storage_options,
            )
        ]

        columns = columns.copy()  # copy since modified below
        n_cols = len(columns)
        args = [f"data{i}" for i in range(n_cols)]
        data_args = args.copy()

        # initialize range index
        assert len(data_args) > 0
        index_arg = f"bodo.hiframes.pd_index_ext.init_range_index(0, len({data_args[0]}), 1, None)"

        # Below we assume that the columns are strings
        func_text = "def _init_df({}):\n".format(", ".join(args))
        func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), {}, __col_name_meta_value_pd_read_json)\n".format(
            ", ".join(data_args),
            index_arg,
        )
        loc_vars = {}
        exec(
            func_text,
            {"__col_name_meta_value_pd_read_json": ColNamesMetaType(tuple(columns))},
            loc_vars,
        )
        _init_df = loc_vars["_init_df"]

        nodes += compile_func_single_block(_init_df, data_arrs, lhs)
        return nodes

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

    def _handle_extended_named_agg(self, assign, lhs, rhs):
        """
        Replace pd.Extended() with equivalent tuple to be handled in groupby typing.
        In order to handle a generic extension argument, all arguments are passed without keyword arguments.
        It's up to the individual function to know which arguments are expected.
        """

        kws = dict(rhs.kws)
        column_var = get_call_expr_arg("pd.NamedAgg", rhs.args, kws, 0, "column")
        aggfunc_var = get_call_expr_arg("pd.NamedAgg", rhs.args, kws, 1, "aggfunc")
        additional_args_var = get_call_expr_arg(
            "pd.NamedAgg", rhs.args, kws, 2, "additional_args"
        )
        assign.value = ir.Expr.build_tuple(
            [column_var, aggfunc_var, additional_args_var], rhs.loc
        )
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

    def _gen_parquet_read(
        self,
        fname,
        lhs: ir.Var,
        columns=None,
        storage_options=None,
        input_file_name_col=None,
        read_as_dict_cols=None,
        use_hive=True,
        _bodo_read_as_table=False,
        chunksize: int | None = None,
        use_index: bool = True,
        sql_op_id: int = -1,
    ):
        (
            columns,
            data_arrs,
            index_cols,
            nodes,
            _,
        ) = self.pq_handler.gen_parquet_read(
            fname,
            lhs,
            columns,
            storage_options=storage_options,
            input_file_name_col=input_file_name_col,
            read_as_dict_cols=read_as_dict_cols,
            use_hive=use_hive,
            chunksize=chunksize,
            use_index=use_index,
            sql_op_id=sql_op_id,
        )
        n_cols = len(columns)

        if chunksize is not None and use_index and index_cols:
            raise BodoError(
                "pd.read_parquet(): Bodo currently does not support batched reads "
                "of Parquet files with an index column"
            )

        if not use_index or len(index_cols) == 0:
            assert n_cols > 0
            agg_index_arg = (
                "bodo.hiframes.pd_index_ext.init_range_index(0, len(T), 1, None)"
            )
        elif len(index_cols) == 1:
            index_col = index_cols[0]
            if isinstance(index_col, dict):
                if index_col["name"] is None:
                    index_col_name = None
                    index_col_name_str = None
                else:
                    index_col_name = index_col["name"]
                    index_col_name_str = f"'{index_col_name}'"
                # ignore range index information in pandas metadata
                agg_index_arg = f"bodo.hiframes.pd_index_ext.init_range_index(0, len(T), 1, {index_col_name_str})"
            else:
                # if the index_col is __index_level_0__, it means it has no name.
                # Thus we do not write the name instead of writing '__index_level_0__' as the name
                field_name = None if "__index_level_" in index_col else index_col
                agg_index_arg = (
                    f"bodo.utils.conversion.convert_to_index(index_arr, {field_name!r})"
                )
        else:
            index_field_names = []
            for index_col in index_cols:
                # I don't think RangeIndex is possible here, but just in case
                assert isinstance(index_col, str)
                index_field_names.append(
                    None if "__index_level_" in index_col else index_col
                )
            agg_index_arg = f"bodo.hiframes.pd_multi_index_ext.init_multi_index(bodo.libs.struct_arr_ext.get_data(index_arr), {tuple(index_field_names)!r})"

        # _bodo_read_as_table = do not wrap the output table in a DataFrame
        if _bodo_read_as_table or chunksize is not None:
            nodes += [ir.Assign(data_arrs[0], lhs, lhs.loc)]
        else:
            func_text = (
                f"def _init_df(T, index_arr):\n"
                f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(\n"
                f"    (T,),\n"
                f"    {agg_index_arg},\n"
                f"    __col_name_meta_value_pq_read\n"
                f"  )\n"
            )
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            _init_df = loc_vars["_init_df"]
            nodes += compile_func_single_block(
                _init_df,
                data_arrs,
                lhs,
                extra_globals={
                    "__col_name_meta_value_pq_read": ColNamesMetaType(tuple(columns))
                },
            )
        return nodes

    def _handle_pd_read_parquet(self, assign, lhs, rhs):
        # get args and check values
        kws = dict(rhs.kws)
        fname = get_call_expr_arg("read_parquet", rhs.args, kws, 0, "path")
        engine = get_call_expr_arg("read_parquet", rhs.args, kws, 1, "engine", "auto")
        columns = get_const_arg(
            "read_parquet",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            2,
            "columns",
            rhs.loc,
            default=-1,
        )
        storage_options = get_const_arg(
            "read_parquet",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e4,
            "storage_options",
            rhs.loc,
            default={},
        )

        # Bodo specific arguments. To avoid constantly needing to update Pandas we
        # make these kwargs only.

        # Equivalent to Spark SQL's input_file_name
        # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.input_file_name.html
        # When specified, create a column in the resulting dataframe with this name
        # that contains the name of the file the row comes from
        _bodo_input_file_name_col = get_const_arg(
            "read_parquet",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e4,
            "_bodo_input_file_name_col",
            rhs.loc,
            use_default=True,
            default=None,
        )

        # Users can use this to specify what columns should be read in as
        # dictionary-encoded string arrays. This is in addition
        # to whatever columns bodo determines should be read in
        # with dictionary encoding.
        _bodo_read_as_dict = get_const_arg(
            "read_parquet",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e4,
            "_bodo_read_as_dict",
            rhs.loc,
            use_default=True,
            default=None,
        )

        _bodo_use_hive = get_const_arg(
            "read_parquet",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e4,
            "_bodo_use_hive",
            rhs.loc,
            use_default=True,
            default=True,
        )

        _bodo_read_as_table: bool = get_const_arg(
            "read_sql",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e5,
            "_bodo_read_as_table",
            rhs.loc,
            default=False,
        )

        # Mimicing the use_index arg from read_csv
        use_index = get_const_arg(
            "read_parquet",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e4,
            "_bodo_use_index",
            rhs.loc,
            use_default=True,
            default=True,
        )

        # If defined, perform batched reads on the dataset
        # Note that we don't want to use Pandas chunksize, since it returns an iterator
        chunksize: int | None = get_const_arg(
            "read_parquet",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e4,
            "_bodo_chunksize",
            rhs.loc,
            use_default=True,
            default=None,
            typ="int",
        )
        if chunksize is not None and (
            not isinstance(chunksize, int) or chunksize < 1
        ):  # pragma: no cover
            raise BodoError(
                "pd.read_parquet() '_bodo_chunksize' must be a constant integer >= 1."
            )

        # Operator ID assigned by the planner for query profile purposes.
        # Only applicable in the streaming case.
        _bodo_sql_op_id_const: int = get_const_arg(
            "read_parquet",
            rhs.args,
            kws,
            self.func_ir,
            self.args,
            10e4,
            "_bodo_sql_op_id",
            rhs.loc,
            use_default=True,
            default=-1,
            typ="int",
        )

        # check unsupported arguments
        supported_args = (
            "path",
            "engine",
            "columns",
            "storage_options",
            "dtype_backend",
            "_bodo_chunksize",
            "_bodo_input_file_name_col",
            "_bodo_read_as_dict",
            "_bodo_use_hive",
            "_bodo_read_as_table",
            "_bodo_use_index",
            "_bodo_sql_op_id",
        )
        unsupported_args = set(kws.keys()) - set(supported_args)
        if unsupported_args:
            raise BodoError(
                f"read_parquet() arguments {unsupported_args} not supported yet"
            )
        _check_storage_options(storage_options, "read_parquet", rhs, check_fields=False)

        if engine not in ("auto", "pyarrow"):
            raise BodoError("read_parquet: only pyarrow engine supported")

        if columns == -1:
            columns = None

        return self._gen_parquet_read(
            fname,
            lhs,
            columns,
            storage_options,
            _bodo_input_file_name_col,
            _bodo_read_as_dict,
            use_hive=_bodo_use_hive,
            _bodo_read_as_table=_bodo_read_as_table,
            chunksize=chunksize,
            use_index=use_index,
            sql_op_id=_bodo_sql_op_id_const,
        )

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
        sep_err_msg = f"{np_fromfile}(): sep argument is not supported"
        _sep = get_const_arg(
            np_fromfile,
            rhs.args,
            kws,
            self.func_ir,
            self.args,
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

    def _handle_metadata(self):
        """remove distributed input annotation from locals and add to metadata"""
        if "distributed" not in self.metadata:
            # TODO: keep updated in variable renaming?
            self.metadata["distributed"] = self.flags.distributed.copy()

        if "distributed_block" not in self.metadata:
            self.metadata["distributed_block"] = self.flags.distributed_block.copy()

        if "replicated" not in self.metadata:
            self.metadata["replicated"] = self.flags.replicated.copy()

        if "threaded" not in self.metadata:
            self.metadata["threaded"] = self.flags.threaded.copy()

        if "is_return_distributed" not in self.metadata:
            self.metadata["is_return_distributed"] = False

        # store args_maybe_distributed to be used in distributed_analysis of a potential
        # JIT caller
        if "args_maybe_distributed" not in self.metadata:
            self.metadata["args_maybe_distributed"] = self.flags.args_maybe_distributed

        # make sure a variable is not marked as both distributed and replicated
        dist_and_rep = self.metadata["distributed"] & self.metadata["replicated"]
        if dist_and_rep:
            raise BodoError(
                f"Variables {dist_and_rep} marked as both 'distributed' and 'replicated'.",
                self.func_ir.loc,
            )

    def _run_return(self, ret_node):
        # TODO: handle distributed analysis, requires handling variable name
        # change in simplify() and replace_var_names()
        flagged_vars = (
            self.metadata["distributed"]
            | self.metadata["distributed_block"]
            | self.metadata["threaded"]
            | self.metadata["replicated"]
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
            if (
                ret_name in self.metadata["distributed"]
                or ret_name in self.metadata["distributed_block"]
                or all_returns_distributed
            ):
                flag = "distributed"
            elif ret_name in self.metadata["replicated"]:
                flag = "replicated"
            else:
                assert ret_name in self.metadata["threaded"], (
                    f"invalid return flag for {ret_name}"
                )
                flag = "threaded"
            # save in metadata that the return value is distributed
            # TODO(ehsan): support other flags like distributed_block?
            if flag == "distributed":
                self.metadata["is_return_distributed"] = True
            if flag == "replicated":
                self.metadata["is_return_distributed"] = False
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
                    if (
                        vname in self.metadata["distributed"]
                        or vname in self.metadata["distributed_block"]
                        or all_returns_distributed
                    ):
                        flag = "distributed"
                    elif vname in self.metadata["replicated"]:
                        flag = "replicated"
                    else:
                        assert vname in self.metadata["threaded"], (
                            f"invalid return flag for {vname}"
                        )
                        flag = "threaded"
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

        elif flag == "replicated":
            func_text = (
                ""
                "def f(_rep_arr):\n"
                "    rep_return = bodo.libs.distributed_api.rep_return(_rep_arr)\n"
            )

        elif flag == "threaded":
            func_text = (
                ""
                "def f(_threaded_arr):\n"
                "    _th_arr = bodo.libs.distributed_api.threaded_return(_threaded_arr)\n"
            )

        else:
            raise BodoError(f"Invalid return flag {flag}")
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
    changed = False
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
                changed = True
            except GuardException:
                pass

    # Remove dead blocks using CFG
    cfg = compute_cfg_from_blocks(func_ir.blocks)
    for dead in cfg.dead_nodes():
        del func_ir.blocks[dead]
        changed = True
    return changed


def _dtype_val_to_arr_type(t, func_name, loc):
    """get array type from type value 't' specified in calls like read_csv()
    e.g. "str" -> string_array_type
    """
    if t is object:
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

        if t.startswith("Float"):  # pragma: no cover
            dtype = bodo.libs.float_arr_ext.typeof_pd_float_dtype(
                pd.api.types.pandas_dtype(t), None
            )
            return FloatingArrayType(dtype.dtype)

        # datetime64 case
        if t == "datetime64[ns]":
            return types.Array(types.NPDatetime("ns"), 1, "C")

        t = "int64" if t == "int" else t
        t = "float64" if t == "float" else t
        t = "bool_" if t == "bool" else t
        # XXX: bool with NA needs to be object, TODO: fix somehow? doc.
        t = "bool_" if t == "O" else t

        if t == "bool_":
            return boolean_array_type

        typ = getattr(types, t)
        typ = types.Array(typ, 1, "C")
        return typ

    if t is int:
        return types.Array(types.int64, 1, "C")

    if t is float:
        return types.Array(types.float64, 1, "C")

    # categorical type
    if isinstance(t, pd.CategoricalDtype):
        cats = tuple(t.categories)
        elem_typ = bodo.types.string_type if len(cats) == 0 else bodo.typeof(cats[0])
        typ = PDCategoricalDtype(cats, elem_typ, t.ordered)
        return CategoricalArrayType(typ)

    # nullable int types
    if isinstance(t, pd.core.arrays.integer.IntegerDtype):  # pragma: no cover
        dtype = bodo.libs.int_arr_ext.typeof_pd_int_dtype(t, None)
        return IntegerArrayType(dtype.dtype)

    # nullable float types
    if isinstance(t, pd.core.arrays.floating.FloatingDtype):
        dtype = bodo.libs.float_arr_ext.typeof_pd_float_dtype(t, None)
        return FloatingArrayType(dtype.dtype)

    # try numpy dtypes
    try:
        dtype = numba.np.numpy_support.from_dtype(t)
        return types.Array(dtype, 1, "C")
    except Exception:
        raise BodoError(f"{func_name}() 'dtype' does not support {t}", loc=loc)


def _get_col_ind_from_name_or_ind(c, col_names_map):
    """get column index from a map {column name -> index}"""
    # TODO(ehsan): error checking
    if isinstance(c, int) and c not in col_names_map:
        return c
    elif c not in col_names_map:
        raise BodoError(
            f"usecols: `{c}` does not match columns: {list(col_names_map.keys())}. "
        )
    return col_names_map[c]


def _get_usecols_as_indices(col_names, usecols, df_type_columns):
    """
    get usecols as column indices (not names)
    If usecols is None, generate indices.
    Otherwise, check if name is passed and replace with its index position

    Args:

    col_names: df list of all columns
    usecols: list of column(s) to load passed from user.
            It could be int (index) or string (column name)
    df_type_columns: original list of columns

    Returns:
    usecols: list of used column as indices only.
    all_cols: flag to indicate whether all columns are used.
    """

    all_cols = True
    if usecols is None:
        # If only `names` argument was passed and it includes only few names
        # Pandas use last len(names) as the columns
        # Otherwise, it starts from begining and takes first len(col_names).
        # In the latter case, `df_type_columns` will be all columns from file or `names`
        usecols = list(
            range(len(df_type_columns) - len(col_names), len(df_type_columns))
        )
    else:
        all_cols = False
        # make sure usecols has column indices (not names)
        col_name_map = {name: i for i, name in enumerate(col_names)}
        usecols = [_get_col_ind_from_name_or_ind(c, col_name_map) for c in usecols]
    return sorted(set(usecols)), all_cols


def _replace_col_names(col_names, usecols):
    """
    Create list of column names with user-defined ones (`names`)
    and `usecols`

    Args:

    col_names: df list of all columns
    usecols: list of column(s) to load passed from user.
            It could be int (index) or string (column name)

    Returns:
    cols: list of column names that are specified by the user
    """
    # User pass subset of names for usecols only (i.e. not for all actual columns)
    # e.g. names = ["A", "B"], usecols = [0, 2] (i.e. column 1 has no name and will not be read)
    if len(usecols) == len(col_names):
        cols = col_names
    # col_names more than usecols.
    # Ex: names=['A', 'B'], usecols=[0]
    # It can also be all names and then usecols use subset of these names
    # names = ['A', 'B', 'C'] usecols=[0,1]
    else:
        cols = [col_names[i] for i in usecols]
    return cols


class JSONFileInfo(FileInfo):
    """FileInfo object passed to ForceLiteralArg for
    file name arguments that refer to a JSON dataset"""

    def __init__(
        self,
        orient,
        convert_dates,
        precise_float,
        lines,
        compression,
        storage_options,
        json_sample_nrows,
    ):
        self.orient = orient
        self.convert_dates = convert_dates
        self.precise_float = precise_float
        self.lines = lines
        self.compression = compression
        self.storage_options = storage_options
        self.json_sample_nrows = json_sample_nrows
        super().__init__()

    # TODO: Return type is not consistent with base class and Parquet?
    def _get_schema(self, fname):
        return _get_json_df_type_from_file(
            fname,
            self.orient,
            self.convert_dates,
            self.precise_float,
            self.lines,
            self.compression,
            self.storage_options,
            self.json_sample_nrows,
        )


def _get_json_df_type_from_file(
    fname_const,
    orient,
    convert_dates,
    precise_float,
    lines,
    compression,
    storage_options,
    json_sample_nrows=100,
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
            is_handler, file_name_or_handler, file_compression, _ = (
                find_file_name_or_handler(fname_const, "json", storage_options)
            )
            if is_handler and compression == "infer":
                # pandas can't infer compression without filename, we need to do it
                compression = file_compression

            # nrows: This can only be passed if lines=True.
            # https://pandas.pydata.org/docs/reference/api/pandas.read_json.html
            # This is safe since code will only reach _get_json_df_type_from_file iff lines=True

            df = pd.read_json(
                file_name_or_handler,
                orient=orient,
                convert_dates=convert_dates,
                precise_float=precise_float,
                lines=lines,
                compression=compression,
                nrows=json_sample_nrows,
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

    df_type_or_e = df_type_or_e.copy(
        data=tuple(to_str_arr_if_dict_array(t) for t in df_type_or_e.data)
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

    df_type_or_e = df_type_or_e.copy(
        data=tuple(to_str_arr_if_dict_array(t) for t in df_type_or_e.data)
    )
    return df_type_or_e


def _get_read_file_col_info(dtype_map, date_cols, col_names, lhs):
    """get column names, ir.Var objects, and data types for file read (sql/csv/json)"""
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
    for i, col_name in enumerate(col_names):
        # Column is alive if its in the dtype_map or date_cols
        if col_name in dtype_map or i in date_cols or col_name in date_cols:
            # Pandas prioritizes dtype_map over date_cols
            col_type = dtype_map.get(
                col_name, types.Array(bodo.types.datetime64ns, 1, "C")
            )
            columns.append(col_name)
            out_types.append(col_type)
            data_arrs.append(ir.Var(lhs.scope, mk_unique_var(col_name), lhs.loc))
    return columns, data_arrs, out_types


def _get_sql_types_arr_colnames(
    sql_const,
    con_const,
    _bodo_read_as_dict,
    lhs,
    loc,
    is_table_input: bool,
    is_independent: bool,
    downcast_decimal_to_double: bool = False,
    orig_table_const: str | None = None,
    orig_table_indices_const: tuple[int] | None = None,
    snowflake_conn_cache: dict[str, SnowflakeConnection] | None = None,
    convert_snowflake_column_names: bool = True,
):
    """
    Wrapper function to determine the db_type, column names,
    array variables, array types, any column names
    that were converted, and any unsupported columns that may
    be possible to type but shouldn't be loaded.

    This is written as a standalone
    function because other packages (i.e. BodoSQL) may need
    to type a SQL query.

    Args:
        sql_const: The sql query to determine the array types for. May be a full query, or a table name.
        con_const (str): The connection string being used to connect to the database.
        _bodo_read_as_dict (bool): Read all string columns as dict encoded strings.
        lhs: The variable being assigned to by the read_sql call. If not converting a read_sql call,
                this is a dummy variable.
        loc: The location of the original read_sql call, if calling this function
                is called while converting a read_sql. Otherwise, it's a dummy location.
        is_table_input (bool): Is sql_const just the string name of the table to read from?
        is_independent (bool): (TODO: add docs)
        downcast_decimal_to_double (bool, default false): (TODO: add docs)
        orig_table_const (Optional str): The string of the table at the outermost
                leaf of this query. Should never be passed for queries that read from multiple tables.
                If provided, this will be used for certain metadata queries.
        orig_table_indices_const (Optional[Tuple[Int]]): The indices for each column
            in the original table. This is to handle renaming and replace name based reads with
            index based reads.
        convert_snowflake_column_names (bool, default True): Should Snowflake column names be
            converted to match SqlAlchemy. This is needed to ensure table path is consistent for
            casing with the SnowflakeCatalog.

    Returns:
        A very large tuple (TODO: add docs)
    """
    # find db type
    db_type, _ = bodo.io.utils.parse_dbtype(con_const)
    # Whether SQL statement is SELECT query
    is_select_query = False
    # Does the SQL node have side effects (e.g. DELETE). If
    # so we cannot perform DCE.
    has_side_effects = False
    # Operations that create or edit objects don't return data.
    # They work with Pandas but then raises an exception
    # and ends program. So we avoid them.
    # sqlalchemy.exc.ResourceClosedError:
    # This result object does not return rows. It has been closed automatically.
    # Ex. : Create, insert, update, delete, drop, ...
    # SELECT goes to full path of getting type, split across ranks, ...
    # Other will be executed by all ranks as it's.
    # Snowflake + Bodo only supports SELECT
    # Snowflake: Show and Describe don't work with get_dataset
    # Only supported by MySQL.
    # Oracle: cx_oracle doesn't support them. Bodo displays same error as Pandas.
    # Postgresql: SELECT and SHOW only.
    # Declare what Bodo supports.  Users may run them to explore their database
    supported_sql_queries = ("SELECT", "SHOW", "DESCRIBE", "DESC", "DELETE")
    sql_word = (
        "SELECT" if is_table_input else sql_const.lstrip().split(maxsplit=1)[0].upper()
    )
    if sql_word not in supported_sql_queries:
        raise BodoError(f"{sql_word} query is not supported.\n")
    elif sql_word == "SELECT":
        is_select_query = True
    elif (
        db_type == "oracle"
        or (db_type == "postgresql" and sql_word in ("DESCRIBE", "DESC", "DELETE"))
        or (db_type == "snowflake" and sql_word in ("DESCRIBE", "DESC", "SHOW"))
    ):
        raise BodoError(f"{sql_word} query is not supported with {db_type}.\n")
    elif sql_word == "DELETE":
        has_side_effects = True
    # find df type
    (
        df_type,
        converted_colnames,
        unsupported_columns,
        unsupported_arrow_types,
        pyarrow_table_schema,
    ) = _get_sql_df_type_from_db(
        sql_const,
        con_const,
        db_type,
        is_select_query,
        sql_word,
        _bodo_read_as_dict,
        loc,
        is_table_input,
        is_independent,
        downcast_decimal_to_double,
        orig_table_const,
        orig_table_indices_const,
        snowflake_conn_cache=snowflake_conn_cache,
        convert_snowflake_column_names=convert_snowflake_column_names,
    )
    dtypes = df_type.data
    dtype_map = {c: dtypes[i] for i, c in enumerate(df_type.columns)}
    col_names = list(df_type.columns)

    # date columns
    date_cols = []

    columns, _, out_types = _get_read_file_col_info(
        dtype_map, date_cols, col_names, lhs
    )
    data_arrs = [
        ir.Var(lhs.scope, mk_unique_var("sql_table"), lhs.loc),
        ir.Var(lhs.scope, mk_unique_var("index_col"), lhs.loc),
    ]
    return (
        db_type,
        columns,
        data_arrs,
        out_types,
        converted_colnames,
        unsupported_columns,
        unsupported_arrow_types,
        is_select_query,
        has_side_effects,
        pyarrow_table_schema,
    )


def _get_sql_df_type_from_db(
    sql_const,
    con_const,
    db_type,
    is_select_query,
    sql_word,
    _bodo_read_as_dict,
    loc,
    is_table_input: bool,
    is_independent: bool,
    downcast_decimal_to_double: bool,
    orig_table_const: str | None = None,
    orig_table_indices_const: tuple[int] | None = None,
    snowflake_conn_cache: dict[str, SnowflakeConnection] | None = None,
    convert_snowflake_column_names: bool = True,
):
    """access the database to find df type for read_sql() output.
    Only rank zero accesses the database, then broadcasts.

    Args:
        sql_query (str): read query or Snowflake table name
        con_const (str): The connection string being used to connect to the database.
        db_type (str): The string name of the type of database being read from.
        is_select_query (bool): TODO: document this
        sql_word: TODO: document this
        _bodo_read_as_dict (bool): Read all string columns as dict encoded strings.
        loc: The location of the original read_sql call, can be a dummy,
            if the original call does not exist
        is_table_input (bool): read query is a just a table name
        is_independent (bool): TODO: document this
        downcast_decimal_to_double (bool): downcast decimal types to double
        orig_table_const (str, optional): Original table name, to be used if sql_query is not
            a table name. If provided, must guarantee that the sql_query only performs
            a selection of a subset of the table's columns, and does not rename
            any of the columns from the input table. Defaults to None.
        orig_table_indices_const (Optional[Tuple[Int]]): The indices for each column
            in the original table. This is to handle renaming and replace name based reads with
            index based reads.
        convert_snowflake_column_names (bool, default True): Should Snowflake column names be
            converted to match SqlAlchemy. This is needed to ensure table path is consistent for
            casing with the SnowflakeCatalog.


    Returns:
        A large tuple containing: (#TODO: document this)

    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    if downcast_decimal_to_double and db_type != "snowflake":  # pragma: no cover
        raise BodoError(
            "pd.read_sql(): The `_bodo_downcast_decimal_to_double` argument is only supported for "
            "Snowflake table reads"
        )

    if _bodo_read_as_dict and db_type != "snowflake":
        if bodo.get_rank() == 0:
            warnings.warn(
                BodoWarning(
                    "_bodo_read_as_dict is only supported when reading from Snowflake."
                )
            )

    # No need to try `import snowflake.connector` here, since the import
    # is handled by bodo.io.snowflake.snowflake_connect()
    if db_type != "snowflake":
        try:
            import sqlalchemy  # noqa
        except ImportError:  # pragma: no cover
            message = (
                "Using URI string without sqlalchemy installed."
                " sqlalchemy can be installed by calling"
                " 'conda install -c conda-forge sqlalchemy'."
            )
            raise BodoError(message)

    message = ""
    df_type = None
    converted_colnames = None
    unsupported_columns = None
    unsupported_arrow_types = None
    pyarrow_table_schema = None
    if bodo.get_rank() == 0 or is_independent:
        try:
            if db_type == "snowflake":  # pragma: no cover
                from bodo.io.snowflake import (
                    SF_READ_DICT_ENCODING_IF_TIMEOUT,
                    get_schema,
                    snowflake_connect,
                )

                # Use Snowflake connection cache if available
                if snowflake_conn_cache is not None:
                    if con_const in snowflake_conn_cache:
                        conn = snowflake_conn_cache[con_const]
                    else:
                        conn = snowflake_connect(con_const)
                        snowflake_conn_cache[con_const] = conn
                else:
                    conn = snowflake_connect(con_const)

                (
                    df_type,
                    converted_colnames,
                    unsupported_columns,
                    unsupported_arrow_types,
                    pyarrow_table_schema,
                    schema_timeout_info,
                    dict_encode_timeout,
                ) = get_schema(
                    conn,
                    sql_const,
                    is_select_query,
                    is_table_input,
                    _bodo_read_as_dict,
                    downcast_decimal_to_double,
                    orig_table_const,
                    orig_table_indices_const,
                    convert_snowflake_column_names=convert_snowflake_column_names,
                )

                # Log the chosen dict-encoding timeout behavior
                if bodo.user_logging.get_verbose_level() >= 2 and (
                    schema_timeout_info or dict_encode_timeout
                ):
                    if schema_timeout_info:
                        msg = (
                            "Timeout occurred during schema probing query for Number types:\n%s\n"
                            "The following columns will be kept as decimal which may impact performance: %s\n"
                        )
                        read_src = loc.strformat()
                        string_col_names = ",".join(schema_timeout_info)
                        bodo.user_logging.log_message(
                            "Decimal Schema Probe Query",
                            msg,
                            read_src,
                            string_col_names,
                        )
                    if dict_encode_timeout:
                        probe_limit, query_args = dict_encode_timeout

                        msg = (
                            "Timeout occurred during probing query at:\n%s\n"
                            "Maximum number of rows queried: %d\n"
                        )
                        if SF_READ_DICT_ENCODING_IF_TIMEOUT:
                            msg += (
                                "The following columns will be dictionary encoded: %s\n"
                            )
                        else:
                            msg += "The following columns will not be dictionary encoded: %s\n"
                        read_src = loc.strformat()
                        string_col_names = ",".join(query_args)
                        bodo.user_logging.log_message(
                            "Dictionary Encoding Probe Query",
                            msg,
                            read_src,
                            probe_limit,
                            string_col_names,
                        )

            else:
                # Any columns that had their name converted. These need to be reverted
                # in any dead column elimination
                converted_colnames = set()
                rows_to_read = 100  # TODO: tune this
                # SHOW/DESCRIBE don't work with LIMIT.
                if not is_select_query:
                    sql_call = f"{sql_const}"
                # oracle does not support LIMIT. Use ROWNUM instead
                elif db_type == "oracle":
                    sql_call = (
                        f"select * from ({sql_const}) WHERE ROWNUM <= {rows_to_read}"
                    )
                else:
                    sql_call = f"select * from ({sql_const}) x LIMIT {rows_to_read}"

                # Unsupported arrow columns is unused by other paths.
                unsupported_columns = []
                unsupported_arrow_types = []
                pyarrow_table_schema = None
                # MySQL+DESCRIBE: has fixed DataFrameType. Created upfront.
                # SHOW has many variation depending on object to show
                # so it will fall in the else-stmt
                if db_type == "mysql" and sql_word in ("DESCRIBE", "DESC"):
                    colnames = ("Field", "Type", "Null", "Key", "Default", "Extra")
                    index_type = bodo.types.RangeIndexType(bodo.types.none)
                    data_type = (
                        bodo.types.string_type,
                        bodo.types.string_type,
                        bodo.types.string_type,
                        bodo.types.string_type,
                        bodo.types.string_type,
                        bodo.types.string_type,
                    )
                    df_type = DataFrameType(data_type, index_type, colnames)
                else:
                    df = pd.read_sql(sql_call, con_const)
                    # https://docs.sqlalchemy.org/en/14/dialects/oracle.html#identifier-casing
                    # Oracle stores column names in UPPER CASE unless it's quoted
                    # pd.read_sql() returns the name with all lower case unless it was quoted
                    # NOTE: BE-2217 this "colnamealllowercase" will fail with this.
                    # If column is lowercase, then it was either converted or all upper with quotes
                    # In both cases, we add it to converted_colnames list that is needed later.
                    # See escape_column_names
                    if db_type == "oracle":
                        new_colnames = []
                        for x in df.columns:
                            if x.islower():
                                converted_colnames.add(x)
                            new_colnames.append(x)
                    df_type = numba.typeof(df)
            # always convert to nullable type since initial rows of a column could be all
            # int for example, but later rows could have NAs
            # Q: Is this needed for snowflake?
            df_type = to_nullable_type(df_type)

        except Exception as e:
            message = f"{type(e).__name__}:'{e}'"

    if not is_independent:
        message = comm.bcast(message)
    raise_error = bool(message)
    if raise_error:
        common_err_msg = f"pd.read_sql(): Error executing query `{sql_const}`."
        # raised general exception since except checks for multiple exceptions (sqlalchemy, snowflake)
        raise RuntimeError(f"{common_err_msg}\n{message}")

    if not is_independent:
        (
            df_type,
            converted_colnames,
            unsupported_columns,
            unsupported_arrow_types,
            pyarrow_table_schema,
        ) = comm.bcast(
            (
                df_type,
                converted_colnames,
                unsupported_columns,
                unsupported_arrow_types,
                pyarrow_table_schema,
            )
        )
    df_type = df_type.copy(data=tuple(t for t in df_type.data))

    return (
        df_type,
        converted_colnames,
        unsupported_columns,
        unsupported_arrow_types,
        pyarrow_table_schema,
    )


def _check_storage_options(
    storage_options, func_name: str, rhs, check_fields: bool = True
):
    """
    Error checking for storage_options usage in read_parquet/json/csv
    """

    if isinstance(storage_options, dict):
        # Early exit when allowing for more storage_options.
        # Used only in pd.read_parquet for now
        if not check_fields:
            return

        supported_storage_options = ("anon",)
        unsupported_storage_options = set(storage_options.keys()) - set(
            supported_storage_options
        )
        if unsupported_storage_options:
            raise BodoError(
                f"pd.{func_name}() arguments {unsupported_storage_options} for 'storage_options' not supported yet",
                loc=rhs.loc,
            )

        if "anon" in storage_options:
            if not isinstance(storage_options["anon"], bool):
                raise BodoError(
                    f"pd.{func_name}(): 'anon' in 'storage_options' must be a constant boolean value",
                    loc=rhs.loc,
                )
    elif storage_options is not None:
        raise BodoError(
            f"pd.{func_name}(): 'storage_options' must be a constant dictionary",
            loc=rhs.loc,
        )


class CSVFileInfo(FileInfo):
    """FileInfo object passed to ForceLiteralArg for
    file name arguments that refer to a CSV dataset"""

    def __init__(
        self,
        sep,
        skiprows,
        header,
        compression,
        low_memory,
        escapechar,
        storage_options,
        csv_sample_nrows,
    ):
        self.sep = sep
        self.skiprows = skiprows
        self.header = header
        self.compression = compression
        self.low_memory = low_memory
        self.escapechar = escapechar
        self.storage_options = storage_options
        self.csv_sample_nrows = csv_sample_nrows
        super().__init__()

    # TODO: Return type is not consistent with base class and Parquet?
    def _get_schema(self, fname):
        return _get_csv_df_type_from_file(
            fname,
            self.sep,
            self.skiprows,
            self.header,
            self.compression,
            self.low_memory,
            self.escapechar,
            self.storage_options,
            self.csv_sample_nrows,
        )


def _get_csv_df_type_from_file(
    fname_const,
    sep,
    skiprows,
    header,
    compression,
    low_memory,
    escapechar,
    csv_storage_options,
    csv_sample_nrows=100,
):
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
            is_handler, file_name_or_handler, file_compression, _ = (
                find_file_name_or_handler(fname_const, "csv", csv_storage_options)
            )

            if is_handler and compression == "infer":
                # pandas can't infer compression without filename, we need to do it
                compression = file_compression

            df = pd.read_csv(
                file_name_or_handler,
                sep=sep,
                nrows=csv_sample_nrows,
                skiprows=skiprows,
                header=header,
                compression=compression,
                # Copy low memory value from runtime.
                low_memory=low_memory,
                escapechar=escapechar,
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


def _check_int_list(list_val):
    """check whether list_val is list/tuple and its elements are of type int"""
    return isinstance(list_val, (list, tuple)) and all(
        isinstance(val, int) for val in list_val
    )
