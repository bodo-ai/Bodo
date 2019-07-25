"""
transforms the IR to remove features that Numba's type inference cannot support
such as non-uniform dictionary input of `pd.DataFrame({})`.
"""
import warnings
import itertools
import pandas as pd
import numpy as np
import math

import numba
from numba import ir, ir_utils, types
from numba.targets.registry import CPUDispatcher

from numba.ir_utils import (
    mk_unique_var, find_topo_order, dprint_func_ir, remove_dead, remove_dels,
    replace_var_names, find_const, compile_to_numba_ir, replace_arg_nodes,
    find_callname, guard, require, get_definition, build_definitions,
    replace_vars_stmt, find_build_sequence)

from numba.inline_closurecall import inline_closure_call
from numba.analysis import compute_cfg_from_blocks

import bodo
from bodo import config
import bodo.io
from bodo.io import pio, parquet_pio
from bodo.io.parquet_pio import ParquetHandler
from bodo.utils.utils import (
    inline_new_blocks, ReplaceFunc, is_call, is_assign)
import bodo.hiframes.api
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.libs.str_arr_ext import string_array_type
import bodo.ir
import bodo.ir.aggregate
import bodo.ir.filter
import bodo.ir.join
import bodo.ir.sort
from bodo.ir import csv_ext

from bodo.hiframes.pd_categorical_ext import (
    PDCategoricalDtype, CategoricalArray)
import bodo.hiframes.pd_dataframe_ext
from bodo.utils.transform import update_locs


def remove_hiframes(rhs, lives, call_list):
    # used in stencil generation of rolling
    if len(call_list) == 1 and call_list[0] in [int, min, max, abs]:
        return True
    # used in stencil generation of rolling
    if (len(call_list) == 1 and isinstance(call_list[0], CPUDispatcher)
            and call_list[0].py_func == numba.stencilparfor._compute_last_ind):
        return True
    # used in stencil generation of rolling
    if call_list == ['ceil', math]:
        return True
    if (len(call_list) == 4 and call_list[1:] == ['api', 'hiframes', bodo] and
            call_list[0] in ['fix_rolling_array',
            'concat', 'count', 'mean', 'quantile', 'var',
            'str_contains_regex', 'str_contains_noregex',
            'nunique', 'init_series', 'init_datetime_index',
            'init_timedelta_index',
            'get_series_data', 'get_series_index', 'get_series_name',
            'convert_tup_to_rec', 'convert_rec_to_tup']):
        return True
    if (len(call_list) == 4
            and call_list[1:] == ['pd_index_ext', 'hiframes', bodo] and
            call_list[0] in ('init_string_index',
            'init_numeric_index', '_dti_val_finalize')):
        return True
    if len(call_list) == 4 and call_list[1:] == ['conversion', 'utils', bodo]:
        # all conversion functions are side effect-free
        return True
    if (len(call_list) == 4 and call_list[1:] == ['series_kernels', 'hiframes', bodo] and
            call_list[0]
            in ['_sum_handle_nan', '_mean_handle_nan', '_var_handle_nan']):
        return True
    if call_list == ['dist_return', 'distributed_api', bodo]:
        return True
    if call_list == ['init_dataframe', 'pd_dataframe_ext', 'hiframes', bodo]:
        return True
    if call_list == ['get_dataframe_data', 'pd_dataframe_ext', 'hiframes', bodo]:
        return True
    if call_list == ['get_dataframe_index', 'pd_dataframe_ext', 'hiframes', bodo]:
        return True
    # if call_list == ['set_parent_dummy', 'pd_dataframe_ext', 'hiframes', bodo]:
    #     return True
    if call_list == ['rolling_dummy', 'pd_rolling_ext', 'hiframes', bodo]:
        return True
    if call_list == [list]:
        return True
    if call_list == ['groupby']:
        return True
    if call_list == ['rolling']:
        return True
    if call_list == [pd.api.types.CategoricalDtype]:
        return True
    # TODO: move to Numba
    if call_list == ['empty_inferred', 'ndarray', 'unsafe', numba]:
        return True
    if call_list == ['chain', itertools]:
        return True
    return False


numba.ir_utils.remove_call_handlers.append(remove_hiframes)


class UntypedPass(object):
    """
    Transformations before typing to enable type inference.
    """

    def __init__(self, func_ir, typingctx, args, _locals, metadata):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.args = args
        self.locals = _locals
        self.metadata = metadata
        ir_utils._max_label = max(func_ir.blocks.keys())
        # replace inst variables as determined previously during the pass
        # currently use to keep lhs of Arg nodes intact
        self.replace_var_dict = {}
        # labels of rhs of assignments to enable finding nodes that create
        # dataframes such as Arg(df_type), pd.DataFrame(), df[['A','B']]...
        # the use is conservative and doesn't assume complete info
        self.rhs_labels = {}

        self.arrow_tables = {}
        self.reverse_copies = {}
        self.pq_handler = ParquetHandler(
            func_ir, typingctx, args, _locals, self.reverse_copies)
        self.h5_handler = pio.PIO(self.func_ir, _locals, self.reverse_copies)


    def run(self):
        # XXX: the block structure shouldn't change in this pass since labels
        # are used in analysis (e.g. df creation points in rhs_labels)
        # FIXME: see why this breaks test_kmeans
        # remove_dels(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "starting untyped pass")
        self._handle_metadata()
        blocks = self.func_ir.blocks
        # call build definition since rewrite pass doesn't update definitions
        # e.g. getitem to static_getitem in test_column_list_select2
        self.func_ir._definitions = build_definitions(blocks)
        # topo_order necessary since df vars need to be found before use
        topo_order = find_topo_order(blocks)
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        while work_list:
            label, block = work_list.pop()
            self._get_reverse_copies(blocks[label].body)
            new_body = []
            self._working_body = new_body
            for inst in block.body:
                self._replace_vars(inst)
                # ir_utils.replace_vars_stmt(inst, self.replace_var_dict)
                out_nodes = [inst]

                # handle potential dataframe set column here
                # df['col'] = arr
                if (isinstance(inst, ir.StaticSetItem)
                        and isinstance(inst.index, str)):
                    # cfg needed for set df column
                    cfg = compute_cfg_from_blocks(blocks)
                    out_nodes = self._run_df_set_column(inst, label, cfg)
                elif isinstance(inst, ir.Assign):
                    self.func_ir._definitions[inst.target.name].remove(inst.value)
                    self.rhs_labels[inst.value] = label
                    out_nodes = self._run_assign(inst, label)
                elif isinstance(inst, ir.Return):
                    out_nodes = self._run_return(inst)

                assert isinstance(out_nodes, list)
                # TODO: fix scope/loc
                new_body.extend(out_nodes)
                self._update_definitions(out_nodes)
                for inst in out_nodes:
                    if is_assign(inst):
                        self.rhs_labels[inst.value] = label

            blocks[label].body = new_body

        self.func_ir.blocks = ir_utils.simplify_CFG(self.func_ir.blocks)
        # self.func_ir._definitions = build_definitions(blocks)
        # XXX: remove dead here fixes h5 slice issue
        # iterative remove dead to make sure all extra code (e.g. df vars) is removed
        # while remove_dead(blocks, self.func_ir.arg_names, self.func_ir):
        #     pass
        self.func_ir._definitions = build_definitions(blocks)
        dprint_func_ir(self.func_ir, "after untyped pass")
        return

    def _replace_vars(self, inst):
        # variable replacement can affect definitions so handling assignment
        # values specifically
        if is_assign(inst):
            lhs = inst.target.name
            self.func_ir._definitions[lhs].remove(inst.value)

        ir_utils.replace_vars_stmt(inst, self.replace_var_dict)

        if is_assign(inst):
            self.func_ir._definitions[lhs].append(inst.value)
            # if lhs changed, TODO: test
            if inst.target.name != lhs:
                self.func_ir._definitions[inst.target.name] = self.func_ir._definitions[lhs]

    def _run_assign(self, assign, label):
        lhs = assign.target.name
        rhs = assign.value

        if isinstance(rhs, ir.Expr):
            if rhs.op == 'call':
                return self._run_call(assign, label)

            # fix type for f['A'][:] dset reads
            if config._has_h5py and rhs.op in ('getitem', 'static_getitem'):
                h5_nodes = self.h5_handler.handle_possible_h5_read(
                    assign, lhs, rhs)
                if h5_nodes is not None:
                    return h5_nodes

            # HACK: delete pd.DataFrame({}) nodes to avoid typing errors
            # TODO: remove when dictionaries are implemented and typing works
            if rhs.op == 'getattr':
                val_def = guard(get_definition, self.func_ir, rhs.value)
                if (isinstance(val_def, ir.Global) and val_def.value == pd
                        and rhs.attr in ('read_csv',
                                        'read_parquet', 'to_numeric')):
                    # TODO: implement to_numeric in typed pass?
                    # put back the definition removed earlier but remove node
                    # enables function matching without node in IR
                    self.func_ir._definitions[lhs].append(rhs)
                    return []

            if rhs.op == 'getattr':
                val_def = guard(get_definition, self.func_ir, rhs.value)
                if (isinstance(val_def, ir.Global) and val_def.value == np
                        and rhs.attr == 'fromfile'):
                    # put back the definition removed earlier but remove node
                    self.func_ir._definitions[lhs].append(rhs)
                    return []

            # HACK: delete pyarrow.parquet.read_table() to avoid typing errors
            if rhs.op == 'getattr' and rhs.attr == 'read_table':
                import pyarrow.parquet as pq
                val_def = guard(get_definition, self.func_ir, rhs.value)
                if isinstance(val_def, ir.Global) and val_def.value == pq:
                    # put back the definition removed earlier but remove node
                    self.func_ir._definitions[lhs].append(rhs)
                    return []

            if (rhs.op == 'getattr' and rhs.value.name in self.arrow_tables
                    and rhs.attr == 'to_pandas'):
                # put back the definition removed earlier but remove node
                self.func_ir._definitions[lhs].append(rhs)
                return []

            #if rhs.op in ('build_list', 'build_tuple'): TODO: test tuple
            if rhs.op == 'build_list':
                # if build_list items are constant, add the constant values
                # to the returned list type as metadata. This enables type
                # inference for calls like pd.merge() where the values
                # determine output dataframe type
                # TODO: add proper metadata to Numba types
                # XXX: when constants are used, all the uses of the list object
                # have to be checked since lists are mutable
                try:
                    vals = tuple(find_const(self.func_ir, v) for v in rhs.items)
                    # a = ['A', 'B'] ->
                    # tmp = ['A', 'B']
                    # a = add_consts_to_type(tmp, 'A', 'B')
                    vals_expr = ", ".join("'{}'".format(c) if isinstance(c, str) else "{}".format(c) for c in vals)
                    func_text = "def _build_f(a):\n"
                    func_text += "  return bodo.utils.typing.add_consts_to_type(a, {})\n".format(vals_expr)
                    loc_vars = {}
                    exec(func_text, {'bodo': bodo}, loc_vars)
                    _build_f = loc_vars['_build_f']
                    target = assign.target
                    tmp_target = ir.Var(
                        target.scope, mk_unique_var(target.name), rhs.loc)
                    tmp_assign = ir.Assign(rhs, tmp_target, rhs.loc)
                    nodes = [tmp_assign]
                    nodes += _compile_func_single_block(
                        _build_f, (tmp_target,), assign.target)
                    return nodes
                except numba.ir_utils.GuardException:
                    pass

            if rhs.op == 'make_function':
                # HACK make globals availabe for typing in series.map()
                rhs.globals = self.func_ir.func_id.func.__globals__

        # pass pivot values to df.pivot_table() calls using a meta
        # variable passed as argument. The meta variable's type
        # is set to MetaType with pivot values baked in.
        pivot_key = lhs + ":pivot"
        if pivot_key in self.locals:
            pivot_values = self.locals.pop(pivot_key)
            # put back the definition removed earlier
            self.func_ir._definitions[lhs].append(rhs)
            pivot_call = guard(get_definition, self.func_ir, lhs)
            assert pivot_call is not None
            meta_var = ir.Var(
                assign.target.scope, mk_unique_var('pivot_meta'), rhs.loc)
            meta_assign = ir.Assign(
                ir.Const(0, rhs.loc), meta_var, rhs.loc)
            self._working_body.insert(0, meta_assign)
            pivot_call.kws = list(pivot_call.kws)
            pivot_call.kws.append(('_pivot_values', meta_var))
            self.locals[meta_var.name] = bodo.utils.typing.MetaType(
                pivot_values)

        # handle copies lhs = f
        if isinstance(rhs, ir.Var) and rhs.name in self.arrow_tables:
            self.arrow_tables[lhs] = self.arrow_tables[rhs.name]
            # enables function matching without node in IR
            self.func_ir._definitions[lhs].append(rhs)
            return []
        return [assign]

    def _run_call(self, assign, label):
        """handle calls and return new nodes if needed
        """
        lhs = assign.target
        rhs = assign.value

        func_name = ""
        func_mod = ""
        fdef = guard(find_callname, self.func_ir, rhs)
        if fdef is None:
            # could be make_function from list comprehension which is ok
            func_def = guard(get_definition, self.func_ir, rhs.func)
            if isinstance(func_def, ir.Expr) and func_def.op == 'make_function':
                return [assign]
            warnings.warn(
                "function call couldn't be found for initial analysis")
            return [assign]
        else:
            func_name, func_mod = fdef

        # handling pd.DataFrame() here since input can be constant dictionary
        if fdef == ('DataFrame', 'pandas'):
            return self._handle_pd_DataFrame(assign, lhs, rhs, label)

        # handling pd.read_csv() here since input can have constants
        # like dictionaries for typing
        if fdef == ('read_csv', 'pandas'):
            return self._handle_pd_read_csv(assign, lhs, rhs, label)

        # match flatmap pd.Series(list(itertools.chain(*A))) and flatten
        if fdef == ('Series', 'pandas'):
            return self._handle_pd_Series(assign, lhs, rhs)

        if fdef == ('read_table', 'pyarrow.parquet'):
            return self._handle_pq_read_table(assign, lhs, rhs)

        if (func_name == 'to_pandas' and isinstance(func_mod, ir.Var)
                and func_mod.name in self.arrow_tables):
            return self._handle_pq_to_pandas(assign, lhs, rhs, func_mod)

        if fdef == ('read_parquet', 'pandas'):
            return self._handle_pd_read_parquet(assign, lhs, rhs)

        if fdef == ('concat', 'pandas'):
            return self._handle_concat(assign, lhs, rhs, label)

        if fdef == ('to_numeric', 'pandas'):
            return self._handle_pd_to_numeric(assign, lhs, rhs)

        if func_name == 'drop' and isinstance(func_mod, ir.Var):
            # handle potential df.drop(inplace=True) here since it needs
            # variable replacement
            kws = dict(rhs.kws)
            inplace_var = self._get_arg(
                'drop', rhs.args, kws, 5, 'inplace', '')
            replace_func = lambda: bodo.hiframes.api.drop_inplace
            return self._handle_df_inplace_func(assign, lhs, rhs, func_mod,
                inplace_var, replace_func, label)

        if func_name == 'sort_values' and isinstance(func_mod, ir.Var):
            # handle potential df.sort_values(inplace=True) here since it needs
            # variable replacement
            kws = dict(rhs.kws)
            inplace_var = self._get_arg(
                'sort_values', rhs.args, kws, 3, 'inplace', '')
            replace_func = lambda: bodo.hiframes.api.sort_values_inplace
            return self._handle_df_inplace_func(assign, lhs, rhs, func_mod,
                inplace_var, replace_func, label)

        if config._has_h5py and fdef == ('File', 'h5py'):
            return self.h5_handler._handle_h5_File_call(assign, lhs, rhs)

        if fdef == ('fromfile', 'numpy'):
            return bodo.io.np_io._handle_np_fromfile(assign, lhs, rhs)

        if fdef == ('list', 'builtins') and len(rhs.args) == 1:
            arg_val = guard(find_const, self.func_ir, rhs.args[0])
            if isinstance(arg_val, str):
                # a = list('AB') ->
                # tmp = ['A', 'B']
                # a = add_consts_to_type(tmp, 'A', 'B')
                vals_expr = ", ".join("'{}'".format(c) for c in arg_val)
                func_text = "def _build_f(a):\n"
                func_text += "  return bodo.utils.typing.add_consts_to_type(a, {})\n".format(vals_expr)
                loc_vars = {}
                exec(func_text, {'bodo': bodo}, loc_vars)
                _build_f = loc_vars['_build_f']
                target = assign.target
                tmp_target = ir.Var(
                    target.scope, mk_unique_var(target.name), rhs.loc)
                tmp_assign = ir.Assign(rhs, tmp_target, rhs.loc)
                nodes = [tmp_assign]
                nodes += _compile_func_single_block(
                    _build_f, (tmp_target,), lhs)
                return nodes

        # TODO: update Xenon handling code
        # if fdef == ('read_xenon', 'bodo.xenon_ext'):
        #     col_items, nodes = bodo.xenon_ext._handle_read(assign, lhs, rhs, self.func_ir)
        #     df_nodes, col_map = self._process_df_build_map(col_items)
        #     self._create_df(lhs.name, col_map, label)
        #     nodes += df_nodes
        #     return nodes

        return [assign]

    def _handle_df_inplace_func(self, assign, lhs, rhs, df_var, inplace_var,
                                                          replace_func, label):
        """handle possible df.func(inplace=True)
        lhs = A.func(inplace=True) -> A1, lhs = func_inplace(...)
        replace A with A1
        """
        inplace = guard(find_const, self.func_ir, inplace_var)
        dominates = False
        df_def = guard(get_definition, self.func_ir, df_var)
        cfg = compute_cfg_from_blocks(self.func_ir.blocks)
        if (df_def in self.rhs_labels
                and label in cfg.post_dominators()[self.rhs_labels[df_def]]):
            dominates = True
        if inplace is not None and inplace == True and dominates:
            # TODO: make sure call post dominates df_var definition or df_var
            # is not used in other code paths
            # replace func variable with replace_func
            f_block = compile_to_numba_ir(
                replace_func,
                {'bodo': bodo}).blocks.popitem()[1]
            nodes = f_block.body[:-2]
            new_func_var = nodes[-1].target
            rhs.func = new_func_var
            rhs.args.insert(0, df_var)
            # new tuple return
            ret_tup = ir.Var(lhs.scope, mk_unique_var('tuple_ret'), lhs.loc)
            assign.target = ret_tup
            nodes.append(assign)
            new_df_var = ir.Var(df_var.scope, mk_unique_var(df_var.name), df_var.loc)
            zero_var = ir.Var(df_var.scope, mk_unique_var('zero'), df_var.loc)
            one_var = ir.Var(df_var.scope, mk_unique_var('one'), df_var.loc)
            nodes.append(ir.Assign(ir.Const(0, lhs.loc), zero_var, lhs.loc))
            nodes.append(ir.Assign(ir.Const(1, lhs.loc), one_var, lhs.loc))
            getitem0 = ir.Expr.static_getitem(ret_tup, 0, zero_var, lhs.loc)
            nodes.append(ir.Assign(getitem0, new_df_var, lhs.loc))
            getitem1 = ir.Expr.static_getitem(ret_tup, 1, one_var, lhs.loc)
            nodes.append(ir.Assign(getitem1, lhs, lhs.loc))
            # replace old variable with new one
            self.replace_var_dict[df_var.name] = new_df_var
            return nodes

        return [assign]

    def _get_reverse_copies(self, body):
        for inst in body:
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var):
                self.reverse_copies[inst.value.name] = inst.target.name
        return

    def _handle_pd_DataFrame(self, assign, lhs, rhs, label):
        """
        Enable typing for dictionary data arg to pd.DataFrame({'A': A}) call.
        Converts constant dictionary to tuple with sentinel if present.
        """
        nodes = [assign]
        kws = dict(rhs.kws)
        data_arg = self._get_arg('pd.DataFrame', rhs.args, kws, 0, 'data')
        index_arg = self._get_arg('pd.DataFrame', rhs.args, kws, 1, 'index', '')

        arg_def = guard(get_definition, self.func_ir, data_arg)
        if isinstance(arg_def, ir.Expr) and arg_def.op == 'build_map':
            # check column names to be string
            col_names = tuple(guard(find_const, self.func_ir, t[0])
                              for t in arg_def.items)
            if not all(isinstance(c, str) for c in col_names):
                # TODO: support int column names?
                raise ValueError(
                    "DataFrame column names should be constant strings")

            # create tuple with sentinel
            sentinel_var = ir.Var(lhs.scope, mk_unique_var('sentinel'), lhs.loc)
            tup_var = ir.Var(lhs.scope, mk_unique_var('dict_tup'), lhs.loc)
            new_nodes = [ir.Assign(ir.Const(
                '__bodo_tup', lhs.loc), sentinel_var, lhs.loc)]
            tup_items = ([sentinel_var] + [t[0] for t in arg_def.items]
                         + [t[1] for t in arg_def.items])
            new_nodes.append(ir.Assign(
                ir.Expr.build_tuple(tup_items, lhs.loc), tup_var, lhs.loc))

            # replace data arg with dict tuple
            if 'data' in kws:
                kws['data'] = tup_var
                rhs.kws = list(kws.items())
            else:
                rhs.args[0] = tup_var

            nodes = new_nodes + nodes
            # HACK replace build_map to avoid inference errors
            # TODO: don't replace if used in other places
            arg_def.op = 'build_list'
            arg_def.items = [v[0] for v in arg_def.items]

        # replace range() with pd.RangeIndex() for index argument
        arg_def = guard(get_definition, self.func_ir, index_arg)
        if is_call(arg_def) and guard(
                find_callname, self.func_ir, arg_def) == ('range', 'builtins'):
            # gen pd.RangeIndex() call
            def _call_range_index():
                return pd.RangeIndex()
            f_block = compile_to_numba_ir(
                _call_range_index, {'pd': pd}).blocks.popitem()[1]
            new_nodes = f_block.body[:-2]
            new_nodes[-1].value.args = arg_def.args
            new_index = new_nodes[-1].target
            # replace index arg
            if 'index' in kws:
                kws['index'] = new_index
                rhs.kws = list(kws.items())
            else:
                rhs.args[1] = new_index
            nodes = new_nodes + nodes

        return nodes

    def _handle_pd_read_csv(self, assign, lhs, rhs, label):
        """transform pd.read_csv(names=[A], dtype={'A': np.int32}) call
        """
        # schema: pd.read_csv(filepath_or_buffer, sep=',', delimiter=None,
        # header='infer', names=None, index_col=None, usecols=None,
        # squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None,
        # engine=None, converters=None, true_values=None, false_values=None,
        # skipinitialspace=False, skiprows=None, nrows=None, na_values=None,
        # keep_default_na=True, na_filter=True, verbose=False,
        # skip_blank_lines=True, parse_dates=False,
        # infer_datetime_format=False, keep_date_col=False, date_parser=None,
        # dayfirst=False, iterator=False, chunksize=None, compression='infer',
        # thousands=None, decimal=b'.', lineterminator=None, quotechar='"',
        # quoting=0, escapechar=None, comment=None, encoding=None,
        # dialect=None, tupleize_cols=None, error_bad_lines=True,
        # warn_bad_lines=True, skipfooter=0, doublequote=True,
        # delim_whitespace=False, low_memory=True, memory_map=False,
        # float_precision=None)

        kws = dict(rhs.kws)
        fname = self._get_arg(
            'read_csv', rhs.args, kws, 0, 'filepath_or_buffer')
        sep = self._get_str_arg('read_csv', rhs.args, kws, 1, 'sep', ',')
        sep = self._get_str_arg('read_csv', rhs.args, kws, 2, 'delimiter', sep)
        header = self._get_str_arg(
            'read_csv', rhs.args, kws, 3, 'header', 'infer')
        names_var = self._get_arg('read_csv', rhs.args, kws, 4, 'names', '')
        index_col = self._get_str_arg(
            'read_csv', rhs.args, kws, 5, 'index_col', -1)
        usecols_var = self._get_arg(
            'read_csv', rhs.args, kws, 6, 'usecols', '')
        dtype_var = self._get_arg('read_csv', rhs.args, kws, 10, 'dtype', '')
        skiprows = self._get_str_arg(
            'read_csv', rhs.args, kws, 16, 'skiprows', 0)

        # check unsupported arguments
        supported_args = ('filepath_or_buffer', 'sep', 'delimiter', 'header',
            'names', 'index_col', 'usecols', 'dtype', 'skiprows',
            'parse_dates')
        unsupported_args = set(kws.keys()) - set(supported_args)
        if unsupported_args:
            raise ValueError(
                "read_csv() arguments {} not supported yet".format(
                    unsupported_args))

        col_names = self._get_str_or_list(names_var, default=0)

        # infer the column names: if no names
        # are passed the behavior is identical to ``header=0`` and column
        # names are inferred from the first line of the file, if column
        # names are passed explicitly then the behavior is identical to
        # ``header=None``
        if header == 'infer':
            header = 0 if col_names == 0 else None

        # TODO: support string usecols
        usecols = None
        if usecols_var != '':
            err_msg = "pd.read_csv() usecols should be constant list of ints"
            usecols = self._get_str_or_list(
                usecols_var, err_msg=err_msg, typ=int)

        # if inference is required
        if dtype_var is '' or col_names == 0:
            # infer column names and types from constant filename
            fname_const = guard(find_const, self.func_ir, fname)
            if fname_const is None:
                raise ValueError("pd.read_csv() requires explicit type"
                    "annotation using 'dtype' if filename is not constant")
            rows_to_read = 100  # TODO: tune this
            df = pd.read_csv(
                fname_const, sep=sep, nrows=rows_to_read, skiprows=skiprows,
                header=header)
            # TODO: categorical, etc.
            dtypes = numba.typeof(df).data

            usecols = list(range(len(dtypes))) if usecols is None else usecols
            # convert Pandas generated integer names if any
            cols = [str(df.columns[i]) for i in usecols]
            # overwrite column names like Pandas if explicitly provided
            if col_names != 0:
                cols[-len(col_names):] = col_names
            col_names = cols
            dtype_map = {c: dtypes[usecols[i]]
                for i, c in enumerate(col_names)}

        usecols = list(range(len(col_names))) if usecols is None else usecols

        if header is not None:
            # data starts after header
            skiprows += header + 1

        # handle dtype arg if provided
        if dtype_var != '':
            dtype_map = guard(get_definition, self.func_ir, dtype_var)
            if (not isinstance(dtype_map, ir.Expr)
                    or dtype_map.op != 'build_map'):  # pragma: no cover
                # try single type for all columns case
                dtype_map = self._get_const_dtype(dtype_var)
            else:
                new_dtype_map = {}
                for n_var, t_var in dtype_map.items:
                    # find constant column name
                    c = guard(find_const, self.func_ir, n_var)
                    if c is None:  # pragma: no cover
                        raise ValueError("dtype column names should be constant")
                    new_dtype_map[c] = self._get_const_dtype(t_var)

                # HACK replace build_map to avoid inference errors
                dtype_map.op = 'build_list'
                dtype_map.items = [v[0] for v in dtype_map.items]
                dtype_map = new_dtype_map

        if col_names == 0:
            raise ValueError("pd.read_csv() names should be constant list")

        # TODO: support other args

        date_cols = []
        if 'parse_dates' in kws:
            err_msg = "pd.read_csv() parse_dates should be constant list"
            date_cols = self._get_str_or_list(kws['parse_dates'], err_msg=err_msg, typ=int)

        columns, data_arrs, out_types = self._get_csv_col_info(
            dtype_map, date_cols, col_names, lhs)

        nodes = [csv_ext.CsvReader(
            fname, lhs.name, sep, columns, data_arrs, out_types, usecols,
            lhs.loc, skiprows)]

        columns = columns.copy()  # copy since modified below
        n_cols = len(columns)
        args = ['data{}'.format(i) for i in range(n_cols)]
        data_args = args.copy()
        index_arg = 'None'

        # one column is index
        if index_col != -1:
            # convert column number to column name
            if isinstance(index_col, int):
                index_col = columns[index_col]
            index_ind = columns.index(index_col)
            index_arg = data_args[index_ind]
            columns.remove(index_col)
            data_args.remove(data_args[index_ind])

        col_args = ", ".join("'{}'".format(c) for c in columns)

        col_var = "bodo.utils.typing.add_consts_to_type([{}], {})".format(col_args, col_args)
        func_text = "def _init_df({}):\n".format(", ".join(args))
        func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), {}, {})\n".format(
            ", ".join(data_args), index_arg, col_var)
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        # print(func_text)
        _init_df = loc_vars['_init_df']

        nodes += _compile_func_single_block(_init_df, data_arrs, lhs)
        return nodes

    def _get_csv_col_info(self, dtype_map, date_cols, col_names, lhs):
        if isinstance(dtype_map, types.Type):
            typ = dtype_map
            data_arrs = [ir.Var(lhs.scope, mk_unique_var(cname), lhs.loc)
                        for cname in col_names]
            return col_names, data_arrs, [typ]*len(col_names)

        columns = []
        data_arrs = []
        out_types = []
        for i, (col_name, typ) in enumerate(dtype_map.items()):
            columns.append(col_name)
            # get array dtype
            if i in date_cols:
                typ = types.Array(types.NPDatetime('ns'), 1, 'C')
            out_types.append(typ)
            # output array variable
            data_arrs.append(
                ir.Var(lhs.scope, mk_unique_var(col_name), lhs.loc))

        return columns, data_arrs, out_types

    def _get_const_dtype(self, dtype_var):
        dtype_def = guard(get_definition, self.func_ir, dtype_var)
        if isinstance(dtype_def, ir.Const) and isinstance(dtype_def.value, str):
            typ_name = dtype_def.value
            if typ_name == 'str':
                return string_array_type
            typ_name = 'int64' if typ_name == 'int' else typ_name
            typ_name = 'float64' if typ_name == 'float' else typ_name
            typ = getattr(types, typ_name)
            typ = types.Array(typ, 1, 'C')
            return typ

        # str case
        if isinstance(dtype_def, ir.Global) and dtype_def.value == str:
            return string_array_type
        # categorical case
        if isinstance(dtype_def, ir.Expr) and dtype_def.op == 'call':
            if (not guard(find_callname, self.func_ir, dtype_def)
                    == ('category', 'pandas.core.dtypes.dtypes')):
                raise ValueError("pd.read_csv() invalid dtype "
                    "(built using a call but not Categorical)")
            cats_var = self._get_arg('CategoricalDtype', dtype_def.args,
                dict(dtype_def.kws), 0, 'categories')
            err_msg = "categories should be constant list"
            cats = self._get_str_or_list(cats_var, list_only=True, err_msg=err_msg)
            typ = PDCategoricalDtype(cats)
            return CategoricalArray(typ)
        if not isinstance(dtype_def, ir.Expr) or dtype_def.op != 'getattr':
            raise ValueError("pd.read_csv() invalid dtype")
        glob_def = guard(get_definition, self.func_ir, dtype_def.value)
        if not isinstance(glob_def, ir.Global) or glob_def.value != np:
            raise ValueError("pd.read_csv() invalid dtype")
        # TODO: extend to other types like string and date, check error
        typ_name = dtype_def.attr
        typ_name = 'int64' if typ_name == 'int' else typ_name
        typ_name = 'float64' if typ_name == 'float' else typ_name
        typ = getattr(types, typ_name)
        typ = types.Array(typ, 1, 'C')
        return typ

    def _handle_pd_Series(self, assign, lhs, rhs):
        """transform pd.Series(A) call for flatmap case
        """
        kws = dict(rhs.kws)
        data = self._get_arg('pd.Series', rhs.args, kws, 0, 'data')

        # match flatmap pd.Series(list(itertools.chain(*A))) and flatten
        data_def = guard(get_definition, self.func_ir, data)
        if (is_call(data_def) and guard(find_callname, self.func_ir, data_def)
                == ('list', 'builtins') and len(data_def.args) == 1):
            arg_def = guard(get_definition, self.func_ir, data_def.args[0])
            if (is_call(arg_def) and guard(find_callname, self.func_ir,
                    arg_def) == ('chain', 'itertools')):
                in_data = arg_def.vararg
                arg_def.vararg = None  # avoid typing error
                return _compile_func_single_block(
                    lambda l: bodo.utils.typing.flatten_to_series(l),
                    (in_data,),
                    lhs
                )

        # pd.Series() is handled in typed pass now
        return [assign]

    def _handle_pd_to_numeric(self, assign, lhs, rhs):
        """transform pd.to_numeric(A, errors='coerce') call here since dtype
        has to be specified in locals and applied
        """
        kws = dict(rhs.kws)
        if 'errors' not in kws or guard(find_const, self.func_ir, kws['errors']) != 'coerce':
            raise ValueError("pd.to_numeric() only supports errors='coerce'")

        if lhs.name not in self.reverse_copies or (self.reverse_copies[lhs.name]) not in self.locals:
            raise ValueError("pd.to_numeric() requires annotation of output type")

        typ = self.locals.pop(self.reverse_copies[lhs.name])
        dtype = numba.numpy_support.as_dtype(typ.dtype)
        arg = rhs.args[0]

        return _compile_func_single_block(
            lambda arr: bodo.hiframes.api.to_numeric(arr, _dtype),
            [arg], lhs, extra_globals={'_dtype': dtype})

    def _handle_pq_read_table(self, assign, lhs, rhs):
        if len(rhs.args) != 1:  # pragma: no cover
            raise ValueError("Invalid read_table() arguments")
        # put back the definition removed earlier but remove node
        self.func_ir._definitions[lhs.name].append(rhs)
        self.arrow_tables[lhs.name] = rhs.args[0]
        return []

    def _handle_pq_to_pandas(self, assign, lhs, rhs, t_var):
        return self._gen_parquet_read(self.arrow_tables[t_var.name], lhs)

    def _gen_parquet_read(self, fname, lhs, columns=None):
        # make sure pyarrow is available
        if not config._has_pyarrow:
            raise RuntimeError("pyarrow is required for Parquet support")

        columns, data_arrs, index_col, nodes = self.pq_handler.gen_parquet_read(
            fname, lhs, columns)
        n_cols = len(columns)
        args = ", ".join('data{}'.format(i) for i in range(n_cols))
        data_args = ", ".join('data{}'.format(i) for i in range(n_cols)
                       if (index_col is None or i != columns.index(index_col)))

        index_arg = 'None' if index_col is None else 'data{}'.format(
            columns.index(index_col))
        col_args = ", ".join("'{}'".format(c) for c in columns
                             if (index_col is None or c != index_col))
        col_var = "bodo.utils.typing.add_consts_to_type([{}], {})".format(
            col_args, col_args)
        func_text = "def _init_df({}):\n".format(args)
        func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), bodo.utils.conversion.convert_to_index({}), {})\n".format(
            data_args, index_arg, col_var)
        loc_vars = {}
        # print(func_text)
        exec(func_text, {}, loc_vars)
        _init_df = loc_vars['_init_df']
        nodes += _compile_func_single_block(_init_df, data_arrs, lhs)
        return nodes

    def _handle_pd_read_parquet(self, assign, lhs, rhs):
        # get args and check values
        kws = dict(rhs.kws)
        fname = self._get_arg('read_parquet', rhs.args, kws, 0, 'path')
        engine = self._get_arg(
            'read_parquet', rhs.args, kws, 1, 'engine', 'auto')
        if engine not in ('auto', 'pyarrow'):
            raise ValueError("read_parquet: only pyarrow engine supported")

        columns = self._get_arg(
            'read_parquet', rhs.args, kws, 2, 'columns', -1)
        if columns == -1:
            columns = None
        elif columns is not None:
            columns = self._get_str_or_list(columns)

        return self._gen_parquet_read(fname, lhs, columns)

    def _handle_concat(self, assign, lhs, rhs, label):
        # converting build_list to build_tuple before type inference to avoid
        # errors
        kws = dict(rhs.kws)
        objs_arg = self._get_arg('concat', rhs.args, kws, 0, 'objs')

        df_list = guard(get_definition, self.func_ir, objs_arg)
        if not isinstance(df_list, ir.Expr) or not (df_list.op
                                            in ['build_tuple', 'build_list']):
            raise ValueError("pd.concat input should be constant list or tuple")

        # XXX convert build_list to build_tuple since Numba doesn't handle list of
        # arrays for np.concatenate()
        if df_list.op == 'build_list':
            df_list.op = 'build_tuple'

        if len(df_list.items) == 0:
            # copied error from pandas
            raise ValueError("No objects to concatenate")

        return [assign]

    def _get_str_arg(self, f_name, args, kws, arg_no, arg_name, default=None,
                                                                 err_msg=None):
        arg = None
        if len(args) > arg_no:
            arg = guard(find_const, self.func_ir, args[arg_no])
        elif arg_name in kws:
            arg = guard(find_const, self.func_ir, kws[arg_name])

        if arg is None:
            if default is not None:
                return default
            if err_msg is None:
                err_msg = ("{} requires '{}' argument as a "
                             "constant string").format(f_name, arg_name)
            raise ValueError(err_msg)
        return arg

    def _get_arg(self, f_name, args, kws, arg_no, arg_name, default=None,
                                                                 err_msg=None):
        arg = None
        if len(args) > arg_no:
            arg = args[arg_no]
        elif arg_name in kws:
            arg = kws[arg_name]

        if arg is None:
            if default is not None:
                return default
            if err_msg is None:
                err_msg = "{} requires '{}' argument".format(f_name, arg_name)
            raise ValueError(err_msg)
        return arg

    def _get_str_or_list(self, by_arg, list_only=False, default=None, err_msg=None, typ=None):
        typ = str if typ is None else typ
        by_arg_def = guard(find_build_sequence, self.func_ir, by_arg)

        if by_arg_def is None:
            # try add_consts_to_type
            by_arg_call = guard(get_definition, self.func_ir, by_arg)
            if guard(find_callname, self.func_ir, by_arg_call) == ('add_consts_to_type', 'bodo.utils.typing'):
                by_arg_def = guard(find_build_sequence, self.func_ir, by_arg_call.args[0])

        if by_arg_def is None:
            # try dict.keys()
            by_arg_call = guard(get_definition, self.func_ir, by_arg)
            call_name = guard(find_callname, self.func_ir, by_arg_call)
            if (call_name is not None and len(call_name) == 2
                    and call_name[0] == 'keys'
                    and isinstance(call_name[1], ir.Var)):
                var_def = guard(get_definition, self.func_ir, call_name[1])
                if isinstance(var_def, ir.Expr) and var_def.op == 'build_map':
                    by_arg_def = [v[0] for v in var_def.items], 'build_map'
                    # HACK replace dict.keys getattr to avoid typing errors
                    keys_getattr = guard(
                        get_definition, self.func_ir, by_arg_call.func)
                    assert isinstance(
                        keys_getattr, ir.Expr) and keys_getattr.attr == 'keys'
                    keys_getattr.attr = 'copy'

        if by_arg_def is None:
            # try single key column
            by_arg_def = guard(find_const, self.func_ir, by_arg)
            if by_arg_def is None:
                if default is not None:
                    return default
                raise ValueError(err_msg)
            key_colnames = [by_arg_def]
        else:
            if list_only and by_arg_def[1] != 'build_list':
                if default is not None:
                    return default
                raise ValueError(err_msg)
            key_colnames = [guard(find_const, self.func_ir, v) for v in by_arg_def[0]]
            if any(not isinstance(v, typ) for v in key_colnames):
                if default is not None:
                    return default
                raise ValueError(err_msg)
        return key_colnames

    def _handle_metadata(self):
        """remove distributed input annotation from locals and add to metadata
        """
        if 'distributed' not in self.metadata:
            # TODO: keep updated in variable renaming?
            self.metadata['distributed'] = self.locals.pop(
                '##distributed', set())

        if 'distributed_varlength' not in self.metadata:
            self.metadata['distributed_varlength'] = self.locals.pop(
                '##distributed_varlength', set())

        if 'threaded' not in self.metadata:
            self.metadata['threaded'] = self.locals.pop('##threaded', set())

        if 'all_args_distributed' not in self.metadata:
            self.metadata['all_args_distributed'] = self.locals.pop(
                '##all_args_distributed', False)

        if 'all_args_distributed_varlength' not in self.metadata:
            self.metadata['all_args_distributed_varlength'] = self.locals.pop(
                '##all_args_distributed_varlength', False)

        if 'all_returns_distributed' not in self.metadata:
            self.metadata['all_returns_distributed'] = self.locals.pop(
                '##all_returns_distributed', False)

        # handle old input flags
        # e.g. {"A:input": "distributed"} -> "A"
        dist_inputs = { var_name.split(":")[0]
                    for (var_name, flag) in self.locals.items()
                    if var_name.endswith(":input") and flag == 'distributed'}

        thread_inputs = { var_name.split(":")[0]
                    for (var_name, flag) in self.locals.items()
                    if var_name.endswith(":input") and flag == 'threaded'}

        # check inputs to be in actuall args
        for arg_name in dist_inputs | thread_inputs:
            if arg_name not in self.func_ir.arg_names:
                raise ValueError(
                    "distributed input {} not found in arguments".format(
                        arg_name))
            self.locals.pop(arg_name + ":input")

        self.metadata['distributed'] |= dist_inputs
        self.metadata['threaded'] |= thread_inputs


        # handle old return flags
        # e.g. {"A:return":"distributed"} -> "A"
        flagged_returns = { var_name.split(":")[0]: flag
                    for (var_name, flag) in self.locals.items()
                    if var_name.endswith(":return") }

        for v, flag in flagged_returns.items():
            if flag == 'distributed':
                self.metadata['distributed'].add(v)
            elif flag == 'threaded':
                self.metadata['threaded'].add(v)

            self.locals.pop(v + ":return")

        return

    def _run_return(self, ret_node):
        # TODO: handle distributed analysis, requires handling variable name
        # change in simplify() and replace_var_names()
        # TODO: include distributed_varlength?
        flagged_vars = self.metadata['distributed'] | self.metadata['threaded']
        all_returns_distributed = self.metadata['all_returns_distributed']
        nodes = [ret_node]
        cast = guard(get_definition, self.func_ir, ret_node.value)
        assert cast is not None, "return cast not found"
        assert isinstance(cast, ir.Expr) and cast.op == 'cast'
        scope = cast.value.scope
        loc = cast.loc
        # XXX: using split('.') since the variable might be renamed (e.g. A.2)
        ret_name = cast.value.name.split('.')[0]

        if ret_name in flagged_vars or all_returns_distributed:
            flag = ('distributed' if (ret_name in self.metadata['distributed']
                        or all_returns_distributed) else 'threaded')
            nodes = self._gen_replace_dist_return(cast.value, flag)
            new_arr = nodes[-1].target
            new_cast = ir.Expr.cast(new_arr, loc)
            new_out = ir.Var(scope, mk_unique_var(flag + "_return"), loc)
            nodes.append(ir.Assign(new_cast, new_out, loc))
            ret_node.value = new_out
            nodes.append(ret_node)
            return nodes

        cast_def = guard(get_definition, self.func_ir, cast.value)
        if (cast_def is not None and isinstance(cast_def, ir.Expr)
                and cast_def.op == 'build_tuple'):
            nodes = []
            new_var_list = []
            for v in cast_def.items:
                vname = v.name.split('.')[0]
                if vname in flagged_vars or all_returns_distributed:
                    flag = ('distributed' if (
                            vname in self.metadata['distributed']
                            or all_returns_distributed) else 'threaded')
                    nodes += self._gen_replace_dist_return(v, flag)
                    new_var_list.append(nodes[-1].target)
                else:
                    new_var_list.append(v)
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
        if flag == 'distributed':
            def f(_dist_arr):  # pragma: no cover
                dist_return = bodo.libs.distributed_api.dist_return(_dist_arr)
        elif flag == 'threaded':
            def f(_threaded_arr):  # pragma: no cover
                _th_arr = bodo.libs.distributed_api.threaded_return(_threaded_arr)
        else:
            raise ValueError("Invalid return flag {}".format(flag))
        f_block = compile_to_numba_ir(
            f, {'bodo': bodo}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [var])
        return f_block.body[:-3]  # remove none return

    def _run_df_set_column(self, inst, label, cfg):
        """replace setitem of string index with a call to handle possible
        dataframe case where schema is changed:
        df['new_col'] = arr  ->  df2 = set_df_col(df, 'new_col', arr)
        dataframe_pass will replace set_df_col() with regular setitem if target
        is not dataframe
        """
        # setting column possible only when:
        #   1) it dominates the df creation, so we can create a new df variable
        #      to replace the existing variable for the rest of the program,
        #      which avoids inplace update and schema change possiblity
        #      TODO: make sure there is no other reference (refcount==1)
        #   2) setting existing column with same type (inplace)
        # invalid case:
        # df = pd.DataFrame({'A': A})
        # if cond:
        #     df['B'] = B
        # return df
        # TODO: add this check back in
        # if label not in cfg.backbone() and label not in cfg.post_dominators()[df_label]:
        #     raise ValueError("setting dataframe columns inside conditionals and"
        #                      " loops not supported yet")
        # see if setitem dominates creation, # TODO: handle changing labels
        df_var = inst.target
        df_def = guard(get_definition, self.func_ir, df_var)
        dominates = False
        if (df_def in self.rhs_labels
                and label in cfg.post_dominators()[self.rhs_labels[df_def]]):
            dominates = True

        # TODO: generalize to more cases
        # for example:
        # df = pd.DataFrame({'A': A})
        # if cond:
        #     df['B'] = B
        # else:
        #     df['B'] = C
        # return df
        # TODO: check for references to df
        # for example:
        # df = pd.DataFrame({'A': A})
        # df2 = df
        # df['B'] = C
        # return df2
        # create var for string index
        cname_var = ir.Var(inst.value.scope, mk_unique_var("$cname_const"), inst.loc)
        nodes = [ir.Assign(ir.Const(inst.index, inst.loc), cname_var, inst.loc)]
        inplace = not dominates

        func = lambda df, cname, arr: bodo.hiframes.api.set_df_col(df, cname, arr, _inplace)
        f_block = compile_to_numba_ir(func, {'bodo': bodo, '_inplace': inplace}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [df_var, cname_var, inst.value])
        nodes += f_block.body[:-2]

        if dominates:
            # rename the dataframe variable to keep schema static
            new_df_var = ir.Var(
                df_var.scope, mk_unique_var(df_var.name), df_var.loc)
            nodes[-1].target = new_df_var
            self.replace_var_dict[df_var.name] = new_df_var
        else:
            # cannot replace variable, but can set existing column with the
            # same data type
            # TODO: check data type and throw clear error
            nodes[-1].target = df_var

        return nodes

    def _update_definitions(self, node_list):
        loc = ir.Loc("", 0)
        dumm_block = ir.Block(ir.Scope(None, loc), loc)
        dumm_block.body = node_list
        build_definitions({0: dumm_block}, self.func_ir._definitions)
        return


def _compile_func_single_block(func, args, ret_var, extra_globals=None):
    """compiles functions that are just a single basic block.
    Does not handle defaults, freevars etc.
    """
    glbls = {'numba': numba, 'np': np, 'bodo': bodo, 'pd': pd}
    if extra_globals is not None:
        glbls.update(extra_globals)
    f_ir = compile_to_numba_ir(func, glbls)
    assert len(f_ir.blocks) == 1
    f_block = f_ir.blocks.popitem()[1]
    replace_arg_nodes(f_block, args)
    nodes = f_block.body[:-2]

    # update Loc objects, avoid changing input arg vars
    update_locs(nodes[len(args):], ret_var.loc)
    for stmt in nodes[:len(args)]:
        stmt.target.loc = ret_var.loc

    nodes[-1].target = ret_var
    return nodes
