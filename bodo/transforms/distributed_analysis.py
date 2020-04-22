# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
analyzes the IR to decide parallelism of arrays and parfors
for distributed transformation.
"""
from collections import namedtuple, defaultdict
import copy
import warnings
import inspect
from enum import Enum

import numba
from numba import ir, ir_utils, types
from numba.ir_utils import (
    find_topo_order,
    guard,
    get_definition,
    require,
    find_callname,
    mk_unique_var,
    compile_to_numba_ir,
    replace_arg_nodes,
    build_definitions,
    find_build_sequence,
    find_const,
)
from numba.parfor import Parfor
from numba.parfor import wrap_parfor_blocks, unwrap_parfor_blocks

import bodo
import bodo.io
import bodo.io.np_io
from bodo.utils.utils import (
    get_constant,
    is_alloc_callname,
    is_whole_slice,
    is_slice_equiv_arr,
    is_array_typ,
    is_np_array_typ,
    find_build_tuple,
    debug_prints,
    is_expr,
    is_distributable_typ,
    is_distributable_tuple_typ,
    is_static_getsetitem,
    get_getsetitem_index_var,
)
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.hiframes.pd_categorical_ext import CategoricalArray
from bodo.utils.transform import get_stmt_defs, get_call_expr_arg
from bodo.utils.typing import BodoWarning, BodoError
from bodo.libs.bool_arr_ext import boolean_array


class Distribution(Enum):
    REP = 1
    Thread = 2
    TwoD = 3
    OneD_Var = 4
    OneD = 5

    def __str__(self):
        name_map = {
            "OneD": "1D_Block",
            "OneD_Var": "1D_Block_Var",
            "TwoD": "2D_Block",
            "Thread": "Multi-thread",
            "REP": "REP",
        }
        return name_map[self.name]


_dist_analysis_result = namedtuple("dist_analysis_result", "array_dists,parfor_dists")


distributed_analysis_extensions = {}
auto_rebalance = False


class DistributedDiagnostics:
    """Gather and print distributed diagnostics information
    """

    def __init__(
        self, parfor_locs, array_locs, array_dists, parfor_dists, diag_info, func_ir
    ):
        self.parfor_locs = parfor_locs
        self.array_locs = array_locs
        self.array_dists = array_dists
        self.parfor_dists = parfor_dists
        self.diag_info = diag_info
        self.func_ir = func_ir

    def _print_dists(self):
        print("Data distributions:")
        if len(self.array_dists) > 0:
            arrname_width = max(len(a) for a in self.array_dists.keys())
            arrname_width = max(arrname_width + 3, 20)
            for arr, dist in self.array_dists.items():
                print("   {0:{1}} {2}".format(arr, arrname_width, dist))
        else:
            print("No distributable data structures to distribute.")

        print("\nParfor distributions:")
        if len(self.parfor_dists) > 0:
            for p, dist in self.parfor_dists.items():
                print("   {0:<20} {1}".format(p, dist))
        else:
            print("No parfors to distribute.")
        return

    def dump(self, level=1):
        name = self.func_ir.func_id.func_qualname
        line = self.func_ir.loc

        print("Distributed diagnostics for function {}, {}\n".format(name, line))
        self._print_dists()

        # similar to ParforDiagnostics.dump()
        func_name = self.func_ir.func_id.func
        try:
            lines = inspect.getsource(func_name).splitlines()
        except OSError:  # generated function
            lines = None

        if not lines:
            print("No source available")
            return

        print("\nDistributed listing for function {}, {}".format(name, line))
        self._print_src_dists(lines)

        # trace diag info
        print()
        for l in self.diag_info:
            print(l)
        print()

    def _print_src_dists(self, lines):
        filename = self.func_ir.loc.filename
        src_width = max(len(x) for x in lines)

        map_line_to_info = defaultdict(list)  # parfors can alias lines
        for p_id, p_dist in self.parfor_dists.items():
            # TODO: fix parfor locs
            loc = self.parfor_locs[p_id]
            if loc.filename == filename:
                l_no = max(0, loc.line - 1)
                map_line_to_info[l_no].append("#{}: {}".format(p_id, p_dist))

        for arr, a_dist in self.array_dists.items():
            if not arr in self.array_locs:
                continue
            loc = self.array_locs[arr]
            if loc.filename == filename:
                l_no = max(0, loc.line - 1)
                map_line_to_info[l_no].append("{}: {}".format(arr, a_dist))

        width = src_width + 4
        newlines = []
        newlines.append(width * "-" + "| parfor_id/variable: distribution")
        fmt = "{0:{1}}| {2}"
        lstart = max(0, self.func_ir.loc.line - 1)
        for no, line in enumerate(lines, lstart):
            l_info = map_line_to_info[no]
            info_str = ", ".join(l_info)
            stripped = line.strip("\n")
            srclen = len(stripped)
            if l_info:
                l = fmt.format(width * "-", width, info_str)
            else:
                l = fmt.format(width * " ", width, info_str)
            newlines.append(stripped + l[srclen:])
        print("\n".join(newlines))


class DistributedAnalysis:
    """
    Analyzes the program for distributed transformation and assigns distributions to
    distributable containers (e.g. arrays) and parfors.
    """

    def __init__(
        self, func_ir, typemap, calltypes, typingctx, metadata, flags, arr_analysis
    ):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.typingctx = typingctx
        self.metadata = metadata
        self.flags = flags
        self.arr_analysis = arr_analysis
        self.parfor_locs = {}
        self.array_locs = {}
        self.diag_info = []

    def _init_run(self):
        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        self._parallel_accesses = set()
        self._T_arrs = set()
        self.second_pass = False
        self.in_parallel_parfor = -1

    def run(self):
        self._init_run()
        blocks = self.func_ir.blocks
        array_dists = {}
        parfor_dists = {}
        topo_order = find_topo_order(blocks)
        self._run_analysis(self.func_ir.blocks, topo_order, array_dists, parfor_dists)
        self.second_pass = True
        self._run_analysis(self.func_ir.blocks, topo_order, array_dists, parfor_dists)
        # rebalance arrays if necessary
        if auto_rebalance and Distribution.OneD_Var in array_dists.values():
            changed = self._rebalance_arrs(array_dists, parfor_dists)
            if changed:
                return self.run()

        # warn when there is no parallel array or parfor
        # only warn for parfor when there is no parallel array since there could be
        # parallel functionality other than parfors
        if (len(array_dists) > 0 and all(is_REP(d) for d in array_dists.values())) or (
            len(array_dists) == 0
            and len(parfor_dists) > 0
            and all(d == Distribution.REP for d in parfor_dists.values())
        ):
            if bodo.get_rank() == 0:
                warnings.warn(
                    BodoWarning(
                        "No parallelism found for function '{}'. This could be due to "
                        "unsupported usage. See distributed diagnostics for more "
                        "information.".format(self.func_ir.func_id.func_name)
                    )
                )

        self.metadata["distributed_diagnostics"] = DistributedDiagnostics(
            self.parfor_locs,
            self.array_locs,
            array_dists,
            parfor_dists,
            self.diag_info,
            self.func_ir,
        )
        return _dist_analysis_result(array_dists, parfor_dists)

    def _run_analysis(self, blocks, topo_order, array_dists, parfor_dists):
        save_array_dists = {}
        save_parfor_dists = {1: 1}  # dummy value
        # fixed-point iteration
        while array_dists != save_array_dists or parfor_dists != save_parfor_dists:
            save_array_dists = copy.copy(array_dists)
            save_parfor_dists = copy.copy(parfor_dists)
            for label in topo_order:
                equiv_set = self.arr_analysis.get_equiv_set(label)
                self._analyze_block(blocks[label], equiv_set, array_dists, parfor_dists)

    def _analyze_block(self, block, equiv_set, array_dists, parfor_dists):
        for inst in block.body:
            inst_defs = get_stmt_defs(inst)
            for a in inst_defs:
                self.array_locs[a] = inst.loc
            if isinstance(inst, ir.Assign):
                self._analyze_assign(inst, equiv_set, array_dists, parfor_dists)
            elif isinstance(inst, Parfor):
                self._analyze_parfor(inst, array_dists, parfor_dists)
            elif isinstance(inst, (ir.SetItem, ir.StaticSetItem)):
                self._analyze_setitem(inst, equiv_set, array_dists)
            elif isinstance(inst, ir.Print):
                continue
            elif type(inst) in distributed_analysis_extensions:
                # let external calls handle stmt if type matches
                f = distributed_analysis_extensions[type(inst)]
                f(inst, array_dists)
            elif isinstance(inst, ir.Return):
                self._analyze_return(inst.value, array_dists)
            else:
                self._set_REP(inst.list_vars(), array_dists)

    def _analyze_assign(self, inst, equiv_set, array_dists, parfor_dists):
        lhs = inst.target.name
        rhs = inst.value
        lhs_typ = self.typemap[lhs]

        # treat return casts like assignments
        if is_expr(rhs, "cast"):
            rhs = rhs.value

        if isinstance(rhs, ir.Var) and (
            is_distributable_typ(lhs_typ) or is_distributable_tuple_typ(lhs_typ)
        ):
            self._meet_array_dists(lhs, rhs.name, array_dists)
            return
        elif is_array_typ(lhs_typ) and is_expr(rhs, "inplace_binop"):
            # distributions of all 3 variables should meet (lhs, arg1, arg2)
            # XXX: arg1 or arg2 (but not both) can be non-array like scalar
            arg1 = rhs.lhs.name
            arg2 = rhs.rhs.name
            arg1_typ = self.typemap[arg1]
            arg2_typ = self.typemap[arg2]
            if is_distributable_typ(arg1_typ):
                dist = self._meet_array_dists(lhs, arg1, array_dists)
            if is_distributable_typ(arg2_typ):
                dist = self._meet_array_dists(lhs, arg2, array_dists, dist)
            if is_distributable_typ(arg1_typ):
                dist = self._meet_array_dists(lhs, arg1, array_dists, dist)
            if is_distributable_typ(arg2_typ):
                self._meet_array_dists(lhs, arg2, array_dists, dist)
            return
        elif isinstance(rhs, ir.Expr) and rhs.op in ("getitem", "static_getitem"):
            self._analyze_getitem(inst, lhs, rhs, equiv_set, array_dists)
            return
        elif is_expr(rhs, "build_tuple") and is_distributable_tuple_typ(lhs_typ):
            # parallel arrays can be packed and unpacked from tuples
            # e.g. boolean array index in test_getitem_multidim
            l_dist = self._get_var_dist(lhs, array_dists)
            new_dist = []
            for d, v in zip(l_dist, rhs.items):
                # some elements might not be distributable
                if d is None:
                    new_dist.append(None)
                    continue
                new_d = self._min_dist(d, self._get_var_dist(v.name, array_dists))
                self._set_var_dist(v.name, array_dists, new_d)
                new_dist.append(new_d)

            array_dists[lhs] = new_dist
            return
        elif is_expr(rhs, "build_list") and (
            is_distributable_tuple_typ(lhs_typ) or is_distributable_typ(lhs_typ)
        ):
            # dist vars can be in lists
            # meet all distributions
            for v in rhs.items:
                self._meet_array_dists(lhs, v.name, array_dists)
            # second round to propagate info fully
            for v in rhs.items:
                self._meet_array_dists(lhs, v.name, array_dists)
            return
        elif is_expr(rhs, "build_map") and (
            is_distributable_tuple_typ(lhs_typ) or is_distributable_typ(lhs_typ)
        ):
            # dist vars can be in dictionary as values
            # meet all distributions
            for _, v in rhs.items:
                self._meet_array_dists(lhs, v.name, array_dists)
            # second round to propagate info fully
            for _, v in rhs.items:
                self._meet_array_dists(lhs, v.name, array_dists)
            return
        elif is_expr(rhs, "exhaust_iter") and is_distributable_tuple_typ(lhs_typ):
            self._meet_array_dists(lhs, rhs.value.name, array_dists)
        elif is_expr(rhs, "getattr"):
            self._analyze_getattr(lhs, rhs, array_dists)
        elif is_expr(rhs, "call"):
            self._analyze_call(
                lhs, rhs, rhs.func.name, rhs.args, dict(rhs.kws), array_dists
            )
        # handle for A in arr_container: ...
        # A = pair_first(iternext(getiter(arr_container)))
        # TODO: support getitem of container
        elif is_expr(rhs, "pair_first") and is_distributable_typ(lhs_typ):
            arr_container = guard(_get_pair_first_container, self.func_ir, rhs)
            if arr_container is not None:
                self._meet_array_dists(lhs, arr_container.name, array_dists)
                return
        elif isinstance(rhs, ir.Expr) and rhs.op in ("getiter", "iternext"):
            # analyze array container access in pair_first
            return
        elif isinstance(rhs, ir.Arg):
            self._analyze_arg(lhs, rhs, array_dists)
            return
        else:
            self._set_REP(inst.list_vars(), array_dists)

    def _analyze_getattr(self, lhs, rhs, array_dists):
        lhs_typ = self.typemap[lhs]
        rhs_typ = self.typemap[rhs.value.name]
        if rhs.attr == "T" and is_array_typ(lhs_typ):
            # array and its transpose have same distributions
            arr = rhs.value.name
            self._meet_array_dists(lhs, arr, array_dists)
            # keep lhs in table for dot() handling
            self._T_arrs.add(lhs)
            return
        elif isinstance(rhs_typ, DataFrameType) and rhs.attr in (
            "to_csv",
            "to_parquet",
        ):
            return
        # list methods
        elif isinstance(rhs_typ, types.List) and rhs.attr in (
            "append",
            "clear",
            "copy",
            "count",
            "extend",
            "index",
            "insert",
            "pop",
            "remove",
            "reverse",
            "sort",
        ):
            return
        elif rhs.attr in [
            "shape",
            "ndim",
            "size",
            "strides",
            "dtype",
            "itemsize",
            "astype",
            "reshape",
            "ctypes",
            "transpose",
            "tofile",
            "copy",
            "view",
        ]:
            pass  # X.shape doesn't affect X distribution
        elif isinstance(
            rhs_typ, bodo.hiframes.pd_index_ext.RangeIndexType
        ) and rhs.attr in ("_start", "_stop", "_step", "_name"):
            return
        elif (
            isinstance(rhs_typ, MultiIndexType)
            and len(rhs_typ.array_types) > 0
            and rhs.attr == "_data"
        ):
            # output of MultiIndex._data is a tuple, with all arrays having the same
            # distribution as input MultiIndex
            # find min of all array distributions
            l_dist = self._get_var_dist(lhs, array_dists)
            m_dist = self._get_var_dist(rhs.value.name, array_dists)
            new_dist = self._min_dist(l_dist[0], m_dist)
            for d in l_dist:
                new_dist = self._min_dist(new_dist, d)
            self._set_var_dist(lhs, array_dists, new_dist)
            self._set_var_dist(rhs.value.name, array_dists, new_dist)
            return
        elif isinstance(rhs_typ, CategoricalArray) and rhs.attr == "_codes":
            # categorical array and its underlying codes array have same distributions
            arr = rhs.value.name
            self._meet_array_dists(lhs, arr, array_dists)

    def _analyze_parfor(self, parfor, array_dists, parfor_dists):
        if parfor.id not in parfor_dists:
            parfor_dists[parfor.id] = Distribution.OneD
            # save parfor loc for diagnostics
            loc = parfor.loc
            # fix loc using pattern if possible
            # TODO: fix parfor loc in transforms
            for pattern in parfor.patterns:
                if (
                    isinstance(pattern, tuple)
                    and pattern[0] == "prange"
                    and pattern[1] == "internal"
                    and isinstance(pattern[2][1], ir.Loc)
                    and pattern[2][1].filename == self.func_ir.loc.filename
                ):
                    loc = pattern[2][1]
                    break
            self.parfor_locs[parfor.id] = loc

        # analyze init block first to see array definitions
        self._analyze_block(
            parfor.init_block, parfor.equiv_set, array_dists, parfor_dists
        )
        out_dist = Distribution.OneD
        if self.in_parallel_parfor != -1:
            out_dist = Distribution.REP

        parfor_arrs = set()  # arrays this parfor accesses in parallel
        array_accesses = _get_array_accesses(
            parfor.loop_body, self.func_ir, self.typemap
        )
        par_index_var = parfor.loop_nests[0].index_variable.name

        for (arr, index, _) in array_accesses:
            # XXX sometimes copy propagation doesn't work for parfor indices
            # so see if the index has a single variable definition and use it
            # e.g. test_to_numeric
            ind_def = self.func_ir._definitions[index]
            if len(ind_def) == 1 and isinstance(ind_def[0], ir.Var):
                index = ind_def[0].name
            if index == par_index_var:
                parfor_arrs.add(arr)
                self._parallel_accesses.add((arr, index))

            # multi-dim case
            tup_list = guard(find_build_tuple, self.func_ir, index)
            if tup_list is not None:
                index_tuple = [var.name for var in tup_list]
                if index_tuple[0] == par_index_var:
                    parfor_arrs.add(arr)
                    self._parallel_accesses.add((arr, index))
                if par_index_var in index_tuple[1:]:
                    out_dist = Distribution.REP
            # TODO: check for index dependency

        for arr in parfor_arrs:
            if arr in array_dists:
                out_dist = Distribution(min(out_dist.value, array_dists[arr].value))
        parfor_dists[parfor.id] = out_dist
        for arr in parfor_arrs:
            if arr in array_dists:
                array_dists[arr] = out_dist

        # TODO: find prange actually coming from user
        # for pattern in parfor.patterns:
        #     if pattern[0] == 'prange' and not self.in_parallel_parfor:
        #         parfor_dists[parfor.id] = Distribution.OneD

        # run analysis recursively on parfor body
        if self.second_pass and out_dist in [Distribution.OneD, Distribution.OneD_Var]:
            self.in_parallel_parfor = parfor.id
        blocks = wrap_parfor_blocks(parfor)
        for l, b in blocks.items():
            # init_block (label 0) equiv set is parfor.equiv_set in array analysis
            eq_set = parfor.equiv_set if l == 0 else self.arr_analysis.get_equiv_set(l)
            self._analyze_block(b, eq_set, array_dists, parfor_dists)
        unwrap_parfor_blocks(parfor)
        if self.in_parallel_parfor == parfor.id:
            self.in_parallel_parfor = -1
        return

    def _analyze_call(self, lhs, rhs, func_var, args, kws, array_dists):
        """analyze array distributions in function calls
        """
        func_name = ""
        func_mod = ""
        fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
        if fdef is None:
            # check ObjModeLiftedWith, we assume distribution doesn't change
            # blocks of data are passed in, TODO: document
            func_def = guard(get_definition, self.func_ir, rhs.func)
            if isinstance(func_def, ir.Const) and isinstance(
                func_def.value, numba.dispatcher.ObjModeLiftedWith
            ):
                return
            warnings.warn("function call couldn't be found for distributed analysis")
            self._analyze_call_set_REP(lhs, args, array_dists, fdef)
            return
        else:
            func_name, func_mod = fdef

        if is_alloc_callname(func_name, func_mod):
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            size_def = guard(get_definition, self.func_ir, rhs.args[0])
            # local 1D_var if local_alloc_size() is used
            if is_expr(size_def, "call") and guard(
                find_callname, self.func_ir, size_def, self.typemap
            ) == ("local_alloc_size", "bodo.libs.distributed_api"):
                in_arr_name = size_def.args[1].name
                # output array is 1D_Var if input array is distributed
                out_dist = Distribution(
                    min(Distribution.OneD_Var.value, array_dists[in_arr_name].value)
                )
                array_dists[lhs] = out_dist
                # input can become REP
                if out_dist != Distribution.OneD_Var:
                    array_dists[in_arr_name] = out_dist
            return

        # numpy direct functions
        if isinstance(func_mod, str) and func_mod == "numpy":
            self._analyze_call_np(lhs, func_name, args, kws, array_dists)
            return

        # handle array.func calls
        if isinstance(func_mod, ir.Var) and is_array_typ(self.typemap[func_mod.name]):
            self._analyze_call_array(lhs, func_mod, func_name, args, array_dists)
            return

        # handle df.func calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], DataFrameType
        ):
            self._analyze_call_df(lhs, func_mod, func_name, args, array_dists)
            return

        if fdef == ("parallel_print", "bodo"):
            return

        # input of gatherv should not be REP (likely a user mistake),
        # but the output is REP
        if fdef == ("gatherv", "bodo") or fdef == ("allgatherv", "bodo"):
            if is_REP(array_dists[rhs.args[0].name]):
                # TODO: test
                raise BodoWarning("Input to gatherv is not distributed array")
            array_dists[lhs] = Distribution.REP
            return

        if fdef == ("setitem_arr_nan", "bodo.ir.join"):
            return

        # bodo.libs.distributed_api functions
        if isinstance(func_mod, str) and func_mod == "bodo.libs.distributed_api":
            self._analyze_call_bodo_dist(lhs, func_name, args, array_dists)
            return

        # len()
        if func_name == "len" and func_mod in ("__builtin__", "builtins"):
            return

        # handle list.func calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], types.List
        ):
            dtype = self.typemap[func_mod.name].dtype
            if is_distributable_typ(dtype) or is_distributable_tuple_typ(dtype):
                if func_name in ("append", "count", "extend", "index", "remove"):
                    self._meet_array_dists(func_mod.name, rhs.args[0].name, array_dists)
                    return
                if func_name == "insert":
                    self._meet_array_dists(func_mod.name, rhs.args[1].name, array_dists)
                    return
                if func_name in ("copy", "pop"):
                    self._meet_array_dists(lhs, func_mod.name, array_dists)
                    return

        if bodo.config._has_h5py and (
            func_mod == "bodo.io.h5_api"
            and func_name in ("h5read", "h5write", "h5read_filter")
        ):
            return

        if bodo.config._has_h5py and (
            func_mod == "bodo.io.h5_api" and func_name == "get_filter_read_indices"
        ):
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return

        if fdef == ("quantile", "bodo.libs.array_kernels"):
            # quantile doesn't affect input's distribution
            return

        if fdef == ("nunique", "bodo.libs.array_kernels"):
            # nunique doesn't affect input's distribution
            return

        if fdef == ("unique", "bodo.libs.array_kernels"):
            # doesn't affect distribution of input since input can stay 1D
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD_Var

            new_dist = Distribution(
                min(array_dists[lhs].value, array_dists[rhs.args[0].name].value)
            )
            array_dists[lhs] = new_dist
            return

        if fdef == ("array_isin", "bodo.libs.array"):
            # out_arr and in_arr should have the same distribution
            new_dist = self._meet_array_dists(
                rhs.args[0].name, rhs.args[1].name, array_dists
            )
            # values array can be distributed only if input is distributed
            new_dist = Distribution(
                min(new_dist.value, array_dists[rhs.args[2].name].value)
            )
            array_dists[rhs.args[2].name] = new_dist
            return

        if fdef == ("rolling_fixed", "bodo.hiframes.rolling"):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("rolling_variable", "bodo.hiframes.rolling"):
            # lhs, in_arr, on_arr should have the same distribution
            new_dist = self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            new_dist = self._meet_array_dists(
                lhs, rhs.args[1].name, array_dists, new_dist
            )
            array_dists[rhs.args[0].name] = new_dist
            return

        if fdef == ("shift", "bodo.hiframes.rolling"):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("pct_change", "bodo.hiframes.rolling"):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("nlargest", "bodo.libs.array_kernels"):
            # data and index arrays have the same distributions
            self._meet_array_dists(rhs.args[0].name, rhs.args[1].name, array_dists)
            # output of nlargest is REP
            self._set_var_dist(lhs, array_dists, Distribution.REP)
            return

        if fdef == ("set_df_column_with_reflect", "bodo.hiframes.pd_dataframe_ext"):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            self._meet_array_dists(lhs, rhs.args[2].name, array_dists)
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("drop_duplicates", "bodo.libs.array_kernels"):
            lhs_typ = self.typemap[lhs]
            arg_dist = min(a.value for a in array_dists[rhs.args[0].name])
            lhs_dist = arg_dist
            if lhs in array_dists:
                lhs_dist = min(a.value for a in array_dists[lhs][0])
                lhs_dist = min(lhs_dist, array_dists[lhs][1].value)
            new_dist = Distribution(
                min(array_dists[rhs.args[1].name].value, arg_dist, lhs_dist)
            )
            array_dists[lhs] = [
                [new_dist for _ in range(len(lhs_typ.types[0]))],
                new_dist,
            ]
            array_dists[rhs.args[0].name] = [
                new_dist for _ in range(len(array_dists[rhs.args[0].name]))
            ]
            array_dists[rhs.args[1].name] = new_dist
            return

        if fdef == ("duplicated", "bodo.libs.array_kernels"):
            arg_dist = min(a.value for a in array_dists[rhs.args[0].name])
            lhs_dist = arg_dist
            if lhs in array_dists:
                lhs_dist = min(a.value for a in array_dists[lhs])
            new_dist = Distribution(
                min(array_dists[rhs.args[1].name].value, arg_dist, lhs_dist)
            )
            array_dists[lhs] = [new_dist for _ in range(len(self.typemap[lhs]))]
            array_dists[rhs.args[0].name] = [
                new_dist for _ in range(len(array_dists[rhs.args[0].name]))
            ]
            array_dists[rhs.args[1].name] = new_dist
            return

        if fdef == ("nancorr", "bodo.libs.array_kernels"):
            array_dists[lhs] = Distribution.REP
            return

        if fdef == ("median", "bodo.libs.array_kernels"):
            return

        if fdef == ("concat", "bodo.libs.array_kernels"):
            # hiframes concat is similar to np.concatenate
            self._analyze_call_np_concatenate(lhs, args, array_dists)
            return

        if fdef == ("isna", "bodo.libs.array_kernels"):
            return

        if fdef == ("get_str_arr_item_length", "bodo.libs.str_arr_ext"):
            return

        if fdef == ("move_str_arr_payload", "bodo.libs.str_arr_ext"):
            self._meet_array_dists(rhs.args[0].name, rhs.args[1].name, array_dists)
            return

        if fdef == ("get_series_name", "bodo.hiframes.pd_series_ext"):
            return

        if fdef == ("get_index_name", "bodo.hiframes.pd_index_ext"):
            return

        # dummy hiframes functions
        if func_mod == "bodo.hiframes.pd_series_ext" and func_name in (
            "get_series_data",
            "get_series_index",
        ):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if func_mod == "bodo.libs.int_arr_ext" and func_name in (
            "get_int_arr_data",
            "get_int_arr_bitmap",
        ):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if func_mod == "bodo.libs.bool_arr_ext" and func_name in (
            "get_bool_arr_data",
            "get_bool_arr_bitmap",
        ):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("get_bit_bitmap_arr", "bodo.libs.int_arr_ext"):
            return

        if fdef == ("set_bit_to_arr", "bodo.libs.int_arr_ext"):
            return

        # from flat map pattern: pd.Series(list(itertools.chain(*A)))
        if fdef == ("flatten_array", "bodo.utils.conversion"):
            # output of flatten_array is variable-length even if input is 1D
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD_Var
            in_dist = array_dists[rhs.args[0].name]
            out_dist = array_dists[lhs]
            out_dist = Distribution(min(out_dist.value, in_dist.value))
            array_dists[lhs] = out_dist
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                array_dists[rhs.args[0].name] = out_dist
            return

        if fdef == ("str_split", "bodo.libs.list_str_arr_ext"):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if func_mod == "bodo.hiframes.pd_index_ext" and func_name in (
            "init_numeric_index",
            "init_string_index",
            "init_datetime_index",
            "init_timedelta_index",
            "get_index_data",
        ):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        # RangeIndexType is technically a distributable type even though the
        # object doesn't require communication
        if fdef == ("init_range_index", "bodo.hiframes.pd_index_ext"):
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return

        if fdef == ("init_multi_index", "bodo.hiframes.pd_multi_index_ext"):
            # input arrays and output index have the same distribution
            tup_list = guard(find_build_tuple, self.func_ir, rhs.args[0])
            assert tup_list is not None
            for v in tup_list:
                self._meet_array_dists(lhs, v.name, array_dists)
            for v in tup_list:
                self._meet_array_dists(lhs, v.name, array_dists)
            return

        if fdef == ("init_series", "bodo.hiframes.pd_series_ext"):
            # lhs, in_arr, and index should have the same distribution
            new_dist = self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            new_dist = self._meet_array_dists(
                lhs, rhs.args[1].name, array_dists, new_dist
            )
            array_dists[rhs.args[0].name] = new_dist
            return

        if fdef == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
            # lhs, data arrays, and index should have the same distribution
            # data arrays
            data_varname = rhs.args[0].name
            ind_varname = rhs.args[1].name
            new_dist_val = min(a.value for a in array_dists[data_varname])
            if lhs in array_dists:
                new_dist_val = min(new_dist_val, array_dists[lhs].value)
            # handle index
            if self.typemap[ind_varname] != types.none:
                new_dist_val = min(new_dist_val, array_dists[ind_varname].value)
            new_dist = Distribution(new_dist_val)
            self._set_var_dist(data_varname, array_dists, new_dist)
            if self.typemap[ind_varname] != types.none:
                self._set_var_dist(ind_varname, array_dists, new_dist)
            self._set_var_dist(lhs, array_dists, new_dist)
            return

        if fdef == ("init_integer_array", "bodo.libs.int_arr_ext"):
            # lhs, data, and bitmap should have the same distribution
            new_dist = self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            new_dist = self._meet_array_dists(
                lhs, rhs.args[1].name, array_dists, new_dist
            )
            array_dists[rhs.args[0].name] = new_dist
            return

        if fdef == ("init_bool_array", "bodo.libs.bool_arr_ext"):
            # lhs, data, and bitmap should have the same distribution
            new_dist = self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            new_dist = self._meet_array_dists(
                lhs, rhs.args[1].name, array_dists, new_dist
            )
            array_dists[rhs.args[0].name] = new_dist
            return

        if fdef == ("get_dataframe_data", "bodo.hiframes.pd_dataframe_ext"):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("get_dataframe_index", "bodo.hiframes.pd_dataframe_ext"):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("compute_split_view", "bodo.hiframes.split_impl"):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("get_split_view_index", "bodo.hiframes.split_impl"):
            # just used in str.get() implementation for now so we know it is
            # parallel
            # TODO: handle index similar to getitem to support more cases
            return

        if fdef == ("get_split_view_data_ptr", "bodo.hiframes.split_impl"):
            return

        if fdef == ("setitem_str_arr_ptr", "bodo.libs.str_arr_ext"):
            return

        if fdef == ("num_total_chars", "bodo.libs.str_arr_ext"):
            return

        if fdef == (
            "_series_dropna_str_alloc_impl_inner",
            "bodo.hiframes.series_kernels",
        ):
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD_Var
            in_dist = array_dists[rhs.args[0].name]
            out_dist = array_dists[lhs]
            out_dist = Distribution(min(out_dist.value, in_dist.value))
            array_dists[lhs] = out_dist
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                array_dists[rhs.args[0].name] = out_dist
            return

        if fdef == ("copy_non_null_offsets", "bodo.libs.str_arr_ext") or fdef == (
            "copy_data",
            "bodo.libs.str_arr_ext",
        ):
            out_arrname = rhs.args[0].name
            in_arrname = rhs.args[1].name
            self._meet_array_dists(out_arrname, in_arrname, array_dists)
            return

        if fdef == ("str_arr_item_to_numeric", "bodo.libs.str_arr_ext"):
            out_arrname = rhs.args[0].name
            in_arrname = rhs.args[2].name
            self._meet_array_dists(out_arrname, in_arrname, array_dists)
            return

        # np.fromfile()
        if fdef == ("file_read", "bodo.io.np_io"):
            return

        # TODO: make sure assert_equiv is not generated unnecessarily
        # TODO: fix assert_equiv for np.stack from df.value
        if fdef == ("assert_equiv", "numba.array_analysis"):
            return

        # handle calling other Bodo functions that have distributed flags
        func_type = self.typemap[func_var]
        if isinstance(func_type, types.Dispatcher):
            self._handle_dispatcher(func_type.dispatcher, lhs, rhs, array_dists)
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, fdef)

    def _analyze_call_np(self, lhs, func_name, args, kws, array_dists):
        """analyze distributions of numpy functions (np.func_name)
        """

        if func_name == "ascontiguousarray":
            self._meet_array_dists(lhs, args[0].name, array_dists)
            return

        if func_name == "ravel":
            self._meet_array_dists(lhs, args[0].name, array_dists)
            return

        if func_name == "concatenate":
            self._analyze_call_np_concatenate(lhs, args, array_dists)
            return

        if func_name == "array" and is_array_typ(self.typemap[args[0].name]):
            self._meet_array_dists(lhs, args[0].name, array_dists)
            return

        # handle array.sum() with axis
        if func_name == "sum":
            axis_var = get_call_expr_arg("sum", args, kws, 1, "axis", "")
            axis = guard(find_const, self.func_ir, axis_var)
            # sum over the first axis produces REP output
            if axis == 0:
                array_dists[lhs] = Distribution.REP
                return
            # sum over other axis doesn't change distribution
            if axis_var != "" and axis != 0:
                self._meet_array_dists(lhs, args[0].name, array_dists)
                return

        if func_name == "dot":
            self._analyze_call_np_dot(lhs, args, array_dists)
            return

        # used in df.values
        if func_name == "stack":
            seq_info = guard(find_build_sequence, self.func_ir, args[0])
            if seq_info is None:
                self._analyze_call_set_REP(lhs, args, array_dists, "np." + func_name)
                return
            in_arrs, _ = seq_info

            axis = 0
            # TODO: support kws
            # if 'axis' in kws:
            #     axis = find_const(self.func_ir, kws['axis'])
            if len(args) > 1:
                axis = find_const(self.func_ir, args[1])

            # parallel if args are 1D and output is 2D and axis == 1
            if axis is not None and axis == 1 and self.typemap[lhs].ndim == 2:
                for v in in_arrs:
                    self._meet_array_dists(lhs, v.name, array_dists)
                return

        if func_name == "reshape":
            # shape argument can be int or tuple of ints
            shape_typ = self.typemap[args[1].name]
            if isinstance(shape_typ, types.Integer):
                shape_vars = [args[1]]
            else:
                isinstance(shape_typ, types.BaseTuple)
                shape_vars = find_build_tuple(self.func_ir, args[1])
            return self._analyze_call_np_reshape(lhs, args[0], shape_vars, array_dists)

        if func_name in [
            "cumsum",
            "cumprod",
            "cummin",
            "cummax",
            "empty_like",
            "zeros_like",
            "ones_like",
            "full_like",
            "copy",
        ]:
            in_arr = args[0].name
            self._meet_array_dists(lhs, in_arr, array_dists)
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, "np." + func_name)

    def _analyze_call_array(self, lhs, arr, func_name, args, array_dists):
        """analyze distributions of array functions (arr.func_name)
        """
        if func_name == "transpose":
            if len(args) == 0:
                raise ValueError("Transpose with no arguments is not" " supported")
            in_arr_name = arr.name
            arg0 = guard(get_constant, self.func_ir, args[0])
            if isinstance(arg0, tuple):
                arg0 = arg0[0]
            if arg0 != 0:
                raise ValueError(
                    "Transpose with non-zero first argument" " is not supported"
                )
            self._meet_array_dists(lhs, in_arr_name, array_dists)
            return

        if func_name == "reshape":
            # array.reshape supports shape input as single tuple, as well as separate
            # arguments
            shape_vars = args
            arg_typ = self.typemap[args[0].name]
            if isinstance(arg_typ, types.BaseTuple):
                shape_vars = find_build_tuple(self.func_ir, args[0])
            return self._analyze_call_np_reshape(lhs, arr, shape_vars, array_dists)

        if func_name in ("astype", "copy", "view"):
            in_arr_name = arr.name
            self._meet_array_dists(lhs, in_arr_name, array_dists)
            return

        # Array.tofile() is supported for all distributions
        if func_name == "tofile":
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, "array." + func_name)

    def _analyze_call_np_reshape(self, lhs, arr, shape_vars, array_dists):
        """distributed analysis for array.reshape or np.reshape calls
        """
        # REP propagates from input to output and vice versa
        if is_REP(array_dists[arr.name]) or (
            lhs in array_dists and is_REP(array_dists[lhs])
        ):
            self._set_var_dist(lhs, array_dists, Distribution.REP)
            self._set_var_dist(arr.name, array_dists, Distribution.REP)
            return

        # reshape to 1 dimension
        # special case: output is 1D_Var since we just reshape locally without data
        # exchange
        if len(shape_vars) == 1:
            self._set_var_dist(lhs, array_dists, Distribution.OneD_Var)
        # all other cases will have data exchange resulting in 1D distribution
        else:
            self._set_var_dist(lhs, array_dists, Distribution.OneD)

    def _analyze_call_df(self, lhs, arr, func_name, args, array_dists):
        # to_csv() and to_parquet() can be parallelized
        if func_name in {"to_csv", "to_parquet"}:
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, "df." + func_name)

    def _analyze_call_bodo_dist(self, lhs, func_name, args, array_dists):
        """analyze distributions of bodo distributed functions
        (bodo.libs.distributed_api.func_name)
        """

        if func_name == "parallel_print":
            return

        if func_name == "set_arr_local":
            return

        if func_name == "local_alloc_size":
            return

        if func_name == "dist_return":
            arr_name = args[0].name
            arr_typ = self.typemap[arr_name]
            assert is_distributable_typ(arr_typ) or is_distributable_tuple_typ(
                arr_typ
            ), "Variable {} is not distributable since it is of type {}".format(
                arr_name, arr_typ
            )
            assert arr_name in array_dists, "array distribution not found"
            if is_REP(array_dists[arr_name]):
                raise ValueError(
                    "distributed return of array {} not valid"
                    " since it is replicated".format(arr_name)
                )
            return

        if func_name == "threaded_return":
            arr_name = args[0].name
            assert arr_name in array_dists, "array distribution not found"
            if is_REP(array_dists[arr_name]):
                raise ValueError(
                    "threaded return of array {} not valid" " since it is replicated"
                )
            array_dists[arr_name] = Distribution.Thread
            return

        if func_name == "rebalance_array":
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            in_arr = args[0].name
            if array_dists[in_arr] == Distribution.OneD_Var:
                array_dists[lhs] = Distribution.OneD
            else:
                self._meet_array_dists(lhs, in_arr, array_dists)
            return

        # set REP if not found
        self._analyze_call_set_REP(
            lhs, args, array_dists, "bodo.libs.distributed_api." + func_name
        )

    def _handle_dispatcher(self, dispatcher, lhs, rhs, array_dists):
        """handles Bodo function calls that have distributed flags.
        finds if input arguments and return value are marked as distributed and makes
        sure distributions are set properly
        """
        dist_flag_vars = dispatcher.targetoptions.get("distributed", ())
        dist_vars = []
        rep_vars = []

        # folds arguments and finds the ones that are flagged as distributed
        # folding arguments similar to:
        # https://github.com/numba/numba/blob/5f474010f8f50b3cf358125ba279d345ae5914ef/numba/core/dispatcher.py#L70
        def normal_handler(index, param, value):
            if param.name in dist_flag_vars:
                dist_vars.append(value.name)
            else:
                rep_vars.append(value.name)
            return self.typemap[value.name]

        def default_handler(index, param, default):
            return types.Omitted(default)

        def stararg_handler(index, param, values):
            if param.name in dist_flag_vars:
                dist_vars.extend(v.name for v in values)
            else:
                rep_vars.extend(v.name for v in values)
            val_types = tuple(self.typemap[v.name] for v in values)
            return types.StarArgTuple(val_types)

        pysig = dispatcher._compiler.pysig
        arg_types = numba.typing.templates.fold_arguments(
            pysig,
            rhs.args,
            dict(rhs.kws),
            normal_handler,
            default_handler,
            stararg_handler,
        )

        # check return value for distributed flag
        metadata = dispatcher.overloads[arg_types].metadata
        is_return_distributed = metadata.get("is_return_distributed", False)
        if is_return_distributed:
            dist_vars.append(lhs)
            if lhs not in array_dists:
                self._set_var_dist(lhs, array_dists, Distribution.OneD_Var)
        else:
            rep_vars.append(lhs)

        # make sure variables marked distributed are not REP
        # otherwise, leave current distribution in place (1D or 1D_Var)
        for v in dist_vars:
            if is_REP(array_dists[v]):
                raise BodoError(
                    "variable {} is marked as distribtued by {} but not possible to"
                    " distribute in caller function {}".format(
                        v, dispatcher.__name__, self.func_ir.func_id.func_name
                    )
                )

        # set REP vars
        for v in rep_vars:
            self._set_var_dist(v, array_dists, Distribution.REP)

    def _analyze_call_np_concatenate(self, lhs, args, array_dists):
        assert len(args) == 1
        tup_def = guard(get_definition, self.func_ir, args[0])
        assert isinstance(tup_def, ir.Expr) and tup_def.op == "build_tuple"
        in_arrs = tup_def.items
        # input arrays have same distribution
        in_dist = Distribution.OneD
        for v in in_arrs:
            in_dist = Distribution(min(in_dist.value, array_dists[v.name].value))

        # OneD_Var since sum of block sizes might not be exactly 1D
        out_dist = Distribution.OneD_Var
        if lhs in array_dists:
            out_dist = Distribution(min(out_dist.value, array_dists[lhs].value))
        out_dist = Distribution(min(out_dist.value, in_dist.value))
        array_dists[lhs] = out_dist

        # output can cause input REP
        if out_dist != Distribution.OneD_Var:
            in_dist = out_dist
        for v in in_arrs:
            array_dists[v.name] = in_dist
        return

    def _analyze_call_np_dot(self, lhs, args, array_dists):
        arg0 = args[0].name
        arg1 = args[1].name
        ndim0 = self.typemap[arg0].ndim
        ndim1 = self.typemap[arg1].ndim
        dist0 = array_dists[arg0]
        dist1 = array_dists[arg1]
        # Fortran layout is caused by X.T and means transpose
        t0 = arg0 in self._T_arrs
        t1 = arg1 in self._T_arrs
        if ndim0 == 1 and ndim1 == 1:
            # vector dot, both vectors should have same layout
            new_dist = Distribution(
                min(array_dists[arg0].value, array_dists[arg1].value)
            )
            array_dists[arg0] = new_dist
            array_dists[arg1] = new_dist
            return
        if ndim0 == 2 and ndim1 == 1 and not t0:
            # special case were arg1 vector is treated as column vector
            # samples dot weights: np.dot(X,w)
            # w is always REP
            array_dists[arg1] = Distribution.REP
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            # lhs and X have same distribution
            self._meet_array_dists(lhs, arg0, array_dists)
            dprint("dot case 1 Xw:", arg0, arg1)
            return
        if ndim0 == 1 and ndim1 == 2 and not t1:
            # reduction across samples np.dot(Y,X)
            # lhs is always REP
            array_dists[lhs] = Distribution.REP
            # Y and X have same distribution
            self._meet_array_dists(arg0, arg1, array_dists)
            dprint("dot case 2 YX:", arg0, arg1)
            return
        if ndim0 == 2 and ndim1 == 2 and t0 and not t1:
            # reduction across samples np.dot(X.T,Y)
            # lhs is always REP
            array_dists[lhs] = Distribution.REP
            # Y and X have same distribution
            self._meet_array_dists(arg0, arg1, array_dists)
            dprint("dot case 3 XtY:", arg0, arg1)
            return
        if ndim0 == 2 and ndim1 == 2 and not t0 and not t1:
            # samples dot weights: np.dot(X,w)
            # w is always REP
            array_dists[arg1] = Distribution.REP
            self._meet_array_dists(lhs, arg0, array_dists)
            dprint("dot case 4 Xw:", arg0, arg1)
            return

        # set REP if no pattern matched
        self._analyze_call_set_REP(lhs, args, array_dists, "np.dot")

    def _analyze_call_set_REP(self, lhs, args, array_dists, fdef=None):
        arrs = []
        for v in args:
            typ = self.typemap[v.name]
            if is_distributable_typ(typ) or is_distributable_tuple_typ(typ):
                dprint("dist setting call arg REP {} in {}".format(v.name, fdef))
                self._set_var_dist(v.name, array_dists, Distribution.REP)
                arrs.append(v.name)
        typ = self.typemap[lhs]
        if is_distributable_typ(typ) or is_distributable_tuple_typ(typ):
            dprint("dist setting call out REP {} in {}".format(lhs, fdef))
            self._set_var_dist(lhs, array_dists, Distribution.REP)
            arrs.append(lhs)
        # save diagnostic info for faild analysis
        fname = fdef
        if isinstance(fdef, tuple) and len(fdef) == 2:
            name, mod = fdef
            if isinstance(mod, ir.Var):
                mod = str(self.typemap[mod.name])
            fname = mod + "." + name
        if len(arrs) > 0:
            info = (
                "Distributed analysis set {} as replicated due "
                "to call to function '{}' (unsupported function or usage)"
            ).format(", ".join(arrs), fname)
            if info not in self.diag_info:
                self.diag_info.append(info)

    def _analyze_getitem(self, inst, lhs, rhs, equiv_set, array_dists):
        in_typ = self.typemap[rhs.value.name]
        # get index_var without changing IR since we are in analysis
        index_var = get_getsetitem_index_var(rhs, self.typemap, [])
        index_typ = self.typemap[index_var.name]
        lhs_typ = self.typemap[lhs]

        # selecting a value from a distributable tuple does not make it REP
        # nested tuples are also possible
        if (
            isinstance(in_typ, types.BaseTuple)
            and is_distributable_tuple_typ(in_typ)
            and isinstance(index_typ, types.IntegerLiteral)
        ):
            # meet distributions if returned value is distributable
            if is_distributable_typ(lhs_typ) or is_distributable_tuple_typ(lhs_typ):
                # meet distributions
                ind_val = index_typ.literal_value
                tup = rhs.value.name
                if tup not in array_dists:
                    self._set_var_dist(tup, array_dists, Distribution.OneD)
                if lhs not in array_dists:
                    self._set_var_dist(lhs, array_dists, Distribution.OneD)
                new_dist = self._min_dist(array_dists[tup][ind_val], array_dists[lhs])
                array_dists[tup][ind_val] = new_dist
                array_dists[lhs] = new_dist
            return

        # getitem on list/dictionary of distributed values
        if isinstance(in_typ, (types.List, types.DictType)) and (
            is_distributable_typ(lhs_typ) or is_distributable_tuple_typ(lhs_typ)
        ):
            # output and dictionary have the same distribution
            self._meet_array_dists(lhs, rhs.value.name, array_dists)
            return

        # indexing into arrays from this point only, check for array type
        if not is_array_typ(in_typ):
            self._set_REP(inst.list_vars(), array_dists)
            return

        if (rhs.value.name, index_var.name) in self._parallel_accesses:
            # XXX: is this always valid? should be done second pass?
            self._set_REP([inst.target], array_dists)
            return

        # in multi-dimensional case, we only consider first dimension
        # TODO: extend to 2D distribution
        tup_list = guard(find_build_tuple, self.func_ir, index_var)
        if tup_list is not None:
            index_var = tup_list[0]
            # rest of indices should be replicated if array
            other_ind_vars = tup_list[1:]
            self._set_REP(other_ind_vars, array_dists)

        if isinstance(index_var, int):
            self._set_REP(inst.list_vars(), array_dists)
            return
        assert isinstance(index_var, ir.Var)
        index_typ = self.typemap[index_var.name]

        # array selection with boolean index
        if (
            is_np_array_typ(index_typ)
            and index_typ.dtype == types.boolean
            or index_typ == boolean_array
        ):
            # input array and bool index have the same distribution
            new_dist = self._meet_array_dists(
                index_var.name, rhs.value.name, array_dists
            )
            out_dist = Distribution(min(Distribution.OneD_Var.value, new_dist.value))
            array_dists[lhs] = out_dist
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                self._meet_array_dists(
                    index_var.name, rhs.value.name, array_dists, out_dist
                )
            return

        # whole slice or strided slice access
        # for example: A = X[:,5], A = X[::2,5]
        if guard(
            is_whole_slice, self.typemap, self.func_ir, index_var, accept_stride=True
        ) or guard(
            is_slice_equiv_arr,
            inst.target,
            index_var,
            self.func_ir,
            equiv_set,
            accept_stride=True,
        ):
            self._meet_array_dists(lhs, rhs.value.name, array_dists)
            return

        # avoid parallel slice/scalar getitem when inside a parfor
        # examples: test_np_dot, logistic_regression_rand
        if self.in_parallel_parfor != -1:
            self._set_REP(inst.list_vars(), array_dists)
            return

        # output of operations like S.head() is REP since it's a "small" slice
        # input can remain 1D
        if isinstance(index_typ, types.SliceType):
            # TODO: since array and its slice alias, make sure array or its
            # slice or their aliases are not written to
            array_dists[lhs] = Distribution.REP
            return

        # int index of dist array
        if isinstance(index_typ, types.Integer):
            # multi-dim not supported yet, TODO: support
            if is_np_array_typ(in_typ) and in_typ.ndim > 1:
                self._set_REP(inst.list_vars(), array_dists)
            if is_distributable_typ(self.typemap[lhs]):
                array_dists[lhs] = Distribution.REP
            return

        self._set_REP(inst.list_vars(), array_dists)
        return

    def _analyze_setitem(self, inst, equiv_set, array_dists):
        index_var = inst.index_var if is_static_getsetitem(inst) else inst.index
        target_typ = self.typemap[inst.target.name]
        value_typ = self.typemap[inst.value.name]

        if index_var is None:
            self._set_REP(inst.list_vars(), array_dists)
            return

        # setitem on list/dictionary of distributed values
        if isinstance(target_typ, (types.List, types.DictType)) and (
            is_distributable_typ(value_typ) or is_distributable_tuple_typ(value_typ)
        ):
            # output and dictionary have the same distribution
            self._meet_array_dists(inst.target.name, inst.value.name, array_dists)
            return

        if (inst.target.name, index_var.name) in self._parallel_accesses:
            # no parallel to parallel array set (TODO)
            return

        tup_list = guard(find_build_tuple, self.func_ir, index_var)
        if tup_list is not None:
            index_var = tup_list[0]
            # rest of indices should be replicated if array
            self._set_REP(tup_list[1:], array_dists)

        if guard(is_whole_slice, self.typemap, self.func_ir, index_var) or guard(
            is_slice_equiv_arr, inst.target, index_var, self.func_ir, equiv_set
        ):
            # for example: X[:,3] = A
            self._meet_array_dists(inst.target.name, inst.value.name, array_dists)
            return

        self._set_REP([inst.value], array_dists)

    def _analyze_arg(self, lhs, rhs, array_dists):
        if (
            rhs.name in self.metadata["distributed_block"]
            or self.flags.all_args_distributed_block
        ):
            if lhs not in array_dists:
                self._set_var_dist(lhs, array_dists, Distribution.OneD)
        elif (
            rhs.name in self.metadata["distributed"]
            or self.flags.all_args_distributed_varlength
        ):
            if lhs not in array_dists:
                self._set_var_dist(lhs, array_dists, Distribution.OneD_Var)
        elif rhs.name in self.metadata["threaded"]:
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.Thread
        else:
            dprint("replicated input ", rhs.name, lhs)
            typ = self.typemap[lhs]
            if is_distributable_typ(typ) or is_distributable_tuple_typ(typ):
                info = (
                    "Distributed analysis replicated input {0} (variable "
                    "{1}). Set distributed flag for {0} if distributed partitions "
                    "are passed (e.g. @bodo.jit(distributed=['{0}']))."
                ).format(rhs.name, lhs)
                if info not in self.diag_info:
                    self.diag_info.append(info)
                self._set_var_dist(lhs, array_dists, Distribution.REP)

    def _analyze_return(self, var, array_dists):
        if self._is_dist_return_var(var):
            return

        if is_distributable_typ(self.typemap[var.name]):
            info = (
                "Distributed analysis replicated output variable "
                "{}. Set distributed flag for the original variable if distributed "
                "partitions should be returned."
            ).format(var.name)
            if info not in self.diag_info:
                self.diag_info.append(info)
        self._set_REP([var], array_dists)

    def _is_dist_return_var(self, var):
        try:
            vdef = get_definition(self.func_ir, var)
            require(is_expr(vdef, "cast"))
            dcall = get_definition(self.func_ir, vdef.value)
            require(is_expr(dcall, "call"))
            require(
                find_callname(self.func_ir, dcall)
                == ("dist_return", "bodo.libs.distributed_api")
            )
            return True
        except:
            return False

    def _meet_array_dists(self, arr1, arr2, array_dists, top_dist=None):
        typ1 = self.typemap[arr1]
        typ2 = self.typemap[arr2]

        if top_dist is None:
            top_dist = Distribution.OneD
        if arr1 not in array_dists:
            self._set_var_dist(arr1, array_dists, top_dist, False)
        if arr2 not in array_dists:
            self._set_var_dist(arr2, array_dists, top_dist, False)

        new_dist = self._min_dist(array_dists[arr1], array_dists[arr2])
        new_dist = self._min_dist_top(new_dist, top_dist)
        array_dists[arr1] = new_dist
        array_dists[arr2] = new_dist
        return new_dist

    def _set_REP(self, var_list, array_dists):
        for var in var_list:
            varname = var.name
            # Handle SeriesType since it comes from Arg node and it could
            # have user-defined distribution
            typ = self.typemap[varname]
            if is_distributable_typ(typ) or is_distributable_tuple_typ(typ):
                dprint("dist setting REP {}".format(varname))
                self._set_var_dist(varname, array_dists, Distribution.REP)

    def _get_var_dist(self, varname, array_dists):
        if varname not in array_dists:
            self._set_var_dist(varname, array_dists, Distribution.OneD, False)
        return array_dists[varname]

    def _set_var_dist(self, varname, array_dists, dist, check_type=True):
        # some non-distributable types could need to be assigned distribution
        # sometimes, e.g. SeriesILocType. check_type=False handles these cases.
        typ = self.typemap[varname]
        dist = self._get_dist(typ, dist)
        # XXX: Index values can be None so they should have distribution
        # TODO: use proper "FullRangeIndex" type
        if not check_type or (
            is_distributable_typ(typ)
            or is_distributable_tuple_typ(typ)
            or typ is types.none
        ):
            array_dists[varname] = dist

    def _get_dist(self, typ, dist):
        if is_distributable_tuple_typ(typ):
            if isinstance(typ, types.List):
                typ = typ.dtype
            return [
                self._get_dist(t, dist)
                if (is_distributable_typ(t) or is_distributable_tuple_typ(t))
                else None
                for t in typ.types
            ]
        return dist

    def _min_dist(self, dist1, dist2):
        if isinstance(dist1, list):
            assert len(dist1) == len(dist2)
            n = len(dist1)
            return [
                self._min_dist(dist1[i], dist2[i]) if dist1[i] is not None else None
                for i in range(n)
            ]
        return Distribution(min(dist1.value, dist2.value))

    def _min_dist_top(self, dist, top_dist):
        if isinstance(dist, list):
            n = len(dist)
            return [
                self._min_dist_top(dist[i], top_dist) if dist[i] is not None else None
                for i in range(n)
            ]
        return Distribution(min(dist.value, top_dist.value))

    def _rebalance_arrs(self, array_dists, parfor_dists):
        # rebalance an array if it is accessed in a parfor that has output
        # arrays or is in a loop

        # find sequential loop bodies
        cfg = numba.analysis.compute_cfg_from_blocks(self.func_ir.blocks)
        loop_bodies = set()
        for loop in cfg.loops().values():
            loop_bodies |= loop.body

        rebalance_arrs = set()

        for label, block in self.func_ir.blocks.items():
            for inst in block.body:
                # TODO: handle hiframes filter etc.
                if (
                    isinstance(inst, Parfor)
                    and parfor_dists[inst.id] == Distribution.OneD_Var
                ):
                    array_accesses = _get_array_accesses(
                        inst.loop_body, self.func_ir, self.typemap
                    )
                    onedv_arrs = set(
                        arr
                        for (arr, ind, _) in array_accesses
                        if arr in array_dists
                        and array_dists[arr] == Distribution.OneD_Var
                    )
                    if label in loop_bodies or _arrays_written(
                        onedv_arrs, inst.loop_body
                    ):
                        rebalance_arrs |= onedv_arrs

        if len(rebalance_arrs) != 0:
            self._gen_rebalances(rebalance_arrs, self.func_ir.blocks)
            return True

        return False

    def _gen_rebalances(self, rebalance_arrs, blocks):
        #
        for block in blocks.values():
            new_body = []
            for inst in block.body:
                # TODO: handle hiframes filter etc.
                if isinstance(inst, Parfor):
                    self._gen_rebalances(rebalance_arrs, {0: inst.init_block})
                    self._gen_rebalances(rebalance_arrs, inst.loop_body)
                if isinstance(inst, ir.Assign) and inst.target.name in rebalance_arrs:
                    out_arr = inst.target
                    self.func_ir._definitions[out_arr.name].remove(inst.value)
                    # hold inst results in tmp array
                    tmp_arr = ir.Var(
                        out_arr.scope, mk_unique_var("rebalance_tmp"), out_arr.loc
                    )
                    self.typemap[tmp_arr.name] = self.typemap[out_arr.name]
                    inst.target = tmp_arr
                    nodes = [inst]

                    def f(in_arr):  # pragma: no cover
                        out_a = bodo.libs.distributed_api.rebalance_array(in_arr)

                    f_block = compile_to_numba_ir(
                        f,
                        {"bodo": bodo},
                        self.typingctx,
                        (self.typemap[tmp_arr.name],),
                        self.typemap,
                        self.calltypes,
                    ).blocks.popitem()[1]
                    replace_arg_nodes(f_block, [tmp_arr])
                    nodes += f_block.body[:-3]  # remove none return
                    nodes[-1].target = out_arr
                    # update definitions
                    dumm_block = ir.Block(out_arr.scope, out_arr.loc)
                    dumm_block.body = nodes
                    build_definitions({0: dumm_block}, self.func_ir._definitions)
                    new_body += nodes
                else:
                    new_body.append(inst)

            block.body = new_body


def _get_pair_first_container(func_ir, rhs):
    assert isinstance(rhs, ir.Expr) and rhs.op == "pair_first"
    iternext = get_definition(func_ir, rhs.value)
    require(isinstance(iternext, ir.Expr) and iternext.op == "iternext")
    getiter = get_definition(func_ir, iternext.value)
    require(isinstance(iternext, ir.Expr) and getiter.op == "getiter")
    return getiter.value


def _arrays_written(arrs, blocks):
    for block in blocks.values():
        for inst in block.body:
            if isinstance(inst, Parfor) and _arrays_written(arrs, inst.loop_body):
                return True
            if (
                isinstance(inst, (ir.SetItem, ir.StaticSetItem))
                and inst.target.name in arrs
            ):
                return True
    return False


# def get_stencil_accesses(parfor, typemap):
#     # if a parfor has stencil pattern, see which accesses depend on loop index
#     # XXX: assuming loop index is not used for non-stencil arrays
#     # TODO support recursive parfor, multi-D, mutiple body blocks

#     # no access if not stencil
#     is_stencil = False
#     for pattern in parfor.patterns:
#         if pattern[0] == 'stencil':
#             is_stencil = True
#             neighborhood = pattern[1]
#     if not is_stencil:
#         return {}, None

#     par_index_var = parfor.loop_nests[0].index_variable
#     body = parfor.loop_body
#     body_defs = build_definitions(body)

#     stencil_accesses = {}

#     for block in body.values():
#         for stmt in block.body:
#             if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
#                 lhs = stmt.target.name
#                 rhs = stmt.value
#                 if (rhs.op == 'getitem' and is_array_typ(typemap[rhs.value.name])
#                         and vars_dependent(body_defs, rhs.index, par_index_var)):
#                     stencil_accesses[rhs.index.name] = rhs.value.name

#     return stencil_accesses, neighborhood


# def vars_dependent(defs, var1, var2):
#     # see if var1 depends on var2 based on definitions in defs
#     if len(defs[var1.name]) != 1:
#         return False

#     vardef = defs[var1.name][0]
#     if isinstance(vardef, ir.Var) and vardef.name == var2.name:
#         return True
#     if isinstance(vardef, ir.Expr):
#         for invar in vardef.list_vars():
#             if invar.name == var2.name or vars_dependent(defs, invar, var2):
#                 return True
#     return False


# array access code is copied from ir_utils to be able to handle specialized
# array access calls such as get_split_view_index()
# TODO: implement extendable version in ir_utils
def get_parfor_array_accesses(parfor, func_ir, typemap, accesses=None):
    if accesses is None:
        accesses = set()
    blocks = wrap_parfor_blocks(parfor)
    accesses = _get_array_accesses(blocks, func_ir, typemap, accesses)
    unwrap_parfor_blocks(parfor)
    return accesses


array_accesses_extensions = {}
array_accesses_extensions[Parfor] = get_parfor_array_accesses


def _get_array_accesses(blocks, func_ir, typemap, accesses=None):
    """returns a set of arrays accessed and their indices.
    """
    if accesses is None:
        accesses = set()

    for block in blocks.values():
        for inst in block.body:
            if isinstance(inst, ir.SetItem):
                accesses.add((inst.target.name, inst.index.name, False))
            if isinstance(inst, ir.StaticSetItem):
                accesses.add((inst.target.name, inst.index_var.name, False))
            if isinstance(inst, ir.Assign):
                rhs = inst.value
                if isinstance(rhs, ir.Expr) and rhs.op == "getitem":
                    accesses.add((rhs.value.name, rhs.index.name, False))
                if isinstance(rhs, ir.Expr) and rhs.op == "static_getitem":
                    index = rhs.index
                    # slice is unhashable, so just keep the variable
                    if index is None or ir_utils.is_slice_index(index):
                        index = rhs.index_var.name
                    accesses.add((rhs.value.name, index, False))
                if isinstance(rhs, ir.Expr) and rhs.op == "call":
                    fdef = guard(find_callname, func_ir, rhs, typemap)
                    if fdef is not None:
                        if fdef == ("isna", "bodo.libs.array_kernels"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                        if fdef == ("get_split_view_index", "bodo.hiframes.split_impl"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                        if fdef == ("setitem_str_arr_ptr", "bodo.libs.str_arr_ext"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                        if fdef == ("setitem_arr_nan", "bodo.ir.join"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                        if fdef == ("str_arr_item_to_numeric", "bodo.libs.str_arr_ext"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                            accesses.add((rhs.args[2].name, rhs.args[3].name, False))
                        if fdef == ("get_str_arr_item_length", "bodo.libs.str_arr_ext"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                        if fdef == ("get_bit_bitmap_arr", "bodo.libs.int_arr_ext"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, True))
                        if fdef == ("set_bit_to_arr", "bodo.libs.int_arr_ext"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, True))
            for T, f in array_accesses_extensions.items():
                if isinstance(inst, T):
                    f(inst, func_ir, typemap, accesses)
    return accesses


def is_REP(d):
    """Check whether a distribution is REP. Supports regular distributables
    like arrays, as well as tuples with some distributable element
    (distribution is a list object with possible None values)
    """
    if isinstance(d, list):
        return all(a is None or is_REP(a) for a in d)
    return d == Distribution.REP


def dprint(*s):
    if debug_prints():
        print(*s)
