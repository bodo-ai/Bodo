# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
analyzes the IR to decide parallelism of arrays and parfors
for distributed transformation.
"""
import copy
import inspect
import operator
import sys
import warnings
from collections import defaultdict, namedtuple
from enum import Enum

import numba
import numpy as np
from numba.core import ir, ir_utils, types
from numba.core.ir_utils import (
    GuardException,
    build_definitions,
    find_build_sequence,
    find_callname,
    find_const,
    find_topo_order,
    get_definition,
    guard,
    require,
)
from numba.parfors.parfor import (
    Parfor,
    get_parfor_reductions,
    unwrap_parfor_blocks,
    wrap_parfor_blocks,
)

import bodo
import bodo.io
import bodo.io.np_io
from bodo.hiframes.pd_categorical_ext import CategoricalArray
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.distributed_api import Reduce_Type
from bodo.utils.transform import (
    get_call_expr_arg,
    get_const_value,
    get_const_value_inner,
    get_stmt_defs,
)
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    is_heterogeneous_tuple_type,
    is_overload_false,
)
from bodo.utils.utils import (
    debug_prints,
    find_build_tuple,
    get_constant,
    get_getsetitem_index_var,
    is_alloc_callname,
    is_array_typ,
    is_call,
    is_call_assign,
    is_distributable_tuple_typ,
    is_distributable_typ,
    is_expr,
    is_np_array_typ,
    is_slice_equiv_arr,
    is_whole_slice,
)


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


_dist_analysis_result = namedtuple(
    "dist_analysis_result", "array_dists,parfor_dists,concat_reduce_varnames"
)


distributed_analysis_extensions = {}


class DistributedDiagnostics:
    """Gather and print distributed diagnostics information"""

    def __init__(
        self, parfor_locs, array_locs, array_dists, parfor_dists, diag_info, func_ir
    ):
        self.parfor_locs = parfor_locs
        self.array_locs = array_locs
        self.array_dists = array_dists
        self.parfor_dists = parfor_dists
        self.diag_info = diag_info
        self.func_ir = func_ir

    def _print_dists(self, level, metadata):
        print("Data distributions:")
        if len(self.array_dists) > 0:
            arrname_width = max(len(a) for a in self.array_dists.keys())
            arrname_width = max(arrname_width + 3, 20)
            printed_vars = set()
            for arr, dist in self.array_dists.items():
                # only show original user variable names in level=1
                # avoid variable repetition (possible with renaming)
                if level < 2 and arr in metadata["parfors"]["var_rename_map"]:
                    arr = metadata["parfors"]["var_rename_map"][arr]
                if level < 2 and (arr in printed_vars or arr.startswith("$")):
                    continue
                printed_vars.add(arr)
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

    # NOTE: adding metadata as input instead of attribute to avoid circular dependency
    # since DistributedDiagnostics object is inside metadata
    def dump(self, level, metadata):
        name = self.func_ir.func_id.func_qualname
        line = self.func_ir.loc

        print("Distributed diagnostics for function {}, {}\n".format(name, line))
        self._print_dists(level, metadata)

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
        self._print_src_dists(lines, level, metadata)

        # trace diag info
        print()
        for l in self.diag_info:
            print(l)
        print()

    def _print_src_dists(self, lines, level, metadata):
        filename = self.func_ir.loc.filename
        src_width = max(len(x) for x in lines)

        map_line_to_info = defaultdict(list)  # parfors can alias lines
        for p_id, p_dist in self.parfor_dists.items():
            # TODO: fix parfor locs
            loc = self.parfor_locs[p_id]
            if loc.filename == filename:
                l_no = max(0, loc.line - 1)
                map_line_to_info[l_no].append("#{}: {}".format(p_id, p_dist))

        printed_vars = set()
        for arr, a_dist in self.array_dists.items():
            if not arr in self.array_locs:
                continue
            loc = self.array_locs[arr]
            if loc.filename == filename:
                l_no = max(0, loc.line - 1)
                # only show original user variable names in level=1
                # avoid variable repetition (possible with renaming)
                if level < 2 and arr in metadata["parfors"]["var_rename_map"]:
                    arr = metadata["parfors"]["var_rename_map"][arr]
                if level < 2 and (arr in printed_vars or arr.startswith("$")):
                    continue
                printed_vars.add(arr)
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
        # keep track of concat reduce vars to handle in concat analysis and
        # transforms properly
        self._concat_reduce_vars = set()

    def _init_run(self):
        """initialize data structures for distribution analysis"""
        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        self._parallel_accesses = set()
        self._T_arrs = set()
        self.second_pass = False
        self.in_parallel_parfor = -1
        self._concat_reduce_vars = set()

    def run(self):
        """run full distribution analysis pass over the IR.
        It consists of two passes inside to be able to consider nested parfors
        (see "test_kmeans" example).
        """
        self._init_run()
        blocks = self.func_ir.blocks
        array_dists = {}
        parfor_dists = {}
        topo_order = find_topo_order(blocks)
        self._run_analysis(self.func_ir.blocks, topo_order, array_dists, parfor_dists)
        self.second_pass = True
        self._run_analysis(self.func_ir.blocks, topo_order, array_dists, parfor_dists)

        # warn when there is no parallel array or parfor
        # only warn for parfor when there is no parallel array since there could be
        # parallel functionality other than parfors
        # avoid warning if there is no array or parfor since not useful.
        if (
            (array_dists or parfor_dists)
            and all(is_REP(d) for d in array_dists.values())
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
        return _dist_analysis_result(
            array_dists, parfor_dists, self._concat_reduce_vars
        )

    def _run_analysis(self, blocks, topo_order, array_dists, parfor_dists):
        """run a pass of distributed analysis (fixed-point iteration algorithm)"""
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
        """analyze basic blocks (ir.Block)"""
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
            elif isinstance(inst, ir.SetAttr):
                self._analyze_setattr(inst.target, inst.attr, inst.value, array_dists)
            else:
                self._set_REP(
                    inst.list_vars(),
                    array_dists,
                    "unsupported statement in distribution analysis",
                )

    def _analyze_assign(self, inst, equiv_set, array_dists, parfor_dists):
        """analyze assignment nodes (ir.Assign)"""
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
                lhs, rhs, rhs.func.name, rhs.args, dict(rhs.kws), equiv_set, array_dists
            )
        # handle for A in arr_container: ...
        # A = pair_first(iternext(getiter(arr_container)))
        elif is_expr(rhs, "pair_first") and is_distributable_typ(lhs_typ):
            arr_container = guard(_get_pair_first_container, self.func_ir, rhs)
            if arr_container is not None:
                self._meet_array_dists(lhs, arr_container.name, array_dists)
                return
            # this path is not possible since pair_first is only used in the pattern
            # above, unless if variable definitions have some issue
            else:  # pragma: no cover
                self._set_REP(inst.list_vars(), array_dists, "invalid pair_first")
        elif isinstance(rhs, ir.Expr) and rhs.op in ("getiter", "iternext"):
            # analyze array container access in pair_first
            return
        elif isinstance(rhs, ir.Arg):
            self._analyze_arg(lhs, rhs, array_dists)
            return
        else:
            self._set_REP(
                inst.list_vars(),
                array_dists,
                "unsupported expression in distributed analysis",
            )

    def _analyze_getattr(self, lhs, rhs, array_dists):
        """analyze getattr nodes (ir.Expr.getattr)"""
        # NOTE: assuming getattr doesn't change distribution by default, since almost
        # all attribute accesses are benign (e.g. A.shape). Exceptions should be handled
        # here.
        lhs_typ = self.typemap[lhs]
        rhs_typ = self.typemap[rhs.value.name]
        attr = rhs.attr
        if attr == "T" and is_array_typ(lhs_typ):
            # array and its transpose have same distributions
            arr = rhs.value.name
            self._meet_array_dists(lhs, arr, array_dists)
            # keep lhs in table for dot() handling
            self._T_arrs.add(lhs)
            return
        elif (
            isinstance(rhs_typ, MultiIndexType)
            and len(rhs_typ.array_types) > 0
            and attr == "_data"
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
        elif isinstance(rhs_typ, CategoricalArray) and attr == "codes":
            # categorical array and its underlying codes array have same distributions
            arr = rhs.value.name
            self._meet_array_dists(lhs, arr, array_dists)
        # jitclass getattr (e.g. df1 = self.df)
        elif (
            isinstance(rhs_typ, types.ClassInstanceType)
            and attr in rhs_typ.class_type.dist_spec
        ):
            # attribute dist spec should be compatible with distribution of value
            attr_dist = rhs_typ.class_type.dist_spec[attr]
            assert is_distributable_typ(lhs_typ) or is_distributable_tuple_typ(
                lhs_typ
            ), "Variable {} is not distributable since it is of type {} (required for getting distributed class field)".format(
                lhs, lhs_typ
            )
            if lhs not in array_dists:
                array_dists[lhs] = attr_dist
            else:
                # value shouldn't have a more restrictive distribution than dist spec
                # e.g. REP vs OneD
                val_dist = array_dists[lhs]
                if val_dist.value < attr_dist.value:
                    raise BodoError(
                        f"distribution of value is not compatible with the class"
                        f" attribute distribution spec of"
                        f" {rhs_typ.class_type.class_name} in"
                        f" {lhs} = {rhs.value.name}.{attr}"
                    )

    def _analyze_parfor(self, parfor, array_dists, parfor_dists):
        """analyze Parfor nodes for distribution. Parfor and its accessed arrays should
        have the same distribution.
        """
        # get reduction info for parfor if not already available.
        # can compute & save it since parfor doesn't change at this compiler stage
        if "redvars" not in parfor._kws:
            parfor.redvars, parfor.reddict = get_parfor_reductions(
                self.func_ir, parfor, parfor.params, self.calltypes
            )
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
        # nested parfors are replicated
        if self.in_parallel_parfor != -1:
            self._add_diag_info(
                "Parfor {} set to REP since it is inside another distributed Parfor".format(
                    parfor.id
                )
            )
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
            index_name = index
            ind_def = self.func_ir._definitions[index]
            if len(ind_def) == 1 and isinstance(ind_def[0], ir.Var):
                index_name = ind_def[0].name
            if index_name == par_index_var:
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
                    self._add_diag_info(
                        "Parfor {} set to REP since index is used in lower dimensions of array access".format(
                            parfor.id
                        )
                    )
                    out_dist = Distribution.REP
            # TODO: check for index dependency

        for arr in parfor_arrs:
            if arr in array_dists:
                out_dist = Distribution(min(out_dist.value, array_dists[arr].value))

        # analyze reductions like concat that can affect parfor distribution
        out_dist = self._get_parfor_reduce_dists(parfor, out_dist, array_dists)

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

    def _get_parfor_reduce_dists(self, parfor, out_dist, array_dists):
        """analyze parfor reductions like concat that can affect parfor distribution
        TODO: support other similar reductions?
        """

        for reduce_varname, (_init_val, reduce_nodes, _op) in parfor.reddict.items():
            reduce_op = guard(
                get_reduce_op, reduce_varname, reduce_nodes, self.func_ir, self.typemap
            )
            if reduce_op == Reduce_Type.Concat:
                # if output array is replicated, parfor should be replicated too
                if is_REP(array_dists[reduce_varname]):
                    self._add_diag_info(
                        "Parfor {} set to REP since its concat reduction variable is REP".format(
                            parfor.id
                        )
                    )
                    out_dist = Distribution.REP
                else:
                    # concat reduce variables are 1D_Var since each rank can produce
                    # variable amount of data
                    array_dists[reduce_varname] = Distribution.OneD_Var
                # if pafor is replicated, output array is replicated
                if is_REP(out_dist):
                    self._add_diag_info(
                        "Variable '{}' set to REP since it is a concat reduction variable for Parfor {} which is REP".format(
                            self._get_user_varname(reduce_varname), parfor.id
                        )
                    )
                    array_dists[reduce_varname] = Distribution.REP
                # keep track of concat reduce vars to handle in concat analysis and
                # transforms properly
                assert len(self.func_ir._definitions[reduce_varname]) == 2
                conc_varname = self.func_ir._definitions[reduce_varname][1].name
                concat_reduce_vars = self._get_concat_reduce_vars(conc_varname)

                # add concat reduce vars only if it is a parallel reduction
                if not is_REP(out_dist):
                    self._concat_reduce_vars |= concat_reduce_vars
                else:
                    self._concat_reduce_vars -= concat_reduce_vars

        return out_dist

    def _analyze_call(self, lhs, rhs, func_var, args, kws, equiv_set, array_dists):
        """analyze array distributions in function calls"""
        func_name = ""
        func_mod = ""
        fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
        if fdef is None:
            # check ObjModeLiftedWith, we assume out data is distributed (1D_Var)
            func_def = guard(get_definition, self.func_ir, rhs.func)
            if isinstance(func_def, ir.Const) and isinstance(
                func_def.value, numba.core.dispatcher.ObjModeLiftedWith
            ):
                if lhs not in array_dists:
                    self._set_var_dist(lhs, array_dists, Distribution.OneD_Var)
                return
            # some functions like overload_bool_arr_op_nin_1 may generate const ufuncs
            if isinstance(func_def, ir.Const) and isinstance(func_def.value, np.ufunc):
                fdef = (func_def.value.__name__, "numpy")
            else:
                # handle calling other Bodo functions that have distributed flags
                # this code path runs when another jit function is passed as argument
                func_type = self.typemap[func_var]
                if isinstance(func_type, types.Dispatcher) and issubclass(
                    func_type.dispatcher._compiler.pipeline_class,
                    bodo.compiler.BodoCompiler,
                ):
                    self._handle_dispatcher(func_type.dispatcher, lhs, rhs, array_dists)
                    return
                warnings.warn(
                    "function call couldn't be found for distributed analysis"
                )
                self._analyze_call_set_REP(lhs, args, array_dists, fdef)
                return
        else:
            func_name, func_mod = fdef

        if (
            func_name in {"fit", "predict"}
            and "bodo.libs.xgb_ext" in sys.modules
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                (
                    bodo.libs.xgb_ext.BodoXGBClassifierType,
                    bodo.libs.xgb_ext.BodoXGBRegressorType,
                ),
            )
        ):  # pragma: no cover
            if func_name == "fit":
                arg0 = rhs.args[0].name
                arg1 = rhs.args[1].name
                self._meet_array_dists(arg0, arg1, array_dists)
                if array_dists[arg0] == Distribution.REP:
                    raise BodoError(f"Arguments of xgboost.fit are not distributed")
            elif func_name == "predict":
                # match input and output distributions
                self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if (
            func_name in {"fit", "predict", "score"}
            and "bodo.libs.sklearn_ext" in sys.modules
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                (
                    bodo.libs.sklearn_ext.BodoRandomForestClassifierType,
                    bodo.libs.sklearn_ext.BodoSGDClassifierType,
                    bodo.libs.sklearn_ext.BodoSGDRegressorType,
                    bodo.libs.sklearn_ext.BodoLogisticRegressionType,
                    bodo.libs.sklearn_ext.BodoMultinomialNBType,
                    bodo.libs.sklearn_ext.BodoLassoType,
                    bodo.libs.sklearn_ext.BodoLinearRegressionType,
                    bodo.libs.sklearn_ext.BodoRidgeType,
                    bodo.libs.sklearn_ext.BodoLinearSVCType,
                ),
            )
        ):
            if func_name == "fit":
                self._meet_array_dists(rhs.args[0].name, rhs.args[1].name, array_dists)
            elif func_name == "predict":
                # match input and output distributions
                self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            elif func_name == "score":
                self._meet_array_dists(rhs.args[0].name, rhs.args[1].name, array_dists)
            return

        if (
            func_name in {"fit", "predict", "score", "transform"}
            and "bodo.libs.sklearn_ext" in sys.modules
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.libs.sklearn_ext.BodoKMeansClusteringType,
            )
        ):
            self._analyze_call_sklearn_cluster_kmeans(
                lhs, func_name, rhs, kws, array_dists
            )
            return

        if (
            func_name in {"fit", "transform", "inverse_transform", "fit_transform"}
            and "bodo.libs.sklearn_ext" in sys.modules
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                (
                    bodo.libs.sklearn_ext.BodoPreprocessingStandardScalerType,
                    bodo.libs.sklearn_ext.BodoPreprocessingMinMaxScalerType,
                    bodo.libs.sklearn_ext.BodoPreprocessingLabelEncoderType,
                ),
            )
        ):
            self._analyze_call_sklearn_preprocessing_scalers(
                lhs, func_name, rhs, kws, array_dists
            )
            return

        if (
            func_name in {"fit_transform"}
            and "bodo.libs.sklearn_ext" in sys.modules
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                (bodo.libs.sklearn_ext.BodoFExtractHashingVectorizerType,),
            )
        ):
            if func_name == "fit_transform":
                # match input and output distributions (y is ignored)
                self._meet_array_dists(lhs, rhs.args[0].name, array_dists)

            return

        if func_mod == "sklearn.metrics._classification":
            if func_name in {"precision_score", "recall_score", "f1_score"}:
                # output is always replicated, and the output can be an array
                # if average=None so we have to set it
                # TODO this shouldn't be done if output is float?
                self._set_REP(lhs, array_dists, "output of {} is REP".format(func_name))
                dist_arg0 = is_distributable_typ(self.typemap[rhs.args[0].name])
                dist_arg1 = is_distributable_typ(self.typemap[rhs.args[1].name])
                if dist_arg0 and dist_arg1:
                    self._meet_array_dists(
                        rhs.args[0].name, rhs.args[1].name, array_dists
                    )
                elif not dist_arg0 and dist_arg1:
                    self._set_REP(
                        rhs.args[1].name,
                        array_dists,
                        "first input of {} is non-distributable".format(func_name),
                    )
                elif not dist_arg1 and dist_arg0:
                    self._set_REP(
                        rhs.args[0].name,
                        array_dists,
                        "second input of {} is non-distributable".format(func_name),
                    )

            if func_name == "accuracy_score":
                self._analyze_sklearn_score_err_ytrue_ypred_optional_sample_weight(
                    lhs, func_name, rhs, kws, array_dists
                )

            return

        if func_mod == "sklearn.metrics._regression":

            if func_name in {"mean_squared_error", "mean_absolute_error", "r2_score"}:

                self._set_REP(lhs, array_dists, "output of {} is REP".format(func_name))
                self._analyze_sklearn_score_err_ytrue_ypred_optional_sample_weight(
                    lhs, func_name, rhs, kws, array_dists
                )

            return
        if func_mod == "sklearn.model_selection._split":
            if func_name == "train_test_split":
                arg0 = rhs.args[0].name
                if lhs not in array_dists:
                    self._set_var_dist(lhs, array_dists, Distribution.OneD, True)

                min_dist = self._min_dist(array_dists[lhs][0], array_dists[lhs][1])
                min_dist = self._min_dist(min_dist, array_dists[arg0])
                if self.typemap[rhs.args[1].name] != types.none:
                    arg1 = rhs.args[1].name
                    min_dist = self._min_dist(min_dist, array_dists[arg1])
                    min_dist = self._min_dist(min_dist, array_dists[lhs][2])
                    min_dist = self._min_dist(min_dist, array_dists[lhs][3])
                    array_dists[arg1] = min_dist

                self._set_var_dist(lhs, array_dists, min_dist)
                array_dists[arg0] = min_dist
            return

        if fdef == ("prepare_data", "bodo.dl"):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            if array_dists[rhs.args[0].name] == Distribution.REP:
                # TODO: more informative error and suggestion
                raise BodoError(f"Argument of bodo.dl.prepare_data is not distributed")
            return

        if fdef == ("datetime_date_arr_to_dt64_arr", "bodo.hiframes.pd_timestamp_ext"):
            # LHS should match RHS
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if is_alloc_callname(func_name, func_mod):
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            self._alloc_call_size_equiv(lhs, rhs.args[0], equiv_set, array_dists)
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

        # cummin/cummax (absent from numpy)
        if func_mod == "bodo.libs.array_kernels" and func_name in {"cummin", "cummax"}:
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
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

        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], SeriesType
        ):
            self._analyze_call_series(lhs, func_mod, func_name, args, array_dists)
            return

        if fdef == ("parallel_print", "bodo"):
            return

        # input of gatherv should not be REP (likely a user mistake),
        # but the output is REP
        if fdef == ("gatherv", "bodo") or fdef == ("allgatherv", "bodo"):
            arg_no = 2 if fdef[0] == "gatherv" else 1
            warn_flag = get_call_expr_arg(
                fdef[0], rhs.args, kws, arg_no, "warn_if_rep", True
            )
            if isinstance(warn_flag, ir.Var):
                # warn if flag is not constant False. Otherwise just raise warning (not
                # an error if flag is not const since not critical)
                warn_flag = not is_overload_false(self.typemap[warn_flag.name])
            if warn_flag and is_REP(array_dists[rhs.args[0].name]):
                # TODO: test
                warnings.warn(BodoWarning("Input to gatherv is not distributed array"))
            self._set_REP(lhs, array_dists, "output of gatherv() is replicated")
            return

        if fdef == ("scatterv", "bodo"):
            # output of scatterv is 1D
            if lhs not in array_dists:
                self._set_var_dist(lhs, array_dists, Distribution.OneD)
            elif is_REP(array_dists[lhs]):
                raise BodoError("Output of scatterv should be a distributed array")

            # input of scatterv should be replicated
            self._set_REP(
                rhs.args[0].name, array_dists, "input of scatterv() is replicated"
            )
            return

        if fdef == ("setna", "bodo.libs.array_kernels"):
            return

        if (
            isinstance(func_mod, str) and func_mod == "bodo"
        ) and func_name == "rebalance":
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if (
            isinstance(func_mod, str) and func_mod == "bodo"
        ) and func_name == "random_shuffle":
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
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

        if fdef == ("cat_replace", "bodo.hiframes.pd_categorical_ext"):
            # LHS should match RHS
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
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
            # index array is passed for apply(raw=False) case
            if self.typemap[rhs.args[1].name] != types.none:
                self._meet_array_dists(lhs, rhs.args[1].name, array_dists)
                self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("rolling_variable", "bodo.hiframes.rolling"):
            # lhs, in_arr, on_arr should have the same distribution
            new_dist = self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            new_dist = self._meet_array_dists(
                lhs, rhs.args[1].name, array_dists, new_dist
            )
            # index array is passed for apply(raw=False) case
            if self.typemap[rhs.args[2].name] != types.none:
                new_dist = self._meet_array_dists(
                    lhs, rhs.args[2].name, array_dists, new_dist
                )
            array_dists[rhs.args[0].name] = new_dist
            array_dists[rhs.args[1].name] = new_dist
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
            self._set_REP(lhs, array_dists, "output of nlargest is REP")
            return

        if fdef in (
            ("set_df_column_with_reflect", "bodo.hiframes.pd_dataframe_ext"),
            ("set_dataframe_data", "bodo.hiframes.pd_dataframe_ext"),
        ):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            self._meet_array_dists(lhs, rhs.args[2].name, array_dists)
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("sample_table_operation", "bodo.libs.array_kernels"):
            in_dist = Distribution(
                min(
                    min(a.value for a in array_dists[rhs.args[0].name]),
                    array_dists[rhs.args[1].name].value,
                )
            )
            self._set_var_dist(rhs.args[0].name, array_dists, in_dist)
            self._set_var_dist(rhs.args[1].name, array_dists, in_dist)
            self._set_REP(lhs, array_dists, "output of sample is REP")
            return

        if fdef == (
            "pandas_string_array_to_datetime",
            "bodo.hiframes.pd_timestamp_ext",
        ):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == (
            "pandas_string_array_to_timedelta",
            "bodo.hiframes.pd_timestamp_ext",
        ):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("nonzero", "bodo.libs.array_kernels"):
            # output of nonzero is variable-length even if input is 1D
            if lhs not in array_dists:
                self._set_var_dist(lhs, array_dists, Distribution.OneD_Var)

            # arg0 is an array
            in_dist = Distribution(array_dists[rhs.args[0].name].value)
            # return is a tuple(array,)
            out_dist = Distribution(
                min(
                    array_dists[lhs][0].value,
                    in_dist.value,
                )
            )
            self._set_var_dist(lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            self._set_var_dist(rhs.args[0].name, array_dists, in_dist)
            return

        if fdef == ("repeat_kernel", "bodo.libs.array_kernels"):
            # output of repeat_kernel is variable-length even if input is 1D
            # because of the boundary case
            # ex repeat(A, 2) where len(A) = 9 -> (10, 8)
            if lhs not in array_dists:
                self._set_var_dist(lhs, array_dists, Distribution.OneD_Var)

            # arg0 is an array
            in_dist = Distribution(array_dists[rhs.args[0].name].value)
            # arg1 could be an array
            if is_array_typ(self.typemap[rhs.args[1].name]):
                in_dist = self._meet_array_dists(
                    rhs.args[0].name, rhs.args[1].name, array_dists
                )
            # return is an array
            out_dist = Distribution(
                min(
                    array_dists[lhs].value,
                    in_dist.value,
                )
            )
            self._set_var_dist(lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            self._set_var_dist(rhs.args[0].name, array_dists, in_dist)
            if is_array_typ(self.typemap[rhs.args[1].name]):
                self._set_var_dist(rhs.args[1].name, array_dists, in_dist)
            return

        if fdef == ("drop_duplicates", "bodo.libs.array_kernels"):
            # output of drop_duplicates is variable-length even if input is 1D
            if lhs not in array_dists:
                self._set_var_dist(lhs, array_dists, Distribution.OneD_Var)

            # arg0 is a tuple of arrays, arg1 is an array
            in_dist = Distribution(
                min(
                    min(a.value for a in array_dists[rhs.args[0].name]),
                    array_dists[rhs.args[1].name].value,
                )
            )
            # return is a tuple(tuple(arrays), array)
            out_dist = Distribution(
                min(
                    min(a.value for a in array_dists[lhs][0]),
                    array_dists[lhs][1].value,
                    in_dist.value,
                )
            )
            self._set_var_dist(lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            self._set_var_dist(rhs.args[0].name, array_dists, in_dist)
            self._set_var_dist(rhs.args[1].name, array_dists, in_dist)
            return

        if fdef == ("duplicated", "bodo.libs.array_kernels"):
            # output of duplicated is variable-length even if input is 1D
            if lhs not in array_dists:
                self._set_var_dist(lhs, array_dists, Distribution.OneD_Var)

            # arg0 is a tuple of arrays, arg1 is an array
            in_dist = Distribution(
                min(
                    min(a.value for a in array_dists[rhs.args[0].name]),
                    array_dists[rhs.args[1].name].value,
                )
            )
            # return is a tuple(array, array)
            out_dist = Distribution(
                min(
                    array_dists[lhs][0].value,
                    array_dists[lhs][1].value,
                    in_dist.value,
                )
            )
            self._set_var_dist(lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            self._set_var_dist(rhs.args[0].name, array_dists, in_dist)
            self._set_var_dist(rhs.args[1].name, array_dists, in_dist)
            return

        if fdef == ("dropna", "bodo.libs.array_kernels"):
            # output of dropna is variable-length even if input is 1D
            if lhs not in array_dists:
                self._set_var_dist(lhs, array_dists, Distribution.OneD_Var)

            in_dist = Distribution(min(a.value for a in array_dists[rhs.args[0].name]))
            out_dist = Distribution(min(a.value for a in array_dists[lhs]))
            out_dist = Distribution(min(out_dist.value, in_dist.value))
            self._set_var_dist(lhs, array_dists, out_dist)
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist
            self._set_var_dist(rhs.args[0].name, array_dists, in_dist)
            return

        if fdef == ("get", "bodo.libs.array_kernels"):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("nancorr", "bodo.libs.array_kernels"):
            self._set_REP(lhs, array_dists, "output of nancorr is REP")
            return

        if fdef == ("series_monotonicity", "bodo.libs.array_kernels"):
            return

        if fdef == ("autocorr", "bodo.libs.array_kernels"):
            return

        if fdef == ("median", "bodo.libs.array_kernels"):
            return

        if fdef == ("concat", "bodo.libs.array_kernels"):
            # hiframes concat is similar to np.concatenate
            self._analyze_call_concat(lhs, args, array_dists)
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
            # NOTE: constant sizes Series/Index is not distributed
            if _is_tuple_like_type(self.typemap[lhs]):
                self._analyze_call_set_REP(lhs, args, array_dists, fdef)
                return

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

        if func_mod == "bodo.hiframes.pd_categorical_ext" and func_name in (
            "get_categorical_arr_codes",
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

        # explode(): both args have same dist, output is a tuple of 1D_Var arrays
        if fdef == ("explode", "bodo.libs.array_kernels"):
            # output of explode is variable-length even if input is 1D
            if lhs not in array_dists:
                self._set_var_dist(lhs, array_dists, Distribution.OneD_Var)

            in_dist = self._meet_array_dists(
                rhs.args[1].name, rhs.args[0].name, array_dists
            )
            out_dist = Distribution(
                min(array_dists[lhs][0].value, array_dists[lhs][1].value, in_dist.value)
            )
            self._set_var_dist(lhs, array_dists, out_dist)
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            array_dists[rhs.args[0].name] = in_dist
            array_dists[rhs.args[1].name] = in_dist
            return

        if fdef == ("str_split", "bodo.libs.str_ext"):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("explode_str_split", "bodo.libs.array_kernels"):
            # output of explode is variable-length even if input is 1D
            if lhs not in array_dists:
                self._set_var_dist(lhs, array_dists, Distribution.OneD_Var)

            # Input array and index array (args 0 and 3) need to be checked
            in_dist = self._meet_array_dists(
                rhs.args[3].name, rhs.args[0].name, array_dists
            )

            out_dist = Distribution(
                min(array_dists[lhs][0].value, array_dists[lhs][1].value, in_dist.value)
            )
            self._set_var_dist(lhs, array_dists, out_dist)
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            array_dists[rhs.args[0].name] = in_dist
            array_dists[rhs.args[3].name] = in_dist
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
            # NOTE: constant sizes Series/Index is not distributed
            if _is_tuple_like_type(self.typemap[rhs.args[0].name]):
                self._analyze_call_set_REP(lhs, args, array_dists, fdef)
                return

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

            # empty dataframe case, Index and df have same distribution
            if len(self.typemap[data_varname].types) == 0:
                self._meet_array_dists(lhs, ind_varname, array_dists)
                return

            new_dist_val = min(a.value for a in array_dists[data_varname])
            if lhs in array_dists:
                new_dist_val = min(new_dist_val, array_dists[lhs].value)
            # handle index
            new_dist_val = min(new_dist_val, array_dists[ind_varname].value)
            new_dist = Distribution(new_dist_val)
            self._set_var_dist(data_varname, array_dists, new_dist)
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

        if fdef == ("init_categorical_array", "bodo.hiframes.pd_categorical_ext"):
            # lhs and codes should have the same distribution
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
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

        if fdef == ("inplace_eq", "bodo.libs.str_arr_ext"):
            return

        if fdef == ("str_arr_setitem_int_to_str", "bodo.libs.str_arr_ext"):
            return

        if fdef == ("str_arr_setitem_NA_str", "bodo.libs.str_arr_ext"):
            return

        if fdef == ("str_arr_set_not_na", "bodo.libs.str_arr_ext"):
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

        # str_arr_from_sequence() applies to lists/tuples so output is REP
        # e.g. column names in df.mean()
        if fdef == ("str_arr_from_sequence", "bodo.libs.str_arr_ext"):
            self._set_REP(lhs, array_dists, "output of str_arr_from_sequence is REP")
            return

        # TODO: make sure assert_equiv is not generated unnecessarily
        # TODO: fix assert_equiv for np.stack from df.value
        if fdef == ("assert_equiv", "numba.parfors.array_analysis"):
            return

        if fdef == ("_bodo_groupby_apply_impl", ""):
            # output is variable-length even if input is 1D
            if lhs not in array_dists:
                self._set_var_dist(lhs, array_dists, Distribution.OneD_Var)

            # arg0 is a tuple of arrays, arg1 is a dataframe
            in_dist = Distribution(
                min(
                    min(a.value for a in array_dists[rhs.args[0].name]),
                    array_dists[rhs.args[1].name].value,
                )
            )
            out_dist = Distribution(
                min(
                    array_dists[lhs].value,
                    in_dist.value,
                )
            )
            self._set_var_dist(lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            self._set_var_dist(rhs.args[0].name, array_dists, in_dist)
            self._set_var_dist(rhs.args[1].name, array_dists, in_dist)
            self._set_REP(
                args[2:],
                array_dists,
                "extra argument in groupby.apply()",
            )
            return

        # handle calling other Bodo functions that have distributed flags
        func_type = self.typemap[func_var]
        if isinstance(func_type, types.Dispatcher) and issubclass(
            func_type.dispatcher._compiler.pipeline_class, bodo.compiler.BodoCompiler
        ):
            self._handle_dispatcher(func_type.dispatcher, lhs, rhs, array_dists)
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, fdef)

    def _analyze_call_sklearn_preprocessing_scalers(
        self,
        lhs,
        func_name,
        rhs,
        kws,
        array_dists,
    ):
        """
        Analyze distribution of sklearn.preprocessing.StandardScaler, sklearn.preprocessing.MinMaxScaler, and
        sklearn.preprocessing.LabelEncoder functions.
        Only need to handle fit_transform, transform and inverse_transform. fit is handled automatically.
        """

        if func_name in {"transform", "inverse_transform", "fit_transform"}:
            # match input (X) and output (X_new) distributions
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)

        return

    def _analyze_call_sklearn_cluster_kmeans(
        self, lhs, func_name, rhs, kws, array_dists
    ):
        """analyze distribution of sklearn cluster kmeans
        functions (sklearn.cluster.kmeans.func_name"""
        if func_name == "fit":
            # match dist of X and sample_weight (if provided)
            X_arg_name = rhs.args[0].name
            if len(rhs.args) >= 3:
                sample_weight_arg_name = rhs.args[2].name
            elif "sample_weight" in kws:
                sample_weight_arg_name = kws["sample_weight"].name
            else:
                sample_weight_arg_name = None

            if sample_weight_arg_name:
                self._meet_array_dists(X_arg_name, sample_weight_arg_name, array_dists)

        elif func_name == "predict":
            # match dist of X and sample_weight (if provided)
            X_arg_name = rhs.args[0].name
            if len(rhs.args) >= 2:
                sample_weight_arg_name = rhs.args[1].name
            elif "sample_weight" in kws:
                sample_weight_arg_name = kws["sample_weight"].name
            else:
                sample_weight_arg_name = None
            if sample_weight_arg_name:
                self._meet_array_dists(X_arg_name, sample_weight_arg_name, array_dists)

            # match input and output distributions
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)

        elif func_name == "score":
            # match dist of X and sample_weight (if provided)
            X_arg_name = rhs.args[0].name
            if len(rhs.args) >= 3:
                sample_weight_arg_name = rhs.args[2].name
            elif "sample_weight" in kws:
                sample_weight_arg_name = kws["sample_weight"].name
            else:
                sample_weight_arg_name = None
            if sample_weight_arg_name:
                self._meet_array_dists(X_arg_name, sample_weight_arg_name, array_dists)

        elif func_name == "transform":
            # match input (X) and output (X_new) distributions
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)

        return

    def _analyze_sklearn_score_err_ytrue_ypred_optional_sample_weight(
        self, lhs, func_name, rhs, kws, array_dists
    ):
        """
        Analyze for sklearn functions like accuracy_score and mean_squared_error.
        In these we have a y_true, y_pred and optionally a sample_weight.
        Distribution of all 3 should match.
        """

        # sample_weight is an optional kw arg, so check if it is provided.
        # if it is provided and it is not none (because if it is none it's
        # as good as not being provided), then get the "name"
        sample_weight_arg_name = None
        if "sample_weight" in kws and (
            self.typemap[kws["sample_weight"].name] != types.none
        ):
            sample_weight_arg_name = kws["sample_weight"].name

        # check if all 3 (y_true, y_pred, sample_weight) are distributable types
        dist_y_true = is_distributable_typ(self.typemap[rhs.args[0].name])
        dist_y_pred = is_distributable_typ(self.typemap[rhs.args[1].name])
        # if sample_weight is not provided, we can act as if it is distributable
        dist_sample_weight = (
            is_distributable_typ(self.typemap[sample_weight_arg_name])
            if sample_weight_arg_name
            else True
        )

        # if any of the 3 are not distributable, set top_dist to REP
        top_dist = Distribution.OneD
        if not (dist_y_true and dist_y_pred and dist_sample_weight):
            top_dist = Distribution.REP

        # Match distribution of y_true and y_pred, with top dist as
        # computed above, i.e. set to REP if any of the types
        # are not distributable
        self._meet_array_dists(
            rhs.args[0].name, rhs.args[1].name, array_dists, top_dist=top_dist
        )
        if sample_weight_arg_name:
            # Match distribution of y_true and sample_weight
            self._meet_array_dists(
                rhs.args[0].name,
                sample_weight_arg_name,
                array_dists,
                top_dist=top_dist,
            )
            # Match distribution of y_pred and sample_weight
            self._meet_array_dists(
                rhs.args[1].name,
                sample_weight_arg_name,
                array_dists,
                top_dist=top_dist,
            )
        return

    def _analyze_call_np(self, lhs, func_name, args, kws, array_dists):
        """analyze distributions of numpy functions (np.func_name)"""
        # TODO: handle kw args properly
        if func_name == "ascontiguousarray":
            self._meet_array_dists(lhs, args[0].name, array_dists)
            return

        if func_name == "ravel":
            self._meet_array_dists(lhs, args[0].name, array_dists)
            return

        if func_name == "digitize":
            in_arr = get_call_expr_arg("digitize", args, kws, 0, "x")
            bins = get_call_expr_arg("digitize", args, kws, 1, "bins")
            self._meet_array_dists(lhs, in_arr.name, array_dists)
            self._set_REP(
                bins.name, array_dists, "'bins' argument of 'digitize' is REP"
            )
            return

        if func_name == "concatenate":
            # get axis argument
            axis_var = get_call_expr_arg("concatenate", args, kws, 1, "axis", "")
            axis = 0
            if axis_var != "":
                msg = "np.concatenate(): 'axis' should be constant"
                axis = get_const_value(axis_var, self.func_ir, msg)
            self._analyze_call_concat(lhs, args, array_dists, axis)
            return

        if func_name == "array":
            arg = get_call_expr_arg("array", args, kws, 0, "object")
            # np.array of another array can be distributed, but not list/tuple
            # NOTE: not supported by Numba yet
            if is_array_typ(self.typemap[arg.name]):  # pragma: no cover
                self._meet_array_dists(lhs, arg.name, array_dists)
            else:
                self._set_REP(
                    lhs, array_dists, "output of np.array() call on non-array is REP"
                )
            return

        if func_name == "asarray":
            arg = get_call_expr_arg("asarray", args, kws, 0, "a")
            # np.asarray of another array can be distributed, but not list/tuple
            if is_array_typ(self.typemap[args[0].name]):
                self._meet_array_dists(lhs, args[0].name, array_dists)
            else:
                self._set_REP(
                    lhs, array_dists, "output of np.asarray() call on non-array is REP"
                )
            return

        # handle array.sum() with axis
        if func_name == "sum":
            axis_var = get_call_expr_arg("sum", args, kws, 1, "axis", "")
            axis = guard(find_const, self.func_ir, axis_var)
            # sum over the first axis produces REP output
            if axis == 0:
                self._set_REP(
                    lhs, array_dists, "sum over the first axis produces REP output"
                )
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
            arr_var = get_call_expr_arg("np.reshape", args, kws, 0, "a")
            shape_var = get_call_expr_arg("np.reshape", args, kws, 1, "newshape")
            shape_typ = self.typemap[shape_var.name]
            if isinstance(shape_typ, types.Integer):
                shape_vars = [shape_var]
            else:
                assert isinstance(
                    shape_typ, types.BaseTuple
                ), "np.reshape(): invalid shape argument"
                shape_vars = find_build_tuple(self.func_ir, shape_var)
            return self._analyze_call_np_reshape(lhs, arr_var, shape_vars, array_dists)

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

    def _alloc_call_size_equiv(self, lhs, size_var, equiv_set, array_dists):
        """match distribution of output variable 'lhs' of allocation with distributions
        of equivalent arrays (as found by allocation size 'size_var').
        See test_1D_Var_alloc4.
        """
        size_def = guard(get_definition, self.func_ir, size_var)
        # find trivial calc_nitems(0, n, 1) call and use n instead
        if (
            guard(find_callname, self.func_ir, size_def)
            == ("calc_nitems", "bodo.libs.array_kernels")
            and guard(find_const, self.func_ir, size_def.args[0]) == 0
            and guard(find_const, self.func_ir, size_def.args[2]) == 1
        ):  # pragma: no cover
            # TODO: unittest for this case
            size_var = size_def.args[1]
            size_def = guard(get_definition, self.func_ir, size_var)

        # find calc_nitems(r._start, r._stop, r._step) for RangeIndex and match dists
        # see index test_1D_Var_alloc4 for example
        r = guard(self._get_calc_n_items_range_index, size_def)
        if r is not None:
            self._meet_array_dists(lhs, r.name, array_dists)

        # all arrays with equivalent size should have same distribution
        var_set = equiv_set.get_equiv_set(size_var)
        if not var_set:
            return

        for v in var_set:
            # array analysis adds "#0" to array name to designate 1st dimension
            if isinstance(v, str) and "#0" in v:
                arr_name = v.split("#")[0]
                if is_distributable_typ(self.typemap[arr_name]):
                    self._meet_array_dists(lhs, arr_name, array_dists)

    def _analyze_call_array(self, lhs, arr, func_name, args, array_dists):
        """analyze distributions of array functions (arr.func_name)"""
        if func_name == "transpose":
            if len(args) == 0:
                raise BodoError("Transpose with no arguments is not" " supported")
            in_arr_name = arr.name
            arg0 = guard(get_constant, self.func_ir, args[0])
            if isinstance(arg0, tuple):
                arg0 = arg0[0]
            if arg0 != 0:
                raise BodoError(
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
        """distributed analysis for array.reshape or np.reshape calls"""
        # REP propagates from input to output and vice versa
        if is_REP(array_dists[arr.name]) or (
            lhs in array_dists and is_REP(array_dists[lhs])
        ):
            self._set_REP(lhs, array_dists, "np.reshape() input is REP")
            self._set_REP(arr.name, array_dists, "np.reshape() output is REP")
            return

        # optimization: no need to distribute if 1-dim array is reshaped to
        # 2-dim with same length (just added a new dimension)
        if (
            self.typemap[arr.name].ndim == 1
            and len(shape_vars) == 2
            and guard(
                get_const_value_inner, self.func_ir, shape_vars[1], typemap=self.typemap
            )
            == 1
        ):
            self._meet_array_dists(lhs, arr.name, array_dists)
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
        if func_name in {"to_csv", "to_parquet", "to_sql"}:
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, "df." + func_name)

    def _analyze_call_series(self, lhs, arr, func_name, args, array_dists):
        if func_name in {"to_csv"}:
            return

        self._analyze_call_set_REP(lhs, args, array_dists, "series." + func_name)

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
                raise BodoError(
                    "distributed return of array {} not valid"
                    " since it is replicated".format(arr_name)
                )
            array_dists[lhs] = array_dists[arr_name]
            return

        if func_name == "threaded_return":
            arr_name = args[0].name
            assert arr_name in array_dists, "array distribution not found"
            if is_REP(array_dists[arr_name]):
                raise BodoError(
                    "threaded return of array {} not valid" " since it is replicated"
                )
            array_dists[arr_name] = Distribution.Thread
            return

        if func_name == "rebalance":
            self._meet_array_dists(lhs, args[0].name, array_dists)
            return

        if func_name == "random_shuffle":
            self._meet_array_dists(lhs, args[0].name, array_dists)
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
        arg_types = numba.core.typing.templates.fold_arguments(
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
        # is_return_distributed is a list in tuple case which specifies distributions of
        # individual elements
        if isinstance(is_return_distributed, list):
            if lhs not in array_dists:
                self._set_var_dist(
                    lhs,
                    array_dists,
                    [
                        Distribution.OneD_Var if v else Distribution.REP
                        for v in is_return_distributed
                    ],
                )
            # check distributions of tuple elements for errors
            for i, dist in enumerate(array_dists[lhs]):
                if is_REP(dist) and is_return_distributed[i]:
                    raise BodoError(
                        "variable {} is marked as distributed by {} but not possible to"
                        " distribute in caller function {}".format(
                            lhs, dispatcher.__name__, self.func_ir.func_id.func_name
                        )
                    )
        elif is_return_distributed:
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
            self._set_REP(
                v,
                array_dists,
                "input/output of another Bodo call without distributed flag",
            )

    def _analyze_call_concat(self, lhs, args, array_dists, axis=0):
        """analyze distribution for bodo.libs.array_kernels.concat and np.concatenate"""
        assert len(args) == 1, "concat call with only one arg supported"
        # concat reduction variables are handled in parfor analysis
        if lhs in self._concat_reduce_vars:
            return

        in_type = self.typemap[args[0].name]
        # list input case
        if isinstance(in_type, types.List):
            in_list = args[0].name
            # OneD_Var since sum of block sizes might not be exactly 1D
            out_dist = Distribution.OneD_Var
            if lhs in array_dists:
                out_dist = Distribution(min(out_dist.value, array_dists[lhs].value))
            out_dist = Distribution(min(out_dist.value, array_dists[in_list].value))
            array_dists[lhs] = out_dist

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                array_dists[in_list] = out_dist
            return

        tup_def = guard(get_definition, self.func_ir, args[0])
        assert isinstance(tup_def, ir.Expr) and tup_def.op == "build_tuple"
        in_arrs = tup_def.items

        # input arrays have same distribution
        in_dist = Distribution.OneD
        for v in in_arrs:
            in_dist = Distribution(min(in_dist.value, array_dists[v.name].value))

        # when input arrays are concatenated along non-zero axis, output row size is the
        # same as input and data distribution doesn't change
        if axis != 0:
            if lhs in array_dists:
                in_dist = Distribution(min(in_dist.value, array_dists[lhs].value))
            for v in in_arrs:
                array_dists[v.name] = in_dist
            array_dists[lhs] = in_dist
            return

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
            self._set_REP(
                arg1, array_dists, "vector multiplied by matrix rows in np.dot()"
            )
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            # lhs and X have same distribution
            self._meet_array_dists(lhs, arg0, array_dists)
            dprint("dot case 1 Xw:", arg0, arg1)
            return
        if ndim0 == 1 and ndim1 == 2 and not t1:
            # reduction across samples np.dot(Y,X)
            # lhs is always REP
            self._set_REP(
                lhs, array_dists, "output of vector-matrix multiply in np.dot()"
            )
            # Y and X have same distribution
            self._meet_array_dists(arg0, arg1, array_dists)
            dprint("dot case 2 YX:", arg0, arg1)
            return
        if ndim0 == 2 and ndim1 == 2 and t0 and not t1:
            # reduction across samples np.dot(X.T,Y)
            # lhs is always REP
            self._set_REP(
                lhs, array_dists, "output of matrix-matrix multiply in np.dot()"
            )
            # Y and X have same distribution
            self._meet_array_dists(arg0, arg1, array_dists)
            dprint("dot case 3 XtY:", arg0, arg1)
            return
        if ndim0 == 2 and ndim1 == 2 and not t0 and not t1:
            # samples dot weights: np.dot(X,w)
            # w is always REP
            self._set_REP(arg1, array_dists, "matrix multiplied by rows in np.dot()")
            self._meet_array_dists(lhs, arg0, array_dists)
            dprint("dot case 4 Xw:", arg0, arg1)
            return

        # set REP if no pattern matched
        self._analyze_call_set_REP(lhs, args, array_dists, "np.dot")

    def _analyze_call_set_REP(self, lhs, args, array_dists, fdef):
        arrs = []
        for v in args:
            typ = self.typemap[v.name]
            if is_distributable_typ(typ) or is_distributable_tuple_typ(typ):
                dprint("dist setting call arg REP {} in {}".format(v.name, fdef))
                self._set_REP(v.name, array_dists)
                arrs.append(v.name)
        typ = self.typemap[lhs]
        if is_distributable_typ(typ) or is_distributable_tuple_typ(typ):
            dprint("dist setting call out REP {} in {}".format(lhs, fdef))
            self._set_REP(lhs, array_dists)
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
            ).format(", ".join(f"'{self._get_user_varname(a)}'" for a in arrs), fname)
            self._add_diag_info(info)

    def _analyze_getitem(self, inst, lhs, rhs, equiv_set, array_dists):
        """analyze getitem nodes for distribution"""
        in_var = rhs.value
        in_typ = self.typemap[in_var.name]
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
            self._set_REP(inst.list_vars(), array_dists, "getitem input not array")
            return

        if (rhs.value.name, index_var.name) in self._parallel_accesses:
            # XXX: is this always valid? should be done second pass?
            self._set_REP(
                [inst.target], array_dists, "output of distributed getitem is REP"
            )
            return

        # in multi-dimensional case, we only consider first dimension
        # TODO: extend to 2D distribution
        tup_list = guard(find_build_tuple, self.func_ir, index_var)
        if tup_list is not None:
            index_var = tup_list[0]
            # rest of indices should be replicated if array
            other_ind_vars = tup_list[1:]
            self._set_REP(
                other_ind_vars, array_dists, "getitem index variables are REP"
            )

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
            out_dist = Distribution.OneD_Var
            if lhs in array_dists:
                out_dist = Distribution(min(out_dist.value, array_dists[lhs].value))
            out_dist = Distribution(min(out_dist.value, new_dist.value))
            array_dists[lhs] = out_dist
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                self._meet_array_dists(
                    index_var.name, rhs.value.name, array_dists, out_dist
                )
            return

        # whole slice access, output has same distribution as input
        # for example: A = X[:,5]
        if guard(is_whole_slice, self.typemap, self.func_ir, index_var) or guard(
            is_slice_equiv_arr,
            # TODO(ehsan): inst.target instead of in_var results in invalid array
            # analysis equivalence sometimes, see test_dataframe_columns_list
            # inst.target,
            in_var,
            index_var,
            self.func_ir,
            equiv_set,
        ):
            self._meet_array_dists(lhs, in_var.name, array_dists)
            return
        # chunked slice or strided slice can be 1D_Var
        # examples: A = X[:n//3], A = X[::2,5]
        elif isinstance(index_typ, types.SliceType):
            # output array is 1D_Var if input array is distributed
            out_dist = Distribution.OneD_Var
            if lhs in array_dists:
                out_dist = self._min_dist(out_dist, array_dists[lhs])
            out_dist = self._min_dist(out_dist, array_dists[in_var.name])
            array_dists[lhs] = out_dist
            # input can become REP
            if out_dist != Distribution.OneD_Var:
                array_dists[in_var.name] = out_dist
            return

        # avoid parallel scalar getitem when inside a parfor
        # examples: test_np_dot, logistic_regression_rand
        if self.in_parallel_parfor != -1:
            self._set_REP(
                inst.list_vars(), array_dists, "getitem inside parallel loop is REP"
            )
            return

        # int index of dist array
        if isinstance(index_typ, types.Integer):
            # multi-dim not supported yet, TODO: support
            if is_np_array_typ(in_typ) and in_typ.ndim > 1:
                self._set_REP(
                    inst.list_vars(),
                    array_dists,
                    "distributed getitem of multi-dimensional array with int index not supported yet",
                )
            if is_distributable_typ(self.typemap[lhs]):
                self._set_REP(
                    lhs,
                    array_dists,
                    "output of distributed getitem with int index is REP",
                )
            return

        self._set_REP(inst.list_vars(), array_dists, "unsupported getitem distribution")

    def _analyze_setitem(self, inst, equiv_set, array_dists):
        """analyze setitem nodes for distribution"""
        # get index_var without changing IR since we are in analysis
        index_var = get_getsetitem_index_var(inst, self.typemap, [])
        index_typ = self.typemap[index_var.name]
        arr = inst.target
        target_typ = self.typemap[arr.name]
        value_typ = self.typemap[inst.value.name]

        # setitem on list/dictionary of distributed values
        if isinstance(target_typ, (types.List, types.DictType)) and (
            is_distributable_typ(value_typ) or is_distributable_tuple_typ(value_typ)
        ):
            # output and dictionary have the same distribution
            self._meet_array_dists(arr.name, inst.value.name, array_dists)
            return

        if (arr.name, index_var.name) in self._parallel_accesses:
            # no parallel to parallel array set (TODO)
            self._set_REP(
                [inst.value], array_dists, "value set in distributed setitem is REP"
            )
            return

        tup_list = guard(find_build_tuple, self.func_ir, index_var)
        if tup_list is not None:
            index_var = tup_list[0]
            # rest of indices should be replicated if array
            self._set_REP(tup_list[1:], array_dists, "index variables are REP")

        # array selection with boolean index
        if (
            is_np_array_typ(index_typ)
            and index_typ.dtype == types.boolean
            or index_typ == boolean_array
        ):
            # setting scalar or lower dimension value, e.g. A[B] = 1
            if not is_array_typ(value_typ) or value_typ.ndim < target_typ.ndim:
                # input array and bool index have the same distribution
                self._meet_array_dists(arr.name, index_var.name, array_dists)
                self._set_REP(
                    [inst.value],
                    array_dists,
                    "scalar/lower-dimension value set in distributed setitem with bool array index is REP",
                )
                return
            # TODO: support bool index setitem across the whole first dimension, which
            # may require shuffling data to match bool index selection

        # whole slice access, output has same distribution as input
        # for example: X[:,3] = A
        if guard(is_whole_slice, self.typemap, self.func_ir, index_var) or guard(
            is_slice_equiv_arr, arr, index_var, self.func_ir, equiv_set
        ):
            self._meet_array_dists(arr.name, inst.value.name, array_dists)
            return
        # chunked slice or strided slice
        # examples: X[:n//3] = v, X[::2,5] = v
        elif isinstance(index_typ, types.SliceType):
            # if the value is scalar/lower dimension
            if not is_array_typ(value_typ) or value_typ.ndim < target_typ.ndim:
                self._set_REP(
                    [inst.value],
                    array_dists,
                    "scalar/lower-dimension value set in distributed setitem with slice index is REP",
                )
                return
            # TODO: support slice index setitem across the whole first dimension, which
            # may require shuffling data to match slice index selection

        # avoid parallel scalar setitem when inside a parfor
        if self.in_parallel_parfor != -1:
            self._set_REP(
                inst.list_vars(), array_dists, "setitem inside parallel loop is REP"
            )
            return

        # int index setitem of dist array
        if isinstance(index_typ, types.Integer):
            self._set_REP(
                [inst.value],
                array_dists,
                "value set in distributed array setitem is REP",
            )
            return

        # Array boolean idx setitem with scalar value, e.g. A[cond] = val
        if (
            is_array_typ(target_typ)
            and is_array_typ(index_typ)
            and index_typ.dtype == types.bool_
            and not is_array_typ(value_typ)
        ):
            self._meet_array_dists(arr.name, index_var.name, array_dists)
            return

        self._set_REP(
            [inst.value, arr, index_var],
            array_dists,
            "unsupported setitem distribution",
        )

    def _analyze_arg(self, lhs, rhs, array_dists):
        """analyze ir.Arg nodes for distribution. Checks for user flags; sets to REP if
        no user flag found"""
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
                    "Distributed analysis replicated argument {0} (variable "
                    "{1}). Set distributed flag for {0} if distributed partitions "
                    "are passed (e.g. @bodo.jit(distributed=['{0}']))."
                ).format(rhs.name, lhs)
                self._set_REP(lhs, array_dists, info)

    def _analyze_setattr(self, target, attr, value, array_dists):
        """Analyze ir.SetAttr nodes for distribution (e.g. A.b = B)"""
        target_type = self.typemap[target.name]
        val_type = self.typemap[value.name]

        # jitclass setattr (e.g. self.df = df1)
        if (
            isinstance(target_type, types.ClassInstanceType)
            and attr in target_type.class_type.dist_spec
        ):
            # attribute dist spec should be compatible with distribution of value
            attr_dist = target_type.class_type.dist_spec[attr]
            assert is_distributable_typ(val_type) or is_distributable_tuple_typ(
                val_type
            ), "Variable {} is not distributable since it is of type {} (required for setting class field)".format(
                value.name, val_type
            )
            assert value.name in array_dists, "array distribution not found"
            val_dist = array_dists[value.name]
            # value shouldn't have a more restrictive distribution than the dist spec
            # e.g. REP vs OneD
            if val_dist.value < attr_dist.value:
                raise BodoError(
                    f"distribution of value is not compatible with the class attribute"
                    f" distribution spec of {target_type.class_type.class_name} in"
                    f" {target.name}.{attr} = {value.name}"
                )
            return

        self._set_REP([target, value], array_dists, "unsupported SetAttr")

    def _analyze_return(self, var, array_dists):
        """analyze ir.Return nodes for distribution. Checks for user flags; sets to REP
        if no user flag found"""
        if self._is_dist_return_var(var):
            return

        # in case of tuple return, individual variables may be flagged separately
        try:
            vdef = get_definition(self.func_ir, var)
            require(is_expr(vdef, "cast"))
            dcall = get_definition(self.func_ir, vdef.value)
            require(is_expr(dcall, "build_tuple"))
            for v in dcall.items:
                self._analyze_return(v, array_dists)
            return
        except GuardException:
            pass

        info = (
            "Distributed analysis replicated return variable "
            "{}. Set distributed flag for the original variable if distributed "
            "partitions should be returned."
        ).format(var.name)
        self._set_REP([var], array_dists, info)

    def _is_dist_return_var(self, var):
        try:
            vdef = get_definition(self.func_ir, var)
            if is_expr(vdef, "cast"):
                dcall = get_definition(self.func_ir, vdef.value)
            else:
                # tuple return variables don't have "cast" separately
                dcall = vdef
            require(is_expr(dcall, "call"))
            require(
                find_callname(self.func_ir, dcall)
                == ("dist_return", "bodo.libs.distributed_api")
            )
            return True
        except GuardException:
            return False

    def _meet_array_dists(self, arr1, arr2, array_dists, top_dist=None):
        """meet distributions of arrays for consistent distribution"""

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

    def _set_REP(self, var_list, array_dists, info=None):
        """set distribution of all variables in 'var_list' to REP if distributable."""
        if isinstance(var_list, (str, ir.Var)):
            var_list = [var_list]
        for var in var_list:
            varname = var.name if isinstance(var, ir.Var) else var
            # Handle SeriesType since it comes from Arg node and it could
            # have user-defined distribution
            typ = self.typemap[varname]
            if is_distributable_typ(typ) or is_distributable_tuple_typ(typ):
                dprint(f"dist setting REP {varname}")
                # keep diagnostics info if the distribution is changing to REP and extra
                # info is available
                if (
                    varname not in array_dists or not is_REP(array_dists[varname])
                ) and info is not None:
                    info = (
                        "Setting distribution of variable '{}' to REP: ".format(
                            self._get_user_varname(varname)
                        )
                        + info
                    )
                    self._add_diag_info(info)
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
        # TODO: use proper "FullRangeIndex" type
        if not check_type or (
            is_distributable_typ(typ) or is_distributable_tuple_typ(typ)
        ):
            array_dists[varname] = dist

    def _get_dist(self, typ, dist):
        """get proper distribution value for type. Returns list of distributions for
        tuples (but just the input 'dist' otherwise).
        """
        if is_distributable_tuple_typ(typ):
            if isinstance(typ, types.List):
                typ = typ.dtype
            if not isinstance(dist, list):
                dist = [dist] * len(typ.types)
            return [
                self._get_dist(t, dist[i])
                if (is_distributable_typ(t) or is_distributable_tuple_typ(t))
                else None
                for i, t in enumerate(typ.types)
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

    def _get_calc_n_items_range_index(self, size_def):
        """match RangeIndex calc_nitems(r._start, r._stop, r._step) call and return r"""
        require(
            find_callname(self.func_ir, size_def)
            == ("calc_nitems", "bodo.libs.array_kernels")
        )
        start_def = get_definition(self.func_ir, size_def.args[0])
        stop_def = get_definition(self.func_ir, size_def.args[1])
        step_def = get_definition(self.func_ir, size_def.args[2])
        require(is_expr(start_def, "getattr") and start_def.attr == "_start")
        r = start_def.value
        require(
            isinstance(self.typemap[r.name], bodo.hiframes.pd_index_ext.RangeIndexType)
        )
        require(
            is_expr(stop_def, "getattr")
            and stop_def.attr == "_stop"
            and stop_def.value.name == r.name
        )
        require(
            is_expr(step_def, "getattr")
            and step_def.attr == "_step"
            and step_def.value.name == r.name
        )
        return r

    def _get_concat_reduce_vars(self, varname, concat_reduce_vars=None):
        """get output variables of array_kernels.concat() calls which are related to
        concat reduction using reduce variable name.
        """
        if concat_reduce_vars is None:
            concat_reduce_vars = set()
        var_def = guard(get_definition, self.func_ir, varname)
        if is_call(var_def):
            fdef = guard(find_callname, self.func_ir, var_def)
            # data and index variables of dataframes are created from concat()
            if fdef == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
                tup_list = guard(find_build_tuple, self.func_ir, var_def.args[0])
                assert tup_list is not None
                for v in tup_list:
                    self._get_concat_reduce_vars(v.name, concat_reduce_vars)
                index_varname = var_def.args[1].name
                # TODO(ehsan): is the index variable name actually needed
                concat_reduce_vars.add(index_varname)
                self._get_concat_reduce_vars(index_varname, concat_reduce_vars)
            # data and index variables of Series are created from concat()
            if fdef == ("init_series", "bodo.hiframes.pd_series_ext"):
                self._get_concat_reduce_vars(var_def.args[0].name, concat_reduce_vars)
                index_varname = var_def.args[1].name
                concat_reduce_vars.add(index_varname)
                self._get_concat_reduce_vars(index_varname, concat_reduce_vars)
            if fdef == ("concat", "bodo.libs.array_kernels"):
                concat_reduce_vars.add(varname)
            # Index is created from concat(), TODO(ehsan): other index init calls
            if fdef == ("init_numeric_index", "bodo.hiframes.pd_index_ext"):
                self._get_concat_reduce_vars(var_def.args[0].name, concat_reduce_vars)

        return concat_reduce_vars

    def _add_diag_info(self, info):
        """append diagnostics info to be displayed in distributed diagnostics output"""
        if info not in self.diag_info:
            self.diag_info.append(info)

    def _get_user_varname(self, v):
        """get original variable name by user for diagnostics info if possible"""
        if v in self.metadata["parfors"]["var_rename_map"]:
            return self.metadata["parfors"]["var_rename_map"][v]
        return v


def get_reduce_op(reduce_varname, reduce_nodes, func_ir, typemap):
    """find reduction operation in parfor reduction IR nodes."""
    if guard(_is_concat_reduce, reduce_varname, reduce_nodes, func_ir, typemap):
        return Reduce_Type.Concat

    require(len(reduce_nodes) >= 1)
    require(isinstance(reduce_nodes[-1], ir.Assign))

    # ignore extra assignments after reduction operator
    # there could be any number of extra assignment after the reduce node due to SSA
    # changes in Numba 0.53.0rc2
    # See: test_reduction_var_reuse in Numba
    last_ind = -1
    while isinstance(reduce_nodes[last_ind].value, ir.Var):
        require(len(reduce_nodes[:last_ind]) >= 1)
        require(isinstance(reduce_nodes[last_ind - 1], ir.Assign))
        require(
            reduce_nodes[last_ind - 1].target.name == reduce_nodes[last_ind].value.name
        )
        last_ind -= 1
    rhs = reduce_nodes[last_ind].value
    require(isinstance(rhs, ir.Expr))

    if rhs.op == "inplace_binop":
        if rhs.fn in ("+=", operator.iadd):
            return Reduce_Type.Sum
        if rhs.fn in ("|=", operator.ior):
            return Reduce_Type.Or
        if rhs.fn in ("*=", operator.imul):
            return Reduce_Type.Prod

    if rhs.op == "call":
        func = find_callname(func_ir, rhs, typemap)
        if func == ("min", "builtins"):
            if isinstance(
                typemap[rhs.args[0].name],
                numba.core.typing.builtins.IndexValueType,
            ):
                return Reduce_Type.Argmin
            return Reduce_Type.Min
        if func == ("max", "builtins"):
            if isinstance(
                typemap[rhs.args[0].name],
                numba.core.typing.builtins.IndexValueType,
            ):
                return Reduce_Type.Argmax
            return Reduce_Type.Max

        # add_nested_counts is internal and only local result is needed later, so reduce
        # is not necessary
        if func == ("add_nested_counts", "bodo.utils.indexing"):
            return Reduce_Type.No_Op

    raise GuardException  # pragma: no cover


def _is_concat_reduce(reduce_varname, reduce_nodes, func_ir, typemap):
    """return True if reduction nodes match concat pattern"""
    # assuming this structure:
    # A = concat((A, B))
    # I = init_range_index()
    # $df12 = init_dataframe((A,), I, ("A",))
    # df = $df12
    # see test_concat_reduction

    # df = $df12
    require(
        isinstance(reduce_nodes[-1], ir.Assign)
        and reduce_nodes[-1].target.name == reduce_varname
        and isinstance(reduce_nodes[-1].value, ir.Var)
    )
    # $df212 = call()
    require(
        is_call_assign(reduce_nodes[-2])
        and reduce_nodes[-2].target.name == reduce_nodes[-1].value.name
    )
    reduce_func_call = reduce_nodes[-2].value
    fdef = find_callname(func_ir, reduce_nodes[-2].value)
    if fdef == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
        arg_def = get_definition(func_ir, reduce_func_call.args[0])
        require(is_expr(arg_def, "build_tuple"))
        require(len(arg_def.items) > 0)
        reduce_func_call = get_definition(func_ir, arg_def.items[0])

    if fdef == ("init_series", "bodo.hiframes.pd_series_ext"):
        reduce_func_call = get_definition(func_ir, reduce_func_call.args[0])

    require(
        find_callname(func_ir, reduce_func_call)
        == ("concat", "bodo.libs.array_kernels")
    )
    return True


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
    """returns a set of arrays accessed and their indices."""
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
                        if fdef == ("setna", "bodo.libs.array_kernels"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                        if fdef == ("str_arr_item_to_numeric", "bodo.libs.str_arr_ext"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                            accesses.add((rhs.args[2].name, rhs.args[3].name, False))
                        if fdef == ("get_str_arr_item_length", "bodo.libs.str_arr_ext"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                        if fdef == ("inplace_eq", "bodo.libs.str_arr_ext"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                        if fdef == (
                            "str_arr_setitem_int_to_str",
                            "bodo.libs.str_arr_ext",
                        ):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                        if fdef == ("str_arr_setitem_NA_str", "bodo.libs.str_arr_ext"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                        if fdef == ("str_arr_set_not_na", "bodo.libs.str_arr_ext"):
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


def _is_tuple_like_type(t):
    """return True of 't' is a tuple-like type such as tuples or literal list that
    could be used in constant sized Series and Index.
    """
    return (
        isinstance(t, types.BaseTuple)
        or is_heterogeneous_tuple_type(t)
        or isinstance(t, bodo.hiframes.pd_index_ext.HeterogeneousIndexType)
    )


def dprint(*s):  # pragma: no cover
    if debug_prints():
        print(*s)
