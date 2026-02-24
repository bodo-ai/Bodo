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
    unwrap_parfor_blocks,
    wrap_parfor_blocks,
)

import bodo
import bodo.io
import bodo.io.np_io
from bodo.hiframes.pd_categorical_ext import CategoricalArrayType
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.table import TableType
from bodo.libs.bodosql_kernels.bodosql_array_kernels import (
    broadcasted_fixed_arg_functions,
    broadcasted_variadic_functions,
)
from bodo.libs.bool_arr_ext import boolean_array_type
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
    get_overload_const_str,
    is_bodosql_context_type,
    is_overload_constant_tuple,
    is_overload_false,
    is_overload_none,
    is_tuple_like_type,
)
from bodo.utils.utils import (
    debug_prints,
    find_build_tuple,
    get_constant,
    get_getsetitem_index_var,
    is_alloc_callname,
    is_array_typ,
    is_bodosql_kernel_mod,
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

    def __repr__(self):
        """
        Set repr() to str() to enable proper print for distribution of tuples, which are
        represented as list of Distributions.
        [1D_Block_Var, 1D_Block_Var] vs.
        [<Distribution.OneD_Var: 4>, <Distribution.OneD_Var: 4>]

        Python calls repr() on list items for some reason (which seems wrong):
        https://stackoverflow.com/questions/727761/python-str-and-lists
        """
        return self.__str__()


_dist_analysis_result = namedtuple(
    "dist_analysis_result", "array_dists,parfor_dists,concat_reduce_varnames,ret_type"
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
                print(f"   {p:<20} {dist}")
        else:
            print("No parfors to distribute.")
        return

    # NOTE: adding metadata as input instead of attribute to avoid circular dependency
    # since DistributedDiagnostics object is inside metadata
    def dump(self, level, metadata):
        name = self.func_ir.func_id.func_qualname
        line = self.func_ir.loc

        print(f"Distributed diagnostics for function {name}, {line}\n")
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

        print(f"\nDistributed listing for function {name}, {line}")
        self._print_src_dists(lines, level, metadata)

        # trace diag info
        print()
        for l, loc in self.diag_info:
            print(l)
            if loc is not None:
                print(loc.strformat())
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
                map_line_to_info[l_no].append(f"#{p_id}: {p_dist}")

        printed_vars = set()
        for arr, a_dist in self.array_dists.items():
            if arr not in self.array_locs:
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
                map_line_to_info[l_no].append(f"{arr}: {a_dist}")

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
        self,
        func_ir,
        typemap,
        calltypes,
        return_type,
        typingctx,
        metadata,
        flags,
        arr_analysis,
    ):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.return_type = return_type
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
        # keep return variables to help update return type
        self.ret_vars = []

    def _init_run(self):
        """initialize data structures for distribution analysis"""
        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        self._parallel_accesses = set()
        self.second_pass = False
        self.in_parallel_parfor = -1
        self._concat_reduce_vars = set()
        self.ret_vars = []

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
                        f"No parallelism found for function "
                        f"'{self.func_ir.func_id.func_name}'. Distributed diagnostics:"
                        f"\n{self._get_diag_info_str()}"
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

        # update distribution info of data types since necessary during return and for
        # potentially replacing calls to other JIT functions
        for vname, dist in array_dists.items():
            self.typemap[vname] = _update_type_dist(self.typemap.pop(vname), dist)

        # update return type since distribution hints may have changed (can cause
        # lowering error otherwise)
        ret_typ = self.return_type
        if is_distributable_typ(self.return_type) or is_distributable_tuple_typ(
            self.return_type
        ):
            ret_typ = self.typingctx.unify_types(
                *tuple(self.typemap[v] for v in self.ret_vars)
            )

        return _dist_analysis_result(
            array_dists, parfor_dists, self._concat_reduce_vars, ret_typ
        )

    def _check_user_distributed_args(self, array_dists, name, loc):
        """check that no arguments in the originally compiled
        function is specified as distributed but marked as REP"""
        err_msg = (
            "Variable '{}' has distributed flag in function '{}', but it's not "
            "possible to distribute it due to non-distributed dependent operations.\n"
            "Distributed diagnostics:\n{}"
        )
        if name in array_dists and is_REP(array_dists[name]):
            raise BodoError(
                err_msg.format(
                    name,
                    self.func_ir.func_id.func_name,
                    self._get_diag_info_str(),
                ),
                loc,
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
                self.ret_vars.append(inst.value.name)
                self._analyze_return(inst.value, array_dists, inst.loc)
            elif isinstance(inst, ir.SetAttr):
                self._analyze_setattr(
                    inst.target, inst.attr, inst.value, array_dists, inst.loc
                )
            else:
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    inst.list_vars(),
                    array_dists,
                    "unsupported statement in distribution analysis",
                    inst.loc,
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
            _meet_array_dists(self.typemap, lhs, rhs.name, array_dists)
            return
        # NOTE: decimal array comparison isn't inlined since it uses Arrow compute
        elif is_array_typ(lhs_typ) and (
            is_expr(rhs, "inplace_binop")
            or (
                is_expr(rhs, "binop")
                and (
                    isinstance(self.typemap[rhs.lhs.name], bodo.types.DecimalArrayType)
                    or isinstance(
                        self.typemap[rhs.rhs.name], bodo.types.DecimalArrayType
                    )
                )
            )
        ):
            # distributions of all 3 variables should meet (lhs, arg1, arg2)
            # XXX: arg1 or arg2 (but not both) can be non-array like scalar
            arg1 = rhs.lhs.name
            arg2 = rhs.rhs.name
            arg1_typ = self.typemap[arg1]
            arg2_typ = self.typemap[arg2]
            dist = Distribution.OneD
            if is_distributable_typ(arg1_typ):
                dist = _meet_array_dists(self.typemap, lhs, arg1, array_dists)
            if is_distributable_typ(arg2_typ):
                dist = _meet_array_dists(self.typemap, lhs, arg2, array_dists, dist)
            if is_distributable_typ(arg1_typ):
                dist = _meet_array_dists(self.typemap, lhs, arg1, array_dists, dist)
            if is_distributable_typ(arg2_typ):
                _meet_array_dists(self.typemap, lhs, arg2, array_dists, dist)
            return
        elif isinstance(rhs, ir.Expr) and rhs.op in ("getitem", "static_getitem"):
            self._analyze_getitem(inst, lhs, rhs, equiv_set, array_dists)
            return
        elif is_expr(rhs, "build_tuple") and is_distributable_tuple_typ(lhs_typ):
            # parallel arrays can be packed and unpacked from tuples
            # e.g. boolean array index in test_getitem_multidim
            l_dist = _get_var_dist(lhs, array_dists, self.typemap)
            new_dist = []
            for d, v in zip(l_dist, rhs.items):
                # some elements might not be distributable
                if d is None:
                    new_dist.append(None)
                    continue
                new_d = _min_dist(d, _get_var_dist(v.name, array_dists, self.typemap))
                _set_var_dist(self.typemap, v.name, array_dists, new_d)
                new_dist.append(new_d)

            array_dists[lhs] = new_dist
            return
        elif is_expr(rhs, "build_list") and (
            is_distributable_tuple_typ(lhs_typ) or is_distributable_typ(lhs_typ)
        ):
            # dist vars can be in lists
            # meet all distributions
            for v in rhs.items:
                _meet_array_dists(self.typemap, lhs, v.name, array_dists)
            # second round to propagate info fully
            for v in rhs.items:
                _meet_array_dists(self.typemap, lhs, v.name, array_dists)
            return
        elif is_expr(rhs, "build_map") and (
            is_distributable_tuple_typ(lhs_typ) or is_distributable_typ(lhs_typ)
        ):
            # dist vars can be in dictionary as values
            # meet all distributions
            for _, v in rhs.items:
                _meet_array_dists(self.typemap, lhs, v.name, array_dists)
            # second round to propagate info fully
            for _, v in rhs.items:
                _meet_array_dists(self.typemap, lhs, v.name, array_dists)
            return
        elif is_expr(rhs, "exhaust_iter") and is_distributable_tuple_typ(lhs_typ):
            _meet_array_dists(self.typemap, lhs, rhs.value.name, array_dists)
        elif is_expr(rhs, "getattr"):
            self._analyze_getattr(lhs, rhs, array_dists)
        elif is_expr(rhs, "call"):
            self._analyze_call(
                inst,
                lhs,
                rhs,
                rhs.func.name,
                rhs.args,
                dict(rhs.kws),
                equiv_set,
                array_dists,
            )
        # handle both
        # for A in arr_container: ...
        # and
        # for i, A in enumerate(arr_container): ...
        #
        #
        # for A in arr_iter:
        #    iternext -> Pair<A, is_valid>
        #    A = pair_first(iternext(getiter(arr_container)))
        # for i, A in enumerate(arr_iter):
        #    iternext -> Pair<types.BaseTuple(i, A), is_valid>
        #    tuple(i, A) = pair_first(iternext(getiter(arr_container)))
        elif is_expr(rhs, "pair_first") and (
            is_distributable_typ(lhs_typ) or is_distributable_tuple_typ(lhs_typ)
        ):
            arr_container = guard(_get_pair_first_container, self.func_ir, rhs)
            if arr_container is not None:
                _meet_array_dists(self.typemap, lhs, arr_container.name, array_dists)
                return
            # this path is not possible since pair_first is only used in the pattern
            # above, unless if variable definitions have some issue
            else:  # pragma: no cover
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    inst.list_vars(),
                    array_dists,
                    "invalid pair_first",
                    rhs.loc,
                )
        elif isinstance(rhs, ir.Expr) and rhs.op in ("getiter", "iternext"):
            # analyze array container access in pair_first
            return
        elif isinstance(rhs, ir.Arg):
            self._analyze_arg(lhs, rhs, array_dists)
            return
        else:
            if is_expr(rhs, "binop") and rhs.fn in (operator.is_, operator.is_not):
                # Provide distributed support for is None
                arg1, arg2 = rhs.lhs, rhs.rhs
                # If arg2 is None, we want to keep arg1's distribution
                if self.typemap[arg2.name] == types.none:
                    # If the first arg is a Series or dataframe, we want to produce a warning
                    # because this is a common bug.
                    arg1_typ = self.typemap[arg1.name]
                    # TODO: Can we move this warning into typing?
                    if rhs.fn == operator.is_:
                        code_expr = "is None"
                        pandas_fn = "isna"
                    else:
                        code_expr = "is not None"
                        pandas_fn = "notna"
                    if isinstance(
                        arg1_typ, (bodo.types.DataFrameType, bodo.types.SeriesType)
                    ):
                        obj_name = (
                            "DataFrame"
                            if isinstance(arg1_typ, bodo.types.DataFrameType)
                            else "Series"
                        )
                        warning_msg = f"User code checks if a {obj_name} {code_expr} at {arg1.loc}. This checks that the {obj_name} object {code_expr}, not the contents, and is a common bug. To check the contents, please use '{obj_name}.{pandas_fn}()'."
                        warnings.warn(BodoWarning(warning_msg))
                    return
            # Handle [df] * n and matrix multiply
            elif is_expr(rhs, "binop") and rhs.fn == operator.mul:
                if (
                    isinstance(rhs.lhs, ir.Var)
                    and rhs.lhs.name in self.typemap
                    and isinstance(self.typemap[rhs.lhs.name], types.List)
                ):
                    # If expanding a list set the distributions equal.
                    if rhs.lhs.name in array_dists:
                        _meet_array_dists(self.typemap, lhs, rhs.lhs.name, array_dists)
                        return
                elif (
                    isinstance(rhs.rhs, ir.Var)
                    and rhs.rhs.name in self.typemap
                    and isinstance(self.typemap[rhs.rhs.name], types.List)
                ):
                    # If expanding a list set the distributions equal.
                    if rhs.rhs.name in array_dists:
                        _meet_array_dists(self.typemap, lhs, rhs.rhs.name, array_dists)
                        return
                # Matrix multiply
                elif (
                    isinstance(rhs.rhs, ir.Var)
                    and rhs.rhs.name in self.typemap
                    and isinstance(
                        self.typemap[rhs.rhs.name], bodo.libs.matrix_ext.MatrixType
                    )
                ) and (
                    isinstance(rhs.lhs, ir.Var)
                    and rhs.lhs.name in self.typemap
                    and isinstance(
                        self.typemap[rhs.lhs.name], bodo.libs.matrix_ext.MatrixType
                    )
                ):
                    # C = A * B
                    # B is replicated since accessed across rows but A and C can be
                    # distributed
                    _set_REP(
                        self.typemap,
                        self.metadata,
                        self.diag_info,
                        rhs.rhs.name,
                        array_dists,
                        "matrix multiply right hand side input",
                        rhs.loc,
                    )
                    _meet_array_dists(self.typemap, lhs, rhs.lhs.name, array_dists)
                    return
            # Handle Tuple append
            elif is_expr(rhs, "binop") and rhs.fn == operator.add:
                lhs_tuple = (
                    isinstance(rhs.lhs, ir.Var)
                    and rhs.lhs.name in self.typemap
                    and isinstance(self.typemap[rhs.lhs.name], types.BaseTuple)
                )
                rhs_tuple = (
                    isinstance(rhs.rhs, ir.Var)
                    and rhs.rhs.name in self.typemap
                    and isinstance(self.typemap[rhs.rhs.name], types.BaseTuple)
                )
                if lhs_tuple and rhs_tuple:
                    # Create a new tuple dist if both parts have distributions
                    # or is the empty tuple.
                    lhs_can_dist = (
                        rhs.lhs.name in array_dists
                        or len(self.typemap[rhs.lhs.name]) == 0
                    )
                    rhs_can_dist = (
                        rhs.rhs.name in array_dists
                        or len(self.typemap[rhs.rhs.name]) == 0
                    )
                    if lhs_can_dist and rhs_can_dist:
                        lhs_tuple_val = array_dists.get(rhs.lhs.name, [])
                        rhs_tuple_val = array_dists.get(rhs.rhs.name, [])
                        output_tuple = lhs_tuple_val + rhs_tuple_val
                        if output_tuple:
                            array_dists[lhs] = output_tuple
                        return

            # treat global values similar to arguments
            if isinstance(rhs, (ir.FreeVar, ir.Global, ir.Const)):
                # add name to ir.Const to handle it similar to Global/FreeVar nodes
                if isinstance(rhs, ir.Const):
                    rhs.name = lhs
                self._analyze_arg(lhs, rhs, array_dists)
                return

            msg = "unsupported expression in distributed analysis"
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                inst.list_vars(),
                array_dists,
                msg,
                rhs.loc,
            )

    def _analyze_getattr(self, lhs, rhs, array_dists):
        """analyze getattr nodes (ir.Expr.getattr)"""
        # NOTE: assuming getattr doesn't change distribution by default, since almost
        # all attribute accesses are benign (e.g. A.shape). Exceptions should be handled
        # here.

        lhs_typ = self.typemap[lhs]
        rhs_typ = self.typemap[rhs.value.name]
        attr = rhs.attr
        if (
            attr == "T"
            and is_array_typ(lhs_typ)
            and (not isinstance(lhs_typ, types.Array) or lhs_typ.ndim <= 2)
        ):
            # array and its transpose have same distributions
            arr = rhs.value.name
            _meet_array_dists(self.typemap, lhs, arr, array_dists)
            return
        elif attr in ("real", "imag") and is_array_typ(lhs_typ, False):
            # complex array and its real/imag parts have same distributions
            arr = rhs.value.name
            _meet_array_dists(self.typemap, lhs, arr, array_dists)
            return
        elif (
            isinstance(rhs_typ, MultiIndexType)
            and len(rhs_typ.array_types) > 0
            and attr == "_data"
        ):
            # output of MultiIndex._data is a tuple, with all arrays having the same
            # distribution as input MultiIndex
            # find min of all array distributions
            l_dist = _get_var_dist(lhs, array_dists, self.typemap)
            m_dist = _get_var_dist(rhs.value.name, array_dists, self.typemap)
            new_dist = _min_dist(l_dist[0], m_dist)
            for d in l_dist:
                new_dist = _min_dist(new_dist, d)
            _set_var_dist(self.typemap, lhs, array_dists, new_dist)
            _set_var_dist(self.typemap, rhs.value.name, array_dists, new_dist)
            return
        elif isinstance(rhs_typ, CategoricalArrayType) and attr == "codes":
            # categorical array and its underlying codes array have same distributions
            arr = rhs.value.name
            _meet_array_dists(self.typemap, lhs, arr, array_dists)
        elif isinstance(rhs_typ, bodo.types.DatetimeArrayType) and attr == "_data":
            # datetime array and its underlying data array have same distributions
            arr = rhs.value.name
            _meet_array_dists(self.typemap, lhs, arr, array_dists)
        # jitclass getattr (e.g. df1 = self.df)
        elif (
            isinstance(rhs_typ, types.ClassInstanceType)
            and attr in rhs_typ.class_type.dist_spec
        ):
            # attribute dist spec should be compatible with distribution of value
            attr_dist = rhs_typ.class_type.dist_spec[attr]
            assert is_distributable_typ(lhs_typ) or is_distributable_tuple_typ(
                lhs_typ
            ), (
                f"Variable {lhs} is not distributable since it is of type {lhs_typ} (required for getting distributed class field)"
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
                        f" {lhs} = {rhs.value.name}.{attr}",
                        rhs.loc,
                    )
        elif (
            "bodo.libs.pyspark_ext" in sys.modules
            and isinstance(rhs_typ, bodo.libs.pyspark_ext.SparkDataFrameType)
            and attr == "_df"
        ):
            # Spark dataframe may be replicated, e.g. sdf.select(F.sum(F.col("A")))
            _meet_array_dists(self.typemap, lhs, rhs.value.name, array_dists)
        elif is_bodosql_context_type(rhs_typ) and attr == "dataframes":
            _meet_array_dists(self.typemap, lhs, rhs.value.name, array_dists)
        elif isinstance(rhs_typ, DataFrameType) and attr == "dtypes":
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                lhs,
                array_dists,
                "Output of DataFrame.dtypes is replicated",
                rhs.loc,
            )

    def _analyze_parfor(self, parfor, array_dists, parfor_dists):
        """analyze Parfor nodes for distribution. Parfor and its accessed arrays should
        have the same distribution.
        """
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
            _add_diag_info(
                self.diag_info,
                f"Parfor {parfor.id} set to REP since it is inside another distributed Parfor",
                self.parfor_locs[parfor.id],
            )
            out_dist = Distribution.REP

        parfor_arrs = set()  # arrays this parfor accesses in parallel
        array_accesses = _get_array_accesses(
            parfor.loop_body, self.func_ir, self.typemap
        )
        par_index_var = parfor.loop_nests[0].index_variable.name

        for arr, index, _ in array_accesses:
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
                    _add_diag_info(
                        self.diag_info,
                        f"Parfor {parfor.id} set to REP since index is used in lower dimensions of array access",
                        self.parfor_locs[parfor.id],
                    )
                    out_dist = Distribution.REP
            # TODO: check for index dependency

        for arr in parfor_arrs:
            if arr in array_dists:
                out_dist = Distribution(min(out_dist.value, array_dists[arr].value))

        # analyze reductions like concat that can affect parfor distribution
        out_dist, non_concat_redvars = self._get_parfor_reduce_dists(
            parfor, out_dist, array_dists
        )

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
            # reduction arrays of distributed parfors are replicated
            # see test_basic.py::test_array_reduce
            # concat reduce variables should stay distributed
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                non_concat_redvars,
                array_dists,
                "reduction arrays of distributed parfor",
                self.parfor_locs[parfor.id],
            )
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
        non_concat_redvars = []
        for reduce_varname, _reduce_var_info in sorted(parfor.reddict.items()):
            reduce_nodes = _reduce_var_info.reduce_nodes
            reduce_op = guard(
                get_reduce_op, reduce_varname, reduce_nodes, self.func_ir, self.typemap
            )
            if reduce_op == Reduce_Type.Concat:
                # if output array is replicated, parfor should be replicated too
                if is_REP(array_dists[reduce_varname]):
                    _add_diag_info(
                        self.diag_info,
                        f"Parfor {parfor.id} set to REP since its concat reduction variable is REP",
                        self.parfor_locs[parfor.id],
                    )
                    out_dist = Distribution.REP
                else:
                    # concat reduce variables are 1D_Var since each rank can produce
                    # variable amount of data
                    array_dists[reduce_varname] = Distribution.OneD_Var
                # if pafor is replicated, output array is replicated
                if is_REP(out_dist):
                    _add_diag_info(
                        self.diag_info,
                        f"Variable '{_get_user_varname(self.metadata, reduce_varname)}' set to REP since it is a concat reduction variable for Parfor {parfor.id} which is REP",
                        self.parfor_locs[parfor.id],
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
            else:
                non_concat_redvars.append(reduce_varname)

        return out_dist, non_concat_redvars

    def _analyze_call(
        self, inst, lhs, rhs: ir.Expr, func_var, args, kws, equiv_set, array_dists
    ):
        """analyze array distributions in function calls"""
        from bodo.decorators import WrapPythonDispatcherType
        from bodo.transforms.distributed_analysis_call_registry import (
            DistributedAnalysisContext,
            call_registry,
        )

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
                    _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)
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
                self._analyze_call_set_REP(lhs, args, array_dists, fdef, rhs.loc)
                return
        else:
            func_name, func_mod = fdef

        # Similar to ObjModeLiftedWith, we assume out data is distributed (1D_Var)
        if isinstance(self.typemap[rhs.func.name], WrapPythonDispatcherType):
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)
            return

        # Check distributed analysis call registry for handler first
        ctx = DistributedAnalysisContext(
            self.typemap,
            array_dists,
            equiv_set,
            func_name,
            self.metadata,
            self.diag_info,
        )
        # Replace ir.Var with type's class name for easy matching
        fdef_str_var = fdef[:-1] + (
            type(self.typemap[fdef[-1].name]).__name__
            if isinstance(fdef[-1], ir.Var)
            else fdef[-1],
        )
        if call_registry.analyze_call(ctx, inst, fdef_str_var):
            return

        if func_mod == "bodo.hiframes.table" and func_name in (
            "table_filter",
            "table_local_filter",
        ):
            in_var = rhs.args[0]
            index_var = rhs.args[1]
            # Filter code matches getitem code.
            self._analyze_getitem_array_table_inputs(
                inst, lhs, in_var, index_var, rhs.loc, equiv_set, array_dists
            )
            return

        if (
            func_name in {"split"}
            and "bodo.ml_support.sklearn_model_selection_ext" in sys.modules
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.ml_support.sklearn_model_selection_ext.BodoModelSelectionKFoldType,
            )
        ):
            # Not checking get_n_splits for KFold since it might not have a first arg
            self._analyze_call_sklearn_cross_validators(
                lhs, func_name, rhs, kws, array_dists
            )
            return

        if (
            func_name in {"split", "get_n_splits"}
            and "bodo.ml_support.sklearn_model_selection_ext" in sys.modules
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.ml_support.sklearn_model_selection_ext.BodoModelSelectionLeavePOutType,
            )
        ):
            self._analyze_call_sklearn_cross_validators(
                lhs, func_name, rhs, kws, array_dists
            )
            return

        if func_mod in ("sklearn.model_selection._split", "sklearn.model_selection"):
            if func_name == "train_test_split":
                arg0 = rhs.args[0].name
                if lhs not in array_dists:
                    _set_var_dist(
                        self.typemap, lhs, array_dists, Distribution.OneD, True
                    )

                min_dist = _min_dist(array_dists[lhs][0], array_dists[lhs][1])
                min_dist = _min_dist(min_dist, array_dists[arg0])
                if self.typemap[rhs.args[1].name] != types.none:
                    arg1 = rhs.args[1].name
                    min_dist = _min_dist(min_dist, array_dists[arg1])
                    min_dist = _min_dist(min_dist, array_dists[lhs][2])
                    min_dist = _min_dist(min_dist, array_dists[lhs][3])
                    array_dists[arg1] = min_dist

                _set_var_dist(self.typemap, lhs, array_dists, min_dist)
                array_dists[arg0] = min_dist
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
            self._analyze_call_np(lhs, func_name, args, kws, array_dists, rhs.loc)
            return

        if fdef == ("norm", "numpy.linalg"):
            # get axis argument
            axis_var = get_call_expr_arg("numpy.linalg.norm", args, kws, 2, "axis", "")
            if axis_var != "":
                msg = "numpy.linalg.norm(): 'axis' should be constant"
                axis = get_const_value(axis_var, self.func_ir, msg)
                if axis == 1:
                    # With axis=1 (the only version supported at the moment), the
                    # input and the output have the same distribution.
                    _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            return

        # handle array.func calls
        if isinstance(func_mod, ir.Var) and is_array_typ(self.typemap[func_mod.name]):
            self._analyze_call_array(
                lhs, func_mod, func_name, args, array_dists, rhs.loc
            )
            return

        # handle bodosql context calls for catalogs
        if isinstance(func_mod, ir.Var) and func_name in (
            "add_or_replace_catalog",
            "remove_catalog",
        ):
            # Updating a catalog doesn't change the underlying dataframes,
            # so we can do a simple meet on each entry.
            _meet_array_dists(self.typemap, lhs, func_mod.name, array_dists)
            return

        # handle bodosql context calls
        if isinstance(func_mod, ir.Var) and func_name in (
            "add_or_replace_view",
            "remove_view",
        ):
            # The BodoSQLContext behaves like a tuple. We need to verify that
            # each entry matches.
            if func_mod.name not in array_dists:
                _set_var_dist(
                    self.typemap, func_mod.name, array_dists, Distribution.OneD
                )
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD)
            old_lhs_dists = array_dists[lhs]
            old_rhs_var_dists = array_dists.get(func_mod.name, ())
            lhs_dists = []
            rhs_var_dists = []
            skip_table_name = get_overload_const_str(self.typemap[rhs.args[0].name])
            old_typ = self.typemap[func_mod.name]
            # Iterate over the old type. If the name matches we skip it.
            new_typ_offset = 0
            for i, name in enumerate(old_typ.names):
                if name != skip_table_name:
                    if old_rhs_var_dists[i] is None:
                        # TablePath is not distributed
                        new_dist = None
                    else:
                        new_dist = Distribution(
                            min(
                                old_rhs_var_dists[i].value,
                                old_lhs_dists[i + new_typ_offset].value,
                            )
                        )
                    rhs_var_dists.append(new_dist)
                    lhs_dists.append(new_dist)
                else:
                    rhs_var_dists.append(old_rhs_var_dists[i])
                    new_typ_offset = -1

            if func_name == "add_or_replace_view":
                # If we are adding a DataFrame it is always
                # at the end.
                if old_lhs_dists[-1] is None:
                    # TablePath is not distributed
                    new_dist = None
                else:
                    new_dist = Distribution(
                        min(
                            old_lhs_dists[-1].value, array_dists[rhs.args[1].name].value
                        )
                    )
                    # Set the distribution for the dataframe
                    array_dists[rhs.args[1].name] = new_dist
                lhs_dists.append(new_dist)
            # Update the final distributions
            array_dists[lhs] = lhs_dists
            array_dists[func_mod.name] = rhs_var_dists
            return

        # handle df.func calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], DataFrameType
        ):
            self._analyze_call_df(lhs, func_mod, func_name, args, array_dists, rhs.loc)
            return

        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], SeriesType
        ):
            self._analyze_call_series(
                lhs, func_mod, func_name, args, array_dists, rhs.loc
            )
            return

        # input of gatherv should be distributed (likely a user mistake),
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
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                lhs,
                array_dists,
                "output of gatherv() is replicated",
                rhs.loc,
            )
            return

        # input of scatterv should be REP (warn since likely a user mistake)
        if fdef == ("scatterv", "bodo"):
            # output of scatterv is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD)
            elif is_REP(array_dists[lhs]):
                raise BodoError(
                    "Output of scatterv should be a distributed array", rhs.loc
                )

            arg_no = 2
            warn_flag = get_call_expr_arg(
                fdef[0], rhs.args, kws, arg_no, "warn_if_dist", True
            )
            if isinstance(warn_flag, ir.Var):
                # warn if flag is not constant False.
                warn_flag = not is_overload_false(self.typemap[warn_flag.name])
            if warn_flag and _is_1D_or_1D_Var_arr(rhs.args[0].name, array_dists):
                warnings.warn(BodoWarning("Input to scatterv() is distributed"))
            # scatterv() is no-op if input is dist, so output can be 1D_Var
            if _is_1D_or_1D_Var_arr(rhs.args[0].name, array_dists):
                _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            return

        # dict-arr extractall
        if func_mod == "bodo.libs.dict_arr_ext" and func_name in (
            "str_extractall",
            "str_extractall_multi",
        ):
            # default is set to 1D_var because there could be no matches
            # for some strings
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            in_dist = Distribution(
                min(
                    # the first argument is the dictionary encoded array
                    array_dists[rhs.args[0].name].value,
                    # the third argument is the index array
                    array_dists[rhs.args[3].name].value,
                )
            )

            # return is a tuple(array, array, list)
            out_dist = Distribution(
                min(
                    array_dists[lhs][0].value,
                    array_dists[lhs][1].value,
                    array_dists[lhs][2].value,
                    in_dist.value,
                )
            )
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)

            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist
            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, in_dist)
            _set_var_dist(self.typemap, rhs.args[3].name, array_dists, in_dist)
            return

        # dict-arr concat
        if fdef == ("cat_dict_str", "bodo.libs.dict_arr_ext"):
            # all input arrays and output array have the same distribution
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD)

            out_dist = Distribution(
                min(
                    min(a.value for a in array_dists[rhs.args[0].name]),
                    array_dists[lhs].value,
                )
            )
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)
            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, out_dist)
            return

        if fdef == ("read_arrow_next", "bodo.io.arrow_reader"):  # pragma: no cover
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            in_dist = array_dists[rhs.args[0].name]

            # return is a tuple(array, bool)
            out_dist = Distribution(
                min(
                    array_dists[lhs][0].value,
                    in_dist.value,
                )
            )
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)

            if out_dist == Distribution.REP:
                in_dist = out_dist
            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, in_dist, False)
            return

        if fdef == (
            "init_join_state",
            "bodo.libs.streaming.join",
        ):  # pragma: no cover
            # Initialize join state to 1D
            if lhs not in array_dists:
                default_dist = (Distribution.OneD, Distribution.OneD)
                _set_var_dist(self.typemap, lhs, array_dists, default_dist, False)
            return

        if fdef == (
            "join_build_consume_batch",
            "bodo.libs.streaming.join",
        ):  # pragma: no cover
            state_dist = array_dists[rhs.args[0].name]
            build_state_dist = state_dist[0]
            build_table_dist = array_dists[rhs.args[1].name]
            build_dist = Distribution(
                min(
                    build_state_dist.value,
                    build_table_dist.value,
                )
            )
            # Update the build table
            _set_var_dist(self.typemap, rhs.args[1].name, array_dists, build_dist)
            # Update the state
            new_state_dist = (build_dist, state_dist[1])
            _set_var_dist(
                self.typemap, rhs.args[0].name, array_dists, new_state_dist, False
            )
            return

        if fdef == ("runtime_join_filter", "bodo.libs.streaming.join"):
            # Simply match input and output array distributions.
            _meet_array_dists(self.typemap, lhs, rhs.args[1].name, array_dists)
            return

        if fdef == (
            "join_probe_consume_batch",
            "bodo.libs.streaming.join",
        ):  # pragma: no cover
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)
            state_dist = array_dists[rhs.args[0].name]
            # Get the build dist
            build_dist = state_dist[0]
            # Get the probe dist
            probe_state_dist = state_dist[1]
            probe_table_dist = array_dists[rhs.args[1].name]
            probe_dist = Distribution(
                min(
                    probe_state_dist.value,
                    probe_table_dist.value,
                )
            )
            # Determine the output dist
            state_output_dist = Distribution(
                max(
                    build_dist.value,
                    probe_dist.value,
                )
            )
            # return is a tuple(table, bool)
            out_dist = Distribution(
                min(
                    array_dists[lhs][0].value,
                    state_output_dist.value,
                )
            )
            # Update the output
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)
            if out_dist == Distribution.REP:
                # Output can convert inputs to REP.
                build_dist = Distribution.REP
                probe_dist = Distribution.REP

            # Update the probe table
            _set_var_dist(self.typemap, rhs.args[1].name, array_dists, probe_dist)
            # Update the state
            new_state_dist = (build_dist, probe_dist)
            _set_var_dist(
                self.typemap, rhs.args[0].name, array_dists, new_state_dist, False
            )
            return

        if fdef == (
            "iceberg_writer_fetch_theta",
            "bodo.io.iceberg.theta",
        ):
            # Used to obtain the current value of a theta sketch collection from
            # an Iceberg writer as an array, where each row is the current estimate
            # for that column of the table. Answer is replicated since there is
            # only 1 correct answer globally.
            _set_REP(self.typemap, self.metadata, self.diag_info, lhs, array_dists)
            return

        if fdef == ("read_puffin_file_ndvs", "bodo.io.iceberg.theta"):
            # Used to the ndvs from a puffin file for testing.
            _set_REP(self.typemap, self.metadata, self.diag_info, lhs, array_dists)
            return

        if fdef in (
            ("init_groupby_state", "bodo.libs.streaming.groupby"),
            ("init_grouping_sets_state", "bodo.libs.streaming.groupby"),
            ("init_table_builder_state", "bodo.libs.table_builder"),
            ("init_union_state", "bodo.libs.streaming.union"),
            ("init_window_state", "bodo.libs.streaming.window"),
            ("init_stream_sort_state", "bodo.libs.streaming.sort"),
        ):  # pragma: no cover
            # Initialize groupby state to 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD, False)
            return

        if fdef in (
            ("groupby_produce_output_batch", "bodo.libs.streaming.groupby"),
            (
                "groupby_grouping_sets_produce_output_batch",
                "bodo.libs.streaming.groupby",
            ),
            ("window_produce_output_batch", "bodo.libs.streaming.window"),
            ("produce_output_batch", "bodo.libs.streaming.sort"),
        ):  # pragma: no cover
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            state_dist = array_dists[rhs.args[0].name]

            out_dist = Distribution(
                min(
                    array_dists[lhs][0].value,
                    state_dist.value,
                )
            )
            # Update the output
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)

            if out_dist == Distribution.REP:
                # Output can convert inputs to REP.
                state_dist = Distribution.REP

            # Update the state
            _set_var_dist(
                self.typemap, rhs.args[0].name, array_dists, state_dist, False
            )
            return

        if fdef == (
            "table_builder_pop_chunk",
            "bodo.libs.table_builder",
        ):  # pragma: no cover
            if lhs not in array_dists:
                # We can pop unequal chunks on each rank. As a result we must
                # initialize everything to 1DVar.
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)
            lhs_dist = array_dists[lhs][0]
            out_dist = Distribution(
                min(
                    lhs_dist.value,
                    array_dists[rhs.args[0].name].value,
                )
            )
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)
            array_dists[rhs.args[0].name] = out_dist
            return

        if fdef == (
            "table_builder_append",
            "bodo.libs.table_builder",
        ):  # pragma: no cover
            if array_dists[rhs.args[0].name] == Distribution.REP:
                # If the builder is REP then we must set the input table to also
                # be REP
                array_dists[rhs.args[1].name] = Distribution.REP
            # If the input is OneD, it the output should be OneD_Var since
            # appending multiple OneD tables together may break the invariant
            # that we have only at most 1 extra row on a given rank.
            out_dist = _min_dist(Distribution.OneD_Var, array_dists[rhs.args[1].name])
            array_dists[rhs.args[0].name] = out_dist
            return

        if fdef == ("union_consume_batch", "bodo.libs.streaming.union"):
            state = self.typemap[rhs.args[0].name]
            if state.all:
                if array_dists[rhs.args[0].name] == Distribution.REP:
                    # If the builder is REP then we must set the input table to also
                    # be REP
                    array_dists[rhs.args[1].name] = Distribution.REP
                # If the input is OneD, it the output should be OneD_Var since
                # appending multiple OneD tables together may break the invariant
                # that we have only at most 1 extra row on a given rank.
                out_dist = _min_dist(
                    Distribution.OneD_Var, array_dists[rhs.args[1].name]
                )
                array_dists[rhs.args[0].name] = out_dist
            else:
                _meet_array_dists(
                    self.typemap, rhs.args[0].name, rhs.args[1].name, array_dists
                )
            return

        if fdef == ("union_produce_batch", "bodo.libs.streaming.union"):
            state = self.typemap[rhs.args[0].name]
            if state.all:
                if lhs not in array_dists:
                    # We can pop unequal chunks on each rank. As a result we must
                    # initialize everything to 1DVar.
                    _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)
                lhs_dist = array_dists[lhs][0]
                out_dist = Distribution(
                    min(
                        lhs_dist.value,
                        array_dists[rhs.args[0].name].value,
                    )
                )
                _set_var_dist(self.typemap, lhs, array_dists, out_dist)
                array_dists[rhs.args[0].name] = out_dist

            else:
                if lhs not in array_dists:
                    _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

                state_dist = array_dists[rhs.args[0].name]

                out_dist = Distribution(
                    min(
                        array_dists[lhs][0].value,
                        state_dist.value,
                    )
                )
                # Update the output
                _set_var_dist(self.typemap, lhs, array_dists, out_dist)

                if out_dist == Distribution.REP:
                    # Output can convert inputs to REP.
                    state_dist = Distribution.REP

                # Update the state
                _set_var_dist(
                    self.typemap, rhs.args[0].name, array_dists, state_dist, False
                )
            return

        # bodo.libs.distributed_api functions
        if isinstance(func_mod, str) and func_mod == "bodo.libs.distributed_api":
            self._analyze_call_bodo_dist(lhs, func_name, args, array_dists, rhs.loc)
            return

        # handle list.func calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], types.List
        ):
            dtype = self.typemap[func_mod.name].dtype
            if is_distributable_typ(dtype) or is_distributable_tuple_typ(dtype):
                if func_name in ("append", "count", "extend", "index", "remove"):
                    _meet_array_dists(
                        self.typemap, func_mod.name, rhs.args[0].name, array_dists
                    )
                    return
                if func_name == "insert":
                    _meet_array_dists(
                        self.typemap, func_mod.name, rhs.args[1].name, array_dists
                    )
                    return
                if func_name in ("copy", "pop"):
                    _meet_array_dists(self.typemap, lhs, func_mod.name, array_dists)
                    return

        if func_mod == "bodo.io.h5_api" and func_name in (
            "h5read",
            "h5write",
            "h5read_filter",
        ):
            bodo.utils.utils.check_h5py()
            return

        if func_mod == "bodo.io.h5_api" and func_name == "get_filter_read_indices":
            bodo.utils.utils.check_h5py()
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return

        if fdef == ("lateral_flatten", "bodosql.kernels.lateral"):
            # If the input is replicated the output is replicated, otherwise
            # the output is always 1D_Var since each rank may explode its
            # rows into different sizes.
            if lhs not in array_dists:
                if (
                    rhs.args[0].name in array_dists
                    and array_dists[rhs.args[0].name] == Distribution.REP
                ):
                    _set_var_dist(self.typemap, lhs, array_dists, Distribution.REP)
                else:
                    _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)
            return

        if fdef == ("interp_bin_search", "bodo.libs.array_kernels"):
            _meet_array_dists(
                self.typemap, rhs.args[1].name, rhs.args[2].name, array_dists
            )
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

        if fdef == ("random_seedless", "bodosql.kernels"):
            if self.typemap[rhs.args[0].name] != bodo.types.none:
                _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            return

        if fdef == (
            "bodosql_listagg_distributed",
            "bodosql.kernels.listagg",
        ) or fdef == ("bodosql_listagg", "bodosql.kernels.listagg"):
            # Output is a string, so we don't need to explicitly set the distribution
            return
        if func_name in broadcasted_fixed_arg_functions and is_bodosql_kernel_mod(
            func_mod
        ):
            # All of the arguments could be scalars or arrays, but all of the
            # arrays need to meet one another
            arrays = [lhs]
            for arg in rhs.args:
                if is_array_typ(self.typemap[arg.name]):
                    arrays.append(arg.name)
            if len(arrays) > 1:
                _meet_several_array_dists(self.typemap, arrays, array_dists)
            return

        if (
            func_name in broadcasted_variadic_functions
            and is_bodosql_kernel_mod(func_mod)
        ) and not is_overload_constant_tuple(self.typemap[rhs.args[0].name]):
            elems = guard(find_build_tuple, self.func_ir, rhs.args[0])
            assert elems is not None, (
                f"Internal error, unable to find build tuple for arg0 of {func_name}"
            )

            arrays = [lhs]
            for arg in elems:
                if is_array_typ(self.typemap[arg.name]):
                    arrays.append(arg.name)
            if len(arrays) > 1:
                _meet_several_array_dists(self.typemap, arrays, array_dists)
            return

        if fdef == ("bodosql_case_kernel", ""):
            # This is a kernel we generate to avoid inlining case statements
            # We always generate a tuple directly

            # Initialize the distribution of the output array
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            elems = guard(find_build_tuple, self.func_ir, rhs.args[0])
            assert elems is not None, (
                "Internal error, unable to find build tuple for arg0 of bodosql_case_kernel"
            )

            arrays = [lhs] + [elem.name for elem in elems]
            if len(arrays) > 1:
                _meet_several_array_dists(self.typemap, arrays, array_dists)
            return

        if fdef == ("concat_ws", "bodosql.kernels"):
            # If the generate tuple is a constant we skip this path and
            # cannot have any arrays.
            if not is_overload_constant_tuple(self.typemap[rhs.args[0].name]):
                elems = guard(find_build_tuple, self.func_ir, rhs.args[0])
                assert elems is not None, (
                    f"Internal error, unable to find build tuple for arg0 of {func_name}"
                )

                arrays = [lhs]
                for arg in elems:
                    if is_array_typ(self.typemap[arg.name]):
                        arrays.append(arg.name)
                # Get the information about the separator
                sep_name = rhs.args[1].name
                if is_array_typ(self.typemap[sep_name]):
                    arrays.append(sep_name)
                if len(arrays) > 1:
                    _meet_several_array_dists(self.typemap, arrays, array_dists)
                return

        if fdef == ("is_in", "bodosql.kernels"):
            # Case 1: DIST DIST -> DIST, is_parallel=True
            # Case 2: REP  REP  -> REP, is_parallel=False
            # Case 3: DIST REP  -> DIST, is_parallel=False
            # Case 4: REP  DIST:   Banned by construction
            if is_array_typ(self.typemap[rhs.args[0].name]):
                # If the input array is distributed, then the output
                # must be an array with matching distribution
                assert is_array_typ(self.typemap[lhs])
                new_dist = _meet_array_dists(
                    self.typemap, rhs.args[0].name, lhs, array_dists
                )

            assert is_array_typ(self.typemap[rhs.args[1].name])

            # if arg0 is replicated, then we must force arg1 to be replicated as well
            if is_REP(array_dists.get(rhs.args[0].name, None)):
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    rhs.args[1].name,
                    array_dists,
                )

            return

        # I've confirmed that this actually runs on currently nightly, but we never hit it since
        # we don't include bodosql tests in the coverage, and all the tests for this function
        # are on the bodosql side
        if fdef == (
            "merge_sorted_dataframes",
            "bodosql.libs.iceberg_merge_into",
        ):  # pragma: no cover
            # If any of the inputs are replicated, then all of the inputs/outputs must be replicated
            # _set_REP
            if is_REP(array_dists.get(rhs.args[0].name, None)) or is_REP(
                array_dists.get(rhs.args[1].name, None)
            ):
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    [lhs, rhs.args[0].name, rhs.args[1].name],
                    array_dists,
                )
                return
            # If rhs.args[1].name and rhs.args[0].name are both distributed,
            # they do not affect the others distribution

            # Output distribution if not replicated should always be 1D_VAR,
            # due to the possibility of deleting
            # varying numbers of rows on each rank at runtime

            _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)
            return

        if fdef == ("array_isin", "bodo.libs.array"):
            # Case 1: DIST DIST -> DIST, is_parallel=True
            # Case 2: REP  REP  -> REP, is_parallel=False
            # Case 3: DIST REP  -> DIST, is_parallel=False
            # Case 4: REP  DIST:   Banned by construction

            # out_arr and in_arr should have the same distribution
            new_dist = _meet_array_dists(
                self.typemap, rhs.args[0].name, rhs.args[1].name, array_dists
            )

            # if the input is replicated, then we must force the values  to be replicated as well
            if is_REP(new_dist) and _is_1D_or_1D_Var_arr(rhs.args[2].name, array_dists):
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    rhs.args[2].name,
                    array_dists,
                )

            return

        if fdef == ("get_search_regex", "bodo.hiframes.series_str_impl"):
            # out_arr and in_arr should have the same distribution
            new_dist = _meet_array_dists(
                self.typemap, rhs.args[0].name, rhs.args[3].name, array_dists
            )
            array_dists[rhs.args[0].name] = new_dist
            array_dists[rhs.args[3].name] = new_dist
            return

        if fdef == ("rolling_fixed", "bodo.hiframes.rolling"):
            _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            # index array is passed for apply(raw=False) case
            if self.typemap[rhs.args[1].name] != types.none:
                _meet_array_dists(self.typemap, lhs, rhs.args[1].name, array_dists)
                _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("rolling_variable", "bodo.hiframes.rolling"):
            # lhs, in_arr, on_arr should have the same distribution
            new_dist = _meet_array_dists(
                self.typemap, lhs, rhs.args[0].name, array_dists
            )
            new_dist = _meet_array_dists(
                self.typemap, lhs, rhs.args[1].name, array_dists, new_dist
            )
            # index array is passed for apply(raw=False) case
            if self.typemap[rhs.args[2].name] != types.none:
                new_dist = _meet_array_dists(
                    self.typemap, lhs, rhs.args[2].name, array_dists, new_dist
                )
            array_dists[rhs.args[0].name] = new_dist
            array_dists[rhs.args[1].name] = new_dist
            return

        if fdef == ("nlargest", "bodo.libs.array_kernels"):
            # data and index arrays have the same distributions
            _meet_array_dists(
                self.typemap, rhs.args[0].name, rhs.args[1].name, array_dists
            )
            # output of nlargest is REP
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                lhs,
                array_dists,
                "output of nlargest is REP",
                rhs.loc,
            )
            return

        if fdef in (
            ("set_df_column_with_reflect", "bodo.hiframes.pd_dataframe_ext"),
            ("set_dataframe_data", "bodo.hiframes.pd_dataframe_ext"),
            ("set_table_data", "bodo.hiframes.table"),
        ):
            _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            _meet_array_dists(self.typemap, lhs, rhs.args[2].name, array_dists)
            _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("sample_table_operation", "bodo.libs.array_kernels"):
            in_dist = Distribution(
                min(
                    min(a.value for a in array_dists[rhs.args[0].name]),
                    array_dists[rhs.args[1].name].value,
                )
            )
            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, in_dist)
            _set_var_dist(self.typemap, rhs.args[1].name, array_dists, in_dist)
            _set_var_dist(self.typemap, lhs, array_dists, in_dist)
            return

        if fdef == ("nonzero", "bodo.libs.array_kernels"):
            # output of nonzero is variable-length even if input is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            # arg0 is an array
            in_dist = Distribution(array_dists[rhs.args[0].name].value)
            # return is a tuple(array,)
            out_dist = Distribution(
                min(
                    array_dists[lhs][0].value,
                    in_dist.value,
                )
            )
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, in_dist)
            return

        if fdef == ("repeat_kernel", "bodo.libs.array_kernels"):
            # output of repeat_kernel is variable-length even if input is 1D
            # because of the boundary case
            # ex repeat(A, 2) where len(A) = 9 -> (10, 8)
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            # arg0 is an array
            in_dist = Distribution(array_dists[rhs.args[0].name].value)
            # arg1 could be an array
            if is_array_typ(self.typemap[rhs.args[1].name]):
                in_dist = _meet_array_dists(
                    self.typemap, rhs.args[0].name, rhs.args[1].name, array_dists
                )
            # return is an array
            out_dist = Distribution(
                min(
                    array_dists[lhs].value,
                    in_dist.value,
                )
            )
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, in_dist)
            if is_array_typ(self.typemap[rhs.args[1].name]):
                _set_var_dist(self.typemap, rhs.args[1].name, array_dists, in_dist)
            return

        if fdef == ("repeat_like", "bodo.libs.array_kernels"):
            # arr has to be replicated for this function
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                rhs.args[0],
                array_dists,
                "repeat_like argument 0 must be replicated",
                rhs.loc,
            )
            like_dist = Distribution(array_dists[rhs.args[1].name].value)

            # output of repeat_like should have distribution matching dist_like_arr
            # arr should also be distributed
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            # return is an array
            out_dist = Distribution(
                min(
                    array_dists[lhs].value,
                    like_dist.value,
                )
            )
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                _set_var_dist(self.typemap, rhs.args[1].name, array_dists, out_dist)
            return

        if fdef == ("make_replicated_array", "bodo.utils.conversion"):
            # output must be replicated
            _set_var_dist(self.typemap, lhs, array_dists, Distribution.REP)
            return

        if fdef == ("list_to_array", "bodo.utils.conversion"):
            # Initialize output to distirbuted.
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD)
            if array_dists[lhs] == Distribution.OneD_Var:
                raise BodoError(
                    "Output of list_to_array must have a OneD distribution", rhs.loc
                )
            return

        if fdef == ("drop_duplicates", "bodo.libs.array_kernels"):
            # output of drop_duplicates is variable-length even if input is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            # df.drop_duplicates(ignore_index=True) passes None as Index to
            # drop_duplicates kernel, which causes it to return None as output Index.
            ignore_index = is_overload_none(self.typemap[rhs.args[1].name])

            # arg0 is a tuple of arrays, arg1 is an array
            in_dist = Distribution(
                min(
                    min(a.value for a in array_dists[rhs.args[0].name]),
                    (
                        Distribution.OneD.value
                        if ignore_index
                        else array_dists[rhs.args[1].name].value
                    ),
                )
            )
            # return is a tuple(tuple(arrays), array)
            out_dist = Distribution(
                min(
                    min(a.value for a in array_dists[lhs][0]),
                    (
                        Distribution.OneD.value
                        if ignore_index
                        else array_dists[lhs][1].value
                    ),
                    in_dist.value,
                )
            )
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, in_dist)
            _set_var_dist(self.typemap, rhs.args[1].name, array_dists, in_dist)
            return

        if fdef == ("drop_duplicates_table", "bodo.utils.table_utils"):
            # output of drop_duplicates is variable-length even if input is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            # df.drop_duplicates(ignore_index=True) passes None as Index to
            # drop_duplicates_table kernel, which causes it to return None as output
            # Index.
            ignore_index = is_overload_none(self.typemap[rhs.args[1].name])

            # arg0 is a table, arg1 is an array
            in_dist = Distribution(
                min(
                    array_dists[rhs.args[0].name].value,
                    (
                        Distribution.OneD.value
                        if ignore_index
                        else array_dists[rhs.args[1].name].value
                    ),
                )
            )
            # return is a tuple(table, array)
            out_dist = Distribution(
                min(
                    array_dists[lhs][0].value,
                    (
                        Distribution.OneD.value
                        if ignore_index
                        else array_dists[lhs][1].value
                    ),
                    in_dist.value,
                )
            )
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, in_dist)
            _set_var_dist(self.typemap, rhs.args[1].name, array_dists, in_dist)
            return

        if fdef == ("drop_duplicates_table", "bodo.utils.table_utils"):
            # output of drop_duplicates is variable-length even if input is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            # arg0 is a table, arg1 is an array
            in_dist = Distribution(
                min(
                    array_dists[rhs.args[0].name].value,
                    array_dists[rhs.args[1].name].value,
                )
            )
            # return is a tuple(table, array)
            out_dist = Distribution(
                min(
                    array_dists[lhs][0].value,
                    array_dists[lhs][1].value,
                    in_dist.value,
                )
            )
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, in_dist)
            _set_var_dist(self.typemap, rhs.args[1].name, array_dists, in_dist)
            return

        if fdef == ("union_tables", "bodo.libs.array"):
            # output of union_tables is variable-length even if input is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            # arg0 is a tuple of "tables". Here each table is either a
            # tuple of arrays or the actual table type. We currently require
            # either all inputs to be replicated or none to be replicated
            arg0 = rhs.args[0]
            rhs_dists = array_dists[arg0.name]
            # First unify the distributions of the inputs.
            table_dists = []
            for col_dists in rhs_dists:
                table_dist = (
                    Distribution(min(a.value for a in col_dists))
                    if isinstance(col_dists, (list, tuple))
                    else col_dists
                )
                table_dists.append(table_dist)
            rhs_total_dist = Distribution(min(a.value for a in table_dists))
            out_dist = Distribution(min(rhs_total_dist.value, array_dists[lhs].value))
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                _set_var_dist(self.typemap, rhs.args[0].name, array_dists, out_dist)
            return

        if fdef == ("drop_duplicates_array", "bodo.libs.array_kernels"):
            # output of drop_duplicates_array is variable-length even if input is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            # arg0 an array
            in_dist = Distribution(array_dists[rhs.args[0].name].value)

            # return is an array
            out_dist = Distribution(min(array_dists[lhs].value, in_dist.value))
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, in_dist)
            return

        if fdef == ("duplicated", "bodo.libs.array_kernels"):
            # output of duplicated is variable-length even if input is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD)

            # input tuple may be empty, see test_df_duplicated
            if len(self.typemap[rhs.args[0].name]) == 0:
                return

            # arg0 is a tuple of arrays, arg1 is an array
            in_dist = Distribution(min(a.value for a in array_dists[rhs.args[0].name]))
            out_dist = Distribution(min(array_dists[lhs].value, in_dist.value))
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)

            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, out_dist)
            return

        if fdef == ("dropna", "bodo.libs.array_kernels"):
            # output of dropna is variable-length even if input is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            in_dist = Distribution(min(a.value for a in array_dists[rhs.args[0].name]))
            out_dist = Distribution(min(a.value for a in array_dists[lhs]))
            out_dist = Distribution(min(out_dist.value, in_dist.value))
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist
            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, in_dist)
            return

        if fdef == ("pivot_impl", "bodo.hiframes.pd_dataframe_ext"):
            # output of pivot_impl is variable-length even if input is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            # arg0 is a tuple of arrays that must share a distribution.
            index_dist = array_dists[rhs.args[0].name]
            index_final_dist = Distribution(min(index_dist, key=lambda x: x.value))
            columns_dist = array_dists[rhs.args[1].name]
            columns_final_dist = Distribution(min(columns_dist, key=lambda x: x.value))
            values_dist = array_dists[rhs.args[2].name]
            values_final_dist = Distribution(min(values_dist, key=lambda x: x.value))
            in_dist = Distribution(
                min(
                    index_final_dist.value,
                    columns_final_dist.value,
                    values_final_dist.value,
                )
            )
            out_dist = array_dists[lhs]
            final_dist = Distribution(min(in_dist.value, out_dist.value))
            _set_var_dist(self.typemap, lhs, array_dists, final_dist)
            # output can cause input REP
            if final_dist != Distribution.OneD_Var:
                in_dist = final_dist
            # Update args 0, 1, 2 so all arrays have the same dist
            index_dist = len(index_dist) * [in_dist]
            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, index_dist)
            columns_dist = len(columns_dist) * [in_dist]
            _set_var_dist(self.typemap, rhs.args[1].name, array_dists, columns_dist)
            values_dist = len(values_dist) * [in_dist]
            _set_var_dist(self.typemap, rhs.args[2].name, array_dists, values_dist)
            # arg 3 must be replicated
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                rhs.args[3].name,
                array_dists,
                "list of unique values for pivot is replicated",
                rhs.loc,
            )
            # arg 6 must be replicated if it is an array
            if self.typemap[rhs.args[6].name] != types.none:
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    rhs.args[6].name,
                    array_dists,
                    "column names for pivot is replicated",
                    rhs.loc,
                )
            return

        if fdef == ("nancorr", "bodo.libs.array_kernels"):
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                lhs,
                array_dists,
                "output of nancorr is REP",
                rhs.loc,
            )
            return

        if fdef in (
            ("concat", "bodo.libs.array_kernels"),
            ("concat_tables", "bodo.utils.table_utils"),
        ):
            # array/table concat() is similar to np.concatenate
            self._analyze_call_concat(lhs, args, array_dists)
            return

        if fdef == ("move_str_binary_arr_payload", "bodo.libs.str_arr_ext"):
            _meet_array_dists(
                self.typemap, rhs.args[0].name, rhs.args[1].name, array_dists
            )
            return

        if fdef == ("enumerate", "builtins"):
            # Enuemrate only has an impact if the iterable contains an array_like value
            if is_distributable_tuple_typ(self.typemap[lhs]):
                # For enumerate the iterator should be the same dist as the value
                # portion of the enumerate tuple.
                if lhs not in array_dists:
                    _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD)
                if rhs.args[0].name not in array_dists:
                    array_dists[rhs.args[0].name] = Distribution.OneD
                rhs_dist = array_dists[rhs.args[0].name]
                # Just look at the value element
                lhs_dist = array_dists[lhs][1]
                out_dist = Distribution(min(rhs_dist.value, lhs_dist.value))
                # Update input and output
                if rhs_dist != out_dist:
                    array_dists[rhs.args[0].name] = out_dist
                if lhs_dist != out_dist:
                    _set_var_dist(self.typemap, lhs, array_dists, out_dist)
            return

        # dummy hiframes functions
        if func_mod == "bodo.hiframes.pd_series_ext" and func_name in (
            "get_series_data",
            "get_series_index",
        ):
            # NOTE: constant sizes Series/Index is not distributed
            if is_tuple_like_type(self.typemap[lhs]):
                self._analyze_call_set_REP(lhs, args, array_dists, fdef, rhs.loc)
                return

            _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            return

        # add proper diagnostic info for tuple/list to array since usually happens
        # when the code creates Series/Dataframe from list, e.g. pd.Series([1, 2, 3])
        if fdef == ("tuple_list_to_array", "bodo.utils.utils"):
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                rhs.args[0],
                array_dists,
                "Tuples and lists are not distributed by default. Convert to array/Series/DataFrame and use bodo.scatterv() to distribute if necessary.",
                rhs.loc,
            )
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
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            in_dist = _meet_array_dists(
                self.typemap, rhs.args[1].name, rhs.args[0].name, array_dists
            )
            out_dist = Distribution(
                min(array_dists[lhs][0].value, array_dists[lhs][1].value, in_dist.value)
            )
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            array_dists[rhs.args[0].name] = in_dist
            array_dists[rhs.args[1].name] = in_dist
            return

        # explode_no_index(): both args have same dist, output is a 1D_Var array
        if fdef == ("explode_no_index", "bodo.libs.array_kernels"):
            # output of explode is variable-length even if input is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            in_dist = _meet_array_dists(
                self.typemap, rhs.args[1].name, rhs.args[0].name, array_dists
            )
            out_dist = Distribution(min(array_dists[lhs].value, in_dist.value))
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            array_dists[rhs.args[0].name] = in_dist
            array_dists[rhs.args[1].name] = in_dist
            return

        if fdef == ("explode_str_split", "bodo.libs.array_kernels"):
            # output of explode is variable-length even if input is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

            # Input array and index array (args 0 and 3) need to be checked
            in_dist = _meet_array_dists(
                self.typemap, rhs.args[3].name, rhs.args[0].name, array_dists
            )

            out_dist = Distribution(
                min(array_dists[lhs][0].value, array_dists[lhs][1].value, in_dist.value)
            )
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            array_dists[rhs.args[0].name] = in_dist
            array_dists[rhs.args[3].name] = in_dist
            return

        if fdef == ("get_index_data", "bodo.hiframes.pd_index_ext"):
            idx_typ = self.typemap[rhs.args[0].name]
            if isinstance(idx_typ, MultiIndexType):
                # If we have a multi-index set each tuple entry to the
                # dist of the array.
                tuple_typ = self.typemap[lhs]
                if lhs not in array_dists:
                    array_dists[lhs] = [Distribution.OneD] * len(tuple_typ)
                tuple_dist = array_dists[lhs]
                out_dist = Distribution(min(tuple_dist, key=lambda x: x.value))
                out_dist = Distribution(
                    min(out_dist.value, array_dists[rhs.args[0].name].value)
                )
                array_dists[lhs] = [out_dist] * len(tuple_typ)
                _set_var_dist(self.typemap, rhs.args[0].name, array_dists, out_dist)
            else:
                _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            return

        # RangeIndexType is technically a distributable type even though the
        # object doesn't require communication
        if fdef == ("init_range_index", "bodo.hiframes.pd_index_ext"):
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD

            # some operations like groupby(as_index=False) create a RangeIndex
            # with the same size as input arrays. This RangeIndex should have the same
            # distribution as the input arrays semantically as well. See [BE-2569]
            is_simple_range = (
                guard(
                    get_const_value_inner,
                    self.func_ir,
                    args[0],
                    typemap=self.typemap,
                )
                == 0
                and guard(
                    get_const_value_inner,
                    self.func_ir,
                    args[2],
                    typemap=self.typemap,
                )
                == 1
            )
            if is_simple_range:
                size_var = args[1]
                for v in equiv_set.get_equiv_set(size_var):
                    # 'v' could be int (size value) or str (varname)
                    if (
                        isinstance(v, str)
                        and "#" in v
                        and v.split("#")[0] in array_dists
                        and not isinstance(array_dists[v.split("#")[0]], list)
                    ):
                        arr_name = v.split("#")[0]
                        _meet_array_dists(self.typemap, lhs, arr_name, array_dists)

            return

        if fdef == ("generate_empty_table_with_rows", "bodo.hiframes.table"):
            # [BSE-2310] investigate interaction with row_number kernel causing the output
            # to be assigned 1DVar when it should not be.
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return

        if fdef == ("init_multi_index", "bodo.hiframes.pd_multi_index_ext"):
            # input arrays and output index have the same distribution
            tup_list = guard(find_build_tuple, self.func_ir, rhs.args[0])
            if tup_list is not None:
                for v in tup_list:
                    _meet_array_dists(self.typemap, lhs, v.name, array_dists)
                for v in tup_list:
                    _meet_array_dists(self.typemap, lhs, v.name, array_dists)
            else:
                tup_arr_dists = array_dists[rhs.args[0].name]
                assert all(tup_arr_dists[0] == arr_dist for arr_dist in tup_arr_dists)
                _set_var_dist(self.typemap, lhs, array_dists, tup_arr_dists[0])
            return

        if fdef == ("init_series", "bodo.hiframes.pd_series_ext"):
            # NOTE: constant sizes Series/Index is not distributed
            if is_tuple_like_type(self.typemap[rhs.args[0].name]):
                self._analyze_call_set_REP(lhs, args, array_dists, fdef, rhs.loc)
                return

            # lhs, in_arr, and index should have the same distribution
            new_dist = _meet_array_dists(
                self.typemap, lhs, rhs.args[0].name, array_dists
            )
            new_dist = _meet_array_dists(
                self.typemap, lhs, rhs.args[1].name, array_dists, new_dist
            )
            array_dists[rhs.args[0].name] = new_dist
            return

        if fdef == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
            # NOTE: constant sizes DataFrame is not distributed
            if any(is_tuple_like_type(t) for t in self.typemap[rhs.args[0].name].types):
                self._analyze_call_set_REP(lhs, args, array_dists, fdef, rhs.loc)
                return

            # lhs, data arrays, and index should have the same distribution
            # data arrays
            data_varname = rhs.args[0].name
            ind_varname = rhs.args[1].name

            # empty dataframe case, Index and df have same distribution
            if len(self.typemap[data_varname].types) == 0:
                _meet_array_dists(self.typemap, lhs, ind_varname, array_dists)
                return

            new_dist_val = min(a.value for a in array_dists[data_varname])
            if lhs in array_dists:
                new_dist_val = min(new_dist_val, array_dists[lhs].value)
            # handle index
            new_dist_val = min(new_dist_val, array_dists[ind_varname].value)
            new_dist = Distribution(new_dist_val)
            _set_var_dist(self.typemap, data_varname, array_dists, new_dist)
            _set_var_dist(self.typemap, ind_varname, array_dists, new_dist)
            _set_var_dist(self.typemap, lhs, array_dists, new_dist)
            return

        if fdef == ("pushdown_safe_init_df", "bodo.hiframes.pd_dataframe_ext"):
            new_dist_val = array_dists[rhs.args[0].name].value
            if lhs in array_dists:
                new_dist_val = min(new_dist_val, array_dists[lhs].value)
            new_dist = Distribution(new_dist_val)
            _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("init_dict_arr", "bodo.libs.dict_arr_ext"):
            # The dictionary is available in the main IR must
            # be replicated. Here we assume other the data risks
            # having incorrect indices. The indices should match
            # the distribution of the output.
            _meet_array_dists(self.typemap, lhs, rhs.args[1].name, array_dists)
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                rhs.args[0].name,
                array_dists,
                "init_dict_arr dictionary array is REP",
                rhs.loc,
            )
            return

        if fdef == ("init_datetime_array", "bodo.libs.pd_datetime_arr_ext"):
            # lhs and data should have the same distribution
            _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            _meet_array_dists(self.typemap, lhs, rhs.args[1].name, array_dists)
            return

        if fdef == ("init_integer_array", "bodo.libs.int_arr_ext"):
            # lhs, data, and bitmap should have the same distribution
            new_dist = _meet_array_dists(
                self.typemap, lhs, rhs.args[0].name, array_dists
            )
            new_dist = _meet_array_dists(
                self.typemap, lhs, rhs.args[1].name, array_dists, new_dist
            )
            array_dists[rhs.args[0].name] = new_dist
            return

        if fdef == ("init_float_array", "bodo.libs.float_arr_ext"):
            # lhs, data, and bitmap should have the same distribution
            new_dist = _meet_array_dists(
                self.typemap, lhs, rhs.args[0].name, array_dists
            )
            new_dist = _meet_array_dists(
                self.typemap, lhs, rhs.args[1].name, array_dists, new_dist
            )
            array_dists[rhs.args[0].name] = new_dist
            return

        if fdef == ("init_categorical_array", "bodo.hiframes.pd_categorical_ext"):
            # lhs and codes should have the same distribution
            _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            return

        if func_mod == "bodo.hiframes.pd_dataframe_ext" and func_name in (
            "get_dataframe_data",
            "get_dataframe_index",
            "get_dataframe_table",
        ):
            # NOTE: constant sizes DataFrame is not distributed
            if any(is_tuple_like_type(t) for t in self.typemap[rhs.args[0].name].data):
                self._analyze_call_set_REP(lhs, args, array_dists, fdef, rhs.loc)
                return
            _meet_array_dists(self.typemap, lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ("get_dataframe_all_data", "bodo.hiframes.pd_dataframe_ext"):
            in_df = rhs.args[0].name
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD, False)

            min_dist = Distribution(
                min(
                    (
                        array_dists[lhs].value
                        if isinstance(self.typemap[lhs], TableType)
                        else min(a.value for a in array_dists[lhs])
                    ),
                    array_dists[in_df].value,
                )
            )
            _set_var_dist(self.typemap, lhs, array_dists, min_dist)
            _set_var_dist(self.typemap, in_df, array_dists, min_dist)
            return

        if fdef == ("logical_table_to_table", "bodo.hiframes.table"):
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD, False)

            # NOTE: replacing dead input arguments with None is done in table column
            # del pass so there is no None input here
            in_table = rhs.args[0].name
            in_extra_arrs = rhs.args[1].name

            in_table_dist = (
                array_dists[in_table]
                if in_table in array_dists
                else [Distribution.OneD]
            )
            # input "table" could be a tuple of arrays (list of distributions)
            in_table_dist = (
                in_table_dist if isinstance(in_table_dist, list) else [in_table_dist]
            )
            in_extra_arrs_dist = (
                array_dists[in_extra_arrs]
                if in_extra_arrs in array_dists
                else [Distribution.OneD]
            )
            in_dists = in_extra_arrs_dist + in_table_dist

            min_dist = Distribution(
                min(
                    min(a.value for a in in_dists),
                    array_dists[lhs].value,
                )
            )
            _set_var_dist(self.typemap, lhs, array_dists, min_dist)
            _set_var_dist(self.typemap, in_table, array_dists, min_dist)
            _set_var_dist(self.typemap, in_extra_arrs, array_dists, min_dist)
            return

        if fdef == ("get_dataframe_column_names", "bodo.hiframes.table"):
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                lhs,
                array_dists,
                "DataFrame column names is REP",
                rhs.loc,
            )
            return

        if fdef == ("generate_table_nbytes", "bodo.utils.table_utils"):
            # Arg1 is the output array and is always replicated.
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                rhs.args[1],
                array_dists,
                "nbytes array in generate_table_nbytes is REP",
                rhs.loc,
            )
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
            _meet_array_dists(self.typemap, out_arrname, in_arrname, array_dists)
            return

        if fdef == ("str_arr_item_to_numeric", "bodo.libs.str_arr_ext"):
            out_arrname = rhs.args[0].name
            in_arrname = rhs.args[2].name
            _meet_array_dists(self.typemap, out_arrname, in_arrname, array_dists)
            return

        if fdef == ("get_data", "bodo.libs.struct_arr_ext"):
            in_arrname = rhs.args[0].name
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD)
            if in_arrname not in array_dists:
                _set_var_dist(self.typemap, in_arrname, array_dists, Distribution.OneD)

            lhs_dists, rhs_dist = array_dists[lhs], array_dists[in_arrname]
            new_dist = Distribution(
                min(min(x.value for x in lhs_dists), rhs_dist.value)
            )
            new_dist = _min_dist_top(new_dist, Distribution.OneD)

            array_dists[lhs] = [new_dist] * len(lhs_dists)
            array_dists[in_arrname] = new_dist
            return

        if fdef == ("init_spark_df", "bodo.libs.pyspark_ext"):
            in_df_name = args[0].name
            # Spark dataframe may be replicated, e.g. sdf.select(F.sum(F.col("A")))
            _meet_array_dists(self.typemap, lhs, in_df_name, array_dists)
            return

        # str_arr_from_sequence() applies to lists/tuples so output is REP
        # e.g. column names in df.mean()
        if fdef == ("str_arr_from_sequence", "bodo.libs.str_arr_ext"):
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                lhs,
                array_dists,
                "output of str_arr_from_sequence is REP",
                rhs.loc,
            )
            return

        if fdef == ("_bodo_groupby_apply_impl", ""):
            # output is variable-length even if input is 1D
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)

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
            _set_var_dist(self.typemap, lhs, array_dists, out_dist)

            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                in_dist = out_dist

            _set_var_dist(self.typemap, rhs.args[0].name, array_dists, in_dist)
            _set_var_dist(self.typemap, rhs.args[1].name, array_dists, in_dist)
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                args[2:],
                array_dists,
                "extra argument in groupby.apply()",
                rhs.loc,
            )
            return
        if fdef == ("fft2", "scipy.fftpack._basic") or fdef == (
            "fft2",
            "scipy.fft._basic",
        ):
            # If input is REP, output is REP
            # If input is 1D_Var, output is 1D_Var
            # If input is 1D, output is 1D_Var
            if lhs not in array_dists:
                # Default to 1D_Var, fftw's distribution does not match our 1D distribution
                # so 1D always becomes 1D_Var
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)
            new_dist = Distribution(
                min(array_dists[lhs].value, array_dists[rhs.args[0].name].value)
            )
            array_dists[lhs] = new_dist
            # If output is REP set input to REP
            if new_dist == Distribution.REP:
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    rhs.args[0],
                    array_dists,
                    "Output of FFT is replicated.",
                    rhs.loc,
                )
            return
        if fdef == ("fftshift", "numpy.fft") or fdef == (
            "fftshift",
            "scipy.fft._helper",
        ):
            # If input is REP, output is REP
            # If input is 1D_Var, output is 1D
            # If input is 1D, output is 1D
            if lhs not in array_dists:
                # Default to 1D, this always does an alltoallv
                # so 1D_Var always becomes 1D
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD)
            new_dist = Distribution(
                min(array_dists[lhs].value, array_dists[rhs.args[0].name].value)
            )
            if new_dist == Distribution.OneD_Var:
                new_dist = Distribution.OneD
            # If output is REP set input to REP
            if new_dist == Distribution.REP:
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    rhs.args[0],
                    array_dists,
                    "Output of fftshift is replicated.",
                    rhs.loc,
                )
            array_dists[lhs] = new_dist
            return

        if func_name == "execute_javascript_udf" and (
            func_mod == "bodosql.kernels.javascript_udf_array_kernels"
            or func_mod == "bodosql.kernels"
        ):  # pragma: no cover
            # All of the arguments could be scalars or arrays, but all of the
            # arrays need to meet one another
            _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD)
            arrays = [lhs]
            for arg in rhs.args:
                if is_array_typ(self.typemap[arg.name]):
                    arrays.append(arg.name)
            if len(arrays) > 1:
                _meet_several_array_dists(self.typemap, arrays, array_dists)
            return

        # handle calling other Bodo functions that have distributed flags
        func_type = self.typemap[func_var]
        if isinstance(func_type, types.Dispatcher) and issubclass(
            func_type.dispatcher._compiler.pipeline_class, bodo.compiler.BodoCompiler
        ):
            self._handle_dispatcher(func_type.dispatcher, lhs, rhs, array_dists)
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, fdef, rhs.loc)

    def _analyze_call_sklearn_cross_validators(
        self, lhs, func_name, rhs, kws, array_dists
    ):
        """
        Analyze distribution of sklearn.model_selection.KFold and LeavePOut functions.
        """

        if func_name in {"split"}:
            # match dist of X and groups (if provided)
            X_arg_name = rhs.args[0].name
            if len(rhs.args) >= 3:
                groups_arg_name = rhs.args[2].name
            elif "groups" in kws:
                groups_arg_name = kws["groups"].name
            else:
                groups_arg_name = None

            if groups_arg_name:
                _meet_array_dists(
                    self.typemap, X_arg_name, groups_arg_name, array_dists
                )

    def _analyze_call_np(self, lhs, func_name, args, kws, array_dists, loc):
        """analyze distributions of numpy functions (np.func_name)"""
        # TODO: handle kw args properly
        if func_name == "ascontiguousarray":
            _meet_array_dists(self.typemap, lhs, args[0].name, array_dists)
            return

        if func_name == "select":
            cond_list = get_call_expr_arg("select", args, kws, 0, "condlist")
            choice_list = get_call_expr_arg("select", args, kws, 1, "choicelist")

            # if the condlist/choicelist are basetuples, the distributions
            # will be lists. If they are lists/unituples, they will be scalar.
            cond_list_typ = self.typemap[cond_list.name]
            choice_list_typ = self.typemap[choice_list.name]
            cond_list_basetuple = isinstance(cond_list_typ, types.BaseTuple)
            choice_list_basetuple = isinstance(choice_list_typ, types.BaseTuple)

            if cond_list.name not in array_dists:
                # initialize to 1D if not found
                if cond_list_basetuple:
                    cond_list_dist = [Distribution.OneD] * len(cond_list_typ)
                else:
                    cond_list_dist = Distribution.OneD
            else:
                cond_list_dist = array_dists[cond_list.name]

            if choice_list.name not in array_dists:
                # initialize to 1D if not found
                if choice_list_basetuple:
                    choice_list_dist = [Distribution.OneD] * len(choice_list_typ)
                else:
                    choice_list_dist = Distribution.OneD
            else:
                choice_list_dist = array_dists[choice_list.name]

            if not cond_list_basetuple:
                cond_list_dist = [cond_list_dist]
            if not choice_list_basetuple:
                choice_list_dist = [choice_list_dist]

            if lhs in array_dists:
                lhs_dist = array_dists[lhs]
            else:
                lhs_dist = Distribution.OneD

            min_dist = lhs_dist

            for dist_val in cond_list_dist + choice_list_dist:
                min_dist = _min_dist(min_dist, dist_val)

            array_dists[lhs] = min_dist
            if cond_list_basetuple:
                array_dists[cond_list.name] = [min_dist] * len(cond_list_dist)
            else:
                array_dists[cond_list.name] = min_dist
            if choice_list_basetuple:
                array_dists[choice_list.name] = [min_dist] * len(choice_list_dist)
            else:
                array_dists[choice_list.name] = min_dist
            return

        if func_name == "ravel":
            if lhs not in array_dists:
                if self.typemap[args[0].name].ndim != 1:
                    # special case: output is 1D_Var since we just reshape
                    # locally without data exchange
                    array_dists[lhs] = Distribution.OneD_Var
                else:
                    array_dists[lhs] = Distribution.OneD
            in_dist = array_dists[args[0].name]
            out_dist = array_dists[lhs]
            out_dist = Distribution(min(out_dist.value, in_dist.value))
            array_dists[lhs] = out_dist
            # output can cause input REP. Ravel can never force
            # the input to go from 1D -> 1DVar, regardless of
            # the initial distribution
            if out_dist == Distribution.REP:
                array_dists[args[0].name] = out_dist
            return

        if func_name == "digitize":
            in_arr = get_call_expr_arg("digitize", args, kws, 0, "x")
            bins = get_call_expr_arg("digitize", args, kws, 1, "bins")
            _meet_array_dists(self.typemap, lhs, in_arr.name, array_dists)
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                bins.name,
                array_dists,
                "'bins' argument of 'digitize' is REP",
                loc,
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
                _meet_array_dists(self.typemap, lhs, arg.name, array_dists)
            else:
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    lhs,
                    array_dists,
                    "output of np.array() call on non-array is REP",
                    loc,
                )
            return

        if func_name == "asarray":
            arg = get_call_expr_arg("asarray", args, kws, 0, "a")
            # np.asarray of another array can be distributed, but not list/tuple
            if is_array_typ(self.typemap[args[0].name]):
                _meet_array_dists(self.typemap, lhs, args[0].name, array_dists)
            else:
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    lhs,
                    array_dists,
                    "output of np.asarray() call on non-array is REP",
                    loc,
                )
            return

        if func_name == "asmatrix":
            input_type = self.typemap[args[0].name]
            # Returns the data as is for 2D array and matrix input so no change in
            # distribution. Other input cases provide matrix rows individually so output
            # is replicated.
            if (
                isinstance(input_type, types.Array) and input_type.ndim == 2
            ) or isinstance(input_type, bodo.libs.matrix_ext.MatrixType):
                _meet_array_dists(self.typemap, lhs, args[0].name, array_dists)
                return
            else:
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    lhs,
                    array_dists,
                    "output of np.asmatrix() call is REP if input is not 2D array or matrix",
                    loc,
                )

        if func_name == "interp":
            # Output matches 1st input, and 2nd/3rd must match each other
            _meet_array_dists(self.typemap, lhs, args[0].name, array_dists)
            _meet_array_dists(self.typemap, args[1].name, args[2].name, array_dists)
            return

        # handle array.sum() with axis
        if func_name == "sum":
            axis_var = get_call_expr_arg("sum", args, kws, 1, "axis", "")
            axis = guard(find_const, self.func_ir, axis_var)
            # sum over the first axis produces REP output
            if axis == 0:
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    lhs,
                    array_dists,
                    "sum over the first axis produces REP output",
                    loc,
                )
                return
            # sum over other axis doesn't change distribution
            if axis_var != "" and axis != 0:
                _meet_array_dists(self.typemap, lhs, args[0].name, array_dists)
                return

        if func_name == "dot":
            self._analyze_call_np_dot(lhs, args, array_dists, loc)
            return

        # used in df.values
        if func_name == "stack":
            seq_info = guard(find_build_sequence, self.func_ir, args[0])
            if seq_info is None:
                self._analyze_call_set_REP(
                    lhs, args, array_dists, "np." + func_name, loc
                )
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
                    _meet_array_dists(self.typemap, lhs, v.name, array_dists)
                return

        if func_name == "reshape":
            # shape argument can be int or tuple of ints
            arr_var = get_call_expr_arg("np.reshape", args, kws, 0, "a")
            shape_var = get_call_expr_arg("np.reshape", args, kws, 1, "newshape")
            shape_typ = self.typemap[shape_var.name]
            if isinstance(shape_typ, types.Integer):
                shape_vars = [shape_var]
            else:
                assert isinstance(shape_typ, types.BaseTuple), (
                    "np.reshape(): invalid shape argument"
                )
                shape_vars = find_build_tuple(self.func_ir, shape_var, True)
            return self._analyze_call_np_reshape(
                lhs, arr_var, shape_vars, array_dists, loc
            )

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
            _meet_array_dists(self.typemap, lhs, in_arr, array_dists)
            return

        # pragma is needed as the only test is marked as slow
        if func_name == "mod":  # pragma: no cover
            arys_to_meet = [lhs]
            if is_distributable_typ(self.typemap[args[0].name]):
                arys_to_meet += [args[0].name]
            if is_distributable_typ(self.typemap[args[1].name]):
                arys_to_meet += [args[1].name]
            if len(arys_to_meet) > 1:
                _meet_several_array_dists(self.typemap, arys_to_meet, array_dists)
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, "np." + func_name, loc)

    def _alloc_call_size_equiv(self, lhs, size_var, equiv_set, array_dists):
        """match distribution of output variable 'lhs' of allocation with distributions
        of equivalent arrays (as found by allocation size 'size_var').
        See test_1D_Var_alloc4.
        """
        size_def = guard(get_definition, self.func_ir, size_var)
        # find trivial calc_nitems(0, n, 1) call and use n instead
        if (
            guard(find_callname, self.func_ir, size_def, self.typemap)
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
            _meet_array_dists(self.typemap, lhs, r.name, array_dists)

        # all arrays with equivalent size should have same distribution
        var_set = equiv_set.get_equiv_set(size_var)
        if not var_set:
            return

        # Avoid matching distributions if one of the arrays is transposed since one
        # dimension of the transposed array can be replicated before transpose but
        # distributed afterwards. Therefore, matching arrays with same size universally
        # can be incorrect.
        if any(
            isinstance(v, str)
            # array analysis adds "#0" to array name to designate 1st dimension
            # See https://github.com/numba/numba/blob/d4460feb8c91213e7b89f97b632d19e34a776cd3/numba/parfors/array_analysis.py#L439
            and "#0" in v
            and guard(_is_transposed_array, self.func_ir, v.split("#")[0])
            for v in var_set
        ):
            return

        for v in var_set:
            # array analysis adds "#0" to array name to designate 1st dimension
            # See https://github.com/numba/numba/blob/d4460feb8c91213e7b89f97b632d19e34a776cd3/numba/parfors/array_analysis.py#L439
            if isinstance(v, str) and "#0" in v:
                arr_name = v.split("#")[0]
                if is_distributable_typ(self.typemap[arr_name]):
                    _meet_array_dists(self.typemap, lhs, arr_name, array_dists)

    def _analyze_call_array(self, lhs, arr, func_name, args, array_dists, loc):
        """analyze distributions of array functions (arr.func_name)"""
        if func_name == "transpose":
            if len(args) == 0:
                raise BodoError("Transpose with no arguments is not supported", loc)
            in_arr_name = arr.name
            arg0 = guard(get_constant, self.func_ir, args[0])
            if isinstance(arg0, tuple):
                arg0 = arg0[0]
            if arg0 != 0:
                raise BodoError(
                    "Transpose with non-zero first argument is not supported", loc
                )
            _meet_array_dists(self.typemap, lhs, in_arr_name, array_dists)
            return

        if func_name == "reshape":
            # array.reshape supports shape input as single tuple, as well as separate
            # arguments
            shape_vars = args
            arg_typ = self.typemap[args[0].name]
            if isinstance(arg_typ, types.BaseTuple):
                shape_vars = find_build_tuple(self.func_ir, args[0], True)
            return self._analyze_call_np_reshape(lhs, arr, shape_vars, array_dists, loc)

        if func_name == "all":
            # array.all() is supported for all distributions
            return

        if func_name in ("astype", "copy", "view", "tz_convert", "to_numpy", "view"):
            in_arr_name = arr.name
            _meet_array_dists(self.typemap, lhs, in_arr_name, array_dists)
            return

        # Array.tofile() is supported for all distributions
        if func_name == "tofile":
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, "array." + func_name, loc)

    def _analyze_call_np_reshape(self, lhs, arr, shape_vars, array_dists, loc):
        """distributed analysis for array.reshape or np.reshape calls"""
        # REP propagates from input to output and vice versa
        if is_REP(array_dists[arr.name]) or (
            lhs in array_dists and is_REP(array_dists[lhs])
        ):
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                [lhs, arr.name],
                array_dists,
                "np.reshape() input/output are REP",
                loc,
            )
            return

        # optimization: no need to distribute if 1-dim array is reshaped to
        # 2-dim with same length (just added a new dimension)
        if (
            self.typemap[arr.name].ndim == 1
            and len(shape_vars) == 2
            and (
                (isinstance(shape_vars[1], int) and shape_vars[1] == 1)
                or (
                    guard(
                        get_const_value_inner,
                        self.func_ir,
                        shape_vars[1],
                        typemap=self.typemap,
                    )
                    == 1
                )
            )
        ):
            _meet_array_dists(self.typemap, lhs, arr.name, array_dists)
            return

        # reshape to 1 dimension
        # special case: output is 1D_Var since we just reshape locally without data
        # exchange
        if len(shape_vars) == 1:
            _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)
        # all other cases will have data exchange resulting in 1D distribution
        else:
            _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD)

    def _analyze_call_df(self, lhs, arr, func_name, args, array_dists, loc):
        # to_csv() and to_parquet() can be parallelized
        if func_name in {"to_csv", "to_parquet", "to_sql"}:
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, "df." + func_name, loc)

    def _analyze_call_series(self, lhs, arr, func_name, args, array_dists, loc):
        if func_name in {"to_csv"}:
            return

        self._analyze_call_set_REP(lhs, args, array_dists, "series." + func_name, loc)

    def _analyze_call_bodo_dist(self, lhs, func_name, args, array_dists, loc):
        """analyze distributions of bodo distributed functions
        (bodo.libs.distributed_api.func_name)
        """

        if func_name == "parallel_print":
            return

        if func_name == "set_arr_local":
            return

        if func_name == "local_alloc_size":
            return

        if func_name == "rep_return":
            _set_var_dist(self.typemap, lhs, array_dists, Distribution.REP)
            _set_var_dist(self.typemap, args[0].name, array_dists, Distribution.REP)
            return

        if func_name == "dist_return":
            arr_name = args[0].name
            arr_typ = self.typemap[arr_name]
            assert is_distributable_typ(arr_typ) or is_distributable_tuple_typ(
                arr_typ
            ), f"Variable {arr_name} is not distributable since it is of type {arr_typ}"
            assert arr_name in array_dists, "array distribution not found"
            if is_REP(array_dists[arr_name]):
                raise BodoError(
                    f"distributed return of array {arr_name} not valid"
                    f" since it is replicated.\nDistributed Diagnostics: {self._get_diag_info_str()}",
                    loc,
                )
            array_dists[lhs] = array_dists[arr_name]
            return

        if func_name == "threaded_return":
            arr_name = args[0].name
            assert arr_name in array_dists, "array distribution not found"
            if is_REP(array_dists[arr_name]):
                raise BodoError(
                    "threaded return of array {} not valid since it is replicated",
                    loc,
                )
            array_dists[arr_name] = Distribution.Thread
            return

        if func_name == "rebalance":
            _meet_array_dists(self.typemap, lhs, args[0].name, array_dists)
            return

        if func_name == "random_shuffle":
            _meet_array_dists(self.typemap, lhs, args[0].name, array_dists)
            return

        # output is replicated but input can stay distributed
        if func_name == "get_chunk_bounds":
            _set_var_dist(self.typemap, lhs, array_dists, Distribution.REP)
            return

        # set REP if not found
        self._analyze_call_set_REP(
            lhs, args, array_dists, "bodo.libs.distributed_api." + func_name, loc
        )

    def _handle_dispatcher(self, dispatcher, lhs, rhs, array_dists):
        """handles Bodo function calls that have distributed flags.
        finds if input arguments and return value are marked as distributed and makes
        sure distributions are set properly.
        Also, recompiles the function if input distribution hints were wrong.
        """
        err_msg = (
            "variable '{}' is marked as distributed by '{}' but not possible to"
            " distribute in caller function '{}'.\nDistributed diagnostics:\n{}"
        )
        metadata = self._handle_dispatcher_args(
            dispatcher, lhs, rhs, array_dists, err_msg
        )

        # check return value for distributed flag
        is_return_distributed = metadata.get("is_return_distributed", False)
        # is_return_distributed is a list in tuple case which specifies distributions of
        # individual elements
        lhs_typ = self.typemap[lhs]
        if isinstance(is_return_distributed, list):
            # check distributions of tuple elements for errors
            for i, is_dist in enumerate(is_return_distributed):
                if is_dist and lhs in array_dists and is_REP(array_dists[lhs][i]):
                    raise BodoError(
                        err_msg.format(
                            lhs,
                            dispatcher.__name__,
                            self.func_ir.func_id.func_name,
                            self._get_diag_info_str(),
                        ),
                        rhs.loc,
                    )
            _set_var_dist(
                self.typemap,
                lhs,
                array_dists,
                _get_ret_new_dist(lhs_typ, is_return_distributed),
            )
        else:
            if (
                is_return_distributed
                and lhs in array_dists
                and is_REP(array_dists[lhs])
            ):
                raise BodoError(
                    err_msg.format(
                        lhs,
                        dispatcher.__name__,
                        self.func_ir.func_id.func_name,
                        self._get_diag_info_str(),
                    ),
                    rhs.loc,
                )
            _set_var_dist(
                self.typemap,
                lhs,
                array_dists,
                _get_ret_new_dist(lhs_typ, is_return_distributed),
            )

    def _handle_dispatcher_args(self, dispatcher, lhs, rhs, array_dists, err_msg):
        """
        Get distributed flags of inputs and raise error if caller passes REP for an
        argument explicitly specified as distributed.
        In automatic distribution detection case, recompile if dispatcher signature
        distributions are not compatible
        """

        dist_flag_vars = tuple(dispatcher.targetoptions.get("distributed", ())) + tuple(
            dispatcher.targetoptions.get("distributed_block", ())
        )
        rep_flag_vars = tuple(dispatcher.targetoptions.get("replicated", ()))
        dist_vars = []
        rep_vars = []
        rep_inds = {}

        # folds arguments and finds the ones that are flagged as distributed
        # folding arguments similar to:
        # https://github.com/numba/numba/blob/5f474010f8f50b3cf358125ba279d345ae5914ef/numba/core/dispatcher.py#L70
        def normal_handler(index, param, value):
            if param.name in dist_flag_vars:
                dist_vars.append(value.name)
            else:
                if param.name in rep_flag_vars:
                    rep_vars.append(value.name)
                rep_inds[index] = value.name
            return self.typemap[value.name]

        def default_handler(index, param, default):
            return types.Omitted(default)

        def stararg_handler(index, param, values):
            if param.name in dist_flag_vars:
                dist_vars.extend(v.name for v in values)
            else:
                if param.name in rep_flag_vars:
                    rep_vars.extend(v.name for v in values)
                rep_inds[index] = values
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

        # make sure variables marked distributed are not REP
        # otherwise, leave current distribution in place (1D or 1D_Var)
        for v in dist_vars:
            if is_REP(array_dists[v]):
                raise BodoError(
                    err_msg.format(
                        v,
                        dispatcher.__name__,
                        self.func_ir.func_id.func_name,
                        self._get_diag_info_str(),
                    ),
                    rhs.loc,
                )

        # check arguments not flagged as distributed and recompile with correct
        # distributions if necessary
        if arg_types not in dispatcher.overloads:
            # some other dispatcher may have changed variable distributions in types,
            # so overload signature may not match anymore and needs recompilation
            # see test_dist_type_change_multi_func2
            self._recompile_func(lhs, rhs, dispatcher.__name__)
        metadata = dispatcher.overloads[arg_types].metadata
        recompile = False
        new_arg_types = list(arg_types)
        for ind, vname in rep_inds.items():
            if metadata["args_maybe_distributed"]:
                if vname not in array_dists:
                    continue
                typ = arg_types[ind]
                if vname in rep_vars:
                    _set_REP(
                        self.typemap,
                        self.metadata,
                        self.diag_info,
                        vname,
                        array_dists,
                        f"replicated flag set for {vname}",
                        rhs.loc,
                    )
                    continue
                if not hasattr(typ, "dist"):
                    _set_REP(
                        self.typemap,
                        self.metadata,
                        self.diag_info,
                        vname,
                        array_dists,
                        f"input of another Bodo call without distributed flag (automatic distribution detection not supported for data type {typ})",
                        rhs.loc,
                    )
                    continue
                var_dist = array_dists[vname]
                # OneD can be passed as OneD_Var to avoid recompilation
                if not _is_compat_dist(typ.dist, var_dist):
                    new_typ = typ.copy(dist=var_dist)
                    new_arg_types[ind] = new_typ
                    self.typemap.pop(vname)
                    self.typemap[vname] = new_typ
                    recompile = True
            else:
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    vname,
                    array_dists,
                    "input of another Bodo call without distributed flag",
                    rhs.loc,
                )
        if recompile:
            arg_types = tuple(new_arg_types)
            self._recompile_func(lhs, rhs, dispatcher.__name__)

        metadata = dispatcher.overloads[arg_types].metadata
        return metadata

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

        if isinstance(in_type, bodo.types.NullableTupleType):
            nullable_tup_def = guard(get_definition, self.func_ir, args[0])
            assert (
                isinstance(nullable_tup_def, ir.Expr) and nullable_tup_def.op == "call"
            ), (
                "bodo.libs.array_kernels.concat only nullable tuples created with build_nullable_tuple"
            )
            fdef = find_callname(self.func_ir, nullable_tup_def, self.typemap)
            assert fdef == (
                "build_nullable_tuple",
                "bodo.libs.nullable_tuple_ext",
            ), (
                "bodo.libs.array_kernels.concat only nullable tuples created with build_nullable_tuple"
            )
            tup_val = nullable_tup_def.args[0]
        else:
            tup_val = args[0]

        tup_def = guard(get_definition, self.func_ir, tup_val)
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
        if out_dist == Distribution.REP:
            in_dist = out_dist

        # If any of input arguments are replicated, then all inputs must be replicated
        if in_dist == Distribution.REP:
            for v in in_arrs:
                array_dists[v.name] = Distribution.REP

        return

    def _analyze_call_np_dot(self, lhs, args, array_dists, loc):
        arg0 = args[0].name
        arg1 = args[1].name
        ndim0 = self.typemap[arg0].ndim
        ndim1 = self.typemap[arg1].ndim
        t0 = guard(_is_transposed_array, self.func_ir, arg0)
        t1 = guard(_is_transposed_array, self.func_ir, arg1)
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
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                arg1,
                array_dists,
                "vector multiplied by matrix rows in np.dot()",
                loc,
            )
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            # lhs and X have same distribution
            _meet_array_dists(self.typemap, lhs, arg0, array_dists)
            dprint("dot case 1 Xw:", arg0, arg1)
            return
        if ndim0 == 1 and ndim1 == 2 and not t1:
            # reduction across samples np.dot(Y,X)
            # lhs is always REP
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                lhs,
                array_dists,
                "output of vector-matrix multiply in np.dot()",
                loc,
            )
            # Y and X have same distribution
            _meet_array_dists(self.typemap, arg0, arg1, array_dists)
            dprint("dot case 2 YX:", arg0, arg1)
            return
        if ndim0 == 2 and ndim1 == 2 and t0 and not t1:
            # reduction across samples np.dot(X.T,Y)
            # lhs is always REP
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                lhs,
                array_dists,
                "output of matrix-matrix multiply in np.dot()",
                loc,
            )
            # Y and X have same distribution
            _meet_array_dists(self.typemap, arg0, arg1, array_dists)
            dprint("dot case 3 XtY:", arg0, arg1)
            return
        if ndim0 == 2 and ndim1 == 2 and not t0 and not t1:
            # samples dot weights: np.dot(X,w)
            # w is always REP
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                arg1,
                array_dists,
                "matrix multiplied by rows in np.dot()",
                loc,
            )
            _meet_array_dists(self.typemap, lhs, arg0, array_dists)
            dprint("dot case 4 Xw:", arg0, arg1)
            return

        # set REP if no pattern matched
        self._analyze_call_set_REP(lhs, args, array_dists, "np.dot", loc)

    def _analyze_call_set_REP(self, lhs, args, array_dists, fdef, loc):
        arrs = []
        for v in args:
            typ = self.typemap[v.name]
            if is_distributable_typ(typ) or is_distributable_tuple_typ(typ):
                dprint(f"dist setting call arg REP {v.name} in {fdef}")
                _set_REP(
                    self.typemap, self.metadata, self.diag_info, v.name, array_dists
                )
                arrs.append(v.name)
        typ = self.typemap[lhs]
        if is_distributable_typ(typ) or is_distributable_tuple_typ(typ):
            dprint(f"dist setting call out REP {lhs} in {fdef}")
            _set_REP(self.typemap, self.metadata, self.diag_info, lhs, array_dists)
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
                "Distributed analysis set '{}' as replicated due "
                "to call to function '{}' (unsupported function or usage)"
            ).format(
                ", ".join(f"'{_get_user_varname(self.metadata, a)}'" for a in arrs),
                fname,
            )
            _add_diag_info(self.diag_info, info, loc)

    def _analyze_getitem_array_table_inputs(
        self,
        inst,
        lhs,
        in_var: ir.Var,
        index_var: ir.Var,
        rhs_loc,
        equiv_set,
        array_dists,
    ):
        """analyze getitem nodes for arrays/tables
        having determined the the variable and index value."""

        if (in_var.name, index_var.name) in self._parallel_accesses:
            # XXX: is this always valid? should be done second pass?
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                [inst.target],
                array_dists,
                "output of distributed getitem is REP",
                rhs_loc,
            )
            return

        # in multi-dimensional case, we only consider first dimension
        # TODO: extend to 2D distribution
        tup_list = guard(find_build_tuple, self.func_ir, index_var)
        if tup_list is not None:
            index_var = tup_list[0]
            # rest of indices should be replicated if array
            other_ind_vars = tup_list[1:]
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                other_ind_vars,
                array_dists,
                "getitem index variables are REP",
                rhs_loc,
            )

        assert isinstance(index_var, ir.Var)
        index_typ = self.typemap[index_var.name]

        # array selection with boolean index
        if (
            is_np_array_typ(index_typ)
            and index_typ.dtype == types.boolean
            or index_typ == boolean_array_type
        ):
            # input array and bool index have the same distribution
            new_dist = _meet_array_dists(
                self.typemap, index_var.name, in_var.name, array_dists
            )
            out_dist = Distribution.OneD_Var
            if lhs in array_dists:
                out_dist = Distribution(min(out_dist.value, array_dists[lhs].value))
            out_dist = Distribution(min(out_dist.value, new_dist.value))
            array_dists[lhs] = out_dist
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                _meet_array_dists(
                    self.typemap, index_var.name, in_var.name, array_dists, out_dist
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
            _meet_array_dists(self.typemap, lhs, in_var.name, array_dists)
            return
        # chunked slice or strided slice can be 1D_Var
        # examples: A = X[:n//3], A = X[::2,5]
        elif isinstance(index_typ, types.SliceType):
            # output array is 1D_Var if input array is distributed
            out_dist = Distribution.OneD_Var
            if lhs in array_dists:
                out_dist = _min_dist(out_dist, array_dists[lhs])
            out_dist = _min_dist(out_dist, array_dists[in_var.name])
            array_dists[lhs] = out_dist
            # input can become REP
            if out_dist != Distribution.OneD_Var:
                array_dists[in_var.name] = out_dist
            return

        # avoid parallel scalar getitem when inside a parfor
        # examples: test_np_dot, logistic_regression_rand
        if self.in_parallel_parfor != -1:
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                inst.list_vars(),
                array_dists,
                "getitem inside parallel loop is REP",
                rhs_loc,
            )
            return

        # int index of dist array
        if isinstance(index_typ, types.Integer):
            if is_distributable_typ(self.typemap[lhs]):
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    lhs,
                    array_dists,
                    "output of distributed getitem with int index is REP",
                    rhs_loc,
                )
            return

        _set_REP(
            self.typemap,
            self.metadata,
            self.diag_info,
            inst.list_vars(),
            array_dists,
            "unsupported getitem distribution",
            rhs_loc,
        )

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
                    _set_var_dist(self.typemap, tup, array_dists, Distribution.OneD)
                if lhs not in array_dists:
                    _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD)
                new_dist = _min_dist(array_dists[tup][ind_val], array_dists[lhs])
                array_dists[tup][ind_val] = new_dist
                array_dists[lhs] = new_dist
            return

        # getitem on list/dictionary of distributed values
        if isinstance(in_typ, (types.List, types.DictType)) and (
            is_distributable_typ(lhs_typ) or is_distributable_tuple_typ(lhs_typ)
        ):
            # output and dictionary have the same distribution
            _meet_array_dists(self.typemap, lhs, rhs.value.name, array_dists)
            return

        # indexing into arrays from this point only, check for array type
        if not (is_array_typ(in_typ) or isinstance(in_typ, TableType)):
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                inst.list_vars(),
                array_dists,
                "getitem input not array",
                rhs.loc,
            )
            return

        self._analyze_getitem_array_table_inputs(
            inst, lhs, in_var, index_var, rhs.loc, equiv_set, array_dists
        )

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
            _meet_array_dists(self.typemap, arr.name, inst.value.name, array_dists)
            return

        if (arr.name, index_var.name) in self._parallel_accesses:
            # no parallel to parallel array set (TODO)
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                [inst.value],
                array_dists,
                "value set in distributed setitem is REP",
                inst.loc,
            )
            return

        tup_list = guard(find_build_tuple, self.func_ir, index_var)
        if tup_list is not None:
            index_var = tup_list[0]
            # rest of indices should be replicated if array
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                tup_list[1:],
                array_dists,
                "index variables are REP",
                inst.loc,
            )

        # array selection with boolean index
        if (
            is_np_array_typ(index_typ)
            and index_typ.dtype == types.boolean
            or index_typ == boolean_array_type
        ):
            # setting scalar or lower dimension value, e.g. A[B] = 1
            if not is_array_typ(value_typ) or value_typ.ndim < target_typ.ndim:
                # input array and bool index have the same distribution
                _meet_array_dists(self.typemap, arr.name, index_var.name, array_dists)
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    [inst.value],
                    array_dists,
                    "scalar/lower-dimension value set in distributed setitem with bool array index is REP",
                    inst.loc,
                )
                return
            # TODO: support bool index setitem across the whole first dimension, which
            # may require shuffling data to match bool index selection

        # whole slice access, output has same distribution as input
        # for example: X[:,3] = A
        if guard(is_whole_slice, self.typemap, self.func_ir, index_var) or guard(
            is_slice_equiv_arr, arr, index_var, self.func_ir, equiv_set
        ):
            _meet_array_dists(self.typemap, arr.name, inst.value.name, array_dists)
            return
        # chunked slice or strided slice
        # examples: X[:n//3] = v, X[::2,5] = v
        elif isinstance(index_typ, types.SliceType):
            # if the value is scalar/lower dimension
            if not is_array_typ(value_typ) or value_typ.ndim < target_typ.ndim:
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    [inst.value],
                    array_dists,
                    "scalar/lower-dimension value set in distributed setitem with slice index is REP",
                    inst.loc,
                )
                return
            # TODO: support slice index setitem across the whole first dimension, which
            # may require shuffling data to match slice index selection

        # avoid parallel scalar setitem when inside a parfor
        if self.in_parallel_parfor != -1:
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                inst.list_vars(),
                array_dists,
                "setitem inside parallel loop is REP",
                inst.loc,
            )
            return

        # int index setitem of dist array
        if isinstance(index_typ, types.Integer):
            _set_REP(
                self.typemap,
                self.metadata,
                self.diag_info,
                [inst.value],
                array_dists,
                "value set in distributed array setitem is REP",
                inst.loc,
            )
            return

        # Array boolean idx setitem with scalar value, e.g. A[cond] = val
        if (
            is_array_typ(target_typ)
            and is_array_typ(index_typ)
            and index_typ.dtype == types.bool_
            and not is_array_typ(value_typ)
        ):
            _meet_array_dists(self.typemap, arr.name, index_var.name, array_dists)
            return

        _set_REP(
            self.typemap,
            self.metadata,
            self.diag_info,
            [inst.value, arr, index_var],
            array_dists,
            "unsupported setitem distribution",
            inst.loc,
        )

    def _analyze_arg(self, lhs, rhs, array_dists):
        """analyze ir.Arg/Global/FreeVar/Const nodes for distribution.
        Checks for user flags; sets to REP if no user flag found
        """
        is_arg = isinstance(rhs, ir.Arg)
        # Check replicated flag first since it takes precedence over others (spawner
        # sets all_args_distributed_block=True but user specified replicated should
        # take precedence)
        if rhs.name in self.metadata["replicated"]:
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.REP)
        elif rhs.name in self.metadata["distributed_block"] or (
            is_arg and self.flags.all_args_distributed_block
        ):
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD)
            # Check if a user specified argument is indicated distributed
            # but transitions to REP. Fixed point iteration dictates that we will
            # eventually fail this check if an argument is ever changed to REP.
            self._check_user_distributed_args(array_dists, lhs, rhs.loc)
        elif rhs.name in self.metadata["distributed"] or (
            is_arg and self.flags.all_args_distributed_varlength
        ):
            if lhs not in array_dists:
                _set_var_dist(self.typemap, lhs, array_dists, Distribution.OneD_Var)
            # Check if a user specified argument is indicated distributed
            # but transitions to REP. Fixed point iteration dictates that we will
            # eventually fail this check if an argument is ever changed to REP.
            self._check_user_distributed_args(array_dists, lhs, rhs.loc)
        elif rhs.name in self.metadata["threaded"]:
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.Thread
        else:
            typ = self.typemap[lhs]
            # get distribution info from data type if available
            # argument handling sets distribution from Bodo metadata in df/... objects
            if (
                self.flags.args_maybe_distributed
                and hasattr(typ, "dist")
                and typ.dist != Distribution.REP
            ):
                array_dists[lhs] = typ.dist
                self._check_user_distributed_args(array_dists, lhs, rhs.loc)
                return

            if is_distributable_typ(typ) or is_distributable_tuple_typ(typ):
                dprint("replicated input ", rhs.name, lhs)
                info = (
                    "Distributed analysis replicated {2} '{0}' (variable "
                    "'{1}'). Set distributed flag for '{0}' if distributed partitions "
                    "are passed (e.g. @bodo.jit(distributed=['{0}']))."
                ).format(rhs.name, lhs, "argument" if is_arg else "global value")
                _set_REP(
                    self.typemap,
                    self.metadata,
                    self.diag_info,
                    lhs,
                    array_dists,
                    info,
                    rhs.loc,
                )

    def _analyze_setattr(self, target, attr, value, array_dists, loc):
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
            ), (
                f"Variable {value.name} is not distributable since it is of type {val_type} (required for setting class field)"
            )
            assert value.name in array_dists, "array distribution not found"
            val_dist = array_dists[value.name]
            # value shouldn't have a more restrictive distribution than the dist spec
            # e.g. REP vs OneD
            if val_dist.value < attr_dist.value:
                raise BodoError(
                    f"distribution of value is not compatible with the class attribute"
                    f" distribution spec of {target_type.class_type.class_name} in"
                    f" {target.name}.{attr} = {value.name}",
                    loc,
                )
            return

        _set_REP(
            self.typemap,
            self.metadata,
            self.diag_info,
            [target, value],
            array_dists,
            "unsupported SetAttr",
            loc,
        )

    def _analyze_return(self, var, array_dists, loc):
        """analyze ir.Return nodes for distribution. Checks for user flags; sets to REP
        if no user flag found"""
        if self.flags.returns_maybe_distributed:
            # no need to update metadata if the returned variable is not distributable
            if var.name not in array_dists:
                typ = self.typemap[var.name]
                assert not (
                    is_distributable_typ(typ) or is_distributable_tuple_typ(typ)
                ), (
                    "Internal error: distributable type does not have assigned distribution at return"
                )
                return
            is_1D_or_1D_Var = lambda d: d in (Distribution.OneD, Distribution.OneD_Var)
            ret_dist = array_dists[var.name]
            # untyped pass usually sets is_return_distributed but in this case
            # distributions are not known until here, see _handle_dispatcher() for usage
            self.metadata["is_return_distributed"] = (
                [is_1D_or_1D_Var(d) for d in ret_dist]
                if isinstance(ret_dist, list)
                else is_1D_or_1D_Var(ret_dist)
            )
            return

        if self._is_dist_return_var(var):
            return

        # in case of tuple return, individual variables may be flagged separately
        try:
            vdef = get_definition(self.func_ir, var)
            require(is_expr(vdef, "cast"))
            dcall = get_definition(self.func_ir, vdef.value)
            require(is_expr(dcall, "build_tuple"))
            for v in dcall.items:
                self._analyze_return(v, array_dists, loc)
            return
        except GuardException:
            pass

        info = (
            "Distributed analysis replicated return variable "
            f"'{var.name}'. Set distributed flag for the original variable if distributed "
            "partitions should be returned."
        )
        _set_REP(
            self.typemap, self.metadata, self.diag_info, [var], array_dists, info, loc
        )

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
                find_callname(self.func_ir, dcall, self.typemap)
                == ("dist_return", "bodo.libs.distributed_api")
            )
            return True
        except GuardException:
            return False

    def _get_calc_n_items_range_index(self, size_def):
        """match RangeIndex calc_nitems(r._start, r._stop, r._step) call and return r"""
        require(
            find_callname(self.func_ir, size_def, self.typemap)
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
            fdef = guard(find_callname, self.func_ir, var_def, self.typemap)
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

    def _recompile_func(self, lhs, rhs, fname):
        """Recompile function call due to distribution change in input data types"""
        _add_diag_info(
            self.diag_info,
            f"Recompiling {fname} since argument data distribution changed after analysis",
            rhs.loc,
        )
        self.calltypes.pop(rhs)
        self.calltypes[rhs] = self.typemap[rhs.func.name].get_call_type(
            self.typingctx,
            tuple(self.typemap[v.name] for v in rhs.args),
            {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()},
        )
        # We need to update the output datatype, as for certain types (DataFrame),
        # The distribution is an field of the type, and recompilation can result
        # in changed data distribution
        self.typemap.pop(lhs)
        self.typemap[lhs] = self.calltypes[rhs].return_type

    def _get_diag_info_str(self):
        """returns all diagnostics info and their locations as a string"""
        return "\n".join(f"{info}\n{loc.strformat()}" for (info, loc) in self.diag_info)


def _set_REP(typemap, metadata, diag_info, var_list, array_dists, info=None, loc=None):
    """set distribution of all variables in 'var_list' to REP if distributable."""
    if isinstance(var_list, (str, ir.Var)):
        var_list = [var_list]
    for var in var_list:
        varname = var.name if isinstance(var, ir.Var) else var
        # Handle SeriesType since it comes from Arg node and it could
        # have user-defined distribution
        typ = typemap[varname]
        if is_distributable_typ(typ) or is_distributable_tuple_typ(typ):
            dprint(f"dist setting REP {varname}")
            # keep diagnostics info if the distribution is changing to REP and extra
            # info is available
            if (
                varname not in array_dists or not is_REP(array_dists[varname])
            ) and info is not None:
                info = (
                    f"Setting distribution of variable '{_get_user_varname(metadata, varname)}' to REP: "
                    + info
                )
                _add_diag_info(diag_info, info, loc)
            _set_var_dist(typemap, varname, array_dists, Distribution.REP)


def _add_diag_info(diag_info, info, loc):
    """append diagnostics info to be displayed in distributed diagnostics output"""
    if (info, loc) not in diag_info:
        diag_info.append((info, loc))


def _get_user_varname(metadata, v):
    """get original variable name by user for diagnostics info if possible"""
    if v in metadata["parfors"]["var_rename_map"]:
        return metadata["parfors"]["var_rename_map"][v]
    return v


def _meet_array_dists(typemap, arr1: str, arr2: str, array_dists, top_dist=None):
    """meet distributions of arrays for consistent distribution"""

    if top_dist is None:
        top_dist = Distribution.OneD
    if arr1 not in array_dists:
        _set_var_dist(typemap, arr1, array_dists, top_dist, False)
    if arr2 not in array_dists:
        _set_var_dist(typemap, arr2, array_dists, top_dist, False)

    new_dist = _min_dist(array_dists[arr1], array_dists[arr2])
    new_dist = _min_dist_top(new_dist, top_dist)
    array_dists[arr1] = new_dist
    array_dists[arr2] = new_dist
    return new_dist


def _meet_several_array_dists(typemap, iterable_of_arrs, array_dists, top_dist=None):
    """meet distributions of arrays for consistent distribution"""

    if top_dist is None:
        top_dist = Distribution.OneD

    new_dist = top_dist
    for arr in iterable_of_arrs:
        if arr not in array_dists:
            _set_var_dist(typemap, arr, array_dists, top_dist, False)

        new_dist = _min_dist(new_dist, array_dists[arr])

    for arr in iterable_of_arrs:
        array_dists[arr] = new_dist

    return new_dist


def _get_var_dist(varname, array_dists, typemap):
    if varname not in array_dists:
        _set_var_dist(typemap, varname, array_dists, Distribution.OneD, False)
    return array_dists[varname]


def _set_var_dist(typemap, varname: str, array_dists, dist, check_type=True):
    # some non-distributable types could need to be assigned distribution
    # sometimes, e.g. SeriesILocType. check_type=False handles these cases.
    typ = typemap[varname]
    dist = _get_dist(typ, dist)
    # TODO: use proper "FullRangeIndex" type
    if not check_type or (is_distributable_typ(typ) or is_distributable_tuple_typ(typ)):
        array_dists[varname] = dist


def _get_dist(typ, dist):
    """get proper distribution value for type. Returns list of distributions for
    tuples (but just the input 'dist' otherwise).
    """

    if is_distributable_tuple_typ(typ):
        if isinstance(typ, types.iterators.EnumerateType):
            typ = typ.yield_type[1]
            # Enumerate is always Tuple(integer, value_type)
            return [None, _get_dist(typ, dist)]
        if isinstance(typ, types.List):
            typ = typ.dtype
        if is_bodosql_context_type(typ):
            typs = typ.dataframes
        else:
            typs = typ.types
        if not isinstance(dist, list):
            dist = [dist] * len(typs)
        return [
            (
                _get_dist(t, dist[i])
                if (is_distributable_typ(t) or is_distributable_tuple_typ(t))
                else None
            )
            for i, t in enumerate(typs)
        ]
    return dist


def _min_dist(dist1, dist2):
    if isinstance(dist1, list):
        assert len(dist1) == len(dist2)
        n = len(dist1)
        return [
            _min_dist(dist1[i], dist2[i]) if dist1[i] is not None else None
            for i in range(n)
        ]
    return Distribution(min(dist1.value, dist2.value))


def _min_dist_top(dist, top_dist):
    if isinstance(dist, list):
        n = len(dist)
        return [
            _min_dist_top(dist[i], top_dist) if dist[i] is not None else None
            for i in range(n)
        ]
    return Distribution(min(dist.value, top_dist.value))


def _update_type_dist(typ, dist):
    """update distribution value of typ if typ has a 'dist' attribute"""
    # TODO(ehsan): need to handle list/dict of tuple cases?
    if isinstance(dist, list) and isinstance(typ, types.BaseTuple):
        return types.BaseTuple.from_types(
            [_update_type_dist(t, dist[i]) for i, t in enumerate(typ.types)]
        )
    # TODO: make sure all distributable types have a 'dist' attribute
    if hasattr(typ, "dist"):
        typ = typ.copy(dist=dist)
    return typ


def _get_ret_new_dist(lhs_typ, is_return_distributed):
    """get new distribution of return value based on is_return_distributed flag"""
    if isinstance(is_return_distributed, list):
        return [
            _get_ret_new_dist(lhs_typ.types[i], v)
            for i, v in enumerate(is_return_distributed)
        ]
    new_dist = Distribution.REP
    if is_return_distributed:
        new_dist = Distribution.OneD_Var
        if hasattr(lhs_typ, "dist"):
            new_dist = lhs_typ.dist
    return new_dist


def _is_1D_or_1D_Var_arr(arr_name, array_dists):
    """return True if arr_name is either 1D or 1D_Var distributed"""
    return arr_name in array_dists and array_dists[arr_name] in (
        Distribution.OneD,
        Distribution.OneD_Var,
    )


def _is_compat_dist(dist1, dist2):
    """return True if 'dist1' and 'dist2' are compatible: they are the same or
    'dist1' is OneD_var and 'dist2' is OneD ('dist2' can be used in places that
    require 'dist1')
    """
    return dist1 == dist2 or (
        dist1 == Distribution.OneD_Var and dist2 == Distribution.OneD
    )


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
            return Reduce_Type.Bit_Or
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
    fdef = find_callname(func_ir, reduce_nodes[-2].value, typemap)
    if fdef == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
        arg_def = get_definition(func_ir, reduce_func_call.args[0])
        require(is_expr(arg_def, "build_tuple"))
        require(len(arg_def.items) > 0)
        reduce_func_call = get_definition(func_ir, arg_def.items[0])

    if fdef == ("init_series", "bodo.hiframes.pd_series_ext"):
        reduce_func_call = get_definition(func_ir, reduce_func_call.args[0])

    require(
        find_callname(func_ir, reduce_func_call, typemap)
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

    # Accesses consist of 3 values:
    #  - Name of the array
    #  - Name of the index
    #  - bool is_bitwise_operation

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
                        if fdef == ("get_str_arr_str_length", "bodo.libs.str_arr_ext"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                        if fdef == ("inplace_eq", "bodo.libs.str_arr_ext"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                        if fdef == ("copy_array_element", "bodo.libs.array_kernels"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                            accesses.add((rhs.args[2].name, rhs.args[3].name, False))
                        if fdef == ("get_str_arr_item_copy", "bodo.libs.str_arr_ext"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
                            accesses.add((rhs.args[2].name, rhs.args[3].name, False))
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
                        if fdef == ("scalar_optional_getitem", "bodo.utils.indexing"):
                            accesses.add((rhs.args[0].name, rhs.args[1].name, False))
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


def dprint(*s):  # pragma: no cover
    if debug_prints():
        print(*s)


def propagate_assign(array_dists: dict[str, Distribution], nodes: list[ir.Stmt]):
    """Function that updates any assignments in a list of nodes
    with matching array distributions for any existing variables.
    This is run when replacing functions after distributed analysis
    has finished, so there is only 1 pass in the forward direction.

    Args:
        array_dists (dict[str, Distribution]): distributed analysis for each variable
        nodes (List[ir.Stmt]): List of IR nodes to check.
    """
    for node in nodes:
        if isinstance(node, ir.Assign) and isinstance(node.value, ir.Var):
            lhs = node.target.name
            rhs = node.value.name
            if lhs in array_dists and rhs not in array_dists:
                array_dists[rhs] = array_dists[lhs]
            elif rhs in array_dists and lhs not in array_dists:
                array_dists[lhs] = array_dists[rhs]
    return


def _is_transposed_array(func_ir, arr):
    """Return True if input is a transposed array using arr.T expression.
    Returns False or raises GuardException if not.
    """
    arr_def = get_definition(func_ir, arr)
    require(is_expr(arr_def, "getattr"))
    # TODO[BSE-2376]: support other transpose forms (np.transpose, arr.transpose)
    return arr_def.attr == "T"
