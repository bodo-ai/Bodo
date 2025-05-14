import typing as pt
from collections.abc import Callable, Iterable
from contextlib import contextmanager

import pandas as pd
import pyarrow as pa
from pandas._typing import AnyArrayLike, IndexLabel, MergeHow, MergeValidate, Suffixes

import bodo
from bodo.ext import plan_optimizer
from bodo.pandas.array_manager import LazyArrayManager
from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.lazy_wrapper import BodoLazyWrapper, ExecState
from bodo.pandas.managers import LazyBlockManager, LazyMetadataMixin
from bodo.pandas.series import BodoSeries
from bodo.pandas.utils import (
    BodoLibNotImplementedException,
    LazyPlan,
    LazyPlanDistributedArg,
    arrow_to_empty_df,
    check_args_fallback,
    get_lazy_manager_class,
    get_n_index_arrays,
    get_proj_expr_single,
    is_single_projection,
    make_col_ref_exprs,
    wrap_plan,
)
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    get_overload_const_str,
    is_overload_none,
)


class BodoDataFrame(pd.DataFrame, BodoLazyWrapper):
    # We need to store the head_df to avoid data pull when head is called.
    # Since BlockManagers are in Cython it's tricky to override all methods
    # so some methods like head will still trigger data pull if we don't store head_df and
    # use it directly when available.
    _head_df: pd.DataFrame | None = None
    _source_plan: LazyPlan | None = None

    @property
    def _plan(self):
        if self.is_lazy_plan():
            return self._mgr._plan
        else:
            """We can't create a new LazyPlan each time that _plan is called
               because filtering checks that the projections that are part of
               the filter all come from the same source and if you create a
               new LazyPlan here each time then they will appear as different
               sources.  We sometimes use a pandas manager which doesn't have
               _source_plan so we have to do getattr check.
            """
            if getattr(self, "_source_plan", None) is not None:
                return self._source_plan

            from bodo.pandas.base import _empty_like

            empty_data = _empty_like(self)
            if bodo.dataframe_library_run_parallel:
                if getattr(self._mgr, "_md_result_id", None) is not None:
                    # If the plan has been executed but the results are still
                    # distributed then re-use those results as is.
                    res_id = self._mgr._md_result_id
                    mgr = self._mgr
                else:
                    # The data has been collected and is no longer distributed
                    # so we need to re-distribute the results.
                    res_id = bodo.spawn.utils.scatter_data(self)
                    mgr = None
                self._source_plan = LazyPlan(
                    "LogicalGetPandasReadParallel",
                    empty_data,
                    LazyPlanDistributedArg(mgr, res_id),
                )
            else:
                self._source_plan = LazyPlan(
                    "LogicalGetPandasReadSeq", empty_data, self
                )

            return self._source_plan

    @staticmethod
    def from_lazy_mgr(
        lazy_mgr: LazyArrayManager | LazyBlockManager,
        head_df: pd.DataFrame | None,
    ):
        """
        Create a BodoDataFrame from a lazy manager and possibly a head_df.
        If you want to create a BodoDataFrame from a pandas manager use _from_mgr
        """
        df = BodoDataFrame._from_mgr(lazy_mgr, [])
        df._head_df = head_df
        return df

    @classmethod
    def from_lazy_metadata(
        cls,
        lazy_metadata: LazyMetadata,
        collect_func: Callable[[str], pt.Any] | None = None,
        del_func: Callable[[str], None] | None = None,
        plan: plan_optimizer.LogicalOperator | None = None,
    ) -> "BodoDataFrame":
        """
        Create a BodoDataFrame from a lazy metadata object.
        """
        assert isinstance(lazy_metadata.head, pd.DataFrame)
        lazy_mgr = get_lazy_manager_class()(
            None,
            None,
            result_id=lazy_metadata.result_id,
            nrows=lazy_metadata.nrows,
            head=lazy_metadata.head._mgr,
            collect_func=collect_func,
            del_func=del_func,
            index_data=lazy_metadata.index_data,
            plan=plan,
        )
        return cls.from_lazy_mgr(lazy_mgr, lazy_metadata.head)

    def update_from_lazy_metadata(self, lazy_metadata: LazyMetadata):
        """
        Update the dataframe with new metadata.
        """
        assert self._lazy
        assert isinstance(lazy_metadata.head, pd.DataFrame)
        # Call delfunc to delete the old data.
        self._mgr._del_func(self._mgr._md_result_id)
        self._head_df = lazy_metadata.head
        self._mgr._md_nrows = lazy_metadata.nrows
        self._mgr._md_result_id = lazy_metadata.result_id
        self._mgr._md_head = lazy_metadata.head._mgr

    def is_lazy_plan(self):
        """Returns whether the BodoDataFrame is represented by a plan."""
        return getattr(self._mgr, "_plan", None) is not None

    def execute_plan(self):
        if self.is_lazy_plan() and not self._mgr._disable_collect:
            return self._mgr.execute_plan()

    def head(self, n: int = 5):
        """
        Return the first n rows. If head_df is available and larger than n, then use it directly.
        Otherwise, use the default head method which will trigger a data pull.
        """
        # Prevent infinite recursion when called from _empty_like and in general
        # data is never required for head(0) so making a plan is never necessary.
        if n == 0:
            if self._exec_state == ExecState.COLLECTED:
                return self.iloc[:0].copy()
            else:
                assert self._head_df is not None
                return self._head_df.head(0).copy()

        if (self._head_df is None) or (n > self._head_df.shape[0]):
            if bodo.dataframe_library_enabled and isinstance(
                self._mgr, LazyMetadataMixin
            ):
                from bodo.pandas.base import _empty_like

                planLimit = LazyPlan(
                    "LogicalLimit",
                    _empty_like(self),
                    self._plan,
                    n,
                )

                return wrap_plan(planLimit)
            else:
                return super().head(n)
        else:
            # If head_df is available and larger than n, then use it directly.
            return self._head_df.head(n)

    def __len__(self):
        self.execute_plan()
        if self._lazy:
            return self._mgr._md_nrows
        return super().__len__()

    @property
    def index(self):
        self.execute_plan()
        return super().index

    @index.setter
    def index(self, value):
        self.execute_plan()
        super()._set_axis(1, value)

    @property
    def shape(self):
        self.execute_plan()
        if self._lazy:
            return self._mgr._md_nrows, len(self._head_df.columns)
        return super().shape

    def to_parquet(
        self,
        path,
        engine="auto",
        compression="snappy",
        index=None,
        partition_cols=None,
        storage_options=None,
        row_group_size=-1,
    ):
        # argument defaults should match that of to_parquet_overload in pd_dataframe_ext.py

        @bodo.jit(spawn=True)
        def to_parquet_wrapper(
            df: pd.DataFrame,
            path,
            engine,
            compression,
            index,
            partition_cols,
            storage_options,
            row_group_size,
        ):
            return df.to_parquet(
                path,
                engine,
                compression,
                index,
                partition_cols,
                storage_options,
                row_group_size,
            )

        # checks string arguments before jit performs conversion to unicode
        if not is_overload_none(engine) and get_overload_const_str(engine) not in (
            "auto",
            "pyarrow",
        ):  # pragma: no cover
            raise BodoError("DataFrame.to_parquet(): only pyarrow engine supported")

        if not is_overload_none(compression) and get_overload_const_str(
            compression
        ) not in {"snappy", "gzip", "brotli"}:
            raise BodoError(
                "to_parquet(): Unsupported compression: "
                + get_overload_const_str(compression)
            )

        return to_parquet_wrapper(
            self,
            path,
            engine,
            compression,
            index,
            partition_cols,
            storage_options,
            row_group_size,
        )

    def _get_result_id(self) -> str | None:
        if isinstance(self._mgr, LazyMetadataMixin):
            return self._mgr._md_result_id
        return None

    def to_sql(
        self,
        name,
        con,
        schema=None,
        if_exists="fail",
        index=True,
        index_label=None,
        chunksize=None,
        dtype=None,
        method=None,
    ):
        # argument defaults should match that of to_sql_overload in pd_dataframe_ext.py
        @bodo.jit(spawn=True)
        def to_sql_wrapper(
            df: pd.DataFrame,
            name,
            con,
            schema,
            if_exists,
            index,
            index_label,
            chunksize,
            dtype,
            method,
        ):
            return df.to_sql(
                name,
                con,
                schema,
                if_exists,
                index,
                index_label,
                chunksize,
                dtype,
                method,
            )

        return to_sql_wrapper(
            self,
            name,
            con,
            schema,
            if_exists,
            index,
            index_label,
            chunksize,
            dtype,
            method,
        )

    def to_csv(
        self,
        path_or_buf=None,
        sep=",",
        na_rep="",
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        mode="w",
        encoding=None,
        compression=None,
        quoting=None,
        quotechar='"',
        lineterminator=None,
        chunksize=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal=".",
        errors="strict",
        storage_options=None,
    ):
        # argument defaults should match that of to_csv_overload in pd_dataframe_ext.py

        @bodo.jit(spawn=True)
        def to_csv_wrapper(
            df: pd.DataFrame,
            path_or_buf,
            sep=sep,
            na_rep=na_rep,
            float_format=float_format,
            columns=columns,
            header=header,
            index=index,
            index_label=index_label,
            compression=compression,
            quoting=quoting,
            quotechar=quotechar,
            lineterminator=lineterminator,
            chunksize=chunksize,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            decimal=decimal,
        ):
            return df.to_csv(
                path_or_buf=path_or_buf,
                sep=sep,
                na_rep=na_rep,
                float_format=float_format,
                columns=columns,
                header=header,
                index=index,
                index_label=index_label,
                compression=compression,
                quoting=quoting,
                quotechar=quotechar,
                lineterminator=lineterminator,
                chunksize=chunksize,
                date_format=date_format,
                doublequote=doublequote,
                escapechar=escapechar,
                decimal=decimal,
                _bodo_concat_str_output=True,
            )

        # checks string arguments before jit performs conversion to unicode
        # checks should match that of to_csv_overload in pd_dataframe_ext.py
        check_unsupported_args(
            "BodoDataFrame.to_csv",
            {
                "encoding": encoding,
                "mode": mode,
                "errors": errors,
                "storage_options": storage_options,
            },
            {
                "encoding": None,
                "mode": "w",
                "errors": "strict",
                "storage_options": None,
            },
            package_name="pandas",
            module_name="IO",
        )

        return to_csv_wrapper(
            self,
            path_or_buf,
            sep=sep,
            na_rep=na_rep,
            float_format=float_format,
            columns=columns,
            header=header,
            index=index,
            index_label=index_label,
            compression=compression,
            quoting=quoting,
            quotechar=quotechar,
            lineterminator=lineterminator,
            chunksize=chunksize,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            decimal=decimal,
        )

    def to_json(
        self,
        path_or_buf=None,
        orient="records",
        date_format=None,
        double_precision=10,
        force_ascii=True,
        date_unit="ms",
        default_handler=None,
        lines=True,
        compression="infer",
        index=None,
        indent=None,
        storage_options=None,
        mode="w",
    ):
        # Argument defaults should match that of to_json_overload in pd_dataframe_ext.py
        # Passing orient and lines as free vars to become literals in the compiler

        @bodo.jit(spawn=True)
        def to_json_wrapper(
            df: pd.DataFrame,
            path_or_buf,
            date_format=date_format,
            double_precision=double_precision,
            force_ascii=force_ascii,
            date_unit=date_unit,
            default_handler=default_handler,
            compression=compression,
            index=index,
            indent=indent,
            storage_options=storage_options,
            mode=mode,
        ):
            return df.to_json(
                path_or_buf,
                orient=orient,
                date_format=date_format,
                double_precision=double_precision,
                force_ascii=force_ascii,
                date_unit=date_unit,
                default_handler=default_handler,
                lines=lines,
                compression=compression,
                index=index,
                indent=indent,
                storage_options=storage_options,
                mode=mode,
                _bodo_concat_str_output=True,
            )

        return to_json_wrapper(
            self,
            path_or_buf,
            date_format=date_format,
            double_precision=double_precision,
            force_ascii=force_ascii,
            date_unit=date_unit,
            default_handler=default_handler,
            compression=compression,
            index=index,
            indent=indent,
            storage_options=storage_options,
            mode=mode,
        )

    def map_partitions(self, func, *args, **kwargs):
        """
        Apply a function to each partition of the dataframe.
        NOTE: this pickles the function and sends it to the workers, so globals are
        pickled. The use of lazy data structures as globals causes issues.
        """
        return bodo.spawn.spawner.submit_func_to_workers(
            func, [], self, *args, **kwargs
        )

    @check_args_fallback(supported=["on"], disable=True)
    def merge(
        self,
        right: "BodoDataFrame | BodoSeries",
        how: MergeHow = "inner",
        on: IndexLabel | AnyArrayLike | None = None,
        left_on: IndexLabel | AnyArrayLike | None = None,
        right_on: IndexLabel | AnyArrayLike | None = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = False,
        suffixes: Suffixes = ("_x", "_y"),
        copy: bool | None = None,
        indicator: str | bool = False,
        validate: MergeValidate | None = None,
    ):  # -> BodoDataFrame:
        from bodo.pandas.base import _empty_like

        zero_size_self = _empty_like(self)
        zero_size_right = _empty_like(right)
        empty_data = zero_size_self.merge(
            zero_size_right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
        )

        if on is None:
            if left_on is None:
                on = tuple(set(self.columns).intersection(set(right.columns)))
            else:
                on = []
        elif not isinstance(on, list):
            on = (on,)
        if left_on is None:
            left_on = []
        if right_on is None:
            right_on = []
        planComparisonJoin = LazyPlan(
            "LogicalComparisonJoin",
            empty_data,
            self._plan,
            right._plan,
            plan_optimizer.CJoinType.INNER,
            [(self.columns.get_loc(c), right.columns.get_loc(c)) for c in on]
            + [
                (self.columns.get_loc(a), right.columns.get_loc(b))
                for a, b in zip(left_on, right_on)
            ],
        )

        return wrap_plan(planComparisonJoin)

    @check_args_fallback("all")
    def __getitem__(self, key):
        """Called when df[key] is used."""

        from bodo.pandas.base import _empty_like

        # Only selecting columns or filtering with BodoSeries is supported
        if not (
            isinstance(key, (str, BodoSeries))
            or (isinstance(key, list) and all(isinstance(k, str) for k in key))
        ):
            raise BodoLibNotImplementedException(
                "only string and BodoSeries keys are supported"
            )

        """ Create 0 length versions of the dataframe and the key and
            simulate the operation to see the resulting type. """
        zero_size_self = _empty_like(self)
        if isinstance(key, BodoSeries):
            """ This is a masking operation. """
            key_plan = (
                # TODO: error checking for key to be a projection on the same dataframe
                # with a binary operator
                get_proj_expr_single(key._plan)
                if key._plan is not None
                else plan_optimizer.LogicalGetSeriesRead(key._mgr._md_result_id)
            )
            zero_size_key = _empty_like(key)
            empty_data = zero_size_self.__getitem__(zero_size_key)
            return wrap_plan(
                plan=LazyPlan("LogicalFilter", empty_data, self._plan, key_plan),
            )
        else:
            """ This is selecting one or more columns. Be a bit more
                lenient than Pandas here which says that if you have
                an iterable it has to be 2+ elements. We will allow
                just one element. """
            if isinstance(key, str):
                key = [key]
            assert isinstance(key, Iterable)
            key = list(key)
            # convert column name to index
            key_indices = [self.columns.get_loc(x) for x in key]

            # Add Index column numbers to select as well if any,
            # assuming Index columns are always at the end of the table (same as Arrow).
            key_indices += [
                len(self.columns) + i
                for i in range(get_n_index_arrays(zero_size_self.index))
            ]

            # Create column reference expressions for selected columns
            exprs = make_col_ref_exprs(key_indices, self._plan)

            empty_data = zero_size_self.__getitem__(key[0] if len(key) == 1 else key)
            return wrap_plan(
                plan=LazyPlan(
                    "LogicalProjection",
                    empty_data,
                    self._plan,
                    exprs,
                ),
            )

    @check_args_fallback("none")
    def __setitem__(self, key, value) -> None:
        """Supports setting columns (df[key] = value) when value is a Series created
        from the same dataframe.
        This is done by creating a new plan that add the new
        column in the existing dataframe plan using a projection.
        """

        # Match cases like df["B"] = df["A"].str.lower()
        if (
            self.is_lazy_plan()
            and isinstance(key, str)
            and isinstance(value, BodoSeries)
            and value.is_lazy_plan()
        ):
            if (
                new_plan := _get_set_column_plan(self._plan, value._plan, key)
            ) is not None:
                # Update internal state
                self._mgr._plan = new_plan
                head_val = value._head_s
                self._head_df[key] = head_val
                with self.disable_collect():
                    # Update internal data manager (e.g. insert a new block or update an
                    # existing one). See:
                    # https://github.com/pandas-dev/pandas/blob/0691c5cf90477d3503834d983f69350f250a6ff7/pandas/core/frame.py#L4481
                    super().__setitem__(key, head_val)
                return

        raise BodoLibNotImplementedException(
            "Only setting a column with a Series created from the same dataframe is supported."
        )

    @check_args_fallback(supported=["func", "axis"])
    def apply(
        self,
        func,
        axis=0,
        raw=False,
        result_type=None,
        args=(),
        by_row="compat",
        engine="python",
        engine_kwargs=None,
        **kwargs,
    ):
        """
        Apply a function along the axis of the dataframe.
        """

        if axis != 1:
            raise BodoLibNotImplementedException(
                "DataFrame.apply(): only axis=1 supported"
            )

        # Get output data type by running the UDF on a sample of the data.
        df_sample = self.head(1).execute_plan()
        pd_sample = pd.DataFrame(df_sample)
        out_sample = pd_sample.apply(func, axis)

        if not isinstance(out_sample, pd.Series):
            raise BodoLibNotImplementedException(
                f"expected output to be Series, got: {type(out_sample)}."
            )

        # TODO [BSE-4788]: Refactor with convert_to_arrow_dtypes util
        empty_df = arrow_to_empty_df(pa.Schema.from_pandas(out_sample.to_frame()))
        empty_series = empty_df.squeeze()
        empty_series.name = out_sample.name

        udf_arg = LazyPlan(
            "PythonScalarFuncExpression",
            empty_series,
            self._plan,
            (
                "apply",
                False,  # is_series
                True,  # is_method
                (func,),  # args
                {"axis": 1},  # kwargs
            ),
            tuple(range(len(self.columns) + get_n_index_arrays(self.head(0).index))),
        )

        # Select Index columns explicitly for output
        n_cols = len(self.columns)
        index_col_refs = tuple(
            make_col_ref_exprs(
                range(n_cols, n_cols + get_n_index_arrays(self.head(0).index)),
                self._plan,
            )
        )
        plan = LazyPlan(
            "LogicalProjection",
            empty_series,
            self._plan,
            (udf_arg,) + index_col_refs,
        )
        return wrap_plan(plan=plan)

    @contextmanager
    def disable_collect(self):
        """Disable collect calls in internal manager to allow updating internal state.
        See __setitem__.
        """
        original_flag = self._mgr._disable_collect
        self._mgr._disable_collect = True
        try:
            yield
        finally:
            self._mgr._disable_collect = original_flag


def _update_func_expr_source(
    func_expr: LazyPlan, new_source_plan: LazyPlan, col_index_offset: int
):
    """Update source plan of PythonScalarFuncExpression and add an offset to its
    input data column index.
    """
    # Previous input data column index
    in_col_ind = func_expr.args[2][0]
    n_source_cols = len(new_source_plan.empty_data.columns)
    # Add Index columns of the new source plan as input
    index_cols = tuple(
        range(
            n_source_cols,
            n_source_cols + get_n_index_arrays(new_source_plan.empty_data.index),
        )
    )
    expr = LazyPlan(
        "PythonScalarFuncExpression",
        func_expr.empty_data,
        new_source_plan,
        func_expr.args[1],
        (in_col_ind + col_index_offset,) + index_cols,
    )
    return expr


def _add_proj_expr_to_plan(
    df_plan: LazyPlan, value_plan: LazyPlan, key: str, replace_func_source=False
):
    """Add a projection on top of dataframe plan that adds or replaces a column
    with output expression of value_plan (which is a single expression projection).
    """
    # Create column reference expressions for each column in the dataframe.
    in_empty_df = df_plan.empty_data

    # Check if the column already exists in the dataframe
    if key in in_empty_df.columns:
        ikey = in_empty_df.columns.get_loc(key)
        is_replace = True
    else:
        ikey = None
        is_replace = False

    # Get the function expression from the value plan to be added
    func_expr = value_plan.args[1][0]
    if func_expr.plan_class != "PythonScalarFuncExpression":
        return None
    func_expr = (
        _update_func_expr_source(func_expr, df_plan, ikey)
        if replace_func_source
        else func_expr
    )

    # Get projection expressions
    n_cols = len(in_empty_df.columns)
    key_indices = [k for k in range(n_cols) if (not is_replace or k != ikey)]
    data_cols = make_col_ref_exprs(key_indices, df_plan)
    if is_replace:
        data_cols.insert(ikey, func_expr)
    else:
        # New column should be at the end of data columns to match Pandas
        data_cols.append(func_expr)
    index_cols = make_col_ref_exprs(
        range(n_cols, n_cols + get_n_index_arrays(in_empty_df.index)), df_plan
    )

    empty_data = df_plan.empty_data.copy()
    empty_data[key] = value_plan.empty_data.copy()
    new_plan = LazyPlan(
        "LogicalProjection",
        empty_data,
        df_plan,
        tuple(data_cols + index_cols),
    )
    return new_plan


def _get_set_column_plan(
    df_plan: LazyPlan,
    value_plan: LazyPlan,
    key: str,
) -> LazyPlan | None:
    """
    Get the plan for setting a column in a dataframe or return None if not supported.
    Creates a projection on top of the dataframe plan that adds original data columns as
    well as the column from the value plan to be set.
    For example, if the df schema is (a, b, c, I) where I is the index column and the
    code is df["D"] = df["b"].str.lower(), then the value plan is:
    ┌───────────────────────────┐
    │         PROJECTION        │
    │    ────────────────────   │
    │        Expressions:       │
    │ "bodo_udf"(#[0.1], #[0.3])│
    │           #[0.3]          │
    └─────────────┬─────────────┘
    ┌─────────────┴─────────────┐
    │        BODO_READ_DF       │
    │    ────────────────────   │
    └───────────────────────────┘
    and the new dataframe plan with new column added is:
    ┌───────────────────────────┐
    │         PROJECTION        │
    │    ────────────────────   │
    │        Expressions:       │
    │           #[0.0]          │
    │           #[0.1]          │
    │           #[0.2]          │
    │ "bodo_udf"(#[0.1], #[0.3])│
    │           #[0.3]          │
    └─────────────┬─────────────┘
    ┌─────────────┴─────────────┐
    │        BODO_READ_DF       │
    │    ────────────────────   │
    └───────────────────────────┘
    """

    # Handle stacked projections like bdf["b"] = bdf["c"].str.lower().str.strip()
    if (
        is_single_projection(value_plan)
        and value_plan.args[0] != df_plan
        and (inner_plan := _get_set_column_plan(df_plan, value_plan.args[0], key))
        is not None
    ):
        return _add_proj_expr_to_plan(inner_plan, value_plan, key, True)

    # Check for simple projections like bdf["b"] = bdf["c"].str.lower()
    if not is_single_projection(value_plan) or value_plan.args[0] != df_plan:
        return None

    return _add_proj_expr_to_plan(df_plan, value_plan, key)
