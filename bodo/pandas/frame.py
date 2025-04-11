import typing as pt
from collections.abc import Callable, Iterable

import pandas as pd
import pyarrow as pa
from pandas._typing import AnyArrayLike, IndexLabel, MergeHow, MergeValidate, Suffixes

import bodo
from bodo.ext import plan_optimizer
from bodo.pandas.array_manager import LazyArrayManager
from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.lazy_wrapper import BodoLazyWrapper
from bodo.pandas.managers import LazyBlockManager, LazyMetadataMixin
from bodo.pandas.series import BodoSeries
from bodo.pandas.utils import (
    LazyPlan,
    check_args_fallback,
    get_lazy_manager_class,
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

    @property
    def _plan(self):
        if hasattr(self._mgr, "_plan"):
            if self._mgr._plan is not None:
                return self._mgr._plan
            else:
                return plan_optimizer.LogicalGetDataframeRead(self._mgr._md_result_id)
        raise NotImplementedError(
            "Plan not available for this manager, recreate this dataframe with from_pandas"
        )

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

    def head(self, n: int = 5):
        """
        Return the first n rows. If head_df is available and larger than n, then use it directly.
        Otherwise, use the default head method which will trigger a data pull.
        """
        if (self._head_df is None) or (n > self._head_df.shape[0]):
            return super().head(n)
        else:
            # If head_df is available and larger than n, then use it directly.
            return self._head_df.head(n)

    def __len__(self):
        if self.is_lazy_plan():
            self._mgr._collect()
        return super().__len__()

    @property
    def shape(self):
        if self.is_lazy_plan():
            self._mgr._collect()
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

    @check_args_fallback(supported=["on"])
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
        new_metadata = zero_size_self.merge(
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
            self._plan,
            right._plan,
            plan_optimizer.CJoinType.INNER,
            [(self.columns.get_loc(c), right.columns.get_loc(c)) for c in on]
            + [
                (self.columns.get_loc(a), right.columns.get_loc(b))
                for a, b in zip(left_on, right_on)
            ],
        )

        return wrap_plan(new_metadata, planComparisonJoin)

    @check_args_fallback("all")
    def __getitem__(self, key):
        """Called when df[key] is used."""
        from bodo.pandas.base import _empty_like

        """ Create 0 length versions of the dataframe and the key and
            simulate the operation to see the resulting type. """
        zero_size_self = _empty_like(self)
        if isinstance(key, BodoSeries):
            """ This is a masking operation. """
            key_plan = (
                key._plan
                if key._plan is not None
                else plan_optimizer.LogicalGetSeriesRead(key._mgr._md_result_id)
            )
            zero_size_key = _empty_like(key)
            new_metadata = zero_size_self.__getitem__(zero_size_key)
            return wrap_plan(
                new_metadata,
                plan=LazyPlan("LogicalFilter", self._plan, key_plan),
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
            pa_schema = pa.Schema.from_pandas(zero_size_self[key])
            if len(key) == 1:
                """ If just one element then have to extract that singular
                    element for the metadata call to Pandas so it doesn't
                    complain. """
                key = key[0]
                new_metadata = zero_size_self.__getitem__(key)
                return wrap_plan(
                    new_metadata,
                    plan=LazyPlan(
                        "LogicalProjection",
                        self._plan,
                        key_indices,
                        pa_schema,
                    ),
                )
            else:
                new_metadata = zero_size_self.__getitem__(key)
                return wrap_plan(
                    new_metadata,
                    plan=LazyPlan(
                        "LogicalProjection",
                        self._plan,
                        key_indices,
                        pa_schema,
                    ),
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
            raise BodoError("DataFrame.apply(): only axis=1 supported")

        # Get output data type by running the UDF on a sample of the data.
        # Saving the plan to avoid hitting LogicalGetDataframeRead gaps with head().
        # TODO: remove when LIMIT plan is properly supported for head().
        mgr_plan = self._mgr._plan
        df_sample = self.head()
        self._mgr._plan = mgr_plan
        out_sample = pd.DataFrame({"OUT": df_sample.apply(func, axis)})
        arrow_schema = pa.Schema.from_pandas(out_sample)

        empty_df = out_sample.iloc[:0]
        empty_df.index = pd.RangeIndex(0)

        plan = LazyPlan("LogicalProjectionUDF", self._plan, func, arrow_schema)
        return wrap_plan(empty_df, plan=plan)
