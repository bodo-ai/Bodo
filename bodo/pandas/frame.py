from __future__ import annotations

import sys
import typing as pt
import warnings
from collections.abc import Callable, Hashable, Iterable, Sequence
from contextlib import contextmanager

import pandas as pd
import pyarrow as pa
from pandas._libs import lib

if pt.TYPE_CHECKING:
    from pandas._typing import (
        AnyArrayLike,
        Axis,
        DropKeep,
        FilePath,
        IgnoreRaise,
        Index,
        IndexLabel,
        Level,
        Manager,
        MergeHow,
        MergeValidate,
        Renamer,
        Self,
        SortKind,
        StorageOptions,
        Suffixes,
        ValueKeyFunc,
        WriteBuffer,
    )
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.table.sorting import SortOrder

    from bodo.ext import plan_optimizer

import numpy as np
from pandas.core.indexing import _LocIndexer

import bodo
from bodo.pandas.array_manager import LazyArrayManager
from bodo.pandas.groupby import DataFrameGroupBy
from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.lazy_wrapper import BodoLazyWrapper, ExecState
from bodo.pandas.managers import LazyBlockManager, LazyMetadataMixin
from bodo.pandas.plan import (
    ConstantExpression,
    LazyPlan,
    LazyPlanDistributedArg,
    LogicalComparisonJoin,
    LogicalDistinct,
    LogicalFilter,
    LogicalGetPandasReadParallel,
    LogicalGetPandasReadSeq,
    LogicalIcebergWrite,
    LogicalInsertScalarSubquery,
    LogicalLimit,
    LogicalOrder,
    LogicalParquetWrite,
    LogicalProjection,
    LogicalS3VectorsWrite,
    _get_df_python_func_plan,
    execute_plan,
    get_proj_expr_single,
    is_colref_projection,
    is_single_colref_projection,
    is_single_projection,
    make_col_ref_exprs,
    maybe_make_list,
    reset_index,
)
from bodo.pandas.series import BodoSeries
from bodo.pandas.utils import (
    BodoCompilationFailedWarning,
    BodoLibFallbackWarning,
    BodoLibNotImplementedException,
    _fix_multi_index_names,
    _get_empty_series_arrow,
    check_args_fallback,
    fallback_warn,
    fallback_wrapper,
    get_lazy_manager_class,
    get_n_index_arrays,
    get_scalar_udf_result_type,
    wrap_module_functions_and_methods,
    wrap_plan,
)


class BodoDataFrameLocIndexer(_LocIndexer):
    def __init__(self, name, obj):
        self.df = obj
        super().__init__(name, obj)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row_sel, col_sel = key
            if row_sel == slice(None, None, None):
                # Handle tuple columns like df.loc[:, ("B", "C")] which are not
                # supported in regular getitem
                if isinstance(col_sel, tuple):
                    col_sel = list(col_sel)
                return self.df.__getitem__(col_sel)
            else:
                fallback_warn("Selected variant of BodoDataFrame.loc[] not supported.")
                return super(pd.DataFrame, self.df).loc.__getitem__(key)

        fallback_warn("Selected variant of BodoDataFrame.loc[] not supported.")
        # Delegate to original behavior
        return super(pd.DataFrame, self.df).loc.__getitem__(key)


class BodoDataFrame(pd.DataFrame, BodoLazyWrapper):
    # We need to store the head_df to avoid data pull when head is called.
    # Since BlockManagers are in Cython it's tricky to override all methods
    # so some methods like head will still trigger data pull if we don't store head_df and
    # use it directly when available.
    _head_df: pd.DataFrame | None = None
    _source_plan: LazyPlan | None = None

    def __new__(cls, *args, **kwargs):
        """Support bodo.pandas.DataFrame() constructor by creating a pandas DataFrame
        and then converting it to a BodoDataFrame.
        """

        # Return regular pandas DataFrame for empty case to avoid internal issues.
        if not args and not kwargs:
            return pd.DataFrame()

        # TODO: Optimize creation from other BodoDataFrames, BodoSeries, or BodoScalars

        df = pd.DataFrame(*args, **kwargs)
        return bodo.pandas.base.from_pandas(df)

    def __init__(self, *args, **kwargs):
        # No-op since already initialized by __new__
        pass

    @classmethod
    def _from_mgr(cls, mgr: Manager, axes: list[Index]) -> Self:
        """Replace pd.DataFrame._from_mgr to create BodoDataFrame instances.
        This avoids calling BodoDataFrame.__new__() which would cause infinite recursion
        """
        from pandas.core.generic import NDFrame

        obj = super().__new__(cls)
        NDFrame.__init__(obj, mgr)
        return obj

    @property
    def loc(self):
        return BodoDataFrameLocIndexer("loc", self)

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
            from bodo.pandas.base import _empty_like

            empty_data = _empty_like(self)

            """This caching also creates issues because you can materialize a
               dataframe then ask for a plan and then add a column through
               some pandas method we don't support and then you ask for the
               _plan and you may get a source plan from before the column was
               added.  To prevent problems, store the schema of the saved
               source plan and only use it if the current schema is the same.
            """
            if getattr(self, "_source_plan", None) is not None:
                # If the schema hasn't changed since the last time the source
                # plan was generated then use the old source plan.
                if empty_data.equals(self._source_plan[0]):
                    return self._source_plan[1]

            if bodo.dataframe_library_run_parallel:
                nrows = len(self)
                self._source_plan = (
                    empty_data,
                    LogicalGetPandasReadParallel(
                        empty_data,
                        nrows,
                        LazyPlanDistributedArg(self),
                    ),
                )
            else:
                self._source_plan = (
                    empty_data,
                    LogicalGetPandasReadSeq(
                        empty_data,
                        self,
                    ),
                )

            return self._source_plan[1]

    def __getattribute__(self, name: str):
        """Custom attribute access that triggers a fallback warning for unsupported attributes."""

        ignore_fallback_attrs = [
            "dtypes",
            "to_string",
            "attrs",
            "flags",
            "columns",
            "ndim",
            "axes",
            "iloc",
            "empty",
        ]

        cls = object.__getattribute__(self, "__class__")
        base = cls.__mro__[0]

        if (
            name not in base.__dict__
            and name not in ignore_fallback_attrs
            and not name.startswith("_")
            and hasattr(pd.DataFrame, name)
        ):
            msg = (
                f"{name} is not implemented in Bodo DataFrames yet. "
                "Falling back to Pandas (may be slow or run out of memory)."
            )
            return fallback_wrapper(
                self, object.__getattribute__(self, name), name, msg
            )

        return object.__getattribute__(self, name)

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
    ) -> BodoDataFrame:
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

        # Negative n like -1 is equivalent to df.iloc[:-1]
        if n < 0:
            n = max(0, len(self) + n)

        if (self._head_df is None) or (n > self._head_df.shape[0]):
            if bodo.dataframe_library_enabled and isinstance(
                self._mgr, LazyMetadataMixin
            ):
                from bodo.pandas.base import _empty_like

                planLimit = LogicalLimit(
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
        from bodo.pandas.plan import count_plan

        if self._exec_state == ExecState.PLAN:
            return count_plan(self)
        if self._exec_state == ExecState.DISTRIBUTED:
            return self._mgr._md_nrows
        if self._exec_state == ExecState.COLLECTED:
            return super().__len__()

    def __repr__(self):
        # Pandas repr implementation calls len() first which will execute an extra
        # count query before the actual plan which is unnecessary.
        if self._exec_state == ExecState.PLAN:
            self.execute_plan()

        # Avoid fallback warnings for prints
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=BodoLibFallbackWarning)
            return super().__repr__()

    def __array__(self, dtype=None, copy=None):
        if dtype is None and copy is None and len(self.dtypes) > 0:
            common_dtype = self.dtypes.iloc[0]
            if (
                isinstance(common_dtype, pd.ArrowDtype)
                and (
                    pa.types.is_integer(common_dtype.pyarrow_dtype)
                    or pa.types.is_floating(common_dtype.pyarrow_dtype)
                )
                and all(self.dtypes == common_dtype)
            ):
                return np.asarray(self, dtype=common_dtype.numpy_dtype)

        return super().__array__(dtype, copy)

    @property
    def index(self):
        self.execute_plan()
        index = super().index

        if isinstance(index, pd.MultiIndex):
            index.names = _fix_multi_index_names(index.names)

        return index

    @index.setter
    def index(self, value):
        self.execute_plan()
        super()._set_axis(1, value)

    @property
    def columns(self):
        return super().columns

    @columns.setter
    def columns(self, value):
        # Validate arguments/Update column names in managers.
        super()._set_axis(0, value)

        # Update column names in head/metadata.
        if self._head_df is not None:
            self._head_df.columns = value
        if (md_head := getattr(self._mgr, "_md_head", None)) is not None:
            md_head.columns = value

        # Update column names in plan.
        if self.is_lazy_plan():
            self._mgr._plan._update_column_names(value)
        elif self._exec_state == ExecState.DISTRIBUTED:
            assert self._head_df is not None
            # Since we can't edit the plan directly,
            # create a new projection with new column names.
            empty_data = self._head_df.head(0)
            col_indices = list(
                range(len(empty_data.columns) + get_n_index_arrays(empty_data.index))
            )
            self._mgr._plan = LogicalProjection(
                empty_data, self._plan, make_col_ref_exprs(col_indices, self._plan)
            )

    @property
    def shape(self):
        from bodo.pandas.plan import count_plan

        if self._exec_state == ExecState.PLAN:
            return (count_plan(self), len(self._head_df.columns))
        if self._exec_state == ExecState.DISTRIBUTED:
            return (self._mgr._md_nrows, len(self._head_df.columns))
        if self._exec_state == ExecState.COLLECTED:
            return super().shape

    @check_args_fallback(supported=["columns", "copy"])
    def rename(
        self,
        mapper: Renamer | None = None,
        *,
        index: Renamer | None = None,
        columns: Renamer | None = None,
        axis: Axis | None = None,
        copy: bool | None = None,
        inplace: bool = False,
        level: Level | None = None,
        errors: IgnoreRaise = "ignore",
    ) -> BodoDataFrame | None:
        orig_plan = self._plan
        # TODO: The value of copy here is ignored.
        # Most cases of copy=False we have are the idiom A=A.rename(copy=False) where there is still
        # only one possible reference to the data so we can treat this case like copy=True since for us
        # we only materialize as needed and so copying isn't an overhead.

        renamed_plan = orig_plan.replace_empty_data(
            orig_plan.empty_data.rename(columns=columns)
        )
        return wrap_plan(renamed_plan)

    @check_args_fallback(supported=["path", "engine", "compression", "row_group_size"])
    def to_parquet(
        self,
        path: FilePath | WriteBuffer[bytes] | None = None,
        engine: pt.Literal["auto", "pyarrow", "fastparquet"] = "auto",
        compression: str | None = "snappy",
        index: bool | None = None,
        partition_cols: list[str] | None = None,
        storage_options: StorageOptions | None = None,
        row_group_size: int = -1,
        **kwargs,
    ) -> bytes | None:
        from bodo.io.fs_io import get_s3_bucket_region_wrapper
        from bodo.pandas.base import _empty_like

        if not isinstance(path, str):
            raise BodoLibNotImplementedException(
                "DataFrame.to_parquet(): path must be a string"
            )

        if engine not in ("auto", "pyarrow"):
            raise BodoLibNotImplementedException(
                "DataFrame.to_parquet(): only 'auto' and 'pyarrow' engines are supported"
            )

        if compression not in (None, "snappy", "gzip", "brotli"):
            raise BodoLibNotImplementedException(
                "DataFrame.to_parquet(): only None, 'snappy', 'gzip' and 'brotli' compressions are supported"
            )

        # Convert None to "none" as expected by the backend.
        # https://github.com/bodo-ai/Bodo/blob/ff39453f07d8691751d95668ab06a72a5f742dff/bodo/hiframes/pd_dataframe_ext.py#L3795
        if compression is None:
            compression = "none"

        if not isinstance(row_group_size, int):
            raise ValueError(
                "DataFrame.to_parquet(): row_group_size must be an integer"
            )

        bucket_region = get_s3_bucket_region_wrapper(path, False)

        write_plan = LogicalParquetWrite(
            _empty_like(self),
            self._plan,
            path,
            compression,
            bucket_region,
            row_group_size,
        )
        execute_plan(write_plan)

    @check_args_fallback(unsupported="none")
    def to_iceberg(
        self,
        table_identifier: str,
        catalog_name: str | None = None,
        *,
        catalog_properties: dict[str, pt.Any] | None = None,
        location: str | None = None,
        append: bool = False,
        partition_spec: PartitionSpec | None = None,
        sort_order: SortOrder | None = None,
        properties: dict[str, pt.Any] | None = None,
        snapshot_properties: dict[str, str] | None = None,
    ) -> None:
        # See Pandas implementation of to_iceberg:
        # https://github.com/pandas-dev/pandas/blob/c5457f61d92b9428a56c619a6c420b122a41a347/pandas/core/frame.py#L3550
        # https://github.com/pandas-dev/pandas/blob/c5457f61d92b9428a56c619a6c420b122a41a347/pandas/io/iceberg.py#L98
        # See Bodo JIT implementation of streaming writes to Iceberg:
        # https://github.com/bodo-ai/Bodo/blob/142678b2fe7217d80e233d201061debae2d47c13/bodo/io/iceberg/stream_iceberg_write.py#L535
        import pyiceberg.catalog
        import pyiceberg.partitioning
        import pyiceberg.table.sorting

        import bodo.io.iceberg
        from bodo.io.iceberg.write_utils import CreateTableMeta
        from bodo.pandas.base import _empty_like

        # Support simple directory only calls like:
        # df.to_iceberg("table", location="/path/to/table")
        if catalog_name is None and catalog_properties is None and location is not None:
            if location.startswith("arn:aws:s3tables:"):
                from bodo.io.iceberg.catalog.s3_tables import (
                    construct_catalog_properties as construct_s3_tables_catalog_properties,
                )

                catalog_properties = construct_s3_tables_catalog_properties(location)
            else:
                catalog_properties = {
                    pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
                    pyiceberg.catalog.WAREHOUSE_LOCATION: location,
                }
            # DirCatalog and S3TablesCatalog do not support extra location argument in create_table
            location = None
        elif catalog_properties is None:
            catalog_properties = {}

        if partition_spec is None:
            partition_spec = pyiceberg.partitioning.UNPARTITIONED_PARTITION_SPEC

        if sort_order is None:
            sort_order = pyiceberg.table.sorting.UNSORTED_SORT_ORDER

        if properties is None:
            properties = ()
        else:
            if not isinstance(properties, dict):
                raise ValueError(
                    "Iceberg write properties must be a dictionary, got: "
                    f"{type(properties)}"
                )
            # Convert properties to a tuple of items to match expected type in
            # CreateTableMetaType
            properties = tuple(properties.items())

        if snapshot_properties is None:
            snapshot_properties = {}

        catalog = pyiceberg.catalog.load_catalog(catalog_name, **catalog_properties)

        if_exists = "append" if append else "replace"
        df_schema = self._plan.pa_schema
        (
            txn,
            fs,
            table_loc,
            output_pa_schema,
            iceberg_schema_str,
            partition_spec,
            partition_tuples,
            sort_order_id,
            sort_tuples,
            properties,
        ) = bodo.io.iceberg.write_utils.start_write_rank_0(
            catalog,
            table_identifier,
            df_schema,
            if_exists,
            False,
            CreateTableMeta(None, None, properties),
            location,
            partition_spec,
            sort_order,
            snapshot_properties,
        )
        bucket_region = bodo.io.fs_io.get_s3_bucket_region_wrapper(table_loc, False)
        max_pq_chunksize = properties.get(
            "write.target-file-size-bytes",
            bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE,
        )
        compression = properties.get("write.parquet.compression-codec", "snappy")
        # TODO: support Theta sketches

        write_plan = LogicalIcebergWrite(
            _empty_like(self),
            self._plan,
            table_loc,
            bucket_region,
            max_pq_chunksize,
            compression,
            partition_tuples,
            sort_tuples,
            iceberg_schema_str,
            output_pa_schema,
            fs,
        )
        all_iceberg_files_infos = execute_plan(write_plan)
        # Flatten the list of lists
        all_iceberg_files_infos = (
            [item for sub in all_iceberg_files_infos for item in sub]
            if all_iceberg_files_infos
            else None
        )
        (
            fnames,
            file_records,
            partition_infos,
        ) = bodo.io.iceberg.write_utils.generate_data_file_info_seq(
            all_iceberg_files_infos
        )

        # Register file names, metrics and schema in transaction
        success = bodo.io.iceberg.write_utils.register_table_write_seq(
            txn,
            fnames,
            file_records,
            partition_infos,
            partition_spec,
            sort_order_id,
            snapshot_properties,
        )
        if not success:
            raise ValueError("Iceberg write failed.")

    @check_args_fallback(unsupported="none")
    def to_s3_vectors(
        self,
        vector_bucket_name: str,
        index_name: str,
        region: str = None,
    ) -> None:
        """
        Write the DataFrame to S3 Vectors storage.
        """
        import pyarrow as pa

        from bodo.pandas.base import _empty_like

        # https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-getting-started.html

        # Check schema
        schema = self._plan.pa_schema
        required_fields = {"key", "data", "metadata"}
        if not required_fields.issubset(schema.names):
            raise ValueError(
                f"DataFrame must have columns {required_fields} to write to S3 Vectors."
            )
        if schema.field("key").type not in (pa.string(), pa.large_string()):
            raise ValueError(
                "DataFrame 'key' column must be strings to write to S3 Vectors."
            )
        if schema.field("data").type not in (
            pa.list_(pa.float32()),
            pa.large_list(pa.float32()),
            pa.list_(pa.float64()),
            pa.large_list(pa.float64()),
        ):
            raise ValueError(
                "DataFrame 'data' column must be a list of floats to write to S3 Vectors."
            )
        if not isinstance(schema.field("metadata").type, pa.StructType):
            raise ValueError(
                "DataFrame 'metadata' column must be a struct type to write to S3 Vectors."
            )

        write_plan = LogicalS3VectorsWrite(
            _empty_like(self),
            self._plan,
            vector_bucket_name,
            index_name,
            region,
        )
        execute_plan(write_plan)

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
        bodo.spawn.utils.import_compiler_on_workers()

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
        # Import compiler lazily
        import bodo.decorators  # isort:skip # noqa
        from bodo.utils.typing import check_unsupported_args
        # argument defaults should match that of to_csv_overload in pd_dataframe_ext.py

        bodo.spawn.utils.import_compiler_on_workers()

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

        bodo.spawn.utils.import_compiler_on_workers()

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

        If self is a lazy plan, then the result will also be a lazy plan
        (assuming result is Series and the dtype can be infered). Otherwise, the lazy
        plan will be evaluated.
        NOTE: this pickles the function and sends it to the workers, so globals are
        pickled. The use of lazy data structures as globals causes issues.

        Args:
            func (Callable): A callable which takes in a DataFrame as its first
                argument and returns a DataFrame or Series that has the same length
                its input.
            *args: Additional positional arguments to pass to func.
            **kwargs: Additional key-word arguments to pass to func.

        Returns:
            DataFrame or Series: The result of applying the func.
        """
        import bodo.spawn.spawner

        if self._exec_state == ExecState.PLAN:
            required_fallback = False
            try:
                empty_series = get_scalar_udf_result_type(
                    self, "map_partitions", func, *args, **kwargs
                )
            except BodoLibNotImplementedException as e:
                required_fallback = True
                msg = (
                    f"map_partitions(): encountered exception: {e}, while trying to "
                    "build lazy plan. Executing plan and running map_partitions on "
                    "workers (may be slow or run out of memory)."
                )
                fallback_warn(msg)

                df_arg = self.execute_plan()

            if not required_fallback:
                return _get_df_python_func_plan(
                    self._plan, empty_series, func, args, kwargs
                )
        else:
            df_arg = self

        return bodo.spawn.spawner.submit_func_to_workers(
            func, [], df_arg, *args, **kwargs
        )

    @check_args_fallback(supported=["on", "left_on", "right_on", "how"])
    def merge(
        self,
        right: BodoDataFrame | BodoSeries,
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

        # Validates only on, left_on and right_on for now
        is_cross = how == "cross"
        left_on, right_on = validate_merge_spec(
            self, right, on, left_on, right_on, is_cross
        )

        zero_size_self = _empty_like(self)
        zero_size_right = _empty_like(right)
        empty_data = zero_size_self.merge(
            zero_size_right,
            how=how,
            on=None,
            left_on=None if is_cross else left_on,
            right_on=None if is_cross else right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
        )

        key_indices = [
            (self.columns.get_loc(a), right.columns.get_loc(b))
            for a, b in zip(left_on, right_on)
        ]

        # Dummy output with all probe/build columns with unique names to enable
        # make_col_ref_exprs() below
        empty_left = _empty_like(self)
        empty_right = _empty_like(right)
        empty_join_out = pd.concat([empty_left, empty_right], axis=1)
        empty_join_out.columns = [
            c + str(i) for i, c in enumerate(empty_join_out.columns)
        ]

        join_type = _get_join_type_from_how(how)
        planComparisonJoin = LogicalComparisonJoin(
            empty_join_out,
            self._plan,
            right._plan,
            join_type,
            key_indices,
        )

        # Column indices in output that need to be selected
        col_indices = list(range(len(self.columns)))
        # Skip index columns
        # TODO [BSE-4820]: unless indexes are a key.
        n_left_indices = get_n_index_arrays(empty_left.index)
        common_keys = set(left_on).intersection(set(right_on))
        for i, col in enumerate(right.columns):
            # Ignore common keys that are in the right side to match Pandas
            if col not in common_keys:
                col_indices.append(len(self.columns) + n_left_indices + i)

        # Create column reference expressions for selected columns
        exprs = make_col_ref_exprs(col_indices, planComparisonJoin)
        proj_plan = LogicalProjection(
            empty_data,
            planComparisonJoin,
            exprs,
        )

        return wrap_plan(proj_plan)

    @check_args_fallback(supported=["by", "as_index", "dropna"])
    def groupby(
        self,
        by=None,
        axis: Axis | lib.NoDefault = lib.no_default,
        level: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = False,
        group_keys: bool = True,
        observed: bool | lib.NoDefault = lib.no_default,
        dropna: bool = True,
    ) -> DataFrameGroupBy:
        """
        Provides support for groupby similar to Pandas:
        https://github.com/pandas-dev/pandas/blob/0691c5cf90477d3503834d983f69350f250a6ff7/pandas/core/frame.py#L9148
        """
        if isinstance(by, str):
            by = [by]

        # Only list of string column names for keys is supported for now.
        if not isinstance(by, (list, tuple)) or not all(isinstance(b, str) for b in by):
            raise BodoLibNotImplementedException(
                "groupby: only string keys are supported"
            )

        return DataFrameGroupBy(self, by, as_index, dropna)

    @check_args_fallback(supported=["columns"])
    def drop(
        self,
        labels: IndexLabel | None = None,
        *,
        axis: Axis = 0,
        index: IndexLabel | None = None,
        columns: IndexLabel | None = None,
        level: Level | None = None,
        inplace: bool = False,
        errors: IgnoreRaise = "raise",
    ) -> BodoDataFrame | None:
        if isinstance(columns, str):
            columns = [columns]
        if not isinstance(columns, list):
            raise ValueError("drop columns must be string or list of string")
        cur_col_names = self.columns.tolist()
        columns_to_use = [x for x in cur_col_names if x not in columns]
        if len(columns_to_use) != len(cur_col_names) - len(columns):
            not_found = [x for x in columns if x not in cur_col_names]
            raise KeyError(
                f"drop columns includes names {not_found} not present in dataframe"
            )
        return self.__getitem__(columns_to_use)

    @check_args_fallback("all")
    def __getitem__(self, key):
        """Called when df[key] is used."""
        import pyarrow as pa

        from bodo.pandas.base import _empty_like

        # Create 0 length versions of the dataframe and the key and
        # simulate the operation to see the resulting type.
        zero_size_self = _empty_like(self)

        # Filter operation
        if isinstance(key, BodoSeries) and key._plan.pa_schema.types[0] == pa.bool_():
            # Pattern match df1[df1.A.isin(df2.B)] case which is a semi-join
            if (out_plan := get_isin_filter_plan(self._plan, key._plan)) is not None:
                return wrap_plan(out_plan)

            plan = self._plan
            # If the key is a scalar subquery, then we use that plan since it's just
            # this plan with the extra column containing the scalar value.
            if isinstance(key._plan.source, LogicalInsertScalarSubquery):
                plan = key._plan.source

            key_expr = get_proj_expr_single(key._plan)
            key_expr = key_expr.replace_source(plan)
            if key_expr is None:
                raise BodoLibNotImplementedException(
                    "DataFrame filter expression must be on the same dataframe."
                )
            zero_size_key = _empty_like(key)
            empty_data = zero_size_self.__getitem__(zero_size_key)
            return wrap_plan(
                plan=LogicalFilter(empty_data, plan, key_expr),
            )
        # Select one or more columns
        elif isinstance(key, str) or (
            isinstance(key, list) and all(isinstance(k, str) for k in key)
        ):
            if isinstance(key, str):
                key = [key]
                output_series = True
            else:
                output_series = False

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

            empty_data = zero_size_self.__getitem__(key[0] if output_series else key)
            return wrap_plan(
                plan=LogicalProjection(
                    empty_data,
                    self._plan,
                    exprs,
                ),
            )

        raise BodoLibNotImplementedException(
            "DataFrame getitem: Only selecting columns or filtering with BodoSeries is supported."
        )

    @check_args_fallback("none")
    def __setitem__(self, key, value) -> None:
        """Supports setting columns (df[key] = value) when value is a Series created
        from the same dataframe.
        This is done by creating a new plan that add the new
        column in the existing dataframe plan using a projection.
        """
        import pyarrow as pa

        from bodo.pandas.base import _empty_like

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
                head_val = value._head_s
                self._update_setitem_internal_state(new_plan, key, head_val)
                return

        # Match cases like df["B"] = 1
        if (
            self.is_lazy_plan()
            and isinstance(key, str)
            and pd.api.types.is_scalar(value)
        ):
            # Create a projection with the scalar column included
            empty_data = _empty_like(self)

            # Check if the column already exists in the dataframe
            if key in empty_data.columns:
                ikey = empty_data.columns.get_loc(key)
                is_replace = True
            else:
                ikey = None
                is_replace = False

            const_expr = ConstantExpression(
                # Dummy empty data for LazyPlan
                empty_data,
                self._plan,
                value,
            )
            proj_exprs = _get_setitem_proj_exprs(
                empty_data, self._plan, ikey, is_replace, const_expr
            )
            empty_data[key] = value

            # Make sure proper Arrow type is used for the column to match backend and
            # there is no object dtype.
            pa_type = pa.scalar(value).type
            if isinstance(pa_type, pa.TimestampType):
                # Convert to nanosecond precision as required by backend
                pa_type = pa.timestamp("ns", pa_type.tz)
            empty_data[key] = empty_data[key].astype(pd.ArrowDtype(pa_type))

            new_plan = LogicalProjection(
                empty_data,
                self._plan,
                proj_exprs,
            )
            self._update_setitem_internal_state(new_plan, key, value)
            return

        # Match cases like df[["B", "C"]] = 1
        if (
            self.is_lazy_plan()
            and isinstance(key, Iterable)
            and all(isinstance(x, str) for x in key)
            and pd.api.types.is_scalar(value)
        ):
            # Implement as recursive calls to the above code segment for
            # single column assignment.
            for new_col in key:
                self.__setitem__(new_col, value)
            return

        raise BodoLibNotImplementedException(
            "Only setting a column with a Series created from the same dataframe is supported."
        )

    def _update_setitem_internal_state(
        self, new_plan: LazyPlan, key: str, head_val: pd.Series
    ):
        """Update internal state of the dataframe for setting a column.
        new_plan: the updated plan that adds the column to the dataframe.
        key: the name of the column to be set.
        head_val: new head value for the column to be set (Series, array or scalar).
        """
        self._mgr._plan = new_plan
        new_column = key not in self.columns
        # Copy and update head in case reused
        new_df_head = self._head_df.copy()
        new_df_head[key] = head_val
        self._head_df = new_df_head
        self._mgr._md_head = new_df_head._mgr
        with self.disable_collect():
            # Update internal data manager (e.g. insert a new block or update an
            # existing one). See:
            # https://github.com/pandas-dev/pandas/blob/0691c5cf90477d3503834d983f69350f250a6ff7/pandas/core/frame.py#L4481
            if new_column:
                self._mgr.insert(
                    len(self._info_axis), key, *self._sanitize_column(head_val)
                )
            else:
                loc = self._info_axis.get_loc(key)
                self._iset_item_mgr(loc, *self._sanitize_column(head_val))

    def _apply_bodo(
        self, func, args: tuple, **kwargs
    ) -> tuple[BodoDataFrame | None, str]:
        """Attempts to create a wrapper for creating a C callback for the provided
        function and determine output types using JIT.

        Raises:
            BodoLibNotImplementedException: If output of UDF is a DataFrame.

        Returns:
            tuple[BodoDataFrame | None, str]: The new lazy dataframe.
                If errors occured during compilation, first value will be None
                followed by the errors.
        """
        # Import compiler lazily
        import bodo.decorators  # isort:skip # noqa
        from bodo.hiframes.table import TableType
        from bodo.pandas.utils_jit import (
            cpp_table_to_df_jit,
            get_udf_cfunc_decorator,
            series_to_cpp_table_jit,
        )
        from bodo.pandas_compat import _prepare_function_arguments
        from bodo.utils.typing import BodoError

        zero_sized_self = self.head(0)

        # Convert kwargs into args to avoid dynamically generating apply wrapper.
        if callable(func):
            try:
                args, _ = _prepare_function_arguments(func, args, kwargs)
            except ValueError:
                # Keyword-only arguments in UDFs are rare.
                return None, "Keyword-only arguments not supported by JIT"
        elif kwargs:
            return (
                None,
                "Keyword arguments are only supported for callable funcs, use args instead",
            )

        # Generate wrappers for calling apply from C++.
        df_type = bodo.typeof(zero_sized_self)
        index_type = df_type.index
        py_table_type = TableType(df_type.data)
        out_cols_arr = np.array(range(len(self.columns)), dtype=np.int64)
        column_names = bodo.utils.typing.ColNamesMetaType(tuple(self.columns))

        @bodo.jit(cache=True, spawn=False, distributed=False)
        def apply_wrapper_inner(df):
            return df.apply(func, axis=1, args=args)

        def apply_wrapper(in_cpp_table):
            series = cpp_table_to_df_jit(
                in_cpp_table, out_cols_arr, column_names, py_table_type, index_type
            )
            out_series = apply_wrapper_inner(series)
            out_cpp_table = series_to_cpp_table_jit(out_series)
            return out_cpp_table

        try:
            # Compile map inner wrapper, get the output type
            out_jit = apply_wrapper_inner(zero_sized_self)
        except BodoError as e:
            # Compilation failed, attempt execute UDF in Python.
            return None, str(e)

        # DataFrame output is supported by JIT but not Bodo DataFr.
        if isinstance(out_jit, pd.DataFrame):
            raise BodoLibNotImplementedException(
                f"DataFrame.apply(): expected output to be Series, got: {type(out_jit)}"
            )
        empty_series = _get_empty_series_arrow(out_jit)

        # Jit failed to determine dtypes, likely from gaps in our Arrow support.
        if pa.types.is_null(empty_series.dtype.pyarrow_dtype):
            return None, "JIT could not determine pyarrow return type from UDF"

        return _get_df_python_func_plan(
            self._plan,
            empty_series,
            apply_wrapper,
            (),
            {},
            cfunc_decorator=get_udf_cfunc_decorator(),
        ), ""

    @check_args_fallback(supported=["func", "axis", "args", "engine"])
    def apply(
        self,
        func,
        axis=0,
        raw=False,
        result_type=None,
        args=(),
        by_row="compat",
        engine="bodo",
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

        if engine == "numba":
            raise BodoLibNotImplementedException(
                "DataFrame.apply(): with engine='numba' not supported yet."
            )
        elif engine not in ("bodo", "python"):
            raise TypeError(
                "DataFrame.apply(): expected engine to be one of (bodo, python)"
            )

        if engine == "bodo":
            apply_result, error_msg = self._apply_bodo(func, args, **kwargs)

            if apply_result is None:
                msg = (
                    "DataFrame.apply(): Compiling user defined function failed or "
                    "encountered an unsupported result type. Falling back to "
                    "Python engine. Add engine='python' to ignore this warning. "
                    "Original error: "
                    f"{error_msg}."
                )
                if bodo.dataframe_library_warn:
                    warnings.warn(BodoCompilationFailedWarning(msg))
            else:
                if bodo.dataframe_library_run_parallel:
                    bodo.spawn.utils.import_compiler_on_workers()
                return apply_result

        # engine == "python" or jit fallthrough
        empty_series = get_scalar_udf_result_type(
            self, "apply", func, axis=axis, args=args, **kwargs
        )

        apply_kwargs = {"axis": 1, "args": args} | kwargs

        return _get_df_python_func_plan(
            self._plan, empty_series, "apply", (func,), apply_kwargs
        )

    @check_args_fallback(supported=["by", "ascending", "na_position", "kind"])
    def sort_values(
        self,
        by: IndexLabel,
        *,
        axis: Axis = 0,
        ascending: bool | list[bool] | tuple[bool, ...] = True,
        inplace: bool = False,
        kind: SortKind | None = None,
        na_position: str | list[str] | tuple[str, ...] = "last",
        ignore_index: bool = False,
        key: ValueKeyFunc | None = None,
    ) -> BodoDataFrame | None:
        from bodo.pandas.base import _empty_like

        # Validate by argument.
        if isinstance(by, str):
            by = [by]
        elif not isinstance(by, (list, tuple)):
            raise ValueError(
                "DataFrame.sort_values(): argument by not a string, list or tuple"
            )

        if not all(isinstance(item, str) for item in by):
            raise ValueError(
                "DataFrame.sort_values(): argument by iterable does not contain only strings"
            )

        # Validate ascending argument.
        if isinstance(ascending, bool):
            ascending = [ascending]
        elif not isinstance(ascending, (list, tuple)):
            raise ValueError(
                "DataFrame.sort_values(): argument ascending not a bool, list or tuple"
            )

        if not all(isinstance(item, bool) for item in ascending):
            raise ValueError(
                "DataFrame.sort_values(): argument ascending iterable does not contain only boolean"
            )

        # Validate na_position argument.
        if isinstance(na_position, str):
            na_position = [na_position]
        elif not isinstance(na_position, (list, tuple)):
            raise ValueError(
                "DataFrame.sort_values(): argument na_position not a string, list or tuple"
            )

        if not all(item in ["first", "last"] for item in na_position):
            raise ValueError(
                "DataFrame.sort_values(): argument na_position iterable does not contain only 'first' or 'last'"
            )

        if kind is not None:
            if bodo.dataframe_library_warn:
                warnings.warn("sort_values() kind argument ignored")

        # Apply singular ascending param to all columns.
        if len(by) != len(ascending):
            if len(ascending) == 1:
                ascending = ascending * len(by)
            else:
                raise ValueError(
                    f"DataFrame.sort_values(): lengths of by {len(by)} and ascending {len(ascending)}"
                )

        # Apply singular na_position param to all columns.
        if len(by) != len(na_position):
            if len(na_position) == 1:
                na_position = na_position * len(by)
            else:
                raise ValueError(
                    f"DataFrame.sort_values(): lengths of by {len(by)} and na_position {len(na_position)}"
                )
        # Convert to True/False list instead of str.
        na_position = [True if x == "first" else False for x in na_position]

        if any(col not in self.columns for col in by):
            raise BodoLibNotImplementedException(
                "sort_values on index column not supported"
            )

        # Convert column names to indices.
        cols = [self.columns.get_loc(col) for col in by]

        """ Create 0 length versions of the dataframe as sorted dataframe
            has the same structure. """
        zero_size_self = _empty_like(self)

        return wrap_plan(
            plan=LogicalOrder(
                zero_size_self,
                self._plan,
                ascending,
                na_position,
                cols,
                self._plan.pa_schema,
            ),
        )

    @check_args_fallback(supported=["subset", "keep"])
    def drop_duplicates(
        self,
        subset: Hashable | Sequence[Hashable] | None = None,
        *,
        keep: DropKeep = "first",
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> BodoDataFrame | None:
        from bodo.pandas.base import _empty_like

        if keep not in ("first", "last"):
            raise BodoLibNotImplementedException(
                "DataFrame.drop_duplicates() keep argument: only 'first' and 'last' are supported."
            )

        zero_size_self = _empty_like(self)

        # If Index columns exist, it's as if a subset of columns are keys and groupby
        # should be used.
        if subset is None and get_n_index_arrays(zero_size_self.index) > 0:
            subset = zero_size_self.columns.tolist()

        if subset is not None:
            subset_group = self.groupby(
                subset, as_index=False, sort=False, dropna=False
            )
            if keep == "first":
                drop_dups = subset_group.first()
            else:
                drop_dups = subset_group.last()

            # Preserve original ordering of columns
            return drop_dups[self.columns.tolist()]

        exprs = make_col_ref_exprs(list(range(len(zero_size_self.columns))), self._plan)
        return wrap_plan(
            plan=LogicalDistinct(
                zero_size_self,
                self._plan,
                exprs,
            ),
        )

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

    @check_args_fallback(supported=["drop", "names", "level"])
    def reset_index(
        self,
        level=None,
        *,
        drop=False,
        inplace=False,
        col_level=0,
        col_fill="",
        allow_duplicates=lib.no_default,
        names=None,
    ):
        """
        Reset the index, or a level of it.
        Reset the index of the DataFrame, and use the default one instead.
        If the DataFrame has a MultiIndex, this method can remove one or more levels.
        """
        return reset_index(self, drop, level, names=names)


def _add_proj_expr_to_plan(
    df_plan: LazyPlan,
    value_plan: LogicalProjection,
    key: str,
    replace_func_source=False,
    out_columns: list[str] | None = None,
):
    """Add a projection on top of dataframe plan that adds or replaces a column
    with output expression of value_plan (which is a single expression projection).

    df_plan: the dataframe plan to add the column to.
    value_plan: the value plan that contains the expression to be added as a column.
    key: the name of the column to be added or replaced.
    replace_func_source: if True, update the input column index of the function
                         expression in value_plan to point to the source dataframe plan.
    out_columns: if provided, only these columns will be included in the output plan
                 (excludes key).
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
    func_expr = get_proj_expr_single(value_plan)

    if replace_func_source:
        func_expr = func_expr.update_func_expr_source(df_plan, ikey)

    # Update output column name
    func_expr = func_expr.replace_empty_data(
        func_expr.empty_data.set_axis([key], axis=1)
    )

    proj_exprs = _get_setitem_proj_exprs(
        in_empty_df, df_plan, ikey, is_replace, func_expr, out_columns
    )
    empty_data = (
        df_plan.empty_data.copy()
        if out_columns is None
        else df_plan.empty_data[out_columns].copy()
    )
    empty_data[key] = value_plan.empty_data.copy()
    new_plan = LogicalProjection(
        empty_data,
        df_plan,
        proj_exprs,
    )
    return new_plan


def _get_setitem_proj_exprs(
    in_empty_df, df_plan, ikey, is_replace, func_expr, out_columns=None
):
    """Create projection expressions for setting a column in a dataframe."""
    n_cols = len(in_empty_df.columns)

    if out_columns is not None:
        data_cols = []
        for c in out_columns:
            if is_replace and c == in_empty_df.columns[ikey]:
                data_cols.append(func_expr)
            else:
                data_cols.append(
                    make_col_ref_exprs([in_empty_df.columns.get_loc(c)], df_plan)[0]
                )

        if not is_replace or in_empty_df.columns[ikey] not in out_columns:
            data_cols.append(func_expr)
    else:
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
    return tuple(data_cols + index_cols)


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

    # Handle extra projection on source plan that only selects columns like:
    # df2 = df1[["B", "C"]]
    # df2["D"] = df1["B"].str.lower()
    if (
        is_single_projection(value_plan)
        and is_colref_projection(df_plan)
        and value_plan.source == df_plan.source
    ):
        out_columns = list(df_plan.empty_data.columns)
        return _add_proj_expr_to_plan(
            df_plan.source, value_plan, key, out_columns=out_columns
        )

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


def get_isin_filter_plan(source_plan: LazyPlan, key_plan: LazyPlan) -> LazyPlan | None:
    """
    Pattern match df1[df1.A.isin(df2.B)] case and return a semi-join plan to implement
    it. Returns None if the plan pattern does not match.
    """
    from bodo.ext import plan_optimizer
    from bodo.pandas.plan import UnaryOpExpression

    # Support not in case like df1[~df1.A.isin(df2.B)] using anti-join
    is_anti = False
    if (
        is_single_projection(key_plan)
        and isinstance(
            (key_expr := get_proj_expr_single(key_plan)),
            UnaryOpExpression,
        )
        and key_expr.op == "__invert__"
        and isinstance(key_expr.source, LogicalComparisonJoin)
        and key_expr.source.join_type == plan_optimizer.CJoinType.MARK
    ):
        # Match df1[~df1.A.isin(df2.B)] and convert to df1[df1.A.isin(df2.B)] for
        # matching below but with anti-join flag set
        is_anti = True
        key_plan = LogicalProjection(
            key_plan.empty_data,
            key_expr.source,
            (key_expr.source_expr,) + key_plan.exprs[1:],
        )

    # Match df1.A.isin(df2.B) case which is a mark join generated in our Series.isin()
    if not (
        is_single_colref_projection(key_plan)
        and isinstance(key_plan.source, LogicalComparisonJoin)
        and key_plan.source.join_type == plan_optimizer.CJoinType.MARK
        and is_single_colref_projection(key_plan.source.left_plan)
        and key_plan.source.left_plan.source == source_plan
    ):
        return None

    left_key_ind = key_plan.source.left_plan.exprs[0].col_index
    planComparisonJoin = LogicalComparisonJoin(
        source_plan.empty_data,
        source_plan,
        key_plan.source.right_plan,
        plan_optimizer.CJoinType.ANTI if is_anti else plan_optimizer.CJoinType.INNER,
        [(left_key_ind, 0)],
    )

    # Ignore right column in output
    exprs = make_col_ref_exprs(
        list(
            range(
                len(source_plan.empty_data.columns)
                + get_n_index_arrays(source_plan.empty_data.index)
            )
        ),
        planComparisonJoin,
    )
    proj_plan = LogicalProjection(
        source_plan.empty_data,
        planComparisonJoin,
        exprs,
    )
    return proj_plan


def validate_on(val):
    """Validates single on-value"""
    if val is not None:
        if not (
            isinstance(val, str)
            or (isinstance(val, (list, tuple)) and all(isinstance(k, str) for k in val))
        ):
            raise ValueError(
                "only str, str list, str tuple, or None are supported for on, left_on and right_on values"
            )


def validate_keys(keys, df):
    """Utilizes set difference to check key membership in DataFrame df"""
    key_diff = set(keys).difference(set(df.columns))
    if len(key_diff) > 0:
        raise KeyError(
            f"merge(): invalid key {key_diff} for on/left_on/right_on\n"
            f"merge supports only valid column names {df.columns}"
        )


def validate_merge_spec(left, right, on, left_on, right_on, is_cross):
    """Check on, left_on and right_on values for type correctness
    (currently only str, str list, str tuple, or None are supported)
    and matching number of elements. If failed to validate, raise error.
    Also checks membership in left and right DFs to validate keys.
    """
    validate_on(on)
    validate_on(left_on)
    validate_on(right_on)

    if is_cross:
        if on is not None or left_on is not None or right_on is not None:
            raise ValueError(
                'Cannot specify "on", "left_on" or "right_on" for cross join.'
            )
        return [], []

    if on is None and left_on is None and right_on is None:
        # Join on common keys if keys not specified
        common_cols = on = tuple(set(left.columns).intersection(set(right.columns)))
        if len(common_cols) == 0:
            raise ValueError(
                "No common columns to perform merge on. "
                f"Merge options: left_on={left_on}, "
                f"right_on={right_on}"
            )
        left_on = right_on = common_cols
        return left_on, right_on

    elif on is not None:
        if left_on is not None or right_on is not None:
            raise ValueError(
                'Can only pass argument "on" OR "left_on" '
                'and "right_on", not a combination of both.'
            )
        left_on = right_on = maybe_make_list(on)

    elif (left_on is not None) ^ (right_on is not None):
        raise ValueError('Must pass both "left_on" and "right_on"')

    elif left_on is not None and right_on is not None:
        left_on, right_on = maybe_make_list(left_on), maybe_make_list(right_on)

    if len(left_on) != len(right_on):
        raise ValueError("len(right_on) must equal len(left_on)")

    validate_keys(left_on, left)
    validate_keys(right_on, right)

    return left_on, right_on


def _get_join_type_from_how(how: str) -> plan_optimizer.CJoinType:
    """Convert how string to DuckDB JoinType enum."""
    from bodo.ext import plan_optimizer

    if how == "inner":
        return plan_optimizer.CJoinType.INNER
    elif how == "left":
        return plan_optimizer.CJoinType.LEFT
    elif how == "right":
        return plan_optimizer.CJoinType.RIGHT
    elif how == "outer":
        return plan_optimizer.CJoinType.OUTER
    elif how == "cross":
        return plan_optimizer.CJoinType.INNER
    else:
        raise ValueError(f"Invalid join type: {how}")


wrap_module_functions_and_methods(sys.modules[__name__])
