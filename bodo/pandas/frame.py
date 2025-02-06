import typing as pt
from collections.abc import Callable

import pandas as pd

import bodo
from bodo.pandas.array_manager import LazyArrayManager
from bodo.pandas.lazy_metadata import LazyMetadata
from bodo.pandas.lazy_wrapper import BodoLazyWrapper
from bodo.pandas.managers import LazyBlockManager, LazyMetadataMixin
from bodo.pandas.utils import get_lazy_manager_class
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

        to_parquet_wrapper(
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

        to_sql_wrapper(
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

        to_csv_wrapper(
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
            )

        to_json_wrapper(
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
