import pathlib
import tempfile
from collections.abc import Callable
from typing import Literal

import polars as pl
from queries.common_utils import (
    check_query_result_pl,
    get_table_path,
    run_query_generic,
)
from settings import Settings

settings = Settings()


def _scan_ds(table_name: str) -> pl.LazyFrame:
    path = get_table_path(table_name)

    if settings.run.io_type == "skip":
        return pl.read_parquet(path, rechunk=True).lazy()
    if settings.run.io_type == "parquet":
        # Bodo Change: rename columns
        return pl.scan_parquet(path).rename(str.lower)
    elif settings.run.io_type == "feather":
        return pl.scan_ipc(path)
    elif settings.run.io_type == "csv":
        return pl.scan_csv(path, try_parse_dates=True)
    else:
        msg = f"unsupported file type: {settings.run.io_type!r}"
        raise ValueError(msg)


def get_line_item_ds() -> pl.LazyFrame:
    return _scan_ds("lineitem")


def get_orders_ds() -> pl.LazyFrame:
    return _scan_ds("orders")


def get_customer_ds() -> pl.LazyFrame:
    return _scan_ds("customer")


def get_region_ds() -> pl.LazyFrame:
    return _scan_ds("region")


def get_nation_ds() -> pl.LazyFrame:
    return _scan_ds("nation")


def get_supplier_ds() -> pl.LazyFrame:
    return _scan_ds("supplier")


def get_part_ds() -> pl.LazyFrame:
    return _scan_ds("part")


def get_part_supp_ds() -> pl.LazyFrame:
    return _scan_ds("partsupp")


def _preload_engine(
    engine: pl.GPUEngine | Literal["in-memory", "streaming", "old-streaming"],
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # GPU engine has one-time lazy-loaded cost in IO, which we
        # remove from timings here.
        f = pathlib.Path(tmpdir) / "test.pq"
        df = pl.DataFrame({"a": [1]})
        df.write_parquet(f)
        pl.scan_parquet(f).collect(engine=engine)  # type: ignore[arg-type]


def obtain_engine_config() -> (
    pl.GPUEngine | Literal["in-memory", "streaming", "old-streaming"]
):
    if settings.run.polars_old_streaming:
        return "old-streaming"
    if settings.run.polars_streaming:
        return "streaming"
    if not settings.run.polars_gpu:
        return "in-memory"

    import cudf_polars
    import rmm
    from cudf_polars.callback import set_device
    from packaging import version

    if version.parse(cudf_polars.__version__) < version.Version("24.10"):
        import cudf._lib.pylibcudf as plc
    else:
        import pylibcudf as plc

    device = settings.run.polars_gpu_device
    mr_type = settings.run.use_rmm_mr
    with set_device(device):
        # Must make sure to create memory resource on the requested device
        free_memory, _ = rmm.mr.available_device_memory()
        # Pick an initial pool of around 80% of the free device
        # memory, must be multiple of 256
        initial_pool_size = 256 * (int(free_memory * 0.8) // 256)
        if mr_type == "cuda":
            mr = rmm.mr.CudaMemoryResource()
        elif mr_type == "cuda-pool":
            mr = rmm.mr.PoolMemoryResource(
                rmm.mr.CudaMemoryResource(), initial_pool_size=initial_pool_size
            )
        elif mr_type == "cuda-async":
            mr = rmm.mr.CudaAsyncMemoryResource(initial_pool_size=initial_pool_size)
        elif mr_type == "managed":
            mr = rmm.mr.ManagedMemoryResource()
        elif mr_type == "managed-pool":
            mr = rmm.mr.PrefetchResourceAdaptor(
                rmm.mr.PoolMemoryResource(
                    rmm.mr.ManagedMemoryResource(), initial_pool_size=initial_pool_size
                )
            )
        else:
            msg = "Unknown memory resource type"
            raise RuntimeError(msg)
        if mr_type in ("managed", "managed-pool"):
            for typ in [
                "column_view::get_data",
                "mutable_column_view::get_data",
                "gather",
                "hash_join",
            ]:
                plc.experimental.enable_prefetching(typ)

        return pl.GPUEngine(device=device, memory_resource=mr, raise_on_fail=True)


def run_query(query_number: int, query_func: Callable) -> None:
    streaming = settings.run.polars_old_streaming
    new_streaming = settings.run.polars_streaming
    eager = settings.run.polars_eager
    gpu = settings.run.polars_gpu

    if sum([eager, streaming, new_streaming, gpu]) > 1:
        msg = "Please specify at most one of eager, streaming, new_streaming, or gpu"
        raise ValueError(msg)

    engine = obtain_engine_config()

    # Eager load engine backend, so we don't time that.
    _preload_engine(engine)

    def query_final():
        lf: pl.LazyFrame = query_func()
        return lf.collect(
            streaming=streaming,
            new_streaming=new_streaming,
            no_optimization=eager,
            engine=engine,
        )

    if gpu:
        library_name = f"polars-gpu-{settings.run.use_rmm_mr}"
    elif eager:
        library_name = "polars-eager"
    else:
        library_name = "polars"

    try:
        run_query_generic(
            query_final,
            query_number,
            library_name,
            library_version=pl.__version__,
            query_checker=check_query_result_pl,
        )
    except Exception as e:
        print(f"q{query_number} FAILED\n{e}")
