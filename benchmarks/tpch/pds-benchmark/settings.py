import os
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

IoType: TypeAlias = Literal["skip", "parquet", "feather", "csv", "duckdb"]


# Set via PATH_<NAME>
class Paths(BaseSettings):
    answers: Path = Path("data/answers")
    tables: Path = Path("data/tables").absolute()
    # BODO Change: override computed path
    data_folder: str | None = None

    timings: Path = Path("output/run")
    timings_filename: str = "timings.csv"

    plots: Path = Path("output/plot")

    model_config = SettingsConfigDict(
        env_prefix="path_", env_file=".env", extra="ignore"
    )


# Set via RUN_<NAME>
class Run(BaseSettings):
    io_type: IoType = "parquet"

    iterations: int = 1
    log_timings: bool = False
    show_results: bool = False
    check_results: bool = False  # Only available for SCALE_FACTOR=1

    polars_show_plan: bool = False
    polars_eager: bool = False
    polars_old_streaming: bool = False
    polars_streaming: bool = False
    polars_cloud: bool = False
    polars_gpu: bool = False  # Use GPU engine?
    polars_gpu_device: int = 0  # The GPU device to run on for polars GPU
    # Which style of GPU memory resource to use
    # cuda -> cudaMalloc
    # cuda-pool -> Pool suballocator wrapped around cudaMalloc
    # managed -> cudaMallocManaged
    # managed-pool -> Pool suballocator wrapped around cudaMallocManaged
    # cuda-async -> cudaMallocAsync (comes with pool)
    # See https://docs.rapids.ai/api/rmm/stable/ for details on RMM memory resources
    use_rmm_mr: Literal[
        "cuda", "cuda-pool", "managed", "managed-pool", "cuda-async"
    ] = "cuda-async"

    modin_memory: int = 8_000_000_000  # Tune as needed for optimal performance

    spark_driver_memory: str = "2g"  # Tune as needed for optimal performance
    spark_executor_memory: str = "1g"  # Tune as needed for optimal performance
    spark_log_level: str = "ERROR"

    pandas_gpu: bool = False  # Use cudf.pandas to run pandas benchmarks
    bodo_use_pandas_backend: bool = False  # Use pandas backend with bodo queries.

    @computed_field  # type: ignore[prop-decorator]
    @property
    def include_io(self) -> bool:
        return self.io_type != "skip"

    model_config = SettingsConfigDict(
        env_prefix="run_", env_file=".env", extra="ignore"
    )


class Plot(BaseSettings):
    show: bool = False
    n_queries: int = 7
    y_limit: float | None = None

    model_config = SettingsConfigDict(
        env_prefix="plot_", env_file=".env", extra="ignore"
    )


class Settings(BaseSettings):
    scale_factor: float = 1.0
    num_batches: int | None = None

    paths: Paths = Paths()
    plot: Plot = Plot()
    run: Run = Run()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_base_dir(self) -> str:
        # BODO Change: override computed path if provided.
        if self.paths.data_folder is not None:
            return self.paths.data_folder
        return os.path.join(self.paths.tables, f"scale-{self.scale_factor}")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
