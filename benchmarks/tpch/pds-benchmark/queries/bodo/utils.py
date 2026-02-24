import inspect
import typing as pt
import warnings

import pandas
from queries.common_utils import run_query_generic
from settings import Settings

import bodo.pandas
import bodo.spawn.spawner as spawner
from bodo.pandas import BodoDataFrame, BodoScalar, BodoSeries

if pt.TYPE_CHECKING:
    from collections.abc import Callable

settings = Settings()


def load_lineitem(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/lineitem.pq"
    df = pd.read_parquet(data_path)
    df["L_SHIPDATE"] = pd.to_datetime(df.L_SHIPDATE, format="%Y-%m-%d")
    df["L_RECEIPTDATE"] = pd.to_datetime(df.L_RECEIPTDATE, format="%Y-%m-%d")
    df["L_COMMITDATE"] = pd.to_datetime(df.L_COMMITDATE, format="%Y-%m-%d")
    return df


def load_part(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/part.pq"
    df = pd.read_parquet(data_path)
    return df


def load_orders(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/orders.pq"
    df = pd.read_parquet(data_path)
    df["O_ORDERDATE"] = pd.to_datetime(df.O_ORDERDATE, format="%Y-%m-%d")
    return df


def load_customer(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/customer.pq"
    df = pd.read_parquet(data_path)
    return df


def load_nation(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/nation.pq"
    df = pd.read_parquet(data_path)
    return df


def load_region(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/region.pq"
    df = pd.read_parquet(data_path)
    return df


def load_supplier(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/supplier.pq"
    df = pd.read_parquet(data_path)
    return df


def load_partsupp(data_folder: str, pd=bodo.pandas):
    data_path = data_folder + "/partsupp.pq"
    df = pd.read_parquet(data_path)
    return df


def run_query(query_number: int, q: "Callable"):
    if settings.run.bodo_use_pandas_backend:
        pandas.options.mode.copy_on_write = True
        backend = pandas
    else:
        backend = bodo.pandas

    if backend is bodo.pandas:
        # Filter performance-related warnings on workers
        spawner.submit_func_to_workers(lambda: warnings.filterwarnings("ignore"), [])

    if settings.run.io_type != "parquet":
        raise NotImplementedError(
            f"Bodo/Pandas queries currently require IO type parquet, got: {settings.run.io_type}"
        )

    def query_func():
        params = list(inspect.signature(q).parameters)
        args = []

        for param in params:
            if param == "scale_factor":
                args.append(settings.scale_factor)
            elif param == "pd":
                args.append(backend)
            else:
                args.append(
                    globals()[f"load_{param}"](
                        str(settings.dataset_base_dir), pd=backend
                    )
                )

        res = q(*args)

        if isinstance(res, (BodoDataFrame, BodoSeries, BodoScalar)):
            return res.execute_plan()
        return res

    try:
        run_query_generic(
            query_func,
            query_number,
            library_name="bodo",
            library_version=backend.__version__,
            query_checker=None,
        )
    except Exception as e:
        print(f"q{query_number} FAILED\n{e}")
