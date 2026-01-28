import os

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import bodo
from bodo.tests.iceberg_database_helpers import spark_reader
from bodo.tests.iceberg_database_helpers.simple_tables import (
    TABLE_MAP as SIMPLE_TABLES_MAP,
)
from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    PartitionField,
    create_iceberg_table,
    get_spark,
)
from bodo.tests.utils import (
    _gather_output,
    _test_equal_guard,
    gen_nonascii_list,
)

pytestmark = [
    pytest.mark.iceberg,
    pytest.mark.skip(
        reason="[BSE-4569] MERGE INTO with PyIceberg is not supported yet"
    ),
]


@pytest.mark.slow
def test_merge_into_cow_write_api(
    iceberg_database,
    iceberg_table_conn,
):
    import bodo_iceberg_connector

    comm = MPI.COMM_WORLD
    bodo.barrier()

    db_schema, warehouse_loc = iceberg_database()
    # Should always only run this test on rank O
    if bodo.get_rank() != 0:
        passed = comm.bcast(False)
        if not passed:
            raise Exception("Exception on Rank 0")
        return

    passed = True
    try:
        # Create a table to work off of
        table_name = "MERGE_INTO_COW_WRITE_API"
        if bodo.get_rank() == 0:
            df = pd.DataFrame({"A": [1, 2, 3, 4]})
            create_iceberg_table(df, [("A", "long", True)], table_name)

        conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
        conn = bodo.io.iceberg.format_iceberg_conn(conn)

        # Get relevant read info from connector
        snapshot_id = bodo_iceberg_connector.bodo_connector_get_current_snapshot_id(
            conn, db_schema, table_name
        )
        old_fnames = [
            x.orig_path
            for x in bodo_iceberg_connector.get_bodo_parquet_info(
                conn,
                db_schema,
                table_name,
                None,
            )[0]
        ]

        # Write new file into warehouse location
        new_fname = "new_file.parquet"
        df_new = pd.DataFrame({"A": [5, 6, 7, 8]})
        df_new.to_parquet(
            os.path.join(warehouse_loc, db_schema, table_name, "data", new_fname)
        )

        # Commit the MERGE INTO COW operation
        success = bodo_iceberg_connector.commit_merge_cow(
            conn,
            db_schema,
            table_name,
            warehouse_loc,
            old_fnames,
            [new_fname],
            [0],
            [{"rowCount": 4}],
            snapshot_id,
        )
        assert success, "MERGE INTO Commit Operation Failed"

        # See if the reported files to read is only the new file
        new_fnames = [
            x.orig_path
            for x in bodo_iceberg_connector.get_bodo_parquet_info(
                conn,
                db_schema,
                table_name,
                None,
            )[0]
        ]

        assert len(new_fnames) == 1 and new_fnames[0] == os.path.join(
            db_schema, table_name, "data", new_fname
        )

    except Exception as e:
        passed = False
        raise e
    finally:
        passed = comm.bcast(passed)


@pytest.mark.slow
def test_merge_into_cow_write_api_partitioned(
    iceberg_database,
    iceberg_table_conn,
):
    """
    Test the Iceberg Connectors MERGE INTO COW Write Operation
    with partitioned Iceberg tables
    """
    import bodo_iceberg_connector

    comm = MPI.COMM_WORLD
    bodo.barrier()

    db_schema, warehouse_loc = iceberg_database()

    # Should always only run this test on rank O
    if bodo.get_rank() != 0:
        passed = comm.bcast(False)
        if not passed:
            raise Exception("Exception on Rank 0")
        return

    passed = True
    try:
        # Create a table to work off of
        table_name = "MERGE_INTO_COW_WRITE_API_PARTITIONED"
        df, sql_schema = SIMPLE_TABLES_MAP["SIMPLE_PRIMITIVES_TABLE"]
        create_iceberg_table(
            df,
            sql_schema,
            table_name,
            par_spec=[PartitionField("C", "identity", -1)],
        )

        conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
        conn = bodo.io.iceberg.format_iceberg_conn(conn)

        # Get relevant read info from connector
        snapshot_id = bodo_iceberg_connector.bodo_connector_get_current_snapshot_id(
            conn, db_schema, table_name
        )
        old_fnames = [
            x.orig_path
            for x in bodo_iceberg_connector.get_bodo_parquet_info(
                conn,
                db_schema,
                table_name,
                None,
            )[0]
        ]

        # Write new partitioned files into warehouse location
        new_paths = []
        record_counts = []

        # Write a new data file to the "warehouse" to use in the operation
        # Note, Bodo will essentially do this internally at Parquet Write
        for part_val, name in [(True, "true"), (False, "false"), (None, "null")]:
            sub_df = df[df["C"] == part_val]
            if len(sub_df) == 0:
                continue

            record_counts.append(len(sub_df))
            new_path = os.path.join(f"C={name}", "new_file.parquet")
            new_paths.append(new_path)

            sub_df.to_parquet(
                os.path.join(warehouse_loc, db_schema, table_name, "data", new_path)
            )

        # Commit the MERGE INTO COW operation
        success = bodo_iceberg_connector.commit_merge_cow(
            conn,
            db_schema,
            table_name,
            warehouse_loc,
            old_fnames,
            new_paths,
            [0] * len(new_paths),
            [{"rowCount": x} for x in record_counts],
            snapshot_id,
        )
        assert success, "MERGE INTO Commit Operation Failed"

        # See if the reported files to read is only the new file
        out_fnames = [
            x.orig_path
            for x in bodo_iceberg_connector.get_bodo_parquet_info(
                conn,
                db_schema,
                table_name,
                None,
            )[0]
        ]

        assert set(out_fnames) == {
            os.path.join(db_schema, table_name, "data", path) for path in new_paths
        }

    except Exception as e:
        passed = False
        raise e
    finally:
        passed = comm.bcast(passed)


@pytest.mark.slow
def test_merge_into_cow_write_api_snapshot_check(
    iceberg_database,
    iceberg_table_conn,
):
    import bodo_iceberg_connector

    comm = MPI.COMM_WORLD
    bodo.barrier()

    # Note that for the connector, conn_str and warehouse_loc are the same
    db_schema, warehouse_loc = iceberg_database()

    # Should always only run this test on rank O
    if bodo.get_rank() != 0:
        passed = comm.bcast(False)
        if not passed:
            raise Exception("Exception on Rank 0")
        return

    passed = True
    try:
        # Create a table to work off of
        table_name = "MERGE_INTO_COW_WRITE_API_SNAPSHOT_CHECK"
        df = pd.DataFrame({"A": [1, 2, 3, 4]})
        create_iceberg_table(df, [("A", "long", True)], table_name)

        conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
        # Format the connection string since we don't go through pd.read_sql_table
        conn = bodo.io.iceberg.format_iceberg_conn(conn)

        # Get relevant read info from connector
        snapshot_id = bodo_iceberg_connector.bodo_connector_get_current_snapshot_id(
            conn, db_schema, table_name
        )
        old_fnames = [
            x.orig_path
            for x in bodo_iceberg_connector.get_bodo_parquet_info(
                conn,
                db_schema,
                table_name,
                None,
            )[0]
        ]

        # Write new file into warehouse location
        new_fname = "new_file.parquet"
        df_new = pd.DataFrame({"A": [5, 6, 7, 8]})
        df_new.to_parquet(
            os.path.join(warehouse_loc, db_schema, table_name, "data", new_fname)
        )

        # Update the current snapshot ID by appending data to the table
        spark = get_spark()
        spark.sql(
            f"INSERT INTO hadoop_prod.{db_schema}.{table_name} VALUES (10), (11), (12), (13)"
        )

        # Attempt to commit a MERGE INTO operation with the old snapshot id
        # Expect it return False (and prints error)
        success = bodo_iceberg_connector.commit_merge_cow(
            conn,
            db_schema,
            table_name,
            warehouse_loc,
            old_fnames,
            [new_fname],
            [0],
            [{"rowCount": 4}],
            snapshot_id,
        )
        assert not success, "MERGE INTO Commit Operation should not have succeeded"

    except Exception as e:
        passed = False
        raise e
    finally:
        passed = comm.bcast(passed)


@pytest.mark.slow
def test_merge_into_cow_simple_e2e(iceberg_database, iceberg_table_conn):
    """
    Tests a simple end to end example of reading with _bodo_merge_into, performing some modifications,
    and that writing the changes back with iceberg_merge_cow_py
    """
    import bodo.io.iceberg.merge_into

    comm = MPI.COMM_WORLD

    # Create a table to work off of
    table_name = "MERGE_INTO_COW_WRITE_SIMPLE_E2E"
    if bodo.get_rank() == 0:
        df = pd.DataFrame({"A": np.arange(10)})
        create_iceberg_table(df, [("A", "long", True)], table_name)
    bodo.barrier()

    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl1(table_name, conn, db_schema):
        df, old_fnames, old_snapshot_id = pd.read_sql_table(
            table_name, conn, db_schema, _bodo_merge_into=True
        )  # type: ignore
        df_update = df[df.A > 5]
        df_update["A"] = df_update["A"] * -10

        df = df.merge(df_update, on="_BODO_ROW_ID", how="left")
        df["A"] = df["A_y"].fillna(df["A_x"])

        df = df.drop(columns=["_BODO_ROW_ID", "A_y", "A_x"])
        bodo.io.iceberg.merge_into.iceberg_merge_cow_py(
            table_name, conn, db_schema, df, old_snapshot_id, old_fnames
        )
        return old_snapshot_id

    old_snapshot_id = bodo.jit(impl1)(table_name, conn, db_schema)
    bodo.barrier()

    passed = True
    if bodo.get_rank() == 0:
        # We had issues with spark caching previously, this alleviates those issues
        spark = get_spark()
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")

        snapshot_id_table = spark.sql(
            f"""
            select snapshot_id from hadoop_prod.{db_schema}.{table_name}.history order by made_current_at DESC
            """
        ).toPandas()

        # We expect to see two snapshot id's, the first from the creation/insertion of the initial values into
        # the table, and the second from when we do the merge into.
        passed = len(snapshot_id_table) == 2 and (
            old_snapshot_id == snapshot_id_table.iloc[1, 0]
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Snapshot ID's do not match expected output"

    passed = True
    if bodo.get_rank() == 0:
        bodo_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
        # See comment above, expected may change if we spark version changes,
        # but we will always see the negative values (-60, -70, -80, -90),
        # and never see values > 5
        expected = pd.DataFrame({"A": [0, 1, 2, 3, 4, 5, -60, -70, -80, -90]})
        passed = _test_equal_guard(
            bodo_out,
            expected,
            check_dtype=False,
            sort_output=True,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Bodo function output doesn't match expected output"


@pytest.mark.slow
def test_merge_into_cow_simple_e2e_partitions(iceberg_database, iceberg_table_conn):
    """
    Tests a simple end to end example of reading with _bodo_merge_into, performing some modifications,
    and then writing the changes back with iceberg_merge_cow_py, this time, on a partitioned table,
    where we should only read/write back certain files.
    """
    import bodo.io.iceberg.merge_into

    comm = MPI.COMM_WORLD

    # Create a table to work off of
    table_name = "MERGE_INTO_COW_WRITE_SIMPLE_E2E_PARTITIONS"

    orig_df = pd.DataFrame(
        {
            "A": list(np.arange(8)) * 8,
            "B": gen_nonascii_list(64),
            "C": np.arange(64),
            "D": pd.date_range("2017-01-03", periods=8).repeat(8).tz_localize("UTC"),
        }
    )

    if bodo.get_rank() == 0:
        create_iceberg_table(
            orig_df,
            [
                ("A", "long", True),
                ("B", "string", True),
                ("C", "long", True),
                ("D", "timestamp", True),
            ],
            table_name,
            par_spec=[
                PartitionField("A", "identity", -1),
                PartitionField("D", "days", -1),
            ],
        )
    bodo.barrier()

    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl1(table_name, conn, db_schema):
        df, old_fnames, old_snapshot_id = pd.read_sql_table(
            table_name, conn, db_schema, _bodo_merge_into=True
        )  # type: ignore
        # df is partitioned on int column A and the date column D
        df_update = df[df.A > 4][["_BODO_ROW_ID", "A"]]
        df_update["A"] = df_update["A"] * -10

        df = df.merge(df_update, on="_BODO_ROW_ID", how="left")
        df["A"] = df["A_y"].fillna(df["A_x"])

        df = df[["A", "B", "C", "D"]]
        bodo.io.iceberg.merge_into.iceberg_merge_cow_py(
            table_name, conn, db_schema, df, old_snapshot_id, old_fnames
        )
        return old_snapshot_id

    old_snapshot_id = bodo.jit(impl1)(table_name, conn, db_schema)
    bodo.barrier()

    passed = False
    if bodo.get_rank() == 0:
        spark = get_spark()
        # We had issues with spark caching previously, this alleviates those issues
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")

        snapshot_id_table = spark.sql(
            f"""
            select snapshot_id from hadoop_prod.{db_schema}.{table_name}.history order by made_current_at DESC
            """
        ).toPandas()

        # We expect to see two snapshot id's, the first from the creation/insertion of the initial values into
        # the table, and the second from when we do the merge into.
        passed = len(snapshot_id_table) == 2 and (
            old_snapshot_id == snapshot_id_table.iloc[1, 0]
        )

    passed = comm.bcast(passed)
    assert passed, "Snapshot ID's do not match expected output"

    # Construct Expected Output
    expected_out = orig_df.copy()
    expected_out.loc[expected_out.A > 4, ["A"]] = (
        expected_out[expected_out.A > 4]["A"] * -10
    )
    expected_out = expected_out.sort_values(by="C", ascending=True)

    passed = False
    err = None
    if bodo.get_rank() == 0:
        try:
            spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
            spark_out = spark_out.sort_values(by="C", ascending=True)
            spark_out["D"] = (
                spark_out["D"].astype("datetime64[ns]").dt.tz_localize("UTC")
            )

            passed = bool(
                _test_equal_guard(
                    spark_out,
                    expected_out,
                    sort_output=True,
                    check_dtype=False,
                    reset_index=True,
                )
            )
        except Exception as e:
            err = e

    passed, err = comm.bcast((passed, err))
    if isinstance(err, Exception):
        raise err
    assert passed, "Spark output doesn't match expected output"

    # Validate Bodo read output
    bodo_out = bodo.jit(all_returns_distributed=True)(
        lambda: pd.read_sql_table(table_name, conn, db_schema)
    )()  # type: ignore
    bodo_out = _gather_output(bodo_out)

    passed = False
    if bodo.get_rank() == 0:
        passed = bool(
            _test_equal_guard(
                expected_out,
                bodo_out,
                sort_output=True,
                check_dtype=False,
                reset_index=True,
            )
        )

    passed = comm.bcast(passed)
    assert passed, "Bodo read output doesn't match expected output"
