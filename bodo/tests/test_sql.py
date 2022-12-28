# Copyright (C) 2022 Bodo Inc. All rights reserved.
import io
import re
import string
import traceback

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import bodo
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
    get_start_end,
    oracle_user_pass_and_hostname,
    reduce_sum,
    sql_user_pass_and_hostname,
)
from bodo.utils.testing import ensure_clean_mysql_psql_table


@pytest.mark.parametrize("chunksize", [None, 4])
def test_write_sql_aws(chunksize, memory_leak_check):
    """This test for a write down on a SQL database"""

    def test_impl_write_sql(df, table_name, conn, chunksize):
        df.to_sql(table_name, conn, if_exists="replace", chunksize=chunksize)

    def test_specific_dataframe(test_impl, is_distributed, df_in, chunksize):
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"

        with ensure_clean_mysql_psql_table(conn, "test_table_ABCD") as table_name:
            bodo_impl = bodo.jit(all_args_distributed_block=is_distributed)(test_impl)
            if is_distributed:
                start, end = get_start_end(len(df_in))
                df_input = df_in.iloc[start:end]
            else:
                df_input = df_in
            bodo_impl(df_input, table_name, conn, chunksize)
            bodo.barrier()
            passed = 1
            npes = bodo.get_size()
            if bodo.get_rank() == 0:
                try:
                    df_load = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                    # The writing does not preserve the order a priori
                    l_cols = df_in.columns.to_list()
                    df_in_sort = df_in.sort_values(l_cols).reset_index(drop=True)
                    df_load_sort = (
                        df_load[l_cols].sort_values(l_cols).reset_index(drop=True)
                    )
                    pd.testing.assert_frame_equal(
                        df_load_sort, df_in_sort, check_column_type=False
                    )
                except Exception as e:
                    print("".join(traceback.format_exception(None, e, e.__traceback__)))
                    passed = 0
            n_passed = reduce_sum(passed)
            n_pes = bodo.get_size()
            assert n_passed == n_pes, "test_write_sql_aws failed"
            bodo.barrier()

    rng = np.random.default_rng(5)
    len_list = 20
    list_int = rng.choice(10, len_list)
    list_double = rng.choice([4.0, np.nan], len_list)
    list_datetime = pd.date_range("2001-01-01", periods=len_list)
    df1 = pd.DataFrame({"A": list_int, "B": list_double, "C": list_datetime})
    test_specific_dataframe(test_impl_write_sql, False, df1, chunksize)
    test_specific_dataframe(test_impl_write_sql, True, df1, chunksize)


# TODO: Add memory_leak_check when bug is resolved.
def test_sql_if_exists_fail_errorchecking():
    """
    This test with the option if_exists="fail" (which is the default)
    It will fail because the database is already present.
    """
    # Create the Table
    conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
    with ensure_clean_mysql_psql_table(conn, "test_table_already_exists") as table_name:
        # Initialize Table with Empty Columns
        if bodo.get_rank() == 0:
            df_empty = pd.DataFrame({"A": [], "B": [], "C": []})
            df_empty.to_sql(table_name, conn)
        bodo.barrier()

        def test_impl_fails(df, table_name, conn):
            df.to_sql(table_name, conn)

        rng = np.random.default_rng(5)
        len_list = 20
        list_int = rng.integers(1, 10, len_list)
        list_double = rng.choice([4.0, np.nan], len_list, p=[0.33, 0.67])
        list_datetime = pd.date_range("2001-01-01", periods=len_list)
        df1 = pd.DataFrame({"A": list_int, "B": list_double, "C": list_datetime})

        bodo_impl = bodo.jit(all_args_distributed_block=True)(test_impl_fails)
        #    with pytest.raises(ValueError, match="Table .* already exists"):
        with pytest.raises(ValueError, match="error in to_sql.* operation"):
            bodo_impl(df1, table_name, conn)


def test_sql_hardcoded_aws(memory_leak_check):
    """This test for an hardcoded request and connection"""

    def test_impl_hardcoded():
        sql_request = "select * from employees"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl_hardcoded, ())


def test_read_sql_hardcoded_time_offset_aws(memory_leak_check):
    """This test does not pass because the type of dates is not supported"""

    def test_impl_offset():
        sql_request = "select * from employees limit 1000 offset 4000"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl_offset, ())


def test_read_sql_hardcoded_limit_aws(memory_leak_check):
    def test_impl_limit():
        sql_request = "select * from employees limit 1000"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl_limit, ())


def test_sql_limit_inference():
    f = bodo.ir.sql_ext.req_limit
    # Simple query
    sql_request = "select * from employees limit 1000"
    assert f(sql_request) == 1000
    # White space
    sql_request = "select * from employees limit 1000   "
    assert f(sql_request) == 1000
    # White space
    sql_request = "select * from employees limit 1000 \n\n\n\n  "
    assert f(sql_request) == 1000

    # Check that we do not match with an offset
    sql_request = "select * from employees limit 1, 1000"
    assert f(sql_request) is None
    sql_request = "select * from employees limit 1000 offset 1"
    assert f(sql_request) is None

    # Check that we select the right limit in a nested query
    sql_request = "with table1 as (select * from employees limit 1000) select * from table1 limit 500"
    assert f(sql_request) == 500

    # Check that we don't select an inner limit
    sql_request = "select * from employees, (select table1.A, table2.B from table1 FULL OUTER join table2 on table1.A = table2.A limit 1000)"
    assert f(sql_request) is None


def test_limit_inferrence_small_table(memory_leak_check):
    """
    Checks that a query where the limit is much larger than the size
    of the table succeeds. We create a very small table and then set
    the limit to be much greater than the table size.
    """
    comm = MPI.COMM_WORLD

    conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
    with ensure_clean_mysql_psql_table(conn) as table_name:
        df = pd.DataFrame({"A": [1.12, 1.1] * 5, "B": [213, -7] * 5})
        # Create the table once.
        write_err = None
        if bodo.get_rank() == 0:
            try:
                df.to_sql(table_name, conn, if_exists="replace")
            except Exception as e:
                write_err = e
        write_err = comm.bcast(write_err)
        if isinstance(write_err, Exception):
            raise write_err

        def test_impl_limit(conn):
            """
            Test receiving the table with limit 1000 while there are only
            10 rows.
            """
            sql_request = f"select A from `{table_name}` limit 1000"
            frame = pd.read_sql(sql_request, conn)
            return frame

        check_func(test_impl_limit, (conn,))


def test_sql_single_column(memory_leak_check):
    """
    Test that loading using a single column has a correct result.
    This can break if dead column elimination is applied incorrectly.
    """
    comm = MPI.COMM_WORLD

    def write_sql(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace")

    conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
    with ensure_clean_mysql_psql_table(conn) as table_name:

        query1 = f"select A, B, C from `{table_name}`"
        query2 = f"select * from `{table_name}`"

        df = pd.DataFrame(
            {"A": [1.12, 1.1] * 5, "B": [213, -7] * 5, "C": [31, 247] * 5}
        )
        # Create the table once.
        write_err = None
        if bodo.get_rank() == 0:
            try:
                write_sql(df, table_name, conn)
            except Exception as e:
                write_err = e
        write_err = comm.bcast(write_err)
        if isinstance(write_err, Exception):
            raise write_err

        def test_impl1(conn):
            sql_request = query1
            frame = pd.read_sql(sql_request, conn)
            return frame["B"]

        def test_impl2(conn):
            sql_request = query2
            frame = pd.read_sql(sql_request, conn)
            return frame["B"]

        check_func(test_impl1, (conn,), check_dtype=False)
        check_func(test_impl2, (conn,), check_dtype=False)


def test_sql_use_index_column(memory_leak_check):
    """
    Test that loading a single and index using index_col has a correct result.
    This can break if dead column elimination is applied incorrectly.
    """
    comm = MPI.COMM_WORLD

    def write_sql(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace")

    conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"

    with ensure_clean_mysql_psql_table(conn) as table_name:
        query = f"select A, B, C from `{table_name}`"

        df = pd.DataFrame(
            {"A": [1.12, 1.1] * 5, "B": [213, -7] * 5, "C": [31, 247] * 5}
        )
        # Create the table once.
        write_err = None
        if bodo.get_rank() == 0:
            try:
                write_sql(df, table_name, conn)
            except Exception as e:
                write_err = e
        write_err = comm.bcast(write_err)
        if isinstance(write_err, Exception):
            raise write_err

        def test_impl(conn):
            sql_request = query
            frame = pd.read_sql(sql_request, conn, index_col="A")
            return frame["B"]

        check_func(test_impl, (conn,), check_dtype=False)
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(test_impl)(conn)
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['B', 'A']")


def test_read_sql_hardcoded_twocol_aws(memory_leak_check):
    """Selecting two columns without dates"""

    def test_impl_hardcoded_twocol():
        sql_request = "select first_name,last_name from employees"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl_hardcoded_twocol, ())


def test_sql_argument_passing(memory_leak_check):
    """Test passing SQL query and connection as arguments"""

    def test_impl_arg_passing(sql_request, conn):
        df = pd.read_sql(sql_request, conn)
        return df

    sql_request = "select * from employees"
    conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
    check_func(test_impl_arg_passing, (sql_request, conn))


def test_read_sql_column_function(memory_leak_check):
    """
    Test a SQL query that uses an unaliased function.
    """
    comm = MPI.COMM_WORLD

    def write_sql(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace")

    conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
    with ensure_clean_mysql_psql_table(conn) as table_name:
        df = pd.DataFrame({"A": [1.12, 1.1] * 5, "B": [213, -7] * 5})
        # Create the table once.
        write_err = None
        if bodo.get_rank() == 0:
            try:
                write_sql(df, table_name, conn)
            except Exception as e:
                write_err = e
        write_err = comm.bcast(write_err)
        if isinstance(write_err, Exception):
            raise write_err

        def test_impl(conn):
            sql_request = f"select B, count(*) from `{table_name}` group by B"
            frame = pd.read_sql(sql_request, conn)
            return frame

        check_func(test_impl, (conn,), check_dtype=False)


# ---------------------Oracle Database------------------------#
# Queries used from
# https://www.oracle.com/news/connect/run-sql-data-queries-with-pandas.html


@pytest.mark.slow
def test_oracle_read_sql_basic(memory_leak_check):
    """Test simple SQL query with Oracle DB"""

    def impl():
        sql_request = "select * from orders"
        conn = "oracle+cx_oracle://" + oracle_user_pass_and_hostname + "/ORACLE"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl, ())

    def impl2():
        sql_request = "select pono, ordate from orders"
        conn = "oracle+cx_oracle://" + oracle_user_pass_and_hostname + "/ORACLE"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl2, ())

    def impl3():
        sql_request = "select pono from orders"
        conn = "oracle+cx_oracle://" + oracle_user_pass_and_hostname + "/ORACLE"
        frame = pd.read_sql(sql_request, conn)
        return frame

    # [left]:  Int64  vs. [right]: int64
    check_func(impl3, (), check_dtype=False)


@pytest.mark.slow
def test_oracle_read_sql_count(memory_leak_check):
    """Test SQL query count(*) and a single column Oracle DB"""

    conn = "oracle+cx_oracle://" + oracle_user_pass_and_hostname + "/ORACLE"

    def write_sql(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace")

    table_name = "test_small_table"

    df = pd.DataFrame({"A": [1.12, 1.1] * 5, "B": [213, -7] * 5})
    # Create the table once.
    if bodo.get_rank() == 0:
        write_sql(df, table_name, conn)
    bodo.barrier()

    def test_impl(conn):
        sql_request = "select B, count(*) from test_small_table group by B"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl, (conn,), check_dtype=False)


@pytest.mark.slow
def test_oracle_read_sql_join(memory_leak_check):
    """Test SQL query join Oracle DB"""

    def impl():
        sql_request = """
        SELECT
            ordate,
            empl,
            price*quantity*(1-discount/100) AS total,
            price*quantity*(discount/100) AS off
            FROM orders INNER JOIN details
            ON orders.pono = details.pono
            ORDER BY ordate
        """
        conn = "oracle+cx_oracle://" + oracle_user_pass_and_hostname + "/ORACLE"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl, ())


@pytest.mark.slow
def test_oracle_read_sql_gb(memory_leak_check):
    """Test SQL query group by, column alias, and round Oracle DB"""

    def impl():
        sql_request = """
        SELECT
            brand,
            ROUND(AVG(price),2) AS MEAN
            FROM details
            GROUP BY brand
        """
        conn = "oracle+cx_oracle://" + oracle_user_pass_and_hostname + "/ORACLE"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl, ())


@pytest.mark.slow
@pytest.mark.parametrize("is_distributed", [True, False])
def test_write_sql_oracle(is_distributed, memory_leak_check):
    """Test to_sql with Oracle database
    Data is compared vs. original DF
    """

    def test_impl_write_sql(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace")

    rng = np.random.default_rng(5)
    len_list = 20
    list_int = rng.integers(1, 10, len_list)
    list_double = rng.choice([4.0, np.nan], len_list, p=[0.33, 0.67])
    list_datetime = pd.date_range("2001-01-01", periods=len_list)
    df_in = pd.DataFrame({"a": list_int, "b": list_double, "c": list_datetime})

    table_name = "to_sql_table"
    if is_distributed:
        start, end = get_start_end(len(df_in))
        df_input = df_in.iloc[start:end]
    else:
        df_input = df_in
    conn = "oracle+cx_oracle://" + oracle_user_pass_and_hostname + "/ORACLE"
    bodo.jit(all_args_distributed_block=is_distributed)(test_impl_write_sql)(
        df_input, table_name, conn
    )
    bodo.barrier()
    passed = 1
    if bodo.get_rank() == 0:
        try:
            # to_sql adds index column by default. Setting it here explicitly.
            df_load = pd.read_sql(
                "select * from " + table_name, conn, index_col="index"
            )
            # The writing does not preserve the order a priori
            l_cols = df_in.columns.to_list()
            df_in_sort = df_in.sort_values(l_cols).reset_index(drop=True)
            df_load_sort = df_load[l_cols].sort_values(l_cols).reset_index(drop=True)
            pd.testing.assert_frame_equal(
                df_load_sort, df_in_sort, check_column_type=False
            )
        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0
    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_write_sql_oracle failed"
    bodo.barrier()


def test_to_sql_invalid_password(memory_leak_check):
    """
    Tests df.to_sql when writing with an invalid password
    and thus triggering an exception. This checks that a
    hang won't occur if df.to_sql raises an exception.
    """

    @bodo.jit
    def impl(df):
        return df.to_sql(
            "mytable",
            "mysql+pymysql://admin:GARBAGEPASSWORD@bodo-engine-ci.copjdp5mkwpk.us-east-2.rds.amazonaws.com",
        )

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(ValueError, match=re.escape("error in to_sql() operation")):
        impl(df)


@pytest.mark.slow
def test_mysql_show(memory_leak_check):
    """Test MySQL: SHOW query"""

    def impl():
        sql_request = "show databases"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl, (), is_out_distributed=False)


@pytest.mark.skip(reason="bad internal function error only here.")
def test_mysql_describe(memory_leak_check):
    """Test MySQL: DESCRIBE query"""

    def impl():
        sql_request = "DESCRIBE salaries"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl, (), is_out_distributed=False)

    def impl():
        sql_request = "DESC salaries"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl, (), is_out_distributed=False)


@pytest.mark.slow
@pytest.mark.skip(reason="disable all postgreSQL tests for now")
def test_postgre_show(memory_leak_check):
    """Test PostgreSQL: SHOW query"""

    def impl():
        sql_request = "show all"
        conn = "postgresql+psycopg2://" + postgres_user_pass_and_hostname + "/TEST_DB"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl, (), is_out_distributed=False)


# ---------------------PostgreSQL Database------------------------#
# Queries used from
# https://www.postgresqltutorial.com/postgresql-select/

postgres_user_pass_and_hostname = "bodo:edJEh6RCUWMefuoZXTIy@bodo-postgre-test.copjdp5mkwpk.us-east-2.rds.amazonaws.com"


@pytest.mark.slow
@pytest.mark.skip(reason="disable all postgreSQL tests for now")
def test_postgres_read_sql_basic(memory_leak_check):
    """Test simple SQL query with PostgreSQL DBMS"""

    conn = "postgresql+psycopg2://" + postgres_user_pass_and_hostname + "/TEST_DB"

    def impl(conn):
        sql_request = "select * from actor"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl, (conn,))

    def impl2(conn):
        sql_request = "SELECT first_name, last_name, email FROM customer"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl2, (conn,))

    def impl3(conn):
        sql_request = (
            "SELECT first_name || ' ' || last_name AS full_name, email FROM customer"
        )
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl3, (conn,))

    def impl4(conn):
        sql_request = "SELECT actor_id from actor"
        frame = pd.read_sql(sql_request, conn)
        return frame

    # [left]:  Int64  vs. [right]: int64
    check_func(impl4, (conn,), check_dtype=False)


@pytest.mark.slow
@pytest.mark.skip(reason="disable all postgreSQL tests for now")
def test_postgres_read_sql_count(memory_leak_check):
    """Test SQL query count(*) and a single column PostgreSQL DB"""

    conn = "postgresql+psycopg2://" + postgres_user_pass_and_hostname + "/TEST_DB"

    def test_impl(conn):
        sql_request = """
        SELECT staff_id,
	            COUNT(*)
        FROM payment
        GROUP BY staff_id
        """
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl, (conn,), check_dtype=False)


@pytest.mark.slow
@pytest.mark.skip(reason="disable all postgreSQL tests for now")
def test_postgres_read_sql_join(memory_leak_check):
    """Test SQL query join PostgreSQL DB"""

    conn = "postgresql+psycopg2://" + postgres_user_pass_and_hostname + "/TEST_DB"
    # 1 and 2 Taken from
    # https://github.com/YashMotwani/Sakila-DVD-Rental-database-Analysis/blob/master/Yash_Investigate_Relational_Database.txt
    def impl(conn):
        sql_request = """
        SELECT f.title, c.name, COUNT(r.rental_id)
        FROM film_category fc
        JOIN category c
        ON c.category_id = fc.category_id
        JOIN film f
        ON f.film_id = fc.film_id
        JOIN inventory i
        ON i.film_id = f.film_id
        JOIN rental r
        ON r.inventory_id = i.inventory_id
        WHERE c.name IN ('Animation', 'Children', 'Classics', 'Comedy', 'Family', 'Music')
        GROUP BY 1, 2
        ORDER BY 2, 1
        """
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl, (conn,))

    def impl2(conn):
        sql_request = """
        SELECT f.title, c.name, f.rental_duration, NTILE(4) OVER (ORDER BY f.rental_duration) AS standard_quartile
        FROM film_category fc
        JOIN category c
        ON c.category_id = fc.category_id
        JOIN film f
        ON f.film_id = fc.film_id
        WHERE c.name IN ('Animation', 'Children', 'Classics', 'Comedy', 'Family', 'Music')
        ORDER BY 3
        """
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl2, (conn,))

    def impl3(conn):
        sql_request = """
        SELECT film.film_id, title, inventory_id
        FROM film
        LEFT JOIN inventory
        ON inventory.film_id = film.film_id
        """
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl3, (conn,))


@pytest.mark.slow
@pytest.mark.skip(reason="disable all postgreSQL tests for now")
def test_postgres_read_sql_gb(memory_leak_check):
    """Test SQL query group by, column alias, and round PostgreSQL DB"""

    conn = "postgresql+psycopg2://" + postgres_user_pass_and_hostname + "/TEST_DB"

    def impl(conn):
        sql_request = """
        SELECT
	        customer_id,
            staff_id,
	        ROUND(AVG(amount), 2) AS Average
        FROM
	        payment
        GROUP BY
        	customer_id, staff_id
        """
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl, (conn,), check_dtype=False)

    def impl2(conn):
        sql_request = """
        SELECT
	        DATE(payment_date) paid_date,
	        SUM(amount) sum
        FROM
        	payment
        GROUP BY
        	DATE(payment_date)
        ORDER BY paid_date
        """
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl2, (conn,), check_dtype=False)


@pytest.mark.slow
@pytest.mark.skip(reason="disable all postgreSQL tests for now")
def test_postgres_read_sql_having(memory_leak_check):
    """Test SQL query HAVING PostgreSQL DB"""

    conn = "postgresql+psycopg2://" + postgres_user_pass_and_hostname + "/TEST_DB"

    def impl(conn):
        sql_request = """
        SELECT customer_id, SUM(amount)
        FROM payment
        GROUP BY customer_id
        HAVING SUM (amount) > 200
        ORDER BY customer_id
        """
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(impl, (conn,), check_dtype=False)


@pytest.mark.slow
@pytest.mark.skip(reason="disable all postgreSQL tests for now")
@pytest.mark.parametrize("is_distributed", [True, False])
def test_to_sql_postgres(is_distributed, memory_leak_check):
    """
    Test to_sql with PostgreSQL database
    Data is compared vs. original DF
    """

    def test_impl_write_sql(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace")

    # Gen Unique Table Name and Reset Table
    conn = "postgresql+psycopg2://" + postgres_user_pass_and_hostname + "/TEST_DB"
    with ensure_clean_mysql_psql_table(conn, "to_sql_table") as table_name:
        # Gen Random DF
        rng = np.random.default_rng(5)
        len_list = 20
        list_int = rng.choice(10, len_list)
        list_double = rng.choice([4.0, np.nan], len_list)
        list_datetime = pd.date_range("2001-01-01", periods=len_list)
        df_in = pd.DataFrame({"a": list_int, "b": list_double, "c": list_datetime})
        if is_distributed:
            start, end = get_start_end(len(df_in))
            df_input = df_in.iloc[start:end]
        else:
            df_input = df_in

        bodo.jit(all_args_distributed_block=is_distributed)(test_impl_write_sql)(
            df_input, table_name, conn
        )
        bodo.barrier()
        passed = 1
        if bodo.get_rank() == 0:
            try:
                # to_sql adds index column by default. Setting it here explicitly.
                df_load = pd.read_sql(
                    "select * from " + table_name, conn, index_col="index"
                )
                # The writing does not preserve the order a priori
                l_cols = df_in.columns.to_list()
                df_in_sort = df_in.sort_values(l_cols).reset_index(drop=True)
                df_load_sort = (
                    df_load[l_cols].sort_values(l_cols).reset_index(drop=True)
                )
                pd.testing.assert_frame_equal(
                    df_load_sort, df_in_sort, check_column_type=False
                )
            except Exception as e:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))
                passed = 0
        n_passed = reduce_sum(passed)
        n_pes = bodo.get_size()
        assert n_passed == n_pes, "test_to_sql_postgres failed"
        bodo.barrier()


# @pytest.mark.slow
@pytest.mark.parametrize("is_distributed", [True, False])
def test_to_sql_oracle(is_distributed, memory_leak_check):
    """Test to_sql with Oracle database
    Data is compared vs. original DF
    """

    def test_impl_write_sql(df, table_name, conn):
        df.to_sql(table_name, conn, index=False, if_exists="replace")

    rng = np.random.default_rng(5)
    # To ensure we're above rank 0 100 row limit
    len_list = 330
    list_int = rng.choice(10, len_list)
    list_double = rng.choice([4.0, np.nan], len_list)
    letters = np.array(list(string.ascii_letters))
    list_string = [
        "".join(rng.choice(letters, size=rng.integers(10, 100)))
        for _ in range(len_list)
    ]
    list_datetime = pd.date_range("2001-01-01", periods=len_list)
    list_date = [d.date() for d in pd.date_range("2021-11-06", periods=len_list)]
    df_in = pd.DataFrame(
        {
            "a": list_int,
            "b": list_double,
            "c": list_string,
            "d": list_date,
            "e": list_datetime,
        }
    )
    table_name = "to_sql_table"
    if is_distributed:
        start, end = get_start_end(len(df_in))
        df_input = df_in.iloc[start:end]
    else:
        df_input = df_in
    conn = "oracle+cx_oracle://" + oracle_user_pass_and_hostname + "/ORACLE"
    bodo.jit(all_args_distributed_block=is_distributed)(test_impl_write_sql)(
        df_input, table_name, conn
    )
    bodo.barrier()
    passed = 1
    if bodo.get_rank() == 0:
        try:
            df_load = pd.read_sql(
                "select * from " + table_name,
                conn,
            )
            # The writing does not preserve the order a priori
            l_cols = df_in.columns.to_list()
            df_in_sort = df_in.sort_values(l_cols).reset_index(drop=True)
            df_load_sort = df_load[l_cols].sort_values(l_cols).reset_index(drop=True)
            # check_dtype is False because Date column is read by Pandas as `object`
            pd.testing.assert_frame_equal(
                df_load_sort, df_in_sort, check_dtype=False, check_column_type=False
            )
        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0
    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_to_sql_oracle failed"
    bodo.barrier()
