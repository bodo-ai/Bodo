from queries.duckdb import utils

if utils.settings.run.io_type == "duckdb":
    from pathlib import Path

    import duckdb
    from queries.common_utils import get_table_path

    db_name = utils.get_persistent_path()
    if not Path(db_name).is_file():
        con = duckdb.connect(db_name)
        table_list = [
            "customer",
            "lineitem",
            "nation",
            "orders",
            "part",
            "partsupp",
            "region",
            "supplier",
        ]
        for table_name in table_list:
            parquet_path = get_table_path(table_name)
            con.sql(f"create table {table_name} as from read_parquet('{parquet_path}')")
        con.close()
        del con
