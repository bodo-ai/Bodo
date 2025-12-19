from ddltest_harness import DDLTestHarness

import bodosql
from bodo.spawn.utils import run_rank0
from bodo.tests.iceberg_database_helpers.utils import SparkAwsIcebergCatalog, get_spark


class RestTestHarness(DDLTestHarness):
    def __init__(self, polaris_catalog, polaris_connection):
        uri, warehouse, credential = polaris_connection
        self.spark_catalog = SparkAwsIcebergCatalog(
            catalog_name=warehouse,
            warehouse=warehouse,
            uri=uri,
            credential=credential,
            default_schema="default",
        )
        self.bc = bodosql.BodoSQLContext(catalog=polaris_catalog)

    @property
    def spark(self):
        return get_spark(self.spark_catalog)

    def run_bodo_query(self, query):
        return self.bc.sql(query)

    @run_rank0
    def run_spark_query(self, query):
        return self.spark.sql(query).toPandas()

    def get_table_identifier(self, table_name, db_schema=None):
        if db_schema is None:
            return table_name
        else:
            return f"{db_schema}.{table_name}"

    @run_rank0
    def create_test_table(self, table_identifier):
        self.spark.sql(
            f"CREATE OR REPLACE TABLE {table_identifier} AS SELECT 'testtable' as A"
        )

    @run_rank0
    def drop_test_table(self, table_identifier):
        self.spark.sql(f"DROP TABLE IF EXISTS {table_identifier}")

    @run_rank0
    def check_table_exists(self, table_identifier):
        if "." in table_identifier:
            table_name = table_identifier.split(".")[1]
            db_schema = table_identifier.split(".")[0]
        else:
            table_name = table_identifier
            db_schema = "default"
        return (
            len(
                self.spark.sql(
                    f"SHOW TABLES IN {db_schema} LIKE '{table_name}'"
                ).toPandas()
            )
            == 1
        )

    @run_rank0
    def show_table_properties(self, table_identifier):
        self.spark.catalog.refreshTable(table_identifier)
        output = self.spark.sql(f"SHOW TBLPROPERTIES {table_identifier}").toPandas()
        return output

    @run_rank0
    def describe_table_extended(self, table_identifier):
        self.spark.catalog.refreshTable(table_identifier)
        output = self.spark.sql(
            f"DESCRIBE TABLE EXTENDED {table_identifier}"
        ).toPandas()
        return output

    @run_rank0
    def describe_table(self, table_identifier, spark=False):
        if not spark:
            # Spark does not support time fields, so we use bodo
            output = self.bc.sql(f"DESCRIBE TABLE {table_identifier}")
        else:
            self.spark.catalog.refreshTable(table_identifier)
            output = self.spark.sql(f"DESCRIBE TABLE {table_identifier}").toPandas()
        return output

    @run_rank0
    def refresh_table(self, table_identifier):
        self.spark.catalog.refreshTable(table_identifier)
        self.spark.sql(f"REFRESH TABLE {table_identifier}")

    @run_rank0
    def create_test_view(self, view_identifier):
        self.spark.sql(f"CREATE OR REPLACE VIEW {view_identifier} AS SELECT 0 as A")

    @run_rank0
    def drop_test_view(self, view_identifier):
        self.spark.sql(f"DROP VIEW IF EXISTS {view_identifier}")

    @run_rank0
    def check_view_exists(self, view_identifier):
        if "." in view_identifier:
            view_name = view_identifier.split(".")[1]
            db_schema = view_identifier.split(".")[0]
        else:
            view_name = view_identifier
            db_schema = "default"
        return (
            len(
                self.spark.sql(
                    f"SHOW VIEWS IN {db_schema} LIKE '{view_name}'"
                ).toPandas()
            )
            == 1
        )

    @run_rank0
    def check_schema_exists(self, schema_name: str) -> bool:
        tables = self.spark.sql(f"SHOW SCHEMAS LIKE '{schema_name}'").toPandas()
        return len(tables) == 1
