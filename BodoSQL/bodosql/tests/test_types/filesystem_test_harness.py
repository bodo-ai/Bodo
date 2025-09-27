from pathlib import Path

from ddltest_harness import DDLTestHarness

import bodosql
from bodo.spawn.utils import run_rank0
from bodo.tests.conftest import iceberg_database  # noqa
from bodo.tests.iceberg_database_helpers.utils import (
    SparkFilesystemIcebergCatalog,
    get_spark,
)


class FilesystemTestHarness(DDLTestHarness):
    def __init__(self, iceberg_filesystem_catalog):
        self.catalog = iceberg_filesystem_catalog
        self.bc = bodosql.BodoSQLContext(catalog=iceberg_filesystem_catalog)
        self.spark_catalog = SparkFilesystemIcebergCatalog(
            catalog_name="ddl_test_filesystem",
            path=iceberg_filesystem_catalog.connection_string,
        )

    @property
    def spark(self):
        return get_spark(self.spark_catalog)

    def run_bodo_query(self, query):
        return self.bc.sql(query)

    @run_rank0
    def run_spark_query(self, query):
        # Spark doesn't allow double quotes as SQL identifiers.
        # when parsing SQL identifiers, BodoSQL is not case-sensitive, but Spark IS case-sensitive. This would not be a problem for the tabular filesystem as everything is in caps anyways, but for the filesystem catalog, the schema is actually `iceberg_db` in lowercase.

        # Thus, the following discrepencies happen.

        # - The `table_identifier` that bodosql recognizes is `"iceberg_db"."table_name"` (Note it must be in quotes, as we need to force BodoSQL to care about case.)
        # - Spark requires us to also pass in `ddl_test_filesystem` at the front. Bodo does not recognize it with `ddl_test_filesystem`.
        # - Spark does not let us use double quotes as sql identifiers, so we cannot just use double quotes everywhere.
        query = query.replace('"', "")
        if "iceberg_db." in query:
            query = query.replace("iceberg_db.", "ddl_test_filesystem.iceberg_db.")
        return self.spark.sql(query).toPandas()

    def get_table_identifier(self, table_name, db_schema=None):
        if db_schema is None:
            return f'"iceberg_db"."{table_name}"'
        else:
            return f'"{db_schema}"."{table_name}"'

    @run_rank0
    def create_test_table(self, table_identifier):
        # Spark doesn't allow double quotes as SQL identifiers
        table_identifier = table_identifier.replace('"', "")
        table_name = table_identifier.split(".")[1]
        db_schema = table_identifier.split(".")[0]
        self.spark.sql(
            f"CREATE TABLE ddl_test_filesystem.{db_schema}.{table_name} as select 'testtable' as A"
        )

    @run_rank0
    def drop_test_table(self, table_identifier):
        # Spark doesn't allow double quotes as SQL identifiers
        table_identifier = table_identifier.replace('"', "")
        table_identifier = "ddl_test_filesystem." + table_identifier
        self.spark.sql(f"DROP TABLE IF EXISTS {table_identifier}")

    @run_rank0
    def check_table_exists(self, table_identifier):
        # Spark doesn't allow double quotes as SQL identifiers
        table_identifier = table_identifier.replace('"', "")
        db_schema = table_identifier.split(".")[0]
        table_name = table_identifier.split(".")[1]
        return (
            len(
                self.spark.sql(
                    f"SHOW TABLES IN ddl_test_filesystem.{db_schema} like '{table_name}'"
                ).toPandas()
            )
            == 1
        )

    @run_rank0
    def show_table_properties(self, table_identifier):
        # Spark doesn't allow double quotes as SQL identifiers
        table_identifier = table_identifier.replace('"', "")
        table_identifier = "ddl_test_filesystem." + table_identifier
        self.spark.catalog.refreshTable(table_identifier)
        output = self.spark.sql(f"SHOW TBLPROPERTIES {table_identifier}").toPandas()
        return output

    @run_rank0
    def describe_table_extended(self, table_identifier):
        table_identifier = table_identifier.replace('"', "")
        table_identifier = "ddl_test_filesystem." + table_identifier
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
            # Spark doesn't allow double quotes as SQL identifiers
            table_identifier = table_identifier.replace('"', "")
            table_identifier = "ddl_test_filesystem." + table_identifier
            self.spark.catalog.refreshTable(table_identifier)
            output = self.spark.sql(f"DESCRIBE TABLE {table_identifier}").toPandas()
        return output

    @run_rank0
    def refresh_table(self, table_identifier):
        table_identifier = table_identifier.replace('"', "")
        table_identifier = "ddl_test_filesystem." + table_identifier
        self.spark.catalog.refreshTable(table_identifier)

    @run_rank0
    def create_test_view(self, view_identifier):
        raise NotImplementedError("FilesystemCatalog does not support views")

    @run_rank0
    def drop_test_view(self, view_identifier):
        raise NotImplementedError("FilesystemCatalog does not support views")

    @run_rank0
    def check_view_exists(self, view_identifier):
        raise NotImplementedError("FilesystemCatalog does not support views")

    def check_schema_exists(self, schema_name: str) -> bool:
        schema_path = Path(self.catalog.connection_string) / schema_name
        return schema_path.exists()
