from bodo.tests.iceberg_database_helpers import (
    filter_pushdown_test_table,
    part_sort_table,
    partitions_dt_table,
)
from bodo.tests.iceberg_database_helpers.partition_tables import (
    create_all_partition_tables,
)
from bodo.tests.iceberg_database_helpers.simple_tables import (
    create_all_simple_tables,
)
from bodo.tests.iceberg_database_helpers.sort_tables import (
    create_all_sort_tables,
)
from bodo.tests.iceberg_database_helpers.utils import DATABASE_NAME, get_spark


def create_all_tables(spark=None):
    if spark is None:
        spark = get_spark()

    create_all_simple_tables(spark)
    create_all_partition_tables(spark)
    create_all_sort_tables(spark)

    for table_mod in [
        filter_pushdown_test_table,
        partitions_dt_table,
        part_sort_table,
        # These are not used in any of the tests at this time.
        # These should be added back when they are.
        #     file_subset_deleted_rows_table,
        #     file_subset_empty_files_table,
        #     file_subset_partial_file_table,
        #     large_delete_table,
        #     partitions_dropped_dt_table,
        #     partitions_general_table,
        #     schema_evolution_eg_table,
    ]:
        table_mod.create_table(spark=spark)

    return DATABASE_NAME


if __name__ == "__main__":
    create_all_tables()
