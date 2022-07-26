from bodo.tests.iceberg_database_helpers import (
    file_subset_deleted_rows_table,
    file_subset_empty_files_table,
    file_subset_partial_file_table,
    filter_pushdown_test_table,
    large_delete_table,
    partitions_dropped_dt_table,
    partitions_dt_table,
    partitions_general_table,
    schema_evolution_eg_table,
    simple_bool_binary_table,
    simple_decimals_list_table,
    simple_decimals_table,
    simple_dt_tsz_table,
    simple_dtype_list_table,
    simple_list_table,
    simple_map_table,
    simple_numeric_table,
    simple_string_table,
    simple_struct_table,
    simple_tz_aware_table,
)
from bodo.tests.iceberg_database_helpers.utils import DATABASE_NAME, get_spark


def create_all_tables(spark=None):
    if spark is None:
        spark = get_spark()
    for table_mod in [
        file_subset_deleted_rows_table,
        file_subset_empty_files_table,
        file_subset_partial_file_table,
        filter_pushdown_test_table,
        large_delete_table,
        partitions_dropped_dt_table,
        partitions_dt_table,
        partitions_general_table,
        schema_evolution_eg_table,
        simple_bool_binary_table,
        simple_dt_tsz_table,
        simple_dtype_list_table,
        simple_list_table,
        simple_map_table,
        simple_numeric_table,
        simple_string_table,
        simple_struct_table,
        simple_decimals_table,
        simple_decimals_list_table,
        simple_tz_aware_table,
    ]:
        table_mod.create_table(spark=spark)

    return DATABASE_NAME


if __name__ == "__main__":
    create_all_tables()
