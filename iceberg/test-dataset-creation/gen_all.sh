set -eo pipefail

echo "Creating simple_numeric_table"
python simple_numeric_table.py
echo "Creating simple_string_table"
python simple_string_table.py
echo "Creating simple_bool_binary_table"
python simple_bool_binary_table.py
echo "Creating simple_struct_table"
python simple_struct_table.py
echo "Creating simple_list_table"
python simple_list_table.py
echo "Creating simple_map_table"
python simple_map_table.py
echo "Creating simple_dt_tsz_table"
python simple_dt_tsz_table.py
echo "Creating schema_evolved_table"
python schema_evolved_table.py
echo "Creating file_subset_deleted_rows_table"
python file_subset_deleted_rows_table.py
echo "Creating file_subset_empty_files_table"
python file_subset_empty_files_table.py
echo "Creating file_subset_partial_file"
python file_subset_partial_file.py
echo "Creating partitions_general_table"
python partitions_general_table.py
echo "Creating partitions_dt_table"
python partitions_dt_table.py
echo "Creating partitions_dropped_dt_table"
python partitions_dropped_dt_table.py
echo "Creating filter_pushdown_test_table"
python filter_pushdown_test_table.py
echo "All done!"
