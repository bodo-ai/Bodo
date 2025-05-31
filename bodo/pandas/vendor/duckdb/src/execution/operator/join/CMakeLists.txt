add_library_unity(
  duckdb_operator_join
  OBJECT
  physical_comparison_join.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_operator_join>
    PARENT_SCOPE)
