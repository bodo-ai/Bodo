add_library_unity(
  duckdb_operator_aggregate
  OBJECT
  aggregate_object.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_operator_aggregate>
    PARENT_SCOPE)
