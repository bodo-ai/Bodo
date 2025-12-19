add_library_unity(
  duckdb_transformer_helpers
  OBJECT
  # Bodo Change: Remove unneeded helpers
  #transform_alias.cpp
  transform_cte.cpp
  #transform_groupby.cpp
  #transform_orderby.cpp
  #transform_sample.cpp
  #transform_typename.cpp
  #nodetype_to_string.cpp)
)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_transformer_helpers>
    PARENT_SCOPE)
