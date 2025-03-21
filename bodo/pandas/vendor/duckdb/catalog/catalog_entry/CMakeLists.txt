add_subdirectory(dependency)

set(NEW_SOURCES
  copy_function_catalog_entry.cpp
  duck_index_entry.cpp
  duck_schema_entry.cpp
  duck_table_entry.cpp
  type_catalog_entry.cpp
  index_catalog_entry.cpp
  macro_catalog_entry.cpp
  pragma_function_catalog_entry.cpp
  schema_catalog_entry.cpp
  sequence_catalog_entry.cpp
  table_catalog_entry.cpp
  column_dependency_manager.cpp
  scalar_function_catalog_entry.cpp
  table_function_catalog_entry.cpp
  view_catalog_entry.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
