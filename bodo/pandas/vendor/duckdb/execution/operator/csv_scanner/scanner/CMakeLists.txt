set(NEW_SOURCES
  base_scanner.cpp
  column_count_scanner.cpp
  csv_schema.cpp
  scanner_boundary.cpp
  skip_scanner.cpp
  string_value_scanner.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
