set(NEW_SOURCES
  catalog_type.cpp
  compression_type.cpp
  date_part_specifier.cpp
  expression_type.cpp
  file_compression_type.cpp
  join_type.cpp
  logical_operator_type.cpp
  metric_type.cpp
  optimizer_type.cpp
  physical_operator_type.cpp
  statement_type.cpp
  relation_type.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
