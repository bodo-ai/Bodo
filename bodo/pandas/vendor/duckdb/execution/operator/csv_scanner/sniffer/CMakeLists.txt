set(NEW_SOURCES
  csv_sniffer.cpp
  dialect_detection.cpp
  header_detection.cpp
  set_columns.cpp
  type_detection.cpp
  type_refinement.cpp
  type_replacement.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
