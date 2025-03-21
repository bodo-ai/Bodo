set(NEW_SOURCES
  csv_file_scanner.cpp
  global_csv_state.cpp csv_multi_file_info.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
