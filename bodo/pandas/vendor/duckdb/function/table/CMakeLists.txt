set(NEW_SOURCES
  arrow.cpp
  arrow_conversion.cpp
  checkpoint.cpp
  glob.cpp
  query_function.cpp
  range.cpp
  repeat.cpp
  repeat_row.cpp
  copy_csv.cpp
  read_csv.cpp
  sniff_csv.cpp
  read_file.cpp
  system_functions.cpp
  summary.cpp
  table_scan.cpp
  unnest.cpp)

prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
  