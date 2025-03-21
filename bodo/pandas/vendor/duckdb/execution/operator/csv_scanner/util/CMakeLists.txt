set(NEW_SOURCES
    csv_error.cpp csv_reader_options.cpp csv_validator.cpp
  )
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
