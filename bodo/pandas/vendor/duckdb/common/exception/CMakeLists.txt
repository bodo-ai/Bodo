set(NEW_SOURCES
  binder_exception.cpp catalog_exception.cpp
  conversion_exception.cpp parser_exception.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
