set(NEW_SOURCES
  row_aggregate.cpp
  row_scatter.cpp
  row_gather.cpp
  row_matcher.cpp
  row_external.cpp
  row_radix_scatter.cpp
  row_heap_scatter.cpp
  row_heap_gather.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)