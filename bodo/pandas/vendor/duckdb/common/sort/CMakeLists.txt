set(NEW_SOURCES
  comparators.cpp
  merge_sorter.cpp
  partition_state.cpp
  radix_sort.cpp
  sort_state.cpp
  sorted_block.cpp
  )
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
