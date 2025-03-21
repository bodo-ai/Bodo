set(NEW_SOURCES
  boolean_operators.cpp
  vector_cast.cpp
  comparison_operators.cpp
  vector_copy.cpp
  generators.cpp
  vector_hash.cpp
  vector_storage.cpp
  null_operations.cpp
  numeric_inplace_operators.cpp
  is_distinct_from.cpp)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)