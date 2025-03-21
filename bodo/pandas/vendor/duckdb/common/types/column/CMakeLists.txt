set(NEW_SOURCES
  column_data_allocator.cpp
  column_data_collection.cpp
  column_data_collection_segment.cpp
  column_data_consumer.cpp
  partitioned_column_data.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
