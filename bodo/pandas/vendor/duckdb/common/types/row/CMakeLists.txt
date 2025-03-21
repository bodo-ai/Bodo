set(NEW_SOURCES
    partitioned_tuple_data.cpp
    row_data_collection.cpp
    row_data_collection_scanner.cpp
    row_layout.cpp
    tuple_data_allocator.cpp
    tuple_data_collection.cpp
    tuple_data_iterator.cpp
    tuple_data_layout.cpp
    tuple_data_scatter_gather.cpp
    tuple_data_segment.cpp
  )
  prepend_source_dir(NEW_SOURCES NEW_SOURCES)
  set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
  