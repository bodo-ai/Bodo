add_subdirectory(column)
add_subdirectory(row)

set(NEW_SOURCES
  batched_data_collection.cpp
  bit.cpp
  blob.cpp
  cast_helpers.cpp
  conflict_manager.cpp
  conflict_info.cpp
  data_chunk.cpp
  date.cpp
  decimal.cpp
  hash.cpp
  hugeint.cpp
  uhugeint.cpp
  uuid.cpp
  hyperloglog.cpp
  interval.cpp
  list_segment.cpp
  selection_vector.cpp
  string_heap.cpp
  string_type.cpp
  timestamp.cpp
  time.cpp
  validity_mask.cpp
  value.cpp
  varint.cpp
  vector_buffer.cpp
  vector.cpp
  vector_cache.cpp
  vector_constants.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)