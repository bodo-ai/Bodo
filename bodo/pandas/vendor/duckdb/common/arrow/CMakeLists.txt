add_subdirectory(appender)


set(NEW_SOURCES
  arrow_appender.cpp
  arrow_converter.cpp
  arrow_type_extension.cpp
  arrow_merge_event.cpp
  arrow_query_result.cpp
  arrow_util.cpp
  arrow_wrapper.cpp
  physical_arrow_collector.cpp
  physical_arrow_batch_collector.cpp
  schema_metadata.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
