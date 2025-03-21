set(NEW_SOURCES
  binary_serializer.cpp
  binary_deserializer.cpp
  buffered_file_reader.cpp
  buffered_file_writer.cpp
  memory_stream.cpp
  serializer.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)