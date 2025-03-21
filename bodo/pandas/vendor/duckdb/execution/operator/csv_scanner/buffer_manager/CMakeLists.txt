set(NEW_SOURCES
    csv_buffer.cpp
    csv_buffer_manager.cpp csv_file_handle.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
