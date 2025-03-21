set(NEW_SOURCES
    arrow_duck_schema.cpp arrow_array_scan_state.cpp arrow_type_info.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
