set(NEW_SOURCES
    bool_data.cpp struct_data.cpp union_data.cpp fixed_size_list_data.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
