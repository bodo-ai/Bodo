set(NEW_SOURCES
    default_functions.cpp
    default_generator.cpp
    default_schemas.cpp
    default_table_functions.cpp
    default_types.cpp
    default_views.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
