set(NEW_SOURCES
    cast_operators.cpp convert_to_string.cpp string_cast.cpp)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
