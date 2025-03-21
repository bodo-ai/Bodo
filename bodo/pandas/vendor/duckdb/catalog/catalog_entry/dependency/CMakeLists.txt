set(NEW_SOURCES
    dependency_entry.cpp
                  dependency_subject_entry.cpp dependency_dependent_entry.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
