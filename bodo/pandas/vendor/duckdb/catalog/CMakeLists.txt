add_subdirectory(catalog_entry)
add_subdirectory(default)

set(NEW_SOURCES
  catalog_entry.cpp
  catalog_entry_retriever.cpp
  catalog.cpp
  catalog_search_path.cpp
  catalog_set.cpp
  catalog_transaction.cpp
  duck_catalog.cpp
  dependency_manager.cpp
  dependency_list.cpp
  dependency_catalog_set.cpp
  similar_catalog_entry.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
