set(NEW_SOURCES
  json_tree_renderer.cpp
  html_tree_renderer.cpp
  graphviz_tree_renderer.cpp
  text_tree_renderer.cpp
  tree_renderer.cpp
)
prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
