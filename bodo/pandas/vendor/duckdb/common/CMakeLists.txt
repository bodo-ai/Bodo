# Bodo change: remove adbc since it is not used
# if(NOT CLANG_TIDY)
#   add_subdirectory(adbc)
# endif()

add_subdirectory(arrow)
# bodo change: remove crypto since it is not used
# add_subdirectory(crypto)
add_subdirectory(enums)
add_subdirectory(exception)
add_subdirectory(operator)
add_subdirectory(progress_bar)
add_subdirectory(tree_renderer)
add_subdirectory(row_operations)
add_subdirectory(serializer)
add_subdirectory(sort)
add_subdirectory(types)
add_subdirectory(value_operations)
add_subdirectory(vector_operations)


set(NEW_SOURCES
  allocator.cpp
  assert.cpp
  bind_helpers.cpp
  box_renderer.cpp
  cgroups.cpp
  complex_json.cpp
  compressed_file_system.cpp
  constants.cpp
  checksum.cpp
  encryption_state.cpp
  exception.cpp
  exception_format_value.cpp
  extra_type_info.cpp
  file_buffer.cpp
  file_system.cpp
  filename_pattern.cpp
  fsst.cpp
  gzip_file_system.cpp
  hive_partitioning.cpp
  http_util.cpp
  pipe_file_system.cpp
  local_file_system.cpp
  multi_file_list.cpp
  multi_file_reader.cpp
  error_data.cpp
  opener_file_system.cpp
  printer.cpp
  radix_partitioning.cpp
  # bodo change: remove crypto since it is not used and causes an extra dependency
  # re2_regex.cpp
  random_engine.cpp
  stacktrace.cpp
  string_util.cpp
  enum_util.cpp
  symbols.cpp
  render_tree.cpp
  tree_renderer.cpp
  types.cpp
  virtual_file_system.cpp
  windows_util.cpp)

prepend_source_dir(NEW_SOURCES NEW_SOURCES)
set(PLANNER_SOURCES ${PLANNER_SOURCES} ${NEW_SOURCES} PARENT_SCOPE)
