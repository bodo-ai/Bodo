# don't export any symbols from the third party stuff
set(CMAKE_C_VISIBILITY_PRESET hidden)

if(NOT AMALGAMATION_BUILD)
  add_subdirectory(fmt)
  add_subdirectory(re2)
  add_subdirectory(utf8proc)
  add_subdirectory(hyperloglog)
  add_subdirectory(fastpforlib)
  add_subdirectory(yyjson)
endif()
