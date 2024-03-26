cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

# Debug Helpers
include(CMakePrintHelpers)

# ------------------------- Enable Compilation Caches -------------------------
# Enable CCache or SCCache if Found
# CCache is very useful when developing C++ code, because on
# rebuild it will only recompile the cpp files that have been modified.
# SCCache is similar, but supports remote S3 caches
# We use CCache for local development and SCCache for CI
# With newer compiler versions on Linux, we have to set both CC
# and CXX variables for (s)ccache to work, and have to set CC to
# the C++ compiler to avoid compile-time and dynamic linking errors
# Note that SCCache doesn't handle linking, so we skip for both

find_program(CCACHE_EXECUTABLE "ccache")
find_program(SCCACHE_EXECUTABLE "sccache")
if (CCACHE_EXECUTABLE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
elseif (SCCACHE_EXECUTABLE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE sccache)
endif()


# ----------------------------- Declare Dependencies -------------------------
# We need to do this before setting CMake Options so they aren't propagated
if (DEFINED ENV{BUILD_WITH_V8})
    # Needed for Prix64 to be defined in g++/Centos
    add_compile_definitions("__STDC_FORMAT_MACROS")
    include(FetchContent)

    FetchContent_Declare(
      v8
      GIT_REPOSITORY https://github.com/bnoordhuis/v8-cmake.git
      GIT_TAG "tags/11.6.189.4"
    )
    FetchContent_MakeAvailable(v8)
endif()


# ---------------------------- Set CMake Options -----------------------------
# Use RelWithDebInfo Mode by Default
cmake_print_variables(CMAKE_BUILD_TYPE)
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

# Ensure We are Inside a Conda Environment
if (NOT DEFINED ENV{CONDA_PREFIX})
  message(FATAL_ERROR "Please activate a Conda Environment before building Bodo")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Don't use GCC / Clang Extensions, only C++ Standard
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Get Pretty Color diagnostics in Ninja and C++ Compiler
# Same as enabling `-fcolor-diagnostics` by default
set(CMAKE_COLOR_DIAGNOSTICS ON)
# Export compile_commands.json for ClangD
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Enable -Werror by Default
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# Ignore CFLAGS and CXXFLAGS from Conda
set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS "")




# ----------------------- Set Compiler Args for Targets -----------------------
# Default Compiler Args
add_compile_options(
  "-Wno-c99-designator"         # Check if still necessary?
  "-Wno-return-type-c-linkage"  # Check if still necessary?
  "-Wno-macro-redefined"        # Check if still necessary?
  "-fwrapv"                     # Conda used to force-add. Remove in follow up PR
  "-fstack-protector-strong"    # Conda used to force-add. Remove in follow up PR
  "-D_FORTIFY_SOURCE=2"         # Conda used to force-add. Remove in follow up PR
)
if (LINUX)
  add_compile_options(
    # -march=haswell is used to enable AVX2 support (required by SIMD bloom filter implementation)
    "-march=haswell"
    # Avoid GCC errors for using int64 in allocations
    "-Wno-alloc-size-larger-than")
endif()

# Debug Compiler Args
if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DebugSanitize")
  add_compile_options(
    "-g"
    "-O1"
    "-fno-omit-frame-pointer"
  )
endif()

# Debug with Sanitizer Flags
if (CMAKE_BUILD_TYPE STREQUAL "DebugSanitize")
  add_compile_options(
    "-fsanitize=address"
    "-fsanitize=undefined"
  )
  add_link_options(
    "-fsanitize=address"
    "-fsanitize=undefined"
  )
endif()

# Release Compiler Args
if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  add_compile_options(
    "-g0"  # Not necessary, but for safety
    "-O3"
    "-ftree-vectorize"
    $<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>
  )
endif()

# ------------------------ Find Libraries + Compilers ------------------------
# Build Package + Program Dependencies
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_program(CYTHON_EXECUTABLE "cython" REQUIRED)

# Include for All Targets in General
set(CONDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
cmake_print_variables(CONDA_INCLUDE_DIR)
set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
cmake_print_variables(CONDA_LIB_DIR)

# This is supposed to work, but isnt. I don't know why
# include_directories(SYSTEM "${CONDA_INCLUDE_DIR}")
# Similarly, the following line doesn't work
# target_include_directories(ext SYSTEM PRIVATE ${CONDA_INCLUDE_DIR})
# In both cases, I'm certain that CMake believes that the directory is included
# But the Ninja file and GCC / Clang command doesn't include the directory
# TODO: Figure out what's going on here
# As an alternative, we can manually construct the generated arguments
# and append to the CMAKE_*_FLAGS variables directly
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isystem ${CONDA_INCLUDE_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CONDA_INCLUDE_DIR}")

# Find PyArrow Include Directory
# TODO: Use Arrow Include Directory from Conda Instead
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import pyarrow; print(pyarrow.get_include())"
  OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import pyarrow; print(pyarrow.get_library_dirs()[0])"
  OUTPUT_VARIABLE PYARROW_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
cmake_print_variables(PYARROW_INCLUDE_DIR)
cmake_print_variables(PYARROW_LIB_DIR)

# Numpy Include Directory is provided by scikit-build-core
# through `Python_NumPy_INCLUDE_DIR`
cmake_print_variables(Python_NumPy_INCLUDE_DIR)

# Combine All 2 Includes
set(BASE_INCLUDE_DIRS ${PYARROW_INCLUDE_DIR} ${Python_NumPy_INCLUDE_DIR})


# ------------------------ Cython Target - bodo.io.pyfs -----------------------
add_custom_command(
  OUTPUT bodo/io/pyfs.cpp
  DEPENDS bodo/io/pyfs.pyx
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "bodo/io/pyfs.cpp" "bodo/io/pyfs.pyx"
  COMMAND cp "bodo/io/pyfs.cpp" "${CMAKE_CURRENT_BINARY_DIR}/bodo/io/pyfs.cpp"
  COMMENT "Cythonizing Source bodo/io/pyfs.pyx into bodo/io/pyfs.cpp"
)

python_add_library(pyfs MODULE WITH_SOABI "bodo/io/pyfs.cpp")
target_include_directories(pyfs PRIVATE ${BASE_INCLUDE_DIRS})
target_link_directories(pyfs PRIVATE ${PYARROW_LIB_DIR})
target_link_libraries(pyfs PRIVATE arrow arrow_python)
install(TARGETS pyfs DESTINATION "bodo/io/")


# ----------------------- Cython Target - bodo.io._hdfs -----------------------
add_custom_command(
  OUTPUT bodo/io/_hdfs.cpp
  DEPENDS bodo/io/_hdfs.pyx
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "bodo/io/_hdfs.cpp" "bodo/io/_hdfs.pyx"
  COMMAND cp "bodo/io/_hdfs.cpp" "${CMAKE_CURRENT_BINARY_DIR}/bodo/io/_hdfs.cpp"
  COMMENT "Cythonizing Source bodo/io/_hdfs.pyx into bodo/io/_hdfs.cpp"
)

python_add_library(_hdfs MODULE WITH_SOABI "bodo/io/_hdfs.cpp")
target_include_directories(_hdfs PUBLIC ${BASE_INCLUDE_DIRS})
target_link_directories(_hdfs PRIVATE ${PYARROW_LIB_DIR})
target_link_libraries(_hdfs PRIVATE arrow arrow_python)
install(TARGETS _hdfs DESTINATION "bodo/io/")


# ---------------------- Cython Target - bodo.io.tracing ----------------------
if (NOT (CMAKE_BUILD_TYPE STREQUAL "Release"))
  set(BODO_DEV_BUILD "1")
else()
  set(BODO_DEV_BUILD "0")
endif()

add_custom_command(
  OUTPUT bodo/utils/tracing.c
  DEPENDS bodo/utils/tracing.pyx
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND "${CYTHON_EXECUTABLE}" -3 -E "BODO_DEV_BUILD=${BODO_DEV_BUILD}" --output-file "${CMAKE_CURRENT_BINARY_DIR}/bodo/utils/tracing.c" "bodo/utils/tracing.pyx"
  COMMENT "Cythonizing Source bodo/utils/tracing.pyx into bodo/utils/tracing.c"
)

python_add_library(tracing MODULE WITH_SOABI "bodo/utils/tracing.c")
target_link_libraries(tracing PRIVATE mpi)
install(TARGETS tracing DESTINATION "bodo/utils/")


# ---------------------- Cython Target - bodo.io.memory -----------------------
add_custom_command(
  OUTPUT bodo/libs/memory.cpp
  DEPENDS bodo/libs/memory.pyx
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "${CMAKE_CURRENT_BINARY_DIR}/bodo/libs/memory.cpp" "bodo/libs/memory.pyx"
  COMMENT "Cythonizing Source bodo/libs/memory.pyx into bodo/libs/memory.cpp"
)

python_add_library(
  memory
  MODULE WITH_SOABI
    "bodo/libs/memory.cpp"
    "bodo/libs/_memory.cpp"
    "bodo/libs/_operator_pool.cpp"
    "bodo/libs/_utils.cpp"
    "bodo/libs/_storage_manager.cpp"
    "bodo/libs/_memory_budget.cpp"
)

# Required when using boost::stacktrace for debugging
target_compile_definitions(memory PRIVATE "BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED=1")
target_include_directories(memory PRIVATE ${BASE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/bodo/libs/")
target_link_directories(memory PRIVATE ${PYARROW_LIB_DIR})
target_link_libraries(memory PRIVATE mpi arrow arrow_python fmt)
install(TARGETS memory DESTINATION "bodo/libs/")


# ------------------------- Main Bodo Extension -------------------------
set(sources_list
  "bodo/io/_csv_json_reader.cpp"
  "bodo/io/_csv_json_writer.cpp"
  "bodo/io/_fs_io.cpp"
  "bodo/io/_fsspec_reader.cpp"
  "bodo/io/_hdf5.cpp"
  "bodo/io/_hdfs_reader.cpp"
  "bodo/io/_io.cpp"
  "bodo/io/_s3_reader.cpp"
  "bodo/io/arrow.cpp"
  "bodo/io/arrow_reader.cpp"
  "bodo/io/iceberg_parquet_reader.cpp"
  "bodo/io/iceberg_parquet_write.cpp"
  "bodo/io/json_col_parser.cpp"
  "bodo/io/parquet_reader.cpp"
  "bodo/io/parquet_write.cpp"
  "bodo/io/snowflake_reader.cpp"
  "bodo/io/timestamptz_parser.cpp"
  "bodo/libs/_array.cpp"
  "bodo/libs/_array_hash.cpp"
  "bodo/libs/_array_operations.cpp"
  "bodo/libs/_array_utils.cpp"
  "bodo/libs/_base64.cpp"
  "bodo/libs/_bodo_common.cpp"
  "bodo/libs/_bodo_tdigest.cpp"
  "bodo/libs/_bodo_to_arrow.cpp"
  "bodo/libs/_datetime_ext.cpp"
  "bodo/libs/_datetime_utils.cpp"
  "bodo/libs/_decimal_ext.cpp"
  "bodo/libs/_distributed.cpp"
  "bodo/libs/_groupby.cpp"
  "bodo/libs/_groupby_agg_funcs.cpp"
  "bodo/libs/_groupby_col_set.cpp"
  "bodo/libs/_groupby_common.cpp"
  "bodo/libs/_groupby_do_apply_to_column.cpp"
  "bodo/libs/_groupby_eval.cpp"
  "bodo/libs/_groupby_ftypes.cpp"
  "bodo/libs/_groupby_groups.cpp"
  "bodo/libs/_groupby_mode.cpp"
  "bodo/libs/_groupby_mpi_exscan.cpp"
  "bodo/libs/_groupby_update.cpp"
  "bodo/libs/_hash_join.cpp"
  "bodo/libs/_nested_loop_join.cpp"
  "bodo/libs/_interval_join.cpp"
  "bodo/libs/_join_hashing.cpp"
  "bodo/libs/_lead_lag.cpp"
  "bodo/libs/_crypto_funcs.cpp"
  "bodo/libs/_memory.cpp"
  "bodo/libs/_memory_budget.cpp"
  "bodo/libs/_memory_budget_pymod.cpp"
  "bodo/libs/_murmurhash3.cpp"
  "bodo/libs/_quantile_alg.cpp"
  "bodo/libs/_lateral.cpp"
  "bodo/libs/_shuffle.cpp"
  "bodo/libs/_str_ext.cpp"
  "bodo/libs/iceberg_transforms.cpp"
  "bodo/libs/_stream_join.cpp"
  "bodo/libs/_stream_nested_loop_join.cpp"
  "bodo/libs/_stream_groupby.cpp"
  "bodo/libs/_dict_builder.cpp"
  "bodo/libs/_table_builder_utils.cpp"
  "bodo/libs/_table_builder.cpp"
  "bodo/libs/_chunked_table_builder.cpp"
  "bodo/libs/_listagg.cpp"
  "bodo/libs/_operator_pool.cpp"
  "bodo/libs/_window_aggfuncs.cpp"
  "bodo/libs/_window_compute.cpp"
  "bodo/libs/_stream_dict_encoding.cpp"
  "bodo/libs/_stream_shuffle.cpp"
  "bodo/libs/_storage_manager.cpp"
  "bodo/libs/_utils.cpp"
  "bodo/libs/_fft.cpp"
  "bodo/libs/_uuid.cpp"
  "bodo/libs/gason/gason.cpp"
)

if (DEFINED ENV{BUILD_WITH_V8})
  set(sources_list "${sources_list}"
    "bodo/libs/_javascript_udf.cpp"
      )
endif()

# TODO: Replace with ctest when integrating Catch2
if (NOT (CMAKE_BUILD_TYPE STREQUAL "Release"))
  set(sources_list "${sources_list}"
    "bodo/tests/test_framework.cpp"
    "bodo/tests/test_example.cpp"
    "bodo/tests/test_dict_builder.cpp"
    "bodo/tests/test_groupby_and_window.cpp"
    "bodo/tests/test_json_col_reader.cpp"
    "bodo/tests/test_memory_budget.cpp"
    "bodo/tests/test_pinnable.cpp"
    "bodo/tests/test_schema.cpp"
    "bodo/tests/test_table_builder.cpp"
    "bodo/tests/test_table_generator.cpp"
    "bodo/tests/test_test_framework.cpp"
    "bodo/tests/test_timestamptz_parser.cpp"
    "bodo/tests/test_nested_array.cpp"
    "bodo/tests/test_stream_shuffle.cpp"
    "bodo/tests/test_timestamptz_array.cpp"
  )
  if (DEFINED ENV{BUILD_WITH_V8})
      set(sources_list "${sources_list}"
      "bodo/tests/test_javascript.cpp"
      )
  endif()
endif()

python_add_library(ext MODULE WITH_SOABI "${sources_list}")

# Dependency Includes Are `-isystem` to Suppress Warnings
target_include_directories(ext SYSTEM PRIVATE ${BASE_INCLUDE_DIRS})
target_include_directories(
  ext
  PRIVATE
    "bodo/libs/HashLibs/TSL/hopscotch-map"
    "bodo/libs/HashLibs/TSL/robin-map"
    "bodo/libs/HashLibs/TSL/sparse-map"
)


if (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(ext PRIVATE "-Wno-unknown-pragmas")

else()
  target_compile_definitions(
    ext
    PRIVATE
      # Required when using boost::stacktrace for debugging
      "BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED=1"
      "IS_TESTING=1"
  )
endif()

target_compile_options(
	ext
	PRIVATE
		# -fno-strict-aliasing required by bloom filter implementation (see comment
		# in simd-block-fixed-fpp.h about violating strict aliasing rules)
		"-fno-strict-aliasing"
    "-Wall"
)

target_link_directories(ext PRIVATE ${PYARROW_LIB_DIR} "${Python_NumPy_INCLUDE_DIR}/../lib")
target_link_libraries(ext PRIVATE
  npymath
  mpi
  hdf5
  arrow
  arrow_python
  parquet
  fftw3
  fftw3f
  fftw3_mpi
  fftw3f_mpi
  m
  fmt)

set(IS_EXPIRED "$ENV{CHECK_LICENSE_EXPIRED}")
set(IS_CORE_COUNT "$ENV{CHECK_LICENSE_CORE_COUNT}")
set(IS_PLATFORM "$ENV{CHECK_LICENSE_PLATFORM}")
cmake_print_variables(IS_EXPIRED IS_CORE_COUNT IS_PLATFORM)

if (IS_EXPIRED OR IS_CORE_COUNT)
  target_link_libraries(ext PRIVATE ssl crypto)
  if (IS_EXPIRED)
    target_compile_definitions(ext PRIVATE "CHECK_LICENSE_EXPIRED=1")
  endif()
  if (IS_CORE_COUNT)
    target_compile_definitions(ext PRIVATE "CHECK_LICENSE_CORE_COUNT=1")
  endif()
endif()

if (IS_PLATFORM)
  if (IS_EXPIRED OR IS_CORE_COUNT)
    message( FATAL_ERROR "CHECK_LICENSE_EXPIRED or CHECK_LICENSE_CORE_COUNT cant be defined if CHECK_LICENSE_PLATFORM is defined" )
  endif()

  target_link_libraries(ext PRIVATE curl)
  target_compile_definitions(ext PRIVATE "CHECK_LICENSE_PLATFORM=1")
  target_include_directories(ext PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bodo/libs/gason")
endif()

if (DEFINED ENV{BUILD_WITH_V8})
  target_link_libraries(ext PRIVATE
    v8_libbase
    v8_libplatform
    v8_base_without_compiler
    v8_compiler
    v8_initializers
    v8-bytecodes-builtin-list
    v8_torque_generated
    v8_snapshot
    v8_libsampler
    dl)
  target_include_directories(ext PRIVATE "${v8_SOURCE_DIR}/v8")
  target_compile_definitions(ext PRIVATE "BUILD_WITH_V8=1")
endif()
add_dependencies(ext pyfs _hdfs tracing)


install(TARGETS ext DESTINATION "bodo/")


# ------------------------- Cythonize Transform Files -------------------------
set(transform_files)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(transform_files
    "bodo/transforms/dataframe_pass.py"
    "bodo/transforms/distributed_analysis.py"
    "bodo/transforms/distributed_pass.py"
    "bodo/transforms/series_pass.py"
    "bodo/transforms/table_column_del_pass.py"
    "bodo/transforms/typing_pass.py"
    "bodo/transforms/untyped_pass.py"
  )
endif()

foreach (file_path ${transform_files})
  cmake_path(GET file_path FILENAME file_name)
  cmake_path(REMOVE_EXTENSION file_name OUTPUT_VARIABLE cython_module)
  cmake_path(REPLACE_EXTENSION file_path ".c" OUTPUT_VARIABLE out_path)

  add_custom_command(
    OUTPUT ${out_path}
    DEPENDS ${file_path}
    VERBATIM
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND "${CYTHON_EXECUTABLE}" -3 --output-file "${CMAKE_CURRENT_BINARY_DIR}/${out_path}" "${file_path}"
    COMMENT "Cythonizing Source ${file_path} into ${out_path}"
  )

  python_add_library(${cython_module} MODULE WITH_SOABI ${out_path})
  install(TARGETS ${cython_module} DESTINATION "bodo/transforms/")
endforeach()
