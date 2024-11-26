cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

# External Packages
include(FetchContent)
# Debug Helpers
include(CMakePrintHelpers)
# LTO Support
include(CheckIPOSupported)
check_ipo_supported()


# ------------------------- Enable Compilation Caches -------------------------
# Enable CCache or SCCache if Found
# CCache is very useful when developing C++ code, because on
# rebuild it will only recompile the cpp files that have been modified.
# SCCache is similar, but supports remote S3 caches
# We use CCache for local development and SCCache for CI
# Note that SCCache doesn't handle linking, so we skip for both

find_program(CCACHE_EXECUTABLE "ccache")
find_program(SCCACHE_EXECUTABLE "sccache")

if ((NOT DEFINED ENV{DISABLE_CCACHE}) AND CCACHE_EXECUTABLE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
elseif ((NOT DEFINED ENV{DISABLE_SCCACHE}) AND SCCACHE_EXECUTABLE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE sccache)
endif()


# ---------------------------- Enable C++ Linting ----------------------------
# Clang-Tidy is only available on MacOS as we use GCC on Linux
find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if (CLANG_TIDY_EXE)
  message(STATUS "Found Clang-Tidy: ${CLANG_TIDY_EXE}")
  set(CLANG_TIDY_COMMAND "${CLANG_TIDY_EXE}" "--checks=.clang-tidy" "--fix-errors")
else()
  message(STATUS "Clang-Tidy not found")
  set(CLANG_TIDY_COMMAND "")
endif()

# ----------------------------- Declare Dependencies -------------------------
# We need to do this before setting CMake Options so they aren't propagated
set(BUILD_WITH_V8 "$ENV{BUILD_WITH_V8}")
cmake_print_variables(BUILD_WITH_V8)
if (BUILD_WITH_V8)
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

# Download and Build 'fmt' from Git Directly
set(FMT_TEST OFF CACHE INTERNAL "disabling fmt tests")
FetchContent_Declare(
  fmt
  GIT_REPOSITORY  https://github.com/fmtlib/fmt.git
  GIT_TAG         10.2.1
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(fmt)
set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)


# ----------------------------- Download Apache Datasketches -------------------------
include(ExternalProject)
ExternalProject_Add(datasketches
  GIT_REPOSITORY https://github.com/apache/datasketches-cpp.git
  GIT_TAG 5.0.2
  GIT_SHALLOW true
  GIT_SUBMODULES ""
  INSTALL_DIR /tmp/datasketches-prefix
  CMAKE_ARGS -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=/tmp/datasketches-prefix
  # Override the install command to add DESTDIR
  # This is necessary to work around an oddity in the RPM (but not other) package
  # generation, as CMake otherwise picks up the Datasketch files when building
  # an RPM for a dependent package. (RPM scans the directory for files in addition to installing
  # those files referenced in an "install" rule in the cmake file)
  INSTALL_COMMAND env DESTDIR= ${CMAKE_COMMAND} --build . --target install
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
)

ExternalProject_Get_property(datasketches INSTALL_DIR)
set(datasketches_INSTALL_DIR ${INSTALL_DIR})
message("Source dir of datasketches = ${datasketches_INSTALL_DIR}")


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

# Enable Link-Time Optimizations (LTO)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()


# ----------------------- Set Compiler Args for Targets -----------------------
# Default Compiler Args
add_compile_options(
  "-Wno-c99-designator"         # Check if still necessary?
  "-Wno-return-type-c-linkage"  # Check if still necessary?
  "-Wno-macro-redefined"        # Check if still necessary?
  "-Wno-implicit-const-int-float-conversion" # Necessary for DataSketches
  "-fwrapv"                     # Conda used to force-add. Remove in follow up PR
  "-fstack-protector-strong"    # Conda used to force-add. Remove in follow up PR
  "-D_FORTIFY_SOURCE=2"         # Conda used to force-add. Remove in follow up PR
)
if (LINUX)
  add_compile_options(
    # -march=haswell is used to enable AVX2 support (required by SIMD bloom filter implementation)
    "-march=haswell"
    # Avoid GCC errors for using int64 in allocations
    "-Wno-alloc-size-larger-than"
    # Avoid GCC errors for unknown pragmas like "region"
    "-Wno-error=unknown-pragmas")
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
    "-O3"
    "-ftree-vectorize"
    # Encourages LTO & increase obfuscations
    $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
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

# Vendor MPI4Py
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bodo/mpi4py")
  message(STATUS "Copying mpi4py to source directory")
  file(COPY "$ENV{CONDA_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/mpi4py" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/bodo/")
endif()
message(STATUS "Copying mpi4py to wheel")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/bodo/mpi4py" DESTINATION "${SKBUILD_PLATLIB_DIR}/bodo/")

# Vendor MPICH
if(DEFINED ENV{BODO_VENDOR_MPICH})
  message(STATUS "Downloading mpich")
  execute_process(COMMAND ${Python_EXECUTABLE} -m pip download mpich==${MPICH_VERSION} -i  https://pypi.anaconda.org/mpi4py/simple)
  execute_process(COMMAND unzip mpich-*.whl "mpich*.data/*" -d mpich-tmp-extract-dir)
  file(GLOB MPICH_DATA_DIR "mpich-tmp-extract-dir/mpich-*.data/data/*")
  file(COPY ${MPICH_DATA_DIR} DESTINATION "${SKBUILD_DATA_DIR}" FOLLOW_SYMLINK_CHAIN FILE_PERMISSIONS OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE GROUP_WRITE)
  file(GLOB MPICH_WHEEL "mpich-*.whl")
  file(REMOVE_RECURSE ${MPICH_WHEEL} mpich-tmp-extract-dir)
  message(STATUS "Moved mpich to ${SKBUILD_DATA_DIR}")
endif()

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
target_link_directories(pyfs PRIVATE ${CONDA_LIB_DIR})
target_link_libraries(pyfs PRIVATE arrow arrow_python)
install(TARGETS pyfs DESTINATION "bodo/io/")

# ------------------------ Cython Target - bodo.io.pyarrow_wrappers ---------------
add_custom_command(
 OUTPUT bodo/io/pyarrow_wrappers.cpp
 DEPENDS bodo/io/pyarrow_wrappers.pyx
 VERBATIM
 WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
 COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "bodo/io/pyarrow_wrappers.cpp" "bodo/io/pyarrow_wrappers.pyx"
 COMMAND cp "bodo/io/pyarrow_wrappers.cpp" "${CMAKE_CURRENT_BINARY_DIR}/bodo/io/pyarrow_wrappers.cpp"
 COMMENT "Cythonizing Source bodo/io/pyarrow_wrappers.pyx into bodo/io/pyarrow_wrappers.cpp"
)
set(pyarrow_wrappers_sources "bodo/io/pyarrow_wrappers.cpp" "bodo/io/arrow_compat.cpp")
python_add_library(pyarrow_wrappers MODULE WITH_SOABI "${pyarrow_wrappers_sources}")
target_include_directories(pyarrow_wrappers PRIVATE ${BASE_INCLUDE_DIRS})
target_link_directories(pyarrow_wrappers PRIVATE ${PYARROW_LIB_DIR})
target_link_directories(pyarrow_wrappers PRIVATE ${CONDA_LIB_DIR})
target_link_libraries(pyarrow_wrappers PRIVATE arrow arrow_python arrow_dataset)
install(TARGETS pyarrow_wrappers DESTINATION "bodo/io/")


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
target_link_directories(_hdfs PRIVATE ${CONDA_LIB_DIR})
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
target_link_directories(tracing PRIVATE ${CONDA_LIB_DIR})
target_link_libraries(tracing PRIVATE mpi)
install(TARGETS tracing DESTINATION "bodo/utils/")


# ---------------------- Cython Target - bodo.memory -----------------------
add_custom_command(
  OUTPUT bodo/memory.cpp
  DEPENDS bodo/memory.pyx
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "${CMAKE_CURRENT_BINARY_DIR}/bodo/memory.cpp" "bodo/memory.pyx"
  COMMENT "Cythonizing Source bodo/memory.pyx into bodo/memory.cpp"
)

python_add_library(memory MODULE WITH_SOABI "bodo/memory.cpp")
target_include_directories(memory PRIVATE ${BASE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/bodo/")
target_link_directories(memory PRIVATE ${PYARROW_LIB_DIR})
target_link_directories(memory PRIVATE ${CONDA_LIB_DIR})
target_link_libraries(memory PRIVATE arrow arrow_python)
install(TARGETS memory DESTINATION "bodo/")


# ---------------------- Cython Target - bodo.tests.memory_tester -----------------------
add_custom_command(
  OUTPUT bodo/tests/memory_tester.cpp
  DEPENDS bodo/tests/memory_tester.pyx
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "${CMAKE_CURRENT_BINARY_DIR}/bodo/tests/memory_tester.cpp" "bodo/tests/memory_tester.pyx"
  COMMENT "Cythonizing Source bodo/tests/memory_tester.pyx into bodo/tests/memory_tester.cpp"
)

python_add_library(
  memory_tester
  MODULE WITH_SOABI
    "bodo/tests/memory_tester.cpp"
    "bodo/libs/_memory.cpp"
    "bodo/libs/_operator_pool.cpp"
    "bodo/libs/_utils.cpp"
    "bodo/libs/_storage_manager.cpp"
    "bodo/libs/_memory_budget.cpp"
)

target_include_directories(memory_tester PRIVATE ${BASE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/bodo/tests/")
target_link_directories(memory_tester PRIVATE ${PYARROW_LIB_DIR})
target_link_directories(memory_tester PRIVATE ${CONDA_LIB_DIR})
target_link_libraries(memory_tester PRIVATE mpi arrow arrow_python fmt::fmt)
if(NOT(CMAKE_BUILD_TYPE STREQUAL "Release"))
  target_compile_definitions(
    memory_tester
    PRIVATE
    # Required when using boost::stacktrace for debugging
    "BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED=1"
  )
endif()
install(TARGETS memory_tester DESTINATION "bodo/tests/")


# ---------------------- Cython Target - bodo.transforms.type_inference.native_typer -----------------------
add_custom_command(
  OUTPUT bodo/transforms/type_inference/native_typer.cpp
  DEPENDS bodo/transforms/type_inference/native_typer.pyx
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "${CMAKE_CURRENT_BINARY_DIR}/bodo/transforms/type_inference/native_typer.cpp" "bodo/transforms/type_inference/native_typer.pyx"
  COMMENT "Cythonizing Source bodo/transforms/type_inference/native_typer.pyx into bodo/transforms/type_inference/native_typer.cpp"
)

python_add_library(
  native_typer
  MODULE WITH_SOABI
    "bodo/transforms/type_inference/native_typer.cpp"
    "bodo/transforms/type_inference/type.cpp"
    "bodo/transforms/type_inference/typeinfer.cpp"
    "bodo/transforms/type_inference/ir.cpp"
)

target_include_directories(native_typer PRIVATE ${BASE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/bodo/transforms/type_inference/" "${CMAKE_CURRENT_BINARY_DIR}/bodo/transforms/type_inference/")
target_link_directories(native_typer PRIVATE ${PYARROW_LIB_DIR})
target_link_directories(native_typer PRIVATE ${CONDA_LIB_DIR})
target_link_libraries(native_typer PRIVATE fmt::fmt)
install(TARGETS native_typer DESTINATION "bodo/transforms/type_inference/")


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
  "bodo/io/arrow_compat.cpp"
  "bodo/io/iceberg_helpers.cpp"
  "bodo/libs/_array.cpp"
  "bodo/libs/_array_build_buffer.cpp"
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
  "bodo/libs/groupby/_groupby.cpp"
  "bodo/libs/groupby/_groupby_agg_funcs.cpp"
  "bodo/libs/groupby/_groupby_col_set.cpp"
  "bodo/libs/groupby/_groupby_common.cpp"
  "bodo/libs/groupby/_groupby_do_apply_to_column.cpp"
  "bodo/libs/groupby/_groupby_eval.cpp"
  "bodo/libs/groupby/_groupby_ftypes.cpp"
  "bodo/libs/groupby/_groupby_groups.cpp"
  "bodo/libs/groupby/_groupby_mode.cpp"
  "bodo/libs/groupby/_groupby_mpi_exscan.cpp"
  "bodo/libs/groupby/_groupby_update.cpp"
  "bodo/libs/_hash_join.cpp"
  "bodo/libs/_nested_loop_join_impl.cpp"
  "bodo/libs/_nested_loop_join.cpp"
  "bodo/libs/_interval_join.cpp"
  "bodo/libs/_join_hashing.cpp"
  "bodo/libs/_lead_lag.cpp"
  "bodo/libs/_crypto_funcs.cpp"
  "bodo/libs/_memory.cpp"
  "bodo/libs/_memory_budget.cpp"
  "bodo/libs/_memory_budget_pymod.cpp"
  "bodo/libs/vendored/_murmurhash3.cpp"
  "bodo/libs/_quantile_alg.cpp"
  "bodo/libs/_lateral.cpp"
  "bodo/libs/_shuffle.cpp"
  "bodo/libs/_str_ext.cpp"
  "bodo/libs/iceberg_transforms.cpp"
  "bodo/libs/streaming/_join.cpp"
  "bodo/libs/streaming/_nested_loop_join.cpp"
  "bodo/libs/streaming/_groupby.cpp"
  "bodo/libs/streaming/_sort.cpp"
  "bodo/libs/streaming/_window.cpp"
  "bodo/libs/_dict_builder.cpp"
  "bodo/libs/_table_builder_utils.cpp"
  "bodo/libs/_table_builder.cpp"
  "bodo/libs/_chunked_table_builder.cpp"
  "bodo/libs/_listagg.cpp"
  "bodo/libs/_operator_pool.cpp"
  "bodo/libs/window/_window_aggfuncs.cpp"
  "bodo/libs/window/_window_calculator.cpp"
  "bodo/libs/window/_window_compute.cpp"
  "bodo/libs/streaming/_dict_encoding.cpp"
  "bodo/libs/streaming/_shuffle.cpp"
  "bodo/libs/_storage_manager.cpp"
  "bodo/libs/_utils.cpp"
  "bodo/libs/_fft.cpp"
  "bodo/libs/_uuid.cpp"
  "bodo/libs/_query_profile_collector.cpp"
  "bodo/libs/vendored/gason/gason.cpp"
  "bodo/libs/_pymemory.cpp"
  "bodo/libs/_theta_sketches.cpp"
  "bodo/libs/_puffin.cpp"
  "bodo/libs/_io_cpu_thread_pool.cpp"
)

if (BUILD_WITH_V8)
  set(sources_list "${sources_list}"
    "bodo/libs/_javascript_udf.cpp"
      )
endif()

# Some files cannot be compiled with -Werror=implict-fallthrough, but we want
# the flag to be enabled for most files, so we create an exclude list of files
# that are incomptible
set(allow_implicit_fallthrough_list
  "bodo/io/arrow.cpp"
  "bodo/io/iceberg_parquet_write.cpp"
  "bodo/libs/_puffin.cpp"
  "bodo/libs/_theta_sketches.cpp"
  "bodo/libs/vendored/_murmurhash3.cpp"
  "bodo/libs/vendored/gason/gason.cpp"
)

# TODO: Replace with ctest when integrating Catch2
if (NOT (CMAKE_BUILD_TYPE STREQUAL "Release") AND NOT DEFINED ENV{BODO_SKIP_CPP_TESTS})
  set(test_list
    "bodo/tests/test_framework.cpp"
    "bodo/tests/test_example.cpp"
    "bodo/tests/test_external_sort.cpp"
    "bodo/tests/test_dict_builder.cpp"
    "bodo/tests/test_groupby_common.cpp"
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
    "bodo/tests/test_streaming/test_shuffle.cpp"
    "bodo/tests/test_streaming/test_groupby_shuffle.cpp"
    "bodo/tests/test_timestamptz_array.cpp"
    "bodo/tests/test_query_profile_collector.cpp"
    "bodo/tests/test_theta_sketches.cpp"
    "bodo/tests/test_puffin.cpp"
    "bodo/tests/test_iceberg_rest_aws_credentials_provider.cpp"
    "bodo/tests/test_sorted_window_computation.cpp"
    "bodo/tests/test_thread_pool.cpp"
    "bodo/tests/test_datatypes.cpp"
  )
  if (BUILD_WITH_V8)
      set(test_list "${test_list}"
      "bodo/tests/test_javascript.cpp"
      )
  endif()
  list(APPEND sources_list "${test_list}")
  list(APPEND allow_implicit_fallthrough_list 
    "bodo/tests/test_puffin.cpp"
    "bodo/tests/test_theta_sketches.cpp")
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
target_link_directories(ext PRIVATE ${CONDA_LIB_DIR})


if (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(ext PRIVATE "-Wno-unknown-pragmas")
elseif (DEFINED ENV{BODO_SKIP_CPP_TESTS})
  # Remove the IS_TESTING flag.
  target_compile_definitions(
    ext
    PRIVATE
      # Required when using boost::stacktrace for debugging
      "BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED=1"
  )
else()
  target_compile_definitions(
    ext
    PRIVATE
      # Required when using boost::stacktrace for debugging
      "BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED=1"
      "IS_TESTING=1"
  )
endif()

# Enable erroring on implicit fallthrough in switch/case statements this is
# critical because we use switch/case statements to handle variants of code for
# different array types in a lot of places.
list(APPEND error_on_fallthrough ${sources_list})
# TODO(aneesh) [BSE-3514] avoid having files exempt from implicit-fallthrough
list(REMOVE_ITEM error_on_fallthrough ${allow_implicit_fallthrough_list})
SET_SOURCE_FILES_PROPERTIES(
  ${error_on_fallthrough}
  PROPERTIES
  COMPILE_FLAGS "-Werror=implicit-fallthrough"
)

# This file includes pyarrow_wrappers_api.h which is generated by Cython
# and the compiler can't tell that the functions are used
SET_SOURCE_FILES_PROPERTIES(
  "bodo/io/arrow_compat.cpp"
  PROPERTIES
  COMPILE_FLAGS "-Wno-unused-function"
)

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
  arrow_dataset
  parquet
  fftw3
  fftw3f
  fftw3_mpi
  fftw3f_mpi
  m
  fmt::fmt
  aws-cpp-sdk-core
)

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
  target_include_directories(ext PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bodo/libs/vendored/gason")
endif()

# Build with our fork of Arrow
if (IS_PLATFORM OR DEFINED ENV{USE_BODO_ARROW_FORK})
  target_compile_definitions(ext PRIVATE "USE_BODO_ARROW_FORK=1")
endif()


if (BUILD_WITH_V8)
  message(STATUS "Building with V8")
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
add_dependencies(ext pyfs _hdfs tracing pyarrow_wrappers)


target_include_directories(ext PRIVATE ${datasketches_INSTALL_DIR}/include/DataSketches)
add_dependencies(ext datasketches)

set_target_properties(ext PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")

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
