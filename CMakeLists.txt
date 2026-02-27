cmake_minimum_required(VERSION 3.29...3.31 FATAL_ERROR)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

# External Packages
include(FetchContent)
# Debug Helpers
include(CMakePrintHelpers)
# LTO Support
if (NOT WIN32)
  include(CheckIPOSupported)
  check_ipo_supported(LANGUAGES CXX)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  set (X86 TRUE)
else()
  set (X86 FALSE)
endif()

if (LINUX)
    # Use mold for linking on Linux systems
    set(CMAKE_LINKER_TYPE MOLD)
endif()

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
if (LINUX)
  message(STATUS "Clang-Tidy is not used on Linux")
  set(CLANG_TIDY_COMMAND "")
elseif (CLANG_TIDY_EXE)
  message(STATUS "Found Clang-Tidy: ${CLANG_TIDY_EXE}")
  set(CLANG_TIDY_COMMAND "${CLANG_TIDY_EXE}" "--checks=.clang-tidy" "--fix-errors")
else()
  message(STATUS "Clang-Tidy was not found")
  set(CLANG_TIDY_COMMAND "")
endif()

# ----------------------------- Declare Dependencies -------------------------
# We need to do this before setting CMake Options so they aren't propagated
set(BUILD_WITH_V8 "$ENV{BUILD_WITH_V8}")
cmake_print_variables(BUILD_WITH_V8)
if (BUILD_WITH_V8)
  # Needed for Prix64 to be defined in g++/Centos
  add_compile_definitions("__STDC_FORMAT_MACROS")
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

#Prevent symbol collisions
add_compile_definitions(duckdb=bododuckdb)
FetchContent_Declare(
    duckdb
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/bodo/pandas/vendor/duckdb
    EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(duckdb)

# If we're building dependencies (not getting them from conda), we need to build AWS SDK
if("$ENV{CONDA_PREFIX}" STREQUAL "")
  set(BUILD_ONLY "core" CACHE INTERNAL "only build core")
  set(ENABLE_TESTING OFF CACHE INTERNAL "disabling fmt tests")
  FetchContent_Declare(
    AWSSDK
    GIT_REPOSITORY   https://github.com/aws/aws-sdk-cpp
    GIT_TAG          1.11.472
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(AWSSDK)
  find_package(AWSSDK REQUIRED COMPONENTS core)
  install(TARGETS aws-cpp-sdk-core DESTINATION "${SKBUILD_PLATLIB_DIR}")
  unset(BUILD_ONLY)
  unset(ENABLE_TESTING)
elseif(WIN32)
  # On Windows we need to link with the import library (.lib) in order to resolve references into dynamically loaded libraries.
  # find_package locates the import library so we can make it availible to our extension.
  # https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-creation
  find_package(AWSSDK REQUIRED COMPONENTS core)
endif()

# Find additional import libraries needed to compile on Windows
if(WIN32)
  find_package(OpenSSL REQUIRED)
endif()

# ----------------------------- Download Apache Datasketches -------------------------
set(BUILD_TESTS OFF CACHE BOOL "Disable datasketches tests" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable datasketches examples" FORCE) 

FetchContent_Declare(
  datasketches
  GIT_REPOSITORY https://github.com/apache/datasketches-cpp.git
  GIT_TAG        5.0.2
  GIT_SHALLOW    TRUE
  GIT_SUBMODULES ""
  EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(datasketches)

# ---------------------------- Set CMake Options -----------------------------

# CMAKE_BUILD_TYPE is not set when using Visual Studio generator because
# technically there could be multiple build configurations in the same build
# environment.
# TODO [BSE-4599] Switch to Ninja on Windows for consistency.
if (DEFINED ENV{BODO_WINDOWS_BUILD_TYPE})
  message(STATUS "Setting build type to $ENV{BODO_WINDOWS_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "$ENV{BODO_WINDOWS_BUILD_TYPE}")
endif()

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
# Don't bother checking for C++ Modules
set(CMAKE_CXX_SCAN_FOR_MODULES 0)

# Get Pretty Color diagnostics in Ninja and C++ Compiler
# Same as enabling `-fcolor-diagnostics` by default
set(CMAKE_COLOR_DIAGNOSTICS ON)
# Export compile_commands.json for ClangD
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Enable -Werror by Default
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# Enable Link-Time Optimizations (LTO)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

option(USE_CUDF "Enable CUDF mode" OFF)
if(USE_CUDF)
    add_compile_definitions(USE_CUDF=1)
endif()

# ----------------------- Apply Compiler Flags -----------------------
message(STATUS "\n*** Apply Compiler Flags for Targets")

# Default Compiler Args
if (LINUX AND X86)
  add_compile_options(
    # -march=haswell is used to enable AVX2 support (required by SIMD bloom filter implementation)
    "-march=haswell"
    # Avoid GCC errors for using int64 in allocations
    "-Wno-alloc-size-larger-than"
    # Avoid GCC errors for unknown pragmas like "region"
    "-Wno-error=unknown-pragmas"
  )
endif()

if (WIN32)
  # TODO [BSE-4559]: enable warnings.
  add_compile_options(
    "/wd4819" # Suppress warnings about file character encoding
    "/wd4996" # Suppress warnings about deprecated functions
    "/wd4312" # Suppress warnings about conversion
    "/wd4365" # Supress warnings about conversions (signed/unsigned)
    "/wd4244" # Supress warning about conversions (loss of data)
    "/wd4061" # Supress warning about switch not being explicitly handled
    "/wd4100" # Supress warning about unreferenced formal parameter.
    "/wd4551" # Suppress warnings about function call missing argument list
    "/w"      # disable other warnings
    "/GS"       # Enable buffer security checks
    "/guard:cf" # Enable control flow guard (security feature)
    "/EHsc"
  )
  add_definitions(-DMS_WINDOWS)  # Globally define MS_WINDOWS
else()
  add_compile_options(
    "-Wno-c99-designator"         # Check if still necessary?
    "-Wno-return-type-c-linkage"  # Check if still necessary?
    "-Wno-macro-redefined"        # Check if still necessary?
    "-Wno-implicit-const-int-float-conversion" # Necessary for DataSketches
    "-fwrapv"                     # Conda used to force-add. Its an optimization for integer overflow
  )
endif()

# Debug Compiler Args
if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DebugSanitize")
  if (NOT WIN32)
    add_compile_options(
      "-g"
      "-O0"
      "-fno-omit-frame-pointer"
    )
  else()
    add_compile_options(
      "/FS"  # Force writes to PDBs to be serialized
      "/Zi"  # Generate debug information
    )
    add_link_options("/DEBUG") # Generate debug symbols
  endif()
endif()

# Debug with Sanitizer Flags
# TODO [BSE-4552]: Add DebugSanitize as a build type for Windows builds.
if (CMAKE_BUILD_TYPE STREQUAL "DebugSanitize")
  if (NOT WIN32)
    add_compile_options(
      "-fsanitize=address"
      "-fsanitize=undefined"
    )
    add_link_options(
      "-fsanitize=address"
      "-fsanitize=undefined"
    )
  else()
    # Note for DebugSanitize builds we have to compile with /bigobj to allow larger object files
    # https://learn.microsoft.com/en-us/cpp/build/reference/bigobj-increase-number-of-sections-in-dot-obj-file?view=msvc-170
    add_compile_options(
      "/bigobj"
      "/fsanitize=address"
    )
    # These definitions are necessary for now because of conflicts between our extensions and other
    # build dependencies e.g. fmt. In the future we can look into enabling them:
    # https://learn.microsoft.com/en-us/cpp/sanitizers/error-container-overflow?view=msvc-170
    # TODO [BSE-4552] Remove once we have fully working DebugSanitize config
    add_definitions(
      -D_DISABLE_STRING_ANNOTATION
      -D_DISABLE_VECTOR_ANNOTATION
    )
  endif()
endif()

# Release Compiler Args
# TODO [BSE-4561] Configure release options for windows.
if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  if (NOT WIN32)
    add_compile_options(
      "-O3"
      "-ftree-vectorize"
      # Encourages LTO & increase obfuscations
      $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
    )
  endif()
endif()


# ------------------------ Find Libraries + Compilers ------------------------
# Build Package + Program Dependencies
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_program(CYTHON_EXECUTABLE "cython" REQUIRED)

# Include for All Targets in General
if (WIN32)
  set(CONDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}\\Library\\include")
  set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}\\Library\\lib")
else()
  set(CONDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
  set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
endif()

cmake_print_variables(CONDA_INCLUDE_DIR)
cmake_print_variables(CONDA_LIB_DIR)

# This is supposed to work, but isn't. I don't know why
# include_directories(SYSTEM "${CONDA_INCLUDE_DIR}")
# Similarly, the following line doesn't work
# target_include_directories(ext SYSTEM PRIVATE ${CONDA_INCLUDE_DIR})
# In both cases, I'm certain that CMake believes that the directory is included
# But the Ninja file and GCC / Clang command doesn't include the directory
# TODO: Figure out what's going on here
# As an alternative, we can manually construct the generated arguments
# and append to the CMAKE_*_FLAGS variables directly
if (WIN32)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /external:I ${CONDA_INCLUDE_DIR}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /external:I ${CONDA_INCLUDE_DIR}")
else()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isystem ${CONDA_INCLUDE_DIR}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CONDA_INCLUDE_DIR}")
endif()

# Vendor MPICH
if(DEFINED ENV{BODO_VENDOR_MPICH})
  message(STATUS "Downloading MPICH")
  execute_process(COMMAND ${Python_EXECUTABLE} -m pip download mpich --default-timeout=100)
  execute_process(COMMAND unzip "mpich-*.whl" "mpich*.data/*" -d mpich-tmp-extract-dir)
  file(GLOB MPICH_DATA_DIR "mpich-tmp-extract-dir/mpich-*.data/data/*")
  file(COPY ${MPICH_DATA_DIR} DESTINATION "${SKBUILD_DATA_DIR}" FOLLOW_SYMLINK_CHAIN
    FILE_PERMISSIONS OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE GROUP_WRITE)
  file(GLOB MPICH_WHEEL "mpich-*.whl")
  file(REMOVE_RECURSE ${MPICH_WHEEL} mpich-tmp-extract-dir)
  message(STATUS "Moved MPICH to ${SKBUILD_DATA_DIR}")

  set(MPI_INCLUDE_DIR "${SKBUILD_DATA_DIR}/include")
  set(MPI_LIB_DIR "${SKBUILD_DATA_DIR}/lib")
else()
  set(MPI_INCLUDE_DIR "${CONDA_INCLUDE_DIR}")
  set(MPI_LIB_DIR "${CONDA_LIB_DIR}")
endif()

# MPI implementation (Intel or MPICH)
if (WIN32)
  set(MPI_LIBRARIES "impi")
else()
 set(MPI_LIBRARIES "mpi")
endif()

cmake_print_variables(MPI_INCLUDE_DIR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PyArrowHelpers)
setup_pyarrow_dirs()

message(STATUS "PyArrow Includes: ${PYARROW_INCLUDE_DIR}")
message(STATUS "PyArrow Lib Dirs: ${PYARROW_LIBRARY_DIRS}")

# Numpy Include Directory is provided by scikit-build-core
# through `Python_NumPy_INCLUDE_DIR`
cmake_print_variables(Python_NumPy_INCLUDE_DIR)

# Combine All 2 Includes
set(BASE_INCLUDE_DIRS ${PYARROW_INCLUDE_DIR} ${Python_NumPy_INCLUDE_DIR} ${MPI_INCLUDE_DIR})


# ------------------------ Cython Target - bodo.io.csv_json_reader -----------------------
add_custom_command(
  OUTPUT bodo/io/csv_json_reader.cpp
  DEPENDS bodo/io/csv_json_reader.pyx
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "bodo/io/csv_json_reader.cpp" "bodo/io/csv_json_reader.pyx"
  COMMAND "${CMAKE_COMMAND}" -E copy "bodo/io/csv_json_reader.cpp" "${CMAKE_CURRENT_BINARY_DIR}/bodo/io/csv_json_reader.cpp"
  COMMENT "Cythonizing Source bodo/io/csv_json_reader.pyx into bodo/io/csv_json_reader.cpp"
)

python_add_library(csv_json_reader
  MODULE WITH_SOABI
  "bodo/io/_csv_json_reader.cpp"
  "bodo/io/csv_json_reader.cpp"
)
target_include_directories(csv_json_reader PRIVATE ${BASE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/bodo/io/" "${CMAKE_CURRENT_BINARY_DIR}/bodo/io/")
target_link_directories(csv_json_reader PRIVATE ${PYARROW_LIBRARY_DIRS} ${CONDA_LIB_DIR} ${MPI_LIB_DIR})
target_link_libraries(csv_json_reader PRIVATE fmt::fmt arrow arrow_python ${MPI_LIBRARIES})
install(TARGETS csv_json_reader DESTINATION "bodo/io/")

# ------------------------ Cython Target - bodo.io.pyarrow_wrappers ---------------
add_custom_command(
 OUTPUT bodo/io/pyarrow_wrappers.cpp
 DEPENDS bodo/io/pyarrow_wrappers.pyx
 VERBATIM
 WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
 COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "bodo/io/pyarrow_wrappers.cpp" "bodo/io/pyarrow_wrappers.pyx"
 COMMAND "${CMAKE_COMMAND}" -E copy "bodo/io/pyarrow_wrappers.cpp" "${CMAKE_CURRENT_BINARY_DIR}/bodo/io/pyarrow_wrappers.cpp"
 COMMENT "Cythonizing Source bodo/io/pyarrow_wrappers.pyx into bodo/io/pyarrow_wrappers.cpp"
)
set(pyarrow_wrappers_sources "bodo/io/pyarrow_wrappers.cpp" "bodo/io/arrow_compat.cpp")
python_add_library(pyarrow_wrappers MODULE WITH_SOABI "${pyarrow_wrappers_sources}")
target_include_directories(pyarrow_wrappers PRIVATE ${BASE_INCLUDE_DIRS})
target_link_directories(pyarrow_wrappers PRIVATE ${PYARROW_LIBRARY_DIRS} ${CONDA_LIB_DIR} ${MPI_LIB_DIR})
# On Windows this is necessary to define due to importing/exporting DLL semantics.
# See https://github.com/apache/arrow/blob/9105a4109a80a1c01eabb24ee4b9f7c94ee942cb/python/CMakeLists.txt#L472
target_compile_definitions(pyarrow_wrappers PRIVATE ARROW_PYTHON_EXPORTING)
target_link_libraries(pyarrow_wrappers PRIVATE arrow arrow_python arrow_dataset "${MPI_LIBRARIES}")
install(TARGETS pyarrow_wrappers DESTINATION "bodo/io/")

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
  COMMAND "${CYTHON_EXECUTABLE}" -3 -E "BODO_DEV_BUILD=${BODO_DEV_BUILD}" --output-file "bodo/utils/tracing.c" "bodo/utils/tracing.pyx"
  COMMAND "${CMAKE_COMMAND}" -E copy "bodo/utils/tracing.c" "${CMAKE_CURRENT_BINARY_DIR}/bodo/utils/tracing.c"
  COMMENT "Cythonizing Source bodo/utils/tracing.pyx into bodo/utils/tracing.c"
)

python_add_library(tracing MODULE WITH_SOABI "bodo/utils/tracing.c")
target_include_directories(tracing PRIVATE ${MPI_INCLUDE_DIR})
target_link_directories(tracing PRIVATE ${CONDA_LIB_DIR} ${MPI_LIB_DIR})
target_link_libraries(tracing PRIVATE "${MPI_LIBRARIES}")
install(TARGETS tracing DESTINATION "bodo/utils/")

# ---------------------- Cython Target - bodo.memory -----------------------
add_custom_command(
  OUTPUT bodo/memory.cpp
  DEPENDS bodo/memory.pyx
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "bodo/memory.cpp" "bodo/memory.pyx"
  COMMAND "${CMAKE_COMMAND}" -E copy "bodo/memory.cpp" "${CMAKE_CURRENT_BINARY_DIR}/bodo/memory.cpp"
  COMMENT "Cythonizing Source bodo/memory.pyx into bodo/memory.cpp"
)

python_add_library(memory MODULE WITH_SOABI "bodo/memory.cpp")
target_include_directories(memory PRIVATE ${BASE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/bodo/")
target_link_directories(memory PRIVATE ${PYARROW_LIBRARY_DIRS} ${CONDA_LIB_DIR})
target_link_libraries(memory PRIVATE arrow arrow_python)
install(TARGETS memory DESTINATION "bodo/")

# ---------------------- Cython Target - bodo.tests.memory_tester -----------------------
add_custom_command(
  OUTPUT bodo/tests/memory_tester.cpp
  DEPENDS bodo/tests/memory_tester.pyx
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "bodo/tests/memory_tester.cpp" "bodo/tests/memory_tester.pyx"
  COMMAND "${CMAKE_COMMAND}" -E copy "bodo/tests/memory_tester.cpp" "${CMAKE_CURRENT_BINARY_DIR}/bodo/tests/memory_tester.cpp"
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
target_link_directories(memory_tester PRIVATE ${PYARROW_LIBRARY_DIRS} ${CONDA_LIB_DIR} ${MPI_LIB_DIR})
target_link_libraries(memory_tester PRIVATE "${MPI_LIBRARIES}" arrow arrow_python fmt::fmt)
if(NOT(CMAKE_BUILD_TYPE STREQUAL "Release"))
  target_compile_definitions(
    memory_tester
    PRIVATE
    # Required when using boost::stacktrace for debugging
    "BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED=1"
  )
endif()
install(TARGETS memory_tester DESTINATION "bodo/tests/")


# TODO[BSE-5071]: Re-enable native typer when its coverage improved
# # ---------------------- Cython Target - bodo.transforms.type_inference.native_typer -----------------------
# add_custom_command(
#   OUTPUT bodo/transforms/type_inference/native_typer.cpp
#   DEPENDS bodo/transforms/type_inference/native_typer.pyx
#   VERBATIM
#   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#   COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "bodo/transforms/type_inference/native_typer.cpp" "bodo/transforms/type_inference/native_typer.pyx"
#   COMMAND "${CMAKE_COMMAND}" -E copy "bodo/transforms/type_inference/native_typer.cpp" "${CMAKE_CURRENT_BINARY_DIR}/bodo/transforms/type_inference/native_typer.cpp"
#   COMMENT "Cythonizing Source bodo/transforms/type_inference/native_typer.pyx into bodo/transforms/type_inference/native_typer.cpp"
# )

# python_add_library(
#   native_typer
#   MODULE WITH_SOABI
#     "bodo/transforms/type_inference/native_typer.cpp"
#     "bodo/transforms/type_inference/type.cpp"
#     "bodo/transforms/type_inference/typeinfer.cpp"
#     "bodo/transforms/type_inference/ir.cpp"
# )
# target_include_directories(native_typer PRIVATE ${BASE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/bodo/transforms/type_inference/" "${CMAKE_CURRENT_BINARY_DIR}/bodo/transforms/type_inference/")
# target_link_directories(native_typer PRIVATE ${CONDA_LIB_DIR})
# target_link_libraries(native_typer PRIVATE fmt::fmt)
# install(TARGETS native_typer DESTINATION "bodo/transforms/type_inference/")


# ---------------------- Cython Target - plan_optimizer -----------------------
add_custom_command(
  OUTPUT bodo/pandas/plan_optimizer.cpp
  DEPENDS bodo/pandas/plan_optimizer.pyx
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND "${CYTHON_EXECUTABLE}" --cplus -3 --output-file "bodo/pandas/plan_optimizer.cpp" "bodo/pandas/plan_optimizer.pyx"
  COMMAND "${CMAKE_COMMAND}" -E copy "bodo/pandas/plan_optimizer.cpp" "${CMAKE_CURRENT_BINARY_DIR}/bodo/pandas/plan_optimizer.cpp"
  COMMENT "Cythonizing Source bodo/pandas/plan_optimizer.pyx into bodo/pandas/plan_optimizer.cpp"
)


# ---------------------- Native Target - crypto_funcs -----------------------

python_add_library(crypto_funcs
  MODULE WITH_SOABI
  "bodo/libs/_crypto_funcs.cpp"
  "bodo/libs/_base64.cpp"
)
target_include_directories(crypto_funcs PRIVATE ${BASE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/bodo/libs/" "${CMAKE_CURRENT_BINARY_DIR}/bodo/libs/")
target_link_directories(crypto_funcs PRIVATE ${CONDA_LIB_DIR})
if (WIN32)
  target_link_libraries(crypto_funcs PRIVATE OpenSSL::SSL OpenSSL::Crypto)
else()
  target_link_libraries(crypto_funcs PRIVATE ssl crypto)
endif()
install(TARGETS crypto_funcs DESTINATION "bodo/libs/")


# ---------------------- Native Target - memory_cpp -----------------------

python_add_library(memory_cpp
  MODULE WITH_SOABI
  "bodo/libs/_pymemory.cpp"
  "bodo/libs/_memory.cpp"
  "bodo/libs/_utils.cpp"
  "bodo/libs/_storage_manager.cpp"
)
target_include_directories(memory_cpp PRIVATE ${BASE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/bodo/libs/" "${CMAKE_CURRENT_BINARY_DIR}/bodo/libs/")
target_link_directories(memory_cpp PRIVATE ${CONDA_LIB_DIR} ${PYARROW_LIBRARY_DIRS} ${MPI_LIB_DIR})
target_link_libraries(memory_cpp PRIVATE fmt::fmt arrow ${MPI_LIBRARIES})
install(TARGETS memory_cpp DESTINATION "bodo/")


# ------------------------- Main Bodo Extension -------------------------
set(sources_list
  "bodo/pandas/plan_optimizer.cpp"
  "bodo/pandas/_bodo_scan_function.cpp"
  "bodo/pandas/_bodo_write_function.cpp"
  "bodo/pandas/_plan.cpp"
  "bodo/pandas/_pipeline.cpp"
  "bodo/pandas/_physical_conv.cpp"
  "bodo/pandas/_executor.cpp"
  "bodo/pandas/_util.cpp"
  "bodo/pandas/physical/expression.cpp"
  "bodo/pandas/physical/operator.cpp"
  "bodo/pandas/physical/read_iceberg.cpp"
  "bodo/pandas/physical/reduce.cpp"
  "bodo/pandas/optimizer/runtime_join_filter.cpp"
  "bodo/io/_csv_json_writer.cpp"
  "bodo/io/_fs_io.cpp"
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
  "bodo/libs/_bodo_ext.cpp"
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
  "bodo/libs/_uuid.cpp"
  "bodo/libs/_query_profile_collector.cpp"
  "bodo/libs/_theta_sketches.cpp"
  "bodo/libs/_puffin.cpp"
  "bodo/libs/_io_cpu_thread_pool.cpp"
)

if (BUILD_WITH_V8)
  set(sources_list "${sources_list}"
    "bodo/libs/_javascript_udf.cpp")
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
)

# TODO: Replace with ctest when integrating Catch2
if (NOT (CMAKE_BUILD_TYPE STREQUAL "Release") AND NOT DEFINED ENV{BODO_SKIP_CPP_TESTS})
  set(test_list
    "bodo/tests/test_framework.cpp"
    "bodo/tests/test_example.cpp"
    "bodo/tests/test_external_sort.cpp"
    "bodo/tests/test_dict_builder.cpp"
    "bodo/tests/test_groupby_common.cpp"
    "bodo/tests/test_groupby.cpp"
    "bodo/tests/test_window.cpp"
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
    "bodo/tests/test_sorted_window_computation.cpp"
    "bodo/tests/test_thread_pool.cpp"
    "bodo/tests/test_datatypes.cpp"
  )
  if (BUILD_WITH_V8)
      set(test_list "${test_list}"
      "bodo/tests/test_javascript.cpp"
      )
  endif()
  if (USE_CUDF)
        set(test_list "${test_list}"
        "bodo/tests/test_streaming/test_gpu_shuffle.cpp"
        "bodo/tests/test_streaming/test_gpu_batch_gen.cpp"
        )
  endif()
  list(APPEND sources_list "${test_list}")
  list(APPEND allow_implicit_fallthrough_list
    "bodo/tests/test_puffin.cpp"
    "bodo/tests/test_theta_sketches.cpp")
endif()

# TODO [BSE-4554] HDF5 support on Windows
if(DEFINED ENV{NO_HDF5} AND "$ENV{NO_HDF5}" STREQUAL "1")
    set(NO_HDF5 TRUE)
else()
    list(APPEND sources_list "bodo/io/_hdf5.cpp")
    set(NO_HDF5 FALSE)
endif()
cmake_print_variables(NO_HDF5)

if (USE_CUDF)
    list(APPEND sources_list
    "bodo/pandas/physical/gpu_expression.cpp"
    "bodo/pandas/physical/gpu_aggregate.cpp"
    "bodo/libs/streaming/cuda_join.cpp"
    "bodo/libs/streaming/cuda_groupby.cpp"
    "bodo/libs/gpu_utils.cpp"
    )
endif()

python_add_library(ext MODULE WITH_SOABI "${sources_list}")

# Dependency Includes Are `-isystem` to Suppress Warnings
target_include_directories(ext SYSTEM PRIVATE ${BASE_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_directories(ext PRIVATE ${CONDA_LIB_DIR} ${MPI_LIB_DIR})

# plan_optimizer build on top of DuckDB
target_include_directories(ext PRIVATE ${BASE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/bodo/pandas/" "${CMAKE_CURRENT_BINARY_DIR}/bodo/pandas/"
)
target_link_directories(ext PRIVATE ${CONDA_LIB_DIR}
)
target_link_libraries(ext PRIVATE
  duckdb_static datasketches
)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  if (NOT WIN32)
    target_compile_options(ext PRIVATE "-Wno-unknown-pragmas")
  endif()
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

if (NOT WIN32)
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
endif()

target_link_directories(ext PRIVATE ${PYARROW_LIBRARY_DIRS} "${Python_NumPy_INCLUDE_DIR}/../lib")
target_compile_definitions(ext PRIVATE ARROW_PYTHON_EXPORTING)
target_link_libraries(ext PRIVATE
  npymath
  arrow
  arrow_python
  arrow_dataset
  parquet
  fmt::fmt
  zstd
  aws-cpp-sdk-core
  ${MPI_LIBRARIES}
)

if (NOT WIN32)
  target_link_libraries(ext PRIVATE pthread)
endif()

set(IS_PLATFORM "$ENV{IS_BODO_PLATFORM}")
cmake_print_variables(IS_PLATFORM)

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

if(NO_HDF5)
  target_compile_definitions(ext PRIVATE "NO_HDF5=1")
else()
  if(WIN32)
    # Needed to fix DLL import on Windows
    # https://forum.hdfgroup.org/t/unresolved-external-symbol-h5t-native-int-g/7550
    target_compile_definitions(ext PRIVATE "H5_BUILT_AS_DYNAMIC_LIB=1")
  endif()
  target_link_libraries(ext PRIVATE hdf5)
endif()

if(USE_CUDF)
  message(STATUS "Configuring for cudf.")
  execute_process( COMMAND "${Python_EXECUTABLE}" -c "import os, pathlib; \ root = os.environ.get('CONDA_PREFIX', ''); \ matches = list(pathlib.Path(root).rglob('cuda_runtime.h')); \ print(matches[0].parent if matches else '')" OUTPUT_VARIABLE CUDA_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE )

  message(STATUS "Using CUDA include dir: ${CUDA_INCLUDE_DIR}")

  target_include_directories(ext BEFORE PRIVATE
    ${CUDA_INCLUDE_DIR} ${CUDA_INCLUDE_DIR}/cccl ${PYLIBCUDF_BASE_DIR} ${RMM_BASE_DIR}
  )

  # RMM library defines
  find_path(RMM_INCLUDE_DIR NAMES rmm/cuda_stream_view.hpp)
  find_library(RMM_LIB NAMES librmm.so)
  get_filename_component(RMM_LIBDIR "${RMM_LIB}" DIRECTORY)
  message(STATUS "RMM include: ${RMM_INCLUDE_DIR} lib: ${RMM_LIBDIR}")
  add_library(rmm SHARED IMPORTED)
  set_target_properties(rmm PROPERTIES IMPORTED_LOCATION "${RMM_LIB}")
  # Rapids defines
  find_path(RAPIDS_INCLUDE_DIR NAMES cuda/memory_resource cuda/mr/any_resource HINTS "$ENV{CONDA_PREFIX}/include"
      PATH_SUFFIXES rapids
      REQUIRED)
  message(STATUS "Rapids include: ${RAPIDS_INCLUDE_DIR}")
  # Rapids logger defines
  find_path(RAPIDS_LOGGER_INCLUDE_DIR NAMES rapids_logger/log_levels.h)
  find_library(RL_LIB NAMES librapids_logger.so)
  get_filename_component(RL_LIBDIR "${RL_LIB}" DIRECTORY)
  message(STATUS "RapidsLogger include: ${RAPIDS_LOGGER_INCLUDE_DIR} lib: ${RL_LIBDIR}")
  add_library(rapids_logger SHARED IMPORTED)
  set_target_properties(rapids_logger PROPERTIES IMPORTED_LOCATION "${RL_LIB}")
  # libcudf defines
  find_path(LIBCUDF_INCLUDE_DIR NAMES cudf/types.hpp)
  find_library(LIBCUDF_LIB NAMES libcudf.so)
  get_filename_component(LIBCUDF_LIBDIR "${LIBCUDF_LIB}" DIRECTORY)
  message(STATUS "libcudf include: ${LIBCUDF_INCLUDE_DIR} lib: ${LIBCUDF_LIB}")
  add_library(libcudf SHARED IMPORTED)
  set_target_properties(libcudf PROPERTIES IMPORTED_LOCATION "${LIBCUDF_LIB}")

  # cudart
  find_library(LIBCUDART_LIB NAMES libcudart.so.13)
  get_filename_component(LIBCUDART_LIBDIR "${LIBCUDART_LIB}" DIRECTORY)
  message(STATUS "libcudart lib: ${LIBCUDART_LIB} libdir: ${LIBCUDART_LIBDIR}")
  add_library(libcudart SHARED IMPORTED)
  set_target_properties(libcudart PROPERTIES IMPORTED_LOCATION "${LIBCUDART_LIB}")

  # nccl
  find_library(NCCL_LIB NAMES libnccl.so)
  get_filename_component(NCCL_LIBDIR "${NCCL_LIB}" DIRECTORY)
  message(STATUS "libnccl lib: ${NCCL_LIB} libdir: ${NCCL_LIBDIR}")
  add_library(nccl SHARED IMPORTED)
  set_target_properties(nccl PROPERTIES IMPORTED_LOCATION "${NCCL_LIB}")

  # RPATH
  set_target_properties(ext PROPERTIES
    BUILD_RPATH "${RMM_LIBDIR};${RL_LIBDIR};${LIBCUDF_LIBDIR};${LIBCUDART_LIBDIR};${NCCL_LIBDIR}"
    INSTALL_RPATH "${RMM_LIBDIR};${RL_LIBDIR};${LIBCUDF_LIBDIR};${LIBCUDART_LIBDIR};${NCCL_LIBDIR}"
  )

  target_compile_definitions(ext PRIVATE LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE=1)
  message(STATUS "cudf include dirs: ${cudf_INCLUDE_DIRS}")
  target_include_directories(ext BEFORE PRIVATE ${RAPIDS_INCLUDE_DIR} ${cudf_INCLUDE_DIRS} ${RMM_INCLUDE_DIR} ${RAPIDS_LOGGER_INCLUDE_DIR} ${LIBCUDF_INCLUDE_DIR} )
  target_link_libraries(ext PRIVATE rmm rapids_logger libcudf libcudart nccl
      "-Wl,--as-needed")
endif()

# TODO: fix excluding plan_optimizer.cpp and enable clang-tidy
# set_target_properties(ext PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
# # Exclude Cython generated code from clang-tidy since it can hang
# set_source_files_properties(bodo/pandas/plan_optimizer.cpp PROPERTIES CXX_CLANG_TIDY "")

add_dependencies(ext csv_json_reader tracing pyarrow_wrappers)
install(TARGETS ext DESTINATION "bodo/")
