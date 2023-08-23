# isort: skip_file
import sys
import os
import glob
import platform
from typing import Any, Dict
from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize

import numpy.distutils.misc_util as np_misc
import pyarrow

import versioneer

cwd = os.getcwd()
setup_py_dir_path = os.path.dirname(os.path.realpath(__file__))
# despite the name, this also works for directories
if not os.path.samefile(cwd, setup_py_dir_path):
    raise Exception(
        "setup.py should only be invoked if the current working directory is in the same directory as Setup.py.\nThis is to prevent having with conflicting .egg-info in the same directory when building Bodo's submodules."
    )

is_win = platform.system() == "Windows"
is_mac = platform.system() == "Darwin"
is_m1_mac = is_mac and platform.machine() == "arm64"

# develop_mode: Compile the code, etc.
# clean_mode: Delete build files
# install_mode: Convert bodo/transform to .pyx files and Cythonize
develop_mode = "develop" in sys.argv
clean_mode = "clean" in sys.argv
install_mode = (
    ("install" in sys.argv) or ("build" in sys.argv) or ("bdist_wheel" in sys.argv)
)

if [develop_mode, clean_mode, install_mode].count(True) != 1:
    raise ValueError(f"Please specify a valid mode [develop | clean | install]")

if develop_mode:
    if "--no-ccache" not in sys.argv:
        import shutil

        if shutil.which("ccache"):
            if "ccache" not in os.environ["CC"]:
                # ccache is very useful when developing C++ code, because on
                # rebuild it will only recompile the cpp files that have been
                # modified.
                # This only modifies the environment variables CC and CXX
                # during execution of `python setup.py develop`.
                # With newer compiler versions on Linux, we have to set both CC
                # and CXX variables for ccache to work, and have to set CC to
                # the C++ compiler to avoid compile-time and dynamic linking
                # errors
                os.environ["CC"] = f"ccache {os.environ['CXX']}"
                os.environ["CXX"] = f"ccache {os.environ['CXX']}"
        else:
            print(
                """
ccache not found. ccache is strongly recommended when developing
C++ code to greatly improve recompilation times. Install ccache
with `conda install ccache -c conda-forge`

Use --no-ccache to build without ccache.
                """
            )
            exit(1)
    else:
        sys.argv.remove("--no-ccache")


def readme():
    with open("README_pypi.md") as f:
        return f.read()


# package environment variable is PREFIX during build time
if "CONDA_BUILD" in os.environ:
    PREFIX_DIR = os.environ["PREFIX"]
else:
    PREFIX_DIR = os.environ["CONDA_PREFIX"]
    # C libraries are in \Library on Windows
    if is_win:
        PREFIX_DIR += "\Library"


try:
    import h5py  # noqa
except ImportError:
    # currently, due to cross compilation, the import fails.
    # building the extension modules still works though.
    # TODO: resolve issue with h5py import
    h5py_version = None
else:
    # NOTE: conda-forge does not have MPI-enabled hdf5 for Windows yet
    # TODO: make sure the available hdf5 library is MPI-enabled automatically
    h5py_version = h5py.version.hdf5_version_tuple[1]


# Define include dirs, library dirs, extra compile args, and extra link args
ind = [PREFIX_DIR + "/include"]
lid = [PREFIX_DIR + "/lib"]
# eca = ["-std=c++20", "-fsanitize=address"]
# ela = ["-std=c++20", "-fsanitize=address"]

# Pass --debug flag to setup.py to compile with debug symbols
opt_flag = "-O3"
profile_flag = []
if "--debug" in sys.argv:
    debug_flag = "-g"
    opt_flag = "-O1"  # This makes many c++ features easier to debug
    sys.argv.remove("--debug")
else:
    debug_flag = "-g0"

is_testing = os.environ.get("NUMBA_DEVELOPER_MODE") == "1"
if "--no-test" in sys.argv:
    is_testing = False
    sys.argv.remove("--no-test")

if "--profile" in sys.argv:
    profile_flag = ["-pg"]
    if debug_flag == "-g0":
        debug_flag = "-g"
    sys.argv.remove("--profile")

address_sanitizer_flag = []
if "--address-sanitizer" in sys.argv:
    sys.argv.remove("--address-sanitizer")
    address_sanitizer_flag = ["-fsanitize=address"]
    opt_flag = "-O1"

if is_win:
    eca = ["/std:c++20", "/O2"]
    eca_c = ["/O2"]
    ela = ["/std:c++20"]
else:
    if is_mac:
        # Mac on CI can't support AVX2
        eca = ["-std=c++20", debug_flag, opt_flag] + address_sanitizer_flag
    else:
        # -march=haswell is used to enable AVX2 support (required by SIMD bloom
        # filter implementation)
        eca = (
            [
                "-std=c++20",
                debug_flag,
                opt_flag,
                "-march=haswell",
            ]
            + address_sanitizer_flag
            + profile_flag
        )
    eca_c = [debug_flag, opt_flag] + profile_flag
    ela = ["-std=c++20"]

if develop_mode:
    eca.append("-Werror")
    # avoid GCC errors for using int64 in allocations
    if not (is_m1_mac or is_mac):
        eca.append("-Wno-alloc-size-larger-than")
eca.append("-Wno-c99-designator")


# Use a single C-extension for all of Bodo
# Copying ind, lid, eca, and ela to avoid aliasing, as we continue to append
ext_metadata: Dict[str, Any] = dict(
    name="bodo.ext",
    sources=[],
    depends=[],
    include_dirs=list(ind),
    define_macros=[],
    library_dirs=list(lid),
    libraries=[],
    extra_compile_args=list(eca),
    extra_link_args=list(ela),
    language="c++",
)

# Add hash includes
ext_metadata["include_dirs"] += [
    os.path.join("bodo", "libs", "HashLibs", "TSL", "hopscotch-map"),
    os.path.join("bodo", "libs", "HashLibs", "TSL", "robin-map"),
    os.path.join("bodo", "libs", "HashLibs", "TSL", "sparse-map"),
]

# Add third-party libraries
mpi_libs = ["mpi"]
if is_win:
    # use Microsoft MPI on Windows
    mpi_libs = ["msmpi"]
    if os.environ.get("BUILD_PIP", "") == "1":
        # building pip package, need to set additional library paths
        pyarrow_dirname = os.path.dirname(pyarrow.__file__)
        ext_metadata["extra_link_args"].append(f"/LIBPATH:{pyarrow_dirname}")
    # hdf5-parallel Windows build uses CMake which needs this flag
    ext_metadata["define_macros"].append(("H5_BUILT_AS_DYNAMIC_LIB", None))

ext_metadata["libraries"] += mpi_libs + ["hdf5", "arrow", "arrow_python", "parquet"]

# Even though we upgraded to 1.12 which changes the API, these flags keep the
# 1.10 API. See the example: https://github.com/openmc-dev/openmc/pull/1533
# This was also verified by looking at the .h files.
# If we still have version 1.10 these are ignored.
ext_metadata["extra_compile_args"] += [
    "-DH5Oget_info_by_name_vers=1",
    "-DH5Oget_info_vers=1",
    "-DH5O_info_t_vers=1",
]

# TODO Windows build fails because ssl.lib not found. Disabling licensing
# check on windows for now
is_expired = os.environ.get("CHECK_LICENSE_EXPIRED", None) == "1"
is_core_count = os.environ.get("CHECK_LICENSE_CORE_COUNT", None) == "1"
is_platform = os.environ.get("CHECK_LICENSE_PLATFORM", None) == "1"

if is_expired:
    ext_metadata["define_macros"].append(("CHECK_LICENSE_EXPIRED", "1"))

if is_core_count:
    ext_metadata["define_macros"].append(("CHECK_LICENSE_CORE_COUNT", "1"))

if is_platform:
    assert (not is_expired) and (not is_core_count)
    ext_metadata["define_macros"].append(("CHECK_LICENSE_PLATFORM", "1"))
    ext_metadata["include_dirs"].append("bodo/libs/gason")
    ext_metadata["sources"].append("bodo/libs/gason/gason.cpp")
    ext_metadata["libraries"].append("curl")

if is_expired or is_core_count:
    if is_win:
        ext_metadata["libraries"] += ["libssl", "libcrypto"]
    else:
        ext_metadata["libraries"] += ["ssl", "crypto"]

# -fno-strict-aliasing required by bloom filter implementation (see comment
# in simd-block-fixed-fpp.h about violating strict aliasing rules)
ext_metadata["extra_compile_args"].append("-fno-strict-aliasing")

# Define sources and dependencies
ext_metadata["sources"] += [
    "bodo/io/_csv_json_reader.cpp",
    "bodo/io/_csv_json_writer.cpp",
    "bodo/io/_fs_io.cpp",
    "bodo/io/_fsspec_reader.cpp",
    "bodo/io/_hdf5.cpp",
    "bodo/io/_hdfs_reader.cpp",
    "bodo/io/_io.cpp",
    "bodo/io/_s3_reader.cpp",
    "bodo/io/arrow.cpp",
    "bodo/io/arrow_reader.cpp",
    "bodo/io/iceberg_parquet_reader.cpp",
    "bodo/io/iceberg_parquet_write.cpp",
    "bodo/io/parquet_reader.cpp",
    "bodo/io/parquet_write.cpp",
    "bodo/io/snowflake_reader.cpp",
    "bodo/libs/_array.cpp",
    "bodo/libs/_array_hash.cpp",
    "bodo/libs/_array_operations.cpp",
    "bodo/libs/_array_utils.cpp",
    "bodo/libs/_bodo_common.cpp",
    "bodo/libs/_bodo_tdigest.cpp",
    "bodo/libs/_bodo_to_arrow.cpp",
    "bodo/libs/_datetime_ext.cpp",
    "bodo/libs/_datetime_utils.cpp",
    "bodo/libs/_decimal_ext.cpp",
    "bodo/libs/_distributed.cpp",
    "bodo/libs/_groupby.cpp",
    "bodo/libs/_groupby_agg_funcs.cpp",
    "bodo/libs/_groupby_col_set.cpp",
    "bodo/libs/_groupby_common.cpp",
    "bodo/libs/_groupby_do_apply_to_column.cpp",
    "bodo/libs/_groupby_eval.cpp",
    "bodo/libs/_groupby_ftypes.cpp",
    "bodo/libs/_groupby_groups.cpp",
    "bodo/libs/_groupby_mode.cpp",
    "bodo/libs/_groupby_mpi_exscan.cpp",
    "bodo/libs/_groupby_update.cpp",
    "bodo/libs/_hash_join.cpp",
    "bodo/libs/_nested_loop_join.cpp",
    "bodo/libs/_interval_join.cpp",
    "bodo/libs/_join_hashing.cpp",
    "bodo/libs/_lead_lag.cpp",
    "bodo/libs/_crypto_funcs.cpp",
    "bodo/libs/_memory.cpp",
    "bodo/libs/_murmurhash3.cpp",
    "bodo/libs/_quantile_alg.cpp",
    "bodo/libs/_shuffle.cpp",
    "bodo/libs/_str_ext.cpp",
    "bodo/libs/iceberg_transforms.cpp",
    "bodo/libs/_stream_join.cpp",
    "bodo/libs/_stream_nested_loop_join.cpp",
    "bodo/libs/_stream_groupby.cpp",
    "bodo/libs/_dict_builder.cpp",
    "bodo/libs/_table_builder_utils.cpp",
    "bodo/libs/_table_builder.cpp",
    "bodo/libs/_chunked_table_builder.cpp",
    "bodo/libs/_listagg.cpp",
    "bodo/libs/_operator_pool.cpp",
    "bodo/libs/_window_aggfuncs.cpp",
    "bodo/libs/_window_compute.cpp",
    "bodo/libs/_stream_dict_encoding.cpp",
]
ext_metadata["depends"] += [
    "bodo/io/_bodo_file_reader.h",
    "bodo/io/_csv_json_reader.h",
    "bodo/io/_fs_io.h",
    "bodo/io/_io.h",
    "bodo/io/arrow_reader.h",
    "bodo/io/parquet_reader.h",
    "bodo/io/parquet_write.h",
    "bodo/libs/_array_hash.h",
    "bodo/libs/_array_operations.h",
    "bodo/libs/_array_utils.h",
    "bodo/libs/_bodo_common.h",
    "bodo/libs/_bodo_tdigest.h",
    "bodo/libs/_bodo_to_arrow.h",
    "bodo/libs/_datetime_utils.h",
    "bodo/libs/_decimal_ext.h",
    "bodo/libs/_distributed.h",
    "bodo/libs/_groupby.h",
    "bodo/libs/_groupby_agg_funcs.h",
    "bodo/libs/_groupby_col_set.h",
    "bodo/libs/_groupby_common.h",
    "bodo/libs/_groupby_do_apply_to_column.h",
    "bodo/libs/_groupby_eval.h",
    "bodo/libs/_groupby_ftypes.h",
    "bodo/libs/_groupby_groups.h",
    "bodo/libs/_groupby_hashing.h",
    "bodo/libs/_groupby_mpi_exscan.h",
    "bodo/libs/_groupby_udf.h",
    "bodo/libs/_groupby_update.h",
    "bodo/libs/_import_py.h",
    "bodo/libs/_join.h",
    "bodo/libs/_join_hashing.h",
    "bodo/libs/_lead_lag.h",
    "bodo/libs/_crypto_funcs.h",
    "bodo/libs/_meminfo.h",
    "bodo/libs/_memory.h",
    "bodo/libs/_murmurhash3.h",
    "bodo/libs/_shuffle.h",
    "bodo/libs/hyperloglog.hpp",
    "bodo/libs/iceberg_transforms.h",
    "bodo/libs/_stream_join.h",
    "bodo/libs/_stream_groupby.h",
    "bodo/libs/simd-block-fixed-fpp.h",
    "bodo/libs/_dict_builder.h",
    "bodo/libs/_table_builder_utils.h",
    "bodo/libs/_table_builder.h",
    "bodo/libs/_listagg.h",
    "bodo/libs/_chunked_table_builder.h",
    "bodo/libs/_nested_loop_join.h",
    "bodo/libs/_operator_pool.h",
    "bodo/libs/_window_aggfuncs.h",
    "bodo/libs/_window_compute.h",
    "bodo/libs/_stream_dict_encoding.h",
    "bodo/libs/_pinnable.h",
    "bodo/libs/_nested_loop_join_impl.h",
]

if is_testing:
    ext_metadata["sources"].extend(
        [
            "bodo/tests/test_framework.cpp",
            "bodo/tests/test_example.cpp",
            "bodo/tests/test_pinnable.cpp",
            "bodo/tests/test_dict_builder.cpp",
            "bodo/tests/test_table_builder.cpp",
            "bodo/tests/test_table_generator.cpp",
        ]
    )
    ext_metadata["define_macros"].append(("IS_TESTING", "1"))


# We cannot compile with -Werror yet because _fsspec_reader.cpp
# depends on pyfs.cpp which generates a warning.
ext_metadata["extra_compile_args"] = [
    x for x in ext_metadata["extra_compile_args"] if x != "-Werror"
]

# Inject required options for extensions compiled against the Numpy
# C API (include dirs, library dirs etc.)
np_compile_args = np_misc.get_info("npymath")
ext_metadata["libraries"] += np_compile_args["libraries"]
ext_metadata["include_dirs"] += np_compile_args["include_dirs"]
ext_metadata["library_dirs"] += np_compile_args["library_dirs"]
ext_metadata["define_macros"] += np_compile_args["define_macros"]

# Inject required options for extensions compiled against
# PyArrow and Arrow
pa_compile_args = {
    "include_dirs": [pyarrow.get_include()],
    "library_dirs": pyarrow.get_library_dirs(),
}
ext_metadata["include_dirs"].extend(pa_compile_args["include_dirs"])
ext_metadata["library_dirs"].extend(pa_compile_args["library_dirs"])

# Compile Bodo extension
bodo_ext = Extension(**ext_metadata)


# Build extensions for Cython files that are part of the code base, and aren't
# just renamed .py files during build
# These .pyx files are always part of Bodo (not generated during build)
builtin_exts = []
pyx_builtins = []

ext_arrow = Extension(
    name="bodo.io.arrow_ext",
    sources=["bodo/io/arrow_ext.pyx"],
    include_dirs=np_compile_args["include_dirs"]
    + ind
    + pa_compile_args["include_dirs"],
    define_macros=[],
    library_dirs=lid + pa_compile_args["library_dirs"],
    libraries=["arrow", "arrow_python"],
    # We cannot compile with -Werror yet because pyfs.cpp
    # generates serveral warnings.
    extra_compile_args=[x for x in eca if x != "-Werror"],
    extra_link_args=ela,
    language="c++",
)
builtin_exts.append(ext_arrow)
pyx_builtins.append(os.path.join("bodo", "io", "arrow_ext.pyx"))

ext_pyfs = Extension(
    name="bodo.io.pyfs",
    sources=["bodo/io/pyfs.pyx"],
    include_dirs=np_compile_args["include_dirs"]
    + ind
    + pa_compile_args["include_dirs"],
    define_macros=[],
    library_dirs=lid + pa_compile_args["library_dirs"],
    libraries=["arrow", "arrow_python"],
    # We cannot compile with -Werror yet because pyfs.cpp
    # generates serveral warnings.
    extra_compile_args=[x for x in eca if x != "-Werror"],
    extra_link_args=ela,
    language="c++",
)
builtin_exts.append(ext_pyfs)
pyx_builtins.append(os.path.join("bodo", "io", "pyfs.pyx"))

ext_hdfs_pyarrow = Extension(
    name="bodo.io._hdfs",
    sources=["bodo/io/_hdfs.pyx"],
    include_dirs=np_compile_args["include_dirs"]
    + ind
    + pa_compile_args["include_dirs"],
    define_macros=[],
    library_dirs=lid + pa_compile_args["library_dirs"],
    libraries=["arrow", "arrow_python"],
    # We cannot compile with -Werror yet because hdfs.cpp
    # generates serveral warnings.
    extra_compile_args=[x for x in eca if x != "-Werror"],
    extra_link_args=ela,
    language="c++",
)
builtin_exts.append(ext_hdfs_pyarrow)
pyx_builtins.append(os.path.join("bodo", "io", "_hdfs.pyx"))

ext_tracing = Extension(
    name="bodo.utils.tracing",
    sources=["bodo/utils/tracing.pyx"],
    include_dirs=ind,
    define_macros=[],
    library_dirs=lid,
    libraries=mpi_libs,
    extra_compile_args=eca_c,
    extra_link_args=ela,
)
builtin_exts.append(ext_tracing)
pyx_builtins.append(os.path.join("bodo", "utils", "tracing.pyx"))

ext_memory = Extension(
    name="bodo.libs.memory",
    sources=[
        "bodo/libs/memory.pyx",
        "bodo/libs/_memory.cpp",
        "bodo/libs/_operator_pool.cpp",
    ],
    include_dirs=np_compile_args["include_dirs"]
    + ind
    + pa_compile_args["include_dirs"],
    define_macros=[],
    library_dirs=lid + pa_compile_args["library_dirs"],
    libraries=["arrow", "arrow_python"] + mpi_libs,
    # Cannot compile with -Werror yet because memory.cpp
    # generated multiple unused variable warnings
    extra_compile_args=[x for x in eca if x != "-Werror"],
    extra_link_args=ela,
    language="c++",
)
builtin_exts.append(ext_memory)
pyx_builtins.append(os.path.join("bodo", "libs", "memory.pyx"))


# Handle additional .pyx files not in pyx_builtins
if clean_mode:
    _cython_ext_mods = [
        f for f in glob.glob("bodo/**/*.pyx", recursive=True) if f not in pyx_builtins
    ]
elif develop_mode:
    _cython_ext_mods = []
    # make sure there are no .pyx files in development mode
    pyxs = [
        f for f in glob.glob("bodo/**/*.pyx", recursive=True) if f not in pyx_builtins
    ]
    assert len(pyxs) == 0, (
        "Found .pyx files in development mode. Please clear all of the "
        "generated files in bodo/transforms and manually restore any deleted "
        "Python files from Git."
    )
elif install_mode:
    import subprocess

    # rename select files to .pyx for cythonizing
    subprocess.run([sys.executable, "rename_to_pyx.py"])
    _cython_ext_mods = [
        f for f in glob.glob("bodo/**/*.pyx", recursive=True) if f not in pyx_builtins
    ]
else:
    _cython_ext_mods = []


setup(
    name="bodo",
    version=versioneer.get_version(),
    description="The Python Supercomputing Analytics Platform",
    long_description=readme(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Compilers",
        "Topic :: System :: Distributed Computing",
    ],
    keywords="data analytics cluster",
    url="https://bodo.ai",
    author="Bodo.ai",
    packages=find_packages(),
    package_data={
        "bodo.tests": [
            "caching_tests/*",
            "data/**",
            "bodosql_array_kernel_tests/*",
        ],
        # on Windows and MacOS we copy libssl and libcrypto DLLs to
        # bodo/libs to bundle them with our package and avoid
        # external dependency
        "bodo": ["pytest.ini", "libs/*.dll", "libs/*.dylib"],
    },
    # When doing `python setup.py develop`, setuptools will try to install whatever is
    # in `install_requires` after building, so we set it to empty (we don't want to
    # install mpi4py_mpich in development mode, and it will also break CI)
    # fsspec >= 2021.09 because it includes Arrow filesystem wrappers (useful for fs.glob() for example)
    install_requires=(
        []
        if develop_mode
        else [
            "numba==0.56.4",
            "pyarrow==11.0.0",
            "pandas>=1.3,<1.5",
            "numpy>=1.18,<1.24",
            "fsspec>=2021.09",
            "mpi4py_mpich==3.1.2",
        ]
    ),
    extras_require={"HDF5": ["h5py"]},
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=(
        [bodo_ext]
        + cythonize(
            _cython_ext_mods + builtin_exts,
            compiler_directives={"language_level": "3"},
            compile_time_env=dict(BODO_DEV_BUILD=develop_mode),
        )
    ),
)
