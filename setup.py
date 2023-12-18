# isort: skip_file
import sys
import os
import glob
import platform
import typing as pt
import shutil

from setuptools import Extension, find_packages, setup
from setuptools.command.install import install
from Cython.Build import cythonize
import numpy as np
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
    use_ccache = False
    if "--no-ccache" not in sys.argv:
        if shutil.which("ccache"):
            use_ccache = True
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
                "ccache not found. Make sure you're using the (latest) conda-lock environment.\n"
                "If this is expected, use --no-ccache to build without ccache."
            )
            exit(1)
    else:
        sys.argv.remove("--no-ccache")

    if "--no-sccache" not in sys.argv:
        # sccache allows us to save cache between different machines
        if shutil.which("sccache"):
            if use_ccache:
                # When using ccache & sccache
                os.environ["CCACHE_PREFIX"] = "sccache"
            elif not use_ccache and "sccache" not in os.environ["CC"]:
                # Without ccache, we only need to modify the environment variables
                # CC and CXX during execution of `python setup.py develop`.
                # With newer compiler versions on Linux, we have to set both CC
                # and CXX variables for sccache to work, and have to set CC to
                # the C++ compiler to avoid compile-time and dynamic linking
                # errors
                os.environ["CC"] = f"sccache {os.environ['CXX']}"
                os.environ["CXX"] = f"sccache {os.environ['CXX']}"
        else:
            print(
                "sccache not found. Make sure you're using the (latest) conda-lock environment.\n"
                "If this is expected, use --no-sccache to build without sccache."
            )
            exit(1)
    else:
        sys.argv.remove("--no-sccache")


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
        PREFIX_DIR += "\\Library"


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


# Enable Tests by default for development
is_testing = develop_mode
if "--no-test" in sys.argv:
    is_testing = False
    sys.argv.remove("--no-test")

if "--profile" in sys.argv:
    profile_flag = ["-pg"]
    if debug_flag == "-g0":
        debug_flag = "-g"
    sys.argv.remove("--profile")

sanitizer_flags = []
if "--address-sanitizer" in sys.argv:
    sys.argv.remove("--address-sanitizer")
    sanitizer_flags.append("address")
    opt_flag = "-O1"

if "--undefined-sanitizer" in sys.argv:
    sys.argv.remove("--undefined-sanitizer")
    sanitizer_flags.append("undefined")
    opt_flag = "-O1"

if sanitizer_flags:
    comma_args = ",".join(sanitizer_flags)
    sanitizer_flag = [f"-fsanitize={comma_args}"]
else:
    sanitizer_flag = []


if is_win:
    eca = ["/std:c++20", "/O2"]
    eca_c = ["/O2"]
    ela = ["/std:c++20"]
else:
    if is_mac:
        # Mac on CI can't support AVX2
        eca = ["-std=c++20", debug_flag, opt_flag] + sanitizer_flag
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
            + sanitizer_flag
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
eca.append("-Wno-return-type-c-linkage")
eca.append("-Wno-macro-redefined")

# Force colored output, even when the the output is not a terminal
if "BODO_FORCE_COLORED_BUILD" in os.environ:
    eca.append("-fcolor-diagnostics")

# Use a single C-extension for all of Bodo
# Copying ind, lid, eca, and ela to avoid aliasing, as we continue to append
ext_metadata: dict[str, pt.Any] = dict(
    name="bodo.ext",
    sources=[],
    depends=[],
    include_dirs=list(ind),
    define_macros=[
        # Required when using boost::stacktrace for debugging
        ("BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED", "1"),
    ],
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

ext_metadata["libraries"] += mpi_libs + [
    "hdf5",
    "arrow",
    "arrow_python",
    "parquet",
    "fftw3",
    "fftw3f",
    "fftw3_mpi",
    "fftw3f_mpi",
    "m",
    "fmt",
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
    "bodo/io/json_col_parser.cpp",
    "bodo/libs/_array.cpp",
    "bodo/libs/_array_hash.cpp",
    "bodo/libs/_array_operations.cpp",
    "bodo/libs/_array_utils.cpp",
    "bodo/libs/_base64.cpp",
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
    "bodo/libs/_memory_budget.cpp",
    "bodo/libs/_memory_budget_pymod.cpp",
    "bodo/libs/_murmurhash3.cpp",
    "bodo/libs/_quantile_alg.cpp",
    "bodo/libs/_lateral.cpp",
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
    "bodo/libs/_stream_shuffle.cpp",
    "bodo/libs/_storage_manager.cpp",
    "bodo/libs/_utils.cpp",
    "bodo/libs/_fft.cpp",
]
ext_metadata["depends"] += [
    "bodo/io/_bodo_file_reader.h",
    "bodo/io/_csv_json_reader.h",
    "bodo/io/_fs_io.h",
    "bodo/io/_io.h",
    "bodo/io/arrow_reader.h",
    "bodo/io/parquet_reader.h",
    "bodo/io/parquet_write.h",
    "bodo/io/json_col_parser.h",
    "bodo/libs/_array_hash.h",
    "bodo/libs/_array_operations.h",
    "bodo/libs/_array_utils.h",
    "bodo/libs/_base64.h",
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
    "bodo/libs/_lateral.h",
    "bodo/libs/_crypto_funcs.h",
    "bodo/libs/_meminfo.h",
    "bodo/libs/_memory.h",
    "bodo/libs/_memory_budget.h",
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
    "bodo/libs/_stream_shuffle.h",
    "bodo/libs/_storage_manager.h",
    "bodo/libs/_utils.h",
    "bodo/libs/_fft.h",
]

if is_testing:
    ext_metadata["sources"].extend(
        [
            "bodo/tests/test_framework.cpp",
            "bodo/tests/test_example.cpp",
            "bodo/tests/test_schema.cpp",
            "bodo/tests/test_pinnable.cpp",
            "bodo/tests/test_dict_builder.cpp",
            "bodo/tests/test_groupby_and_window.cpp",
            "bodo/tests/test_memory_budget.cpp",
            "bodo/tests/test_table_builder.cpp",
            "bodo/tests/test_table_generator.cpp",
            "bodo/tests/test_test_framework.cpp",
            "bodo/tests/test_json_col_reader.cpp",
            "bodo/tests/test_nested_array.cpp",
            "bodo/tests/test_stream_shuffle.cpp",
        ]
    )
    ext_metadata["define_macros"].append(("IS_TESTING", "1"))


# Inject required options for extensions compiled against the Numpy
# C API (include dirs, library dirs etc.)
# TODO(ehsan): avoid top-level np include if fails for pip
# See: https://github.com/numba/numba/blob/04e81073b2c1e3ff4afa1da8513738e5e136775b/setup.py#L138
np_compile_args = {
    "include_dirs": [np.get_include()],
    "library_dirs": [os.path.join(np.get_include(), "..", "lib")],
    "define_macros": [],
    "libraries": ["npymath"],
}
ext_metadata["libraries"] += np_compile_args["libraries"]
# Include all Numpy headers as system includes (prevents warnings from being
# emitted)
for dir_ in np_compile_args["include_dirs"]:
    ext_metadata["extra_compile_args"].append(f"-isystem{dir_}")
ext_metadata["library_dirs"] += np_compile_args["library_dirs"]
ext_metadata["define_macros"] += np_compile_args["define_macros"]

# Inject required options for extensions compiled against
# PyArrow and Arrow
pa_compile_args = {
    "include_dirs": [pyarrow.get_include()],
    "library_dirs": pyarrow.get_library_dirs(),
}

# Include all pyarrow headers as system includes (prevents warnings from
# being emitted)
for dir_ in pa_compile_args["include_dirs"]:
    ext_metadata["extra_compile_args"].append(f"-isystem{dir_}")
ext_metadata["library_dirs"].extend(pa_compile_args["library_dirs"])

# Compile Bodo extension
bodo_ext = Extension(**ext_metadata)


# Build extensions for Cython files that are part of the code base, and aren't
# just renamed .py files during build
# These .pyx files are always part of Bodo (not generated during build)
builtin_exts = []
pyx_builtins = []

ext_pyfs = Extension(
    name="bodo.io.pyfs",
    sources=["bodo/io/pyfs.pyx"],
    include_dirs=np_compile_args["include_dirs"]
    + ind
    + pa_compile_args["include_dirs"],
    define_macros=[],
    library_dirs=lid + pa_compile_args["library_dirs"],
    libraries=["arrow", "arrow_python"],
    extra_compile_args=eca + ["-Wno-unused-variable"],
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
    extra_compile_args=eca + ["-Wno-unused-variable"],
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
        "bodo/libs/_utils.cpp",
        "bodo/libs/_storage_manager.cpp",
        "bodo/libs/_memory_budget.cpp",
    ],
    depends=[
        "bodo/libs/_storage_manager.h",
        "bodo/libs/_utils.h",
        "bodo/libs/_memory_budget.h",
    ],
    include_dirs=np_compile_args["include_dirs"]
    + ind
    + pa_compile_args["include_dirs"],
    define_macros=[
        # Required when using boost::stacktrace for debugging
        ("BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED", "1"),
    ],
    library_dirs=lid + pa_compile_args["library_dirs"],
    libraries=["arrow", "arrow_python", "fmt"] + mpi_libs,
    # Cannot compile with -Werror yet because memory.cpp
    # generated multiple unused variable warnings
    extra_compile_args=eca + ["-Wno-unused-variable"],
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
    _cython_ext_mods = glob.glob("bodo/transforms/*.py", recursive=True)
    _cython_ext_mods.remove(os.path.join("bodo", "transforms", "__init__.py"))
else:
    _cython_ext_mods = []


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        # Installation has completed, run post-install steps
        if install_mode:
            # find the directory bodo was installed to
            for p in sys.path:
                if os.path.isdir(p) and "bodo" in os.listdir(p):
                    # For every cythonized file, attempt to delete the source
                    # python file. This is because cythonized files cannot be
                    # obfuscated before cythonization, and aren't needed after
                    # compilation. If we leave them as is, we risk leaking IP.
                    for f in _cython_ext_mods:
                        cythonized_install_path = os.path.join(p, f)
                        if os.path.exists(cythonized_install_path):
                            os.remove(cythonized_install_path)


cmdclass = versioneer.get_cmdclass()
assert "install" not in cmdclass
cmdclass["install"] = PostInstallCommand

setup(
    name="bodo",
    version=versioneer.get_version(),
    description="The Python Supercomputing Analytics Platform",
    long_description=readme(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
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
            "data/**/*",
            "data/**/**/*",
            "data/**/**/**/*",
            "data/**/**/**/**/*",
            "data/**/**/**/**/**/*",
            "data/**/**/**/**/**/**/*",
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
            "numba==0.57.1",
            "pyarrow==13.0.0",
            "pandas>=2,<2.1",
            "numpy>=1.23,<1.25",
            "fsspec>=2021.09",
            "mpi4py_mpich==3.1.2",
        ]
    ),
    extras_require={"HDF5": ["h5py"]},
    cmdclass=cmdclass,
    ext_modules=(
        [bodo_ext]
        + cythonize(
            _cython_ext_mods + builtin_exts,
            compiler_directives={"language_level": "3"},
            compile_time_env={"BODO_DEV_BUILD": develop_mode},
        )
    ),
)
