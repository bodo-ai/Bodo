# isort: skip_file
import sys
import os
import glob
import platform
from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize


import numpy.distutils.misc_util as np_misc

import versioneer

cwd = os.getcwd()
setup_py_dir_path = os.path.dirname(os.path.realpath(__file__))
# despite the name, this also works for directories
if not os.path.samefile(cwd, setup_py_dir_path):
    raise Exception(
        "setup.py should only be invoked if the current working directory is in the same directory as Setup.py.\nThis is to prevent having with conflicting .egg-info in the same directory when building Bodo's submodules."
    )

# Inject required options for extensions compiled against the Numpy
# C API (include dirs, library dirs etc.)
np_compile_args = np_misc.get_info("npymath")

is_win = platform.system() == "Windows"
is_mac = platform.system() == "Darwin"
is_m1_mac = is_mac and platform.machine() == "arm64"

development_mode = "develop" in sys.argv
if development_mode:
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
clean_mode = "clean" in sys.argv


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
    import pyarrow  # noqa
except ImportError:
    # currently, due to cross compilation, the import fails.
    # building the extension modules still works though.
    # TODO: resolve issue with pyarrow import
    _has_pyarrow = False or is_m1_mac
else:
    _has_pyarrow = True


try:
    import h5py  # noqa
except ImportError:
    # currently, due to cross compilation, the import fails.
    # building the extension modules still works though.
    # TODO: resolve issue with h5py import
    _has_h5py = False or is_m1_mac
    h5py_version = None
else:
    # NOTE: conda-forge does not have MPI-enabled hdf5 for Windows yet
    # TODO: make sure the available hdf5 library is MPI-enabled automatically
    _has_h5py = not is_win
    h5py_version = h5py.version.hdf5_version_tuple[1]

ind = [PREFIX_DIR + "/include"]
extra_hash_ind1 = ["bodo/libs/HashLibs/TSL/hopscotch-map"]
extra_hash_ind2 = ["bodo/libs/HashLibs/TSL/robin-map"]
extra_hash_ind3 = ["bodo/libs/HashLibs/TSL/sparse-map"]
extra_hash_ind = extra_hash_ind1 + extra_hash_ind2 + extra_hash_ind3
lid = [PREFIX_DIR + "/lib"]
# eca = ["-std=c++17", "-fsanitize=address"]
# ela = ["-std=c++17", "-fsanitize=address"]
if is_win:
    eca = ["/std:c++17", "/O2"]
    eca_c = ["/O2"]
    ela = ["/std:c++17"]
else:
    if is_mac:
        # Mac on CI can't support AVX2
        eca = ["-std=c++17", "-g0", "-O3"]
    else:
        # -march=haswell is used to enable AVX2 support (required by SIMD bloom
        # filter implementation)
        eca = ["-std=c++17", "-g0", "-O3", "-march=haswell"]
    eca_c = ["-g0", "-O3"]
    ela = ["-std=c++17"]

if development_mode:
    eca.append("-Werror")

MPI_LIBS = ["mpi"]
H5_CPP_FLAGS = []


if is_win:
    # use Microsoft MPI on Windows
    MPI_LIBS = ["msmpi"]
    if os.environ.get("BUILD_PIP", "") == "1":
        # building pip package, need to set additional include and library paths
        ind.append(os.path.join(os.path.dirname(pyarrow.__file__), "include"))
        ela.append(f"/LIBPATH:{os.path.join(os.path.dirname(pyarrow.__file__))}")
    # hdf5-parallel Windows build uses CMake which needs this flag
    H5_CPP_FLAGS = [("H5_BUILT_AS_DYNAMIC_LIB", None)]

hdf5_libs = MPI_LIBS + ["hdf5"]
io_libs = MPI_LIBS + ["arrow"]


ext_io = Extension(
    name="bodo.libs.hio",
    sources=[
        "bodo/io/_io.cpp",
        "bodo/io/_fs_io.cpp",
    ],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_distributed.h",
        "bodo/libs/_import_py.h",
        "bodo/io/_io.h",
        "bodo/io/_bodo_file_reader.h",
        "bodo/io/_fs_io.h",
    ],
    libraries=io_libs,
    include_dirs=ind + np_compile_args["include_dirs"],
    library_dirs=lid,
    define_macros=H5_CPP_FLAGS,
    extra_compile_args=eca,
    extra_link_args=ela,
    language="c++",
)


s3_reader_libraries = MPI_LIBS + ["arrow"]
ext_s3 = Extension(
    name="bodo.io.s3_reader",
    sources=["bodo/io/_s3_reader.cpp"],
    depends=["bodo/io/_bodo_file_reader.h"],
    libraries=s3_reader_libraries,
    include_dirs=ind + np_compile_args["include_dirs"],
    library_dirs=lid,
    define_macros=[],
    extra_compile_args=eca,
    extra_link_args=ela,
    language="c++",
)


fsspec_reader_libraries = MPI_LIBS + ["arrow", "arrow_python"]
ext_fsspec = Extension(
    name="bodo.io.fsspec_reader",
    sources=["bodo/io/_fsspec_reader.cpp"],
    depends=["bodo/io/_bodo_file_reader.h"],
    libraries=fsspec_reader_libraries,
    include_dirs=ind + np_compile_args["include_dirs"],
    library_dirs=lid,
    define_macros=[],
    # We cannot compile with -Werror yet because _fsspec_reader.cpp
    # depends on pyfs.cpp which generates a warning.
    extra_compile_args=[x for x in eca if x != "-Werror"],
    extra_link_args=ela,
    language="c++",
)


hdfs_reader_libraries = MPI_LIBS + ["arrow"]
ext_hdfs = Extension(
    name="bodo.io.hdfs_reader",
    sources=["bodo/io/_hdfs_reader.cpp"],
    depends=["bodo/io/_bodo_file_reader.h"],
    libraries=hdfs_reader_libraries,
    include_dirs=ind + np_compile_args["include_dirs"],
    library_dirs=lid,
    define_macros=[],
    extra_compile_args=eca,
    extra_link_args=ela,
    language="c++",
)

# Even though we upgraded to 1.12 which changes the API, these flags keep the
# 1.10 API. See the example: https://github.com/openmc-dev/openmc/pull/1533
# This was also verified by looking at the .h files.
# If we still have version 1.10 these are ignored.
extra_eca_hdf5 = [
    "-DH5Oget_info_by_name_vers=1",
    "-DH5Oget_info_vers=1",
    "-DH5O_info_t_vers=1",
]

ext_hdf5 = Extension(
    name="bodo.io._hdf5",
    sources=["bodo/io/_hdf5.cpp"],
    depends=[],
    libraries=hdf5_libs,
    include_dirs=ind,
    library_dirs=lid,
    define_macros=H5_CPP_FLAGS,
    extra_compile_args=eca + extra_eca_hdf5,
    extra_link_args=ela,
    language="c++",
)


# TODO Windows build fails because ssl.lib not found. Disabling licensing
# check on windows for now
dist_macros = []
dist_includes = []
dist_sources = []
dist_libs = []
if os.environ.get("CHECK_LICENSE_EXPIRED", None) == "1":
    dist_macros.append(("CHECK_LICENSE_EXPIRED", "1"))

if os.environ.get("CHECK_LICENSE_CORE_COUNT", None) == "1":
    dist_macros.append(("CHECK_LICENSE_CORE_COUNT", "1"))

if os.environ.get("CHECK_LICENSE_PLATFORM", None) == "1":
    assert os.environ.get("CHECK_LICENSE_EXPIRED", None) != "1"
    assert os.environ.get("CHECK_LICENSE_CORE_COUNT", None) != "1"
    dist_macros.append(("CHECK_LICENSE_PLATFORM", "1"))
    dist_includes += ["bodo/libs/gason"]
    dist_sources += ["bodo/libs/gason/gason.cpp"]
    dist_libs += ["curl"]

if (
    os.environ.get("CHECK_LICENSE_EXPIRED", None) == "1"
    or os.environ.get("CHECK_LICENSE_CORE_COUNT", None) == "1"
):
    if is_win:
        dist_libs += ["libssl", "libcrypto"]
    else:
        dist_libs += ["ssl", "crypto"]


ext_hdist = Extension(
    name="bodo.libs.hdist",
    sources=["bodo/libs/_distributed.cpp"] + dist_sources,
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_distributed.h",
    ],
    libraries=MPI_LIBS + dist_libs,
    define_macros=dist_macros,
    extra_compile_args=eca,
    extra_link_args=ela,
    include_dirs=ind + dist_includes,
    library_dirs=lid,
)


ext_str = Extension(
    name="bodo.libs.hstr_ext",
    sources=[
        "bodo/libs/_str_ext.cpp",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_bodo_to_arrow.cpp",
        "bodo/libs/_datetime_utils.cpp",
    ],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_bodo_to_arrow.h",
        "bodo/libs/_datetime_utils.h",
    ],
    libraries=MPI_LIBS + np_compile_args["libraries"] + ["arrow", "arrow_python"],
    define_macros=np_compile_args["define_macros"],
    extra_compile_args=eca,
    extra_link_args=ela,
    include_dirs=np_compile_args["include_dirs"] + ind,
    library_dirs=np_compile_args["library_dirs"] + lid,
)


# TODO: make Arrow optional in decimal extension similar to parquet extension?
ext_decimal = Extension(
    name="bodo.libs.decimal_ext",
    sources=["bodo/libs/_decimal_ext.cpp", "bodo/libs/_bodo_common.cpp"],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_bodo_common.cpp",
    ],
    libraries=MPI_LIBS + np_compile_args["libraries"] + ["arrow"],
    define_macros=np_compile_args["define_macros"],
    extra_compile_args=eca,
    extra_link_args=ela,
    include_dirs=np_compile_args["include_dirs"] + ind,
    library_dirs=np_compile_args["library_dirs"] + lid,
)

ext_arr = Extension(
    name="bodo.libs.array_ext",
    sources=[
        "bodo/libs/_array.cpp",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_decimal_ext.cpp",
        "bodo/libs/_array_utils.cpp",
        "bodo/libs/_array_hash.cpp",
        "bodo/libs/_shuffle.cpp",
        "bodo/libs/_join.cpp",
        "bodo/libs/_join_hashing.cpp",
        "bodo/libs/_groupby.cpp",
        "bodo/libs/_array_operations.cpp",
        "bodo/libs/_murmurhash3.cpp",
    ],
    depends=[
        "bodo/libs/_array_utils.h",
        "bodo/libs/_decimal_ext.h",
        "bodo/libs/_decimal_ext.cpp",
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_murmurhash3.h",
        "bodo/libs/_distributed.h",
        "bodo/libs/_join.h",
        "bodo/libs/_join_hashing.h",
        "bodo/libs/hyperloglog.hpp",
        "bodo/libs/simd-block-fixed.h",
    ],
    libraries=MPI_LIBS + np_compile_args["libraries"] + ["arrow", "arrow_python"],
    # -fno-strict-aliasing required by bloom filter implementation (see comment
    # in simd-block-fixed-fpp.h about violating strict aliasing rules)
    extra_compile_args=eca + ["-fno-strict-aliasing"],
    extra_link_args=ela,
    define_macros=np_compile_args["define_macros"],
    include_dirs=np_compile_args["include_dirs"] + ind + extra_hash_ind,
    library_dirs=np_compile_args["library_dirs"] + lid,
)


ext_dt = Extension(
    name="bodo.libs.hdatetime_ext",
    sources=[
        "bodo/libs/_datetime_ext.cpp",
        "bodo/libs/_datetime_utils.cpp",
        "bodo/libs/_bodo_common.cpp",
    ],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_datetime_utils.h",
    ],
    libraries=MPI_LIBS + np_compile_args["libraries"] + ["arrow"],
    define_macros=np_compile_args["define_macros"],
    extra_compile_args=eca,
    extra_link_args=ela,
    include_dirs=np_compile_args["include_dirs"] + ind,
    library_dirs=np_compile_args["library_dirs"] + lid,
    language="c++",
)


ext_quantile = Extension(
    name="bodo.libs.quantile_alg",
    sources=[
        "bodo/libs/_quantile_alg.cpp",
        "bodo/libs/_decimal_ext.cpp",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_array_utils.cpp",
    ],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_decimal_ext.cpp",
        "bodo/libs/_decimal_ext.h",
    ],
    libraries=MPI_LIBS + np_compile_args["libraries"] + ["arrow"],
    extra_compile_args=eca,
    extra_link_args=ela,
    include_dirs=np_compile_args["include_dirs"] + ind + extra_hash_ind,
    library_dirs=np_compile_args["library_dirs"] + lid,
)


# pq_libs = MPI_LIBS + ['arrow', 'parquet']
pq_libs = MPI_LIBS.copy()

csv_libs = pq_libs + ["arrow"]
pq_libs += ["arrow", "parquet"]

ext_csv = Extension(
    name="bodo.io.csv_cpp",
    sources=[
        "bodo/io/_io.cpp",
        "bodo/io/_fs_io.cpp",
        "bodo/io/_csv_json_reader.cpp",
        "bodo/io/_csv_json_writer.cpp",
    ],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_distributed.h",
        "bodo/libs/_import_py.h",
        "bodo/io/_io.h",
        "bodo/io/_fs_io.h",
        "bodo/io/_csv_json_reader.h",
        "bodo/io/_bodo_file_reader.h",
    ],
    libraries=csv_libs,
    include_dirs=["."] + ind,
    define_macros=[],
    extra_compile_args=eca,
    extra_link_args=ela,
    library_dirs=lid,
)

ext_json = Extension(
    name="bodo.io.json_cpp",
    sources=[
        "bodo/io/_io.cpp",
        "bodo/io/_fs_io.cpp",
        "bodo/io/_csv_json_reader.cpp",
        "bodo/io/_csv_json_writer.cpp",
    ],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_distributed.h",
        "bodo/libs/_import_py.h",
        "bodo/io/_fs_io.h",
        "bodo/io/_csv_json_reader.h",
        "bodo/io/_bodo_file_reader.h",
    ],
    libraries=csv_libs,
    include_dirs=["."] + ind,
    define_macros=[],
    extra_compile_args=eca,
    extra_link_args=ela,
    library_dirs=lid,
)

ext_parquet = Extension(
    name="bodo.io.arrow_cpp",
    sources=[
        "bodo/io/_fs_io.cpp",
        "bodo/io/arrow.cpp",
        "bodo/io/arrow_reader.cpp",
        "bodo/io/parquet_reader.cpp",
        "bodo/io/iceberg_parquet_reader.cpp",
        "bodo/io/snowflake_reader.cpp",
        "bodo/io/parquet_write.cpp",
        "bodo/io/iceberg_parquet_write.cpp",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_bodo_to_arrow.cpp",
        "bodo/libs/_decimal_ext.cpp",
        "bodo/libs/_array_utils.cpp",
        "bodo/libs/_array_hash.cpp",
        "bodo/libs/_murmurhash3.cpp",
        "bodo/io/_fsspec_reader.cpp",
        "bodo/io/_hdfs_reader.cpp",
        "bodo/io/_s3_reader.cpp",
        "bodo/libs/iceberg_transforms.cpp",
        "bodo/libs/_array_operations.cpp",
        "bodo/libs/_datetime_utils.cpp",
        "bodo/libs/_shuffle.cpp",
    ],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_bodo_to_arrow.h",
        "bodo/libs/_bodo_to_arrow.cpp",
        "bodo/libs/_decimal_ext.h",
        "bodo/io/_fs_io.h",
        "bodo/io/arrow_reader.h",
        "bodo/io/parquet_reader.h",
        "bodo/io/parquet_write.h",
        "bodo/libs/_murmurhash3.h",
        "bodo/libs/iceberg_transforms.h",
        "bodo/libs/_array_operations.h",
        "bodo/libs/_datetime_utils.h",
        "bodo/libs/_shuffle.h",
    ],
    libraries=pq_libs + np_compile_args["libraries"] + ["arrow", "arrow_python"],
    include_dirs=["."] + np_compile_args["include_dirs"] + ind + extra_hash_ind,
    define_macros=[],
    # We cannot compile with -Werror yet because _fsspec_reader.cpp
    # depends on pyfs.cpp which generates a warning.
    extra_compile_args=[x for x in eca if x != "-Werror"],
    extra_link_args=ela,
    library_dirs=np_compile_args["library_dirs"] + lid,
)

# Cython files that are part of the code base that aren't just renamed .py
# files during build
pyx_builtins = []

ext_pyfs = Extension(
    name="bodo.io.pyfs",
    sources=[
        "bodo/io/pyfs.pyx",
    ],
    include_dirs=np_compile_args["include_dirs"] + ind,
    define_macros=[],
    # We cannot compile with -Werror yet because pyfs.cpp
    # generates serveral warnings.
    extra_compile_args=[x for x in eca if x != "-Werror"],
    extra_link_args=ela,
    library_dirs=lid,
)
# the bodo/io/pyfs.pyx file is always part of Bodo (not generated during build)
pyx_builtins += [os.path.join("bodo", "io", "pyfs.pyx")]


ext_hdfs_pyarrow = Extension(
    name="bodo.io._hdfs",
    sources=[
        "bodo/io/_hdfs.pyx",
    ],
    libraries=["arrow", "arrow_python"],
    include_dirs=np_compile_args["include_dirs"] + ind,
    define_macros=[],
    # We cannot compile with -Werror yet because hdfs.cpp
    # generates serveral warnings.
    extra_compile_args=[x for x in eca if x != "-Werror"],
    extra_link_args=ela,
    library_dirs=lid,
    language="c++",
)
# the bodo/io/_hdfs.pyx file is always part of Bodo (not generated during build)
pyx_builtins += [os.path.join("bodo", "io", "_hdfs.pyx")]


ext_tracing = Extension(
    name="bodo.utils.tracing",
    sources=[
        "bodo/utils/tracing.pyx",
    ],
    include_dirs=ind,
    libraries=MPI_LIBS,
    define_macros=[],
    extra_compile_args=eca_c,
    extra_link_args=ela,
    library_dirs=lid,
)
pyx_builtins += [os.path.join("bodo", "utils", "tracing.pyx")]


_ext_mods = [
    ext_hdist,
    ext_str,
    ext_decimal,
    ext_quantile,
    ext_dt,
    ext_io,
    ext_arr,
    ext_s3,
    ext_fsspec,
    ext_hdfs,
]


if clean_mode:
    assert not development_mode
    _cython_ext_mods = [
        f for f in glob.glob("bodo/**/*.pyx", recursive=True) if f not in pyx_builtins
    ]
elif development_mode:
    _cython_ext_mods = []
    # make sure there are no .pyx files in development mode
    pyxs = [
        f for f in glob.glob("bodo/**/*.pyx", recursive=True) if f not in pyx_builtins
    ]
    assert len(pyxs) == 0
else:
    import subprocess

    # rename select files to .pyx for cythonizing
    subprocess.run([sys.executable, "rename_to_pyx.py"])
    _cython_ext_mods = [
        f for f in glob.glob("bodo/**/*.pyx", recursive=True) if f not in pyx_builtins
    ]


if _has_h5py:
    _ext_mods.append(ext_hdf5)
if _has_pyarrow:
    _ext_mods.append(ext_parquet)
    _ext_mods.append(ext_csv)
    _ext_mods.append(ext_json)


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
            "data/*",
            "data/*/*",
            "data/*/*/*",
            "data/*/*/*/*",
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
    install_requires=[]
    if development_mode
    else [
        "numba==0.55.2",
        "pyarrow==8.0.0",
        "pandas>=1.3.*,<1.5",
        "numpy>=1.18,<1.22",
        "fsspec>=2021.09",
        "mpi4py_mpich==3.1.2",
    ],
    extras_require={"HDF5": ["h5py"], "Parquet": ["pyarrow"]},
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=_ext_mods
    + cythonize(
        _cython_ext_mods + [ext_pyfs, ext_hdfs_pyarrow, ext_tracing],
        compiler_directives={"language_level": "3"},
        compile_time_env=dict(BODO_DEV_BUILD=development_mode),
    ),
)
