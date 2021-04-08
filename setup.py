# isort: skip_file
import sys
import os
import glob
import platform
from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize

# Note we don't import Numpy at the toplevel, since setup.py
# should be able to run without Numpy for pip to discover the
# build dependencies
import numpy.distutils.misc_util as np_misc

import versioneer

# Inject required options for extensions compiled against the Numpy
# C API (include dirs, library dirs etc.)
np_compile_args = np_misc.get_info("npymath")

is_win = platform.system() == "Windows"
development_mode = "develop" in sys.argv
clean_mode = "clean" in sys.argv


def readme():
    with open("README.md") as f:
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
    _has_pyarrow = False
else:
    _has_pyarrow = True


try:
    import h5py  # noqa
except ImportError:
    _has_h5py = False
else:
    # NOTE: conda-forge does not have MPI-enabled hdf5 for Windows yet
    # TODO: make sure the available hdf5 library is MPI-enabled automatically
    _has_h5py = not is_win


ind = [PREFIX_DIR + "/include"]
extra_hash_ind1 = ["bodo/libs/HashLibs/TSL/hopscotch-map"]
extra_hash_ind2 = ["bodo/libs/HashLibs/TSL/robin-map"]
extra_hash_ind3 = ["bodo/libs/HashLibs/TSL/sparse-map"]
extra_hash_ind = extra_hash_ind1 + extra_hash_ind2 + extra_hash_ind3
lid = [PREFIX_DIR + "/lib"]
# eca = ["-std=c++11", "-fsanitize=address"]
# ela = ["-std=c++11", "-fsanitize=address"]
if is_win:
    eca = ["/std=c++latest", "/O2"]
else:
    eca = ["-std=c++11", "-g0", "-O3"]
ela = ["-std=c++11"]

MPI_LIBS = ["mpi"]
if "setup_centos7" in os.environ:
    MPI_LIBS = ["mpich"]
H5_CPP_FLAGS = []

use_impi = False
if use_impi:
    MPI_ROOT = os.environ["I_MPI_ROOT"]
    MPI_INC = MPI_ROOT + "/include64/"
    MPI_LIBDIR = MPI_ROOT + "/lib64/"
    MPI_LIBS = ["mpifort", "mpi", "mpigi"]
    ind = [PREFIX_DIR + "/include", MPI_INC]
    lid = [PREFIX_DIR + "/lib", MPI_LIBDIR]

if is_win:
    # use Intel MPI on Windows
    MPI_LIBS = ["impi"]
    # hdf5-parallel Windows build uses CMake which needs this flag
    H5_CPP_FLAGS = [("H5_BUILT_AS_DYNAMIC_LIB", None)]

hdf5_libs = MPI_LIBS + ["hdf5"]
io_libs = MPI_LIBS + ["arrow"]
if not is_win:
    io_libs += ["boost_filesystem", "boost_system"]


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
if "setup_centos7" in os.environ:
    s3_reader_libraries += ["boost_system"]
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


gcs_reader_libraries = MPI_LIBS + ["arrow", "arrow_python"]
if "setup_centos7" in os.environ:
    gcs_reader_libraries += ["boost_system"]
ext_gcs = Extension(
    name="bodo.io.gcs_reader",
    sources=["bodo/io/_gcs_reader.cpp"],
    depends=["bodo/io/_bodo_file_reader.h"],
    libraries=gcs_reader_libraries,
    include_dirs=ind + np_compile_args["include_dirs"],
    library_dirs=lid,
    define_macros=[],
    extra_compile_args=eca,
    extra_link_args=ela,
    language="c++",
)


hdfs_reader_libraries = MPI_LIBS + ["arrow"]
if "setup_centos7" in os.environ:
    hdfs_reader_libraries += ["boost_system"]
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

ext_hdf5 = Extension(
    name="bodo.io._hdf5",
    sources=["bodo/io/_hdf5.cpp"],
    depends=[],
    libraries=hdf5_libs,
    include_dirs=ind,
    library_dirs=lid,
    define_macros=H5_CPP_FLAGS,
    extra_compile_args=eca,
    extra_link_args=ela,
    language="c++",
)


# TODO Windows build fails because ssl.lib not found. Disabling licensing
# check on windows for now
dist_macros = []
dist_includes = []
dist_sources = []
dist_libs = []
if os.environ.get("CHECK_LICENSE_EXPIRED", None) == "1" and not is_win:
    dist_macros.append(("CHECK_LICENSE_EXPIRED", "1"))

if os.environ.get("CHECK_LICENSE_CORE_COUNT", None) == "1" and not is_win:
    dist_macros.append(("CHECK_LICENSE_CORE_COUNT", "1"))

if os.environ.get("CHECK_LICENSE_PLATFORM_AWS", None) == "1":
    assert os.environ.get("CHECK_LICENSE_EXPIRED", None) != "1"
    assert os.environ.get("CHECK_LICENSE_CORE_COUNT", None) != "1"
    dist_macros.append(("CHECK_LICENSE_PLATFORM_AWS", "1"))
    dist_includes += ["bodo/libs/gason"]
    dist_sources += ["bodo/libs/gason/gason.cpp"]
    dist_libs += ["curl"]

if not is_win and (
    os.environ.get("CHECK_LICENSE_EXPIRED", None) == "1"
    or os.environ.get("CHECK_LICENSE_CORE_COUNT", None) == "1"
):
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
    sources=["bodo/libs/_str_ext.cpp", "bodo/libs/_bodo_common.cpp"],
    libraries=MPI_LIBS + np_compile_args["libraries"] + ["arrow"],
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
    depends=["bodo/libs/_bodo_common.h", "bodo/libs/_bodo_common.cpp"],
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
        "bodo/libs/_groupby.cpp",
        "bodo/libs/_array_operations.cpp",
        "bodo/libs/_murmurhash3.cpp",
    ],
    depends=[
        "bodo/libs/_decimal_ext.h",
        "bodo/libs/_decimal_ext.cpp",
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_murmurhash3.h",
        "bodo/libs/_distributed.h",
    ],
    libraries=MPI_LIBS + np_compile_args["libraries"] + ["arrow"],
    extra_compile_args=eca,
    extra_link_args=ela,
    define_macros=np_compile_args["define_macros"],
    include_dirs=np_compile_args["include_dirs"] + ind + extra_hash_ind,
    library_dirs=np_compile_args["library_dirs"] + lid,
)


ext_dt = Extension(
    name="bodo.libs.hdatetime_ext",
    sources=["bodo/libs/_datetime_ext.cpp", "bodo/libs/_bodo_common.cpp"],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_bodo_common.cpp",
    ],
    libraries=MPI_LIBS + np_compile_args["libraries"] + ["arrow"],
    define_macros=np_compile_args["define_macros"],
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"],
    include_dirs=np_compile_args["include_dirs"] + ind,
    library_dirs=np_compile_args["library_dirs"],
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


# pq_libs = MPI_LIBS + ['boost_filesystem', 'arrow', 'parquet']
pq_libs = MPI_LIBS.copy()

# Windows MSVC can't have boost library names on command line
# auto-link magic of boost should be used
if not is_win:
    pq_libs += ["boost_filesystem"]

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
    name="bodo.io.parquet_cpp",
    sources=[
        "bodo/io/_fs_io.cpp",
        "bodo/io/_parquet.cpp",
        "bodo/io/_parquet_reader.cpp",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_decimal_ext.cpp",
        "bodo/libs/_array_utils.cpp",
        "bodo/libs/_array_hash.cpp",
        "bodo/libs/_murmurhash3.cpp",
    ],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_bodo_common.cpp",
        "bodo/io/_fs_io.h",
        "bodo/io/_parquet_reader.h",
        "bodo/libs/_murmurhash3.h",
    ],
    libraries=pq_libs + np_compile_args["libraries"],
    include_dirs=["."] + np_compile_args["include_dirs"] + ind + extra_hash_ind,
    define_macros=[],
    extra_compile_args=eca,
    extra_link_args=ela,
    library_dirs=np_compile_args["library_dirs"] + lid,
)


ext_pyfs = Extension(
    name="bodo.io.pyfs",
    sources=[
        "bodo/io/pyfs.pyx",
    ],
    include_dirs=np_compile_args["include_dirs"] + ind,
    define_macros=[],
    extra_compile_args=eca,
    extra_link_args=ela,
    library_dirs=lid,
)

_ext_mods = [
    ext_hdist,
    ext_str,
    ext_decimal,
    ext_quantile,
    ext_dt,
    ext_io,
    ext_arr,
    ext_s3,
    ext_gcs,
    ext_hdfs,
]


# the bodo/io/pyfs.pyx file is always part of Bodo (not generated during build)
pyfs_pyx_fpath = os.path.join("bodo", "io", "pyfs.pyx")
if clean_mode:
    assert not development_mode
    _cython_ext_mods = [
        f for f in glob.glob("bodo/**/*.pyx", recursive=True) if f != pyfs_pyx_fpath
    ]
elif development_mode:
    _cython_ext_mods = []
    # make sure there are no .pyx files in development mode
    pyxs = [
        f for f in glob.glob("bodo/**/*.pyx", recursive=True) if f != pyfs_pyx_fpath
    ]
    assert len(pyxs) == 0
else:
    import subprocess

    # rename select files to .pyx for cythonizing
    subprocess.run([sys.executable, "rename_to_pyx.py"])
    _cython_ext_mods = [
        f for f in glob.glob("bodo/**/*.pyx", recursive=True) if f != pyfs_pyx_fpath
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
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Compilers",
        "Topic :: System :: Distributed Computing",
    ],
    keywords="data analytics cluster",
    url="https://bodo.ai",
    author="Bodo.ai",
    packages=find_packages(),
    package_data={
        "bodo.tests": [
            "data/*",
            "data/*/*",
        ],
        "bodo": ["pytest.ini"],
    },
    install_requires=["numba"],
    extras_require={"HDF5": ["h5py"], "Parquet": ["pyarrow"]},
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=_ext_mods
    + cythonize(
        _cython_ext_mods + [ext_pyfs], compiler_directives={"language_level": "3"}
    ),
)
