from setuptools import setup, Extension, find_packages
import platform, os, time

# Note we don't import Numpy at the toplevel, since setup.py
# should be able to run without Numpy for pip to discover the
# build dependencies
import numpy.distutils.misc_util as np_misc

# import copy
import versioneer

# Inject required options for extensions compiled against the Numpy
# C API (include dirs, library dirs etc.)
np_compile_args = np_misc.get_info("npymath")

is_win = platform.system() == "Windows"


def readme():
    with open("README.md") as f:
        return f.read()


_has_h5py = False
HDF5_DIR = ""

if "HDF5_DIR" in os.environ:
    _has_h5py = True
    HDF5_DIR = os.environ["HDF5_DIR"]

# PANDAS_DIR = ""
# if 'PANDAS_DIR' in os.environ:
#    PANDAS_DIR = os.environ['PANDAS_DIR']

# package environment variable is PREFIX during build time
if "CONDA_BUILD" in os.environ:
    PREFIX_DIR = os.environ["PREFIX"]
else:
    PREFIX_DIR = os.environ["CONDA_PREFIX"]
    # C libraries are in \Library on Windows
    if is_win:
        PREFIX_DIR += "\Library"


try:
    import pyarrow
except ImportError:
    _has_pyarrow = False
else:
    _has_pyarrow = True


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
io_libs = MPI_LIBS
if not is_win:
    io_libs += ["boost_filesystem", "boost_system"]


ext_io = Extension(
    name="bodo.libs.hio",
    sources=["bodo/io/_io.cpp", "bodo/io/_csv.cpp"],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_distributed.h",
        "bodo/libs/_import_py.h",
        "bodo/io/_csv.h",
        "bodo/io/_bodo_csv_file_reader.h",
    ],
    libraries=io_libs,
    include_dirs=ind + np_compile_args["include_dirs"],
    library_dirs=lid,
    define_macros=H5_CPP_FLAGS,
    extra_compile_args=eca,
    extra_link_args=ela,
    language="c++",
)


ext_s3 = Extension(
    name="bodo.io.s3_reader",
    sources=["bodo/io/_s3_reader.cpp"],
    depends=["bodo/io/_bodo_csv_file_reader.h"],
    libraries=["arrow"],
    include_dirs=ind + np_compile_args["include_dirs"],
    library_dirs=lid,
    define_macros=[],
    extra_compile_args=eca,
    extra_link_args=ela,
    language="c++",
)

ext_hdfs = Extension(
    name="bodo.io.hdfs_reader",
    sources=["bodo/io/_hdfs_reader.cpp"],
    depends=["bodo/io/_bodo_csv_file_reader.h"],
    libraries=["arrow"],
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
    include_dirs=[HDF5_DIR + "/include"] + ind,
    library_dirs=[HDF5_DIR + "/lib"] + lid,
    define_macros=H5_CPP_FLAGS,
    extra_compile_args=eca,
    extra_link_args=ela,
    language="c++",
)


dist_macros = []
if "TRIAL_PERIOD" in os.environ and os.environ["TRIAL_PERIOD"] != "":
    trial_period = os.environ["TRIAL_PERIOD"]
    trial_start = str(int(time.time()))
    dist_macros.append(("TRIAL_PERIOD", trial_period))
    dist_macros.append(("TRIAL_START", trial_start))

if "MAX_CORE_COUNT" in os.environ and os.environ["MAX_CORE_COUNT"] != "":
    max_core_count = os.environ["MAX_CORE_COUNT"]
    dist_macros.append(("MAX_CORE_COUNT", max_core_count))

ext_hdist = Extension(
    name="bodo.libs.hdist",
    sources=["bodo/libs/_distributed.cpp"],
    depends=[
        "bodo/libs/_bodo_common.h", 
        "bodo/libs/_distributed.h",],
    libraries=MPI_LIBS,
    define_macros=dist_macros,
    extra_compile_args=eca,
    extra_link_args=ela,
    include_dirs=ind,
    library_dirs=lid,
)


ext_dict = Extension(
    name="bodo.libs.hdict_ext",
    sources=["bodo/libs/_dict_ext.cpp"],
    extra_compile_args=eca,
    extra_link_args=ela,
    include_dirs=ind,
    library_dirs=lid,
)


ext_str = Extension(
    name="bodo.libs.hstr_ext",
    sources=["bodo/libs/_str_ext.cpp", "bodo/libs/_bodo_common.cpp"],
    libraries=np_compile_args["libraries"],
    define_macros=np_compile_args["define_macros"],
    extra_compile_args=eca,
    extra_link_args=ela,
    include_dirs=np_compile_args["include_dirs"] + ind,
    library_dirs=np_compile_args["library_dirs"] + lid,
)


# TODO: make Arrow optional in decimal extension similar to parquet extension?
ext_decimal = Extension(
    name="bodo.libs.decimal_ext",
    sources=["bodo/libs/_decimal_ext.cpp"],
    depends=["bodo/libs/_bodo_common.h"],
    libraries=np_compile_args["libraries"] + ["arrow"],
    define_macros=np_compile_args["define_macros"],
    extra_compile_args=eca,
    extra_link_args=ela,
    include_dirs=np_compile_args["include_dirs"] + ind,
    library_dirs=np_compile_args["library_dirs"] + lid,
)


ext_arr = Extension(
    name="bodo.libs.array_ext",
    sources=["bodo/libs/_array.cpp",
        "bodo/libs/_bodo_common.cpp",
        "bodo/libs/_array_utils.cpp",
        "bodo/libs/_array_hash.cpp",
        "bodo/libs/_shuffle.cpp",
        "bodo/libs/_join.cpp",
        "bodo/libs/_groupby.cpp",
        "bodo/libs/_array_operations.cpp",
        "bodo/libs/_murmurhash3.cpp",
    ],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_murmurhash3.h",
        "bodo/libs/_distributed.h",
    ],
    libraries=MPI_LIBS,
    extra_compile_args=eca,
    extra_link_args=ela,
    define_macros=np_compile_args["define_macros"],
    include_dirs=np_compile_args["include_dirs"] + ind + extra_hash_ind,
    library_dirs=np_compile_args["library_dirs"] + lid,
)


ext_dt = Extension(
    name="bodo.libs.hdatetime_ext",
    sources=["bodo/libs/_datetime_ext.cpp"],
    libraries=np_compile_args["libraries"] + ["arrow"],
    define_macros=np_compile_args["define_macros"],
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"],
    include_dirs=np_compile_args["include_dirs"] + ind,
    library_dirs=np_compile_args["library_dirs"],
    language="c++",
)


ext_quantile = Extension(
    name="bodo.libs.quantile_alg",
    sources=["bodo/libs/_quantile_alg.cpp"],
    depends=["bodo/libs/_bodo_common.h"],
    libraries=MPI_LIBS,
    extra_compile_args=eca,
    extra_link_args=ela,
    include_dirs=ind,
    library_dirs=lid,
)


# pq_libs = MPI_LIBS + ['boost_filesystem', 'arrow', 'parquet']
pq_libs = MPI_LIBS.copy()

# Windows MSVC can't have boost library names on command line
# auto-link magic of boost should be used
if not is_win:
    pq_libs += ["boost_filesystem"]

pq_libs += ["arrow", "parquet"]


ext_parquet = Extension(
    name="bodo.io.parquet_cpp",
    sources=["bodo/io/_parquet.cpp", "bodo/io/_parquet_reader.cpp"],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/io/_parquet_reader.h"
    ],
    libraries=pq_libs,
    include_dirs=["."] + ind,
    define_macros=[],
    extra_compile_args=eca,
    extra_link_args=ela,
    library_dirs=lid,
)

_ext_mods = [
    ext_hdist,
    ext_dict,
    ext_str,
    ext_decimal,
    ext_quantile,
    ext_dt,
    ext_io,
    ext_arr,
    ext_s3,
    ext_hdfs,
]

if _has_h5py:
    _ext_mods.append(ext_hdf5)
if _has_pyarrow:
    _ext_mods.append(ext_parquet)


setup(
    name="bodo",
    version=versioneer.get_version(),
    description="compiling Python code for clusters",
    long_description=readme(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Compilers",
        "Topic :: System :: Distributed Computing",
    ],
    keywords="data analytics cluster",
    url="https://bodo-inc.com",
    author="Bodo",
    packages=find_packages(),
    package_data={
        "bodo.tests": ["data/*", "data/int_nulls_multi.pq/*", "data/sdf_dt.pq/*"]
    },
    install_requires=["numba"],
    extras_require={"HDF5": ["h5py"], "Parquet": ["pyarrow"]},
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=_ext_mods,
)
