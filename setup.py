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
    with open("README.rst") as f:
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

_has_xenon = False

if "BODO_XE_SUPPORT" in os.environ and os.environ["BODO_XE_SUPPORT"] != "0":
    _has_xenon = True

ind = [PREFIX_DIR + "/include"]
lid = [PREFIX_DIR + "/lib"]
eca = ["-std=c++11"]  # '-g', '-O0']
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
        "bodo/libs/_datetime_ext.h",
    ],
    libraries=io_libs,
    include_dirs=ind + np_compile_args["include_dirs"],
    library_dirs=lid,
    define_macros=H5_CPP_FLAGS,
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


ext_hdist = Extension(
    name="bodo.libs.hdist",
    sources=["bodo/libs/_distributed.cpp"],
    depends=["bodo/libs/_bodo_common.h"],
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


str_libs = np_compile_args["libraries"]

if not is_win:
    str_libs += ["boost_regex"]

ext_str = Extension(
    name="bodo.libs.hstr_ext",
    sources=["bodo/libs/_str_ext.cpp"],
    libraries=str_libs,
    define_macros=np_compile_args["define_macros"] + [("USE_BOOST_REGEX", None)],
    extra_compile_args=eca,
    extra_link_args=ela,
    include_dirs=np_compile_args["include_dirs"] + ind,
    library_dirs=np_compile_args["library_dirs"] + lid,
)

ext_arr = Extension(
    name="bodo.libs.array_tools_ext",
    sources=["bodo/libs/_array_tools.cpp"],
    depends=[
        "bodo/libs/_bodo_common.h",
        "bodo/libs/_murmurhash3.h",
        "bodo/libs/_murmurhash3.cpp",
        "bodo/libs/_distributed.h",
    ],
    libraries=MPI_LIBS,
    extra_compile_args=eca,
    extra_link_args=ela,
    define_macros=np_compile_args["define_macros"],
    include_dirs=np_compile_args["include_dirs"] + ind,
    library_dirs=np_compile_args["library_dirs"] + lid,
)


ext_dt = Extension(
    name="bodo.libs.hdatetime_ext",
    sources=["bodo/libs/_datetime_ext.cpp"],
    libraries=np_compile_args["libraries"],
    define_macros=np_compile_args["define_macros"],
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"],
    include_dirs=np_compile_args["include_dirs"],
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
    libraries=pq_libs,
    include_dirs=["."] + ind,
    define_macros=[("BUILTIN_PARQUET_READER", None)],
    extra_compile_args=eca,
    extra_link_args=ela,
    library_dirs=lid,
)


ext_xenon_wrapper = Extension(
    name="bodo.hxe_ext",
    sources=["bodo/io/_xe_wrapper.cpp"],
    # include_dirs = ['/usr/include'],
    include_dirs=["."] + ind,
    library_dirs=["."] + lid,
    libraries=["xe"],
    extra_compile_args=eca,
    extra_link_args=ela,
)

_ext_mods = [
    ext_hdist,
    ext_dict,
    ext_str,
    ext_quantile,
    ext_dt,
    ext_io,
    ext_arr,
]

if _has_h5py:
    _ext_mods.append(ext_hdf5)
if _has_pyarrow:
    _ext_mods.append(ext_parquet)


if _has_xenon:
    _ext_mods.append(ext_xenon_wrapper)

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
    install_requires=["numba"],
    extras_require={"HDF5": ["h5py"], "Parquet": ["pyarrow"]},
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=_ext_mods,
)
