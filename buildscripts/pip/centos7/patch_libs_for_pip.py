import os
import subprocess
from subprocess import check_call, check_output
import shutil
import re

# See https://bodo.atlassian.net/wiki/spaces/DD/pages/946929672/Bodo+Linux+pip+package
# for more information.

# This script needs to be run in bodo sources top-level directory. The script:
# - Modifies RPATH of .so libraries in Bodo:
#     1) clears the existing RPATH
#     2) Adds mpi4py_mpich.libs directory with relative path to RPATH
#        and point to the libmpi-XXX.so.a.b.c contained there
#     3) Add bodo.libs with relative path to the RPATH (ssl libraries are there,
#         bundled by `auditwheel repair`)
#     4) Undo the changes made by auditwheel repair that refer to Arrow libraries
# - Removes libmpi and any arrow libraries from the contents of Bodo package

# XXX Maybe it's better to do all of this as part of `auditwheel repair`, but
# it would require [monkey]patching it


def patch_lib(fpath, new_libmpi_name, arrow_libs):
    # clear rpath from library
    check_call(["patchelf", "--remove-rpath", fpath])

    # set rpath that points to libmpi location (of mpi4py_mpich package).
    # Note that this is a relative path and requires mpi4py_mpich package to be
    # installed in same site-packages folder as Bodo when running Bodo
    RPATH = "$ORIGIN/../../mpi4py_mpich.libs:$ORIGIN/../../bodo.libs"
    check_call(["patchelf", "--force-rpath", "--set-rpath", RPATH, fpath])

    # get original libmpi library name
    old_libmpi_name = None
    try:
        old_libmpi_name = (
            check_output(f"ldd {fpath} | grep libmpi", shell=True).split()[0].decode()
        )
    except:
        pass
    if old_libmpi_name:
        print(f"patching {fpath} : {old_libmpi_name} -> {new_libmpi_name}")
        # replace with new name
        check_call(
            ["patchelf", "--replace-needed", old_libmpi_name, new_libmpi_name, fpath]
        )

    # `auditwheel repair` (which runs before this script) bundles Arrow libraries with Bodo,
    # and includes a hash in their filename (for example: libarrow-777f8346.so.500). But for
    # Bodo to work correctly with pyarrow, it needs to use the same libraries (the library files
    # that pyarrow loads at runtime), so we have to rename to their original names
    regexp = re.compile("lib(.*)-.*\.so\.(\d+)$")
    for arrow_lib in arrow_libs:
        m = regexp.match(arrow_lib)
        libname = m.group(1)
        arrow_version = m.group(2)
        print(f"patching {fpath} : {arrow_lib} -> lib{libname}.so.{arrow_version}")
        check_call(
            [
                "patchelf",
                "--replace-needed",
                arrow_lib,
                f"lib{libname}.so.{arrow_version}",
                fpath,
            ]
        )


def patch_libs(path, new_libmpi_name, arrow_libs):
    """patch any .so files found recursively in path"""
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".so"):
                patch_lib(os.path.join(root, f), new_libmpi_name, arrow_libs)


def is_arrow_lib(fname):
    return "libarrow" in fname or "libparquet" in fname


def get_libmpi_filename():
    # get name of file that starts with libmpi in mpi4py_mpich.__file__ + "../mpi4py_mpich.libs"
    import mpi4py

    libpath = os.path.join(os.path.dirname(mpi4py.__file__), "..", "mpi4py_mpich.libs")
    for fname in os.listdir(libpath):
        if fname.startswith("libmpi"):
            return fname
    assert False, "mpi4py_mpich libmpi not found"


# Bodo pip needs to use the libmpi-xxx.so.yyy file that is part of mpi4py_mpich.
# We get the name of that file in case it doesn't match the one that Bodo
# was built against, and will edit the RPATH entry accordingly in patch_libs()
new_libmpi_name = get_libmpi_filename()
print("libmpi filename in mpi4py_mpich package is", new_libmpi_name)

# patch the wheels in "wheelhouse" directory
os.chdir("wheelhouse")
for f in os.listdir("."):
    if f.endswith(".whl"):
        print("\nPATCHING", f)
        # unpack wheel in "whl_tmp" directory
        check_call([f"{os.environ['PYBIN']}/wheel", "unpack", f, "-d", "whl_tmp"])
        # contents of wheel are in the first path in "whl_tmp"
        dirname = os.listdir("whl_tmp")[0]
        # get file names of Arrow libraries that were included in bodo/libs by `auditwheel repair`
        arrow_libs = [
            f for f in os.listdir(f"whl_tmp/{dirname}/bodo.libs") if is_arrow_lib(f)
        ]
        patch_libs("whl_tmp", new_libmpi_name, arrow_libs)
        # remove Arrow and MPI libraries that were added by `auditwheel repair`
        # because Bodo needs to use the libmpi.so that is part of mpi4py_mpich package
        # and the Arrow libraries that are part of pyarrow
        subprocess.call(f"rm whl_tmp/{dirname}/bodo.libs/libarrow*", shell=True)
        subprocess.call(f"rm whl_tmp/{dirname}/bodo.libs/libparquet*", shell=True)
        subprocess.call(f"rm whl_tmp/{dirname}/bodo.libs/libmpi*", shell=True)
        check_call([f"{os.environ['PYBIN']}/wheel", "pack", "whl_tmp/" + dirname])
        shutil.rmtree("whl_tmp")
        os.remove(f)  # delete old wheel
