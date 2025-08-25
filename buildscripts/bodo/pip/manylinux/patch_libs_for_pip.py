import argparse
import os
import shutil
import zipfile
from subprocess import check_call
from zipfile import ZipFile, ZipInfo

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


class ZipFileWithPermissions(ZipFile):
    """Custom ZipFile class handling file permissions.
    https://stackoverflow.com/a/54748564"""

    def _extract_member(self, member, targetpath, pwd):
        if not isinstance(member, ZipInfo):
            member = self.getinfo(member)

        targetpath = super()._extract_member(member, targetpath, pwd)

        attr = member.external_attr >> 16
        if attr != 0:
            os.chmod(targetpath, attr)
        return targetpath


def patch_lib(fpath):
    # clear rpath from library
    print("Patching: ", fpath)
    try:
        check_call(["patchelf", "--remove-rpath", fpath])
    except Exception:
        # Patchelf doesn't like libraries that are just links to other libraries
        # so we just ignore them
        print("Error removing rpath from", fpath)
        return

    # set rpath that points to libmpi location (of mpi4py_mpich package).
    # Note that this is a relative path and requires mpi4py_mpich package to be
    # installed in same site-packages folder as Bodo when running Bodo
    RPATH = "$ORIGIN/../../..:$ORIGIN/../../../../..:$ORIGIN/../bodo.libs:$ORIGIN/../lib64:$ORIGIN/../pyarrow"
    check_call(["patchelf", "--force-rpath", "--set-rpath", RPATH, fpath])


def patch_libs(path):
    """patch any .so files found recursively in path"""
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".so"):
                patch_lib(os.path.join(root, f))


def patch_wheel(path):
    for wheel in os.listdir(path):
        print("\nPATCHING", wheel)
        # unpack wheel in "whl_tmp" directory
        with ZipFileWithPermissions(os.path.join(path, wheel), "r") as z:
            z.extractall("whl_tmp")

        # get file names of Arrow libraries that were included in bodo/libs by `auditwheel repair`
        patch_libs("whl_tmp")
        # Delete the original wheel
        os.remove(os.path.join(path, wheel))
        # This packs the wheel back
        with zipfile.ZipFile(os.path.join(path, wheel), "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk("whl_tmp"):
                for f in files:
                    z.write(
                        os.path.join(root, f),
                        os.path.relpath(os.path.join(root, f), "whl_tmp"),
                    )
        shutil.rmtree("whl_tmp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        help="Path to a wheel or directory containing .so files to patch",
    )
    args = parser.parse_args()
    patch_wheel(args.path)
