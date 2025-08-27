import argparse
import os
import shutil
import zipfile
from subprocess import check_call, check_output
from zipfile import ZipFile, ZipInfo

# This script modifies the load path of libmpi in the main Bodo extension


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


def patch_lib(fpath, prefix_path):
    # Update the LC_LOAD_DYLIB call for libmpi if it exists.
    # https://medium.com/@donblas/fun-with-rpath-otool-and-install-name-tool-e3e41ae86172
    load_libs = check_output(["otool", "-L", fpath]).decode("utf-8").split("\n")
    # Example output
    # whl_tmp/bodo-2021.12.1+58.g027d8c402.dirty/bodo/libs/hdatetime_ext.cpython-39-darwin.so:
    #     /Users/nicholasriasanovsky/Documents/bodo/mpich-info/mpich/lib/libmpi.12.dylib (compatibility version 14.0.0, current version 14.8.0)
    #     @rpath/libarrow.500.dylib (compatibility version 500.0.0, current version 500.0.0)
    #     @rpath/libc++.1.dylib (compatibility version 1.0.0, current version 1.0.0)
    #     /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1311.0.0)
    print("Loaded Libs for", fpath)

    for lib in load_libs[1:]:
        print(lib)
        if is_libmpi(lib) or is_libpmpi(lib):
            print(f"Patching {lib}")
            # Each line looks like
            #      path ...
            # So we strip whitespace and split on spaces,
            # taking the first element for the path
            lib_path = lib.strip().split(" ")[0]
            # Update the mpi path to be relative to the loader.
            # https://stackoverflow.com/questions/4824885/remove-dependent-shared-library-from-a-dylib
            filename = lib_path.split("/")[-1]
            check_call(
                [
                    "install_name_tool",
                    "-change",
                    lib_path,
                    f"{prefix_path}/{filename}",
                    fpath,
                ]
            )
            check_call(["codesign", "--force", "--sign", "-", fpath])


def is_libmpi(fname):
    return "libmpi" in fname


def is_libpmpi(fname):
    return "libpmpi" in fname


def patch_libs(path):
    """patch .so files found recursively in path"""
    ext_prefix = "@loader_path/../../.."
    # Assumes MPI.cpython.*.so is located in bodo/mpi4py/_vendored_mpi4py
    mpi_prefix = ext_prefix + "/../.."
    for root, _, files in os.walk(path):
        for f in files:
            # Only patch the main extension + MPI since other extensions might have different relative paths
            # other libraries will find the library in the cache
            if f.startswith("ext.cpython") and f.endswith(".so"):
                patch_lib(os.path.join(root, f), ext_prefix)
            elif f.startswith("MPI.cpython") and f.endswith(".so"):
                patch_lib(os.path.join(root, f), mpi_prefix)


def patch_wheel(path):
    print("\nPATCHING", path)
    # unpack wheel in "whl_tmp" directory
    with ZipFileWithPermissions(path, "r") as z:
        z.extractall("whl_tmp")

    patch_libs("whl_tmp")
    # Delete the original wheel
    os.remove(path)
    # This packs the wheel back
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
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
