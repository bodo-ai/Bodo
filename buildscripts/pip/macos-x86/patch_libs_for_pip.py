import os
import shutil
from subprocess import check_call, check_output

# This script needs to be run in bodo sources top-level directory. The script:
# - Modifies RPATH of .so libraries in Bodo:
#     1) clears the existing RPATHs since these are system specific
#     2) Updates the mpi4py_mpich path to be relative to the loader for each library.


def patch_lib(fpath):
    # First we determine if there are existing rpaths by
    # looking at the otool -l output.
    # https://stackoverflow.com/questions/12521802/print-rpath-of-an-executable-on-macos
    rpath_output = check_output(["otool", "-l", fpath]).decode("utf-8").split("\n")
    # Example Output:
    # ...
    # Load command 13
    #          cmd LC_LOAD_DYLIB
    #      cmdsize 56
    #         name /usr/lib/libSystem.B.dylib (offset 24)
    #   time stamp 2 Wed Dec 31 19:00:02 1969
    #      current version 1311.0.0
    # compatibility version 1.0.0
    # Load command 14
    #         cmd LC_RPATH
    #     cmdsize 72
    #         path /Users/nicholasriasanovsky/miniconda3/envs/BUILDPIP/lib (offset 12)
    # Load command 15
    #         cmd LC_RPATH
    #     cmdsize 72
    #         path /Users/nicholasriasanovsky/miniconda3/envs/BUILDPIP/lib (offset 12)
    # Load command 16
    #         cmd LC_RPATH
    #     cmdsize 24
    #         path /mpich/lib (offset 12)
    # ...
    #
    # Here we want to delete each LC_RPATH because they are absolute paths on the build machine
    # Output here is spread across multiple lines. However, each line is of
    # the form:
    # Load command #
    #   cmd ...

    # We care about the sections that have LC_RPATH. These refer to absolute paths
    # on the current machine. We want to delete the line that says path.
    is_lc_rpath = False
    parsing_load_command = False
    for line in rpath_output:
        if "Load command" in line:
            is_lc_rpath = False  # reset
            parsing_load_command = True
        if parsing_load_command:
            if "LC_RPATH" in line:
                is_lc_rpath = True
            elif is_lc_rpath:
                # If we have found an lc_rpath we are looking for the path name
                # to remove it.
                if "path" in line:
                    sections = line.strip().split(" ")
                    # The path should immediately follow the space after the word "path"
                    index = sections.index("path") + 1
                    path_section = sections[index]
                    assert (
                        "/" in path_section
                    ), "Something went wrong, path wasn't where it was expected to be"
                    check_call(
                        ["install_name_tool", "-delete_rpath", path_section, fpath]
                    )
                    is_lc_rpath = False
                    parsing_load_command = False

    # Update the LC_LOAD_DYLIB call for libmpi if it exists.
    # https://medium.com/@donblas/fun-with-rpath-otool-and-install-name-tool-e3e41ae86172
    load_libs = check_output(["otool", "-L", fpath]).decode("utf-8").split("\n")
    # Example output
    # whl_tmp/bodo-2021.12.1+58.g027d8c402.dirty/bodo/libs/hdatetime_ext.cpython-39-darwin.so:
    #     /Users/nicholasriasanovsky/Documents/bodo/mpich-info/mpich/lib/libmpi.12.dylib (compatibility version 14.0.0, current version 14.8.0)
    #     @rpath/libarrow.500.dylib (compatibility version 500.0.0, current version 500.0.0)
    #     @rpath/libc++.1.dylib (compatibility version 1.0.0, current version 1.0.0)
    #     /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1311.0.0)

    for lib in load_libs[1:]:
        if is_libmpi(lib):
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
                    f"@loader_path/../../mpi4py/.dylibs/{filename}",
                    fpath,
                ]
            )
        elif is_ssl_lib(lib):
            # Each line looks like
            #      path ...
            # So we strip whitespace and split on spaces,
            # taking the first element for the path
            lib_path = lib.strip().split(" ")[0]
            # Update the mpi path to be relative to the loader. It should be
            # in bodo/libs
            # https://stackoverflow.com/questions/4824885/remove-dependent-shared-library-from-a-dylib
            filename = lib_path.split("/")[-1]
            check_call(
                [
                    "install_name_tool",
                    "-change",
                    lib_path,
                    f"@loader_path/{filename}",
                    fpath,
                ]
            )


def is_libmpi(fname):
    return "libmpi" in fname


def is_ssl_lib(fname):
    return "libssl" in fname or "libcrypto" in fname


def patch_libs(path):
    """patch any .so files found recursively in path"""
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".so"):
                patch_lib(os.path.join(root, f))


# patch the wheels in "wheelhouse" directory
os.chdir("wheelhouse")
for f in os.listdir("."):
    if f.endswith(".whl"):
        print("\nPATCHING", f)
        # unpack wheel in "whl_tmp" directory
        check_call(["python", "-m", "wheel", "unpack", f, "-d", "whl_tmp"])
        patch_libs("whl_tmp")
        # contents of wheel are in the first path in "whl_tmp"
        dirname = os.listdir("whl_tmp")[0]
        # This packs the wheel back into f
        check_call(["python", "-m", "wheel", "pack", "whl_tmp/" + dirname])
        shutil.rmtree("whl_tmp")
