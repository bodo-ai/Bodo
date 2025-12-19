"""
Fix module paths in Bodo vendored mpi4py to enable finding impi DLLs and mpiexec
in Windows pip.
"""

import argparse
import os
import shutil
import zipfile


def patch_mpi4py(wheel_dir: str) -> None:
    """
    Fix module path references in mpi4py.__init__.py to reflect the
    fact that the module is part of the bodo package.
    """
    mpi_init_path = os.path.join(
        wheel_dir, "bodo", "mpi4py", "_vendored_mpi4py", "__init__.py"
    )

    with open(mpi_init_path) as init_f:
        mpi_init_text: str = init_f.read()

    patched_mpi_init_text = mpi_init_text.replace(
        "mpi4py.MPI", "bodo.mpi4py._vendored_mpi4py.MPI"
    )

    with open(mpi_init_path, "w") as init_f:
        init_f.write(patched_mpi_init_text)


def patch_wheels(path: str) -> None:
    for wheel in os.listdir(path):
        print("\nPATCHING", wheel)

        tmp_dir = "whl_tmp"

        with zipfile.ZipFile(os.path.join(path, wheel), "r") as z:
            z.extractall(tmp_dir)

        # apply patches
        patch_mpi4py(tmp_dir)

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
        help="Path to a wheel directory containing files to patch",
    )
    args = parser.parse_args()
    patch_wheels(args.path)
