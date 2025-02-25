"""
Fix module paths in Bodo vendored mpi4py to enable finding impi DLLs and mpiexec
in Windows pip.
"""

import argparse
import os


def patch_mpi4py(path: str) -> None:
    """
    Fix module path references in mpi4py.__init__.py to reflect the
    fact that the module is part of the bodo package.
    """
    with open(os.path.join(path, "__init__.py")) as init_f:
        mpi_init_text: str = init_f.read()

    patched_mpi_init_text = mpi_init_text.replace("mpi4py.MPI", "bodo.mpi4py.MPI")
    patched_mpi_init_text = "# patch successful!\n" + patched_mpi_init_text

    with open(os.path.join(path, "__init__.py"), "w") as init_f:
        init_f.write(patched_mpi_init_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        help="Path to the directory containing mpi4py __init__.py to patch.",
    )
    args = parser.parse_args()
    patch_mpi4py(args.path)
