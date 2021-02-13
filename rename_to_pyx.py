""" This script can rename any Python files under the bodo source code directory
    to .pyx except the ones excluded by excludes_pattern below """
import glob
import os

excludes_pattern = [
    "bodo/tests/**/*.py",  # all python files under bodo/tests
    "bodo/**/__init__.py",  # any __init__.py files
    "bodo/bench/**/*.py",  # all python files under bodo/bench
    "bodo/_version.py",
    "bodo/runtests.py",
]
excludes = set()
for pat in excludes_pattern:
    excludes.update([os.path.normcase(path) for path in glob.glob(pat, recursive=True)])


def py_to_pyx(path):
    # using normcase to normalize slashes on Windows
    path = os.path.normcase(path)
    for f in os.listdir(path):
        fpath = path + os.sep + f
        if f.endswith(".py") and fpath not in excludes:
            os.rename(fpath, fpath[:-3] + ".pyx")
        elif os.path.isdir(fpath):
            py_to_pyx(fpath)


# We are only cythonizing files in bodo/transforms (at least for now)
py_to_pyx("bodo/transforms")
