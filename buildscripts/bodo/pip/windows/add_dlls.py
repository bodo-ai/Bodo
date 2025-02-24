# Begin bodo pip patch
def add_pyarrow_dlls():
    import os

    import pyarrow as pa

    for lib_dir in pa.get_library_dirs():
        os.add_dll_directory(lib_dir)


add_pyarrow_dlls()
del add_pyarrow_dlls
# End bodo pip patch
