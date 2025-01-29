# Remove definition of c_PyFileSystemBodo to prevent compilation errors on windows
import os

print("############################# APPLYING PATCH TO GENERATED _pyfs.h")

cython_def = """
struct c_PyFileSystemBodo {
  struct __pyx_obj_7pyarrow_3_fs_PyFileSystem __pyx_base;
};
"""

new_def = """
struct __pyx_obj_7pyarrow_3_fs_PyFileSystem {
};

"""


path = os.path.join("bodo", "io", "pyfs.h")

with open(path) as f:
    c_code = f.read()

c_code = c_code.replace(cython_def, new_def + cython_def)

with open(path, "w") as f:
    f.write(c_code)
