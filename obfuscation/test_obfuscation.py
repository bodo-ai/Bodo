#!/usr/bin/env python
import subprocess
import sys
import os
import shutil


list_test_files = []

complete_dir = os.path.join("./", "examples")
content_dir = os.listdir(complete_dir)
for e_file in content_dir:
    if e_file.startswith("test"):
        list_test_files.append(e_file)

# Now treating the files one by one.

def get_func_text(e_file):
    f = open(e_file, "r")
    list_lines = f.readlines()
    f.close()
    func_text = ""
    for e_line in list_lines:
        func_text += e_line
    return func_text

def get_variable_expression(func_text, var_name):
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    return loc_vars[var_name]


for e_file in list_test_files:
    # Copying the script obfuscating it
    e_test_file = os.path.join(complete_dir, e_file)
    TmpFile = "/tmp/test.py"
    shutil.copy(e_test_file, TmpFile)
    list_command_args = [sys.executable, "./obfuscate.py", "file", TmpFile]
    returncode = subprocess.run(list_command_args).returncode
    if returncode!=0:
        print("obfuscate.py code failed, returncode=", returncode)
        exit(0)
    # Reading the python programs
    func_text1 = get_func_text(TmpFile)
    func_text2 = get_func_text(e_test_file)
    if func_text1 == func_text2:
        print("obfuscation failed to create a different program")
        exit(0)
    # Compiling the function
    f1 = get_variable_expression(func_text1, "f")
    f2 = get_variable_expression(func_text2, "f")
    # Loading the data
    l_split = e_file.split("_")
    prefix = l_split[0]
    input_file = os.path.join(complete_dir, 'input' + prefix[4:])
    data_text = get_func_text(input_file)
    kwargs = get_variable_expression(data_text, "data")
    # Running them
    val1 = f1(**kwargs)
    val2 = f2(**kwargs)
    if val1 != val2:
        print("obfuscation failed to work correctly")
        exit(0)

print("Successful termination of the obfuscation tests")
