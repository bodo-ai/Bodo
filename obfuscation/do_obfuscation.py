#!/usr/bin/env python
# Copyright (C) 2020 Bodo Inc. All rights reserved.
import subprocess
import sys
import os

"""obfuscation of the ir, hiframes and transforms directories using the obfuscate script

Use:

  ./do_obfuscation.py

The output shows the result of the operations."""

list_dir=['ir', 'io', 'utils', 'hiframes', 'transforms', 'libs']

list_files=[]
for e_dir in list_dir:
    complete_dir = os.path.join("../bodo", e_dir)
    content_dir = os.listdir(complete_dir)
    for e_file in content_dir:
        if e_file.endswith(".py"):
            list_files.append(os.path.join(complete_dir, e_file))

list_command_args = [sys.executable, './obfuscate.py', 'file'] + list_files

subprocess.run(list_command_args)
