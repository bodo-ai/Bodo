#!/usr/bin/env python
# Copyright (C) 2020 Bodo Inc. All rights reserved.
import os
import subprocess
import sys

"""obfuscation of the ir, hiframes, etc. directories using the obfuscate script

Use:

  ./do_obfuscation.py

The output shows the result of the operations."""

list_dir = ["", "dl", "ir", "io", "utils", "hiframes", "libs"]

file_dir = os.path.dirname(os.path.realpath(__file__))

if "pre-commit" in os.listdir(os.path.join(file_dir, "../.git/hooks")):
    raise Exception(
        "Error! Pre-commit hooks are installed. This can cause issues when building locally, due to obfuscation removing the directives that disable pre-commits for specific sections of code.\nTo fix, please run `pre-commit uninstall` from within the Bodo repository."
    )

list_files = []
for e_dir in list_dir:
    complete_dir = os.path.join(file_dir, os.path.join("../bodo", e_dir))
    content_dir = os.listdir(complete_dir)
    for e_file in content_dir:
        if e_file.endswith(".py"):
            list_files.append(os.path.join(complete_dir, e_file))

list_command_args = [
    sys.executable,
    os.path.join(file_dir, "obfuscate.py"),
    "file",
] + list_files

exit(subprocess.run(list_command_args).returncode)
