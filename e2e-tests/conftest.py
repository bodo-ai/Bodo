import os
import sys

e2e_tests_base_dir = os.path.dirname(__file__)
if e2e_tests_base_dir not in sys.path:
    sys.path.append(e2e_tests_base_dir)
if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] += os.pathsep + e2e_tests_base_dir
else:
    os.environ["PYTHONPATH"] = e2e_tests_base_dir
