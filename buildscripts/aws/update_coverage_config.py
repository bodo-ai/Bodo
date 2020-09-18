"""
    File that modifies the coverage configuration to
    enable distributed testing. This step is needed because
    AWS Codebuild seems to use a random hash as part of the
    absolute path to the code it downloads, of the form:
    /codebuild/output/<RANDOMHASH>/src/github.com/Bodo-inc/Bodo

    This script appends a path portion to the configuration file
    that tells pytest-cov that the path using the hash on the machine
    executing SonarQube is equivalent to all those used in the testing
    step. Unfortunately the first given path must be a valid absolute
    path on the given machine, so this must be produced at runtime.
"""
import os

with open("setup.cfg", "a") as f:
    print("", file=f)
    print("[coverage:paths]", file=f)
    print("source = ", file=f)
    print("    {}".format(os.getcwd()), file=f)
    print("    /codebuild/output/*/src/github.com/Bodo-inc/Bodo", file=f)
