import os
import shutil
from distutils.errors import DistutilsExecError

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

cwd = os.getcwd()
setup_py_dir_path = os.path.dirname(os.path.realpath(__file__))
# despite the name, this also works for directories
if not os.path.samefile(cwd, setup_py_dir_path):
    raise Exception(
        "setup.py should only be invoked if the current working directory is in the same directory as Setup.py.\nThis is to prevent having with conflicting .egg-info in the same directory when building Bodo's submodules."
    )


def build_libs(obj, dev_mode=False):
    """Build maven and then calls the original run command"""
    try:
        pom_dir = os.path.join("bodo_azurefs_sas_token_provider", "pom.xml")
        cmd_list = ["mvn", "clean", "install"]

        # Batch Mode (--batch-mode or -B) will assume your running in CI
        # --no-transfer-progress or -ntp will suppress additional download messages
        # Both significantly reduce output
        if not dev_mode:
            cmd_list += ["--batch-mode", "--no-transfer-progress"]

        cmd_list += [
            "-Dmaven.test.skip=true",
            "-f",
            pom_dir,
        ]

        obj.spawn(cmd_list)
        curr_path = os.path.dirname(os.path.realpath(__file__))
        executable_jar_dir = os.path.join(
            curr_path,
            "bodo_azurefs_sas_token_provider/target",
        )
        to_jar_path = os.path.join(curr_path, "bodo_azurefs_sas_token_provider/jars")
        os.makedirs(to_jar_path, exist_ok=True)
        os.rename(
            os.path.join(executable_jar_dir, "bodo-azurefs-sas-token-provider-1.0.jar"),
            # This name should match the name in __init__.py
            os.path.join(to_jar_path, "bodo-azurefs-sas-token-provider.jar"),
        )
        # Delete libs if they already exist
        jar_lib_dst = os.path.join(to_jar_path, "libs/")
        shutil.rmtree(jar_lib_dst, ignore_errors=True)
        os.rename(os.path.join(executable_jar_dir, "libs/"), jar_lib_dst)
    except DistutilsExecError as e:
        obj.error("Maven Build Failed with Error:", e)


class CustomDevelopCommand(develop):
    """Custom command to build the jars with python setup.py develop"""

    def run(self):
        build_libs(self, dev_mode=True)
        super().run()


class CustomBuildCommand(build_py):
    def run(self):
        """Creates the generated library/tests, builds maven, and then calls the original run command"""
        build_libs(self)
        super().run()


setup(
    name="bodo-azurefs-sas-token-provider",
    version="1.0",
    description="Bodo's customer SAS token provider",
    long_description="Bodo's customer SAS token provider",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Compilers",
        "Topic :: System :: Distributed Computing",
    ],
    keywords="data analytics cluster",
    url="https://bodo.ai",
    author="Bodo.ai",
    packages=find_packages(),
    # This is needed so that the jars are included if/when the package is installed via pip.
    package_data={
        "bodo_azurefs_sas_token_provider": [
            "jars/bodo-azurefs-sas-token-provider.jar",
            "jars/libs/*.jar",
        ]
    },
    # When doing `python setup.py develop`, setuptools will try to install whatever is
    # in `install_requires` after building, so we set it to empty (we don't want to
    # install bodo in development mode, and it will also break CI
    install_requires=[],
    python_requires=">=3.8,<3.11",
    cmdclass={"develop": CustomDevelopCommand, "build_py": CustomBuildCommand},
)
