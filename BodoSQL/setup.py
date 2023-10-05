import os
import sys
from distutils.errors import DistutilsExecError

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

import versioneer

versioneer_cmdclass = versioneer.get_cmdclass()

cwd = os.getcwd()
setup_py_dir_path = os.path.dirname(os.path.realpath(__file__))
# despite the name, this also works for directories
if not os.path.samefile(cwd, setup_py_dir_path):
    raise Exception(
        "setup.py should only be invoked if the current working directory is in the same directory as Setup.py.\nThis is to prevent having with conflicting .egg-info in the same directory when building Bodo's submodules."
    )

try:
    # wheel is an optional import that may
    # only exist when building for pip
    from wheel.bdist_wheel import bdist_wheel

    bdist_wheel_parent = bdist_wheel
    # Set the parent to extend versioneer
    if "bdist_wheel" in versioneer_cmdclass:
        bdist_wheel_parent = versioneer_cmdclass["bdist_wheel"]

    class CustomBDistWheelCommand(bdist_wheel_parent):
        def run(self):
            """Creates the generated library/tests, builds maven, and then calls the original run command"""
            build_libs(self)
            super().run()

    bdist_wheel_command = CustomBDistWheelCommand
except ImportError:
    # If we cannot import wheel skip it
    bdist_wheel_command = None


development_mode = "develop" in sys.argv
# TODO: Figure out how to just pass the command to develop
if "--no_update_calcite" in sys.argv:
    update_calcite = False
    sys.argv.remove("--no_update_calcite")
else:
    update_calcite = True


def build_libs(obj):
    """Creates the generated library/tests, builds maven, and then calls the original run command"""
    import buildscripts.python_library_build.write_generated_lib

    buildscripts.python_library_build.write_generated_lib.generate_and_write_library()
    try:
        pom_dir = os.path.join("calcite_sql", "pom.xml")
        dmvn_repo = os.path.dirname("bodosql-protocol-mvn/")
        cmd_list = ["mvn", "clean", "install"]
        if update_calcite:
            cmd_list.append("-U")

        cmd_list.extend(
            [
                "-Dmaven.test.skip=true",
                "-f",
                pom_dir,
                "-Dmaven.repo.local=" + dmvn_repo,
            ]
        )
        obj.spawn(cmd_list)

        executable_jar_path = os.path.join(
            "calcite_sql",
            "bodosql-calcite-application",
            "target",
            "BodoSqlCalcite.jar",
        )
        to_jar_path = os.path.join("bodosql", "jars")
        os.makedirs(to_jar_path, exist_ok=True)
        os.rename(
            executable_jar_path, os.path.join(to_jar_path, "bodosql-executable.jar")
        )
    except DistutilsExecError as e:
        obj.error("maven build failed with error:", e)


def readme():
    with open("README.md") as f:
        return f.read()


develop_parent = develop
if "develop" in versioneer_cmdclass:
    develop_parent = versioneer_cmdclass["develop"]

build_py_parent = build_py
if "build_py" in versioneer_cmdclass:
    build_py_parent = versioneer_cmdclass["build_py"]


class CustomDevelopCommand(develop_parent):
    def run(self):
        """Creates the generated library/tests, builds maven, and then calls the original run command"""
        build_libs(self)
        super().run()


class CustomBuildCommand(build_py_parent):
    def run(self):
        """Creates the generated library/tests, builds maven, and then calls the original run command"""
        build_libs(self)
        super().run()


# TODO: Include a clean command that cleans Maven + deletes the generated lib

cmdclass = versioneer_cmdclass.copy()
# Replace any subclass. Update always takes the value from
# the new dictionary.
cmdclass.update({"develop": CustomDevelopCommand, "build_py": CustomBuildCommand})
if bdist_wheel_command is not None:
    # If wheel is installed add the bdist wheel command.
    cmdclass["bdist_wheel"] = bdist_wheel_command

setup(
    name="bodosql",
    version=versioneer.get_version(),
    description="compile SQL for clusters",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Compilers",
        "Topic :: System :: Distributed Computing",
    ],
    keywords="data analytics cluster",
    url="https://bodo.ai",
    author="Bodo.ai",
    packages=find_packages(),
    package_data={
        "bodosql.tests": ["*/*"],
        "bodosql": ["pytest.ini", "jars/*.jar", "opensource/*.NOTICE"],
    },
    # When doing `python setup.py develop`, setuptools will try to install whatever is
    # in `install_requires` after building, so we set it to empty (we don't want to
    # install bodo in development mode, and it will also break CI)
    # match Bodo version to install with BodoSQL version
    install_requires=[]
    if development_mode
    else [f"bodo=={versioneer.get_version()}", "py4j==0.10.9.7"],
    python_requires=">=3.9,<3.12",
    # Update the build and develop commands
    cmdclass=cmdclass,
)
