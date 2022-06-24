import os
import sys
from distutils.errors import DistutilsExecError

from setuptools import find_packages, setup
from setuptools.command.develop import develop

# ----- Trick to pass the Bodo Version in CI -----
# During CI, conda copies and isolates the iceberg subfolder from the monorep
# Thus, we can not import the top-level versioneer.py to get the version
# Instead, we first get the version, save it in the CONNECTOR_VERSION environment variable
# and then pass that in during the build step
if "CONNECTOR_VERSION" in os.environ:
    version = os.environ["CONNECTOR_VERSION"]
else:
    sys.path.insert(0, "..")
    from versioneer import get_version

    version = get_version()
    version += "alpha"


# Automatically Build Java Project on `python setup.py develop`
development_mode = "develop" in sys.argv


def build_libs(obj):
    """Build maven and then calls the original run command"""

    try:
        pom_dir = os.path.join("bodo_iceberg_connector", "iceberg-java", "pom.xml")
        cmd_list = ["mvn", "install"]
        cmd_list += [
            "-Dmaven.test.skip=true",
            "-f",
            pom_dir,
        ]

        obj.spawn(cmd_list)
    except DistutilsExecError as e:
        obj.error("Maven Build Failed with Error:", e)


class CustomDevelopCommand(develop):
    """Custom command to build the jars with python setup.py develop"""

    def run(self):
        build_libs(self)
        super().run()


setup(
    name="bodo-iceberg-connector",
    version=version,
    description="Bodo Connector for Iceberg",
    long_description="Bodo Connector for Iceberg",
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
        "bodo_iceberg_connector": ["bodo_iceberg_connector/iceberg-java/target/*.jar",
            "bodo_iceberg_connector/iceberg-java/target/libs/*.jar",
        ]
    },
    # When doing `python setup.py develop`, setuptools will try to install whatever is
    # in `install_requires` after building, so we set it to empty (we don't want to
    # install bodo in development mode, and it will also break CI
    install_requires=[] if development_mode else ["py4j==0.10.9.3"],
    python_requires=">=3.8,<3.11",
    cmdclass={"develop": CustomDevelopCommand},
)
