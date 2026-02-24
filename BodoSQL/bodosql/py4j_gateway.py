"""
APIs used to launch and connect to the Java py4j calcite gateway server.
"""

import os
import sys
import warnings
from typing import cast

from mpi4py import MPI
from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

import bodo

# This gateway is always None on every rank but rank 0,
# it is initialized by the get_gateway call.
gateway = None


def get_java_path() -> str:
    """
    Ensure that the Java executable Py4J uses is the OpenJDK package
    installed in the current conda environment
    """

    # Currently inside a conda sub-environment
    # except for platform
    if (
        "CONDA_PREFIX" in os.environ
        and "BODO_PLATFORM_WORKSPACE_UUID" not in os.environ
    ):
        conda_prefix = os.environ["CONDA_PREFIX"]
        if "JAVA_HOME" in os.environ:
            java_home = os.environ["JAVA_HOME"]
            if java_home != os.path.join(conda_prefix, "lib", "jvm"):
                warnings.warn(
                    "$JAVA_HOME is currently set to a location that isn't installed by Conda. "
                    "It is recommended that you use OpenJDK v17 from Conda with BodoSQL. To do so, first run\n"
                    "    conda install openjdk=17 -c conda-forge\n"
                    "and then reactivate your environment via\n"
                    f"    conda deactivate && conda activate {conda_prefix}"
                )
            return os.path.join(java_home, "bin", "java")

        else:
            warnings.warn(
                "$JAVA_HOME is currently unset. This occurs when OpenJDK is not installed in your conda environment or when your environment has recently changed by not reactivates. BodoSQL will default to using you system's Java."
                "It is recommended that you use OpenJDK v17 from Conda with BodoSQL. To do so, first run\n"
                "    conda install openjdk=17 -c conda-forge\n"
                "and then reactivate your environment via\n"
                f"    conda deactivate && conda activate {conda_prefix}"
            )
            # TODO: In this case, should we default to conda java?
            return "java"

    # Don't do anything in a pip environment
    # TODO: Some debug logging would be good here
    return "java"


def get_gateway():
    """
    Launches the Java gateway server on rank 0 if not already initialized,
    and returns the gateway for rank 0.
    Has no effect and returns None when called on any rank other than rank 0.

    Note that whenever this function is called, it must be called on every rank, so that errors
    are properly propagated, and we don't hang.
    """
    global gateway

    failed = False
    msg = ""

    if bodo.get_rank() == 0 and gateway is None:
        cur_file_path = os.path.dirname(os.path.abspath(__file__))
        # Get the jar path
        full_path = os.path.join(cur_file_path, "jars/bodosql-executable.jar")

        # Die on exit will close the gateway server when this python process exits or is killed.
        try:
            out_fd = sys.stdout
            err_fd = sys.stderr

            # Jupyter does not support writing to stderr
            # https://discourse.jupyter.org/t/how-to-know-from-python-script-if-we-are-in-jupyterlab/23993/4
            if bodo.spawn.utils.is_jupyter_on_windows():
                out_fd = None
                err_fd = None

            java_path = get_java_path()
            port_no = cast(
                int,
                launch_gateway(
                    jarpath=full_path,
                    java_path=java_path,
                    die_on_exit=True,
                    redirect_stdout=out_fd,
                    redirect_stderr=err_fd,
                    # Required by Arrow: https://arrow.apache.org/docs/java/install.html
                    javaopts=["--add-opens=java.base/java.nio=ALL-UNNAMED"],
                ),
            )
            gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port_no))
        except Exception as e:
            msg = f"Error when launching the BodoSQL JVM. {str(e)}"
            failed = True

    comm = MPI.COMM_WORLD
    failed = comm.bcast(failed)
    msg = comm.bcast(msg)
    if failed:
        raise Exception(msg)
    return gateway


def configure_java_logging(level: int):
    """Configure the java logging based on the given logging level.
    This just turns the loggers on/off, but in the future should ideally
    enable support for changing the logging destination in Java based
    upon the default verbose logger.

    Args:
        level (int): The verbose level.
    """
    # Java logging is only on rank 0
    if bodo.get_rank() == 0:
        from bodosql.imported_java_classes import JavaEntryPoint

        JavaEntryPoint.configureJavaLogging(level)
