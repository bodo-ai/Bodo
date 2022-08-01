"""
APIs used to launch and connect to the Java py4j calcite gateway server.
"""


import os

import bodo
from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

# This gateway is always None on every rank but rank 0,
# it is initialized by the get_gateway call.
gateway = None

from bodo.libs.distributed_api import bcast_scalar


def get_gateway():
    """
    Launches the Java gateway server on rank 0 if not already intialized,
    and returns the gateway for rank 0.
    Has no effect and returns None when called on any rank other than rank 0.

    Note that whenever this function is called, it must be called on every rank, so that errors
    are properly propgated, and we don't hang.
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
            port_no = launch_gateway(jarpath=full_path, die_on_exit=True)
            gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port_no))
        except Exception as e:
            msg = f"Error when launching the BodoSQL JVM. {str(e)}"

    failed = bcast_scalar(failed)
    msg = bcast_scalar(msg)
    if failed:
        raise Exception(msg)
    return gateway
