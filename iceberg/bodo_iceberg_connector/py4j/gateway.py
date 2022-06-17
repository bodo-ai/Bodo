"""
APIs used to launch a connection to the Java py4j gateway server.
This should be used whenever performing a compile time or runtime
API to validate that there is a connection.

We currently assume that since we are only gathering metadata, all
of these APIs.
"""

import os
import subprocess
import time

from bodo_iceberg_connector.config import DEFAULT_PORT
from py4j.java_gateway import GatewayParameters, JavaGateway

# Dictionary of port number -> gateway
gateway = {}
# Dictionary of port number -> dictionary key (table info) -> reader obj
# We use two levels to support shutting down/deleting the gateway.
iceberg_java_table_readers = {}


def launch_default_java_process_async():
    """
    Launches the Java process used to read from Iceberg
    with the default port in an async manner. This is intended
    to be done outside of the regular Bodo read.
    """
    return launch_java_process_async(DEFAULT_PORT)


def launch_java_process_async(port):
    """
    Launches the Java process used to read from Iceberg
    with the given port in an async manner. This is intended
    to be done outside of the regular Bodo read.
    """
    iceberg_poc_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the jar path
    full_path = (
        iceberg_poc_dir + "/../iceberg-java/target/iceberg-java-1.0-SNAPSHOT.jar"
    )
    jar_command = [f"java -jar {full_path} {port}"]
    java_process = subprocess.Popen(jar_command, shell=True)
    # Wait 1 second for the jar to execute.
    time.sleep(1)
    return java_process


def get_py4j_gateway(port):
    """
    Returns a py4j gateaway used to create Java objects.
    """
    if port not in gateway:
        gateway[port] = JavaGateway(gateway_parameters=GatewayParameters(port=port))
    return gateway[port]


def get_iceberg_java_table_reader(port, warehouse, schema, table):
    """
    Returns a Java object for accessing an Iceberg Table
    """
    gateway = get_py4j_gateway(port)
    if port not in iceberg_java_table_readers:
        iceberg_java_table_readers[port] = {}
    reader_dict = iceberg_java_table_readers[port]
    key = (warehouse, schema, table)
    if key not in reader_dict:
        reader_dict[key] = gateway.getBodoIcebergReader(warehouse, schema, table)
    return reader_dict[key]


def shutdown_gateway(port):
    """
    Shuts down a gateway for a given port
    if it exists.
    """
    if port in gateway:
        gateway[port].shutdown()
        del gateway[port]
    if port in iceberg_java_table_readers:
        del iceberg_java_table_readers[port]


def shutdown_default_gateway():
    """
    Shuts down a gateway for the default port
    if it exists.
    """
    shutdown_gateway(DEFAULT_PORT)
