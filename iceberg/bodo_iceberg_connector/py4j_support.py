"""
Contains information used to access the Java package via py4j.
"""
import atexit
import os
import sys

from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

import bodo

# The gateway object used to communicate with the JVM.
gateway = None
# Java Classes used by the connector
BodoReaderIcebergReaderClass = None
LinkedListClass = None
OpEnumClass = None
LiteralConverterClass = None


from bodo_iceberg_connector.config import DEFAULT_PORT

# Dictionary mapping table info -> Reader obj
table_dict = {}


def launch_jvm():
    """
    Launches the gateway server, if it is not already running, and returns
    a gateway object.
    """
    global gateway
    assert (
        bodo.get_rank() == 0
    ), f"Only rank 0 should launch the JVM. Rank {bodo.get_rank()} is trying to launch the JVM."

    if gateway is None:
        cur_file_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(
            cur_file_path, "iceberg-java", "target", "bodo-iceberg-reader.jar"
        )

        # TODO: we do not currently specify a port here. In this case, launch_gateway launches
        # using an arbitrary free ephemeral port. Launching with specified port can lead to
        # hangs if the port is already in use, and has caused errors on CI, so we are ommiting it
        # for now

        # Die on exit will close the gateway server when this python process exits or is killed.
        # We don't need to specify a classpath here, as the executable JAR has a baked in default
        # classpath, which will point to the folder that contains all the needed dependencies.
        gateway_port = launch_gateway(
            jarpath=full_path,
            redirect_stderr=sys.stderr,
            redirect_stdout=sys.stdout,
            die_on_exit=True,
        )
        gateway = JavaGateway(gateway_parameters=GatewayParameters(port=gateway_port))

    # NOTE: currently, gateway.entry_point returns a non existent java object. Additionally, the
    # "main" function of the IcebergReadEntryPoint never seems to run. This is very strange.
    # I suspect it may have somthing to do with the fact that we don't store any state in the
    # gateway object class, and/or the fact that I'm generating a classpath aware executable JAR, as
    # opposed to BodoSQl where I'm packaging it as a sigular executable JAR with all dependencies
    # included. In any case, it doesn't actually impact us, so we can safely ignore it.
    return gateway


def get_literal_converter_class():
    """
    Wrapper around getting the LiteralConverterClass on first request. py4j will often coerce primitive
    java types (float, str, etc) into their equivalent python counterpart, which can make creating literals.
    of a specific type difficult (float vs double, int vs long, etc). This literal converter class helps to get around that

    """
    global LiteralConverterClass

    # We may need to launch the JVM if we haven't loaded the class.
    if LiteralConverterClass is None:
        launch_jvm()
        LiteralConverterClass = gateway.jvm.com.bodo.iceberg.LiteralConverters

    return LiteralConverterClass


def get_bodo_iceberg_reader_class():
    """
    Wrapper around getting the constructor to
    load the BodoReaderIcebergReaderClass on first request.
    """
    global BodoReaderIcebergReaderClass

    # We may need to launch the JVM if we haven't loaded the class.
    if BodoReaderIcebergReaderClass is None:
        launch_jvm()
        BodoReaderIcebergReaderClass = gateway.jvm.com.bodo.iceberg.BodoIcebergReader

    return BodoReaderIcebergReaderClass


def get_linkedlist_class():
    """
    Wrapper around getting the constructor to
    load the LinkedListClass on first request.
    """
    global LinkedListClass

    # We may need to launch the JVM if we haven't loaded the class.
    if LinkedListClass is None:
        launch_jvm()
        LinkedListClass = gateway.jvm.java.util.LinkedList
    return LinkedListClass


def get_java_list(val):
    """
    Converts a Python list to a Java ArrayList
    """
    return gateway.jvm.java.util.ArrayList(val)


def get_op_enum_class():
    """
    Wrapper around getting the constructor to
    load the OpEnumClass on first request.
    """
    global OpEnumClass

    # We may need to launch the JVM if we haven't loaded the class.
    if OpEnumClass is None:
        launch_jvm()
        OpEnumClass = gateway.jvm.com.bodo.iceberg.OpEnum
    return OpEnumClass


def get_iceberg_java_table_reader(warehouse, schema, table):
    reader_class = get_bodo_iceberg_reader_class()
    key = (warehouse, schema, table)
    if key not in table_dict:
        table_dict[key] = reader_class(warehouse, schema, table)
    return table_dict[key]
