"""
Contains information used to access the Java package via py4j.
"""
import os
import sys
from typing import Any, List

from mpi4py import MPI
from py4j.java_collections import ListConverter
from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

# The gateway object used to communicate with the JVM.
gateway = None

# Java Classes used by the Python Portion
CLASSES = {}

# Dictionary mapping table info -> Reader obj
table_dict = {}


def launch_jvm():
    """
    Launches the gateway server, if it is not already running, and returns
    a gateway object.
    """
    global CLASSES, gateway

    rank = MPI.COMM_WORLD.Get_rank()
    assert (
        rank == 0
    ), f"Rank {rank} is trying to launch the JVM. Only rank 0 should launch it."

    if gateway is None:
        cur_file_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(cur_file_path, "jars", "bodo-iceberg-reader.jar")

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

        # TODO: Test out auto_convert=True for converting collections (esp lists)
        # https://www.py4j.org/advanced_topics.html#collections-conversion
        gateway = JavaGateway(gateway_parameters=GatewayParameters(port=gateway_port))
        CLASSES = {}

    # NOTE: currently, gateway.entry_point returns a non existent java object. Additionally, the
    # "main" function of the IcebergReadEntryPoint never seems to run. This is very strange.
    # I suspect it may have somthing to do with the fact that we don't store any state in the
    # gateway object class, and/or the fact that I'm generating a classpath aware executable JAR, as
    # opposed to BodoSQl where I'm packaging it as a sigular executable JAR with all dependencies
    # included. In any case, it doesn't actually impact us, so we can safely ignore it.
    return gateway


def get_class_wrapper(class_name: str, class_inst):
    """
    Wrapper around getting the constructor for a specified Java class
    on first request, and caching the rest.
    """

    def impl():
        # We may need to launch the JVM if we haven't loaded the class.
        if class_name not in CLASSES:
            gateway = launch_jvm()
            CLASSES[class_name] = class_inst(gateway)
        return CLASSES[class_name]

    return impl


def convert_list_to_java(vals: List[Any]):
    """
    Converts a Python list to a Java ArrayList
    """
    gateway = launch_jvm()
    return ListConverter().convert(vals, gateway._gateway_client)


def get_literal_converter_class():
    """
    Wrapper around getting the LiteralConverterClass on first request. py4j will often coerce primitive
    java types (float, str, etc) into their equivalent python counterpart, which can make creating literals.
    of a specific type difficult (float vs double, int vs long, etc). This literal converter class helps to get around that
    """
    return get_class_wrapper(
        "LiteralConverterClass",
        lambda gateway: gateway.jvm.com.bodo.iceberg.LiteralConverters,
    )()


# TODO: Better way than this?
# Built-in Classes
get_linkedlist_class = get_class_wrapper(
    "LinkedListClass",
    lambda gateway: gateway.jvm.java.util.LinkedList,
)

# Iceberg Classes
get_iceberg_schema_class = get_class_wrapper(
    "IcebergSchemaClass",
    lambda gateway: gateway.jvm.org.apache.iceberg.Schema,
)
get_iceberg_type_class = get_class_wrapper(
    "IcebergTypeClass",
    lambda gateway: gateway.jvm.org.apache.iceberg.types.Types,
)

# Bodo Classes
get_bodo_iceberg_handler_class = get_class_wrapper(
    "BodoIcebergHandlerClass",
    lambda gateway: gateway.jvm.com.bodo.iceberg.BodoIcebergHandler,
)
get_op_enum_class = get_class_wrapper(
    "OpEnumClass",
    lambda gateway: gateway.jvm.com.bodo.iceberg.OpEnum,
)
get_data_file_class = get_class_wrapper(
    "DataFileClass",
    lambda gateway: gateway.jvm.com.bodo.iceberg.DataFileInfo,
)


def get_java_table_handler(conn_str: str, catalog_type: str, db_name: str, table: str):
    reader_class = get_bodo_iceberg_handler_class()
    key = (conn_str, db_name, table)
    if key not in table_dict:
        table_dict[key] = reader_class(conn_str, catalog_type, db_name, table)
    return table_dict[key]
