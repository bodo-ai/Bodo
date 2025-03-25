"""
Contains information used to access the Java package via py4j.
"""

import os
import sys
import typing as pt
import warnings

from py4j.java_collections import ListConverter, MapConverter
from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

if pt.TYPE_CHECKING:
    from py4j.java_gateway import JavaClass


# The gateway object used to communicate with the JVM.
gateway: JavaGateway | None = None

# Java Classes used by the Python Portion
CLASSES: dict[str, "JavaClass"] = {}

# Dictionary mapping table info -> Reader obj
catalog_dict = {}


# Core site location.
_CORE_SITE_PATH = ""


def set_core_site_path(path: str):
    global _CORE_SITE_PATH
    _CORE_SITE_PATH = path


def get_core_site_path() -> str:
    return _CORE_SITE_PATH


def get_java_path() -> str:
    """
    Ensure that the Java executable Py4J uses is the OpenJDK package
    installed in the current conda environment
    """

    # Currently inside a conda subenvironment
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
                    "It is recommended that you use OpenJDK v11 from Conda with the Bodo Iceberg Connector. To do so, first run\n"
                    "    conda install openjdk=11 -c conda-forge\n"
                    "and then reactivate your environment via\n"
                    f"    conda deactivate && conda activate {conda_prefix}"
                )
            return os.path.join(java_home, "bin", "java")

        else:
            warnings.warn(
                "$JAVA_HOME is currently unset. This occurs when OpenJDK is not installed in your conda environment or when your environment has recently changed but not reactivated. The Bodo Iceberg Connector will default to using you system's Java."
                "It is recommended that you use OpenJDK v11 from Conda with the Bodo Iceberg Connector. To do so, first run\n"
                "    conda install openjdk=11 -c conda-forge\n"
                "and then reactivate your environment via\n"
                f"    conda deactivate && conda activate {conda_prefix}"
            )
            # TODO: In this case, should we default to conda java?
            return "java"

    # Don't do anything in a pip environment
    # TODO: Some debug logging would be good here
    return "java"


def launch_jvm() -> JavaGateway:
    """
    Launches Py4J's Java Gateway server if it is not already running.
    The gateway server manages a backend Java process that has access to
    various Iceberg, Hive, and Hadoop Java libraries.

    Returns:
        The active Py4J java gateway instance
    """
    global CLASSES, gateway

    if gateway is None:
        cur_file_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(cur_file_path, "jars", "bodo-iceberg-reader.jar")

        # Die on exit will close the gateway server when this python process exits or is killed.
        # We don't need to specify a classpath here, as the executable JAR has a baked in default
        # classpath, which will point to the folder that contains all the needed dependencies.
        java_path = get_java_path()
        print(f"Launching JVM with Java executable: {java_path}", file=sys.stderr)

        gateway_port = pt.cast(
            int,
            launch_gateway(
                classpath=full_path,
                java_path=java_path,
                redirect_stderr=sys.stderr,
                redirect_stdout=sys.stdout,
                die_on_exit=True,
            ),
        )

        # TODO: Test out auto_convert=True for converting collections (esp lists)
        # https://www.py4j.org/advanced_topics.html#collections-conversion
        gateway = JavaGateway(gateway_parameters=GatewayParameters(port=gateway_port))
        CLASSES.clear()
        catalog_dict.clear()

    # NOTE: currently, gateway.entry_point returns a non existent java object. Additionally, the
    # "main" function of the IcebergReadEntryPoint never seems to run. This is very strange.
    # I suspect it may have something to do with the fact that we don't store any state in the
    # gateway object class, and/or the fact that I'm generating a classpath aware executable JAR, as
    # opposed to BodoSQl where I'm packaging it as a singular executable JAR with all dependencies
    # included. In any case, it doesn't actually impact us, so we can safely ignore it.
    return gateway


def get_class_wrapper(
    class_name: str, class_inst: pt.Callable[[JavaGateway], "JavaClass"]
):
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


def convert_dict_to_java(python_dict: dict):
    """
    Converts a Python dictionary to a Java Map
    """
    gateway = launch_jvm()
    return MapConverter().convert(python_dict, gateway._gateway_client)


def convert_list_to_java(vals: list[pt.Any]):
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
        lambda gateway: gateway.jvm.com.bodo.iceberg.LiteralConverters,  # type: ignore
    )()


# TODO: Better way than this?
# Built-in Classes
get_system_class = get_class_wrapper(
    "SystemClass",
    lambda gateway: gateway.jvm.System,  # type: ignore
)

# Iceberg Classes
get_iceberg_schema_class = get_class_wrapper(
    "IcebergSchemaClass",
    lambda gateway: gateway.jvm.org.apache.iceberg.Schema,  # type: ignore
)
get_iceberg_type_class = get_class_wrapper(
    "IcebergTypeClass",
    lambda gateway: gateway.jvm.org.apache.iceberg.types.Types,  # type: ignore
)
get_hadoop_conf_class = get_class_wrapper(
    "ConfigurationClass",
    lambda gateway: gateway.jvm.org.apache.hadoop.conf.Configuration,  # type: ignore
)

# Bodo Classes
get_bodo_iceberg_handler_class = get_class_wrapper(
    "BodoIcebergHandlerClass",
    lambda gateway: gateway.jvm.com.bodo.iceberg.BodoIcebergHandler,  # type: ignore
)
get_data_file_class = get_class_wrapper(
    "DataFileClass",
    lambda gateway: gateway.jvm.com.bodo.iceberg.DataFileInfo,  # type: ignore
)
get_bodo_arrow_schema_utils_class = get_class_wrapper(
    "BodoArrowSchemaUtil",
    lambda gateway: gateway.jvm.com.bodo.iceberg.BodoArrowSchemaUtil,  # type: ignore
)

# Bodo Filter Pushdown Classes
get_column_ref_class = get_class_wrapper(
    "ColumnRefClass",
    lambda gateway: gateway.jvm.com.bodo.iceberg.filters.ColumnRef,  # type: ignore
)
get_array_const_class = get_class_wrapper(
    "ArrayConstClass",
    lambda gateway: gateway.jvm.com.bodo.iceberg.filters.ArrayConst,  # type: ignore
)
get_filter_expr_class = get_class_wrapper(
    "FilterExprClass",
    lambda gateway: gateway.jvm.com.bodo.iceberg.filters.FilterExpr,  # type: ignore
)


def get_catalog(conn_str: str, catalog_type: str):
    """
    Get the catalog object from the global cache
    """
    conn_str = conn_str.removeprefix("iceberg+")
    reader_class = get_bodo_iceberg_handler_class()
    if conn_str not in catalog_dict:
        created_core_site = get_core_site_path()
        # Use the defaults if the user didn't override the core site.
        core_site = created_core_site if os.path.exists(created_core_site) else ""
        catalog_dict[conn_str] = reader_class(conn_str, catalog_type, core_site)
    return catalog_dict[conn_str]
