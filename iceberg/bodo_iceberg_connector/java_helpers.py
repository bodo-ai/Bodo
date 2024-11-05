import typing as pt

from bodo_iceberg_connector.py4j_support import (
    get_hadoop_conf_class,
    get_system_class,
)

if pt.TYPE_CHECKING:
    from py4j.java_gateway import JavaObject


def set_system_property(key: str, value: str):
    """Run Java's `System.setProperty` Function on the Internal Java Process"""
    System = get_system_class()
    System.setProperty(key, value)  # type: ignore


def build_hadoop_conf(d: dict[str, str]):
    """Build a Hadoop Configuration Object on the Internal Java Process"""
    Configuration = get_hadoop_conf_class()
    conf: JavaObject = Configuration()  # type: ignore
    for k, v in d.items():
        conf.set(k, v)  # type: ignore

    return conf
