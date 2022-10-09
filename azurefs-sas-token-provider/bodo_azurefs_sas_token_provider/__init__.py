# Add the jars to the CLASSPATH
import os

# This should match the path in setup.py (look for corresponding comment
# in the build_libs function).
# Note that it's important for it to be the exact location of the jar,
# and not just the directory, otherwise the JVM might not pick it up.
JAR_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "jars",
    "bodo-azurefs-sas-token-provider.jar",
)
os.environ["CLASSPATH"] = f"{JAR_PATH}:" + os.environ.get("CLASSPATH", "")
