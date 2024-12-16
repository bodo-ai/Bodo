import os

# Create a temp dir name that will be used for writing temporary
# core-site files for ADLS read/writes. At this time, this is
# used by Snowflake write when writing to an ADLS backed Snowflake
# stage. This can also be used by users for regular ADLS read/writes.
# This is *required* when both Snowflake write and regular ADLS I/O
# need to happen as part of the same process since Hadoop picks the
# first core-site.xml it finds in the CLASSPATH (but the contents
# of the core-site.xml file can be modified dynamically once it's
# picked). We are using a LazyTemporaryDirectory, i.e. the name is
# decided and synchronized across all rank (since is_parallel=True) now,
# but the directory needs to be initialized when there's an actual
# use for it. This is to avoid any unnecessary filesystem operations.
# Note that the initialization step must be called from all ranks
# since is_parallel=True, else it will lead to a hang. Initialization
# is idempotent, and can be safely repeated.
from bodo.io.lazy_tempdir import LazyTemporaryDirectory

HDFS_CORE_SITE_LOC_DIR = LazyTemporaryDirectory(is_parallel=True)
HDFS_CORE_SITE_LOC = os.path.join(HDFS_CORE_SITE_LOC_DIR.name, "core-site.xml")

# Add this location to the front of the CLASSPATH so that when/if we create
# a core-site, it gets picked up. If we never create one ourselves, Hadoop
# will continue looking in the CLASSPATH, so there shouldn't be any negative
# side-effects.
os.environ["CLASSPATH"] = f"{HDFS_CORE_SITE_LOC_DIR.name}:" + os.environ.get(
    "CLASSPATH", ""
)
# Also expose the location as an evironment variable. This will be used
# by bodo_azurefs_sas_token_provider to locate the SAS token file location.
# See BodoSASTokenProvider.java.
os.environ["BODO_HDFS_CORE_SITE_LOC_DIR"] = HDFS_CORE_SITE_LOC_DIR.name

# Try to import our SAS Token Provider implementation which will
# add the jar to the CLASSPATH from the onset, avoiding issues
# with interleaving with other JVM related operations. e.g.
# if we don't do this import, and the user doesn't either in their code,
# then if we read from / write to ADLS before a Snowflake write operation,
# the JVM would already be initialized from the first operation without
# the jar for the SAS token provider class in the CLASSPATH, and then
# the Snowflake write would fail (even though this import will be done
# as part of Snowflake write). If this import creates issues or
# degrades performance, we could require it to be done by the user
# instead.
try:
    import bodo_azurefs_sas_token_provider  # noqa: F401  isort:skip
except ImportError:
    pass

# Try set the core site path for the iceberg connector.
# Note: This assumes importing the connector doesn't start the JVM
try:
    import bodo_iceberg_connector

    bodo_iceberg_connector.set_core_site_path(HDFS_CORE_SITE_LOC)
except ImportError:
    pass
