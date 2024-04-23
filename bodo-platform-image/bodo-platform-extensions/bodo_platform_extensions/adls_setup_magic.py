import os

from IPython.core.magic import (
    line_magic,
    magics_class,
)
from IPython.core import magic_arguments

import dicttoxml
from xml.dom.minidom import parseString

from .magics_base import PlatformMagicsBase
from .helper import copy_file_to_all_nodes


@magics_class
class ADLSSetupMagics(PlatformMagicsBase):
    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        "--authtype",
        type=str,
        default="OAuth",
        help="The auth type to use. Sets the 'fs.azure.account.auth.type' property. Default: 'OAuth'.",
    )
    @magic_arguments.argument(
        "--provider",
        type=str,
        default="org.apache.hadoop.fs.azurebfs.oauth2.MsiTokenProvider",
        help="The auth provider type to use. Sets the 'fs.azure.account.oauth.provider.type' property. "
        "Default: 'org.apache.hadoop.fs.azurebfs.oauth2.MsiTokenProvider'.",
    )
    @magic_arguments.argument(
        "--impl",
        type=str,
        default="org.apache.hadoop.fs.azurebfs.AzureBlobFileSystem",
        help="The ABFS implementation to use. Sets the 'fs.abfs.impl' property. "
        "Default: 'org.apache.hadoop.fs.azurebfs.AzureBlobFileSystem'.",
    )
    @magic_arguments.argument(
        "--identity",
        type=str,
        default=None,
        help="The client-id for the Managed Identity you want to use. "
        "Sets the 'fs.azure.account.oauth2.client.id' property. It's not set if no value is provided.",
    )
    @magic_arguments.argument(
        "--timeout", type=int, default=None, help="number of seconds to timeout after."
    )
    @magic_arguments.argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print information such as the mpiexec commands being run.",
    )
    @magic_arguments.argument(
        "--hadoop_home",
        type=str,
        default=None,
        help="Location of hadoop installation. By default it uses the environment 'HADOOP_HOME. "
        "If not provided and the environment is not set either, it will throw an error. "
        "The credentials are written to: $HADOOP_HOME/etc/hadoop/core-site.xml",
    )
    def setup_adls(self, line="", local_ns=None):
        """
        Bodo IPython Magic to setup ADLS credentials in core-site.xml on all instances.
        Note that this will replace the existing file with a new file.
        """
        # Parse the arguments
        args = magic_arguments.parse_argstring(self.setup_adls, line)

        adls_settings = [
            {
                "name": "fs.azure.account.auth.type",
                "value": args.authtype,
                "description": "Use OAuth authentication",
            },
            {
                "name": "fs.azure.account.oauth.provider.type",
                "value": args.provider,
                "description": "Use MSI for issuing OAuth tokens",
            },
            {
                "name": "fs.abfs.impl",
                "value": "org.apache.hadoop.fs.azurebfs.AzureBlobFileSystem",
            },
        ]
        if args.identity:
            adls_settings.append(
                {
                    "name": "fs.azure.account.oauth2.client.id",
                    "value": args.identity,
                    "description": "Optional Client ID",
                }
            )

        # Conver adls_settings into an XML string
        template_managed_identity = dicttoxml.dicttoxml(
            adls_settings,
            custom_root="configuration",
            attr_type=False,
            item_func=lambda x: "property",
        )
        dom = parseString(template_managed_identity)
        template_managed_identity = dom.toprettyxml()

        if args.verbose:
            print("Output File:\n", template_managed_identity)

        HADOOP_HOME = args.hadoop_home
        if HADOOP_HOME is None:
            HADOOP_HOME = os.environ.get("HADOOP_HOME", "")
            if HADOOP_HOME == "":
                self.__display_error_message__(
                    "'--hadoop_home' not provided and 'HADOOP_HOME' environment variable isn't set either."
                )
                return
        core_site_file_loc = os.path.join(HADOOP_HOME, "etc/hadoop/core-site.xml")

        with open(core_site_file_loc, "w") as f:
            f.write(template_managed_identity)

        num_timed_out, num_errored_out, total_nodes = copy_file_to_all_nodes(
            core_site_file_loc,
            core_site_file_loc,
            verbose=args.verbose,
            timeout=args.timeout,
        )

        if num_timed_out > 0:
            self.__display_error_message__(
                f"Timed out on {num_timed_out}/{total_nodes} nodes!"
            )
        if num_errored_out > 0:
            self.__display_error_message__(
                f"Errored out on {num_timed_out}/{total_nodes} nodes!"
            )
