import json
from signal import SIGKILL
from ipykernel.ipkernel import IPythonKernel
import ipyparallel as ipp

from .platform_hostfile_update import update_hostfile
from .utils import BColor, colorize_output

IPYPARALLEL_LINE_MAGICS = ("px", "autopx", "pxconfig", "pxresult")
IPYPARALLEL_CELL_MAGICS = ("px",)
BODO_PLATFORM_CUSTOM_LINE_MAGICS = (
    "%pconda",
    "%ppip",
    "%psh",
    f"%setup_adls",
    f"%update_hostfile",
)

IPYPARALLEL_MAGICS = tuple(
    [f"%{m}" for m in IPYPARALLEL_LINE_MAGICS]
    + [f"%%{m}" for m in IPYPARALLEL_CELL_MAGICS]
)

BODO_SQL_MAGICS = (
    f"%sql",
    f"%%sql",
)

# Should be kept in sync with https://github.com/Bodo-inc/bodo-platform-jupyterlab-extension/blob/master/src/types.ts
class SupportedLanguages:
    PARALLEL_PYTHON = "Parallel-Python"
    PYTHON = "Python"
    SQL = "SQL"


class IPyParallelKernel(IPythonKernel):
    banner = "IPyParallel Kernel"

    def start(self):
        super().start()
        self.ipyparallel_cluster_started = False
        self.ipyparallel_cluster = None

    def ipyparallel_magics_already_registered(self) -> bool:
        # Check if any IPyParallel magics are already registered
        return any(
            [
                x in self.shell.magics_manager.magics["line"]
                for x in IPYPARALLEL_LINE_MAGICS
            ]
            + [
                x in self.shell.magics_manager.magics["cell"]
                for x in IPYPARALLEL_CELL_MAGICS
            ]
        )

    def start_ipyparallel_cluster(self):
        if not self.ipyparallel_cluster_started:
            self.log.info("Updating Hostfile...")
            update_hostfile(self.log)

            self.log.info("Starting IPyParallel Cluster...")

            try:
                # Use our custom engine set launcher
                self.ipyparallel_cluster = ipp.Cluster(
                    engines="bodo"
                )  # Config is taken from ipcluster_config.py
                self.ipyparallel_rc = self.ipyparallel_cluster.start_and_connect_sync()
                self.ipyparallel_view = self.ipyparallel_rc.broadcast_view()
                self.ipyparallel_view.block = True
                self.ipyparallel_view.activate()
            except Exception as e:
                self.log.error(
                    "Something went wrong while trying to start the IPyParallel cluster..."
                )
                self.log.error(f"Error: {e}")
                self.log.info("Shutting Cluster down...")
                # Cluster might have been started, if so then stop it and remove any
                # lingering processes
                if self.ipyparallel_view is not None:
                    # In some cases just stop_cluster_sync left hanging engine processes so view.shutdown was added
                    self.ipyparallel_view.shutdown(hub=True)
                if self.ipyparallel_cluster is not None:
                    self.ipyparallel_cluster.stop_cluster_sync()
                raise e
            else:
                self.ipyparallel_cluster_started = True

    def _log_message(self, message: str):
        stream_content = {"name": "stderr", "text": message}
        self.send_response(self.iopub_socket, "stream", stream_content)

    async def do_execute(
        self,
        code: str,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
    ):
        """
        Thin layer on top of do_execute to handle IPyParallel cluster lifecycle,
        and slight modifications to "code" based on it.
        Assumptions about "code":
        - The "code" cannot be `%autopx`, since those cases are caught by `execute_request`
          This is because we have a PARALLEL_PYTHON language mode, which eliminates the
          need of autopx. `execute_request` adds `%%px` to anything that must be parallelized,
          e.g. when running in Parallel-Python or SQL mode. Therefore, this function
          doesn't make any adjustments to the code, such as stripping of %%px, etc.

        Args:
            code (str): Code to execute (after cleanup in `execute_request`)
            The rest of the arguments are passed through to super as is.
        """

        # %autopx is not supported, so convert code to "pass"
        # and show users a warning.
        if code.strip().startswith(f"%autopx"):
            code = "pass"
            message = colorize_output(
                BColor.FAIL,
                f"Running `%autopx` is not supported.",
            )
            self._log_message(message)

        # TODO: Handle %%px (no code) case. Ideally this should be a nop.

        # Start the IPyParallel cluster if any IPyParallel
        # magic is used
        if (
            code.startswith(IPYPARALLEL_MAGICS)
            # This is already checked in ipyparallel_magics_already_registered,
            # so it can be a bit redundant, however, it should be cheaper
            # than checking if magics are registered, so we'll get short-circuiting
            # benefits.
            and not self.ipyparallel_cluster_started
            # If magics are already registered, that means user has started an
            # IPyParallel cluster manually, so we shouldn't start one ourselves.
            # This lets users use this kernel as a direct replacement for a regular
            # kernel.
            and not self.ipyparallel_magics_already_registered()
        ):
            try:
                self.start_ipyparallel_cluster()
            except Exception:
                # Don't run any code since cluster creation failed
                code = "pass"

        # If the engines are not running, e.g. they were previously started but killed,
        # and code starts with "px" or "autopx", rather than executing the code,
        # we throw an error
        if self.ipyparallel_cluster_started and code.startswith(IPYPARALLEL_MAGICS):
            # If engines are not running, e.g. they previously errored out with an exit code, prompt user to restart the kernel and return
            # We use a try-except block here to make sure code doest throw an error for instances when
            # ipyparallel_cluster does not have a property engine_set. The other alternative would be to use a getter
            try:
                if (
                    not self.ipyparallel_cluster.engine_set.running
                    # still allow users to run %pxresult.
                    and not code.startswith("%pxresult")
                    # We don't need to handle %autopx since that's not
                    # allowed in general.
                ):
                    message = colorize_output(
                        BColor.FAIL,
                        "IPyParallel Cluster engines previously exited with an error. Please restart the kernel and try again",
                    )
                    self._log_message(message)
                    # Don't run any code since cluster is in a failed state
                    code = "pass"
            except:
                pass

        return await super().do_execute(
            code=code,
            silent=silent,
            store_history=store_history,
            user_expressions=user_expressions,
            allow_stdin=allow_stdin,
        )

    async def stop_ipyparallel_cluster(self):
        if self.ipyparallel_cluster_started:
            self.log.info("Stopping IPyParallel Cluster...")

            # If a MPI process is hanging/waiting for other processes
            # the normal IPP shutdown process leaves zombie engines
            # which can create OOM issues.
            # Sending SIGKILL to the engines ensures the processes are
            # stopped and their resources are released.
            # passing signal's int value instead of signal object since Slurm launcher
            # cannot handle objects (passed to scancel).
            await self.ipyparallel_cluster.signal_engines(SIGKILL.value)
            # Stop the controller separately since the engines should be
            # removed already.
            await self.ipyparallel_cluster.stop_controller()

            # If a cluster is left in a broken state the cluster file
            # isn't always removed so manually remove it
            self.ipyparallel_cluster.remove_cluster_file()

            try:
                # Just in case the above methods don't work
                self.ipyparallel_cluster.stop_cluster_sync()
            except FileNotFoundError:
                # Cluster.stop_engines will throw FileNotFoundError if logs are already removed.
                # Stop_cluster_sync calls cluster.stop_engines.
                pass

    async def do_shutdown(self, restart):
        await self.stop_ipyparallel_cluster()
        return super().do_shutdown(restart)

    @staticmethod
    def handle_sql_input(code: str, metadata: dict):
        """
        Parse and validate SQL input.
        In case of valid input, we add the `%%sql` magic with the
        catalog and output variable details to the code.
        In case of invalid input, we convert the code to "pass"
        and return a (colorized) message that should be
        displayed to the users.

        Args:
            code (str): Code to parse and validate
            metadata (dict): Associated metadata for this execution
                request. In particular, we will use this to
                extract the catalog details.

        Returns:
            code (str): Modified code that should be executed.
            message (str or None): Colorized message to display to users in case
                of invalid input.
        """

        message = None
        if code.startswith(IPYPARALLEL_MAGICS):
            message = colorize_output(
                BColor.FAIL,
                "Cannot use IPyParallel magics in SQL mode. Please use Parallel-Python mode instead.",
            )
            # We don't want to run the actual code, so replace it with `pass`
            code = "pass"
        elif code.startswith(BODO_SQL_MAGICS):
            message = colorize_output(
                BColor.FAIL,
                "Cannot use SQL magics in SQL mode. Either use Parallel-Python mode or write the SQL query directly in SQL mode.",
            )
            code = "pass"
        elif code.startswith(BODO_PLATFORM_CUSTOM_LINE_MAGICS):
            message = colorize_output(
                BColor.FAIL,
                "Cannot use Platform magics in SQL mode. Use Parallel-Python or Python mode instead.",
            )
            code = "pass"
        else:
            # Else we assume that the code is valid SQL code.
            # In case it isn't, the behavior is undefined.
            catalog = metadata.get("bodo-catalog")
            # Handle the case where 'catalog' is "" or None. The "" handling is
            # essential since json.loads("") would error out.
            if not catalog:
                catalog = "{}"
            catalog_name = json.loads(catalog).get("name")
            if not catalog_name:
                # If catalog_name is not specified, the `%%sql` magic would complain,
                # but without a very intuitive error for users. So we show a warning
                # and modify the code to be "pass" so there's no side effects.
                message = colorize_output(
                    BColor.WARNING,
                    "Please select a catalog before executing a SQL cell.",
                )
                code = "pass"
            else:
                # We store the output in `LAST_SQL_OUTPUT`.
                # Users can choose to take this and store it in some other
                # variable if they want.
                # We don't create a unique/random name every time since
                # that would fill up the memory. This way, if the user
                # doesn't want to keep the output around, python
                # garbage collection will get rid of it.
                code = (
                    f"%%px\n"
                    + f"%%sql --catalog_name {catalog_name} --output LAST_SQL_OUTPUT\n"
                    + code
                )

        return code, message

    @staticmethod
    def handle_python_input(code, metadata):
        """
        Parse and validate regular Python (not Parallel Python) input.
        If the code uses a IPyParallel magic, we show users an error
        message (and convert the code to "pass"). Else, we pass the
        code as is.

        Args:
            code (str): Code to parse and validate
            metadata (dict): Associated metadata for this execution
                request. This is unused and only included for
                standardization reasons

        Returns:
            code (str): Modified code that should be executed.
            message (str or None): Colorized message to display to users in case
                of invalid input.
        """
        message = None
        if code.startswith(IPYPARALLEL_MAGICS):
            message = colorize_output(
                BColor.FAIL,
                "Cannot use IPyParallel magics in Python mode. Please use Parallel-Python mode instead.",
            )
            code = "pass"
        # Else pass it through as is
        return code, message

    @staticmethod
    def handle_parallel_python_input(code, metadata):
        """
        Parse and validate regular Parallel Python (not regular Python) input.
        If the code is "%autopx", we show users a warning and ignore the input
        (convert code to "pass").
        If the code uses IPyParallel or our custom line magics, we pass it
        through as is.
        Else we add `%%px` to the code.

        Args:
            code (str): Code to parse and validate
            metadata (dict): Associated metadata for this execution
                request. This is unused and only included for
                standardization reasons

        Returns:
            code (str): Modified code that should be executed.
            message (str or None): Colorized message to display to users in case
                of invalid input.
        """
        message = None
        if code.startswith(f"%autopx"):
            message = colorize_output(
                BColor.WARNING,
                f"Code in Parallel-Python mode is run in parallel by default and '%autopx' isn't required. This will be a no-op.",
            )
            code = "pass"
        elif not (
            code.startswith(BODO_PLATFORM_CUSTOM_LINE_MAGICS)
            or code.startswith(IPYPARALLEL_MAGICS)
        ):
            # If code uses custom or ipp magics, pass it as is.
            # This is fine since we don't allow autopx, so there's no
            # chance of running the custom or ipp magics in parallel
            # (which is not supported).
            # Prepend %%px to everything else.
            code = f"%%px\n" + code
        return code, message

    async def execute_request(self, stream, ident, parent):
        """
        Modify the input code for handling the different language types
        (SupportedLanguages). This is where we have access to the metadata
        to get the required information, like the selected language,
        selected catalog, etc.
        In case of SQL, we add the `%%sql` magic.
        In case of SQL or PARALLEL_PYTHON, we add `%%px` if autopx
        is not enabled.
        Else we pass it through as is.
        """

        metadata = parent.get("metadata", {})

        # Default to Parallel-Python for backward compatibility purposes.
        # It's possible to run on a single core (using %%px --targets 0)
        # from the Parallel Python mode, but not possible to run on all
        # cores using the regular Python mode.
        lang = metadata.get("lang", SupportedLanguages.PARALLEL_PYTHON)
        code = parent["content"]["code"]

        # Modify code and display the appropriate error message based on lang.
        if lang == SupportedLanguages.SQL:
            code, message = IPyParallelKernel.handle_sql_input(code, metadata)
            if message is not None:
                self._log_message(message)
        elif lang == SupportedLanguages.PYTHON:
            code, message = IPyParallelKernel.handle_python_input(code, metadata)
            if message is not None:
                self._log_message(message)
        elif lang == SupportedLanguages.PARALLEL_PYTHON:
            code, message = IPyParallelKernel.handle_parallel_python_input(
                code, metadata
            )
            if message is not None:
                self._log_message(message)
        else:
            # Show an error in case of unrecognized language mode.
            message = colorize_output(
                BColor.FAIL,
                f"Unrecognized Language Mode: {str(lang)}",
            )
            self._log_message(message)
            code = "pass"

        # Replace code with cleaned up code
        parent["content"]["code"] = code

        return await super().execute_request(stream, ident, parent)
