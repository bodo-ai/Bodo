"""Utility functions for writing Bodo tests using tracing."""
# Copyright (C) 2022 Bodo Inc. All rights reserved.
import json
import os
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

from mpi4py import MPI

import bodo
from bodo.utils import tracing


class TracingContextManager:
    """
    Class to manage enabling tracing and storing the result as a decoded dictionary.
    Implementation is based on this stack overflow post: https://stackoverflow.com/a/62243608
    """

    def __init__(self):
        self._tracing_events = list()
        self._old_trace_dev = os.environ.get("BODO_TRACE_DEV", None)
        # If we run on Azure nightly we need to decrypt the output of
        # any tracing. AGENT_NAME distinguishes Azure from AWS
        self._needs_decryption = "AGENT_NAME" in os.environ
        # If we need to decrypt the data we run the decryption script
        # at /obfuscation/decompress_traces.py. To avoid errors with packages
        # we pass this is as an environment variable, BODO_TRACING_DECRYPTION_FILE_PATH.
        self._decryption_path = os.environ.get(
            "BODO_TRACING_DECRYPTION_FILE_PATH", None
        )

    def __enter__(self):
        # Enable tracing
        os.environ["BODO_TRACE_DEV"] = "1"
        tracing.start()

    def __exit__(self, type=None, value=None, traceback=None):
        comm = MPI.COMM_WORLD
        # Generate the trace data
        with NamedTemporaryFile() as f:
            # Write the tracing result to the file.
            tracing.dump(f.name)
            tracing_events = None
            if self._needs_decryption:
                decryption_error = None
                if bodo.get_rank() == 0:
                    try:
                        if self._decryption_path is None:
                            raise EnvironmentError(
                                "Current testing setup requires decrypting traces but no tracing file is found. Please set the absolute path for decompress_traces.py with the environment variable BODO_TRACING_DECRYPTION_FILE_PATH"
                            )
                        # Replace the file with the decrypted result.
                        ret = subprocess.run(
                            ["python", self._decryption_path, f.name, f.name]
                        )
                        # Throw an exception if the file doesn't exist.
                        ret.check_returncode()
                    except Exception as e:
                        decryption_error = e
                decryption_error = comm.bcast(decryption_error)
                if isinstance(decryption_error, Exception):
                    raise decryption_error
            if bodo.get_rank() == 0:
                with open(f.name, "r") as g:
                    # Reload the file and decode the data.
                    tracing_events = json.load(g)["traceEvents"]
            self._tracing_events = comm.bcast(tracing_events)

        # Map the event names to tracing locations
        self._event_names = dict()
        for i, x in enumerate(self._tracing_events):
            name = x["name"]
            if name not in self._event_names:
                # We use a list to support the same event occurring multiple times
                self._event_names[name] = []
            self._event_names[name].append(i)

        # Reset tracing
        if self._old_trace_dev is None:
            del os.environ["BODO_TRACE_DEV"]
        else:
            os.environ["BODO_TRACE_DEV"] = self._old_trace_dev

    @property
    def tracing_events(self) -> List[Dict[str, Any]]:
        """Return the fully list of tracing events.

        Returns:
            List[Dict[str, Any]]: List of json events.
        """
        return self._tracing_events

    def get_event(self, event_name: str, event_idx: int) -> Dict[str, Any]:
        """Returns the last event with a given name.

        Args:
            event_name (str): Name of the event to return.
            event_idx (int): The index of the event in question. If the same event
                happens multiple times this indicates which event to look at.

        Raises:
            ValueError: Event does not exist

        Returns:
            Dict[str, Any]: Dictionary containing the event.
        """
        if event_name not in self._event_names:
            raise ValueError(
                f"Event {event_name} not found in tracing. Possible events: {self._event_names.keys()}"
            )
        idx = self._event_names[event_name][event_idx]
        return self._tracing_events[idx]

    def get_event_attribute(
        self, event_name: str, attribute_name: str, event_idx: int
    ) -> Any:
        """Returns the attribute of the given attribute_name in the last
        event with the given event_name

        Args:
            event_name (str): Name of the event to search.
            attribute_name (str): Name of the event to return.
            event_idx (int): The index of the event in question. If the same event
                happens multiple times this indicates which event to look at.

        Raises:
            ValueError: Attribute does not exist in the event.

        Returns:
            Any: Attribute in question. Type depends on the attribute.
        """
        event = self.get_event(event_name, event_idx)
        event_args = event["args"]
        if attribute_name not in event_args:
            raise ValueError(
                f"Attribute {attribute_name} not found in {event_name}. Possible attributes: {event_args.keys()}"
            )
        return event_args[attribute_name]
