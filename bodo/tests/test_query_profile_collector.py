import os
import tempfile

import bodo
from bodo.tests.utils import temp_env_override


def test_query_profile_collection_compiles():
    """Check that all query profile collector functions compile"""

    @bodo.jit
    def impl():
        bodo.libs.query_profile_collector.init()
        bodo.libs.query_profile_collector.start_pipeline(1)
        bodo.libs.query_profile_collector.end_pipeline(1, 10)
        bodo.libs.query_profile_collector.finalize()
        return

    impl()


def test_output_directory_can_be_set():
    """Check that the output directory can be set"""

    with tempfile.TemporaryDirectory() as test_dir:
        with temp_env_override(
            {"BODO_TRACING_OUTPUT_DIR": test_dir, "BODO_TRACING_LEVEL": "1"}
        ):

            @bodo.jit
            def impl():
                bodo.libs.query_profile_collector.init()
                bodo.libs.query_profile_collector.start_pipeline(1)
                bodo.libs.query_profile_collector.end_pipeline(1, 10)
                bodo.libs.query_profile_collector.finalize()
                return

            impl()
            for f in os.listdir(test_dir):
                assert f.startswith("query_profile")
                assert f.endswith(".json")
