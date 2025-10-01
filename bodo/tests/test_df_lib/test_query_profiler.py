import numpy as np
import pyarrow as pa
import pytest

import bodo.pandas as pd
from bodo.tests.utils import temp_env_override


@pytest.mark.skip(
    reason="Doesn't seem to deactivate and causes problems for subsequent tests."
)
def test_df_lib_project_metrics_collection(memory_leak_check, tmp_path):
    """
    Test that generated query profile has the metrics that we expect
    to be reported by DataFrame library project.
    """

    df = pd.DataFrame(
        {
            "A": pa.array(list(np.arange(200)) * 160, type="Int64"),
            "B": pa.array(
                [None, "apple", "pie", "egg", "salad", "banana", "kiwi", "pudding"]
                * 4000,
                type=pa.dictionary(pa.int32(), pa.string()),
            ),
            "C": pa.array(list(np.arange(100)) * 320, type="Int64"),
        }
    )

    with temp_env_override(
        {"BODO_TRACING_LEVEL": "1", "BODO_TRACING_OUTPUT_DIR": str(tmp_path)}
    ):
        df["D"] = df["A"] + df["C"]
        df.execute_plan()

    # TODO check that the query profile is generated
    # and the expected metrics are present
    # with open(tmp_path / "query_profile.json") as f:
    #    profile_json = json.load(f)
