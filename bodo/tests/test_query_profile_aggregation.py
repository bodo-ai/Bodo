import pytest

import bodo.utils.aggregate_query_profiles as aggregate_query_profiles


def test_aggregate_empty():
    """Check that no input yields an empty output"""
    agg = aggregate_query_profiles.aggregate([])
    assert agg == {}


def test_aggregate_one_input():
    """Check that passing in 1 input yields the input as output"""
    agg = aggregate_query_profiles.aggregate([{"a": 0}])
    assert agg == {"a": 0}


def test_aggregate_rank_omitted():
    """Check that keys that don't have a defined aggregation strategy are preserved in the output"""
    log0 = {"rank": 0}
    log1 = {"rank": 1}
    log2 = {"rank": 2}
    log3 = {"rank": 3}
    agg = aggregate_query_profiles.aggregate([log0, log1, log2, log3])
    assert agg == {}


def test_aggregate_inconsistent_trace_level():
    """Check that all profiles need the same trace level"""
    log0 = {"trace_level": 0}
    log1 = {"trace_level": 1}
    log2 = {"trace_level": 0}
    log3 = {"trace_level": 0}
    with pytest.raises(AssertionError, match="Inconsistent trace levels"):
        aggregate_query_profiles.aggregate([log0, log1, log2, log3])

    agg = aggregate_query_profiles.aggregate([log0, log2, log3])
    assert agg == {"trace_level": 0}


def test_aggregate_inconsistent_keys():
    """Check that all profiles need the same set of keys"""
    log0 = {"a": 0}
    log1 = {"a": 1}
    log2 = {"a": 0}
    log3 = {"b": 0}
    with pytest.raises(AssertionError, match="Inconsistent keys"):
        aggregate_query_profiles.aggregate([log0, log1, log2, log3])


def test_aggregate_arbitrary_keys():
    """Check that keys that don't have a defined aggregation strategy are preserved in the output"""
    log0 = {"a": 0}
    log1 = {"a": 1}
    log2 = {"a": 2}
    log3 = {"a": 3}
    agg = aggregate_query_profiles.aggregate([log0, log1, log2, log3])
    assert agg == {"a": [0, 1, 2, 3]}


def test_aggregate_initial_operator_budget():
    """Check that inital_operator_budgets isn't duplicated in the output"""
    log0 = {"initial_operator_budgets": {"0": -1, "1": -1, "2": -1, "3": -1}}
    log1 = {"initial_operator_budgets": {"0": -1, "1": -1, "2": -1, "3": -1}}
    agg = aggregate_query_profiles.aggregate([log0, log1])
    assert agg == {"initial_operator_budgets": {"0": -1, "1": -1, "2": -1, "3": -1}}


def test_aggregate_pipelines():
    """Test aggregating pipeline objects"""
    logs = [
        {
            "pipelines": {
                0: {
                    "start": 0,
                    "end": 10,
                    "num_iterations": 1,
                },
                1: {
                    "start": 0,
                    "end": 10,
                    "num_iterations": i * 10,
                },
            },
        }
        for i in range(10)
    ]

    agg = aggregate_query_profiles.aggregate(logs)
    pipeline0 = agg["pipelines"][0]
    duration0 = pipeline0["duration"]
    assert duration0["data"] == [10] * 10
    # The five number summary should be all 10s
    assert set(duration0["summary"].keys()) == {"min", "q1", "median", "q3", "max"}
    assert all(x == 10 for x in duration0["summary"].values())

    pipeline1 = agg["pipelines"][1]
    num_iterations1 = pipeline1["num_iterations"]
    assert num_iterations1["data"] == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert set(num_iterations1["summary"].keys()) == {
        "min",
        "q1",
        "median",
        "q3",
        "max",
    }
    assert num_iterations1["summary"]["min"] == 0
    assert num_iterations1["summary"]["q1"] == 22.5
    assert num_iterations1["summary"]["median"] == 45
    assert num_iterations1["summary"]["q3"] == 67.5
    assert num_iterations1["summary"]["max"] == 90


def test_aggregate_invalid_bufferpool_stats():
    """Test aggregating buffer pool stats"""
    profiles = [
        {"buffer_pool_stats": {"a": 0}},
        {"buffer_pool_stats": {"b": 0}},
    ]

    with pytest.raises(AssertionError, match="Inconsistent buffer pool stat keys"):
        aggregate_query_profiles.aggregate(profiles)


def test_aggregate_bufferpool_stats():
    """Test aggregating buffer pool stats"""
    profile = {
        "buffer_pool_stats": {
            "general stats": {
                "curr_bytes_allocated": 0,
            },
            "SizeClassMetrics": {
                "64KiB": {
                    "Num Spilled": 0,
                    "Spill Time": 0,
                    "Num Readback": 0,
                    "Readback Time": 0,
                    "Num Madvise": 0,
                    "Madvise Time": 0,
                    "Unmapped Time": 0,
                },
                "128KiB": {
                    "Num Spilled": 0,
                    "Spill Time": 0,
                    "Num Readback": 0,
                    "Readback Time": 0,
                    "Num Madvise": 0,
                    "Madvise Time": 0,
                    "Unmapped Time": 0,
                },
            },
        }
    }

    agg = aggregate_query_profiles.aggregate([profile, profile])
    assert "buffer_pool_stats" in agg

    assert "general stats" in agg["buffer_pool_stats"]
    assert "curr_bytes_allocated" in agg["buffer_pool_stats"]["general stats"]
    assert (
        "summary" in agg["buffer_pool_stats"]["general stats"]["curr_bytes_allocated"]
    )

    assert "SizeClassMetrics" in agg["buffer_pool_stats"]
    assert "64KiB" in agg["buffer_pool_stats"]["SizeClassMetrics"]
    assert "128KiB" in agg["buffer_pool_stats"]["SizeClassMetrics"]
    for k in profile["buffer_pool_stats"]["SizeClassMetrics"]["64KiB"]:
        assert k in agg["buffer_pool_stats"]["SizeClassMetrics"]["64KiB"]
        assert "data" in agg["buffer_pool_stats"]["SizeClassMetrics"]["64KiB"][k]
        assert "summary" in agg["buffer_pool_stats"]["SizeClassMetrics"]["64KiB"][k]
