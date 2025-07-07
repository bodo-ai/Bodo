from __future__ import annotations

import difflib
import json

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

    with pytest.raises(AssertionError, match="Inconsistent keys"):
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
                    "Num Spilled": 1,
                    "Spill Time": 1,
                    "Num Readback": 1,
                    "Readback Time": 1,
                    "Num Madvise": 1,
                    "Madvise Time": 1,
                    "Unmapped Time": 1,
                },
            },
        }
    }

    agg = aggregate_query_profiles.aggregate([profile, profile])
    assert "buffer_pool_stats" in agg

    assert "general stats" in agg["buffer_pool_stats"]
    assert "curr_bytes_allocated" in agg["buffer_pool_stats"]["general stats"]
    assert agg["buffer_pool_stats"]["general stats"]["curr_bytes_allocated"] == 0

    assert "SizeClassMetrics" in agg["buffer_pool_stats"]

    assert "64KiB" in agg["buffer_pool_stats"]["SizeClassMetrics"]
    for k in profile["buffer_pool_stats"]["SizeClassMetrics"]["64KiB"]:
        assert k in agg["buffer_pool_stats"]["SizeClassMetrics"]["64KiB"]
        assert agg["buffer_pool_stats"]["SizeClassMetrics"]["64KiB"][k] == 0

    assert "128KiB" in agg["buffer_pool_stats"]["SizeClassMetrics"]
    for k in profile["buffer_pool_stats"]["SizeClassMetrics"]["128KiB"]:
        assert k in agg["buffer_pool_stats"]["SizeClassMetrics"]["128KiB"]
        assert "data" in agg["buffer_pool_stats"]["SizeClassMetrics"]["128KiB"][k]
        assert "summary" in agg["buffer_pool_stats"]["SizeClassMetrics"]["128KiB"][k]


def test_aggregate_operator_reports():
    """Test aggregating operator reports"""
    profile = {
        "operator_reports": {
            "1001": {
                "stage_0": {"time": 0.00013100000023769098},
                "stage_1": {
                    "time": 0.00008000000025276677,
                    "output_row_count": 0,
                    "metrics": [
                        {
                            "name": "bcast_join",
                            "type": "STAT",
                            "global": True,
                            "stat": 0,
                        },
                        {
                            "name": "bcast_time",
                            "type": "TIMER",
                            "global": False,
                            "stat": 0,
                        },
                        {
                            "name": "appends_active_time",
                            "type": "TIMER",
                            "global": False,
                            "stat": 4,
                        },
                        {
                            "name": "final_partitioning_state",
                            "type": "BLOB",
                            "global": False,
                            "stat": "[(0, 0b0),]",
                        },
                    ],
                },
            },
            "2001": {
                "stage_0": {"time": 0.000003000000106112566},
                "stage_1": {"time": 0.000016000000414351234, "output_row_count": 3},
            },
        }
    }

    agg = aggregate_query_profiles.aggregate([profile, profile])
    expected = json.dumps(
        {
            "operator_reports": {
                "1001": {
                    "stage_0": {
                        "time": {
                            "max": 0.00013100000023769098,
                            "data": [0.00013100000023769098, 0.00013100000023769098],
                            "summary": {
                                "min": 0.00013100000023769098,
                                "q1": 0.00013100000023769098,
                                "median": 0.00013100000023769098,
                                "q3": 0.00013100000023769098,
                                "max": 0.00013100000023769098,
                            },
                        }
                    },
                    "stage_1": {
                        "time": {
                            "max": 8.000000025276677e-05,
                            "data": [8.000000025276677e-05, 8.000000025276677e-05],
                            "summary": {
                                "min": 8.000000025276677e-05,
                                "q1": 8.000000025276677e-05,
                                "median": 8.000000025276677e-05,
                                "q3": 8.000000025276677e-05,
                                "max": 8.000000025276677e-05,
                            },
                        },
                        "output_row_count": 0,
                        "metrics": [
                            {
                                "name": "bcast_join",
                                "type": "STAT",
                                "global": True,
                                "stat": 0,
                            },
                            {
                                "name": "bcast_time",
                                "type": "TIMER",
                                "global": False,
                                "sum": 0,
                                "max": 0,
                                "summary": {
                                    "min": 0.0,
                                    "q1": 0.0,
                                    "median": 0.0,
                                    "q3": 0.0,
                                    "max": 0.0,
                                },
                                "data": [0, 0],
                            },
                            {
                                "name": "appends_active_time",
                                "type": "TIMER",
                                "global": False,
                                "sum": 8,
                                "max": 4,
                                "summary": {
                                    "min": 4.0,
                                    "q1": 4.0,
                                    "median": 4.0,
                                    "q3": 4.0,
                                    "max": 4.0,
                                },
                                "data": [4, 4],
                            },
                            {
                                "name": "final_partitioning_state",
                                "type": "BLOB",
                                "global": False,
                                "data": [
                                    "[(0, 0b0),]",
                                    "[(0, 0b0),]",
                                ],
                            },
                        ],
                    },
                },
                "2001": {
                    "stage_0": {
                        "time": {
                            "max": 3.000000106112566e-06,
                            "data": [3.000000106112566e-06, 3.000000106112566e-06],
                            "summary": {
                                "min": 3.000000106112566e-06,
                                "q1": 3.000000106112566e-06,
                                "median": 3.000000106112566e-06,
                                "q3": 3.000000106112566e-06,
                                "max": 3.000000106112566e-06,
                            },
                        }
                    },
                    "stage_1": {
                        "time": {
                            "max": 1.6000000414351234e-05,
                            "data": [1.6000000414351234e-05, 1.6000000414351234e-05],
                            "summary": {
                                "min": 1.6000000414351234e-05,
                                "q1": 1.6000000414351234e-05,
                                "median": 1.6000000414351234e-05,
                                "q3": 1.6000000414351234e-05,
                                "max": 1.6000000414351234e-05,
                            },
                        },
                        "output_row_count": {
                            "data": [3, 3],
                            "summary": {
                                "min": 3.0,
                                "q1": 3.0,
                                "median": 3.0,
                                "q3": 3.0,
                                "max": 3.0,
                            },
                            "sum": 6,
                        },
                    },
                },
            }
        },
        indent=4,
    )
    actual = json.dumps(agg, indent=4)
    diff = list(difflib.unified_diff(expected.splitlines(), actual.splitlines()))
    assert diff == [], "\n".join(diff)
