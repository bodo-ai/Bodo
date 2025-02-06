"""Utilities for aggregating query profiles from multiple ranks into a single
profile"""

from typing import Any

import numpy as np


def five_number_summary(data: list[float]) -> dict[str, float]:
    """Produce a five number summary (min, 1st quartile, median, 3rd quartile,
    max) of the data"""
    min_, q1, median, q3, max_ = np.percentile(data, [0, 25, 50, 75, 100])
    return {
        "min": min_,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": max_,
    }


def generate_summary(data: list[float]) -> float | dict[str, Any]:
    if all(d == 0 for d in data):
        return 0
    return {
        "data": data,
        "summary": five_number_summary(data),
    }


def assert_keys_consistent(objects: list[dict[str, Any]]) -> list[str]:
    """Assert that all objects have the same keys and returns those keys"""
    # Using a list here to preserve the order of the keys
    keys = list(objects[0].keys())
    assert all(set(obj.keys()) == set(keys) for obj in objects[1:]), "Inconsistent keys"
    return keys


def aggregate_buffer_pool_stats(stats: list[dict[str, Any]]) -> dict[str, Any]:
    stat0 = stats[0]
    aggregated_stats = {}
    # assert that all ranks have the same buffer pool stat keys
    assert_keys_consistent(stats)
    assert_keys_consistent([stat["general stats"] for stat in stats])

    general_stats = {}
    for k in stat0["general stats"]:
        data = [stat["general stats"][k] for stat in stats]
        general_stats[k] = generate_summary(data)
    aggregated_stats["general stats"] = general_stats

    # Helper since both SizeClassMetrics and StorageManagerStats have the same
    # logic - we want to iterate over all keys under the initial key, and for
    # each key, we have an object where every subkey has a data point we want a
    # summary for.
    # e.g.
    # {
    #     "buffer_pool_stats": {
    #         "general stats": {
    #           "curr_bytes_allocated": 0,
    #           ...
    #         },
    #         "SizeClassMetrics": {
    #           "64KiB": {
    #             "Num Spilled": 0,
    #             "Spill Time": 0,
    #             "Num Readback": 0,
    #             "Readback Time": 0,
    #             "Num Madvise": 0,
    #             "Madvise Time": 0,
    #             "Unmapped Time": 0
    #           },
    #           "128KiB": {
    #             "Num Spilled": 0,
    #             "Spill Time": 0,
    #             "Num Readback": 0,
    #             "Readback Time": 0,
    #             "Num Madvise": 0,
    #             "Madvise Time": 0,
    #             "Unmapped Time": 0
    #           },
    #           ...
    #         },
    #         ...
    #       },
    #     ...
    # }
    def aggregate_bufferpool_subkeys(key: str):
        assert key in stat0, f"Missing {key}"
        subkeys = assert_keys_consistent([stat[key] for stat in stats])

        aggregated_stats[key] = {subkey: {} for subkey in subkeys}
        for subkey in subkeys:
            for k in stat0[key][subkey]:
                data = [stat[key][subkey][k] for stat in stats]
                aggregated_stats[key][subkey][k] = generate_summary(data)

    aggregate_bufferpool_subkeys("SizeClassMetrics")
    # StorageManagerStats is optional
    if "StorageManagerStats" in stat0:
        aggregate_bufferpool_subkeys("StorageManagerStats")
    return aggregated_stats


def aggregate_operator_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    op_ids = assert_keys_consistent(reports)
    res = {}
    for op in op_ids:
        agg_op = {}
        stage_ids = assert_keys_consistent([report[op] for report in reports])
        for stage in stage_ids:
            agg_stage = {}
            keys = set(
                assert_keys_consistent([report[op][stage] for report in reports])
            )
            assert keys <= {"time", "output_row_count", "metrics"}
            if "time" in keys:
                times = [report[op][stage]["time"] for report in reports]
                agg_stage["time"] = {
                    "max": max(times),
                    "data": times,
                    "summary": five_number_summary(times),
                }
            if "output_row_count" in keys:
                counts = [report[op][stage]["output_row_count"] for report in reports]
                agg_stage["output_row_count"] = generate_summary(counts)
                if not all(c == 0 for c in counts):
                    agg_stage["output_row_count"]["sum"] = sum(counts)

            if "metrics" in keys:
                """Metrics are in the from:
                    "metrics": [
                        {
                          "name": "bcast_join",
                          "type": "STAT",
                          "stat": 0,
                          "global": true
                        },
                        {
                          "name": "bcast_time",
                          "type": "TIMER",
                          "stat": 0,
                          "global": false
                        },
                        ...
                    ]
                We aggregate them with a custom strategy per type.
                """

                metrics = [report[op][stage]["metrics"] for report in reports]
                assert all(len(metric) == len(metrics[0]) for metric in metrics[1:])
                metrics_res = []
                for i in range(len(metrics[0])):
                    name = metrics[0][i]["name"]
                    assert all(metric[i]["name"] == name for metric in metrics)
                    type_ = metrics[0][i]["type"]
                    assert all(metric[i]["type"] == type_ for metric in metrics)

                    is_global = metrics[0][i]["global"]
                    if is_global:
                        agg_metric = metrics[0][i]
                    else:
                        assert type_ in ["TIMER", "STAT", "BLOB"]
                        data = [metric[i]["stat"] for metric in metrics]

                        agg_metric = {
                            "name": name,
                            "type": type_,
                            "global": False,
                        }
                        if type_ == "TIMER" or type_ == "STAT":
                            agg_metric.update(
                                {
                                    "sum": sum(data),
                                    "max": max(data),
                                    "summary": five_number_summary(data),
                                }
                            )
                        agg_metric["data"] = data
                    metrics_res.append(agg_metric)

                agg_stage["metrics"] = metrics_res
            agg_op[stage] = agg_stage
        res[op] = agg_op
    return res


def aggregate_helper(
    profiles: list[dict[str, Any]], key: str, aggregated: dict[str, Any]
) -> None:
    """Aggregate the profiles with custom per-key aggregation strategies"""
    profile0 = profiles[0]
    if key == "rank":
        # We're aggregating, so omit the rank
        return

    if key == "trace_level":
        trace_level = profile0[key]
        # Assert that the trace_level is consistent across all profiles
        assert all(profile[key] == trace_level for profile in profiles), (
            "Inconsistent trace levels"
        )
        # Keep the trace level since it might be useful to double check when
        # analyzing profiles
        aggregated[key] = profile0[key]
        return

    if key == "operator_reports":
        aggregated[key] = aggregate_operator_reports(
            [profile[key] for profile in profiles]
        )
        return

    if key == "pipelines":
        pipeline_ids = assert_keys_consistent([profile[key] for profile in profiles])
        aggregated_pipeline = {}
        for pipeline_id in pipeline_ids:

            def get_duration(pipeline_stage: dict[str, Any]) -> float:
                return pipeline_stage["end"] - pipeline_stage["start"]

            durations = [
                get_duration(profile[key][pipeline_id]) for profile in profiles
            ]
            iterations = [
                profile[key][pipeline_id]["num_iterations"] for profile in profiles
            ]
            aggregated_pipeline[pipeline_id] = {
                "duration": {
                    "data": durations,
                    "summary": five_number_summary(durations),
                },
                "num_iterations": {
                    "data": iterations,
                    "summary": five_number_summary(iterations),
                },
            }
        aggregated[key] = aggregated_pipeline
        return

    if key == "initial_operator_budgets":
        # Assumes that all ranks have the same initial operator budgets
        aggregated[key] = profile0[key]
        return

    if key == "buffer_pool_stats":
        aggregated[key] = aggregate_buffer_pool_stats(
            [profile[key] for profile in profiles]
        )
        return

    # default to aggregating as a list
    aggregated[key] = [profile[key] for profile in profiles]
    return


def aggregate(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Given a set of query profiles from different ranks, aggregate them into a
    single profile, summarizing the data as necessary"""
    if len(profiles) == 0:
        return {}
    elif len(profiles) == 1:
        return profiles[0]

    aggregated = {}
    keys = assert_keys_consistent(profiles)
    for k in keys:
        aggregate_helper(profiles, k, aggregated)
    return aggregated
