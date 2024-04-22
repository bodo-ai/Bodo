# Copyright (C) 2024 Bodo Inc. All rights reserved.
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


def aggregate_buffer_pool_stats(stats: list[dict[str, Any]]) -> dict[str, Any]:
    stat0 = stats[0]
    aggregated_stats = {}
    # assert that all ranks have the same buffer pool stat keys
    assert all(
        set(stat.keys()) == set(stat0.keys()) for stat in stats[1:]
    ), "Inconsistent buffer pool stat keys"

    assert all(
        set(stat["general stats"].keys()) == set(stat0["general stats"].keys())
        for stat in stats[1:]
    ), "Inconsistent buffer pool stat keys"

    general_stats = {}
    for k in stat0["general stats"]:
        data = [stat["general stats"][k] for stat in stats]
        general_stats[k] = {
            "data": data,
            "summary": five_number_summary(data),
        }
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
        subkeys = stat0[key].keys()
        assert all(
            set(stat[key].keys()) == set(subkeys) for stat in stats[1:]
        ), "Inconsistent keys"

        aggregated_stats[key] = {subkey: {} for subkey in subkeys}
        for subkey in subkeys:
            for k in stat0[key][subkey]:
                data = [stat[key][subkey][k] for stat in stats]
                aggregated_stats[key][subkey][k] = {
                    "data": data,
                    "summary": five_number_summary(data),
                }

    aggregate_bufferpool_subkeys("SizeClassMetrics")
    # StorageManagerStats is optional
    if "StorageManagerStats" in stat0:
        aggregate_bufferpool_subkeys("StorageManagerStats")
    return aggregated_stats


def aggregate_helper(
    profiles: list[dict[str, Any]], key: str, aggregated: dict[str, Any]
) -> None:
    """Aggregate the profiles with custom per-key aggregation strategies"""
    profile0 = profiles[0]
    if key == "rank":
        # We're aggregating, so omit the rank
        return

    if key == "trace_level":
        # Assert that the trace_level is consistent across all profiles
        assert all(
            profile[key] == profile0[key] for profile in profiles
        ), "Inconsistent trace levels"
        # Keep the trace level since it might be useful to double check when
        # analyzing profiles
        aggregated[key] = profile0[key]
        return

    if key == "pipelines":
        pipeline_ids = profile0[key].keys()
        assert all(
            set(profile[key].keys()) == set(pipeline_ids) for profile in profiles[1:]
        ), "Inconsistent pipeline IDs"
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

    # Using a list here to preserve the order of the keys
    keys = list(profiles[0].keys())
    assert all(
        set(profile.keys()) == set(keys) for profile in profiles[1:]
    ), "Inconsistent keys"

    for k in keys:
        aggregate_helper(profiles, k, aggregated)
    return aggregated
