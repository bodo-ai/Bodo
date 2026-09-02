"""Aggregate query profile logs:
usage: python -m bodo.utils.query_profile_collector_aggregator <dir>
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

from bodo.utils.aggregate_query_profiles import aggregate


def find_metric_value(metrics, name):
    """
    Search metrics list for an entry with "name" == name and "type" == "STAT".
    Return the integer value found in the "stat" field, or None if not present.
    """
    for m in metrics:
        if not isinstance(m, dict):
            continue
        if m.get("name") == name and m.get("type") == "STAT":
            # Accept either "stat" or numeric fields
            val = m.get("stat")
            if isinstance(val, (int, float)):
                return int(val)
    return None


def stage_max_time(stage):
    """
    Return the max time for a stage.
    Prefer stage["time"]["max"] if present, otherwise compute max(stage["time"]["data"]).
    Return None if no time information is available.
    """
    time_block = stage.get("time")
    if not isinstance(time_block, dict):
        return None
    # Prefer explicit max
    max_val = time_block.get("max")
    if isinstance(max_val, (int, float)):
        return float(max_val)
    # Otherwise compute from data array
    data = time_block.get("data")
    if isinstance(data, list) and data:
        numeric = [x for x in data if isinstance(x, (int, float))]
        if numeric:
            return float(max(numeric))
    return None


def extract_pipeline_times(obj):
    results = []
    pipeline_ret = {}

    pipelines = obj.get("pipelines")
    if not isinstance(pipelines, dict):
        return [], {}

    for pipeline_num, pipeline_data in pipelines.items():
        if not isinstance(pipeline_data, dict):
            raise TypeError("pipeline_data is not a dict")
        duration_dict = pipeline_data["duration"]
        if not isinstance(duration_dict, dict):
            raise TypeError("duration_dict is not a dict")
        summary_dict = duration_dict["summary"]
        if not isinstance(summary_dict, dict):
            raise TypeError("summary_dict is not a dict")
        pipeline_ret[int(pipeline_num)] = summary_dict["max"]

    operator_reports = obj.get("operator_reports")
    if not isinstance(operator_reports, dict):
        return [], {}

    for operator_name, operator_val in operator_reports.items():
        # Expect stage_val to be a dict
        if not isinstance(operator_val, dict):
            continue
        for stage_name, stage_val in operator_val.items():
            if not stage_name.startswith("stage_"):
                continue
            metrics = stage_val.get("metrics")
            if not isinstance(metrics, list):
                continue
            pipeline_num = find_metric_value(metrics, "pipeline_num")
            pipeline_pos = find_metric_value(metrics, "pipeline_position")
            if pipeline_num is None or pipeline_pos is None:
                continue
            max_time = stage_max_time(stage_val)
            if max_time is None:
                continue
            addendum = ""
            if operator_name.startswith("PhysicalJoin "):
                if stage_name == "stage_1":
                    addendum = " build"
                elif stage_name == "stage_2":
                    addendum = " probe"
            elif operator_name.startswith("PhysicalAggregate "):
                if stage_name == "stage_1":
                    addendum = " sink"
                elif stage_name == "stage_2":
                    addendum = " source"
            results.append(
                (
                    pipeline_num,
                    pipeline_pos,
                    operator_name + addendum,
                    max_time / 1000000.0,
                )
            )

    return results, pipeline_ret


def print_sorted(outdir, results, pipeline_ret):
    results_sorted = sorted(results, key=lambda t: (t[0], t[1]))
    if not results_sorted:
        print("No stages with both pipeline_num and pipeline_position and time found.")
        return

    last_pipeline_num = None
    df_data = []
    for pn, pp, stage, mt in results_sorted:
        if pn != last_pipeline_num:
            print("\nPipeline", pn, pipeline_ret[pn], end=" ")
            last_pipeline_num = pn
        print(f"({stage}, {mt})", end=" ")
        df_data.append((pn, pipeline_ret[pn], stage, mt))
    print("")

    df = pd.DataFrame(df_data, columns=["pipeline", "pipeline_total", "stage", "time"])
    df["pipeline_label"] = df["pipeline"].apply(lambda x: f"Pipeline {x}")

    fig = px.sunburst(
        df,
        path=["pipeline_label", "stage"],
        values="time",
        color="stage",
        title="Pipeline stage time breakdown (sunburst)",
        hover_data={"time": True, "pipeline_total": True},
    )
    fig.update_traces(textinfo="label+value+percent entry")
    fig.write_html(
        outdir / "pipeline_sunburst.html", include_plotlyjs="cdn", full_html=True
    )


# The pragma: no cover comments are used to skip coverage because this is just a
# wraper around the functionality in __init__.py, which is covered by tests.


def main(argv: list[str]):  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--print", dest="print", default=False, action="store_true")
    parser.add_argument(
        "--no-pipeline", dest="pipeline", default=True, action="store_false"
    )
    args = parser.parse_args(argv[1:])
    assert args.dir.is_dir(), f"'{args.dir}' is not a directory."

    logs = []
    for path in args.dir.iterdir():
        if not path.stem.startswith("query_profile"):
            continue
        with path.open() as f:
            data = json.load(f)
        logs.append(data)

    aggregated = json.dumps(aggregate(logs), indent=4)
    if args.print:
        print(aggregated)
    with open(args.dir / "aggregated.json", "w") as f:
        f.write(aggregated)
        print(
            f"Aggregated logs written to {args.dir / 'aggregated.json'}",
            file=sys.stderr,
        )

    if args.pipeline:
        with open(args.dir / "aggregated.json", encoding="utf-8") as f:
            data = json.load(f)

        results, pipeline_ret = extract_pipeline_times(data)
        if len(results) != 0 and len(pipeline_ret) != 0:
            print_sorted(args.dir, results, pipeline_ret)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv)
