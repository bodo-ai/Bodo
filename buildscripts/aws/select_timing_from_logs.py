"""
    Script that runs on a logfile with all of the tests using pytest -vv --durations=0
    and filters the log file to group together test timings. Alternatively, the logfile
    can be produced by cating a series of log files, as the tests are already presumed
    to lie in different regions.
"""

import argparse
import collections
import datetime
import re

# Example that matches timing_start from a log:
# ============================== slowest durations ===============================
timing_start = re.compile(r"^\s*=+\s*slowest durations\s*=+\s*$")
# Example that matches timing_end from a log:
# ============== 62 passed, 4579 deselected, 78 warnings in 46.29s ===============
timing_end = re.compile(r"^\s*=+")
# Example that matches time_regex from a log:
# 1.42s call     tests/test_strings.py::test_cmp_binary_op[ge]
time_regex = re.compile(r"^\s*((\d+\.\d+)|(\d+))s\s*$")

"""
    Input: File path corresponding to the log file
    Output: List of lines of the form: time(s) type functionname[input_if_paramertized]
"""


def filter_log_file(logfile):
    with open(logfile, "r") as f:
        found_timing = False
        lines = []
        for line in f:
            if found_timing:
                if timing_end.match(line):
                    found_timing = False
                else:
                    lines.append(line)
            else:
                if timing_start.match(line):
                    found_timing = True
    return lines


"""
    Input: List of strings outputted by filter_log_file
    Output: Dictionary[function_name] -> Total time in seconds
"""


def match_function_timings(timings):
    output_dict = collections.defaultdict(lambda: 0.0)
    for timing in timings:
        # Split line into 3 important parts: times, type and function
        # We will aggregate on function (summing the times) and drop the type
        parts = timing.split(" ")
        # If extra spaces remove ''
        parts = [p for p in parts if p != ""]
        # Output seems to be entirely in seconds so drop the seconds.
        # Raise an exception if this doesn't match
        # Example of matching results:
        #  1.42s call     tests/test_strings.py::test_cmp_binary_op[ge]
        matched_output = time_regex.match(parts[0])
        if matched_output:
            # Example of matched timing:
            # 1.42
            timing = matched_output.group(1)
            # Filter the output to only contain the filename + test name (no path) for pytest
            # Example select only: test_strings.py::test_cmp_binary_op[ge]
            function = parts[2].rsplit("/", 1)[-1].strip()
            output_dict[function] += float(timing)
        else:
            raise Exception("Test time not given in seconds. Script needs revising")

    return output_dict


"""
    Input: Dictionary[function_name] -> Total time in seconds, number of groups
    Output: Dictionary[function] -> GroupNumber
    This function groups together tests by a greedy algorithm.
"""


def group_timings(function_dicts, groups):
    # Create a dictionary for this grouping
    # Anything missing will run in group 0
    groups_dict = collections.defaultdict(lambda: 0)
    # Actually create the timing dict keys because want to take a min
    timing_dict = dict()
    for i in range(groups):
        timing_dict[str(i)] = 0.0

    to_remove = []
    for func_name, timetaken in function_dicts.items():
        # Group together all the S3 tests to place in group 0. This
        # is because they have dependencies between tests
        if "test_s3.py" in func_name:
            timing_dict["0"] += timetaken
            to_remove.append(func_name)
    for func_name in to_remove:
        del function_dicts[func_name]
    # Sort the function dicts
    sorted_functions = sorted(function_dicts.items(), key=lambda x: x[1], reverse=True)
    # Iterate over each of the functions
    for func_name, time_taken in sorted_functions:
        group_selected = min(timing_dict, key=timing_dict.get)
        groups_dict[func_name] = group_selected
        timing_dict[group_selected] += time_taken
    print(timing_dict)
    return groups_dict


def generate_marker_groups(logfile, num_groups):
    contents = filter_log_file(logfile)
    aggregated_functions = match_function_timings(contents)
    grouped_contents = group_timings(aggregated_functions, num_groups)
    return grouped_contents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logfile", type=str, help="Path to the logfile that needs to be parsed."
    )
    parser.add_argument(
        "num_groups", type=int, help="Number of groups to split the tests into."
    )
    args = parser.parse_args()
    generate_marker_groups(args.logfile, args.num_groups)
