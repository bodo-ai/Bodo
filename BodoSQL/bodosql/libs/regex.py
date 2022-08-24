# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Library of BodoSQL functions used in support SQL -> Python Regex
    conversions.
"""
import re

import bodo


@bodo.jit
def sql_to_python(sql_pattern):
    """
    Helper Function to convert a sql pattern with
    % and _ as valid wildcard into a matching python pattern.
    """
    if len(sql_pattern) == 0:
        # Empty pattern should only match a completely empty string.
        return "^$"
    if sql_pattern[0] == "%":
        front = ""
    else:
        front = "^"
    if sql_pattern[-1] == "%":
        end = ""
    else:
        end = "$"
    # Remove any leading or trailing % to simplify the regex
    # This could change the group matched by a regex, but it will
    # not change if a pattern matches.
    sql_pattern = sql_pattern.strip("%")
    pat = front + re.escape(sql_pattern).replace("%", ".*").replace("_", ".") + end
    return pat
