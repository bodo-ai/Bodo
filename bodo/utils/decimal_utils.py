from __future__ import annotations


def decimal_addition_subtraction_output_precision_scale(p1, s1, p2, s2):
    """
    Calculate the output precision and scale for a addition/subtraction of two decimals.
    See: https://docs.snowflake.com/en/sql-reference/operators-arithmetic#addition-and-subtraction
    """
    l1 = p1 - s1
    l2 = p2 - s2
    l = max(l1, l2) + 1
    s = max(s1, s2)
    p = min(l + s, 38)
    return p, s


def decimal_multiplication_output_precision_scale(p1, s1, p2, s2):
    """
    Calculate the output precision and scale for a multiplication of two decimals.
    See: https://docs.snowflake.com/en/sql-reference/operators-arithmetic#multiplication
    """
    l1 = p1 - s1
    l2 = p2 - s2
    l = l1 + l2
    s = min(s1 + s2, max(s1, s2, 12))
    p = min(l + s, 38)
    return p, s


def decimal_division_output_precision_scale(p1, s1, p2, s2):
    """
    Calculate the output precision and scale for a division of two decimals.
    See: https://docs.snowflake.com/en/sql-reference/operators-arithmetic#division
    """
    l1 = p1 - s1
    l = l1 + s2
    s = max(s1, min(s1 + 6, 12))
    p = min(l + s, 38)
    return p, s
