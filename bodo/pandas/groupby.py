"""
Provides a Bodo implementation of the pandas groupby API.
"""


class DataFrameGroupBy:
    """
    Similar to pandas DataFrameGroupBy. See Pandas code for reference:
    https://github.com/pandas-dev/pandas/blob/0691c5cf90477d3503834d983f69350f250a6ff7/pandas/core/groupby/generic.py#L1329
    """

    def __init__(self, obj, keys):
        self._obj = obj
        self._keys = keys

    def sum(self):
        """
        Compute the sum of each group.
        """
