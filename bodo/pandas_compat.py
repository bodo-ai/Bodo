import hashlib
import inspect
import warnings

import pandas as pd

# flag for checking whether the functions we are replacing have changed in a later Pandas
# release. Needs to be checked for every new Pandas release so we update our changes.
_check_pandas_change = False

# c_parser_wrapper change
# Bodo Change: Upgrade to Pandas 1.4 implementation which replaces
# col_indices with a dictionary
def _set_noconvert_columns(self):
    """
    Set the columns that should not undergo dtype conversions.

    Currently, any column that is involved with date parsing will not
    undergo such conversions.
    """
    assert self.orig_names is not None
    # error: Cannot determine type of 'names'

    # Bodo Change vs 1.3.4 Replace orig_names.index(x) with
    # dictionary. This is already merged into Pandas 1.4
    # much faster than using orig_names.index(x) xref GH#44106
    names_dict = {x: i for i, x in enumerate(self.orig_names)}
    col_indices = [names_dict[x] for x in self.names]  # type: ignore[has-type]
    # error: Cannot determine type of 'names'
    noconvert_columns = self._set_noconvert_dtype_columns(
        col_indices,
        self.names,  # type: ignore[has-type]
    )
    for col in noconvert_columns:
        self._reader.set_noconvert(col)


if _check_pandas_change:
    # make sure run_frontend hasn't changed before replacing it
    lines = inspect.getsource(
        pd.io.parsers.c_parser_wrapper.CParserWrapper._set_noconvert_columns
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "afc2d738f194e3976cf05d61cb16dc4224b0139451f08a1cf49c578af6f975d3"
    ):  # pragma: no cover
        warnings.warn(
            "pd.io.parsers.c_parser_wrapper.CParserWrapper._set_noconvert_columns has changed"
        )

pd.io.parsers.c_parser_wrapper.CParserWrapper._set_noconvert_columns = (
    _set_noconvert_columns
)
