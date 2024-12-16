"""
File used to handle the logic for BodoSQL specific calls in the BodoSQL package.
"""

import typing as pt

import numba.core.ir_utils


def remove_bodosql_calls(rhs, lives, call_list: list[pt.Any]) -> bool:
    """Remove any calls in the Bodo IR that originate from the BodoSQL
    module. Once we move bodosql array kernels to the BodoSQL module
    this may need additional updates/specifications.

    At the time of writing every BodoSQL call is a pure function.

    Args:
        rhs: unused
        lives: unused
        call_list (List[types.Any]): List of components that define the call information.

    Returns:
        bool: Can we definitely remove the call. If this returns False Numba will try
        any additional handlers.
    """
    import bodosql

    if call_list[-1] == bodosql:
        return True

    return False


numba.core.ir_utils.remove_call_handlers.append(remove_bodosql_calls)
