# Copyright (C) 2019 Bodo Inc.
import os
import pytest


# similar to Pandas
@pytest.fixture
def datapath():
    """Get the path to a test data file.

    Parameters
    ----------
    path : str
        Path to the file, relative to ``bodo/tests/data``

    Returns
    -------
    path : path including ``bodo/tests/data``.

    Raises
    ------
    ValueError
        If the path doesn't exist.
    """
    BASE_PATH = os.path.join(os.path.dirname(__file__), "data")

    def deco(*args):
        path = os.path.join(BASE_PATH, *args)
        if not os.path.exists(path):
            msg = "Could not find file {}."
            raise ValueError(msg.format(path))
        return path

    return deco
