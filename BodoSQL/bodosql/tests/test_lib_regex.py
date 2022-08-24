# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of bodosql.libs.regex functions
"""
import re

import bodosql
import pytest


@pytest.mark.slow
def test_sql_to_python_empty():
    pat = ""
    new_pat = bodosql.libs.regex.sql_to_python(pat)
    assert new_pat == "^$"


@pytest.mark.slow
def test_sql_to_python_no_wildcards():
    pat = "dfwfwe"
    new_pat = bodosql.libs.regex.sql_to_python(pat)
    assert new_pat == "^dfwfwe$"


@pytest.mark.slow
def test_sql_to_python_pythonwildcards():
    pat = "dfwfwe.*..."
    new_pat = bodosql.libs.regex.sql_to_python(pat)
    assert new_pat == ("^" + re.escape(pat) + "$")


@pytest.mark.slow
def test_sql_to_python_leading_per():
    pat = "%%dfwfwe"
    new_pat = bodosql.libs.regex.sql_to_python(pat)
    assert new_pat == "dfwfwe$"


@pytest.mark.slow
def test_sql_to_python_trailing_per():
    pat = "dfwfwe%%"
    new_pat = bodosql.libs.regex.sql_to_python(pat)
    assert new_pat == "^dfwfwe"


@pytest.mark.slow
def test_sql_to_python_leading_trailing_per():
    pat = "%%dfwfwe%%"
    new_pat = bodosql.libs.regex.sql_to_python(pat)
    assert new_pat == "dfwfwe"


@pytest.mark.slow
def test_sql_to_python_only_per():
    pat = "%%%%"
    new_pat = bodosql.libs.regex.sql_to_python(pat)
    assert new_pat == ""


@pytest.mark.slow
def test_sql_to_python_inner_per():
    pat = "a%nc"
    new_pat = bodosql.libs.regex.sql_to_python(pat)
    assert new_pat == "^a.*nc$"


@pytest.mark.slow
def test_sql_to_python_underscore():
    pat = "a_nc"
    new_pat = bodosql.libs.regex.sql_to_python(pat)
    assert new_pat == "^a.nc$"


@pytest.mark.slow
def test_sql_to_python_underscore_per():
    pat = "%_n.%_c"
    new_pat = bodosql.libs.regex.sql_to_python(pat)
    assert new_pat == ".n\\..*.c$"
