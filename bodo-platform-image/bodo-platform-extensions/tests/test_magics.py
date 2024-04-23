from pytest import raises

def test_sql_no_catalog_name_specified(ipython):
    with raises(ValueError, match="No catalog name specified."):
        ipython.run_cell_magic(magic_name="sql", line="", cell="x = _99")


