import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from ddltest_harness import DDLTestHarness
from filesystem_test_harness import FilesystemTestHarness
from mpi4py import MPI
from numba.core.config import shutil
from rest_test_harness import RestTestHarness

import bodo
import bodosql
from bodo.tests.iceberg_database_helpers.utils import get_spark
from bodo.tests.utils import pytest_one_rank, pytest_polaris, temp_env_override

# Add test harnesses when adding new catalogs.
from BodoSQL.bodosql.bodosql_types.rest_catalog import RESTCatalog
from bodosql.tests.utils import assert_equal_par, replace_type_varchar

pytestmark = pytest_polaris + pytest_one_rank + [pytest.mark.slow]


@pytest.fixture(scope="session")
def test_harness_path(tmp_path_factory):
    # BSE-4273: Use the current relative path + iceberg_db database like
    # other iceberg tests
    path = None
    if bodo.get_rank() == 0:
        path = str(Path("test_harness").resolve())
        os.makedirs(path, exist_ok=True)
    path = MPI.COMM_WORLD.bcast(path)
    assert Path(path).exists(), "Failed to create filesystem catalog across all ranks"
    yield path
    bodo.barrier()
    if bodo.get_rank() == 0:
        shutil.rmtree(path)
    bodo.barrier()


@pytest.fixture(scope="session")
def filesystem_test_harness(test_harness_path):
    catalog = bodosql.FileSystemCatalog(test_harness_path)
    return FilesystemTestHarness(catalog)


@pytest.fixture(scope="session", autouse=True)
def setup_spark():
    # Get Spark to start it with all environment variables set, not overriden by polaris_connection fixture
    with temp_env_override({"AWS_REGION": "us-east-2"}):
        get_spark()


@pytest.fixture
def rest_test_harness(aws_polaris_catalog, aws_polaris_connection):
    # This is needed for Spark
    with temp_env_override({"AWS_REGION": "us-east-2"}):
        catalog = RESTCatalog(
            aws_polaris_catalog.warehouse,
            aws_polaris_catalog.rest_uri,
            aws_polaris_catalog.token,
            aws_polaris_catalog.credential,
            aws_polaris_catalog.scope,
            "default",
        )
        bc = bodosql.BodoSQLContext(catalog=catalog)
        bc.sql("CREATE SCHEMA IF NOT EXISTS BODOSQL_DDL_TESTS")
        bc.sql("CREATE SCHEMA IF NOT EXISTS BODOSQL_DDL_TESTS_ALTERNATE")
        harness = RestTestHarness(catalog, aws_polaris_connection)
        assert harness.check_schema_exists("BODOSQL_DDL_TESTS")
        assert harness.check_schema_exists("BODOSQL_DDL_TESTS_ALTERNATE")
        yield harness


def trim_describe_table_output(output: pd.DataFrame):
    """Retrieve only the name and type field from the output of calling
    describe_table. Additionally, erase precision from VARCHAR types"""
    trimmed = output[["NAME", "TYPE"]]
    return replace_type_varchar(trimmed)


###############################################
#                   Testing                   #
###############################################


# CREATE SCHEMA
@pytest.mark.parametrize(
    "harness_name, ifExists",
    [
        pytest.param("rest_test_harness", True, id="rest-if_exists"),
        pytest.param("rest_test_harness", False, id="rest-no_if_exists"),
        pytest.param("filesystem_test_harness", True, id="filesystem-if_exists"),
        pytest.param("filesystem_test_harness", False, id="filesystem-no_if_exists"),
    ],
)
def test_create_schema(request, harness_name: str, ifExists: bool):
    """Tests that Bodo can create a schema."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    schema_name = harness.gen_unique_id("TEST_SCHEMA_DDL").upper()
    try:
        # Query
        ifExistsClause = "IF NOT EXISTS" if ifExists else ""
        bodo_output = harness.run_bodo_query(
            f"CREATE SCHEMA {ifExistsClause} {schema_name}"
        )
        py_output = pd.DataFrame(
            {"STATUS": [f"Schema '{schema_name}' successfully created."]}
        )
        assert_equal_par(bodo_output, py_output)

        # Check schema exists
        assert harness.check_schema_exists(schema_name)
    finally:
        harness.run_spark_query(f"DROP SCHEMA IF EXISTS {schema_name}")


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_create_schema_already_exists(request, harness_name: str):
    """Tests that Bodo can create a schema."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    schema_name = harness.gen_unique_id("TEST_SCHEMA_DDL").upper()
    try:
        # Query
        bodo_output = harness.run_bodo_query(f"CREATE SCHEMA {schema_name}")
        py_output = pd.DataFrame(
            {"STATUS": [f"Schema '{schema_name}' successfully created."]}
        )
        assert_equal_par(bodo_output, py_output)

        # Check schema exists
        assert harness.check_schema_exists(schema_name)

        # Create schema again
        with pytest.raises(ValueError, match="already exists"):
            bodo_output = harness.run_bodo_query(f"CREATE SCHEMA {schema_name}")

    finally:
        harness.run_spark_query(f"DROP SCHEMA IF EXISTS {schema_name}")


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_create_schema_ifnotexists_already_exists(request, harness_name: str):
    """Tests that Bodo can create a schema."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    schema_name = harness.gen_unique_id("TEST_SCHEMA_DDL").upper()
    try:
        # Query
        bodo_output = harness.run_bodo_query(f"CREATE SCHEMA {schema_name}")
        py_output = pd.DataFrame(
            {"STATUS": [f"Schema '{schema_name}' successfully created."]}
        )
        assert_equal_par(bodo_output, py_output)

        # Check schema exists
        assert harness.check_schema_exists(schema_name)

        # Create schema again
        bodo_output = harness.run_bodo_query(
            f"CREATE SCHEMA IF NOT EXISTS {schema_name}"
        )
        py_output = pd.DataFrame(
            {"STATUS": [f"'{schema_name}' already exists, statement succeeded."]}
        )
        assert_equal_par(bodo_output, py_output)
    finally:
        harness.run_spark_query(f"DROP SCHEMA IF EXISTS {schema_name}")


# DESCRIBE SCHEMA


@pytest.mark.parametrize(
    "harness_name, test_views",
    [
        pytest.param("rest_test_harness", True, id="rest-views"),
        pytest.param("filesystem_test_harness", False, id="filesystem-no_views"),
    ],
)
def test_describe_schema(request, harness_name: str, test_views: bool):
    """Tests that Bodo can show tables in a terse format."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    table_name = harness.gen_unique_id("test_table").upper()
    db_schema = harness.gen_unique_id("test_schema").upper()
    table_identifier = harness.get_table_identifier(table_name, db_schema)
    view_name = harness.gen_unique_id("test_view").upper()
    view_identifier = harness.get_table_identifier(view_name, db_schema)
    try:
        # Create blank schema
        query = f"CREATE SCHEMA {db_schema}"
        harness.run_spark_query(query)
        harness.check_schema_exists(db_schema)
        # Create table
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)
        # Create view if catalog supports it
        if test_views:
            harness.create_test_view(view_identifier)
            assert harness.check_view_exists(view_identifier)
            py_output = pd.DataFrame(
                {
                    "CREATED_ON": pd.array(
                        [None, None], dtype=pd.ArrowDtype(pa.string())
                    ),
                    "NAME": [table_name, view_name],
                    "KIND": ["TABLE", "VIEW"],
                }
            )
        else:
            py_output = pd.DataFrame(
                {
                    "CREATED_ON": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                    "NAME": [table_name],
                    "KIND": ["TABLE"],
                }
            )
        # Describe schema
        query = f"DESCRIBE SCHEMA {db_schema}"
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
    finally:
        harness.drop_test_table(table_identifier)
        if test_views:
            harness.drop_test_view(view_identifier)
        harness.run_spark_query(f"DROP SCHEMA {db_schema}")


# ALTER TABLE RENAME TO


@pytest.mark.parametrize(
    "harness_name, ifExists",
    [
        pytest.param("rest_test_harness", True, id="rest_if_exists"),
        pytest.param("rest_test_harness", False, id="rest_no_if_exists"),
    ],
)
def test_alter_table_rename(request, harness_name: str, ifExists: bool):
    """Tests that Bodo can rename a table via ALTER TABLE."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    try:
        # Create table
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name)
        new_table_identifier = harness.get_table_identifier(table_name + "_renamed")
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)

        # Query
        ifExistsClause = "IF EXISTS" if ifExists else ""
        bodo_output = harness.run_bodo_query(
            f"ALTER TABLE {ifExistsClause} {table_identifier} RENAME TO {new_table_identifier}"
        )
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        assert_equal_par(bodo_output, py_output)

        # Check renamed
        assert harness.check_table_exists(new_table_identifier)
    finally:
        harness.drop_test_table(new_table_identifier)
        harness.drop_test_table(table_identifier)


@pytest.mark.parametrize(
    "harness_name",
    [pytest.param("rest_test_harness", id="rest")],
)
def test_alter_table_rename_compound(request, harness_name: str):
    """Tests that Bodo can rename a table via ALTER TABLE, where we can specify a schema."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name, "BODOSQL_DDL_TESTS")
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)

        rename_table_identifier = harness.get_table_identifier(
            table_name + "_renamed", "BODOSQL_DDL_TESTS"
        )
        # Verify no properties are set
        query = f"ALTER TABLE {table_identifier} RENAME TO {rename_table_identifier}"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        assert harness.check_table_exists(rename_table_identifier)

    finally:
        harness.drop_test_table(rename_table_identifier)
        harness.drop_test_table(table_identifier)


@pytest.mark.parametrize(
    "harness_name",
    [pytest.param("rest_test_harness", id="rest")],
)
def test_alter_table_rename_diffschema(request, harness_name: str):
    """Tests that Bodo can rename a table via ALTER TABLE into a different schema."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name, "BODOSQL_DDL_TESTS")
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)

        with pytest.raises(ValueError, match="Namespace does not exist"):
            rename_table_identifier = harness.get_table_identifier(
                table_name + "_renamed", "NONEXISTENT_SCHEMA"
            )
            query = (
                f"ALTER TABLE {table_identifier} RENAME TO {rename_table_identifier}"
            )
            py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
            bodo_output = harness.run_bodo_query(query)
            assert_equal_par(bodo_output, py_output)

        rename_table_identifier = harness.get_table_identifier(
            table_name + "_renamed", "BODOSQL_DDL_TESTS_ALTERNATE"
        )
        query = f"ALTER TABLE {table_identifier} RENAME TO {rename_table_identifier}"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        assert harness.check_table_exists(rename_table_identifier)

    finally:
        harness.drop_test_table(rename_table_identifier)
        harness.drop_test_table(table_identifier)


@pytest.mark.parametrize(
    "harness_name",
    [pytest.param("rest_test_harness", id="rest")],
)
def test_alter_table_rename_not_found(request, harness_name: str):
    """Tests that Bodo throws an error when trying to rename a table that does not exist."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    table_name = harness.gen_unique_id("test_table").upper()
    table_identifier = harness.get_table_identifier(table_name)
    # Table should not exist.
    assert not harness.check_table_exists(table_identifier)
    rename_table_identifier = harness.get_table_identifier(table_name + "_renamed")
    with pytest.raises(ValueError, match="does not exist or not authorized"):
        query = f"ALTER TABLE {table_identifier} RENAME TO {rename_table_identifier}"
        harness.run_bodo_query(query)


@pytest.mark.parametrize(
    "harness_name",
    [pytest.param("rest_test_harness", id="rest")],
)
def test_alter_table_rename_ifexists_not_found(request, harness_name: str):
    """Tests that Bodo returns gracefully when trying to rename a table that does not exist,
    when the IF EXISTS clause is used."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    table_name = harness.gen_unique_id("test_table").upper()
    table_identifier = harness.get_table_identifier(table_name)
    # Table should not exist.
    assert not harness.check_table_exists(table_identifier)
    rename_table_identifier = harness.get_table_identifier(table_name + "_renamed")
    query = (
        f"ALTER TABLE IF EXISTS {table_identifier} RENAME TO {rename_table_identifier}"
    )
    bodo_output = harness.run_bodo_query(query)
    py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
    assert_equal_par(bodo_output, py_output)


@pytest.mark.parametrize(
    "harness_name",
    [pytest.param("rest_test_harness", id="rest")],
)
def test_alter_table_rename_to_existing(request, harness_name: str):
    """
    Tests that Bodo throws an error when renaming a table to an existing table
    via ALTER TABLE.
    """
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    try:
        # Create table
        table_identifier = harness.get_table_identifier(
            harness.gen_unique_id("test_table").upper()
        )
        table_identifier_b = harness.get_table_identifier(
            harness.gen_unique_id("test_table_b").upper()
        )
        harness.create_test_table(table_identifier)
        harness.create_test_table(table_identifier_b)
        assert harness.check_table_exists(table_identifier)
        assert harness.check_table_exists(table_identifier_b)

        # Query
        with pytest.raises(ValueError, match="already exists"):
            harness.run_bodo_query(
                f"ALTER TABLE {table_identifier} RENAME TO {table_identifier_b}"
            )
    finally:
        harness.drop_test_table(table_identifier)
        harness.drop_test_table(table_identifier_b)


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_unsupported_commands(request, harness_name: str):
    """Tests that Bodo throws an error when running unsupported ALTER commands."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    # Unsupported query
    query = "ALTER VIEW placeholder_name SET SECURE"

    # This should throw an error
    with pytest.raises(ValueError, match="Unable to parse"):
        harness.run_bodo_query(query)

    # Unsupported query
    query = "ALTER TABLE placeholder_name SWAP WITH placeholder_name_swap"

    # This should throw an error
    with pytest.raises(ValueError, match="currently unsupported"):
        harness.run_bodo_query(query)

    # Unsupported query
    query = "ALTER TABLE placeholder_name CLUSTER BY junk_column"

    # This should throw an error
    with pytest.raises(ValueError, match="Unable to parse"):
        harness.run_bodo_query(query)


# ALTER TABLE SET PROPERTY


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_set_property(request, harness_name: str):
    """Tests that Bodo can set table properties."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name)
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)
        # Verify no properties are set
        output = harness.show_table_properties(table_identifier)
        for i in [1, 2, 3]:
            test_row = {"key": f"test_tag{i}", "value": f"test_value{i}"}
            assert not harness.check_row_exists(output, test_row)

        # Set a property on the table
        query = f"ALTER TABLE {table_identifier} SET PROPERTY 'test_tag1'='test_value1'"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        # Verify the properties were set
        output = harness.show_table_properties(table_identifier)
        test_row = {"key": "test_tag1", "value": "test_value1"}
        assert harness.check_row_exists(output, test_row)

        # Set multiple properties
        query = f"ALTER TABLE {table_identifier} SET PROPERTY 'test_tag2'='test_value2', 'test_tag3'='test_value3'"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        # Verify the properties were set
        output = harness.show_table_properties(table_identifier)
        for i in [2, 3]:
            test_row = {"key": f"test_tag{i}", "value": f"test_value{i}"}
            assert harness.check_row_exists(output, test_row)
    finally:
        harness.drop_test_table(table_identifier)


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_set_property_rename(request, harness_name: str):
    """Tests that Bodo can rename table properties."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name)
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)

        # Create property
        query = f"ALTER TABLE {table_identifier} SET PROPERTY 'test_tag1'='test_value1'"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        # Check it exists
        output = harness.show_table_properties(table_identifier)
        test_row = {"key": "test_tag1", "value": "test_value1"}
        assert harness.check_row_exists(output, test_row)

        # Rename tag
        query = f"ALTER TABLE {table_identifier} SET PROPERTY 'test_tag1'='test_value2'"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        # Check that it was successfully renamed
        output = harness.show_table_properties(table_identifier)
        test_row = {"key": "test_tag1", "value": "test_value1"}
        assert not harness.check_row_exists(output, test_row)
        test_row = {"key": "test_tag1", "value": "test_value2"}
        assert harness.check_row_exists(output, test_row)
    finally:
        harness.drop_test_table(table_identifier)


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_set_property_error(request, harness_name: str):
    """Tests that ALTER TABLE SET PROPERTIES will error appropriately
    on malformed queries."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    # Invalid query
    with pytest.raises(ValueError, match="Unable to parse SQL Query"):
        query = "ALTER TABLE placeholder_table SET PROPERTY"
        harness.run_bodo_query(query)

    # Missing property value
    with pytest.raises(ValueError, match="Unable to parse SQL Query."):
        query = "ALTER TABLE placeholder_table SET PROPERTY 'key'"
        harness.run_bodo_query(query)

    # Invalid property key (non-string)
    with pytest.raises(ValueError, match="Unable to parse SQL Query."):
        query = "ALTER TABLE placeholder_table SET PROPERTY invalid_key = 'value'"
        harness.run_bodo_query(query)

    # Invalid property value (non-string)
    with pytest.raises(ValueError, match="Unable to parse SQL Query."):
        query = "ALTER TABLE placeholder_table SET PROPERTY 'key' = 123"
        harness.run_bodo_query(query)


# ALTER TABLE UNSET PROPERTY


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_unset_property(request, harness_name: str):
    """Tests that Bodo can unset table properties."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    table_name = harness.gen_unique_id("TEST_TABLE").upper()
    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name)
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)

        # set 3 properties
        query = f"ALTER TABLE {table_identifier} SET PROPERTY 'test_tag1'='test_value1', 'test_tag2'='test_value2', 'test_tag3'='test_value3'"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        # Check that properties were set
        output = harness.show_table_properties(table_identifier)
        for i in [1, 2, 3]:
            test_row = {"key": f"test_tag{i}", "value": f"test_value{i}"}
            assert harness.check_row_exists(output, test_row)

        # unset 1 property
        query = f"ALTER TABLE {table_identifier} UNSET PROPERTY 'test_tag1'"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)

        # Check that properties were unset
        output = harness.show_table_properties(table_identifier)
        test_row = {"key": "test_tag1", "value": "test_value1"}
        assert not harness.check_row_exists(output, test_row)
        test_row = {"key": "test_tag2", "value": "test_value2"}
        assert harness.check_row_exists(output, test_row)
        test_row = {"key": "test_tag3", "value": "test_value3"}
        assert harness.check_row_exists(output, test_row)

        # unset 2 properties
        query = (
            f"ALTER TABLE {table_identifier} UNSET PROPERTIES 'test_tag2', 'test_tag3'"
        )
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)

        # Check that properties were unset
        output = harness.show_table_properties(table_identifier)
        for i in [1, 2, 3]:
            test_row = {"key": f"test_tag{i}", "value": f"test_value{i}"}
            assert not harness.check_row_exists(output, test_row)

    finally:
        harness.drop_test_table(table_identifier)


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_unset_property_error(request, harness_name: str):
    """Tests that ALTER TABLE UNSET PROPERTIES will error appropriately
    on malformed queries."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    table_name = harness.gen_unique_id("TEST_TABLE").upper()
    table_identifier = harness.get_table_identifier(table_name)
    try:
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)
        # set property
        query = f"ALTER TABLE {table_identifier} SET PROPERTY 'test_tag1'='test_value1'"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        # Check that properties were set
        output = harness.show_table_properties(table_identifier)
        test_row = {"key": "test_tag1", "value": "test_value1"}
        assert harness.check_row_exists(output, test_row)

        # unset non-existent property
        with pytest.raises(
            ValueError, match="Property nonexistent_tag does not exist."
        ):
            query = f"ALTER TABLE {table_identifier} UNSET PROPERTY 'nonexistent_tag'"
            bodo_output = harness.run_bodo_query(query)

        # Unset non-existent property with IF EXISTS tag (should not error)
        query = f"ALTER TABLE {table_identifier} UNSET PROPERTY IF EXISTS 'nonexistent_tag', 'test_tag1'"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        # Check that properties were unset
        output = harness.show_table_properties(table_identifier)
        test_row = {"key": "test_tag1", "value": "test_value1"}
        assert not harness.check_row_exists(output, test_row)

    finally:
        harness.drop_test_table(table_identifier)


# SET / UNSET COMMENT


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_comment(request, harness_name: str):
    """Tests that Bodo can set/unset table comments."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name)
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)

        def assert_table_has_comment(expected_comment):
            output = harness.describe_table_extended(table_identifier)
            comment_str = output.loc[
                output["col_name"] == "Comment", "data_type"
            ].values[0]
            return comment_str == expected_comment

        # Set comment
        query = f"ALTER TABLE {table_identifier} SET COMMENT 'test_comment1'"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        # Check that comment was set
        assert assert_table_has_comment("test_comment1")

        # Set empty comment while renaming
        query = f"ALTER TABLE {table_identifier} SET COMMENT ''"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        assert assert_table_has_comment("")

        # Remove comment
        query = f"ALTER TABLE {table_identifier} UNSET COMMENT"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        harness.refresh_table(table_identifier)
        output = harness.describe_table_extended(table_identifier)
        assert not (output["col_name"] == "Comment").any(), (
            "Comment is not unset correctly"
        )

    finally:
        harness.drop_test_table(table_identifier)


# ADD COLUMN


def get_sqlnode_type_names():
    # A dict of types and the sqlnode names types they should be converted to.
    # Note these are not the same as the iceberg type names -- refer to the BodoSQL ALTER TABLE docs for that.
    return {
        # iceberg type: decimal(38, 0)
        "NUMBER": "DECIMAL(38, 0)",
        "DECIMAL": "DECIMAL(38, 0)",
        "NUMERIC": "DECIMAL(38, 0)",
        # iceberg type: decimal(p, s)
        "DECIMAL(10, 5)": "DECIMAL(10, 5)",
        "NUMBER(10, 5)": "DECIMAL(10, 5)",
        # iceberg type: int
        "INT": "INTEGER",
        "INTEGER": "INTEGER",
        "SMALLINT": "INTEGER",
        "TINYINT": "INTEGER",
        "BYTEINT": "INTEGER",
        # iceberg type: long
        "BIGINT": "BIGINT",
        # iceberg type: float
        "FLOAT": "FLOAT",
        "FLOAT4": "FLOAT",
        "FLOAT8": "FLOAT",
        # iceberg type: double
        "DOUBLE": "DOUBLE",
        "DOUBLE PRECISION": "DOUBLE",
        "REAL": "DOUBLE",
        # iceberg type: string
        "VARCHAR": "VARCHAR",
        "CHAR": "VARCHAR",
        "CHARACTER": "VARCHAR",
        "STRING": "VARCHAR",
        "TEXT": "VARCHAR",
        "BINARY": "VARCHAR",
        "VARBINARY": "VARCHAR",
        # iceberg type: boolean
        "BOOLEAN": "BOOLEAN",
        # iceberg type: date
        "DATE": "DATE",
        # iceberg type: time
        # Spark doesn't support TIME type
        # "TIME": "TIME(6)",
        # iceberg type: timestamp
        "DATETIME": "TIMESTAMP(6)",
        "TIMESTAMP": "TIMESTAMP(6)",
        "TIMESTAMP_NTZ": "TIMESTAMP(6)",
        # iceberg type: timestamptz (but not supported officially by bodo)
        "TIMESTAMP_LTZ": "TIMESTAMP_WITH_LOCAL_TIME_ZONE(6)",
        "TIMESTAMP_TZ": "TIMESTAMP_WITH_LOCAL_TIME_ZONE(6)",
    }


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_add_column(request, harness_name: str):
    """
    Tests that Bodo can add columns of the appropriate type.
    """
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name)
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)
        harness.refresh_table(table_identifier)
        sqlnode_type_names = get_sqlnode_type_names()
        # # Convert to list to maintain order throughout testing
        typeNames = list(sqlnode_type_names.keys())
        for t in typeNames:
            query = f"ALTER TABLE {table_identifier} add column COL_{t.translate(str.maketrans('(), ', '____'))} {t}"
            py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
            bodo_output = harness.run_bodo_query(query)
            assert_equal_par(bodo_output, py_output)

        # Drop extraneous column created during table creation
        harness.run_spark_query(f"ALTER TABLE {table_identifier} DROP COLUMN A")
        # Check column names and types
        output = harness.describe_table(table_identifier)
        # Create dataframe with expected output
        data = {
            "NAME": [
                "COL_" + name.translate(str.maketrans("(), ", "____"))
                for name in typeNames
            ],
            "TYPE": [sqlnode_type_names[name] for name in typeNames],
        }
        answer = pd.DataFrame(data)

        assert_equal_par(trim_describe_table_output(output), answer)

    finally:
        harness.drop_test_table(table_identifier)


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_add_column_ifnotexists(request, harness_name: str):
    """Tests that Bodo exhibits appropriate behavior when adding columns with IF NOT EXISTS."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    table_name = harness.gen_unique_id("test_table").upper()
    table_identifier = harness.get_table_identifier(table_name)
    try:
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)

        # Preexisting column name
        with pytest.raises(ValueError, match="Cannot add column, name already exists"):
            query = f"ALTER TABLE {table_identifier} add column A integer"
            bodo_output = harness.run_bodo_query(query)

        # Preexisting column name with IF NOT EXISTS
        query = f"ALTER TABLE {table_identifier} add column IF NOT EXISTS A integer"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

    finally:
        harness.drop_test_table(table_identifier)


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_drop_column(request, harness_name: str):
    """Tests that Bodo can drop columns and nested columns."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name)
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)
        harness.refresh_table(table_identifier)

        # Create test columns
        harness.run_spark_query(
            f"ALTER TABLE {table_identifier} add column TESTCOL1 INT"
        )
        harness.run_spark_query(
            f"ALTER TABLE {table_identifier} add column TESTCOL2 struct<X: double, Y: double>"
        )
        harness.run_spark_query(
            f"ALTER TABLE {table_identifier} add column TESTCOL3 INT"
        )
        harness.run_spark_query(
            f"ALTER TABLE {table_identifier} add column TESTCOL4 INT"
        )
        # Drop extraneous column created during table creation
        harness.run_spark_query(f"ALTER TABLE {table_identifier} DROP COLUMN A")
        # Check
        output = harness.describe_table(table_identifier)
        answer = pd.DataFrame(
            {
                "NAME": ["TESTCOL1", "TESTCOL2", "TESTCOL3", "TESTCOL4"],
                "TYPE": [
                    "INTEGER",
                    "RecordType(DOUBLE X, DOUBLE Y)",
                    "INTEGER",
                    "INTEGER",
                ],
            }
        )

        assert_equal_par(trim_describe_table_output(output), answer)

        # Drop top level column
        query = f"ALTER TABLE {table_identifier} DROP COLUMN TESTCOL1"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
        # Check
        output = harness.describe_table(table_identifier)
        answer = pd.DataFrame(
            {
                "NAME": ["TESTCOL2", "TESTCOL3", "TESTCOL4"],
                "TYPE": ["RecordType(DOUBLE X, DOUBLE Y)", "INTEGER", "INTEGER"],
            }
        )

        assert_equal_par(trim_describe_table_output(output), answer)

        # Drop nested column
        query = f"ALTER TABLE {table_identifier} DROP COLUMN TESTCOL2.X"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
        # Check
        output = harness.describe_table(table_identifier)
        answer = pd.DataFrame(
            {
                "NAME": ["TESTCOL2", "TESTCOL3", "TESTCOL4"],
                "TYPE": ["RecordType(DOUBLE Y)", "INTEGER", "INTEGER"],
            }
        )
        assert_equal_par(trim_describe_table_output(output), answer)

        # Drop top level column of nested column
        query = f"ALTER TABLE {table_identifier} DROP COLUMN TESTCOL2"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
        # Check
        output = harness.describe_table(table_identifier)
        answer = pd.DataFrame(
            {"NAME": ["TESTCOL3", "TESTCOL4"], "TYPE": ["INTEGER", "INTEGER"]}
        )
        assert_equal_par(trim_describe_table_output(output), answer)

        # Drop multiple columns
        query = f"ALTER TABLE {table_identifier} DROP COLUMN TESTCOL3, TESTCOL4"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
        # Check
        output = harness.describe_table(table_identifier)
        answer = pd.DataFrame({"NAME": [], "TYPE": []})
        assert_equal_par(trim_describe_table_output(output), answer)

    finally:
        harness.drop_test_table(table_identifier)


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_drop_column_ifexists(request, harness_name: str):
    """Tests that Bodo can drop columns and nested columns
    with the IF EXISTS keyword."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name)
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)
        harness.refresh_table(table_identifier)

        # Create test columns
        harness.run_spark_query(
            f"ALTER TABLE {table_identifier} add column TESTCOL1 INT"
        )
        harness.run_spark_query(
            f"ALTER TABLE {table_identifier} add column TESTCOL2 struct<X: double, Y: double>"
        )
        # Drop extraneous column created during table creation
        harness.run_spark_query(f"ALTER TABLE {table_identifier} drop column A")
        harness.run_spark_query(f"REFRESH TABLE {table_identifier}")
        # Check
        output = harness.describe_table(table_identifier)
        answer = pd.DataFrame(
            {
                "NAME": ["TESTCOL1", "TESTCOL2"],
                "TYPE": ["INTEGER", "RecordType(DOUBLE X, DOUBLE Y)"],
            }
        )
        assert_equal_par(trim_describe_table_output(output), answer)

        # Drop non-existent column
        with pytest.raises(ValueError, match="Cannot delete missing column"):
            query = f"ALTER TABLE {table_identifier} DROP COLUMN TESTCOL3"
            bodo_output = harness.run_bodo_query(query)

        # Drop with IF EXISTS -- should not error
        query = f"ALTER TABLE {table_identifier} DROP COLUMN IF EXISTS TESTCOL3"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        # Drop non-existent nested column with IF EXISTS -- should not error
        query = f"ALTER TABLE {table_identifier} DROP COLUMN IF EXISTS TESTCOL1.X"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        # Drop columns now
        query = f"ALTER TABLE {table_identifier} DROP COLUMN IF EXISTS TESTCOL1"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        output = harness.describe_table(table_identifier)
        answer = pd.DataFrame(
            {"NAME": ["TESTCOL2"], "TYPE": ["RecordType(DOUBLE X, DOUBLE Y)"]}
        )
        assert_equal_par(trim_describe_table_output(output), answer)

    finally:
        harness.drop_test_table(table_identifier)


# ALTER TABLE RENAME COLUMN


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_rename_column(request, harness_name: str):
    """Tests that Bodo can drop columns and nested columns
    with the IF EXISTS keyword."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    assert isinstance(harness, DDLTestHarness)
    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name)
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)
        harness.refresh_table(table_identifier)

        # Create test columns
        harness.run_spark_query(
            f"ALTER TABLE {table_identifier} add column TESTCOL1 INT"
        )
        harness.run_spark_query(
            f"ALTER TABLE {table_identifier} add column TESTCOL2 struct<X: double, Y: double>"
        )
        harness.run_spark_query(
            f"ALTER TABLE {table_identifier} add column TESTCOL3 INT"
        )
        # Drop extraneous column created during table creation
        harness.run_spark_query(f"ALTER TABLE {table_identifier} drop column A")
        # Check
        output = harness.describe_table(table_identifier)
        answer = pd.DataFrame(
            {
                "NAME": ["TESTCOL1", "TESTCOL2", "TESTCOL3"],
                "TYPE": ["INTEGER", "RecordType(DOUBLE X, DOUBLE Y)", "INTEGER"],
            }
        )
        assert_equal_par(trim_describe_table_output(output), answer)

        # Rename top level column
        query = (
            f"ALTER TABLE {table_identifier} RENAME COLUMN TESTCOL1 TO TESTCOL1_RENAMED"
        )
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
        # Check
        output = harness.describe_table(table_identifier)
        answer = pd.DataFrame(
            {
                "NAME": ["TESTCOL1_RENAMED", "TESTCOL2", "TESTCOL3"],
                "TYPE": [
                    "INTEGER",
                    "RecordType(DOUBLE X, DOUBLE Y)",
                    "INTEGER",
                ],
            }
        )
        assert_equal_par(trim_describe_table_output(output), answer)

        # Rename again
        query = f"ALTER TABLE {table_identifier} RENAME COLUMN TESTCOL1_RENAMED TO TESTCOL1_RENAMED2"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
        # Check
        output = harness.describe_table(table_identifier)
        answer = pd.DataFrame(
            {
                "NAME": ["TESTCOL1_RENAMED2", "TESTCOL2", "TESTCOL3"],
                "TYPE": [
                    "INTEGER",
                    "RecordType(DOUBLE X, DOUBLE Y)",
                    "INTEGER",
                ],
            }
        )
        assert_equal_par(trim_describe_table_output(output), answer)

        # Rename nested column
        # Note that we cannot change the hierarchy of nesting.
        # The nested field to be renamed should be specified using the dot syntax;
        # which will then be renamed to the new name (without changing the hierarchy).
        query = f"ALTER TABLE {table_identifier} RENAME COLUMN TESTCOL2.X TO X_RENAMED"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
        # Check
        output = harness.describe_table(table_identifier)
        answer = pd.DataFrame(
            {
                "NAME": ["TESTCOL1_RENAMED2", "TESTCOL2", "TESTCOL3"],
                "TYPE": [
                    "INTEGER",
                    "RecordType(DOUBLE X_RENAMED, DOUBLE Y)",
                    "INTEGER",
                ],
            }
        )
        assert_equal_par(trim_describe_table_output(output), answer)

        # Rename to existing column (should error)
        query = f"ALTER TABLE {table_identifier} RENAME COLUMN TESTCOL1_RENAMED2 TO TESTCOL3"
        with pytest.raises(ValueError, match="already exists; cannot rename"):
            bodo_output = harness.run_bodo_query(query)

        with pytest.raises(ValueError, match="Cannot rename missing column"):
            # Rename non-existent column (should error)
            query = f"ALTER TABLE {table_identifier} RENAME COLUMN TESTCOL4 to TESTCOL4_RENAMED"
            harness.run_bodo_query(query)

    finally:
        harness.drop_test_table(table_identifier)


# ALTER TABLE ALTER COLUMN

# COLUMN COMMENT


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_alter_column_comment(request, harness_name: str):
    """Tests that Bodo can alter column comments."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name)
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)
        harness.refresh_table(table_identifier)

        # Create test columns
        harness.run_spark_query(
            f"ALTER TABLE {table_identifier} add column TESTCOL1 INT"
        )
        harness.run_spark_query(
            f"ALTER TABLE {table_identifier} add column TESTCOL2 struct<X: double, Y: double>"
        )
        # Drop extraneous column created during table creation
        harness.run_spark_query(f"ALTER TABLE {table_identifier} drop column A")
        # Check
        output = harness.describe_table(table_identifier)
        answer = pd.DataFrame(
            {
                "NAME": ["TESTCOL1", "TESTCOL2"],
                "COMMENT": [
                    None,
                    None,
                ],
            }
        )
        assert_equal_par(output[["NAME", "COMMENT"]], answer)

        # Change column comment
        query = f"ALTER TABLE {table_identifier} ALTER COLUMN TESTCOL1 COMMENT 'test_comment1'"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
        # Check
        output = harness.describe_table(table_identifier, spark=True)
        answer = pd.DataFrame(
            {
                "col_name": ["TESTCOL1", "TESTCOL2"],
                "comment": [
                    "test_comment1",
                    None,
                ],
            }
        )
        assert_equal_par(output[["col_name", "comment"]], answer)

        # Change column comment again
        query = f"ALTER TABLE {table_identifier} ALTER COLUMN TESTCOL1 COMMENT 'test_comment2'"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
        # Check
        output = harness.describe_table(table_identifier, spark=True)
        answer = pd.DataFrame(
            {
                "col_name": ["TESTCOL1", "TESTCOL2"],
                "comment": [
                    "test_comment2",
                    None,
                ],
            }
        )
        assert_equal_par(output[["col_name", "comment"]], answer)

        # Comment on nested column (should do nothing)
        query = f"ALTER TABLE {table_identifier} ALTER COLUMN TESTCOL2.X COMMENT 'test_comment2'"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
        # Check
        output = harness.describe_table(table_identifier, spark=True)
        answer = pd.DataFrame(
            {
                "col_name": ["TESTCOL1", "TESTCOL2"],
                "comment": [
                    "test_comment2",
                    None,
                ],
            }
        )
        assert_equal_par(output[["col_name", "comment"]], answer)

        # comment on nonexistent column
        with pytest.raises(
            ValueError, match="Invalid column name or column does not exist"
        ):
            query = f"ALTER TABLE {table_identifier} ALTER COLUMN NONEXISTENT_COLUMN COMMENT 'test_comment'"
            bodo_output = harness.run_bodo_query(query)

    finally:
        harness.drop_test_table(table_identifier)


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_alter_table_alter_column_dropnotnull(request, harness_name: str):
    """Tests that Bodo can change a column to nullable."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name)
        harness.run_spark_query(
            f"CREATE OR REPLACE TABLE {table_identifier} (A INT NOT NULL)"
        )
        assert harness.check_table_exists(table_identifier)

        # Check that the column is not nullable
        output = harness.describe_table(table_identifier)
        assert output.loc[output["NAME"] == "A", "NULL?"].values[0] == "N"

        # Change column to nullable
        query = f"ALTER TABLE {table_identifier} ALTER COLUMN A DROP NOT NULL"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        # Check that it is nullable
        output = harness.describe_table(table_identifier)
        assert output.loc[output["NAME"] == "A", "NULL?"].values[0] == "Y"

        # Try changing an already nullable column
        query = f"ALTER TABLE {table_identifier} ALTER COLUMN A DROP NOT NULL"
        py_output = pd.DataFrame({"STATUS": ["Statement executed successfully."]})
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)

        # Should remain unchanged
        output = harness.describe_table(table_identifier)
        assert output.loc[output["NAME"] == "A", "NULL?"].values[0] == "Y"

    finally:
        harness.drop_test_table(table_identifier)


# SHOW


@pytest.mark.parametrize(
    "harness_name, db_schema",
    [
        pytest.param("rest_test_harness", "BODOSQL_SHOW_TESTS", id="rest"),
        pytest.param("filesystem_test_harness", "iceberg_db", id="filesystem"),
    ],
)
def test_show_tables_terse(request, harness_name: str, db_schema: str):
    """Tests that Bodo can show tables in a terse format."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    table_name = harness.gen_unique_id("test_table").upper()
    db_schema = harness.gen_unique_id(db_schema).upper()
    table_identifier = harness.get_table_identifier(table_name, db_schema)
    try:
        # Create blank schema
        query = f"CREATE SCHEMA {db_schema}"
        harness.run_spark_query(query)
        harness.check_schema_exists(db_schema)
        # Create table
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)
        # Show tables
        query = f"SHOW TERSE TABLES in {db_schema}"
        py_output = pd.DataFrame(
            {
                "CREATED_ON": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "NAME": [table_name],
                "KIND": ["TABLE"],
                "SCHEMA_NAME": [db_schema.replace('"', "")],
            }
        )
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
    finally:
        harness.drop_test_table(table_identifier)
        harness.run_spark_query(f"DROP SCHEMA {db_schema}")


@pytest.mark.parametrize(
    "harness_name, db_schema",
    [
        pytest.param("rest_test_harness", "BODOSQL_SHOW_TESTS", id="rest"),
        pytest.param("filesystem_test_harness", "iceberg_db", id="filesystem"),
    ],
)
def test_show_tables(request, harness_name: str, db_schema: str):
    """Tests that Bodo can show tables in the full format."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    table_name = harness.gen_unique_id("test_table").upper()
    db_schema = harness.gen_unique_id(db_schema).upper()
    table_identifier = harness.get_table_identifier(table_name, db_schema)
    try:
        # Create blank schema
        query = f"CREATE SCHEMA {db_schema}"
        harness.run_spark_query(query)
        harness.check_schema_exists(db_schema)
        # Create table
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)
        # Show tables
        query = f"SHOW TABLES in {db_schema}"
        py_output = pd.DataFrame(
            {
                "CREATED_ON": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "NAME": [table_name.replace('"', "")],
                "SCHEMA_NAME": [db_schema.replace('"', "")],
                "KIND": ["TABLE"],
                "COMMENT": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "CLUSTER_BY": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "ROWS": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "BYTES": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "OWNER": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "RETENTION_TIME": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "AUTOMATIC_CLUSTERING": pd.array(
                    [None], dtype=pd.ArrowDtype(pa.string())
                ),
                "CHANGE_TRACKING": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "IS_EXTERNAL": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "ENABLE_SCHEMA_EVOLUTION": pd.array(
                    [None], dtype=pd.ArrowDtype(pa.string())
                ),
                "OWNER_ROLE_TYPE": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "IS_EVENT": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "IS_HYBRID": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "IS_ICEBERG": ["Y"],
                "IS_IMMUTABLE": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
            }
        )
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
    finally:
        harness.drop_test_table(table_identifier)
        harness.run_spark_query(f"DROP SCHEMA {db_schema}")


@pytest.mark.parametrize(
    "harness_name, db_schema",
    [pytest.param("rest_test_harness", "BODOSQL_SHOW_TESTS", id="rest")],
)
def test_show_views_terse(request, harness_name: str, db_schema: str):
    """Tests that Bodo can show views in a terse format."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    view_name = harness.gen_unique_id("test_view").upper()
    db_schema = harness.gen_unique_id(db_schema).upper()
    view_identifier = harness.get_table_identifier(view_name, db_schema)
    try:
        # Create blank schema
        query = f"CREATE SCHEMA {db_schema}"
        harness.run_spark_query(query)
        harness.check_schema_exists(db_schema)
        # Create view
        harness.create_test_view(view_identifier)
        assert harness.check_view_exists(view_identifier)
        # Show views
        query = f"SHOW TERSE VIEWS in {db_schema}"
        py_output = pd.DataFrame(
            {
                "CREATED_ON": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "NAME": [view_name],
                "KIND": ["VIEW"],
                "SCHEMA_NAME": [db_schema.replace('"', "")],
            }
        )
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
    finally:
        harness.drop_test_view(view_identifier)
        harness.run_spark_query(f"DROP SCHEMA {db_schema}")


@pytest.mark.parametrize(
    "harness_name, db_schema",
    [pytest.param("rest_test_harness", "BODOSQL_SHOW_TESTS", id="rest")],
)
def test_show_views(request, harness_name: str, db_schema: str):
    """Tests that Bodo can show views in a full format."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    view_name = harness.gen_unique_id("test_view").upper()
    db_schema = harness.gen_unique_id(db_schema).upper()
    view_identifier = harness.get_table_identifier(view_name, db_schema)
    try:
        # Create blank schema
        query = f"CREATE SCHEMA {db_schema}"
        harness.run_spark_query(query)
        harness.check_schema_exists(db_schema)
        # Create view
        harness.create_test_view(view_identifier)
        assert harness.check_view_exists(view_identifier)
        # Show views
        query = f"SHOW VIEWS in {db_schema}"
        py_output = pd.DataFrame(
            {
                "CREATED_ON": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "NAME": [view_name.replace('"', "")],
                "RESERVED": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "SCHEMA_NAME": [db_schema.replace('"', "")],
                "COMMENT": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "OWNER": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "TEXT": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "IS_SECURE": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "IS_MATERIALIZED": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "OWNER_ROLE_TYPE": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "CHANGE_TRACKING": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
            }
        )
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
    finally:
        harness.drop_test_view(view_identifier)
        harness.run_spark_query(f"DROP SCHEMA {db_schema}")


@pytest.mark.parametrize(
    "harness_name, db_schema",
    [
        pytest.param("rest_test_harness", "BODOSQL_SHOW_TESTS", id="rest"),
        pytest.param("filesystem_test_harness", "iceberg_db", id="filesystem"),
    ],
)
def test_show_objects_terse(request, harness_name: str, db_schema: str):
    """Tests that Bodo can show objects in a terse format."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    table_name = harness.gen_unique_id("test_table").upper()
    db_schema = harness.gen_unique_id(db_schema).upper()
    table_identifier = harness.get_table_identifier(table_name, db_schema)
    try:
        # Create blank schema
        query = f"CREATE SCHEMA {db_schema}"
        harness.run_spark_query(query)
        harness.check_schema_exists(db_schema)
        # Create test table
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)
        # Show objects
        query = f"SHOW TERSE OBJECTS in {db_schema}"
        py_output = pd.DataFrame(
            {
                "CREATED_ON": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                "NAME": [table_name],
                "KIND": ["TABLE"],
                "SCHEMA_NAME": [db_schema.replace('"', "")],
            }
        )
        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
    finally:
        harness.drop_test_table(table_identifier)
        harness.run_spark_query(f"DROP SCHEMA {db_schema}")


@pytest.mark.parametrize(
    "harness_name, db_schema, test_views",
    [
        pytest.param("rest_test_harness", "BODOSQL_SHOW_TESTS", True, id="rest-views"),
        pytest.param(
            "filesystem_test_harness", "iceberg_db", False, id="filesystem-no_views"
        ),
    ],
)
def test_show_objects(request, harness_name: str, db_schema: str, test_views: bool):
    """Tests that Bodo can show objects in a terse format."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    table_name = harness.gen_unique_id("test_table").upper()
    db_schema = harness.gen_unique_id(db_schema).upper()
    table_identifier = harness.get_table_identifier(table_name, db_schema)
    view_name = harness.gen_unique_id("test_view").upper()
    view_identifier = harness.get_table_identifier(view_name, db_schema)
    try:
        # Create blank schema
        query = f"CREATE SCHEMA {db_schema}"
        harness.run_spark_query(query)
        harness.check_schema_exists(db_schema)
        # Create test table
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)
        # Create view (not for filesystem)
        if test_views:
            harness.create_test_view(view_identifier)
            assert harness.check_view_exists(view_identifier)
            py_output = pd.DataFrame(
                {
                    "CREATED_ON": pd.array(
                        [None, None], dtype=pd.ArrowDtype(pa.string())
                    ),
                    "NAME": [table_name, view_name],
                    "SCHEMA_NAME": [db_schema, db_schema],
                    "KIND": ["TABLE", "VIEW"],
                    "COMMENT": pd.array([None, None], dtype=pd.ArrowDtype(pa.string())),
                    "CLUSTER_BY": pd.array(
                        [None, None], dtype=pd.ArrowDtype(pa.string())
                    ),
                    "ROWS": pd.array([None, None], dtype=pd.ArrowDtype(pa.string())),
                    "BYTES": pd.array([None, None], dtype=pd.ArrowDtype(pa.string())),
                    "OWNER": pd.array([None, None], dtype=pd.ArrowDtype(pa.string())),
                    "RETENTION_TIME": pd.array(
                        [None, None], dtype=pd.ArrowDtype(pa.string())
                    ),
                    "OWNER_ROLE_TYPE": pd.array(
                        [None, None], dtype=pd.ArrowDtype(pa.string())
                    ),
                }
            )
        else:
            py_output = pd.DataFrame(
                {
                    "CREATED_ON": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                    "NAME": [table_name],
                    "SCHEMA_NAME": [db_schema],
                    "KIND": ["TABLE"],
                    "COMMENT": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                    "CLUSTER_BY": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                    "ROWS": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                    "BYTES": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                    "OWNER": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
                    "RETENTION_TIME": pd.array(
                        [None], dtype=pd.ArrowDtype(pa.string())
                    ),
                    "OWNER_ROLE_TYPE": pd.array(
                        [None], dtype=pd.ArrowDtype(pa.string())
                    ),
                }
            )
        # Show objects
        query = f"SHOW OBJECTS in {db_schema}"

        bodo_output = harness.run_bodo_query(query)
        assert_equal_par(bodo_output, py_output)
    finally:
        harness.drop_test_table(table_identifier)
        if test_views:
            harness.drop_test_view(view_identifier)
        harness.run_spark_query(f"DROP SCHEMA {db_schema}")


@pytest.mark.parametrize(
    "harness_name, db_schema",
    [pytest.param("filesystem_test_harness", "iceberg_db", id="filesystem")],
)
def test_show_schemas_terse(request, harness_name: str, db_schema: str):
    """Tests that Bodo can show schemas in a terse format."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    # Don't test rest, as doesn't support nested schemas?
    try:
        db_name = harness.gen_unique_id(db_schema).upper()
        query = f'CREATE SCHEMA "{db_name}"'
        harness.run_bodo_query(query)
        schema_name1 = harness.gen_unique_id("TEST_SCHEMA_DDL").upper()
        query = f'CREATE SCHEMA "{db_name}"."{schema_name1}"'
        harness.run_bodo_query(query)
        schema_name2 = harness.gen_unique_id("TEST_SCHEMA_DDL").upper()
        query = f'CREATE SCHEMA "{db_name}"."{schema_name2}"'
        harness.run_bodo_query(query)

        query = f'SHOW TERSE SCHEMAS IN "{db_name}"'
        py_output = pd.DataFrame(
            {
                "CREATED_ON": pd.array([None, None], dtype=pd.ArrowDtype(pa.string())),
                "NAME": [schema_name1, schema_name2],
                "KIND": pd.array([None] * 2, dtype=pd.ArrowDtype(pa.string())),
                "SCHEMA_NAME": [
                    f"{db_name}.{schema_name1}",
                    f"{db_name}.{schema_name2}",
                ],
            }
        )
        bodo_output = harness.run_bodo_query(query)
        bodo_output_sorted = bodo_output.sort_values(
            by=list(bodo_output.columns)
        ).reset_index(drop=True)
        py_output_sorted = py_output.sort_values(
            by=list(py_output.columns)
        ).reset_index(drop=True)
        assert_equal_par(bodo_output_sorted, py_output_sorted)
    finally:
        query = f'DROP SCHEMA "{db_name}"."{schema_name1}"'
        harness.run_bodo_query(query)
        query = f'DROP SCHEMA "{db_name}"."{schema_name2}"'
        harness.run_bodo_query(query)
        query = f'DROP SCHEMA "{db_name}"'
        harness.run_bodo_query(query)


@pytest.mark.parametrize(
    "harness_name, db_schema",
    [pytest.param("filesystem_test_harness", "iceberg_db", id="filesystem")],
)
def test_show_schemas(request, harness_name: str, db_schema: str):
    """Tests that Bodo can show schemas in a full format."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)
    # Don't test rest, as doesn't support nested schemas?
    try:
        db_name = harness.gen_unique_id(db_schema).upper()
        query = f'CREATE SCHEMA "{db_name}"'
        harness.run_bodo_query(query)
        schema_name1 = harness.gen_unique_id("TEST_SCHEMA_DDL").upper()
        query = f'CREATE SCHEMA "{db_name}"."{schema_name1}"'
        harness.run_bodo_query(query)
        schema_name2 = harness.gen_unique_id("TEST_SCHEMA_DDL").upper()
        query = f'CREATE SCHEMA "{db_name}"."{schema_name2}"'
        harness.run_bodo_query(query)

        query = f'SHOW SCHEMAS IN "{db_name}"'
        py_output = pd.DataFrame(
            {
                "CREATED_ON": pd.array([None, None], dtype=pd.ArrowDtype(pa.string())),
                "NAME": [schema_name1, schema_name2],
                "IS_DEFAULT": pd.array([None, None], dtype=pd.ArrowDtype(pa.string())),
                "IS_CURRENT": pd.array([None, None], dtype=pd.ArrowDtype(pa.string())),
                "DATABASE_NAME": [db_name, db_name],
                "OWNER": pd.array([None, None], dtype=pd.ArrowDtype(pa.string())),
                "COMMENT": pd.array([None, None], dtype=pd.ArrowDtype(pa.string())),
                "OPTIONS": pd.array([None, None], dtype=pd.ArrowDtype(pa.string())),
                "RETENTION_TIME": pd.array(
                    [None, None], dtype=pd.ArrowDtype(pa.string())
                ),
                "OWNER_ROLE_TYPE": pd.array(
                    [None, None], dtype=pd.ArrowDtype(pa.string())
                ),
            }
        )
        bodo_output = harness.run_bodo_query(query)
        bodo_output_sorted = bodo_output.sort_values(
            by=list(bodo_output.columns)
        ).reset_index(drop=True)
        py_output_sorted = py_output.sort_values(
            by=list(py_output.columns)
        ).reset_index(drop=True)
        assert_equal_par(bodo_output_sorted, py_output_sorted)
    finally:
        query = f'DROP SCHEMA "{db_name}"."{schema_name1}"'
        harness.run_bodo_query(query)
        query = f'DROP SCHEMA "{db_name}"."{schema_name2}"'
        harness.run_bodo_query(query)
        query = f'DROP SCHEMA "{db_name}"'
        harness.run_bodo_query(query)


@pytest.mark.parametrize(
    "harness_name",
    [
        pytest.param("rest_test_harness", id="rest"),
        pytest.param("filesystem_test_harness", id="filesystem"),
    ],
)
def test_show_tblproperties(request, harness_name: str):
    """Tests that Bodo can show table properties."""
    harness: DDLTestHarness = request.getfixturevalue(harness_name)

    try:
        table_name = harness.gen_unique_id("test_table").upper()
        table_identifier = harness.get_table_identifier(table_name)
        harness.create_test_table(table_identifier)
        assert harness.check_table_exists(table_identifier)

        # Set a few properties
        query = f"ALTER TABLE {table_identifier} SET TBLPROPERTIES 'test_tag1'='test_value1', 'test_tag2'='test_value2'"
        harness.run_bodo_query(query)

        # test SHOW to show all properties
        query = f"SHOW TBLPROPERTIES {table_identifier}"
        output = harness.run_bodo_query(query)
        answer = pd.DataFrame(
            {"KEY": ["test_tag1", "test_tag2"], "VALUE": ["test_value1", "test_value2"]}
        )
        for i in range(len(answer)):
            row = answer.iloc[i]
            assert (
                (output["KEY"] == row["KEY"]) & (output["VALUE"] == row["VALUE"])
            ).any()

        # test SHOW to show one property
        query = f"SHOW TBLPROPERTIES {table_identifier} ('test_tag1')"
        output = harness.run_bodo_query(query)
        answer = pd.DataFrame({"VALUE": ["test_value1"]})
        assert_equal_par(output, answer)

        # change a property
        query = f"ALTER TABLE {table_identifier} SET TBLPROPERTIES 'test_tag1'='new_test_value1'"
        harness.run_bodo_query(query)

        # test SHOW to see if changed
        query = f"SHOW TBLPROPERTIES {table_identifier} ('test_tag1')"
        output = harness.run_bodo_query(query)
        answer = pd.DataFrame({"VALUE": ["new_test_value1"]})
        assert_equal_par(output, answer)

        # Test aliases such as PROPERTIES or TAGS
        query = f"SHOW PROPERTIES {table_identifier} ('test_tag1')"
        output = harness.run_bodo_query(query)
        answer = pd.DataFrame({"VALUE": ["new_test_value1"]})
        assert_equal_par(output, answer)
        query = f"SHOW TAGS {table_identifier} ('test_tag1')"
        output = harness.run_bodo_query(query)
        answer = pd.DataFrame({"VALUE": ["new_test_value1"]})
        assert_equal_par(output, answer)

        # show non-existent property (should error)
        with pytest.raises(
            ValueError, match="The property nonexistent_tag was not found"
        ):
            query = f"SHOW TBLPROPERTIES {table_identifier} ('nonexistent_tag')"
            output = harness.run_bodo_query(query)

    finally:
        harness.drop_test_table(table_identifier)
