def gen_bodosql_code(
    sql: str,
    result_var: str,
    catalog_name: str,
) -> str:
    """
    Generate boilerplate BodoSQL code that should be executed
    on each rank for the given sql code using the provided catalog.

    :param: sql (str): The SQL code to execute
    :param: result_var (str): The variable to store the output DataFrame in.
    :param: catalog_name (str): Name of the catalog registered on the Platform
      that should be used for this SQL execution. We use bodo-platform-utils
      to get the details for the specified catalog.

    :returns: (str) The boilerplate BodoSQL code to execute.
    """

    # Make all hashes positive so the leading minus sign doesn't create an invalid function name.
    # This needs to be deterministic since it will run in parallel on all ranks.
    cell_hash = hex(abs(hash(sql)))

    code = f"""
        import bodo
        import bodosql
        from bodo_platform_utils import catalog

        # Get Snowflake catalog details using bodo-platform-utils.
        # At this time only Snowflake catalogs are supported.
        sf_credentials = catalog.get_data("{catalog_name}")

        # BodoSQL Catalogs and Contexts need to be created outside
        # JIT functions.
        bsql_catalog = bodosql.SnowflakeCatalog(
                username=sf_credentials["username"],
                password=sf_credentials["password"],
                account=sf_credentials["accountName"],
                warehouse=sf_credentials["warehouse"],
                database=sf_credentials["database"],
            )

        bc = bodosql.BodoSQLContext(catalog=bsql_catalog)

        @bodo.jit
        def f_catalog_{cell_hash}(bc):
            df = bc.sql(\"\"\"{sql}\"\"\")
            print(\"Shape of output: \", df.shape)
            return df

        {result_var} = f_catalog_{cell_hash}(bc)

        if bodo.get_rank() == 0:
            print('Saved output in "{result_var}"')
    """

    return code
