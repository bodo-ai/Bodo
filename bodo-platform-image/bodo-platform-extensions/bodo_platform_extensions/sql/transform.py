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
        from bodo_platform_utils.bodosqlwrapper import CatalogType

        # Get catalog details using bodo-platform-utils.
        credentials = catalog.get_data("{catalog_name}")
        
        # check whether the catalog is tabular or snowflake
        # add more conditions once more catalog types are supported
        
        if "icebergRestUrl" in catalog.get_data("{catalog_name}"):
            catalog_type_str = "TABULAR"
        else:
            catalog_type_str = "SNOWFLAKE"
            
        if catalog_type_str is None:
            catalog_type = CatalogType.SNOWFLAKE
        else:
            catalog_type = CatalogType(catalog_type_str)

        # BodoSQL Catalogs and Contexts need to be created outside
        # JIT functions.

        
        if catalog_type == CatalogType.TABULAR:
            bsql_catalog = bodosql.TabularCatalog(
                warehouse=credentials["warehouse"],
                rest_uri=credentials["icebergRestUrl"],
                credential=credentials["credential"],
            )
        else:
            bsql_catalog = bodosql.SnowflakeCatalog(
                    username=credentials["username"],
                    password=credentials["password"],
                    account=credentials["accountName"],
                    warehouse=credentials["warehouse"],
                    database=credentials["database"],
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
