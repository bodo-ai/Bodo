package com.bodosql.calcite.application.testing;

import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.catalog.TabularCatalog;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.LocalSchema;
import java.util.Map;
import org.jetbrains.annotations.NotNull;

/** Class for locally testing codegen using a Tabular REST Iceberg catalog */
public class TabularCatalogGenTest extends GenTestFixture {
  @Override
  public boolean isIceberg() {
    return true;
  }

  @Override
  public boolean supportsTimestampTZ() {
    return false;
  }

  @NotNull
  @Override
  public BodoSQLCatalog getCatalog() {
    String sql = "SELECT * FROM \"examples\".\"nyc_taxi_locations\"";
    Map envVars = System.getenv();
    String credential = (String) envVars.get("TABULAR_CREDENTIAL");
    String warehouse = "Bodo-Test-Iceberg-Warehouse";
    return new TabularCatalog(warehouse, null, credential);
  }

  @NotNull
  @Override
  public BodoSqlSchema getSchema() {
    return new LocalSchema("__BODOLOCAL__");
  }

  public static void main(String[] args) throws Exception {
    String sql = "SELECT * FROM \"examples\".\"nyc_taxi_locations\"";
    boolean generateCode = true;
    new TabularCatalogGenTest().run(sql, generateCode);
  }
}
