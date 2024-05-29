package com.bodosql.calcite.application.testing;

import com.bodosql.calcite.application.write.WriteTarget;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.catalog.FileSystemCatalog;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.LocalSchema;
import java.util.Map;
import org.jetbrains.annotations.NotNull;

/** Class for locally testing codegen using a FileSystem Catalog */
public class FileSystemCatalogGenTest extends GenTestFixture {
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
    Map envVars = System.getenv();
    return new FileSystemCatalog(
        (String) envVars.get("ROOT_PATH"),
        WriteTarget.WriteTargetEnum.ICEBERG,
        (String) envVars.get("DEFAULT_SCHEMA"));
  }

  @NotNull
  @Override
  public BodoSqlSchema getSchema() {
    return new LocalSchema("__BODOLOCAL__");
  }

  public static void main(String[] args) throws Exception {
    String sql = "Select * from SIMPLE_BOOL_BINARY_TABLE";
    boolean generateCode = true;
    new FileSystemCatalogGenTest().run(sql, generateCode);
  }
}
