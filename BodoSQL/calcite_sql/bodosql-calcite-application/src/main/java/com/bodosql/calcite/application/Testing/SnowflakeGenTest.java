package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.catalog.SnowflakeCatalogImpl;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.CatalogSchemaImpl;
import java.util.Map;
import java.util.Properties;

/** Class for locally testing codegen using a snowflake catalog */
public class SnowflakeGenTest {
  public static void main(String[] args) throws Exception {
    String sql = " select * from PUBLIC.LINEITEM1";
    Map envVars = System.getenv();
    Properties prop = new Properties();
    prop.put("warehouse", "DEMO_WH");
    BodoSQLCatalog catalog =
        new SnowflakeCatalogImpl(
            (String) envVars.get("SF_USER"),
            (String) envVars.get("SF_PASSWORD"),
            (String) envVars.get("SF_ACCOUNT"),
            (String) envVars.get("SF_CATALOG"),
            prop);
    BodoSqlSchema schema = new CatalogSchemaImpl("PUBLIC", catalog);

    RelationalAlgebraGenerator generator =
        new RelationalAlgebraGenerator(schema, "dummy_param_table_name");
    System.out.println("SQL query:");
    System.out.println(sql + "\n");
    String unOptimizedPlanStr = generator.getRelationalAlgebraString(sql, false);
    System.out.println("UnOptimized plan:");
    System.out.println(unOptimizedPlanStr + "\n");
    String optimizedPlanStr = generator.getRelationalAlgebraString(sql, true);
    System.out.println("Optimized plan:");
    System.out.println(optimizedPlanStr + "\n");
    String pandasStr = generator.getPandasString(sql);
    System.out.println("Generated code:");
    System.out.println(pandasStr + "\n");
  }
}
