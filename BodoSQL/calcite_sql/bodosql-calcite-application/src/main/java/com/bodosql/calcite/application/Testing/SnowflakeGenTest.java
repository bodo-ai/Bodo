package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.catalog.connection.SnowflakeCatalog;
import com.bodosql.calcite.catalog.connection.SnowflakeCatalogImpl;
import com.bodosql.calcite.schema.BodoSQLSnowflakeSchema;
import com.bodosql.calcite.schema.BodoSqlSchema;
import java.util.Map;
import java.util.Properties;

/** Class for locally testing codegen using a snowflake catalog */
public class SnowflakeGenTest {
  public static void main(String[] args) throws Exception {
    String sql = " select * from PUBLIC.LINEITEM1";
    Map envVars = System.getenv();
    Properties prop = new Properties();
    prop.put("user", envVars.get("SF_USER"));
    prop.put("password", envVars.get("SF_PASSWORD"));
    SnowflakeCatalog catalog =
        new SnowflakeCatalogImpl(
            (String) envVars.get("SF_ACCOUNT"), (String) envVars.get("SF_CATALOG"), prop);
    BodoSqlSchema schema = new BodoSQLSnowflakeSchema("PUBLIC", catalog);

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
