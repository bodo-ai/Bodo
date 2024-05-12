package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.application.PandasCodeSqlPlanPair;
import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.catalog.TabularCatalog;
import com.bodosql.calcite.schema.LocalSchema;
import com.bodosql.calcite.traits.BatchingProperty;
import java.util.Map;

/** Class for locally testing codegen using a Tabular REST Iceberg catalog */
public class TabularCatalogGenTest {
  public static void main(String[] args) throws Exception {
    String sql = "SELECT * FROM \"examples\".\"nyc_taxi_locations\"";
    Map envVars = System.getenv();
    String credential = (String) envVars.get("TABULAR_CREDENTIAL");
    String warehouse = "Bodo-Test-Iceberg-Warehouse";

    BodoSQLCatalog catalog = new TabularCatalog(warehouse, null, credential);
    LocalSchema schema = new LocalSchema("__BODOLOCAL__");

    RelationalAlgebraGenerator generator =
        new RelationalAlgebraGenerator(
            catalog,
            schema,
            RelationalAlgebraGenerator.STREAMING_PLANNER,
            0,
            0,
            BatchingProperty.defaultBatchSize,
            true, // Always hide credentials
            true, // Enable Iceberg for testing
            false, // Do not enable TIMESTAMPTZ for Iceberg testing
            true, // Enable Join Runtime filters for Testing
            "SNOWFLAKE" // Maintain case sensitivity in the Snowflake style by default
            );

    // Controls if we are generating code or executing DDL. You can set this
    // to false if you want to observe the actual execution of DDL statements.
    boolean generateCode = true;

    System.out.println("SQL query:");
    System.out.println(sql + "\n");

    if (generateCode) {
      PandasCodeSqlPlanPair pair = generator.getPandasAndPlanString(sql, true);
      System.out.println("Optimized plan:");
      System.out.println(pair.getSqlPlan() + "\n");
      System.out.println("Generated code:");
      System.out.println(pair.getPdCode() + "\n");
    } else {
      System.out.println("DDL OUTPUT:");
      System.out.println(generator.executeDDL(sql));
    }
  }
}
