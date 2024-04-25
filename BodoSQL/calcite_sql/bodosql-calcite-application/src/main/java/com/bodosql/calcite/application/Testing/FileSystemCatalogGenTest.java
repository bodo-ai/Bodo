package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.application.PandasCodeSqlPlanPair;
import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.application.write.WriteTarget;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.catalog.FileSystemCatalog;
import com.bodosql.calcite.schema.LocalSchema;
import com.bodosql.calcite.traits.BatchingProperty;
import java.util.Map;

/** Class for locally testing codegen using a FileSystem Catalog */
public class FileSystemCatalogGenTest {
  public static void main(String[] args) throws Exception {
    String sql = "DROP TABLE SIMPLE_BOOL_BINARY_TABLE";
    Map envVars = System.getenv();
    BodoSQLCatalog catalog =
        new FileSystemCatalog(
            (String) envVars.get("ROOT_PATH"),
            WriteTarget.WriteTargetEnum.fromString("parquet"),
            (String) envVars.get("DEFAULT_SCHEMA"));
    LocalSchema schema = new LocalSchema("__BODOLOCAL__");

    RelationalAlgebraGenerator generator =
        new RelationalAlgebraGenerator(
            catalog,
            schema,
            RelationalAlgebraGenerator.STREAMING_PLANNER,
            0,
            1,
            BatchingProperty.defaultBatchSize,
            true, // Always hide credentials
            true, // Enable Iceberg for testing
            false, // Do not enable TIMESTAMPTZ for Iceberg testing
            true, // Enable Join Runtime filters for Testing
            "SNOWFLAKE" // Maintain case sensitivity in the Snowflake style by default
            );
    System.out.println("SQL query:");
    System.out.println(sql + "\n");
    PandasCodeSqlPlanPair pair = generator.getPandasAndPlanString(sql, true);
    System.out.println("Optimized plan:");
    System.out.println(pair.getSqlPlan() + "\n");
    System.out.println("Generated code:");
    System.out.println(pair.getPdCode() + "\n");
  }
}
