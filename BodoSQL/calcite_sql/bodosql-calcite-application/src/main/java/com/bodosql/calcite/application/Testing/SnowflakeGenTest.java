package com.bodosql.calcite.application.Testing;

import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelRoot;

import com.bodosql.calcite.adapter.pandas.PandasUtilKt;
import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.catalog.SnowflakeCatalogImpl;
import com.bodosql.calcite.schema.LocalSchemaImpl;
import com.bodosql.calcite.table.BodoSQLColumn;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.LocalTableImpl;
import com.bodosql.calcite.traits.BatchingProperty;
import java.util.ArrayList;
import java.util.Map;
import java.util.Properties;

/** Class for locally testing codegen using a snowflake catalog */
public class SnowflakeGenTest {
  public static void main(String[] args) throws Exception {
    String sql = " select * from LINEITEM1 r limit 10";
    Map envVars = System.getenv();
    Properties prop = new Properties();
    prop.put("schema", "PUBLIC");
    prop.put("queryTimeout", 5);
    BodoSQLCatalog catalog =
        new SnowflakeCatalogImpl(
            (String) envVars.get("SF_USERNAME"),
            (String) envVars.get("SF_PASSWORD"),
            (String) envVars.get("SF_ACCOUNT"),
            (String) envVars.get("SF_CATALOG"),
            "DEMO_WH",
            prop);
    LocalSchemaImpl schema = new LocalSchemaImpl("__bodolocal__");

    // Add a local table to also resolve
    ArrayList arr = new ArrayList();
    BodoSQLColumn.BodoSQLColumnDataType dataType = BodoSQLColumn.BodoSQLColumnDataType.INT64;
    BodoSQLColumnImpl column = new BodoSQLColumnImpl("A", dataType, true);
    arr.add(column);
    BodoSQLColumnImpl column2 = new BodoSQLColumnImpl("D", dataType, true);
    arr.add(column2);
    BodoSQLColumnImpl column3 = new BodoSQLColumnImpl("C", dataType, true);
    arr.add(column3);

    BodoSqlTable table =
        new LocalTableImpl("localtable", schema, arr, false, "", "", false, "MEMORY", null);
    schema.addTable(table);

    RelationalAlgebraGenerator generator =
        new RelationalAlgebraGenerator(
            catalog, schema, "dummy_param_table_name", 0, 0, BatchingProperty.defaultBatchSize, false);
    System.out.println("SQL query:");
    System.out.println(sql + "\n");
    String unOptimizedPlanStr = getRelationalAlgebraString(generator, sql, false);
    System.out.println("UnOptimized plan:");
    System.out.println(unOptimizedPlanStr + "\n");
    String optimizedPlanStr = getRelationalAlgebraString(generator, sql, true);
    System.out.println("Optimized plan:");
    System.out.println(optimizedPlanStr + "\n");
    String pandasStr = generator.getPandasString(sql);
    System.out.println("Generated code:");
    System.out.println(pandasStr + "\n");
  }

  private static String getRelationalAlgebraString(RelationalAlgebraGenerator generator, String sql, boolean optimizePlan) {
    try {
      RelRoot root = generator.getRelationalAlgebra(sql, optimizePlan);
      return RelOptUtil.toString(PandasUtilKt.pandasProject(root));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}
