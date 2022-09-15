package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.catalog.SnowflakeCatalogImpl;
import com.bodosql.calcite.schema.LocalSchemaImpl;
import com.bodosql.calcite.table.BodoSQLColumn;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.LocalTableImpl;
import java.util.ArrayList;
import java.util.Map;
import java.util.Properties;

/** Class for locally testing codegen using a snowflake catalog */
public class SnowflakeGenTest {
  public static void main(String[] args) throws Exception {
    String sql = " select * from LINEITEM1 r full outer join localtable on l_orderkey = a";
    Map envVars = System.getenv();
    Properties prop = new Properties();
    prop.put("schema", "PUBLIC");
    prop.put("queryTimeout", 5);
    BodoSQLCatalog catalog =
        new SnowflakeCatalogImpl(
            (String) envVars.get("SF_USER"),
            (String) envVars.get("SF_PASSWORD"),
            (String) envVars.get("SF_ACCOUNT"),
            (String) envVars.get("SF_CATALOG"),
            "DEMO_WH",
            prop);
    LocalSchemaImpl schema = new LocalSchemaImpl("__bodolocal__");

    // Add a local table to also resolve
    ArrayList arr = new ArrayList();
    BodoSQLColumn.BodoSQLColumnDataType dataType = BodoSQLColumn.BodoSQLColumnDataType.INT64;
    BodoSQLColumnImpl column = new BodoSQLColumnImpl("A", dataType);
    arr.add(column);
    BodoSQLColumnImpl column2 = new BodoSQLColumnImpl("D", dataType);
    arr.add(column2);
    BodoSQLColumnImpl column3 = new BodoSQLColumnImpl("C", dataType);
    arr.add(column3);

    BodoSqlTable table = new LocalTableImpl("localtable", schema, arr, false, "localtable", "");
    schema.addTable(table);

    RelationalAlgebraGenerator generator =
        new RelationalAlgebraGenerator(catalog, schema, "dummy_param_table_name");
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
