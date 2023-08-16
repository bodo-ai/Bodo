package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.adapter.pandas.PandasUtilKt;
import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.application.utils.RelCostWriter;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.catalog.SnowflakeCatalogImpl;
import com.bodosql.calcite.schema.LocalSchemaImpl;
import com.bodosql.calcite.table.BodoSQLColumn;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.LocalTableImpl;
import com.bodosql.calcite.traits.BatchingProperty;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Map;
import java.util.Properties;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;

/** Class for locally testing codegen using a snowflake catalog */
public class SnowflakeGenTest {
  public static void main(String[] args) throws Exception {
    String sql =
        "select\n"
            + "    n_name,\n"
            + "    sum(l_extendedprice * (1 - l_discount)) as revenue\n"
            + "from\n"
            + "    customer,\n"
            + "    orders,\n"
            + "    lineitem,\n"
            + "    supplier,\n"
            + "    nation,\n"
            + "    region\n"
            + "where\n"
            + "    c_custkey = o_custkey\n"
            + "    and l_orderkey = o_orderkey\n"
            + "    and l_suppkey = s_suppkey\n"
            + "    and c_nationkey = s_nationkey\n"
            + "    and s_nationkey = n_nationkey\n"
            + "    and n_regionkey = r_regionkey\n"
            + "    and r_name = 'ASIA'\n"
            + "    and o_orderdate >= '1994-01-01'\n"
            + "    and o_orderdate < '1995-01-01'\n"
            + "group by\n"
            + "    n_name\n"
            + "order by\n"
            + "    revenue desc\n";
    Map envVars = System.getenv();
    Properties prop = new Properties();
    prop.put("schema", "TPCH_SF10");
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
            catalog,
            schema,
            "dummy_param_table_name",
            RelationalAlgebraGenerator.STREAMING_PLANNER,
            0,
            BatchingProperty.defaultBatchSize,
            false);
    System.out.println("SQL query:");
    System.out.println(sql + "\n");
    String optimizedPlanStr = getRelationalAlgebraString(generator, sql, true);
    System.out.println("Optimized plan:");
    System.out.println(optimizedPlanStr + "\n");
    String pandasStr = generator.getPandasString(sql);
    System.out.println("Generated code:");
    System.out.println(pandasStr + "\n");
  }

  private static String getRelationalAlgebraString(
      RelationalAlgebraGenerator generator, String sql, boolean optimizePlan) {
    try {
      RelRoot root = generator.getRelationalAlgebra(sql, optimizePlan);
      RelNode newRoot = PandasUtilKt.pandasProject(root);
      StringWriter sw = new StringWriter();
      RelCostWriter costWriter = new RelCostWriter(new PrintWriter(sw), newRoot);
      newRoot.explain(costWriter);
      return sw.toString();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}
