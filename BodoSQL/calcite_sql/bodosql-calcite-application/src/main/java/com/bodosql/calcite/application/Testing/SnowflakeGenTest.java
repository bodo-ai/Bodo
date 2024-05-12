package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.adapter.bodo.BodoUtilKt;
import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.application.utils.RelCostAndMetaDataWriter;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.catalog.SnowflakeCatalog;
import com.bodosql.calcite.schema.LocalSchema;
import com.bodosql.calcite.table.BodoSQLColumn;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.ColumnDataTypeInfo;
import com.bodosql.calcite.table.LocalTable;
import com.bodosql.calcite.traits.BatchingProperty;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Map;
import java.util.Properties;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.commons.lang3.tuple.Pair;

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
    BodoSQLCatalog catalog =
        new SnowflakeCatalog(
            (String) envVars.get("SF_USERNAME"),
            (String) envVars.get("SF_PASSWORD"),
            (String) envVars.get("SF_ACCOUNT"),
            (String) envVars.get("SF_DATABASE"),
            "DEMO_WH",
            prop,
            null);
    LocalSchema schema = new LocalSchema("__BODOLOCAL__");

    // Add a local table to also resolve
    ArrayList arr = new ArrayList();
    BodoSQLColumn.BodoSQLColumnDataType dataType = BodoSQLColumn.BodoSQLColumnDataType.INT64;
    ColumnDataTypeInfo dataTypeInfo = new ColumnDataTypeInfo(dataType, true);
    BodoSQLColumnImpl column = new BodoSQLColumnImpl("A", dataTypeInfo);
    arr.add(column);
    BodoSQLColumnImpl column2 = new BodoSQLColumnImpl("B", dataTypeInfo);
    arr.add(column2);
    BodoSQLColumnImpl column3 = new BodoSQLColumnImpl("C", dataTypeInfo);
    arr.add(column3);

    BodoSqlTable table =
        new LocalTable(
            "LOCAL_TABLE", schema.getFullPath(), arr, false, "", "", false, "MEMORY", null, null);
    schema.addTable(table);
    // Second table to use for testing joins
    BodoSqlTable table2 =
        new LocalTable(
            "LOCAL_TABLE2", schema.getFullPath(), arr, false, "", "", false, "MEMORY", null, null);
    schema.addTable(table2);

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
            true, // Enable TIMESTAMP_TZ for testing
            true, // Enable Join Runtime filters for Testing
            "SNOWFLAKE" // Maintain case sensitivity in the Snowflake style by default
            );

    // Controls if we are generating code or executing DDL. You can set this
    // to false if you want to observe the actual execution of DDL statements.
    boolean generateCode = true;

    System.out.println("SQL query:");
    System.out.println(sql + "\n");

    if (generateCode) {
      String optimizedPlanStr = getRelationalAlgebraString(generator, sql);
      System.out.println("Optimized plan:");
      System.out.println(optimizedPlanStr + "\n");
      String pandasStr = generator.getPandasString(sql);
      System.out.println("Generated code:");
      System.out.println(pandasStr + "\n");
    } else {
      System.out.println("DDL OUTPUT:");
      System.out.println(generator.executeDDL(sql));
    }
  }

  private static String getRelationalAlgebraString(
      RelationalAlgebraGenerator generator, String sql) {
    try {
      Pair<RelRoot, Map<Integer, Integer>> root = generator.getRelationalAlgebra(sql);
      RelNode newRoot = BodoUtilKt.bodoPhysicalProject(root.getLeft());
      StringWriter sw = new StringWriter();
      RelCostAndMetaDataWriter costWriter =
          new RelCostAndMetaDataWriter(new PrintWriter(sw), newRoot, root.getRight(), true);
      newRoot.explain(costWriter);
      costWriter.explainCachedNodes();
      return sw.toString();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}
