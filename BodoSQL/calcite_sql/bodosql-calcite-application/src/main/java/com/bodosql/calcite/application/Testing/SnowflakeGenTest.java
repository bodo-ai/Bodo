package com.bodosql.calcite.application.testing;

import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.catalog.SnowflakeCatalog;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.LocalSchema;
import com.bodosql.calcite.table.BodoSQLColumn;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.ColumnDataTypeInfo;
import com.bodosql.calcite.table.LocalTable;
import java.util.ArrayList;
import java.util.Map;
import java.util.Properties;
import org.jetbrains.annotations.NotNull;

/** Class for locally testing codegen using a snowflake catalog */
public class SnowflakeGenTest extends GenTestFixture {
  @Override
  public boolean isIceberg() {
    return true;
  }

  @Override
  public boolean supportsTimestampTZ() {
    return true;
  }

  @NotNull
  @Override
  public BodoSQLCatalog getCatalog() {
    Map envVars = System.getenv();
    Properties prop = new Properties();
    return new SnowflakeCatalog(
        (String) envVars.get("SF_USERNAME"),
        (String) envVars.get("SF_PASSWORD"),
        (String) envVars.get("SF_ACCOUNT"),
        (String) envVars.get("SF_DATABASE"),
        "DEMO_WH",
        prop,
        null);
  }

  @NotNull
  @Override
  public BodoSqlSchema getSchema() {
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

    return schema;
  }

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

    boolean generateCode = true;
    new SnowflakeGenTest().run(sql, generateCode);
  }
}
