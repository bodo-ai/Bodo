package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.adapter.pandas.PandasUtilKt;
import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.schema.LocalSchema;
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.ColumnDataTypeInfo;
import com.bodosql.calcite.table.LocalTable;
import com.bodosql.calcite.traits.BatchingProperty;
import java.util.ArrayList;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelRoot;

/** Class for locally testing codegen. */
public class PandasGenTest {

  public static void main(String[] args) throws Exception {

    String sql = "select CURRENT_DATE()";
    int plannerChoice = RelationalAlgebraGenerator.STREAMING_PLANNER;

    LocalSchema schema = new LocalSchema("__BODOLOCAL__");
    ArrayList arr = new ArrayList();
    BodoSQLColumnDataType dataType = BodoSQLColumnDataType.INT64;
    BodoSQLColumnDataType paramType = BodoSQLColumnDataType.INT64;
    ColumnDataTypeInfo dataTypeInfo = new ColumnDataTypeInfo(dataType, true);
    ColumnDataTypeInfo paramTypeInfo = new ColumnDataTypeInfo(paramType, true);
    BodoSQLColumnImpl column = new BodoSQLColumnImpl("A", dataTypeInfo);
    arr.add(column);
    BodoSQLColumnImpl column2 = new BodoSQLColumnImpl("D", dataTypeInfo);
    arr.add(column2);
    BodoSQLColumnImpl column3 = new BodoSQLColumnImpl("C", dataTypeInfo);
    arr.add(column3);

    BodoSqlTable table =
        new LocalTable(
            "TABLE1",
            schema.getFullPath(),
            arr,
            true,
            "table1",
            "TABLE_1 WRITE HERE (%s, %s)",
            false,
            "MEMORY",
            null);

    schema.addTable(table);
    arr = new ArrayList();
    arr.add(column);
    BodoSQLColumnImpl column4 = new BodoSQLColumnImpl("B", dataTypeInfo);
    arr.add(column4);
    BodoSQLColumnImpl column5 = new BodoSQLColumnImpl("C", dataTypeInfo);
    arr.add(column5);

    BodoSqlTable table2 =
        new LocalTable(
            "TABLE2",
            schema.getFullPath(),
            arr,
            true,
            "table2",
            "TABLE_2 WRITE HERE (%s, %s)",
            false,
            "MEMORY",
            null);
    schema.addTable(table2);
    BodoSqlTable table3 =
        new LocalTable(
            "TABLE3",
            schema.getFullPath(),
            arr,
            true,
            "table3",
            "TABLE_3 WRITE HERE (%s, %s)",
            false,
            "MEMORY",
            null);

    schema.addTable(table3);

    // Define the Parameter table
    String paramTableName = "PARAMTABLE";
    arr = new ArrayList();
    arr.add(column);
    BodoSQLColumnImpl param1 = new BodoSQLColumnImpl("B", paramTypeInfo);
    arr.add(param1);
    BodoSQLColumnImpl param2 = new BodoSQLColumnImpl("cwsfe_21", paramTypeInfo);
    arr.add(param2);
    BodoSqlTable paramTable =
        new LocalTable(
            paramTableName,
            schema.getFullPath(),
            arr,
            true,
            paramTableName,
            "PARAM_TABLE WRITE HERE (%s, %s)",
            false,
            "MEMORY",
            null);

    schema.addTable(paramTable);

    RelationalAlgebraGenerator generator =
        new RelationalAlgebraGenerator(
            schema,
            paramTableName,
            plannerChoice,
            0,
            BatchingProperty.defaultBatchSize,
            true, // Always hide credentials
            true, // Enable Iceberg for testing
            true, // Enable TIMESTMAP_TZ for testing
            true // Enable Join Runtime filters for Testing
            );
    System.out.println("SQL query:");
    System.out.println(sql + "\n");
    String optimizedPlanStr = getRelationalAlgebraString(generator, sql);
    System.out.println("Optimized plan:");
    System.out.println(optimizedPlanStr + "\n");

    String pandasStr = generator.getPandasString(sql);
    System.out.println("Generated code:");
    System.out.println(pandasStr + "\n");
    System.out.println("Lowered globals:");
    System.out.println(generator.getLoweredGlobalVariables() + "\n");
  }

  private static String getRelationalAlgebraString(
      RelationalAlgebraGenerator generator, String sql) {
    try {
      RelRoot root = generator.getRelationalAlgebra(sql);
      return RelOptUtil.toString(PandasUtilKt.pandasProject(root));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}
