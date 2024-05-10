package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.adapter.bodo.BodoUtilKt;
import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.schema.LocalSchema;
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.ColumnDataTypeInfo;
import com.bodosql.calcite.table.LocalTable;
import com.bodosql.calcite.traits.BatchingProperty;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelRoot;
import org.apache.commons.lang3.tuple.Pair;

/** Class for locally testing codegen. */
public class PandasGenTest {

  public static void main(String[] args) throws Exception {

    String sql =
        "select CASE WHEN A > 10 THEN @c + 1 WHEN A > 100 THEN @c + 2 WHEN A > 1000 THEN @c + 3"
            + " ELSE @c + 4 END from table1";
    int plannerChoice = RelationalAlgebraGenerator.STREAMING_PLANNER;

    LocalSchema schema = new LocalSchema("__BODOLOCAL__");
    ArrayList arr = new ArrayList();
    BodoSQLColumnDataType dataType = BodoSQLColumnDataType.INT64;
    ColumnDataTypeInfo dataTypeInfo = new ColumnDataTypeInfo(dataType, true);
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
            null,
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
            null,
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
            null,
            null);

    schema.addTable(table3);

    RelationalAlgebraGenerator generator =
        new RelationalAlgebraGenerator(
            schema,
            plannerChoice,
            0,
            1,
            BatchingProperty.defaultBatchSize,
            true, // Always hide credentials
            true, // Enable Iceberg for testing
            true, // Enable TIMESTAMP_TZ for testing
            true, // Enable Join Runtime filters for Testing
            "SNOWFLAKE" // Maintain case sensitivity in the Snowflake style by default
            );
    List<ColumnDataTypeInfo> paramTypes =
        List.of(new ColumnDataTypeInfo(BodoSQLColumnDataType.INT64, false));
    Map<String, ColumnDataTypeInfo> namedParamTypes =
        Map.of(
            "a",
            new ColumnDataTypeInfo(BodoSQLColumnDataType.INT64, false),
            "c",
            new ColumnDataTypeInfo(BodoSQLColumnDataType.INT64, false));
    System.out.println("SQL query:");
    System.out.println(sql + "\n");
    String optimizedPlanStr =
        getRelationalAlgebraString(generator, sql, paramTypes, namedParamTypes);
    System.out.println("Optimized plan:");
    System.out.println(optimizedPlanStr + "\n");

    String pandasStr = generator.getPandasString(sql, paramTypes, namedParamTypes);
    System.out.println("Generated code:");
    System.out.println(pandasStr + "\n");
    System.out.println("Lowered globals:");
    System.out.println(generator.getLoweredGlobalVariables() + "\n");
  }

  private static String getRelationalAlgebraString(
      RelationalAlgebraGenerator generator,
      String sql,
      List<ColumnDataTypeInfo> paramTypes,
      Map<String, ColumnDataTypeInfo> namedParamTypes) {
    try {
      Pair<RelRoot, Map<Integer, Integer>> root =
          generator.getRelationalAlgebra(sql, paramTypes, namedParamTypes);
      return RelOptUtil.toString(BodoUtilKt.bodoPhysicalProject(root.getLeft()));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}
