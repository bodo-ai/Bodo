package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.adapter.pandas.PandasUtilKt;
import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.schema.LocalSchemaImpl;
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.LocalTableImpl;
import com.bodosql.calcite.traits.BatchingProperty;
import java.util.ArrayList;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelRoot;

/** Class for locally testing codegen. */
public class PandasGenTest {

  public static void main(String[] args) throws Exception {

    String sql = "select CURRENT_DATE()";
    int plannerChoice = RelationalAlgebraGenerator.VOLCANO_PLANNER;

    LocalSchemaImpl schema = new LocalSchemaImpl("__bodolocal__");
    ArrayList arr = new ArrayList();
    BodoSQLColumnDataType dataType = BodoSQLColumnDataType.INT64;
    BodoSQLColumnDataType paramType = BodoSQLColumnDataType.INT64;
    BodoSQLColumnImpl column = new BodoSQLColumnImpl("A", dataType, true);
    arr.add(column);
    BodoSQLColumnImpl column2 = new BodoSQLColumnImpl("D", dataType, true);
    arr.add(column2);
    BodoSQLColumnImpl column3 = new BodoSQLColumnImpl("C", dataType, true);
    arr.add(column3);

    BodoSqlTable table =
        new LocalTableImpl(
            "table1",
            schema,
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
    BodoSQLColumnImpl column4 = new BodoSQLColumnImpl("B", dataType, true);
    arr.add(column4);
    BodoSQLColumnImpl column5 = new BodoSQLColumnImpl("C", dataType, true);
    arr.add(column5);

    BodoSqlTable table2 =
        new LocalTableImpl(
            "table2",
            schema,
            arr,
            true,
            "table2",
            "TABLE_2 WRITE HERE (%s, %s)",
            false,
            "MEMORY",
            null);
    schema.addTable(table2);
    BodoSqlTable table3 =
        new LocalTableImpl(
            "table3",
            schema,
            arr,
            true,
            "table3",
            "TABLE_3 WRITE HERE (%s, %s)",
            false,
            "MEMORY",
            null);

    schema.addTable(table3);

    // Define the Parameter table
    String paramTableName = "ParamTable";
    arr = new ArrayList();
    arr.add(column);
    BodoSQLColumnImpl param1 = new BodoSQLColumnImpl("B", paramType, true);
    arr.add(param1);
    BodoSQLColumnImpl param2 = new BodoSQLColumnImpl("cwsfe_21", paramType, true);
    arr.add(param2);
    BodoSqlTable paramTable =
        new LocalTableImpl(
            paramTableName,
            schema,
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
            true // Always hide credentials
            );
    System.out.println("SQL query:");
    System.out.println(sql + "\n");
    String unOptimizedPlanStr = getRelationalAlgebraString(generator, sql, false);
    System.out.println("Unoptimized plan:");
    System.out.println(unOptimizedPlanStr + "\n");
    String optimizedPlanStr = getRelationalAlgebraString(generator, sql, true);
    System.out.println("Optimized plan:");
    System.out.println(optimizedPlanStr + "\n");

    String pandasStr = generator.getPandasString(sql);
    System.out.println("Generated code:");
    System.out.println(pandasStr + "\n");
    System.out.println("Lowered globals:");
    System.out.println(generator.getLoweredGlobalVariables() + "\n");
  }

  private static String getRelationalAlgebraString(
      RelationalAlgebraGenerator generator, String sql, boolean optimizePlan) {
    try {
      RelRoot root = generator.getRelationalAlgebra(sql, optimizePlan);
      return RelOptUtil.toString(PandasUtilKt.pandasProject(root));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}
