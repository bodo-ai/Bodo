package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.schema.LocalSchemaImpl;
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.LocalTableImpl;
import java.util.ArrayList;

/** Class for locally testing codegen. */
public class PandasGenTest {

  public static void main(String[] args) throws Exception {

    String sql = "SELECT CAST('-8454757700450211157' AS INTEGER)";

    LocalSchemaImpl schema = new LocalSchemaImpl("__bodolocal__");
    ArrayList arr = new ArrayList();
    BodoSQLColumnDataType dataType = BodoSQLColumnDataType.DATETIME;
    BodoSQLColumnDataType paramType = BodoSQLColumnDataType.INT64;
    BodoSQLColumnImpl column = new BodoSQLColumnImpl("A", dataType);
    arr.add(column);
    BodoSQLColumnImpl column2 = new BodoSQLColumnImpl("D", dataType);
    arr.add(column2);
    BodoSQLColumnImpl column3 = new BodoSQLColumnImpl("C", dataType);
    arr.add(column3);

    BodoSqlTable table = new LocalTableImpl("table1", schema, arr, false, "table1", "");
    schema.addTable(table);

    arr = new ArrayList();
    arr.add(column);
    BodoSQLColumnImpl column4 = new BodoSQLColumnImpl("B", dataType);
    arr.add(column4);
    BodoSQLColumnImpl column5 = new BodoSQLColumnImpl("C", dataType);
    arr.add(column5);
    BodoSqlTable table2 = new LocalTableImpl("table2", schema, arr, false, "table2", "");
    schema.addTable(table2);
    BodoSqlTable table3 = new LocalTableImpl("table3", schema, arr, false, "table3", "");
    schema.addTable(table3);

    // Define the Parameter table
    String paramTableName = "ParamTable";
    arr = new ArrayList();
    arr.add(column);
    BodoSQLColumnImpl param1 = new BodoSQLColumnImpl("B", paramType);
    arr.add(param1);
    BodoSQLColumnImpl param2 = new BodoSQLColumnImpl("cwsfe_21", paramType);
    arr.add(param2);
    BodoSqlTable paramTable =
        new LocalTableImpl(paramTableName, schema, arr, false, paramTableName, "");
    schema.addTable(paramTable);

    RelationalAlgebraGenerator generator = new RelationalAlgebraGenerator(schema, paramTableName);
    System.out.println("SQL query:");
    System.out.println(sql + "\n");
    String unOptimizedPlanStr = generator.getRelationalAlgebraString(sql, false);
    System.out.println("Unoptimized plan:");
    System.out.println(unOptimizedPlanStr + "\n");
    String optimizedPlanStr = generator.getRelationalAlgebraString(sql, true);
    System.out.println("Optimized plan:");
    System.out.println(optimizedPlanStr + "\n");

    String pandasStr = generator.getPandasString(sql);
    System.out.println("Generated code:");
    System.out.println(pandasStr + "\n");
    System.out.println("Lowered globals:");
    System.out.println(generator.getLoweredGlobalVariables() + "\n");
  }
}
