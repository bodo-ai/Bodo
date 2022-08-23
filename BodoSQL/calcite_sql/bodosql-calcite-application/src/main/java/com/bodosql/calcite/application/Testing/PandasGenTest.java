package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.catalog.domain.BodoSQLColumn.BodoSQLColumnDataType;
import com.bodosql.calcite.catalog.domain.BodoSQLColumnImpl;
import com.bodosql.calcite.catalog.domain.CatalogDatabaseImpl;
import com.bodosql.calcite.catalog.domain.CatalogTableImpl;
import com.bodosql.calcite.schema.BodoSqlSchema;
import java.util.ArrayList;

/** Class for locally testing codegen. */
public class PandasGenTest {

  public static void main(String[] args) throws Exception {

    String sql = "select a from __bodolocal__.table1 limit @cwsfe_21";

    CatalogDatabaseImpl db = new CatalogDatabaseImpl("__bodolocal__");
    ArrayList arr = new ArrayList();
    BodoSQLColumnDataType dataType = BodoSQLColumnDataType.DATETIME;
    BodoSQLColumnDataType paramType = BodoSQLColumnDataType.INT64;
    BodoSQLColumnImpl column = new BodoSQLColumnImpl("A", dataType);
    arr.add(column);
    BodoSQLColumnImpl column2 = new BodoSQLColumnImpl("D", dataType);
    arr.add(column2);
    BodoSQLColumnImpl column3 = new BodoSQLColumnImpl("C", dataType);
    arr.add(column3);

    CatalogTableImpl table = new CatalogTableImpl("table1", db, arr);
    db.addTable(table);

    arr = new ArrayList();
    arr.add(column);
    BodoSQLColumnImpl column4 = new BodoSQLColumnImpl("B", dataType);
    arr.add(column4);
    BodoSQLColumnImpl column5 = new BodoSQLColumnImpl("C", dataType);
    arr.add(column5);
    CatalogTableImpl table2 = new CatalogTableImpl("table2", db, arr);
    db.addTable(table2);
    CatalogTableImpl table3 = new CatalogTableImpl("table3", db, arr);
    db.addTable(table3);

    // Define the Parameter table
    String paramTableName = "ParamTable";
    arr = new ArrayList();
    arr.add(column);
    BodoSQLColumnImpl param1 = new BodoSQLColumnImpl("B", paramType);
    arr.add(param1);
    BodoSQLColumnImpl param2 = new BodoSQLColumnImpl("cwsfe_21", paramType);
    arr.add(param2);
    CatalogTableImpl paramTable = new CatalogTableImpl(paramTableName, db, arr);
    db.addTable(paramTable);

    BodoSqlSchema schema = new BodoSqlSchema(db);
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
