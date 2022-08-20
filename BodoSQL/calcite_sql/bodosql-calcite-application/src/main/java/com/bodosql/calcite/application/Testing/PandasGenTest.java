package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.catalog.domain.CatalogColumnDataType;
import com.bodosql.calcite.catalog.domain.CatalogColumnImpl;
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
    CatalogColumnDataType dataType = CatalogColumnDataType.DATETIME;
    CatalogColumnDataType paramType = CatalogColumnDataType.INT64;
    CatalogColumnImpl column = new CatalogColumnImpl("A", dataType, 0);
    arr.add(column);
    CatalogColumnImpl column2 = new CatalogColumnImpl("D", dataType, 0);
    arr.add(column2);
    CatalogColumnImpl column3 = new CatalogColumnImpl("C", dataType, 0);
    arr.add(column3);

    CatalogTableImpl table = new CatalogTableImpl("table1", db, arr);
    db.addTable(table);

    arr = new ArrayList();
    arr.add(column);
    CatalogColumnImpl column4 = new CatalogColumnImpl("B", dataType, 0);
    arr.add(column4);
    CatalogColumnImpl column5 = new CatalogColumnImpl("C", dataType, 0);
    arr.add(column5);
    CatalogTableImpl table2 = new CatalogTableImpl("table2", db, arr);
    db.addTable(table2);
    CatalogTableImpl table3 = new CatalogTableImpl("table3", db, arr);
    db.addTable(table3);

    // Define the Parameter table
    String paramTableName = "ParamTable";
    arr = new ArrayList();
    arr.add(column);
    CatalogColumnImpl param1 = new CatalogColumnImpl("B", paramType, 0);
    arr.add(param1);
    CatalogColumnImpl param2 = new CatalogColumnImpl("cwsfe_21", paramType, 0);
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
