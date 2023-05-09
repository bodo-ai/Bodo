package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.application.Utils.TypeEquivalentSimplifier;
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
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.pretty.SqlPrettyWriter;
import org.apache.calcite.sql.util.SqlString;
import org.apache.calcite.util.Pair;

public class GenSimplificationUtil {
  public static void main(String[] args) throws Exception {

    // Takes an input SQL query, and reduces it to a simple sql query that returns the same types
    String sql = " select * from LINEITEM1 r limit 10";
    Map envVars = System.getenv();
    Properties prop = new Properties();
    prop.put("schema", "PUBLIC");
    prop.put("queryTimeout", 5);
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
        new LocalTableImpl("localtable", schema, arr, false, "", "", false, "MEMORY");
    schema.addTable(table);

    RelationalAlgebraGenerator generator =
        new RelationalAlgebraGenerator(catalog, schema, "dummy_param_table_name", 0);

    Pair<SqlNode, RelDataType> out = generator.validateQueryAndGetType(sql);
    SqlNode reducedOut = TypeEquivalentSimplifier.reduceToSimpleSqlNode(out.left, out.right);
    // TODO: may eventually need an SF specific version of this writer
    SqlWriter writer = new SqlPrettyWriter();

    reducedOut.unparse(writer, 1, 1);
    SqlString outSqlString = writer.toSqlString();
    String outString = outSqlString.getSql();
    System.out.println(outString);
  }
}
