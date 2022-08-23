package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.catalog.domain.BodoSQLColumn.BodoSQLColumnDataType;
import com.bodosql.calcite.catalog.domain.BodoSQLColumnImpl;
import com.bodosql.calcite.catalog.domain.CatalogDatabaseImpl;
import com.bodosql.calcite.catalog.domain.CatalogTableImpl;
import com.bodosql.calcite.schema.BodoSqlSchema;
import java.util.ArrayList;

public class TPCHSchemaTest {
  public static void main(String[] args) throws Exception {
    String sql =
        "select\n"
            + "                       s_name,\n"
            + "                       s_address\n"
            + "                     from\n"
            + "                       supplier, nation\n"
            + "                     where\n"
            + "                       s_suppkey in (\n"
            + "                         select\n"
            + "                           ps_suppkey\n"
            + "                         from\n"
            + "                           partsupp\n"
            + "                         where\n"
            + "                           ps_partkey in (\n"
            + "                             select\n"
            + "                               p_partkey\n"
            + "                             from\n"
            + "                               part\n"
            + "                             where\n"
            + "                               p_name like 'forest%'\n"
            + "                           )\n"
            + "                         and ps_availqty > (\n"
            + "                           select\n"
            + "                             0.5 * sum(l_quantity)\n"
            + "                           from\n"
            + "                             lineitem\n"
            + "                           where\n"
            + "                             l_partkey = ps_partkey\n"
            + "                             and l_suppkey = ps_suppkey\n"
            + "                             and l_shipdate >= date '1994-01-01'\n"
            + "                             and l_shipdate < date '1994-01-01' + interval '1'"
            + " year\n"
            + "                         )\n"
            + "                       )\n"
            + "                       and s_nationkey = n_nationkey\n"
            + "                       and n_name = 'CANADA'\n"
            + "                     order by\n"
            + "                       s_name";

    System.out.println(sql);

    CatalogDatabaseImpl db = new CatalogDatabaseImpl("__bodolocal__");
    /*
     * Schema derived from: http://www.tpc.org/tpc_documents_current_versions/pdf
     */

    /*
     * We will use uint64 for identifiers.
     */
    int identifer_type = 8;
    int integer_type = 3;
    int date_type = 12;
    /*
     * TPCH uses a decimal type, but we will check with float64 unless this is an issue
     */
    int decimal_type = 10;
    /*
     * We will use string type for all of the text types
     */
    int fixed_text = 16;
    int variable_text = 16;

    ArrayList arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("c_custkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("c_name", BodoSQLColumnDataType.fromTypeId(variable_text)));
    arr.add(new BodoSQLColumnImpl("c_address", BodoSQLColumnDataType.fromTypeId(variable_text)));
    arr.add(new BodoSQLColumnImpl("c_nationkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("c_phone", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("c_acctbal", BodoSQLColumnDataType.fromTypeId(decimal_type)));
    arr.add(new BodoSQLColumnImpl("c_mktsegment", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("c_comment", BodoSQLColumnDataType.fromTypeId(variable_text)));
    CatalogTableImpl table = new CatalogTableImpl("customer", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("o_orderkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("o_custkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("o_orderstatus", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("o_totalprice", BodoSQLColumnDataType.fromTypeId(decimal_type)));
    arr.add(new BodoSQLColumnImpl("o_orderdate", BodoSQLColumnDataType.fromTypeId(date_type)));
    arr.add(new BodoSQLColumnImpl("o_orderpriority", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("o_clerk", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(
        new BodoSQLColumnImpl("o_shippriority", BodoSQLColumnDataType.fromTypeId(integer_type)));
    arr.add(new BodoSQLColumnImpl("o_comment", BodoSQLColumnDataType.fromTypeId(variable_text)));
    table = new CatalogTableImpl("orders", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("l_orderkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("l_partkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("l_suppkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("l_linenumber", BodoSQLColumnDataType.fromTypeId(integer_type)));
    arr.add(new BodoSQLColumnImpl("l_quantity", BodoSQLColumnDataType.fromTypeId(decimal_type)));
    arr.add(
        new BodoSQLColumnImpl("l_extendedprice", BodoSQLColumnDataType.fromTypeId(decimal_type)));
    arr.add(new BodoSQLColumnImpl("l_discount", BodoSQLColumnDataType.fromTypeId(decimal_type)));
    arr.add(new BodoSQLColumnImpl("l_tax", BodoSQLColumnDataType.fromTypeId(decimal_type)));
    arr.add(new BodoSQLColumnImpl("l_returnflag", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("l_linestatus", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("l_shipdate", BodoSQLColumnDataType.fromTypeId(date_type)));
    arr.add(new BodoSQLColumnImpl("l_commitdate", BodoSQLColumnDataType.fromTypeId(date_type)));
    arr.add(new BodoSQLColumnImpl("l_receiptdate", BodoSQLColumnDataType.fromTypeId(date_type)));
    arr.add(new BodoSQLColumnImpl("l_shipinstruct", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("l_shipmode", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("l_comment", BodoSQLColumnDataType.fromTypeId(variable_text)));
    table = new CatalogTableImpl("lineitem", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("n_nationkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("n_name", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("n_regionkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("n_comment", BodoSQLColumnDataType.fromTypeId(variable_text)));
    table = new CatalogTableImpl("nation", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("r_regionkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("r_name", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("r_comment", BodoSQLColumnDataType.fromTypeId(variable_text)));
    table = new CatalogTableImpl("region", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("s_suppkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("s_name", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("s_address", BodoSQLColumnDataType.fromTypeId(variable_text)));
    arr.add(new BodoSQLColumnImpl("s_nationkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("s_phone", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("s_acctbal", BodoSQLColumnDataType.fromTypeId(decimal_type)));
    arr.add(new BodoSQLColumnImpl("s_comment", BodoSQLColumnDataType.fromTypeId(variable_text)));
    table = new CatalogTableImpl("supplier", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("ps_partkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("ps_suppkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("ps_availqty", BodoSQLColumnDataType.fromTypeId(integer_type)));
    arr.add(new BodoSQLColumnImpl("ps_supplycost", BodoSQLColumnDataType.fromTypeId(decimal_type)));
    arr.add(new BodoSQLColumnImpl("ps_comment", BodoSQLColumnDataType.fromTypeId(variable_text)));
    table = new CatalogTableImpl("partsupp", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("p_partkey", BodoSQLColumnDataType.fromTypeId(identifer_type)));
    arr.add(new BodoSQLColumnImpl("p_name", BodoSQLColumnDataType.fromTypeId(variable_text)));
    arr.add(new BodoSQLColumnImpl("p_mfgr", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("p_brand", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("p_type", BodoSQLColumnDataType.fromTypeId(variable_text)));
    arr.add(new BodoSQLColumnImpl("p_size", BodoSQLColumnDataType.fromTypeId(integer_type)));
    arr.add(new BodoSQLColumnImpl("p_container", BodoSQLColumnDataType.fromTypeId(fixed_text)));
    arr.add(new BodoSQLColumnImpl("p_retailprice", BodoSQLColumnDataType.fromTypeId(decimal_type)));
    arr.add(new BodoSQLColumnImpl("p_comment", BodoSQLColumnDataType.fromTypeId(variable_text)));
    table = new CatalogTableImpl("part", db, arr);
    db.addTable(table);

    BodoSqlSchema schema = new BodoSqlSchema(db);
    RelationalAlgebraGenerator generator = new RelationalAlgebraGenerator(schema, "");
    System.out.println("SQL query:");
    System.out.println(sql + "\n");
    String planStr = generator.getRelationalAlgebraString(sql, true);
    System.out.println("Optimized plan:");
    System.out.println(planStr + "\n");
    String pandasStr = generator.getPandasString(sql);
    System.out.println("Generated code:");
    System.out.println(pandasStr + "\n");
  }
}
