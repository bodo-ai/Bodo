package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.catalog.domain.CatalogColumnDataType;
import com.bodosql.calcite.catalog.domain.CatalogColumnImpl;
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

    CatalogDatabaseImpl db = new CatalogDatabaseImpl("main");
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
    arr.add(
        new CatalogColumnImpl("c_custkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(new CatalogColumnImpl("c_name", CatalogColumnDataType.fromTypeId(variable_text), 0));
    arr.add(new CatalogColumnImpl("c_address", CatalogColumnDataType.fromTypeId(variable_text), 0));
    arr.add(
        new CatalogColumnImpl("c_nationkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(new CatalogColumnImpl("c_phone", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(new CatalogColumnImpl("c_acctbal", CatalogColumnDataType.fromTypeId(decimal_type), 0));
    arr.add(new CatalogColumnImpl("c_mktsegment", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(new CatalogColumnImpl("c_comment", CatalogColumnDataType.fromTypeId(variable_text), 0));
    CatalogTableImpl table = new CatalogTableImpl("customer", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(
        new CatalogColumnImpl("o_orderkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(
        new CatalogColumnImpl("o_custkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(
        new CatalogColumnImpl("o_orderstatus", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(
        new CatalogColumnImpl("o_totalprice", CatalogColumnDataType.fromTypeId(decimal_type), 0));
    arr.add(new CatalogColumnImpl("o_orderdate", CatalogColumnDataType.fromTypeId(date_type), 0));
    arr.add(
        new CatalogColumnImpl("o_orderpriority", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(new CatalogColumnImpl("o_clerk", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(
        new CatalogColumnImpl("o_shippriority", CatalogColumnDataType.fromTypeId(integer_type), 0));
    arr.add(new CatalogColumnImpl("o_comment", CatalogColumnDataType.fromTypeId(variable_text), 0));
    table = new CatalogTableImpl("orders", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(
        new CatalogColumnImpl("l_orderkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(
        new CatalogColumnImpl("l_partkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(
        new CatalogColumnImpl("l_suppkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(
        new CatalogColumnImpl("l_linenumber", CatalogColumnDataType.fromTypeId(integer_type), 0));
    arr.add(new CatalogColumnImpl("l_quantity", CatalogColumnDataType.fromTypeId(decimal_type), 0));
    arr.add(
        new CatalogColumnImpl(
            "l_extendedprice", CatalogColumnDataType.fromTypeId(decimal_type), 0));
    arr.add(new CatalogColumnImpl("l_discount", CatalogColumnDataType.fromTypeId(decimal_type), 0));
    arr.add(new CatalogColumnImpl("l_tax", CatalogColumnDataType.fromTypeId(decimal_type), 0));
    arr.add(new CatalogColumnImpl("l_returnflag", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(new CatalogColumnImpl("l_linestatus", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(new CatalogColumnImpl("l_shipdate", CatalogColumnDataType.fromTypeId(date_type), 0));
    arr.add(new CatalogColumnImpl("l_commitdate", CatalogColumnDataType.fromTypeId(date_type), 0));
    arr.add(new CatalogColumnImpl("l_receiptdate", CatalogColumnDataType.fromTypeId(date_type), 0));
    arr.add(
        new CatalogColumnImpl("l_shipinstruct", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(new CatalogColumnImpl("l_shipmode", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(new CatalogColumnImpl("l_comment", CatalogColumnDataType.fromTypeId(variable_text), 0));
    table = new CatalogTableImpl("lineitem", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(
        new CatalogColumnImpl("n_nationkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(new CatalogColumnImpl("n_name", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(
        new CatalogColumnImpl("n_regionkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(new CatalogColumnImpl("n_comment", CatalogColumnDataType.fromTypeId(variable_text), 0));
    table = new CatalogTableImpl("nation", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(
        new CatalogColumnImpl("r_regionkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(new CatalogColumnImpl("r_name", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(new CatalogColumnImpl("r_comment", CatalogColumnDataType.fromTypeId(variable_text), 0));
    table = new CatalogTableImpl("region", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(
        new CatalogColumnImpl("s_suppkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(new CatalogColumnImpl("s_name", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(new CatalogColumnImpl("s_address", CatalogColumnDataType.fromTypeId(variable_text), 0));
    arr.add(
        new CatalogColumnImpl("s_nationkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(new CatalogColumnImpl("s_phone", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(new CatalogColumnImpl("s_acctbal", CatalogColumnDataType.fromTypeId(decimal_type), 0));
    arr.add(new CatalogColumnImpl("s_comment", CatalogColumnDataType.fromTypeId(variable_text), 0));
    table = new CatalogTableImpl("supplier", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(
        new CatalogColumnImpl("ps_partkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(
        new CatalogColumnImpl("ps_suppkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(
        new CatalogColumnImpl("ps_availqty", CatalogColumnDataType.fromTypeId(integer_type), 0));
    arr.add(
        new CatalogColumnImpl("ps_supplycost", CatalogColumnDataType.fromTypeId(decimal_type), 0));
    arr.add(
        new CatalogColumnImpl("ps_comment", CatalogColumnDataType.fromTypeId(variable_text), 0));
    table = new CatalogTableImpl("partsupp", db, arr);
    db.addTable(table);
    arr = new ArrayList();
    arr.add(
        new CatalogColumnImpl("p_partkey", CatalogColumnDataType.fromTypeId(identifer_type), 0));
    arr.add(new CatalogColumnImpl("p_name", CatalogColumnDataType.fromTypeId(variable_text), 0));
    arr.add(new CatalogColumnImpl("p_mfgr", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(new CatalogColumnImpl("p_brand", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(new CatalogColumnImpl("p_type", CatalogColumnDataType.fromTypeId(variable_text), 0));
    arr.add(new CatalogColumnImpl("p_size", CatalogColumnDataType.fromTypeId(integer_type), 0));
    arr.add(new CatalogColumnImpl("p_container", CatalogColumnDataType.fromTypeId(fixed_text), 0));
    arr.add(
        new CatalogColumnImpl("p_retailprice", CatalogColumnDataType.fromTypeId(decimal_type), 0));
    arr.add(new CatalogColumnImpl("p_comment", CatalogColumnDataType.fromTypeId(variable_text), 0));
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
