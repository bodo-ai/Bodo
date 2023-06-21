package com.bodosql.calcite.application.Testing;

import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.schema.LocalSchemaImpl;
import com.bodosql.calcite.table.*;
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType;
import com.bodosql.calcite.traits.BatchingProperty;
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

    LocalSchemaImpl schema = new LocalSchemaImpl("__bodolocal__");
    /*
     * Schema derived from: http://www.tpc.org/tpc_documents_current_versions/pdf
     */

    /*
     * We will use uint64 for identifiers.
     */
    BodoSQLColumnDataType identifer_type = BodoSQLColumnDataType.UINT64;
    BodoSQLColumnDataType integer_type = BodoSQLColumnDataType.INT32;
    BodoSQLColumnDataType date_type = BodoSQLColumnDataType.DATE;
    /*
     * TPCH uses a decimal type, but we will check with float64 unless this is an issue
     */
    BodoSQLColumnDataType decimal_type = BodoSQLColumnDataType.FLOAT64;
    /*
     * We will use string type for all text types
     */
    BodoSQLColumnDataType fixed_text = BodoSQLColumnDataType.STRING;
    BodoSQLColumnDataType variable_text = BodoSQLColumnDataType.STRING;

    ArrayList arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("c_custkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("c_name", variable_text, true));
    arr.add(new BodoSQLColumnImpl("c_address", variable_text, true));
    arr.add(new BodoSQLColumnImpl("c_nationkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("c_phone", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("c_acctbal", decimal_type, true));
    arr.add(new BodoSQLColumnImpl("c_mktsegment", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("c_comment", variable_text, true));
    BodoSqlTable table =
        new LocalTableImpl("customer", schema, arr, false, "customer", "", false, "MEMORY", null);
    schema.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("o_orderkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("o_custkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("o_orderstatus", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("o_totalprice", decimal_type, true));
    arr.add(new BodoSQLColumnImpl("o_orderdate", date_type, true));
    arr.add(new BodoSQLColumnImpl("o_orderpriority", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("o_clerk", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("o_shippriority", integer_type, true));
    arr.add(new BodoSQLColumnImpl("o_comment", variable_text, true));
    table = new LocalTableImpl("orders", schema, arr, false, "orders", "", false, "MEMORY", null);
    schema.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("l_orderkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("l_partkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("l_suppkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("l_linenumber", integer_type, true));
    arr.add(new BodoSQLColumnImpl("l_quantity", decimal_type, true));
    arr.add(new BodoSQLColumnImpl("l_extendedprice", decimal_type, true));
    arr.add(new BodoSQLColumnImpl("l_discount", decimal_type, true));
    arr.add(new BodoSQLColumnImpl("l_tax", decimal_type, true));
    arr.add(new BodoSQLColumnImpl("l_returnflag", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("l_linestatus", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("l_shipdate", date_type, true));
    arr.add(new BodoSQLColumnImpl("l_commitdate", date_type, true));
    arr.add(new BodoSQLColumnImpl("l_receiptdate", date_type, true));
    arr.add(new BodoSQLColumnImpl("l_shipinstruct", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("l_shipmode", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("l_comment", variable_text, true));
    table =
        new LocalTableImpl("lineitem", schema, arr, false, "lineitem", "", false, "MEMORY", null);
    schema.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("n_nationkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("n_name", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("n_regionkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("n_comment", variable_text, true));
    table = new LocalTableImpl("nation", schema, arr, false, "nation", "", false, "MEMORY", null);
    schema.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("r_regionkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("r_name", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("r_comment", variable_text, true));
    table = new LocalTableImpl("region", schema, arr, false, "region", "", false, "MEMORY", null);
    schema.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("s_suppkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("s_name", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("s_address", variable_text, true));
    arr.add(new BodoSQLColumnImpl("s_nationkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("s_phone", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("s_acctbal", decimal_type, true));
    arr.add(new BodoSQLColumnImpl("s_comment", variable_text, true));
    table =
        new LocalTableImpl("supplier", schema, arr, false, "supplier", "", false, "MEMORY", null);
    schema.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("ps_partkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("ps_suppkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("ps_availqty", integer_type, true));
    arr.add(new BodoSQLColumnImpl("ps_supplycost", decimal_type, true));
    arr.add(new BodoSQLColumnImpl("ps_comment", variable_text, true));
    table =
        new LocalTableImpl("partsupp", schema, arr, false, "partsupp", "", false, "MEMORY", null);
    schema.addTable(table);
    arr = new ArrayList();
    arr.add(new BodoSQLColumnImpl("p_partkey", identifer_type, true));
    arr.add(new BodoSQLColumnImpl("p_name", variable_text, true));
    arr.add(new BodoSQLColumnImpl("p_mfgr", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("p_brand", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("p_type", variable_text, true));
    arr.add(new BodoSQLColumnImpl("p_size", integer_type, true));
    arr.add(new BodoSQLColumnImpl("p_container", fixed_text, true));
    arr.add(new BodoSQLColumnImpl("p_retailprice", decimal_type, true));
    arr.add(new BodoSQLColumnImpl("p_comment", variable_text, true));
    table = new LocalTableImpl("part", schema, arr, false, "part", "", false, "MEMORY", null);
    schema.addTable(table);

    RelationalAlgebraGenerator generator =
        new RelationalAlgebraGenerator(schema, "", 0, 0, BatchingProperty.defaultBatchSize);
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
