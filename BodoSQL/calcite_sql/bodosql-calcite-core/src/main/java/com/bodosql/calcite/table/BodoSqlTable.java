/*
 * Copyright 2018 Bodo, Inc.
 */

package com.bodosql.calcite.table;

import com.bodosql.calcite.schema.BodoSqlSchema;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.*;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlNode;

public abstract class BodoSqlTable implements Table {
  /**
   * Abstract class definition for tables in BodoSQL. This contains all common shared components by
   * various types of tables, such as managing columns.
   *
   * <p>See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Table
   */
  private final String name;

  private final BodoSqlSchema schema;
  private final List<BodoSQLColumn> columns;

  /**
   * Common table constructor. This sets relevant information like the table name, columns, and a
   * backlink to the schema.
   *
   * @param name Name of the table.
   * @param schema Schema to which this table belongs.
   * @param columns List of columns in this table, in order.
   */
  protected BodoSqlTable(String name, BodoSqlSchema schema, List<BodoSQLColumn> columns) {
    this.name = name;
    this.schema = schema;
    this.columns = columns;
  }

  /** @return This table's name. */
  public String getName() {
    return this.name;
  }

  /** @return This table's schema. */
  public BodoSqlSchema getSchema() {
    return this.schema;
  }

  /**
   * Return a row type for the table that can be used in calcite.
   *
   * @param rdtf Data Factory used for building types.
   * @return RelDataType for a row of this table.
   */
  @Override
  public RelDataType getRowType(RelDataTypeFactory rdtf) {
    RelDataTypeFactory.Builder builder = rdtf.builder();
    for (BodoSQLColumn column : columns) {
      builder.add(column.getColumnName(), column.convertToSqlType(rdtf));
      builder.nullable(true);
    }
    return builder.build();
  }

  /**
   * Any statistic information for this table. This is not currently used by BodoSQL.
   *
   * @return Statistic information for a table.
   */
  @Override
  public Statistic getStatistic() {
    return Statistics.UNKNOWN;
  }

  /** @return A JDBC type for this table. This should be unused by BodoSQL. */
  @Override
  public Schema.TableType getJdbcTableType() {
    return Schema.TableType.TABLE;
  }

  /**
   * Is the current column rolledUp. This is not supported in BodoSQL so it always returns false.
   *
   * @param colName Column in question.
   * @return false
   */
  @Override
  public boolean isRolledUp(String colName) {
    return false;
  }

  /**
   * Can a rolledUp column be used inside an aggregation. This is not used in BodoSQL as no columns
   * are rolled up.
   *
   * @param colName Column in question.
   * @param sc SQL Aggregation call
   * @param sn SQLNode containing the call.
   * @param ccc Calcite connection configuration used to look up defined behaviors.
   * @return UnsupportedOperationException
   */
  @Override
  public boolean rolledUpColumnValidInsideAgg(
      String colName, SqlCall sc, SqlNode sn, CalciteConnectionConfig ccc) {
    throw new UnsupportedOperationException("rolledUpColumnValidInsideAgg Not supported yet.");
  }

  /**
   * Generates the code needed to cast a read table into types that can be supported by BodoSQL.
   * This using the cast information in the columns and generates appropriate Python to convert to
   * the desired output types.
   *
   * <p>If there are no casts that need to be performed this returns the empty string.
   *
   * @param varName Name of the variable containing the loaded.
   * @return Generated code used to cast the Table.
   */
  public String generateCastCode(String varName) {
    // Name of the columns to cast
    List<String> castColNames = new ArrayList<>();
    // List of string to use to perform the cast
    List<String> castStrings = new ArrayList<>();
    for (BodoSQLColumn col : this.columns) {
      if (col.requiresCast()) {
        castColNames.add(col.getColumnName());
        castStrings.add(col.getCastString(varName));
      }
    }
    if (castColNames.isEmpty()) {
      // No cast is need, return ""
      return "";
    }
    // Construct tuples to pass to __bodosql_replace_columns_dummy
    StringBuilder namesBuilder = new StringBuilder();
    StringBuilder typesBuilder = new StringBuilder();
    namesBuilder.append("(");
    typesBuilder.append("(");
    for (int i = 0; i < castColNames.size(); i++) {
      namesBuilder.append("'").append(castColNames.get(i)).append("'").append(", ");
      typesBuilder.append(castStrings.get(i)).append(", ");
    }
    namesBuilder.append(")");
    typesBuilder.append(")");
    return String.format(
        "bodo.hiframes.dataframe_impl.__bodosql_replace_columns_dummy(%s, %s, %s)",
        varName, namesBuilder, typesBuilder);
  }

  // BodoSQL Table abstract classes

  /** @return Can BodoSQL write to this table, updating it in a remote DB/location. */
  public abstract boolean isWriteable();

  /**
   * Generate the code needed to write the given variable to storage.
   *
   * @param varName Name of the variable to write.
   * @return The generated code to write the table.
   */
  public abstract String generateWriteCode(String varName);

  /**
   * Generate the code needed to read the table.
   *
   * @return The generated code to read the table.
   */
  public abstract String generateReadCode();
}
