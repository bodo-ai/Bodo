/*
 * Copyright 2018 Bodo, Inc.
 */

package com.bodosql.calcite.table;

import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.schema.BodoSqlSchema;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.schema.*;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlNode;

public abstract class BodoSqlTable implements ExtensibleTable {
  /**
   * Abstract class definition for tables in BodoSQL. This contains all common shared components by
   * various types of tables, such as managing columns.
   *
   * <p>See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Table
   */
  private final String name;

  private final BodoSqlSchema schema;
  protected final List<BodoSQLColumn> columns;

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

  public List<String> getColumnNames() {
    List<String> colNames = new ArrayList<>();
    for (BodoSQLColumn col : this.columns) {
      colNames.add(col.getColumnName());
    }
    return colNames;
  }

  /** @return A list of names to use for columns when writing this table. */
  public List<String> getWriteColumnNames() {
    List<String> colNames = new ArrayList<>();
    for (BodoSQLColumn col : this.columns) {
      colNames.add(col.getWriteColumnName());
    }
    return colNames;
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
      builder.add(
          column.getColumnName(),
          column.convertToSqlType(
              rdtf, column.isNullable(), column.getTZInfo(), column.getPrecision()));
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
   * Generates the code needed to cast a table currently in memory for BodoSQL into the output types
   * of the source table. This is done using the cast information in the columns and generates
   * appropriate Python to convert to the desired output types.
   *
   * <p>If there are no casts that need to be performed this returns the empty string.
   *
   * @param varName Name of the variable containing the data to write.
   * @return Generated code used to cast the Table being written.
   */

  /**
   * Generates the code needed to cast a read table into types that can be supported by BodoSQL.
   * This is done using the cast information in the columns and generates appropriate Python to
   * convert to the desired output types.
   *
   * <p>If there are no casts that need to be performed this returns the empty string.
   *
   * @param varName Name of the variable containing the loaded data.
   * @return Generated code used to cast the Table being read.
   */
  public Expr generateReadCastCode(Variable varName) {
    return generateCommonCastCode(varName, false);
  }

  /**
   * Generates the code needed to cast a BodoSQL table into types that are used when writing the
   * DataFrame back to its destination. This is done using the cast information in the columns and
   * generates appropriate Python to convert to the desired output types.
   *
   * <p>If there are no casts that need to be performed this returns the empty string.
   *
   * @param varName Name of the variable containing the data to write.
   * @return Generated code used to cast the Table being written.
   */
  public Expr generateWriteCastCode(Variable varName) {
    return generateCommonCastCode(varName, true);
  }

  /**
   * Generate common code shared by the cast operations for converting data being read in or about
   * to be written. This generates code using __bodosql_replace_columns_dummy to convert data while
   * maintaining table format if possible.
   *
   * <p>If there are no columns to cast this returns the input variable
   *
   * @param varName Name of the variable to cast.
   * @param isWrite Is the cast for a read or write. This determines the cast direction.
   * @return The generated Python code or the empty string.
   */
  private Expr generateCommonCastCode(Variable varName, boolean isWrite) {
    // Name of the columns to cast
    List<String> castColNames = new ArrayList<>();
    // List of string to use to perform the cast
    List<String> castExprs = new ArrayList<>();
    for (BodoSQLColumn col : this.columns) {
      if (!isWrite && col.requiresReadCast()) {
        castColNames.add(col.getColumnName());
        castExprs.add(col.getReadCastExpr(varName));
      }
    }
    if (castColNames.isEmpty()) {
      return varName;
    }
    // Construct tuples to pass to __bodosql_replace_columns_dummy
    StringBuilder namesBuilder = new StringBuilder();
    StringBuilder typesBuilder = new StringBuilder();
    namesBuilder.append("(");
    typesBuilder.append("(");
    for (int i = 0; i < castColNames.size(); i++) {
      namesBuilder.append("'").append(castColNames.get(i)).append("'").append(", ");
      typesBuilder.append(castExprs.get(i)).append(", ");
    }
    namesBuilder.append(")");
    typesBuilder.append(")");
    return new Expr.Raw(
        String.format(
            "bodo.hiframes.dataframe_impl.__bodosql_replace_columns_dummy(%s, %s, %s)",
            varName.emit(), namesBuilder, typesBuilder));
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
  public abstract Expr generateWriteCode(Variable varName);

  /**
   * Generate the code needed to write the given variable to storage.
   *
   * @param varName Name of the variable to write.
   * @param extraArgs Extra arguments to pass to the Python API. They are assume to be escaped by
   *     the calling function and are of the form "key1=value1, ..., keyN=valueN".
   * @return The generated code to write the table.
   */
  public abstract Expr generateWriteCode(Variable varName, String extraArgs);

  /**
   * Return the location from which the table is generated. The return value is always entirely
   * capitalized.
   *
   * @return The source DB location.
   */
  public abstract String getDBType();

  /**
   * Generate the code needed to read the table.
   *
   * @return The generated code to read the table.
   */
  public abstract Expr generateReadCode();

  /**
   * Generate the code needed to read the table. This function is called by specialized IO
   * implementations that require passing 1 or more additional arguments.
   *
   * @param extraArgs Extra arguments to pass to the Python API. They are assume to be escaped by
   *     the calling function and are of the form "key1=value1, ..., keyN=valueN".
   * @return The generated code to read the table.
   */
  public abstract Expr generateReadCode(String extraArgs);

  /**
   * Generates the code necessary to submit the remote query to the catalog DB. This is not
   * supported for local tables.
   *
   * @param query Query to submit.
   * @return The generated code.
   */
  public abstract Expr generateRemoteQuery(String query);

  public abstract Table extend(List<RelDataTypeField> extensionFields);

  /** This is never used by BodoSQL, so we always return -1 */
  @Override
  public int getExtendedColumnOffset() {
    return -1;
  }

  /**
   * Returns if calling `generateReadCode()` for a table will result in an IO operation in the Bodo
   * generated code.
   *
   * @return Does the table require IO?
   */
  public abstract boolean readRequiresIO();
}
