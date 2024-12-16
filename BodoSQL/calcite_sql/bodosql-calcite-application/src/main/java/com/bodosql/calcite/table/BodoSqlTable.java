/*
 * Copyright 2018 Bodo, Inc.
 */

package com.bodosql.calcite.table;

import com.bodosql.calcite.adapter.bodo.StreamingOptions;
import com.bodosql.calcite.application.BodoCodeGenVisitor;
import com.bodosql.calcite.application.write.WriteTarget;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.schema.ExtensibleTable;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.schema.Table;
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

  // Full path of schemas to reach this, including the
  // table name.
  private final ImmutableList<String> fullPath;

  protected final List<BodoSQLColumn> columns;

  /**
   * Common table constructor. This sets relevant information like the table name, columns, and a
   * backlink to the schema.
   *
   * @param name Name of the table.
   * @param schemaPath A list of schemas names that must be traversed from the root to reach this
   *     table.
   * @param columns List of columns in this table, in order.
   */
  protected BodoSqlTable(
      String name, ImmutableList<String> schemaPath, List<BodoSQLColumn> columns) {
    this.name = name;
    this.columns = columns;
    ImmutableList.Builder<String> builder = new ImmutableList.Builder<>();
    builder.addAll(schemaPath);
    builder.add(name);
    fullPath = builder.build();
  }

  /**
   * @return This table's name.
   */
  public String getName() {
    return this.name;
  }

  /**
   * @return The full path for this table. *
   */
  public ImmutableList<String> getFullPath() {
    return fullPath;
  }

  /**
   * @return The full path for this table's parent. This is the full path without the table name. *
   */
  public ImmutableList<String> getParentFullPath() {
    return fullPath.subList(0, fullPath.size() - 1);
  }

  public List<String> getColumnNames() {
    List<String> colNames = new ArrayList<>();
    for (BodoSQLColumn col : this.columns) {
      colNames.add(col.getColumnName());
    }
    return colNames;
  }

  /**
   * @return A list of names to use for columns when writing this table.
   */
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
      builder.add(column.getColumnName(), column.getDataTypeInfo().convertToSqlType(rdtf));
    }
    return builder.build();
  }

  /**
   * Determine the estimated approximate number of distinct values for the column. This is
   * implemented by each individual table which is responsible for caching the results.
   *
   * @return Estimated distinct count for this table.
   */
  public Double getColumnDistinctCount(int column) {
    return null;
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

  /**
   * @return A JDBC type for this table. This should be unused by BodoSQL.
   */
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
   * This is done using the cast information in the columns and generates appropriate Python to
   * convert to the desired output types.
   *
   * <p>If there are no casts that need to be performed this returns the empty string.
   *
   * @param varName Name of the variable containing the loaded data.
   * @return Generated code used to cast the Table being read.
   */
  public Expr generateReadCastCode(Variable varName) {
    // Name of the columns to cast
    List<String> castColNames = new ArrayList<>();
    // List of string to use to perform the cast
    List<String> castExprs = new ArrayList<>();
    for (BodoSQLColumn col : this.columns) {
      if (col.requiresReadCast()) {
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

  /**
   * @return Can BodoSQL write to this table, updating it in a remote DB/location.
   */
  public abstract boolean isWriteable();

  /**
   * Do we have read access for this table. This should be overridden by tables with a catalog
   * source.
   *
   * @return Do we have read access?
   */
  public boolean canRead() {
    return true;
  }

  /**
   * Generate the code needed to write the given variable to storage.
   *
   * @param varName Name of the variable to write.
   * @return The generated code to write the table.
   */
  public abstract Expr generateWriteCode(BodoCodeGenVisitor visitor, Variable varName);

  /**
   * Generate the code needed to write the given variable to storage.
   *
   * @param varName Name of the variable to write.
   * @param extraArgs Extra arguments to pass to the Python API. They are assume to be escaped by
   *     the calling function and are of the form "key1=value1, ..., keyN=valueN".
   * @return The generated code to write the table.
   */
  public abstract Expr generateWriteCode(
      BodoCodeGenVisitor visitor, Variable varName, String extraArgs);

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
   * @param useStreaming Should we generate code to read the table as streaming (currently only
   *     supported for snowflake tables)
   * @param streamingOptions The options to be used if streaming is selected.
   * @return The generated code to read the table.
   */
  public abstract Expr generateReadCode(boolean useStreaming, StreamingOptions streamingOptions);

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

  /**
   * Get the insert into write target for a particular table. Most tables must maintain the same
   * table type as already exists for the table, so this will generally be a property of the table.
   *
   * @param columnNamesGlobal The global variable containing the column names. This should be
   *     possible to remove in the future since we append to a table.
   * @return The WriteTarget for the table.
   */
  public abstract WriteTarget getInsertIntoWriteTarget(Variable columnNamesGlobal);
}
