package com.bodosql.calcite.table;

import static com.bodosql.calcite.table.ColumnDataTypeInfo.fromSqlType;

import com.bodosql.calcite.adapter.pandas.StreamingOptions;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.sql.ddl.SnowflakeCreateTableMetadata;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Table;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * Definition of a table that is not associated with any schema. These tables include in memory
 * DataFrames and
 */
public class LocalTable extends BodoSqlTable {
  /**
   * A BodoSQL Table that is loaded locally (either using an in memory DataFrame or the table path
   * API). Since these tables are loaded individually, each table is responsible for its own code
   * regarding how to read and write the data.
   *
   * <p>See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Table
   */
  // Can we write to this table? This will only be True for certain
  // TablePath API Tables
  private final boolean isWriteable;

  // Code generated to read this table.
  private final String readCode;
  // Code generated to write this table. This should be a format
  // string containing exactly one %s, the location of the variable
  // name.

  private final String writeCodeFormatString;

  // Will the generated read code result in an IO operation inside Bodo.
  private final boolean useIORead;

  // String used to determine the source input type. This is used for
  // operations potentially supported by both TablePath and Catalogs
  // but only for certain DBs.
  private final String dbType;

  private final @Nullable Long estimatedRowCount;

  private final Statistic statistic = new StatisticImpl();

  /**
   * Constructor used for a LocalTable. In addition to the normal table components the table is
   * required to provide code need to read/write it. If a table is not writeable then the write code
   * is garbage and should be ignored.
   *
   * @param name Name of the table.
   * @param schemaPath A list of schemas names that must be traversed from the root to reach this
   *     table.
   * @param columns The list of columns for the table in order.
   * @param isWriteable Can we write to this table.
   * @param readCode The generated code to read this table.
   * @param writeCodeFormatString A format string to write a given variable. The format is described
   *     above the class member of the same name. If isWriteable=false this value can be garbage.
   * @param useIORead Does calling generateReadCode produce codegen that requires IO?
   * @param dbType What is the database source.
   * @param estimatedRowCount Estimated row count passed from Python.
   */
  public LocalTable(
      String name,
      ImmutableList<String> schemaPath,
      List<BodoSQLColumn> columns,
      boolean isWriteable,
      String readCode,
      String writeCodeFormatString,
      boolean useIORead,
      String dbType,
      @Nullable Long estimatedRowCount) {

    super(name, schemaPath, columns);
    this.isWriteable = isWriteable;
    this.readCode = readCode;
    this.writeCodeFormatString = writeCodeFormatString;
    this.useIORead = useIORead;
    this.dbType = dbType;
    this.estimatedRowCount = estimatedRowCount;
  }

  /**
   * Can we write to this table? Currently, only certain tables in TablePath API support writing.
   *
   * @return Is this table writeable?
   */
  @Override
  public boolean isWriteable() {
    return isWriteable;
  }

  /**
   * Generate the code needed to write the given variable to storage. Since each table is unique and
   * not tied to a common catalog, this table is initialized with the relevant write code.
   *
   * @param varName Name of the variable to write.
   * @return The generated code to write the table.
   */
  @Override
  public Expr generateWriteCode(PandasCodeGenVisitor visitor, Variable varName) {
    assert this.isWriteable
        : "Internal error: Local table not writeable in call to generateWriteCode";
    return new Expr.Raw(String.format(this.writeCodeFormatString, varName.emit(), ""));
  }

  /**
   * Generate the code needed to write the given variable to storage.
   *
   * @param varName Name of the variable to write.
   * @param extraArgs Extra arguments to pass to the Python API. They are assume to be escaped by
   *     the calling function and are of the form "key1=value1, ..., keyN=valueN".
   * @return The generated code to write the table.
   */
  public Expr generateWriteCode(PandasCodeGenVisitor visitor, Variable varName, String extraArgs) {
    assert this.isWriteable;
    return new Expr.Raw(String.format(this.writeCodeFormatString, varName.emit(), extraArgs));
  }

  /**
   * Generate the streaming code needed to initialize a writer for the given variable.
   *
   * @return The generated streaming code to write the table.
   */
  @Override
  public Expr generateStreamingWriteInitCode(Expr.IntegerLiteral operatorID) {
    throw new RuntimeException("Internal error: Streaming not supported for non-snowflake tables");
  }

  public Expr generateStreamingWriteAppendCode(
      PandasCodeGenVisitor visitor,
      Variable stateVarName,
      Variable dfVarName,
      Variable colNamesGlobal,
      Variable isLastVarName,
      Variable iterVarName,
      Expr columnPrecisions,
      SnowflakeCreateTableMetadata meta) {
    throw new RuntimeException("Internal error: Streaming not supported for non-snowflake tables");
  }

  /**
   * Return the location from which the table is generated. The return value is always entirely
   * capitalized.
   *
   * @return The source DB location.
   */
  @Override
  public String getDBType() {
    return dbType.toUpperCase();
  }

  /**
   * Generate the code needed to read the table. Since each table is unique and not tied to a common
   * catalog, this table is initialized with the relevant read code.
   *
   * @param useStreaming Should we generate code to read the table as streaming (currently only
   *     supported for snowflake tables)
   * @param streamingOptions The streaming options to use if streaming is enabled.
   * @return The generated code to read the table.
   */
  @Override
  public Expr generateReadCode(boolean useStreaming, StreamingOptions streamingOptions) {
    String extraArg = "";
    if (useStreaming) {
      extraArg = String.format("_bodo_chunksize=%d", streamingOptions.getChunkSize());
    }
    return new Expr.Raw(String.format(this.readCode, extraArg));
  }

  /**
   * Generate the code needed to read the table. This function is called by specialized IO
   * implementations that require passing 1 or more additional arguments.
   *
   * @param extraArgs: Extra arguments to pass to the Python API. They are assume to be escaped by
   *     the calling function and are of the form "key1=value1, ..., keyN=valueN".
   * @return The generated code to read the table.
   */
  @Override
  public Expr generateReadCode(String extraArgs) {
    return new Expr.Raw(String.format(this.readCode, extraArgs));
  }

  /**
   * Generates the code necessary to submit the remote query to the catalog DB. This is not
   * supported for local tables.
   *
   * @param query Query to submit.
   * @return The generated code.
   */
  @Override
  public Expr generateRemoteQuery(String query) {
    throw new UnsupportedOperationException(
        "A remote query cannot be submitted with a local table");
  }

  @Override
  public Table extend(List<RelDataTypeField> extensionFields) {
    String name = this.getName();
    List<BodoSQLColumn> extendedColumns = new ArrayList<>();
    extendedColumns.addAll(this.columns);
    for (int i = 0; i < extensionFields.size(); i++) {
      RelDataTypeField curField = extensionFields.get(0);
      String fieldName = curField.getName();
      RelDataType colType = curField.getType();
      ColumnDataTypeInfo newColType = fromSqlType(colType);
      BodoSQLColumn newCol = new BodoSQLColumnImpl(fieldName, newColType);
      extendedColumns.add(newCol);
    }
    return new LocalTable(
        name,
        getParentFullPath(),
        extendedColumns,
        isWriteable,
        readCode,
        writeCodeFormatString,
        useIORead,
        dbType,
        null);
  }

  /**
   * Returns if calling `generateReadCode()` for a table will result in an IO operation in the Bodo
   * generated code.
   *
   * @return Does the table require IO?
   */
  @Override
  public boolean readRequiresIO() {
    return useIORead;
  }

  @Override
  public Statistic getStatistic() {
    return statistic;
  }

  private class StatisticImpl implements Statistic {
    /**
     * Retrieves the estimated row count for this table based on the provided constructor
     * information. If the information provided is NULL we don't have an estimate and return NULL.
     *
     * @return estimated row count for this table.
     */
    @Override
    public @Nullable Double getRowCount() {
      if (estimatedRowCount == null) {
        return null;
      }
      return Double.valueOf(estimatedRowCount);
    }
  }
}
