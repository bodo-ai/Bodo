package com.bodosql.calcite.table;

import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.LocalSchemaImpl;
import java.util.*;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.type.*;

/**
 * Definition of a table that is not associated with any schema. These tables include in memory
 * DataFrames and
 */
public class LocalTableImpl extends BodoSqlTable {
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

  /**
   * Constructor used for a LocalTableImpl. In addition to the normal table components the table is
   * required to provide code need to read/write it. If a table is not writeable then the write code
   * is garbage and should be ignored.
   *
   * @param name Name of the table.
   * @param schema The table's schema. This must be a LocalSchemaImpl.
   * @param columns The list of columns for the table in order.
   * @param isWriteable Can we write to this table.
   * @param readCode The generated code to read this table.
   * @param writeCodeFormatString A format string to write a given variable. The format is described
   *     above the class member of the same name. If isWriteable=false this value can be garbage.
   * @param useIORead Does calling generateReadCode produce codegen that requires IO?
   */
  public LocalTableImpl(
      String name,
      BodoSqlSchema schema,
      List<BodoSQLColumn> columns,
      boolean isWriteable,
      String readCode,
      String writeCodeFormatString,
      boolean useIORead,
      String dbType) {

    super(name, schema, columns);
    if (!(schema instanceof LocalSchemaImpl)) {
      throw new RuntimeException("Local table must be implemented with a Local Schema.");
    }
    this.isWriteable = isWriteable;
    this.readCode = readCode;
    this.writeCodeFormatString = writeCodeFormatString;
    this.useIORead = useIORead;
    this.dbType = dbType;
  }

  /**
   * Can we write to this table? Currently only certain tables in TablePath API support writing.
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
  public String generateWriteCode(String varName) {
    assert this.isWriteable;
    return String.format(this.writeCodeFormatString, varName, "");
  }

  /**
   * Generate the code needed to write the given variable to storage.
   *
   * @param varName Name of the variable to write.
   * @param extraArgs Extra arguments to pass to the Python API. They are assume to be escaped by
   *     the calling function and are of the form "key1=value1, ..., keyN=valueN".
   * @return The generated code to write the table.
   */
  public String generateWriteCode(String varName, String extraArgs) {
    assert this.isWriteable;
    return String.format(this.writeCodeFormatString, varName, extraArgs);
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
   * @return The generated code to read the table.
   */
  @Override
  public String generateReadCode(boolean useDateRuntime) {
    return String.format(this.readCode, "");
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
  public String generateReadCode(String extraArgs, boolean useDateRuntime) {
    return String.format(this.readCode, extraArgs);
  }

  /**
   * Generates the code necessary to submit the remote query to the catalog DB. This is not
   * supported for local tables.
   *
   * @param query Query to submit.
   * @return The generated code.
   */
  @Override
  public String generateRemoteQuery(String query) {
    throw new UnsupportedOperationException(
        "A remote query cannot be submitted with a local table");
  }

  @Override
  public Table extend(List<RelDataTypeField> extensionFields) {
    String name = this.getName();
    BodoSqlSchema schema = this.getSchema();
    List<BodoSQLColumn> extendedColumns = new ArrayList<>();
    extendedColumns.addAll(this.columns);
    for (int i = 0; i < extensionFields.size(); i++) {
      RelDataTypeField curField = extensionFields.get(0);
      String fieldName = curField.getName();
      RelDataType colType = curField.getType();
      BodoSQLColumn.BodoSQLColumnDataType newColType =
          BodoSQLColumn.BodoSQLColumnDataType.fromSqlType(colType);
      // getTZInfo() returns null if the type is not TZAware Timestamp
      BodoTZInfo tzInfo = colType.getTZInfo();
      BodoSQLColumn newCol = new BodoSQLColumnImpl(fieldName, newColType, false, tzInfo);
      extendedColumns.add(newCol);
    }
    return new LocalTableImpl(
        name,
        schema,
        extendedColumns,
        isWriteable,
        readCode,
        writeCodeFormatString,
        useIORead,
        dbType);
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
}
