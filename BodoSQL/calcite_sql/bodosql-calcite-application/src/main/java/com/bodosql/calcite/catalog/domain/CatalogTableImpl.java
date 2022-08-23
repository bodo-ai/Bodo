package com.bodosql.calcite.catalog.domain;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 *
 *
 * <h1>Stores a table with its corresponding columns</h1>
 *
 * @author bodo
 */
public class CatalogTableImpl implements CatalogTable {
  /**
   * This constructor is used to fill in all the values for the table.
   *
   * @param name the name of the table that is being created
   * @param db the database this table is being added to
   * @param columns list of columns to be added to the table.
   */
  public CatalogTableImpl(String name, CatalogDatabaseImpl db, List<BodoSQLColumnImpl> columns) {
    this.name = name;
    this.database = db;
    this.columnNameMap = new HashMap<>();
    this.columnList = new ArrayList<>();
    for (BodoSQLColumnImpl column : columns) {
      this.columnNameMap.put(column.getColumnName(), column);
      columnList.add(column);
    }
  }

  /**
   * This constructor is used to fill in all the values for the table including values used for
   * writing the output.
   *
   * @param name the name of the table that is being created
   * @param db the database this table is being added toBodoSqlCatalogTest
   * @param columns list of columns to be added to the table.
   * @param writeName Original table name to use when writing.
   * @param schema (Nullable) Schema name to use when writing.
   * @param connStr connection string to use when writing.
   */
  public CatalogTableImpl(
      String name,
      CatalogDatabaseImpl db,
      List<BodoSQLColumnImpl> columns,
      String writeName,
      String schema,
      String connStr) {
    this.name = name;
    this.database = db;
    this.columnNameMap = new HashMap<>();
    this.columnList = new ArrayList<>();
    for (BodoSQLColumnImpl column : columns) {
      this.columnNameMap.put(column.getColumnName(), column);
      columnList.add(column);
    }
    this.writeName = writeName;
    this.schema = schema;
    this.connStr = connStr;
  }

  /** The name of the table. */
  private String name;

  /** A map of name to {@see BodoSQLColumnImpl} */
  private Map<String, BodoSQLColumnImpl> columnNameMap;

  /** A list of column names in order. Used for integer access */
  private ArrayList<BodoSQLColumnImpl> columnList;

  /** The {@see CatalogDatabaseImpl} this table belongs to. */
  private CatalogDatabaseImpl database;

  // TODO: Fix the general design of tables/catalogs
  // inside of BodoSQL.
  private String writeName = null;

  private String schema = null;

  private String connStr = null;

  @Override
  public CatalogDatabaseImpl getDatabase() {
    return database;
  }

  @Override
  public String getTableName() {
    return this.name;
  }

  /**
   * Converts the columns list into a Set of columns
   *
   * @return the set of {@see CatalogColumn} that belong to this table
   */
  @Override
  public Set<BodoSQLColumn> getColumns() {
    Set<BodoSQLColumn> tempColumns = new LinkedHashSet<BodoSQLColumn>();

    for (BodoSQLColumnImpl col : columnList) {
      tempColumns.add(col);
    }

    return tempColumns;
  }

  public @Nullable String getWriteName() {
    return this.writeName;
  }

  public @Nullable String getSchema() {
    return this.schema;
  }

  public @Nullable String getConnStr() {
    return this.connStr;
  }
}
