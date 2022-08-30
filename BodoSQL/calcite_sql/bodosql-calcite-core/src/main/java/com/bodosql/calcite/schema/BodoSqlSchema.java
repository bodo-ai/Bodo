/*
 * Copyright 2018 Bodo, Inc.
 */

package com.bodosql.calcite.schema;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import org.apache.calcite.linq4j.tree.Expression;
import org.apache.calcite.rel.type.RelProtoDataType;
import org.apache.calcite.schema.Function;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.schema.SchemaVersion;

public abstract class BodoSqlSchema implements Schema {
  /**
   * Generic abstract class for various possible BodoSQL schema types.
   *
   * <p>See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Schema
   */
  private final String name;

  /**
   * Constructor utilized by implementing constructors.
   *
   * @param name The schema's name.
   */
  protected BodoSqlSchema(String name) {
    this.name = name;
  }

  /** @return The schema's name. */
  public String getName() {
    return this.name;
  }

  /**
   * Returns all functions defined in this schema with a given name. This is likely used for a
   * stored procedure syntax but is not implemented for BodoSQL.
   *
   * @param funcName Name of functions with a given name.
   * @return Collection of all functions with that name.
   */
  @Override
  public Collection<Function> getFunctions(String funcName) {
    Collection<Function> functionCollection = new HashSet<>();
    return functionCollection;
  }

  /**
   * Returns the name of all functions defined in this schema. This is likely used for a stored
   * procedure syntax but is not implemented for BodoSQL.
   *
   * @return Set of all function names in this schema.
   */
  @Override
  public Set<String> getFunctionNames() {
    Set<String> functionSet = new HashSet<>();
    return functionSet;
  }

  /**
   * Returns a subschema with the given name. This will be replaced by implementations with multiple
   * levels of schema.
   *
   * @param schemaName Name of the subschema.
   * @return The subschema object.
   */
  @Override
  public Schema getSubSchema(String schemaName) {
    return null;
  }

  /**
   * Returns the names of all possible subschemas. This will be replaced by implementations with
   * multiple levels of schema.
   *
   * @return The Set of subschema names.
   */
  @Override
  public Set<String> getSubSchemaNames() {
    Set<String> hs = new HashSet<>();
    return hs;
  }

  /**
   * Returns all type names defined in this schema. This is not implemented for BodoSQL.
   *
   * @return Set of all type names.
   */
  @Override
  public Set<String> getTypeNames() {
    Set<String> hs = new HashSet<>();
    return hs;
  }

  /**
   * Returns the Calcite type registered in this schema with the given name. This is not implemented
   * for BodoSQL.
   *
   * @param name Name of the type to select.
   * @return RelProtoDataType object.
   */
  @Override
  public RelProtoDataType getType(String name) {
    return null;
  }

  /**
   * Generate API for expressions supported on this schema. This is not implemented for BodoSQL.
   *
   * @param sp A wrapper around a schema that can move to parent schemas and add/remove parents.
   * @param exprName The name of the expression
   * @return This operation is not supported.
   */
  @Override
  public Expression getExpression(SchemaPlus sp, String exprName) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  /** @return Can tables be added/removed from this schema. */
  @Override
  public boolean isMutable() {
    return true;
  }

  /**
   * Snapshotting information for a current state of the schema. This is not supported in BodoSQL.
   *
   * @param sv Version information for the given schema.
   * @return This operation is not supported.
   */
  @Override
  public Schema snapshot(SchemaVersion sv) {
    throw new UnsupportedOperationException("Not supported yet.");
  }
}
