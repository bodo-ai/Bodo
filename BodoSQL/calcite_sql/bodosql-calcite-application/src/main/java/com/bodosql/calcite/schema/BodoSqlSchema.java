/*
 * Copyright 2018 Bodo, Inc.
 */

package com.bodosql.calcite.schema;

import com.google.common.collect.ImmutableList;
import java.util.Collection;
import java.util.List;
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
   * The depth level for the given schema. We define the depth as the number of parents that must be
   * visited before reaching the root. The root schema should have a depth of 0 and be the only
   * schema with depth 0.
   */
  private final int depth;

  // Full path of schemas to reach this, including the
  // table name.
  private final ImmutableList<String> fullPath;

  /**
   * Constructor utilized by implementing constructors.
   *
   * @param name The schema's name.
   */
  protected BodoSqlSchema(String name, int depth, ImmutableList<String> schemaPath) {
    this.name = name;
    this.depth = depth;
    ImmutableList.Builder<String> builder = new ImmutableList.Builder<>();
    builder.addAll(schemaPath);
    builder.add(name);
    fullPath = builder.build();
  }

  /** @return The schema's name. */
  public String getName() {
    return this.name;
  }

  /**
   * Get the full path of schemas traversed from the root to reach this schema.
   *
   * @return An immutable list wrapping the name.
   */
  public ImmutableList<String> getFullPath() {
    return fullPath;
  }

  /**
   * Create the full path of for a table by concatenating a table name with the full path.
   *
   * @param tableName The name of the table to append.
   * @return A table path combining getFullPath with the tableName.
   */
  public ImmutableList<String> createTablePath(String tableName) {
    ImmutableList.Builder<String> builder = new ImmutableList.Builder();
    builder.addAll(getFullPath());
    builder.add(tableName);
    return builder.build();
  }

  /**
   * Returns all functions defined in this schema with a given name.
   *
   * @param funcName Name of functions with a given name.
   * @return Collection of all functions with that name.
   */
  @Override
  public Collection<Function> getFunctions(String funcName) {
    return List.of();
  }

  /**
   * Returns the name of all functions defined in this schema. This is likely used for a stored
   * procedure syntax but is not implemented for BodoSQL.
   *
   * @return Set of all function names in this schema.
   */
  @Override
  public Set<String> getFunctionNames() {
    return Set.of();
  }

  /**
   * Returns a subSchema with the given name. This will be replaced by implementations with multiple
   * levels of schema.
   *
   * @param schemaName Name of the subSchema.
   * @return The subSchema object.
   */
  @Override
  public Schema getSubSchema(String schemaName) {
    return null;
  }

  /**
   * Returns the names of all possible subSchemas. This will be replaced by implementations with
   * multiple levels of schema.
   *
   * @return The Set of subSchema names.
   */
  @Override
  public Set<String> getSubSchemaNames() {
    return Set.of();
  }

  /**
   * Returns all type names defined in this schema. This is not implemented for BodoSQL.
   *
   * @return Set of all type names.
   */
  @Override
  public Set<String> getTypeNames() {
    return Set.of();
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

  /**
   * Return the depth level for the given schema. We will define the depth as the length of the path
   * to the root including the root. As a result, the root has schemaDepth 0 and a schema just below
   * the root would have depth 1.
   *
   * @return The depth of the schema.
   */
  int getSchemaDepth() {
    return depth;
  }
}
