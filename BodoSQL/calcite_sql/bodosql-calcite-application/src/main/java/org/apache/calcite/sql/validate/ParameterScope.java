/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.calcite.sql.validate;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlTableIdentifierWithID;

import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.Map;

/**
 * A scope which contains nothing besides a few parameters. Like
 * {@link EmptyScope} (which is its base class), it has no parent scope.
 *
 * @see ParameterNamespace
 */
public class ParameterScope extends EmptyScope {
  //~ Instance fields --------------------------------------------------------

  /**
   * Map from the simple names of the parameters to types of the parameters
   * ({@link RelDataType}).
   */
  private final Map<String, RelDataType> nameToTypeMap;

  //~ Constructors -----------------------------------------------------------

  public ParameterScope(
      SqlValidatorImpl validator,
      Map<String, RelDataType> nameToTypeMap) {
    super(validator);
    this.nameToTypeMap = nameToTypeMap;
  }

  //~ Methods ----------------------------------------------------------------

  @Override public SqlQualified fullyQualify(SqlIdentifier identifier) {
    return SqlQualified.create(this, 1, null, identifier);
  }


  /**
   * Converts a table identifier with an ID column into a fully-qualified identifier.
   * For example, the dept in "select empno from emp natural join dept" may become
   * "myschema.dept".
   *
   * @param identifier SqlTableIdentifierWithID to qualify.
   * @return A qualified identifier, never null
   */
  @Override public SqlTableIdentifierWithIDQualified fullyQualify(
      SqlTableIdentifierWithID identifier) {
    return SqlTableIdentifierWithIDQualified.create(
        this, 1, null, identifier);
  }

  @Override public SqlValidatorScope getOperandScope(SqlCall call) {
    return this;
  }

  @Override public @Nullable RelDataType resolveColumn(String name, SqlNode ctx) {
    return nameToTypeMap.get(name);
  }

  // Bodo Change: Extensions to the interface

  /** Returns the fullyQualify result of evaluating of the identifier
   * in this parameter scope. If the identifier
   * cannot be found within this scope then this returns null. */
  @Override
  @Nullable
  public SqlQualified fullyQualifyIdentifierIfParameter(SqlIdentifier identifier) {
    if (identifier.isSimple() && nameToTypeMap.containsKey(identifier.getSimple())) {
      return fullyQualify(identifier);
    } else {
      return null;
    }
  }

  /**
   * Resolves a single identifier to a column, and returns the datatype of
   * that column if it is a valid identifier in this parameter scope.
   * If the identifier cannot be found within this scope then this returns null.
   *
   * @param name Name of column
   * @param ctx  Context for exception
   * @return Type of column, if found and unambiguous; null if not found
   */
  public @Nullable RelDataType resolveColumnIfParameter(String name, SqlNode ctx) {
    if (nameToTypeMap.containsKey(name)) {
      return resolveColumn(name, ctx);
    } else {
      return null;
    }
  }
}
