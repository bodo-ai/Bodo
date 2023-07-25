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
package com.bodosql.calcite.sql.validate;

import com.bodosql.calcite.sql.ddl.SqlSnowflakeUpdate;

import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.prepare.CalciteCatalogReader;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorImpl;
import org.apache.calcite.sql.validate.SqlValidatorScope;

import static org.apache.calcite.util.Static.RESOURCE;

/** Duplication of the CalciteSqlValidator class from Calcite. */
public class BodoSqlValidator extends SqlValidatorImpl {
  public BodoSqlValidator(
      SqlOperatorTable opTab,
      CalciteCatalogReader catalogReader,
      JavaTypeFactory typeFactory,
      SqlValidator.Config config) {
    super(opTab, catalogReader, typeFactory, config);
  }

  @Override
  protected RelDataType getLogicalSourceRowType(RelDataType sourceRowType, SqlInsert insert) {
    final RelDataType superType = super.getLogicalSourceRowType(sourceRowType, insert);
    return ((JavaTypeFactory) typeFactory).toSql(superType);
  }

  @Override
  protected RelDataType getLogicalTargetRowType(RelDataType targetRowType, SqlInsert insert) {
    final RelDataType superType = super.getLogicalTargetRowType(targetRowType, insert);
    return ((JavaTypeFactory) typeFactory).toSql(superType);
  }

  /**
   * Creates the SELECT statement that putatively feeds rows into an UPDATE statement to be updated.
   *
   * @param call Call to the UPDATE operator
   * @return select statement
   */
  protected SqlSelect createSourceSelectForUpdate(SqlUpdate call) {
    SqlSelect select = super.createSourceSelectForUpdate(call);
    // Cast to the BodoSQL subclass of Update to extract the FROM field.
    // If it exists, this table is cross joined with the source table.
    if (call instanceof SqlSnowflakeUpdate) {
      SqlNode from = ((SqlSnowflakeUpdate) call).getFrom();
      if (from != null) {
        select.setFrom(
            new SqlJoin(
                SqlParserPos.ZERO,
                select.getFrom(),
                SqlLiteral.createBoolean(false, SqlParserPos.ZERO),
                JoinType.CROSS.symbol(SqlParserPos.ZERO),
                from,
                JoinConditionType.NONE.symbol(SqlParserPos.ZERO),
                null));
      }
    }
    return select;
  }

  @Override
  protected void validateSelect(SqlSelect select, RelDataType targetRowType) {
    // Validate the select.
    super.validateSelect(select, targetRowType);

    // Handle fetch and offset directly. The super call only handles literals and dynamic parameters.
    // It does not handle named parameters.
    handleFetchOffset(select);
  }

  private void handleFetchOffset(SqlSelect select) {
    SqlValidatorScope scope = getEmptyScope();

    SqlNode fetchNode = select.getFetch();
    if (fetchNode instanceof SqlCall) {
      SqlCall fetch = (SqlCall) fetchNode;
      RelDataType type = deriveType(scope, fetch);
      if (!SqlTypeUtil.isIntType(type)) {
        throw newValidationError(fetch, RESOURCE.typeNotSupported(type.getFullTypeString()));
      }
    }

    SqlNode offsetNode = select.getOffset();
    if (offsetNode instanceof SqlCall) {
      SqlCall offset = (SqlCall) offsetNode;
      RelDataType type = deriveType(scope, offset);
      if (!SqlTypeUtil.isIntType(type)) {
        throw newValidationError(offset, RESOURCE.typeNotSupported(type.getFullTypeString()));
      }
    }
  }
}
