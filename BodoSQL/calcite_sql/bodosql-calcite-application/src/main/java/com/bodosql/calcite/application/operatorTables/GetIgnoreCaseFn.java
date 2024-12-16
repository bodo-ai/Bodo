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
package com.bodosql.calcite.application.operatorTables;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperandCountRange;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.type.SqlOperandCountRanges;
import org.apache.calcite.sql.type.SqlTypeName;

/**
 * The GET_IGNORE_CASE function. Used to access a map or struct without regard to the case if the
 * key.
 */
class GetIgnoreCaseFn extends SqlFunction {

  GetIgnoreCaseFn() {
    super(
        "GET_IGNORE_CASE",
        null,
        SqlKind.OTHER_FUNCTION,
        null,
        null,
        null,
        null,
        SqlFunctionCategory.USER_DEFINED_SPECIFIC_FUNCTION);
  }

  @Override
  public SqlOperandCountRange getOperandCountRange() {
    return SqlOperandCountRanges.of(2);
  }

  @Override
  public String getAllowedSignatures(String name) {
    return name + "(<MAP>, <VARCHAR>|<VARIANT>)\n" + name + "(<VARIANT>, <ANY>)\n";
  }

  @Override
  public boolean checkOperandTypes(SqlCallBinding callBinding, boolean throwOnFailure) {
    // GET_IGNORE_CASE has the same operand type checking as the generic GET function,
    // but it doesn't allow Rows or arrays for Arg0

    SqlTypeName arg0TypeName = callBinding.getOperandType(0).getSqlTypeName();

    if (arg0TypeName.equals(SqlTypeName.ROW) || arg0TypeName.equals(SqlTypeName.ARRAY)) {
      if (throwOnFailure) {
        throw callBinding.newValidationSignatureError();
      } else {
        return false;
      }
    }

    return BodoSqlItemOperatorUtils.itemOpCheckOperandTypes(callBinding, throwOnFailure);
  }

  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    return BodoSqlItemOperatorUtils.itemOpInferReturnType(opBinding);
  }
}
