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

/**
 * The function syntax of the item operator, used to access a given element of an array, map or
 * struct. For example, {@code GET(myArray, 3)}, {@code GET(myMap, 'foo')}, etc.
 */
class BodoSqlItemOperatorGetFnSyntax extends SqlFunction {

  BodoSqlItemOperatorGetFnSyntax() {
    super(
        "GET",
        null,
        SqlKind.ITEM,
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
    return name
        + "(<ARRAY>|<MAP>|<VARIANT>, <INTEGER>|<STRING>)\n"
        + name
        + "(<ROW>, <CHARACTER>|<INTEGER>)";
  }

  @Override
  public boolean checkOperandTypes(SqlCallBinding callBinding, boolean throwOnFailure) {
    return BodoSqlItemOperatorUtils.itemOpCheckOperandTypes(callBinding, throwOnFailure);
  }

  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    return BodoSqlItemOperatorUtils.itemOpInferReturnType(opBinding);
  }
}
