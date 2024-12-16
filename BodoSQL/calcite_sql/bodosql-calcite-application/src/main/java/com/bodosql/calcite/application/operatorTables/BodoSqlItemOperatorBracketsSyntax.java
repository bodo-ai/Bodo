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

import static java.util.Objects.requireNonNull;

import java.util.Arrays;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperandCountRange;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlOperandCountRanges;

/**
 * The item operator {@code [ ... ]}, used to access a given element of an array, map or struct. For
 * example, {@code myArray[3]}, {@code "myMap['foo']"}, {@code myStruct[2]} or {@code
 * myStruct['fieldName']}.
 *
 * <p>This Operator is heavily based on org.apache.calcite.sql.fun.SqlItemOperator. However, the
 * changes that were needed to make this function work with variant types could not be done within
 * Calcite, due to needing behavior to check for VariantSqlType, which only exists within Bodo.
 */
class BodoSqlItemOperatorBracketsSyntax extends SqlSpecialOperator {

  BodoSqlItemOperatorBracketsSyntax() {
    super("ITEM", SqlKind.ITEM, 100, true, null, null, null);
  }

  @Override
  public SqlSpecialOperator.ReduceResult reduceExpr(
      int ordinal, SqlSpecialOperator.TokenSequence list) {
    SqlNode left = list.node(ordinal - 1);
    SqlNode right = list.node(ordinal + 1);
    return new SqlSpecialOperator.ReduceResult(
        ordinal - 1,
        ordinal + 2,
        createCall(
            SqlParserPos.sum(
                Arrays.asList(
                    requireNonNull(left, "left").getParserPosition(),
                    requireNonNull(right, "right").getParserPosition(),
                    list.pos(ordinal))),
            left,
            right));
  }

  @Override
  public SqlOperandCountRange getOperandCountRange() {
    return SqlOperandCountRanges.of(2);
  }

  @Override
  public void unparse(SqlWriter writer, SqlCall call, int leftPrec, int rightPrec) {
    call.operand(0).unparse(writer, leftPrec, 0);
    final SqlWriter.Frame frame = writer.startList("[", "]");
    call.operand(1).unparse(writer, 0, 0);
    writer.endList(frame);
  }

  @Override
  public String getAllowedSignatures(String name) {
    return "<ARRAY>|<MAP>|<VARIANT>[<INTEGER>|<STRING>]\n" + "<ROW>[<CHARACTER>|<INTEGER>]";
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
