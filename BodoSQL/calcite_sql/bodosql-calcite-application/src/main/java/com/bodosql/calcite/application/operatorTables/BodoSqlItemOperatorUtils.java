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
import static org.apache.calcite.sql.type.NonNullableAccessors.getComponentTypeOrThrow;
import static org.apache.calcite.sql.validate.SqlNonNullableAccessors.getOperandLiteralValueOrThrow;

import com.bodosql.calcite.rel.type.BodoRelDataTypeFactory;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.BodoOperandTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.SqlSingleOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.type.VariantSqlType;
import org.apache.calcite.sql.validate.implicit.BodoTypeCoercionImpl;

/**
 * A class containing functions common to both of the Item Operator functions.
 * (BodoSqlItemOperatorBracketsSyntax, and BodoSqlItemOperatorGetFnSyntax).
 *
 * <p>Several of the functions within this class are based on functions present in
 * org.apache.calcite.sql.fun.SqlItemOperator. However, the changes that were needed to make this
 * function work with variant types could not be done within Calcite, due to needing behavior to
 * check for VariantSqlType, which only exists within Bodo.
 */
public class BodoSqlItemOperatorUtils {

  private static final SqlSingleOperandTypeChecker ARRAY_OR_MAP_OR_VARIANT =
      OperandTypes.family(SqlTypeFamily.ARRAY)
          .or(OperandTypes.family(SqlTypeFamily.MAP))
          .or(BodoOperandTypes.VARIANT);

  public static boolean itemOpCheckOperandTypes(
      SqlCallBinding callBinding, boolean throwOnFailure) {
    final SqlNode left = callBinding.operand(0);
    final SqlNode right = callBinding.operand(1);
    if (!ARRAY_OR_MAP_OR_VARIANT.checkSingleOperandType(callBinding, left, 0, throwOnFailure)) {
      return false;
    }

    final SqlSingleOperandTypeChecker checker = getChecker(callBinding);
    if (checker != null) {
      if (!checker.checkSingleOperandType(callBinding, right, 0, throwOnFailure)) {
        return false;
      }
    }

    // Handle casting of the index if necessary. All numeric values should be cast to int and
    // rounded away from 0.
    // All other types should be cast to string.
    RelDataTypeFactory typeFactory = callBinding.getTypeFactory();
    RelDataType arg1CastType;
    if (SqlTypeFamily.NUMERIC.contains(callBinding.getOperandType(1))) {
      // Need to inject a call to make sure that value is properly rounded up
      List<SqlNode> args = new ArrayList();
      args.add(callBinding.operand(1));
      SqlCall roundedArg1 =
          SqlStdOperatorTable.ROUND.createCall(callBinding.getCall().getParserPosition(), args);
      callBinding.getCall().setOperand(1, roundedArg1);
      arg1CastType =
          typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.BIGINT),
              callBinding.getOperandType(1).isNullable());
    } else {
      // Cast to string
      arg1CastType =
          typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.VARCHAR),
              callBinding.getOperandType(1).isNullable());
    }
    callBinding.getTypeFactory().getTypeSystem();

    ((BodoTypeCoercionImpl) callBinding.getValidator().getTypeCoercion())
        .coerceOperandType(callBinding.getScope(), callBinding.getCall(), 1, arg1CastType);
    return true;
  }

  static SqlSingleOperandTypeChecker getChecker(SqlCallBinding callBinding) {
    final RelDataType operandType = callBinding.getOperandType(0);
    // BODO CHANGE: added a check for variant type here
    if (operandType instanceof VariantSqlType) {
      return null;
    }
    switch (operandType.getSqlTypeName()) {
        // BODO CHANGE: SF allows for indexing with pretty much anything,
        // so we allow any index value
      case ARRAY:
      case MAP:
      case ANY:
      case DYNAMIC_STAR:
        return null;
        // I'm not screwing with Row indexing though, because I'm not entirely certain where this is
        // used,
        // and I don't want to make unnecessary changes that end up breaking things
      case ROW:
        return OperandTypes.family(SqlTypeFamily.INTEGER)
            .or(OperandTypes.family(SqlTypeFamily.CHARACTER));
      default:
        throw callBinding.newValidationSignatureError();
    }
  }

  public static RelDataType itemOpInferReturnType(SqlOperatorBinding opBinding) {
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    final RelDataType operandType = opBinding.getOperandType(0);
    // BODO CHANGE: added check for variant type
    if (operandType instanceof VariantSqlType) {
      assert typeFactory instanceof BodoRelDataTypeFactory;
      // Indexing into variant may return null values
      return typeFactory.createTypeWithNullability(
          ((BodoRelDataTypeFactory) typeFactory).createVariantSqlType(), true);
    }
    switch (operandType.getSqlTypeName()) {
      case ARRAY:
        // This can be null if the index is out of bounds.
        return typeFactory.createTypeWithNullability(getComponentTypeOrThrow(operandType), true);
      case MAP:
        // This can be null if the key doesn't exist in the map.
        return typeFactory.createTypeWithNullability(
            requireNonNull(
                operandType.getValueType(),
                () -> "operandType.getValueType() is null for " + operandType),
            true);
      case ROW:
        RelDataType fieldType;
        RelDataType indexType = opBinding.getOperandType(1);

        if (SqlTypeUtil.isString(indexType)) {
          final String fieldName = getOperandLiteralValueOrThrow(opBinding, 1, String.class);
          RelDataTypeField field = operandType.getField(fieldName, false, false);
          if (field == null) {
            throw new AssertionError(
                "Cannot infer type of field '" + fieldName + "' within ROW type: " + operandType);
          } else {
            fieldType = field.getType();
          }
        } else if (SqlTypeUtil.isIntType(indexType)) {
          Integer index = opBinding.getOperandLiteralValue(1, Integer.class);
          if (index == null || index < 1 || index > operandType.getFieldCount()) {
            throw new AssertionError(
                "Cannot infer type of field at position "
                    + index
                    + " within ROW type: "
                    + operandType);
          } else {
            fieldType = operandType.getFieldList().get(index - 1).getType(); // 1 indexed
          }
        } else {
          throw new AssertionError("Unsupported field identifier type: '" + indexType + "'");
        }
        if (fieldType != null && operandType.isNullable()) {
          fieldType = typeFactory.createTypeWithNullability(fieldType, true);
        }
        return fieldType;
      case ANY:
      case DYNAMIC_STAR:
        return typeFactory.createTypeWithNullability(
            typeFactory.createSqlType(SqlTypeName.ANY), true);
      default:
        throw new AssertionError();
    }
  }
}
