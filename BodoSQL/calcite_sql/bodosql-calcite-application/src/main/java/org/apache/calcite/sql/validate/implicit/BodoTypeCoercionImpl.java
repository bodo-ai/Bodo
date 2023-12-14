package org.apache.calcite.sql.validate.implicit;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLTypeSystems.CoalesceTypeCastingUtils;
import kotlin.Pair;
import kotlin.jvm.functions.Function3;
import kotlin.jvm.functions.Function4;
import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeFactoryImpl;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlDynamicParam;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.BodoSqlTypeUtil;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.type.VariantSqlType;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorScope;

import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.bodosql.calcite.application.operatorTables.CastingOperatorTable.TO_TIMESTAMP_LTZ;
import static com.bodosql.calcite.application.operatorTables.CastingOperatorTable.TO_TIMESTAMP_TZ;
import static com.bodosql.calcite.application.operatorTables.NumericOperatorTable.TO_NUMBER;
import static java.util.Objects.requireNonNull;
import static org.apache.calcite.sql.validate.SqlNonNullableAccessors.getScope;

public class BodoTypeCoercionImpl extends TypeCoercionImpl {
  public BodoTypeCoercionImpl(RelDataTypeFactory typeFactory, SqlValidator validator) {
    super(typeFactory, validator);
  }

  public static TypeCoercionFactory FACTORY = new TypeCoercionFactoryImpl();

  @Override
  public @Nullable RelDataType getTightestCommonType(
      @Nullable RelDataType type1, @Nullable RelDataType type2) {
    if (SqlTypeUtil.isNumeric(type1) && SqlTypeUtil.isBoolean(type2)) {
      return type2;
    }
    if (SqlTypeUtil.isNumeric(type2) && SqlTypeUtil.isBoolean(type1)) {
      return type1;
    }
    return super.getTightestCommonType(type1, type2);
  }

  /**
   * Cast a variant type for a given operator name and argument number to its intended
   * cast. If no cast is provided or the argument does not need to be cast it returns
   * the original type.
   *
   * @param variantType The original type to cast.
   * @param operatorName The name of the function. This is presumed to be a unique identifier. If an
   *                     overloaded function is encountered additional information may be required.
   * @param argNum Which argument is being cast. This is important for functions where arguments have different
   *               defined types.
   * @param operandTypes The original list of types to the function call.
   * @return A new type or the original type.
   */
  protected RelDataType variantImplicitCast(RelDataType variantType, String operatorName, int argNum, List<RelDataType> operandTypes) {
    // TODO(njriasan): Define a simpler name for this type
    Map<String, Function4<RelDataType, RelDataTypeFactory, Integer, List<? extends RelDataType>, RelDataType>> variantMap = VariantCastTable.Companion.getVariantNameMapping();
    if (variantMap.containsKey(operatorName)) {
      Function4<RelDataType, RelDataTypeFactory, Integer, List<? extends RelDataType>, RelDataType> castFunction = variantMap.get(operatorName);
      return castFunction.invoke(variantType, factory, argNum, operandTypes);
    } else {
      return variantType;
    }
  }

  /**
   * Bodo extension of builtinFunctionCoercion to control variant behavior.
   * Within Snowflake Variants do not seem to follow simple rules like
   * Variant -> Boolean if a boolean is accepted. Instead, the behavior
   * seems to be function dependent.
   *
   * If we wanted to make global changes to implicitCast, then we wouldn't
   * have the level of control necessary to specify the action for each function.
   * As a result, instead we cast variants on a per-function basis before reaching
   * the general casting behavior. After we handle variants we return to the parent's
   * behavior.
   *
   * @param binding          Call binding
   * @param operandTypes     Types of the operands passed in
   * @param expectedFamilies Expected SqlTypeFamily list by user specified
   * @return Was any input coerced to another type?
   */
  @Override public boolean builtinFunctionCoercion(
          SqlCallBinding binding,
          List<RelDataType> operandTypes,
          List<SqlTypeFamily> expectedFamilies) {
    boolean coerced = false;
    String operatorName = binding.getOperator().getName();
    for (int i = 0; i < operandTypes.size(); i++) {
      RelDataType operandType = operandTypes.get(i);
      if (operandType instanceof VariantSqlType) {
        RelDataType implicitType = variantImplicitCast(operandType, operatorName, i, operandTypes);
        coerced = null != implicitType
                && operandTypes.get(i) != implicitType
                && coerceOperandType(binding.getScope(), binding.getCall(), i, implicitType)
                || coerced;
      }
    }
    // Perform any other casting
    boolean otherCoerced = super.builtinFunctionCoercion(binding, operandTypes, expectedFamilies);
    return otherCoerced || coerced;
  }

  @Override
  public @Nullable RelDataType implicitCast(RelDataType in, SqlTypeFamily expected) {
    // Calcite natively enables casting STRING -> BINARY AND BINARY -> STRING.
    // We don't want this behavior, so we disable it.
    if ((SqlTypeUtil.isBinary(in) && expected == SqlTypeFamily.CHARACTER) || (SqlTypeUtil.isCharacter(in) && expected == SqlTypeFamily.BINARY)) {
      return null;
    }
    if ((SqlTypeUtil.isNumeric(in) || SqlTypeUtil.isCharacter(in))
        && expected == SqlTypeFamily.BOOLEAN) {
      return factory.createSqlType(SqlTypeName.BOOLEAN);
    }

    if ((SqlTypeUtil.isCharacter(in)) && (expected == SqlTypeFamily.DATETIME)) {
      return factory.createSqlType(SqlTypeName.TIMESTAMP);
    }
    return super.implicitCast(in, expected);
  }

  @Override
  public @Nullable RelDataType commonTypeForBinaryComparison(
      @Nullable RelDataType type1, @Nullable RelDataType type2) {
    if (type1 == null || type2 == null) {
      return null;
    }
    SqlTypeName typeName1 = type1.getSqlTypeName();
    SqlTypeName typeName2 = type2.getSqlTypeName();
    if (typeName1 == null || typeName2 == null) {
      return null;
    }
    // Any variant type should be the common type.
    if (type1 instanceof VariantSqlType) {
      return type1;
    }
    if (type2 instanceof VariantSqlType) {
      return type2;
    }

    // BOOLEAN + NUMERIC -> BOOLEAN
    if (SqlTypeUtil.isBoolean(type1) && SqlTypeUtil.isNumeric(type2)) {
      return type1;
    }
    if (SqlTypeUtil.isNumeric(type1) && SqlTypeUtil.isBoolean(type2)) {
      return type2;
    }
    return super.commonTypeForBinaryComparison(type1, type2);
  }

  @Override
  protected boolean coerceOperandType(
      @Nullable SqlValidatorScope scope,
      SqlCall call,
      int index,
      RelDataType targetType) {
    // Mostly duplicated from calcite. We need to customize this operation
    // because calcite's support for custom datatypes isn't very well supported.
    // It's easier to just copy the code for this method and intercept the
    // coercion logic for tz aware timestamps than it is to try and figure out how
    // to get calcite to recognize the custom datatype.
    //
    // This method is really only required because of tz aware timestamps.
    // Tz aware timestamps are only needed because Bodo enforces that columns
    // must have a single timezone and that operations must be performed on two
    // columns with the same timezone information so we need to explicitly add
    // a cast when they differ. Snowflake doesn't have this requirement and snowflake
    // treats timezones as only the offset and the offsets as being part of the value
    // itself and not the column/data type. If we chose to lighten the requirement
    // for timezone logic, this method and the tz aware timestamp itself wouldn't
    // be needed.
    //
    // Keeping this note here just in case we decide to revisit this decision in
    // the future to make it easier for anyone looking at this to understand
    // why this is here.

    // Original Calcite code follows from here.

    // Transform the JavaType to SQL type because the SqlDataTypeSpec
    // does not support deriving JavaType yet.
    if (RelDataTypeFactoryImpl.isJavaType(targetType)) {
      targetType = ((JavaTypeFactory) factory).toSql(targetType);
    }

    SqlNode operand = call.getOperandList().get(index);
    if (operand instanceof SqlDynamicParam) {
      // Do not support implicit type coercion for dynamic param.
      return false;
    }
    requireNonNull(scope, "scope");
    // Check it early.
    if (!needToCast(scope, operand, targetType)) {
      return false;
    }
    // Fix up nullable attr.
    RelDataType targetType1 = syncAttributes(validator.deriveType(scope, operand), targetType);
    // Customize the castTo operation to use our own implementation.
    SqlNode desired = castTo(operand, targetType1);
    call.setOperand(index, desired);
    updateInferredType(desired, targetType1);
    return true;
  }

  private static SqlNode castTo(SqlNode node, RelDataType type) {
    // Utilize our own version of convertTypeToSpec.
    return SqlStdOperatorTable.CAST.createCall(SqlParserPos.ZERO, node,
        BodoSqlTypeUtil.convertTypeToSpec(type).withNullable(type.isNullable()));
  }

  private static class TypeCoercionFactoryImpl implements TypeCoercionFactory {
    @Override
    public TypeCoercion create(RelDataTypeFactory typeFactory, SqlValidator validator) {
      return new BodoTypeCoercionImpl(typeFactory, validator);
    }
  }

  /**
   * COALESCE type coercion, only used when not decomposing coalesce. In Bodo's version,
   * we follow the snowflake typing semantics for COALESCE coercion. Snowflake semantics is
   * a pairwise right fold, IE:
   * COALESCE(typ1, typ2, typ3) == COALESCE(GET_COMMON(typ1, GET_COMMON(typ2, typ3))
   *
   * Our version of this type coercion can differ from snowflake's semantics due to evaluation order difference,
   * but should otherwise be comprable
   */
  @Override public boolean coalesceCoercion(SqlCallBinding callBinding) {
    // For sql statement like:
    // `case when ... then (a, b, c) when ... then (d, e, f) else (g, h, i)`
    // an exception throws when entering this method.
    SqlCall call = callBinding.getCall();
    assert call.getOperator().getKind() == SqlKind.COALESCE;
    // Note, we have to create a newSqlNodeList here to use coerceColumnType
    // set's on this node list will not affect the actual operand list of the call, so we have
    // to propagate changes manually
    SqlNodeList originalOperandList = new SqlNodeList(call.getOperandList(), call.getParserPosition());
    List<RelDataType> originalArgTypes = new ArrayList<RelDataType>();
    SqlValidatorScope scope = getScope(callBinding);
    for (SqlNode node : originalOperandList) {
      originalArgTypes.add(
              validator.deriveType(
                      scope, node));
    }

    Boolean coerced = false;
    CoalesceTypeCastingUtils.SF_TYPE curRhsType = CoalesceTypeCastingUtils.Companion.TO_SF_TYPE(
            originalArgTypes.get(originalArgTypes.size()-1));

    // If we ever encounter a type that does not have an equivalent type in SF (Mainly interval types),
    // default to the super class's handling to avoid breaking any existing functionality.
    if (curRhsType == null) {
      return super.coalesceCoercion(callBinding) || coerced;
    }

    for (int i = originalArgTypes.size() - 2; i >= 0; i--) {
      CoalesceTypeCastingUtils.SF_TYPE lhsType = CoalesceTypeCastingUtils.Companion.TO_SF_TYPE(originalArgTypes.get(i));


      // If we ever encounter a type that does not have an equivalent type in SF (Mainly interval types),
      // default to the super class's handling to avoid breaking any existing functionality.
      if (lhsType == null) {
        return super.coalesceCoercion(callBinding) || coerced;
      }

      if (lhsType.equals(curRhsType)){
        continue;
      }
      // If we ever get here, set coerced to true
      coerced = true;

      Pair<CoalesceTypeCastingUtils.SF_TYPE, SqlOperator> newRhsTypeAndCastingFn =
              CoalesceTypeCastingUtils.Companion.sfGetCoalesceTypeAndCastingFn(lhsType, curRhsType);

      CoalesceTypeCastingUtils.SF_TYPE newRhsType = newRhsTypeAndCastingFn.getFirst();
      SqlOperator castingFn = newRhsTypeAndCastingFn.getSecond();
      if (castingFn.equals(TO_TIMESTAMP_TZ)){
        //We should never generate TIMESTAMP_TZ, even if that would be the SF behavior
        castingFn = TO_TIMESTAMP_LTZ;
      }

      if (curRhsType.equals(newRhsType)) {
        //In this case, only need to cast the LHS argument
        SqlNode valToCast = call.getOperandList().get(i);
        SqlNode newCall;
        //Snowflake semantics, when casting string to int, we use 18, 5 for scale/precision
        if (castingFn.equals(TO_NUMBER) && lhsType.equals(CoalesceTypeCastingUtils.SF_TYPE.VARCHAR)) {
          newCall = castingFn.createCall(valToCast.getParserPosition(), valToCast, SqlLiteral.createExactNumeric("18", SqlParserPos.ZERO), SqlLiteral.createExactNumeric("5", SqlParserPos.ZERO));
        } else {
          newCall = castingFn.createCall(valToCast.getParserPosition(), valToCast);
        }
        call.setOperand(i, newCall);
      } else{
        //In this case, need to cast all the RHS arguments
        for (int j = i + 1; j < originalArgTypes.size(); j++) {
          SqlNode valToCast = call.getOperandList().get(j);
          SqlNode newCall;
          //Snowflake semantics, when casting string to int, we use 18, 5 for scale/precision
          if (castingFn.equals(TO_NUMBER) && curRhsType.equals(CoalesceTypeCastingUtils.SF_TYPE.VARCHAR)) {
            newCall = castingFn.createCall(valToCast.getParserPosition(), valToCast, SqlLiteral.createExactNumeric("18", SqlParserPos.ZERO), SqlLiteral.createExactNumeric("5", SqlParserPos.ZERO));
          } else {
            newCall = castingFn.createCall(valToCast.getParserPosition(), valToCast);
          }
          call.setOperand(j, newCall);
        }
      }

      curRhsType = newRhsType;
    }

    // Finally,
    // Call super to handle potential need to unify precisions etc.
    return super.coalesceCoercion(callBinding) || coerced;
  }

}
