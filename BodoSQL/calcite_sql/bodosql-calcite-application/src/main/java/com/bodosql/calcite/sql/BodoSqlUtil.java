package com.bodosql.calcite.sql;

import static org.apache.calcite.sql.SqlUtil.permuteArgTypes;
import static org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE;
import static org.apache.calcite.util.Static.RESOURCE;

import com.bodosql.calcite.sql.validate.BodoCoercionUtil;
import com.google.common.collect.Lists;
import java.math.BigDecimal;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SnowflakeUserDefinedBaseFunction;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSampleSpec;
import org.apache.calcite.sql.SqlUtil;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlOperandMetadata;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.apache.calcite.sql.validate.SqlUserDefinedFunction;
import org.apache.calcite.sql.validate.SqlUserDefinedTableFunction;
import org.apache.calcite.util.Pair;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;

public class BodoSqlUtil {
  /** Creates a table sample spec from the parser information. */
  public static @Nullable SqlSampleSpec createTableSample(
      @NonNull SqlParserPos pos,
      boolean isBernoulli,
      BigDecimal rate,
      boolean sampleByRows,
      boolean isRepeatable,
      int repeatableSeed) {
    if (rate == null) {
      return null;
    }

    if (sampleByRows) {
      final BigDecimal maxRows = BigDecimal.valueOf(1000000L);
      if (rate.compareTo(BigDecimal.ZERO) < 0 || rate.compareTo(maxRows) > 0) {
        throw SqlUtil.newContextException(
            pos, BODO_SQL_RESOURCE.invalidSampleSize(BigDecimal.ZERO, maxRows));
      }

      return isRepeatable
          ? SqlTableSampleRowLimitSpec.createTableSample(isBernoulli, rate, repeatableSeed)
          : SqlTableSampleRowLimitSpec.createTableSample(isBernoulli, rate);
    }

    final BigDecimal oneHundred = BigDecimal.valueOf(100L);
    if (rate.compareTo(BigDecimal.ZERO) < 0 || rate.compareTo(oneHundred) > 0) {
      throw SqlUtil.newContextException(
          pos, BODO_SQL_RESOURCE.invalidSampleSize(BigDecimal.ZERO, oneHundred));
    }

    BigDecimal samplePercentage = rate.divide(oneHundred);
    if (samplePercentage.compareTo(BigDecimal.ZERO) < 0
        || samplePercentage.compareTo(BigDecimal.ONE) > 0) {
      throw SqlUtil.newContextException(pos, RESOURCE.invalidSampleSize());
    }
    return isRepeatable
        ? SqlSampleSpec.createTableSample(isBernoulli, samplePercentage, repeatableSeed)
        : SqlSampleSpec.createTableSample(isBernoulli, samplePercentage);
  }

  /**
   * Return if this OP represents a SQL default value.
   *
   * @param op The Sql operator.
   * @return Is this SqlStdOperatorTable.DEFAULT
   */
  public static boolean isDefaultCall(SqlOperator op) {
    return op == SqlStdOperatorTable.DEFAULT;
  }

  /**
   * Return if this SqlNode represents a SQL default value.
   *
   * @param node The SqlNode.
   * @return Is this call the default?
   */
  public static boolean isDefaultCall(SqlNode node) {
    if (node instanceof SqlCall) {
      return isDefaultCall(((SqlCall) node).getOperator());
    }
    return false;
  }

  /**
   * Return if this RexNode represents a SQL default value.
   *
   * @param node The RexNode.
   * @return Is this call the default?
   */
  public static boolean isDefaultCall(RexNode node) {
    if (node instanceof RexCall) {
      return isDefaultCall(((RexCall) node).getOperator());
    }
    return false;
  }

  /**
   * Filter a set of functions that can all be supported with implicit casting and selects the
   * implementation that casts the least by distance. This currently only supports Snowflake UDFs
   * and UDTFs and does nothing for all other functions.
   *
   * <p>The design can be found here:
   * https://bodo.atlassian.net/wiki/spaces/B/pages/1497169943/Inlining+Snowflake+SQL+UDFs+and+UDTFs#Bodo%E2%80%99s-UDF-Signature-Match-Score
   *
   * @param typeFactory Type factory for generating types
   * @param routines The possible routines from which to choose.
   * @param argTypes The types of the passed arguments.
   * @param argNames The names of the passed arguments.
   * @param nameMatcher The name matcher for resolving functions by name.
   * @return An iterator of functions that are selected. For Snowflake UDFs this will be an iterator
   *     containing a single function as we have found the best match.
   */
  public static Iterator<SqlOperator> filterRoutinesByScore(
      RelDataTypeFactory typeFactory,
      Iterator<SqlOperator> routines,
      List<RelDataType> argTypes,
      @Nullable List<String> argNames,
      final SqlNameMatcher nameMatcher) {
    final List<SqlOperator> list = Lists.newArrayList(routines);
    boolean isSnowflakeUDF =
        list.stream()
            .allMatch(
                x ->
                    (x instanceof SqlUserDefinedFunction
                            && (((SqlUserDefinedFunction) x).getFunction()
                                instanceof SnowflakeUserDefinedBaseFunction))
                        || (x instanceof SqlUserDefinedTableFunction
                            && (((SqlUserDefinedTableFunction) x).getFunction()
                                instanceof SnowflakeUserDefinedBaseFunction)));
    if (isSnowflakeUDF) {
      int minScore = -1;
      SqlOperator selectedRoutine = null;
      for (SqlOperator function : list) {
        SqlOperandTypeChecker operandTypeChecker =
            Objects.requireNonNull(function, "function").getOperandTypeChecker();
        final SqlOperandMetadata operandMetadata = (SqlOperandMetadata) operandTypeChecker;
        // Compute the sum of cast scores across all arguments.
        final List<@Nullable RelDataType> paramTypes = operandMetadata.paramTypes(typeFactory);
        final List<@Nullable RelDataType> permutedArgTypes;
        if (argNames != null) {
          final List<String> paramNames = operandMetadata.paramNames();
          // Bodo Change: Pass the name matcher.
          permutedArgTypes = permuteArgTypes(paramNames, argNames, argTypes, nameMatcher);
        } else {
          permutedArgTypes = Lists.newArrayList(argTypes);
          while (permutedArgTypes.size() < argTypes.size()) {
            paramTypes.add(null);
          }
        }
        int score = 0;
        for (Pair<@Nullable RelDataType, @Nullable RelDataType> p :
            Pair.zip(paramTypes, permutedArgTypes)) {
          final RelDataType argType = p.right;
          final RelDataType paramType = p.left;
          if (argType != null && paramType != null) {
            score += BodoCoercionUtil.Companion.getCastingMatchScore(argType, paramType);
          }
        }
        if (minScore == -1 || (score < minScore)) {
          minScore = score;
          selectedRoutine = function;
        } else if (score == minScore) {
          // If we have a match then we compare by timestamp.
          final SnowflakeUserDefinedBaseFunction baseFunctionNew;
          final SnowflakeUserDefinedBaseFunction baseFunctionKept;
          if (function instanceof SqlUserDefinedFunction) {
            baseFunctionNew =
                (SnowflakeUserDefinedBaseFunction)
                    ((SqlUserDefinedFunction) function).getFunction();
            baseFunctionKept =
                (SnowflakeUserDefinedBaseFunction)
                    ((SqlUserDefinedFunction) selectedRoutine).getFunction();
          } else {
            baseFunctionNew =
                (SnowflakeUserDefinedBaseFunction)
                    ((SqlUserDefinedTableFunction) function).getFunction();
            baseFunctionKept =
                (SnowflakeUserDefinedBaseFunction)
                    ((SqlUserDefinedTableFunction) selectedRoutine).getFunction();
          }
          if (baseFunctionNew.getCreatedOn().compareTo(baseFunctionKept.getCreatedOn()) < 0) {
            minScore = score;
            selectedRoutine = function;
          }
        }
      }
      return List.of(selectedRoutine).iterator();
    } else {
      // Maintain the existing behavior for non-udfs
      return list.iterator();
    }
  }
}
