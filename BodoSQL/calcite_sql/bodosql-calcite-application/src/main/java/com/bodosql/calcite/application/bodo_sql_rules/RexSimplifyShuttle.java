package com.bodosql.calcite.application.bodo_sql_rules;

import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.format.DateTimeParseException;
import java.time.temporal.ChronoField;
import java.util.*;
import net.snowflake.client.jdbc.internal.org.checkerframework.checker.nullness.qual.Nullable;
import org.apache.calcite.avatica.util.DateTimeUtils;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.*;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.fun.SqlCastFunction;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.util.Pair;

/**
 * RexSimplifyShuttle is used exclusively in RexSimplificationRule. As a part of that rule, this
 * shuttle is applied to ALL RexNodes in the given plan. Therefore, whenever doing a replacement,
 * this shuttle must always return equivalent expressions, with the same typing.
 *
 * <p>The current list of optimizations performed: converts COALESCE(date/timestamp_expr::string,
 * constant) <COMPARISON OP> constant into COALESCE(date/timestamp_expr, constant::date/timestamp)
 * <COMPARISON OP> constant::date/timestamp, provided that the constants can be safeley converted to
 * date/timestamp (whichever is the type of arg0)
 *
 * <p>This shuttle can and may be extended in the future, by overriding the associated visitX
 * functions, and matching/replacing whatever RexNodes you need to replace.
 */
public class RexSimplifyShuttle extends RexShuttle {

  final RexBuilder builder;

  public RexSimplifyShuttle(RexBuilder builder) {
    this.builder = builder;
  }

  public static final EnumSet<SqlKind> ComparisonOperators =
      EnumSet.of(
          SqlKind.EQUALS,
          SqlKind.NULL_EQUALS,
          SqlKind.NOT_EQUALS,
          SqlKind.GREATER_THAN,
          SqlKind.GREATER_THAN_OR_EQUAL,
          SqlKind.LESS_THAN,
          SqlKind.LESS_THAN_OR_EQUAL,
          SqlKind.NULL_EQUALS);

  @Override
  public RexNode visitCall(final RexCall call) {

    // Optimization in the case that we have an expression of the form
    // COALESCE(expr, constant) <COMPARISON OP> constant
    if (ComparisonOperators.contains(call.getOperator().getKind())
        && call.getOperands().size() == 2) {
      RexNode arg0 = call.getOperands().get(0);
      RexNode arg1 = call.getOperands().get(1);

      Pair<RexNode, RexLiteral> coalesceArgs;
      RexNode output = null;
      if (isConstantCoalesceWithArg0Cast(arg0) && isNonNullLiteralOrCastedLiteral(arg1)) {
        // If we have an expression of the form
        // COALESCE(expr, constant)::typ <COMPARISON OP> constant
        // OR
        // COALESCE(expr::typ, constant) <COMPARISON OP> constant
        // attempt to do the transformation. This will return NULL in the case that
        // we are unable to due to typing/string parsing issues.
        coalesceArgs = getConstantCoalesceArguments((RexCall) arg0);
        output =
            attemptCoalesceTransform(
                call.getOperator(), coalesceArgs, extractLiteralIfCast(arg1), true);
      } else if (isConstantCoalesceWithArg0Cast(arg1) && isNonNullLiteralOrCastedLiteral(arg0)) {
        // ditto, but for
        // constant <COMPARISON OP> COALESCE(expr::typ, constant)
        coalesceArgs = getConstantCoalesceArguments((RexCall) arg1);
        output =
            attemptCoalesceTransform(
                call.getOperator(), coalesceArgs, extractLiteralIfCast(arg0), false);
      }

      if (output != null) {
        // check that we don't change any types.
        // This check should always be true, but in the case that it isn't,
        // we don't attempt to perform the optimisation.
        if (output.getType() == call.getType()) {
          return output;
        }
      }
    }

    // Continue visiting
    return super.visitCall(call);
  }

  public boolean isConstantCoalesceWithArg0Cast(RexNode node) {
    return node instanceof RexCall && isConstantCoalesceWithArg0Cast((RexCall) node);
  }

  /**
   * Returns True if the argument is a "constant coalesce" where the 0th argument is cast to a
   * different type.
   *
   * <p>IE an expression of the form: COALESCE(expr0, constant)::not_expr0_type OR
   * COALESCE(expr0::not_expr0_type, constant)
   *
   * <p>(expr, constant, or the coalesce call itself may be wrapped in an arbitrary number of casts,
   * IE COALESCE(expr::varchar, constant::varchar)::date is still considered a "constant coalesce" )
   *
   * @param call
   * @return
   */
  public boolean isConstantCoalesceWithArg0Cast(RexCall call) {
    RexNode unWrappedNode = unWrapIfCast(call);
    if (!(unWrappedNode instanceof RexCall)) {
      return false;
    }
    RexCall unWrappedCall = (RexCall) unWrappedNode;
    return unWrappedCall.isA(SqlKind.COALESCE)
        && unWrappedCall.getOperands().size() == 2
        && isNonNullLiteralOrCastedLiteral(unWrappedCall.getOperands().get(1))
        && (isCast(call) | isCast(unWrappedCall.getOperands().get(0)));
  }

  /** Helper function that returns true if the given rexNode is a cast function */
  public Boolean isCast(RexNode node) {
    return node instanceof RexCall && node.isA(SqlKind.CAST);
  }

  /**
   * Extracts the arguments to a "constant coalesce" (see isConstantCoalesce), stripping away all
   * intermediate casts.
   *
   * @param call
   * @return
   */
  public Pair<RexNode, RexLiteral> getConstantCoalesceArguments(RexCall call) {
    assert isConstantCoalesceWithArg0Cast(call)
        : "Internal error: getConstantCoalesceArguments requires constant coalesce input";
    RexCall unWrappedCall = (RexCall) unWrapIfCast(call);
    return Pair.of(
        unWrapIfCast(unWrappedCall.getOperands().get(0)),
        extractLiteralIfCast(unWrappedCall.getOperands().get(1)));
  }

  public boolean isNonNullLiteralOrCastedLiteral(RexNode node) {
    RexNode unWrapedNode = unWrapIfCast(node);
    return unWrapedNode instanceof RexLiteral
        && ((RexLiteral) unWrapedNode).getTypeName() != SqlTypeName.NULL;
  }

  /**
   * Takes an expression that is a RexLiteral, or a RexLiteral wrapped in an arbitrary number of
   * casts. Returns the inner RexLiteral.
   */
  public RexLiteral extractLiteralIfCast(RexNode node) {
    assert isNonNullLiteralOrCastedLiteral(node)
        : "Internal error: getLiteralOrWrappedLiteral must be called on a non-null literal, or"
            + " wrapped literal value";
    return (RexLiteral) unWrapIfCast(node);
  }

  /**
   * Helper function that strips all casts from a given node, and returns the innermost value.
   *
   * <p>Note this doesn't strip any arguments to the function, IE unWrapCast(SQL_FN(x::varchar))
   * returns SQL_FN(x::varchar)
   *
   * <p>but unWrapCast(SQL_FN(x)::varchar) returns SQL_FN(x)
   */
  public RexNode unWrapIfCast(RexNode node) {
    if (node instanceof RexCall) {
      RexCall call = (RexCall) node;
      if (call.getOperator() instanceof SqlCastFunction) {
        return unWrapIfCast(call.operands.get(0));
      }
    }
    return node;
  }

  /**
   * At this point, we've determined we have a call with the form: COALESCE(Expr0, Literal1)::typ
   * <COMPARISON OP> Literal2 OR COALESCE(Expr0::typ, Literal1) <COMPARISON OP> Literal2
   *
   * <p>Where Expr0, Literal1, and Literal2 and the COALESCE call itself may be wrapped in an
   * arbitrary number of casts.
   *
   * <p>This function will return COALESCE(Expr0, Literal1::expr0_type) <COMPARISON OP>
   * Literal2::expr0_type
   *
   * <p>In the case that this conversion is safe, and NULL if the conversion cannot be safely
   * performed.
   *
   * @param op The comparison op in question.
   * @param coalesceArgs The arguments to the COALESCE, stripped of any intermediate casts
   * @param literalNode The argument that is being compared with the constant coalesce, stripped of
   *     any intermediate casts.
   * @param literalNodeIsRhs Is the literalNode on the RHS on the LHS
   * @return
   */
  public @Nullable RexNode attemptCoalesceTransform(
      SqlOperator op,
      Pair<RexNode, RexLiteral> coalesceArgs,
      RexLiteral literalNode,
      boolean literalNodeIsRhs) {

    RexNode coalesceDefaultValue = coalesceArgs.left;

    // In the case that the coalesce default value is a null literal, don't proceed with the
    // optimization
    if (coalesceDefaultValue instanceof RexLiteral
        && ((RexLiteral) coalesceDefaultValue).getTypeName() == SqlTypeName.NULL) {
      return null;
    }

    // First, verify we can cast the two literals to the type of columnar argument
    RelDataType newArgumentType = coalesceDefaultValue.getType();

    RelDataType newArgumentTypeNonNullable = builder.makeNotNull(coalesceDefaultValue).getType();

    RexLiteral coalesceFillValue = coalesceArgs.right;

    RelDataType coalesceFillValueType = coalesceFillValue.getType();
    RelDataType rhsNodeType = literalNode.getType();

    // For both of the literal types, check that the cast is safe, and won't change the overall
    // output.
    // If it isn't, we can't perform this optimization
    if (!(isValidCast(rhsNodeType, newArgumentType, literalNode)
        && isValidCast(coalesceFillValueType, newArgumentType, coalesceFillValue))) {

      return null;
    }

    // Note: we have to explicitly tell the builder that these casts are guaranteed to not create a
    // nullable output.
    // This is safe because we've already checked that the literal can be safely cast.
    // If we don't cast, the nullability of the argument may propagate, which can lead to changing
    // types, which can lead to errors.
    // For example, if we have `COALESCE(Expr0, Literal1)` where expr0 is null able, Calcite will
    // correctly infer that, because literal1 can never be null, the overall type of the
    // COALESCE expression is non null.
    //
    // However, if we have `COALESCE(Expr0, Literal1::Expr0_typ)`,
    // Calcite will not be able to infer that the the overall type of
    // the COALESCE expression is not null, because
    // `Literal1::Expr0_typ` is potentially null.
    //
    // This can end up propagating null ability, which can cause weird issues,
    //
    RexNode newLiteralNode = builder.makeAbstractCast(newArgumentTypeNonNullable, literalNode);
    RexNode newCoalesceFillVal =
        builder.makeAbstractCast(newArgumentTypeNonNullable, coalesceFillValue);

    RexNode newCoalesceCall =
        builder.makeCall(SqlStdOperatorTable.COALESCE, coalesceDefaultValue, newCoalesceFillVal);

    if (literalNodeIsRhs) {
      return builder.makeCall(op, newCoalesceCall, newLiteralNode);
    } else {
      return builder.makeCall(op, newLiteralNode, newCoalesceCall);
    }
  }

  public static boolean canParseAsTimestamp(String s) {
    // https://docs.snowflake.com/en/sql-reference/parameters#label-timestamp-input-format
    // Which is apparently:
    // https://docs.snowflake.com/en/user-guide/date-time-input-output#label-date-time-input-output-timestamp-formats

    // NOTE: I am not covering all the supported formats, because there are many formats,
    // and most of them require TZ info (which I'm not handling).
    // Basically just trying to cover the most common cases.
    List<String> TimestampFormatStrings =
        Arrays.asList(
            "yyyy-MM-ddHH:mm",
            "yyyy-MM-dd'T'HH:mm",
            "yyyy-MM-ddHH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss");

    for (String formatString : TimestampFormatStrings) {

      if (DateTimeUtils.parseDateFormat(
              s, new SimpleDateFormat(formatString, Locale.ROOT), DateTimeUtils.UTC_ZONE)
          != null) {
        return true;
      }

      // Workaround for parsing fractional seconds here:
      // https://stackoverflow.com/questions/22588051/is-java-time-failing-to-parse-fraction-of-second
      // Note that this is sufficient for checking the formatting,
      // but the parsed value (which we're not using right now) will be slightly different.
      DateTimeFormatter dtf =
          new DateTimeFormatterBuilder()
              .appendPattern(formatString)
              .appendFraction(ChronoField.MILLI_OF_SECOND, 1, 6, true)
              .toFormatter();
      try {
        LocalDateTime.parse(s, dtf);
        return true;
      } catch (DateTimeParseException d) {

      }
    }
    return false;
  }

  public static boolean canParseAsDate(String s) {
    // https://docs.snowflake.com/en/sql-reference/parameters#label-date-input-format
    // Which is apparently:
    // https://docs.snowflake.com/en/user-guide/date-time-input-output#label-date-time-input-output-date-formats
    List<String> dateFormatStrings = Arrays.asList("yyyy-MM-dd", "dd-MMMM-yyyy", "MM/dd/yyyy");

    for (String formatString : dateFormatStrings) {
      if (DateTimeUtils.parseDateFormat(
              s, new SimpleDateFormat(formatString, Locale.ROOT), DateTimeUtils.UTC_ZONE)
          != null) {
        return true;
      }
    }
    return false;
  }

  /**
   * Checks if we can cast the given literal from the specified type to the specified type, without
   * any possibility of failure.
   *
   * <p>Currently, this is only supported for all combinations of string, Date, and timezone naive
   * timestamps.
   *
   * <p>In the future, this could be extended to handle more type combinations.
   *
   * @param fromType
   * @param toType
   * @param fromValue
   * @return
   */
  boolean isValidCast(RelDataType fromType, RelDataType toType, RexLiteral fromValue) {

    if (SqlTypeUtil.canCastFrom(toType, fromType, false)) {
      return true;
    } else {
      SqlTypeName toTypeSql = toType.getSqlTypeName();
      SqlTypeFamily toTypeSqlFamily = toTypeSql.getFamily();
      switch (fromType.getSqlTypeName().getFamily()) {
        case STRING:
        case CHARACTER:
          // GetValue2 returns the value without the charset preceding it
          // IE getValue -> _ISO-8859-1'2023-03-08 13:12:12'
          // GetValue2 -> '2023-03-08 13:12:12'
          String stringValue = fromValue.getValue2().toString();
          boolean canParseAsDate = canParseAsDate(stringValue);
          boolean canParseAsTimestamp = canParseAsTimestamp(stringValue);
          if (toTypeSqlFamily == SqlTypeFamily.DATE && canParseAsDate && !canParseAsTimestamp) {
            return true;
          } else if (toTypeSqlFamily == SqlTypeFamily.TIMESTAMP
              && (canParseAsTimestamp || canParseAsDate)) {
            return true;
          }
          break;
        case DATE:
          return SqlTypeFamily.TIMESTAMP.contains(toType) || SqlTypeFamily.DATE.contains(toType);
        case TIMESTAMP:
        case DATETIME:
          return SqlTypeFamily.TIMESTAMP.contains(toType);
        default:
          break;
      }
    }

    return false;
  }
}
