package com.bodo.iceberg;

import org.apache.iceberg.expressions.Literal;
import org.apache.iceberg.types.Types;

public class LiteralConverters {
  /**
   * Class holding helper functions used to create apache literal types from python. py4j will often
   * coerce primitive java types (float, str, etc) into their equivalent python counterpart, and
   * visa versa. This can make creating literals of a specific type difficult (float vs double, int
   * vs long, etc). This literal converter class helps to get around that by creating the literals
   * in Java, and returning them to python *
   */
  public static Literal<Integer> asIntLiteral(int val) {
    return Literal.of(val);
  }

  public static Literal<Long> asLongLiteral(long val) {
    return Literal.of(val);
  }

  public static Literal<Float> asFloatLiteral(float val) {
    return Literal.of(val);
  }

  public static Literal<Double> asDoubleLiteral(double val) {
    return Literal.of(val);
  }

  public static Literal<Types.TimestampType> microsecondsToTimestampLiteral(long val) {
    return Literal.of(val).to(Types.TimestampType.withoutZone());
  }

  public static Literal<Types.DateType> numDaysToDateLiteral(long val) {
    return Literal.of(val).to(Types.DateType.get());
  }

  public static Literal<CharSequence> asStringLiteral(CharSequence val) {
    return Literal.of(val);
  }

  public static Literal<Boolean> asBoolLiteral(boolean val) {
    return Literal.of(val);
  }
}
