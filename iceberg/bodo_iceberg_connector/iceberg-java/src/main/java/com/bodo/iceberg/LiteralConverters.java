package com.bodo.iceberg;

import com.bodo.iceberg.filters.Const;
import org.apache.iceberg.expressions.Literal;
import org.apache.iceberg.types.Types;

public class LiteralConverters {
  /**
   * Class holding helper functions used to create apache literal types from python. Py4J will often
   * coerce primitive java types (float, str, etc) into their equivalent python counterpart, and
   * vice versa. This can make creating literals of a specific type difficult (float vs double, int
   * vs long, etc.). This literal converter class helps to get around that by creating the literals
   * in Java, and returning them to python *
   */
  public static Const asTimeLiteral(long val) {
    return new Const(Literal.of(val).to(Types.TimeType.get()));
  }

  public static Const asIntLiteral(int val) {
    return new Const(Literal.of(val));
  }

  public static Const asLongLiteral(long val) {
    return new Const(Literal.of(val));
  }

  public static Const asFloatLiteral(float val) {
    return new Const(Literal.of(val));
  }

  public static Const asDoubleLiteral(double val) {
    return new Const(Literal.of(val));
  }

  public static Const microsecondsToTimestampLiteral(long val) {
    return new Const(Literal.of(val).to(Types.TimestampType.withoutZone()));
  }

  public static Const numDaysToDateLiteral(long val) {
    return new Const(Literal.of(val).to(Types.DateType.get()));
  }

  public static Const asStringLiteral(CharSequence val) {
    return new Const(Literal.of(val));
  }

  public static Const asBoolLiteral(boolean val) {
    return new Const(Literal.of(val));
  }

  public static Const asBinaryLiteral(byte[] val) {
    return new Const(Literal.of(val));
  }
}
