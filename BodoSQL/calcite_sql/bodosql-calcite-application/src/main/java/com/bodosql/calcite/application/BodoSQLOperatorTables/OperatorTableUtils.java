package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.*;
import org.apache.calcite.rel.type.*;

public class OperatorTableUtils {
  /**
   * Helper function to determine output type nullability for functions whose output can only be
   * null if there is a null input.
   *
   * @param operandTypes List of input types. The output is nullable if any input is nullable.
   * @return Is the output nullable.
   */
  public static boolean isOutputNullableCompile(List<RelDataType> operandTypes) {
    for (RelDataType operandType : operandTypes) {
      if (operandType.isNullable()) {
        return true;
      }
    }
    return false;
  }
}
