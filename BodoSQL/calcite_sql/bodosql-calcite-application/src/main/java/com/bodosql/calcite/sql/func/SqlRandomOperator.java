package com.bodosql.calcite.sql.func;

import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;

public class SqlRandomOperator extends SqlFunction {
  public SqlRandomOperator() {
    super(
        "RANDOM",
        SqlKind.RANDOM,
        ReturnTypes.BIGINT,
        null,
        OperandTypes.NILADIC,
        SqlFunctionCategory.NUMERIC);
  }

  @Override
  public boolean isDeterministic() {
    return false;
  }
}
