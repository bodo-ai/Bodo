package com.bodosql.calcite.application.BodoSQLCodeGen;

import org.apache.calcite.avatica.SqlType;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.sql.type.SqlTypeName;

import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.ir.Expr;
import java.util.List;
import static com.bodosql.calcite.application.Utils.BodoArrayHelpers.sqlTypeToBodoArrayType;

public class NestedDataCodeGen {

  /**
   * Handles codegen for Snowflake TO_ARRAY function.
   *
   * @param operands List of operands
   * @return RexVisitorInfo for the TO_ARRAY function, if the input is NULL or ARRAY, it will return the input
   */
  public static Expr generateToArrayFnCode(
      PandasCodeGenVisitor visitor,
      RexCall fnOperation,
      List<Expr> operands) {
    assert operands.size() == 1;
    RelDataType OperandType = fnOperation.getOperands().get(0).getType();
    if (OperandType.getSqlTypeName().equals(SqlTypeName.ARRAY)
        || OperandType.getSqlTypeName().equals(SqlTypeName.NULL))
      // if the input is array or null, just return it
      return operands.get(0);
    Expr BodoSqlArrayType = sqlTypeToBodoArrayType(OperandType, false);
    Expr global_var = visitor.lowerAsGlobal(BodoSqlArrayType);
    return new Expr.Call("bodo.libs.bodosql_array_kernels.to_array", operands.get(0), global_var);
  }
}
