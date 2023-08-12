package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.utils.BodoArrayHelpers.sqlTypeToBodoArrayType;

import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import com.bodosql.calcite.ir.Variable;
import java.util.ArrayList;
import java.util.List;
import kotlin.Pair;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.sql.type.SqlTypeName;

public class NestedDataCodeGen {

  /**
   * Handles codegen for Snowflake TO_ARRAY function.
   *
   * @param visitor The PandasCodeGenVisitor for lowering globals.
   * @param fnOperation The RexNode for the TO_ARRAY call.
   * @param operands List of operands
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Expr for the TO_ARRAY function, if the input is NULL or ARRAY, it will return the input
   */
  public static Expr generateToArrayFnCode(
      PandasCodeGenVisitor visitor,
      RexCall fnOperation,
      List<Expr> operands,
      List<Pair<String, Expr>> streamingNamedArgs) {
    assert operands.size() == 1;
    RelDataType OperandType = fnOperation.getOperands().get(0).getType();
    if (OperandType.getSqlTypeName().equals(SqlTypeName.ARRAY)
        || OperandType.getSqlTypeName().equals(SqlTypeName.NULL)) {
      // if the input is array or null, just return it
      return operands.get(0);
    }
    Expr BodoSqlArrayType = sqlTypeToBodoArrayType(OperandType, false);
    Variable destTypeVar = visitor.lowerAsGlobal(BodoSqlArrayType);
    List<Expr> args = new ArrayList<>();
    args.addAll(operands);
    args.add(destTypeVar);
    return ExprKt.BodoSQLKernel("to_array", args, streamingNamedArgs);
  }
}
