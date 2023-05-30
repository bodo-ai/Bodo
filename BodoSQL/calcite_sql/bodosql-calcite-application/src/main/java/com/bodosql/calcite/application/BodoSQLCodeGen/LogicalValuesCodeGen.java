package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.BodoArrayHelpers.sqlTypeToBodoArrayType;

import com.bodosql.calcite.application.*;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import com.bodosql.calcite.ir.Variable;
import java.util.*;
import java.util.Collections;
import java.util.List;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;

/** Class that returns the generated code for Logical Values after all inputs have been visited. */
public class LogicalValuesCodeGen {

  /**
   * Function that return the necessary generated code for a LogicalValues expression.
   *
   * @param argExprs The expression for each argument.
   * @param rowType The row type of the output. This is used if there are no initial values.
   * @param pdVisitorClass The PandasCodeGenVisitor used to lower globals.
   * @return The code generated for the LogicalValues expression.
   */
  public static Expr generateLogicalValuesCode(
      List<String> argExprs, RelDataType rowType, PandasCodeGenVisitor pdVisitorClass) {

    List<String> columnNames = rowType.getFieldNames();
    List<RelDataTypeField> sqlTypes = rowType.getFieldList();
    List<kotlin.Pair<Expr, Expr>> dfItems = new ArrayList<>();

    final int columnLength;
    if (argExprs.size() == 0) {
      columnLength = 0;
      // Generate a list of all Nones for consistent codegen.
      argExprs = Collections.nCopies(columnNames.size(), "None");
    } else {
      columnLength = 1;
    }
    // Logical Values contain a tuple of entries to fill one row
    for (int i = 0; i < columnNames.size(); i++) {
      // Scalars require separate code path to handle null.
      Variable global =
          pdVisitorClass.lowerAsGlobal(sqlTypeToBodoArrayType(sqlTypes.get(i).getType(), true));
      String colName = columnNames.get(i);

      // TODO (allai5): need to refactor this
      Expr expression = new Expr.Raw(argExprs.get(i));
      Expr length = new Expr.IntegerLiteral(columnLength);

      List<Expr> scalarToArrayArgs = List.of(expression, length, global);
      Expr.Call value =
          new Expr.Call("bodo.utils.conversion.coerce_scalar_to_array", scalarToArrayArgs);
      dfItems.add(new kotlin.Pair<>(new Expr.StringLiteral(colName), value));
    }

    Expr length = new Expr.IntegerLiteral(columnLength);
    List<Expr> indexArgs =
        List.of(new Expr.IntegerLiteral(0), length, new Expr.IntegerLiteral(1), Expr.None.INSTANCE);
    Expr index = ExprKt.initRangeIndex(indexArgs, List.of());
    return new Expr.Call("pd.DataFrame",
        List.of(new Expr.Dict(dfItems)),
        List.of(new kotlin.Pair<String, Expr>("index", index)));
  }
}
