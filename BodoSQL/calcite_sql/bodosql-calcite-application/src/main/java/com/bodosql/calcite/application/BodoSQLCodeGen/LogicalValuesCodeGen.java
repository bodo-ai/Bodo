package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.utils.BodoArrayHelpers.sqlTypeToBodoArrayType;
import static com.bodosql.calcite.application.utils.Utils.integerLiteralArange;

import com.bodosql.calcite.application.BodoCodeGenVisitor;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;

/** Class that returns the generated code for Logical Values after all inputs have been visited. */
public class LogicalValuesCodeGen {

  /**
   * Function that return the necessary generated code for a LogicalValues expression.
   *
   * @param argRows The expressions for each row.
   * @param rowType The row type of the output. This is used if there are no initial values.
   * @param bodoVisitorClass The BodoCodeGenVisitor used to lower globals.
   * @return The code generated for the LogicalValues expression.
   */
  public static Expr generateLogicalValuesCode(
      final List<List<Expr>> argRows,
      final RelDataType rowType,
      final BodoCodeGenVisitor bodoVisitorClass) {

    List<RelDataTypeField> sqlTypes = rowType.getFieldList();
    // Create the types for the output arrays.
    final List<Expr> arrayTypes = new ArrayList<>();
    // Empty table doesn't support dictionary encoding yet.
    boolean allowDictArrays = argRows.size() != 0;
    for (RelDataTypeField field : sqlTypes) {
      arrayTypes.add(
          sqlTypeToBodoArrayType(
              field.getType(), allowDictArrays, bodoVisitorClass.genDefaultTZ().getZoneExpr()));
    }
    // Generate the lists to insert
    final int numArrays = rowType.getFieldCount();
    final List<Expr> dataValues = new ArrayList<>();
    for (int i = 0; i < numArrays; i++) {
      List<Expr> col = new ArrayList<>();
      for (List<Expr> row : argRows) {
        col.add(row.get(i));
      }
      dataValues.add(new Expr.List(col));
    }

    if (argRows.size() == 0) {
      // Special code path with length 0 data. This is necessary to avoid
      // issues with lists.
      Expr.Tuple typeTuple = new Expr.Tuple(arrayTypes);
      Expr tableType = new Expr.Call("bodo.types.TableType", typeTuple);
      // Move the table type to a global
      Variable global = bodoVisitorClass.lowerAsGlobal(tableType);
      return new Expr.Call("bodo.hiframes.table.create_empty_table", global);
    } else {
      // Generate a list of columns to wrap in a table.
      List<Expr> columns = new ArrayList<>();
      for (int i = 0; i < dataValues.size(); i++) {
        // Convert the list to an array.
        Variable global = bodoVisitorClass.lowerAsGlobal(arrayTypes.get(i));
        Expr arrCall =
            new Expr.Call("bodo.utils.conversion.list_to_array", dataValues.get(i), global);
        columns.add(arrCall);
      }
      List<Expr.IntegerLiteral> indices = integerLiteralArange(numArrays);
      Variable buildIndices = bodoVisitorClass.lowerAsMetaType(new Expr.Tuple(indices));
      // Create the table
      return new Expr.Call(
          "bodo.hiframes.table.logical_table_to_table",
          new Expr.Tuple(),
          new Expr.Tuple(columns),
          buildIndices,
          new Expr.IntegerLiteral(0));
    }
  }
}
