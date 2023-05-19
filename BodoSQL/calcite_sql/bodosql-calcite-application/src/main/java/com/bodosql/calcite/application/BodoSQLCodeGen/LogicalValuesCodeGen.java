package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.BodoArrayHelpers.sqlTypeToBodoArrayType;
import static com.bodosql.calcite.application.Utils.Utils.getBodoIndent;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.ir.Variable;
import java.util.Collections;
import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;

/** Class that returns the generated code for Logical Values after all inputs have been visited. */
public class LogicalValuesCodeGen {

  /**
   * Function that return the necessary generated code for a LogicalValues expression.
   *
   * @param outVar The output variable.
   * @param argExprs The expression for each argument.
   * @param rowType The row type of the output. This is used if there are no initial values.
   * @param pdVisitorClass The PandasCodeGenVisitor used to lower globals.
   * @return The code generated for the LogicalValues expression.
   */
  public static String generateLogicalValuesCode(
      String outVar,
      List<String> argExprs,
      RelDataType rowType,
      PandasCodeGenVisitor pdVisitorClass) {

    final String indent = getBodoIndent();

    StringBuilder outputStr = new StringBuilder();
    List<String> columnNames = rowType.getFieldNames();
    List<RelDataTypeField> sqlTypes = rowType.getFieldList();
    outputStr.append(indent).append(outVar).append(" = pd.DataFrame({");

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
      outputStr
          .append(makeQuoted(colName))
          .append(
              String.format(
                  ": bodo.utils.conversion.coerce_scalar_to_array(%s, %d, %s), ",
                  argExprs.get(i), columnLength, global.getName()));
    }
    outputStr.append(
        String.format(
            "}, index=bodo.hiframes.pd_index_ext.init_range_index(0, %d, 1, None))\n",
            columnLength));
    return outputStr.toString();
  }
}
