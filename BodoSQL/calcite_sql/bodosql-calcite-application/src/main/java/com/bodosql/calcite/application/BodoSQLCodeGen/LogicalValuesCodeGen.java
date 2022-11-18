package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.*;

import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.type.SqlTypeName;

/** Class that returns the generated code for Logical Values after all inputs have been visited. */
public class LogicalValuesCodeGen {

  /**
   * Function that return the necessary generated code for a LogicalValues expression.
   *
   * @param outVar The output variable.
   * @param argExprs The expression for each argument.
   * @param rowType The row type of the output. This is used if there are no initial values.
   * @return The code generated for the LogicalValues expression.
   */
  public static String generateLogicalValuesCode(
      String outVar, List<String> argExprs, RelDataType rowType) {

    final String indent = getBodoIndent();

    StringBuilder outputStr = new StringBuilder();
    List<String> columnNames = rowType.getFieldNames();
    outputStr.append(indent).append(outVar).append(" = pd.DataFrame({");
    if (argExprs.size() == 0) {
      for (int i = 0; i < columnNames.size(); i++) {
        SqlTypeName typeName = rowType.getFieldList().get(i).getValue().getSqlTypeName();
        String dType = sqlTypenameToPandasTypename(typeName, false, false);
        outputStr
            .append(makeQuoted(rowType.getFieldList().get(i).getKey()))
            .append(": pd.Series(dtype=")
            .append(dType)
            .append("), ");
      }
      outputStr.append("})\n");
    } else {
      /* to the best of my knowledge, it seems that the logical value node contains one tuple, with the needed number
      of values to fill in the rows in the table */
      for (int i = 0; i < columnNames.size(); i++) {
        String colName = columnNames.get(i);
        outputStr.append(makeQuoted(colName)).append(": ");
        outputStr.append(argExprs.get(i)).append(",");
      }
      outputStr.append("}, index=pd.RangeIndex(0, 1, 1))\n");
    }
    return outputStr.toString();
  }
}
