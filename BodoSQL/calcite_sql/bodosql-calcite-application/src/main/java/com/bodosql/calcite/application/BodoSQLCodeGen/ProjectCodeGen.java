package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.BodoArrayHelpers.sqlTypeToBodoArrayType;
import static com.bodosql.calcite.application.Utils.Utils.escapePythonQuotes;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.application.RexNodeVisitorInfo;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.IntegerLiteral;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.Variable;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;

/**
 * Class that returns the generated code for Project expressions after all inputs have been visited.
 */
public class ProjectCodeGen {

  /*-
   * Generate Python code for a Project expression. Uses logical_table_to_table() to always create a
   * table in output for lower compilation time. For example, if input df has 3 columns and
   * projection takes the second column and adds another one created through computation:
   *   col_inds = MetaType((1, 3))
   *   T2 = bodo.hiframes.table.logical_table_to_table(
   *          bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (A,) col_inds, df.shape[1])
   *   df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T2,), df.index, col_names)
   *
   * @param inVar The input variable. This is only used if all dataframe members are scalars
   * @param outVar The output variable.
   * @param childExprs RexNodeVisitorInfos for each childColumn being projected.
   * @param exprTypes ExprType for each childColumn being projected.
   * @param sqlTypes SQLTypeName for each childColumn being projected. This is used by scalars to
   *     generate the correct array.
   * @param pdVisitorClass The calling pandas visitor class. Used in the case that we need to lower
   *     a global column name variable.
   * @param numInputTableColumns number of columns in input table (used for logical column number
   *     calculations)
   * @return The code generated for the Project expression.
   */
  public static List<Op.Assign> generateProjectCode(
      Variable inVar,
      Variable outVar,
      List<String> outputColumns,
      List<RexNodeVisitorInfo> childExprs,
      List<BodoSQLExprType.ExprType> exprTypes,
      List<RelDataType> sqlTypes,
      PandasCodeGenVisitor pdVisitorClass,
      int numInputTableColumns) {

    List<Op.Assign> outAssigns = new ArrayList<>();
    List<String> seriesNames = new ArrayList<>();

    List<String> scalarSeriesNames = new ArrayList<>();
    List<Integer> scalarSeriesIdxs = new ArrayList<>();

    // First pass to generate Series for Non-Scalars
    // current logical column number for non-table (series) input
    // extra columns are appended after the table input so numInputTableColumns is the first array
    int currentNonTableInd = numInputTableColumns;
    // logical column numbers in logical input table to logical_table_to_table() for creating output
    // table
    List<IntegerLiteral> outColInds = new ArrayList<>();
    for (int i = 0; i < childExprs.size(); i++) {
      BodoSQLExprType.ExprType exprType = exprTypes.get(i);
      RexNodeVisitorInfo childExpr = childExprs.get(i);

      int childIndex = childExpr.getIndex();
      // output is just a column of input table (RexInputRef case)
      if (childIndex != -1) {
        outColInds.add(new Expr.IntegerLiteral(childIndex));
        continue;
      }
      // logical column number in input for non-table data
      outColInds.add(new Expr.IntegerLiteral(currentNonTableInd));
      currentNonTableInd++;

      Variable seriesVar = pdVisitorClass.genSeriesVar();
      seriesNames.add(seriesVar.getName());

      switch (exprType) {
        case COLUMN:
          outAssigns.add(new Op.Assign(seriesVar, new Expr.Raw(childExpr.getExprCode())));
          break;

        case SCALAR:
          scalarSeriesNames.add(seriesVar.getName());
          scalarSeriesIdxs.add(i);
          break;

        case DATAFRAME:
          throw new IllegalArgumentException("Internal Error: Expression can't be a Dataframe");
      }
    }

    // Throw away previous Index and define a dummy Index since BodoSQL never uses Index values.
    // This avoids MultiIndex issues and allows Bodo to optimize more.
    Variable indexVar = pdVisitorClass.genIndexVar();
    outAssigns.add(
        new Op.Assign(
            indexVar, new Expr.Raw(String.format("pd.RangeIndex(0, len(%s), 1)", inVar.emit()))));

    // Generate array for scalar columns
    for (int i = 0; i < scalarSeriesNames.size(); i++) {
      Expr curExpr;
      int idx = scalarSeriesIdxs.get(i);
      if (exprTypes.get(idx) == BodoSQLExprType.ExprType.SCALAR) {
        // Scalars require separate code path to handle null.
        Variable global =
            pdVisitorClass.lowerAsGlobal(sqlTypeToBodoArrayType(sqlTypes.get(idx), true));
        curExpr =
            new Expr.Raw(
                new StringBuilder()
                    .append("bodo.utils.conversion.coerce_scalar_to_array(")
                    .append(childExprs.get(idx).getExprCode())
                    // Generate a unique name based on the output variable
                    .append(String.format(", len(%s), %s)", inVar.emit(), global.getName()))
                    .toString());
      } else {
        curExpr =
            new Expr.Raw(
                new StringBuilder()
                    .append("bodo.utils.conversion.coerce_to_array(")
                    .append(childExprs.get(idx).getExprCode())
                    .append(", True)")
                    .toString());
      }
      outAssigns.add(new Op.Assign(new Variable(scalarSeriesNames.get(i)), curExpr));
    }

    // generate output table, e.g. logical_table_to_table((in_table, S0, S1), col_indices)
    // logical_table_to_table() takes a tuple of input data (which forms a logical input table) and
    // returns a TableType.
    // The first element of input data is input dataframe's data, which can be a table or a tuple of
    // arrays.
    // The other elements are arrays. It also takes logical column indices in input to form the
    // output.
    // For example, if input T1 has 3 columns:
    // logical_table_to_table((T1, S0), (2, 3, 1)) creates a table with (T1_2, S0, T1_1)
    // logical_table_to_table(((A, B, C), S0), (2, 3, 1)) creates a table with (C, S0, B)
    Variable outTableVar = pdVisitorClass.genTableVar();
    StringBuilder tableExprRawString =
        new StringBuilder()
            .append(
                "bodo.hiframes.table.logical_table_to_table(bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(")
            .append(inVar.emit())
            .append("), (");
    for (String seriesName : seriesNames) {
      tableExprRawString.append(String.format("%s, ", seriesName));
    }
    tableExprRawString.append("), ");
    // generate tuple of column indices, e.g. (4, 2, 1)
    Expr.Tuple colIndTuple = new Expr.Tuple(outColInds);
    Variable colIndiceGlobalVarName = pdVisitorClass.lowerAsMetaType(colIndTuple);
    tableExprRawString.append(colIndiceGlobalVarName.emit());
    tableExprRawString.append(String.format(", %s.shape[1])", inVar.emit()));

    outAssigns.add(new Op.Assign(outTableVar, new Expr.Raw(tableExprRawString.toString())));
    // generate output column names for ColNamesMetaType
    List<Expr.StringLiteral> colNamesExpr = new ArrayList<>();
    for (String colName : outputColumns) {
      colNamesExpr.add(new Expr.StringLiteral(colName));
    }
    Expr.Tuple colNameTuple = new Expr.Tuple(colNamesExpr);
    Variable globalVar = pdVisitorClass.lowerAsColNamesMetaType(colNameTuple);

    // output dataframe is always in table format
    Expr tableTuple = new Expr.Tuple(List.of(outTableVar));
    Expr.Call initDf =
        new Expr.Call(
            "bodo.hiframes.pd_dataframe_ext.init_dataframe",
            List.of(tableTuple, indexVar, globalVar));
    outAssigns.add(new Op.Assign(outVar, initDf));
    return outAssigns;
  }

  /**
   * Function that returns the necessary generated code for a Project expression.
   *
   * @param inVar The input variable. This is only used if all dataframe members are scalars
   * @param childExprs RexNodeVisitorInfos for each childColumn being projected.
   * @param exprTypes ExprType for each childColumn being projected.
   * @return The code generated for the Project expression.
   */
  public static String generateProjectedDataframe(
      String inVar,
      List<String> childExprNames,
      List<Expr> childExprs,
      List<BodoSQLExprType.ExprType> exprTypes) {
    StringBuilder projectString = new StringBuilder("pd.DataFrame({");
    boolean allScalars = true;
    for (int i = 0; i < childExprs.size(); i++) {
      Expr column = childExprs.get(i);
      BodoSQLExprType.ExprType exprType = exprTypes.get(i);
      String name = escapePythonQuotes(childExprNames.get(i));
      projectString.append(makeQuoted(name)).append(": ");
      projectString.append(column.emit());

      projectString.append(", ");
      allScalars = (exprType == BodoSQLExprType.ExprType.SCALAR) && allScalars;
    }
    projectString.append("}");
    // If we have only scalar columns we need to provide an Index which matches
    // the size of the input Var.
    if (allScalars) {
      projectString.append(String.format(", index=pd.RangeIndex(0, len(%s), 1)", inVar));
    }
    projectString.append(")");
    return projectString.toString();
  }

  /**
   * Function that return the necessary generated code for a Project expression that can be
   * optimized to use df.loc.
   *
   * @param inVar The input variable. This is only used if all dataframe members are scalars
   * @param inputRefs List of RexNodes that map to inputRefs.
   * @param inColNames List containing the inVar column names.
   * @param outColNames List to populate with the column names for the outvar.
   * @return The code generated for the Project expression.
   */
  public static Expr generateLocCode(
      Variable inVar, List<RexNode> inputRefs, List<String> inColNames, List<String> outColNames) {
    StringBuilder locString = new StringBuilder().append(inVar.emit()).append(".loc[:, [");
    for (RexNode r : inputRefs) {
      RexInputRef inputRef = (RexInputRef) r;
      String colName = inColNames.get(inputRef.getIndex());
      outColNames.add(colName);
      locString.append(makeQuoted(colName)).append(", ");
    }
    locString.append("]]");
    return new Expr.Raw(locString.toString());
  }

  public static Expr generateRenameCode(Expr inExpr, TreeMap<String, String> colsToRename) {
    StringBuilder renameString =
        new StringBuilder()
            .append("(")
            .append(inExpr.emit())
            .append(")")
            .append(".rename(columns={");
    for (String key : colsToRename.keySet()) {
      renameString.append(makeQuoted(key));
      renameString.append(": ");
      renameString.append(makeQuoted(colsToRename.get(key)));
      renameString.append(", ");
    }
    renameString.append("}, copy=False)");
    return new Expr.Raw(renameString.toString());
  }
}
