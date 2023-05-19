package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.getBodoIndent;
import static com.bodosql.calcite.application.Utils.Utils.getDummyColNameBase;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;
import static com.bodosql.calcite.application.Utils.Utils.renameColumns;

import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.IntegerLiteral;
import com.bodosql.calcite.ir.Variable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Class that returns the generated code for Set operations after all the inputs have been visited.
 */
public class SetOpCodeGen {

  /**
   * Function that return the necessary generated code for a Union expression.
   *
   * @param outVar The output variable.
   * @param outputColumnNames a list containing the expected output column names
   * @param childExprs The child expressions to be Unioned together. The expressions must all be
   *     dataframes
   * @param isAll Is the union a UnionAll Expression.
   * @param pdVisitorClass Pandas Visitor used to create global variables.
   * @return The code generated for the Union expression.
   */
  public static String generateUnionCode(
      String outVar,
      List<String> outputColumnNames,
      List<String> childExprs,
      boolean isAll,
      PandasCodeGenVisitor pdVisitorClass) {
    StringBuilder unionBuilder = new StringBuilder();
    unionBuilder
        .append(getBodoIndent())
        .append(outVar)
        .append(" = ")
        .append("bodo.hiframes.pd_dataframe_ext.union_dataframes((");
    for (int i = 0; i < childExprs.size(); i++) {
      unionBuilder.append(childExprs.get(i)).append(", ");
    }
    unionBuilder.append("), ");
    if (isAll) {
      unionBuilder.append("False");
    } else {
      unionBuilder.append("True");
    }
    // generate output column names for ColNamesMetaType
    StringBuilder colNameTupleString = new StringBuilder("(");
    for (String colName : outputColumnNames) {
      colNameTupleString.append(makeQuoted(colName)).append(", ");
    }
    colNameTupleString.append(")");
    List<Expr.StringLiteral> colNamesExpr = new ArrayList<>();
    for (String colName : outputColumnNames) {
      colNamesExpr.add(new Expr.StringLiteral(colName));
    }
    Expr.Tuple colNameTuple = new Expr.Tuple(colNamesExpr);
    Variable globalVarName = pdVisitorClass.lowerAsColNamesMetaType(colNameTuple);
    unionBuilder.append(", ").append(globalVarName.getName()).append(")\n");
    return unionBuilder.toString();
  }

  /**
   * Function that return the necessary generated code for a Intersect expression.
   *
   * @param outVar The output variable.
   * @param lhsExpr The expression of the left hand table
   * @param lhsColNames The names of columns of the left hand table
   * @param rhsExpr The expression of the right hand table
   * @param rhsColNames The names of columns of the right hand table
   * @param columnNames a list containing the expected output column names
   * @param isAll Is the intersect an IntersectAll expression.
   * @param pdVisitorClass The calling pandas visitor class, used to generate temp var names
   * @return The code generated for the Intersect expression.
   */
  public static String generateIntersectCode(
      String outVar,
      String lhsExpr,
      List<String> lhsColNames,
      String rhsExpr,
      List<String> rhsColNames,
      List<String> columnNames,
      boolean isAll,
      PandasCodeGenVisitor pdVisitorClass) {
    // We need there to be at least one column, in the right/left table, so we can perform the merge
    // This may be incorrect if Calcite does not optimize out empty intersects
    assert lhsColNames.size() == rhsColNames.size()
        && lhsColNames.size() == columnNames.size()
        && lhsColNames.size() > 0;

    StringBuilder intersectBuilder = new StringBuilder();

    // Rename all lhs and rhs columns to match the expected output columns
    HashMap<String, String> lhsRenameMap = new HashMap<>();
    HashMap<String, String> rhsRenameMap = new HashMap<>();
    for (int i = 0; i < lhsColNames.size(); i++) {
      lhsRenameMap.put(lhsColNames.get(i), columnNames.get(i));
      rhsRenameMap.put(rhsColNames.get(i), columnNames.get(i));
    }

    final String indent = getBodoIndent();

    if (isAll) {
      // For IntersectAll, we use groupby to compute a unique cumulative count per row,
      // then we perform an inner join on all the columns as well as the cumcount.
      // Rows with x copies in lhsExpr and y copies in rhsExpr will have min(x, y) copies in outVar
      //
      // Simply avoiding drop_duplicates is not enough: pd.merge(x * [1], y * [1])
      // gives (x * y) * [1], rather than min(x, y) * [1] which is correct.
      //
      // Example Codegen:
      //   outVar = lhsDfCnt.rename(columns={...lhsRenameMap...}, copy=False)
      //                    .merge(rhsDfCnt.rename(columns={...rhsRenameMap...}, copy=False),
      //                           on=[...columnNames..., "__bodo_dummy__"])
      //                    .drop(columns="__bodo_dummy__"])
      // where lhsDfCnt and rhsDfCnt are generated by generateCumcountDf().

      // Generate temp vars
      final String lhsDfCnt = pdVisitorClass.genDfVar();
      final String rhsDfCnt = pdVisitorClass.genDfVar();

      String lhsCumcountCode = generateCumcountDf(lhsDfCnt, lhsExpr, lhsColNames, pdVisitorClass);
      String rhsCumcountCode = generateCumcountDf(rhsDfCnt, rhsExpr, rhsColNames, pdVisitorClass);

      // Generate list of output column names
      StringBuilder dummyColumnNamesListString = new StringBuilder("[");
      for (int i = 0; i < columnNames.size(); i++) {
        dummyColumnNamesListString.append(makeQuoted(columnNames.get(i))).append(", ");
      }
      dummyColumnNamesListString.append(makeQuoted(getDummyColNameBase())).append(",]");
      String dummyColumnNamesList = dummyColumnNamesListString.toString();

      intersectBuilder
          .append(lhsCumcountCode)
          .append(rhsCumcountCode)
          .append(indent)
          .append(outVar)
          .append(" = ")
          .append(lhsDfCnt)
          .append(".rename(columns=")
          .append(renameColumns(lhsRenameMap))
          .append(", copy=False).merge(")
          .append(rhsDfCnt)
          .append(".rename(columns=")
          .append(renameColumns(rhsRenameMap))
          .append(", copy=False), on=")
          .append(dummyColumnNamesList)
          .append(").drop(columns=[")
          .append(makeQuoted(getDummyColNameBase()))
          .append("])\n");

    } else {
      // For Intersect, we drop duplicates in lhsExpr and rhsExpr, then perform an inner join
      // on all the columns. Need to perform a final drop to account for values in both tables.
      //
      // Example Codegen:
      //   df_out = <LHS>.merge(<RHS>, on=[...columnNames...]).drop_duplicates()
      //   where <LHS> = df_lhs.drop_duplicates().rename(columns={...lhsRenameMap...}, copy=False)
      //   and   <RHS> = df_rhs.drop_duplicates().rename(columns={...rhsRenameMap...}, copy=False)

      // Generate list of output column names
      StringBuilder columnNamesListString = new StringBuilder("[");
      for (int i = 0; i < columnNames.size(); i++) {
        columnNamesListString.append(makeQuoted(columnNames.get(i))).append(", ");
      }
      columnNamesListString.append("]");
      String columnNamesList = columnNamesListString.toString();

      intersectBuilder
          .append(indent)
          .append(outVar)
          .append(" = ")
          .append(lhsExpr)
          .append(".drop_duplicates().rename(columns=")
          .append(renameColumns(lhsRenameMap))
          .append(", copy=False).merge(")
          .append(rhsExpr)
          .append(".drop_duplicates().rename(columns=")
          .append(renameColumns(rhsRenameMap))
          .append(", copy=False), on=")
          .append(columnNamesList)
          .append(").drop_duplicates()\n");
    }

    return intersectBuilder.toString();
  }

  /**
   * Function that return the necessary generated code for a Except expression.
   *
   * @param outVar The output variable.
   * @param lhsExpr The expression of the left hand table
   * @param lhsColNames The names of columns of the left hand table
   * @param rhsExpr The expression of the right hand table
   * @param rhsColNames The names of columns of the right hand table
   * @param columnNames a list containing the expected output column names
   * @param isAll Is the except an ExceptAll expression.
   * @param pdVisitorClass The calling pandas visitor class, used to generate temp var names
   * @return The code generated for the Except expression.
   */
  public static String generateExceptCode(
      String outVar,
      String lhsExpr,
      List<String> lhsColNames,
      String rhsExpr,
      List<String> rhsColNames,
      List<String> columnNames,
      boolean isAll,
      PandasCodeGenVisitor pdVisitorClass) {
    assert lhsColNames.size() == rhsColNames.size() && lhsColNames.size() == columnNames.size();

    StringBuilder exceptBuilder = new StringBuilder();

    // Rename all lhs and rhs columns to match the expected output columns
    HashMap<String, String> lhsRenameMap = new HashMap<>();
    HashMap<String, String> rhsRenameMap = new HashMap<>();
    for (int i = 0; i < lhsColNames.size(); i++) {
      lhsRenameMap.put(lhsColNames.get(i), columnNames.get(i));
      rhsRenameMap.put(rhsColNames.get(i), columnNames.get(i));
    }

    final String indent = getBodoIndent();

    if (isAll) {
      // For ExceptAll, we use groupby to compute a unique cumulative count per row.
      // Then, we concatenate the dataframes [lhsExpr, rhsExpr, rhsExpr] and use drop_duplicates
      // with `keep=False` to remove rows present in rhsExpr. We rely on the cumulative count
      // column to avoid dropping rows that repeat in lhsExpr but aren't found in rhsExpr.
      // Rows with x copies in lhsExpr and y copies in rhsExpr will have x - y copies in outVar,
      // or zero copies if x - y is negative.
      //
      // Example Codegen:
      //   rhsDfTmp = rhsDfCnt.rename(columns={...rhsRenameMap...}, copy=False)
      //   outVar = pd.concat([
      //       lhsDfCnt.rename(columns={...lhsRenameMap...}, copy=False),
      //       rhsDfTmp,
      //       rhsDfTmp
      //   ]).drop_duplicates(keep=False).drop(columns=["__bodo_dummy__"])
      // where lhsDfCnt and rhsDfCnt are generated by generateCumcountDf().

      // Generate temp vars
      final String lhsDfCnt = pdVisitorClass.genDfVar();
      final String rhsDfCnt = pdVisitorClass.genDfVar();
      final String rhsDfTmp = pdVisitorClass.genDfVar();

      String lhsCumcountCode = generateCumcountDf(lhsDfCnt, lhsExpr, lhsColNames, pdVisitorClass);
      String rhsCumcountCode = generateCumcountDf(rhsDfCnt, rhsExpr, rhsColNames, pdVisitorClass);

      exceptBuilder
          .append(lhsCumcountCode)
          .append(rhsCumcountCode)
          .append(indent)
          .append(rhsDfTmp)
          .append(" = ")
          .append(rhsDfCnt)
          .append(".rename(columns=")
          .append(renameColumns(rhsRenameMap))
          .append(", copy=False)\n")
          .append(indent)
          .append(outVar)
          .append(" = pd.concat([")
          .append(lhsDfCnt)
          .append(".rename(columns=")
          .append(renameColumns(lhsRenameMap))
          .append(", copy=False), ")
          .append(rhsDfTmp)
          .append(", ")
          .append(rhsDfTmp)
          .append("]).drop_duplicates(keep=False).drop(columns=[")
          .append(makeQuoted(getDummyColNameBase()))
          .append("])\n");

    } else {
      // For Except, we concatenate the dataframes [lhsExpr, rhsExpr, rhsExpr] and use
      // drop_duplicates with `keep=False` to remove rows present in rhsExpr.
      //
      // Example Codegen:
      //   rhsDfTmp = rhsExpr.rename(columns={...rhsRenameMap...}, copy=False).drop_duplicates()
      //   outVar = pd.concat([
      //      lhsExpr.rename(columns={...lhsRenameMap...}, copy=False).drop_duplicates(),
      //      rhsDfTmp,
      //      rhsDfTmp
      //   ]).drop_duplicates(keep=False)

      // Generate temp vars
      final String rhsDfTmp = pdVisitorClass.genDfVar();

      exceptBuilder
          .append(indent)
          .append(rhsDfTmp)
          .append(" = ")
          .append(rhsExpr)
          .append(".rename(columns=")
          .append(renameColumns(rhsRenameMap))
          .append(", copy=False).drop_duplicates()\n")
          .append(indent)
          .append(outVar)
          .append(" = pd.concat([")
          .append(lhsExpr)
          .append(".rename(columns=")
          .append(renameColumns(lhsRenameMap))
          .append(", copy=False).drop_duplicates(), ")
          .append(rhsDfTmp)
          .append(", ")
          .append(rhsDfTmp)
          .append("]).drop_duplicates(keep=False)\n");
    }

    return exceptBuilder.toString();
  }

  /**
   * Helper function that inserts a new column of cumulative counts into an existing dataframe. Used
   * in INTERSECT ALL and EXCEPT ALL codegen. Cumulative counts are stored in a new column named
   * getDummyColNameBase().
   *
   * @param outVar The output variable.
   * @param expr The expression of the input dataframe
   * @param colNames The names of columns in the input dataframe
   * @param pdVisitorClass Pandas Visitor used to create temps and global variables
   */
  private static String generateCumcountDf(
      String outVar, String expr, List<String> colNames, PandasCodeGenVisitor pdVisitorClass) {
    StringBuilder cumcountBuilder = new StringBuilder();
    final String indent = getBodoIndent();

    // Since cumcount currently lacks JIT support, we use cumsum on a column of one's instead.
    // We use init_dataframe to efficiently add columns in place to an existing table.
    // Since BodoSQL never uses Index values, we replace the index with a dummy RangeIndex:
    // this avoids MultiIndex issues and allows Bodo to optimize more.
    // We also replace the Groupby index to inform the compiler that len(expr) == len(groupby).
    //
    // Example Codegen:
    //   colOnes = np.ones((len(expr),), dtype=np.int64)
    //   tableOnes = logical_table_to_table(get_dataframe_all_data(expr), (colOnes,),
    //                                      MetaType(...dummyColIdxs...), expr.shape[1])
    //   dfOnes = init_dataframe((tableOnes,), pd.RangeIndex(0, len(expr), 1),
    //                           ColNamesMetaType(...colNames..., "__bodo_dummy__"))
    //   colCnt = dfOnes.groupby([...colNames...], dropna=False).cumsum()["__bodo_dummy__"]
    //   tableCnt = logical_table_to_table(get_dataframe_all_data(expr), (colCnt,),
    //                                     MetaType(...dummyColIdxs...), expr.shape[1])
    //   dfCnt = init_dataframe((tableCnt,), pd.RangeIndex(0, len(expr), 1),
    //                          ColNamesMetaType(...colNames..., "__bodo_dummy__"))

    // Generate temp vars
    final String colOnes = pdVisitorClass.genSeriesVar();
    final String tableOnes = pdVisitorClass.genTableVar();
    final String dfOnes = pdVisitorClass.genDfVar();
    final String colCnt = pdVisitorClass.genSeriesVar();
    final String tableCnt = pdVisitorClass.genTableVar();
    final String dfCnt = outVar;

    // Generate dummyColIdxsGlobal, dummyColNamesGlobal, and colNamesList
    List<IntegerLiteral> dummmyColIdxs = new ArrayList<>();
    List<Expr.StringLiteral> dummyColNameExprs = new ArrayList<>();
    List<Expr.StringLiteral> colNameExprs = new ArrayList<>();

    for (int i = 0; i < colNames.size(); i++) {
      dummmyColIdxs.add(new Expr.IntegerLiteral(i));
      Expr.StringLiteral colName = new Expr.StringLiteral(colNames.get(i));
      dummyColNameExprs.add(colName);
      colNameExprs.add(colName);
    }
    dummmyColIdxs.add(new IntegerLiteral(colNames.size()));
    Expr.StringLiteral dummyColNameExpr = new Expr.StringLiteral(getDummyColNameBase());
    dummyColNameExprs.add(dummyColNameExpr);

    // Create tuples
    Expr.Tuple dummyColIdxsTuple = new Expr.Tuple(dummmyColIdxs);
    Expr.Tuple dummyColNamesTuple = new Expr.Tuple(dummyColNameExprs);
    // Create a list
    Expr.List colNamesList = new Expr.List(colNameExprs);

    Variable dummyColIdxsGlobal = pdVisitorClass.lowerAsMetaType(dummyColIdxsTuple);
    Variable dummyColNamesGlobal = pdVisitorClass.lowerAsColNamesMetaType(dummyColNamesTuple);

    // TODO: Refactor to use Exprs
    // Compute dfOnes
    cumcountBuilder
        .append(indent)
        .append(colOnes)
        .append(" = np.ones((len(")
        .append(expr)
        .append("),), dtype=np.int64)\n")
        .append(indent)
        .append(tableOnes)
        .append(" = bodo.hiframes.table.logical_table_to_table(")
        .append("bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(")
        .append(expr)
        .append("), (")
        .append(colOnes)
        .append(",), ")
        .append(dummyColIdxsGlobal.getName())
        .append(", ")
        .append(expr)
        .append(".shape[1])\n")
        .append(indent)
        .append(dfOnes)
        .append(" = bodo.hiframes.pd_dataframe_ext.init_dataframe((")
        .append(tableOnes)
        .append(",), pd.RangeIndex(0, len(")
        .append(expr)
        .append("), 1), ")
        .append(dummyColNamesGlobal.getName())
        .append(")\n");

    // Compute dfCnt
    cumcountBuilder
        .append(indent)
        .append(colCnt)
        .append(" = ")
        .append(dfOnes)
        .append(".groupby(")
        .append(colNamesList.emit())
        .append(", dropna=False).cumsum()[")
        .append(makeQuoted(getDummyColNameBase()))
        .append("]\n")
        .append(indent)
        .append(tableCnt)
        .append(" = bodo.hiframes.table.logical_table_to_table(")
        .append("bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(")
        .append(expr)
        .append("), (")
        .append(colCnt)
        .append(",), ")
        .append(dummyColIdxsGlobal.getName())
        .append(", ")
        .append(expr)
        .append(".shape[1])\n")
        .append(indent)
        .append(dfCnt)
        .append(" = bodo.hiframes.pd_dataframe_ext.init_dataframe((")
        .append(tableCnt)
        .append(",), pd.RangeIndex(0, len(")
        .append(expr)
        .append("), 1), ")
        .append(dummyColNamesGlobal.getName())
        .append(")\n");

    return cumcountBuilder.toString();
  }
}
