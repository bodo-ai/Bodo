package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.utils.Utils.getBodoIndent;
import static com.bodosql.calcite.application.utils.Utils.getDummyColNameBase;
import static com.bodosql.calcite.application.utils.Utils.makeQuoted;
import static com.bodosql.calcite.application.utils.Utils.renameColumns;

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel;
import com.bodosql.calcite.application.BodoCodeGenVisitor;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.IntegerLiteral;
import com.bodosql.calcite.ir.Op;
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
   * @param outputColumnNames a list containing the expected output column names
   * @param childExprs The child expressions to be Unioned together. The expressions must all be
   *     dataframes
   * @param isAll Is the union a UnionAll Expression.
   * @param ctx BuildContext used to create global variables.
   * @return The code generated for the Union expression.
   */
  public static Expr generateUnionCode(
      List<String> outputColumnNames,
      List<Variable> childExprs,
      boolean isAll,
      BodoPhysicalRel.BuildContext ctx) {

    Expr.Tuple dfTup = new Expr.Tuple(childExprs);

    // (isAll = True) -> False
    Expr.BooleanLiteral dropDuplicates = new Expr.BooleanLiteral(!isAll);

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
    Variable globalVarName = ctx.lowerAsColNamesMetaType(colNameTuple);

    Expr.Call unionExpr =
        new Expr.Call(
            "bodo.hiframes.pd_dataframe_ext.union_dataframes",
            List.of(dfTup, dropDuplicates, globalVarName));

    return unionExpr;
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
   * @param isAll Is this intersect an IntersectAll expression.
   * @param bodoVisitorClass The calling Bodo visitor class, used to generate temp var names
   * @return The code generated for the Intersect expression.
   */
  public static List<Op> generateIntersectCode(
      Variable outVar,
      Variable lhsExpr,
      List<String> lhsColNames,
      Variable rhsExpr,
      List<String> rhsColNames,
      List<String> columnNames,
      boolean isAll,
      BodoCodeGenVisitor bodoVisitorClass) {
    // We need there to be at least one column, in the right/left table, so we can
    // perform the merge
    // This may be incorrect if Calcite does not optimize out empty intersects
    assert lhsColNames.size() == rhsColNames.size()
        && lhsColNames.size() == columnNames.size()
        && lhsColNames.size() > 0;

    // Rename all lhs and rhs columns to match the expected output columns
    HashMap<String, String> lhsRenameMap = new HashMap<>();
    HashMap<String, String> rhsRenameMap = new HashMap<>();
    for (int i = 0; i < lhsColNames.size(); i++) {
      lhsRenameMap.put(lhsColNames.get(i), columnNames.get(i));
      rhsRenameMap.put(rhsColNames.get(i), columnNames.get(i));
    }

    final String indent = getBodoIndent();
    List<Op> outputOperations = new ArrayList<>();

    if (isAll) {
      // For IntersectAll, we use groupby to compute a unique cumulative count per
      // row,
      // then we perform an inner join on all the columns as well as the cumcount.
      // Rows with x copies in lhsExpr and y copies in rhsExpr will have min(x, y)
      // copies in outVar
      //
      // Simply avoiding drop_duplicates is not enough: pd.merge(x * [1], y * [1])
      // gives (x * y) * [1], rather than min(x, y) * [1] which is correct.
      //
      // Example Codegen:
      // outVar = lhsDfCnt.rename(columns={...lhsRenameMap...}, copy=False)
      // .merge(rhsDfCnt.rename(columns={...rhsRenameMap...}, copy=False),
      // on=[...columnNames..., "__bodo_dummy__"])
      // .drop(columns="__bodo_dummy__"])
      // where lhsDfCnt and rhsDfCnt are generated by generateCumcountDf().

      // Generate temp vars
      final Variable lhsDfCnt = bodoVisitorClass.genDfVar();
      final Variable rhsDfCnt = bodoVisitorClass.genDfVar();

      Op lhsCumcountCode = generateCumcountDf(lhsDfCnt, lhsExpr, lhsColNames, bodoVisitorClass);
      Op rhsCumcountCode = generateCumcountDf(rhsDfCnt, rhsExpr, rhsColNames, bodoVisitorClass);
      outputOperations.add(lhsCumcountCode);
      outputOperations.add(rhsCumcountCode);

      // Generate list of output column names
      StringBuilder dummyColumnNamesListString = new StringBuilder("[");
      for (int i = 0; i < columnNames.size(); i++) {
        dummyColumnNamesListString.append(makeQuoted(columnNames.get(i))).append(", ");
      }
      dummyColumnNamesListString.append(makeQuoted(getDummyColNameBase())).append(",]");
      String dummyColumnNamesList = dummyColumnNamesListString.toString();

      outputOperations.add(
          new Op.Code(
              new StringBuilder()
                  .append(indent)
                  .append(outVar.emit())
                  .append(" = ")
                  .append(lhsDfCnt.emit())
                  .append(".rename(columns=")
                  .append(renameColumns(lhsRenameMap))
                  .append(", copy=False).merge(")
                  .append(rhsDfCnt.emit())
                  .append(".rename(columns=")
                  .append(renameColumns(rhsRenameMap))
                  .append(", copy=False), on=")
                  .append(dummyColumnNamesList)
                  .append(").drop(columns=[")
                  .append(makeQuoted(getDummyColNameBase()))
                  .append("])\n")
                  .toString()));

    } else {
      // For Intersect, we drop duplicates in lhsExpr and rhsExpr, then perform an
      // inner join
      // on all the columns. Need to perform a final drop to account for values in
      // both tables.
      //
      // Example Codegen:
      // df_out = <LHS>.merge(<RHS>, on=[...columnNames...]).drop_duplicates()
      // where <LHS> = df_lhs.drop_duplicates().rename(columns={...lhsRenameMap...},
      // copy=False)
      // and <RHS> = df_rhs.drop_duplicates().rename(columns={...rhsRenameMap...},
      // copy=False)

      // Generate list of output column names
      StringBuilder columnNamesListString = new StringBuilder("[");
      for (int i = 0; i < columnNames.size(); i++) {
        columnNamesListString.append(makeQuoted(columnNames.get(i))).append(", ");
      }
      columnNamesListString.append("]");
      String columnNamesList = columnNamesListString.toString();

      outputOperations.add(
          new Op.Code(
              new StringBuilder()
                  .append(indent)
                  .append(outVar.emit())
                  .append(" = ")
                  .append(lhsExpr.emit())
                  .append(".drop_duplicates(ignore_index=True).rename(columns=")
                  .append(renameColumns(lhsRenameMap))
                  .append(", copy=False).merge(")
                  .append(rhsExpr.emit())
                  .append(".drop_duplicates(ignore_index=True).rename(columns=")
                  .append(renameColumns(rhsRenameMap))
                  .append(", copy=False), on=")
                  .append(columnNamesList)
                  .append(").drop_duplicates(ignore_index=True)\n")
                  .toString()));
    }

    return outputOperations;
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
   * @param bodoVisitorClass The calling Bodo visitor class, used to generate temp var names
   * @return The code generated for the Except expression.
   */
  public static List<Op> generateExceptCode(
      Variable outVar,
      Variable lhsExpr,
      List<String> lhsColNames,
      Variable rhsExpr,
      List<String> rhsColNames,
      List<String> columnNames,
      boolean isAll,
      BodoCodeGenVisitor bodoVisitorClass) {
    assert lhsColNames.size() == rhsColNames.size() && lhsColNames.size() == columnNames.size();

    // Rename all lhs and rhs columns to match the expected output columns
    HashMap<String, String> lhsRenameMap = new HashMap<>();
    HashMap<String, String> rhsRenameMap = new HashMap<>();
    for (int i = 0; i < lhsColNames.size(); i++) {
      lhsRenameMap.put(lhsColNames.get(i), columnNames.get(i));
      rhsRenameMap.put(rhsColNames.get(i), columnNames.get(i));
    }

    final String indent = getBodoIndent();
    List<Op> outputOperations = new ArrayList<>();

    if (isAll) {
      // For ExceptAll, we use groupby to compute a unique cumulative count per row.
      // Then, we concatenate the dataframes [lhsExpr, rhsExpr, rhsExpr] and use
      // drop_duplicates
      // with `keep=False` to remove rows present in rhsExpr. We rely on the
      // cumulative count
      // column to avoid dropping rows that repeat in lhsExpr but aren't found in
      // rhsExpr.
      // Rows with x copies in lhsExpr and y copies in rhsExpr will have x - y copies
      // in outVar,
      // or zero copies if x - y is negative.
      //
      // Example Codegen:
      // rhsDfTmp = rhsDfCnt.rename(columns={...rhsRenameMap...}, copy=False)
      // outVar = pd.concat([
      // lhsDfCnt.rename(columns={...lhsRenameMap...}, copy=False),
      // rhsDfTmp,
      // rhsDfTmp
      // ]).drop_duplicates(keep=False).drop(columns=["__bodo_dummy__"])
      // where lhsDfCnt and rhsDfCnt are generated by generateCumcountDf().

      // Generate temp vars
      final Variable lhsDfCnt = bodoVisitorClass.genDfVar();
      final Variable rhsDfCnt = bodoVisitorClass.genDfVar();
      final Variable rhsDfTmp = bodoVisitorClass.genDfVar();

      Op lhsCumcountCode = generateCumcountDf(lhsDfCnt, lhsExpr, lhsColNames, bodoVisitorClass);
      Op rhsCumcountCode = generateCumcountDf(rhsDfCnt, rhsExpr, rhsColNames, bodoVisitorClass);
      outputOperations.add(lhsCumcountCode);
      outputOperations.add(rhsCumcountCode);

      outputOperations.add(
          new Op.Code(
              new StringBuilder()
                  .append(indent)
                  .append(rhsDfTmp.emit())
                  .append(" = ")
                  .append(rhsDfCnt.emit())
                  .append(".rename(columns=")
                  .append(renameColumns(rhsRenameMap))
                  .append(", copy=False)\n")
                  .append(indent)
                  .append(outVar.emit())
                  .append(" = pd.concat([")
                  .append(lhsDfCnt.emit())
                  .append(".rename(columns=")
                  .append(renameColumns(lhsRenameMap))
                  .append(", copy=False), ")
                  .append(rhsDfTmp.emit())
                  .append(", ")
                  .append(rhsDfTmp.emit())
                  .append("]).drop_duplicates(keep=False, ignore_index=True).drop(columns=[")
                  .append(makeQuoted(getDummyColNameBase()))
                  .append("])\n")
                  .toString()));

    } else {
      // For Except, we concatenate the dataframes [lhsExpr, rhsExpr, rhsExpr] and use
      // drop_duplicates with `keep=False` to remove rows present in rhsExpr.
      //
      // Example Codegen:
      // rhsDfTmp = rhsExpr.rename(columns={...rhsRenameMap...},
      // copy=False).drop_duplicates()
      // outVar = pd.concat([
      // lhsExpr.rename(columns={...lhsRenameMap...}, copy=False).drop_duplicates(),
      // rhsDfTmp,
      // rhsDfTmp
      // ]).drop_duplicates(keep=False)

      // Generate temp vars
      final Variable rhsDfTmp = bodoVisitorClass.genDfVar();

      outputOperations.add(
          new Op.Code(
              new StringBuilder()
                  .append(indent)
                  .append(rhsDfTmp.emit())
                  .append(" = ")
                  .append(rhsExpr.emit())
                  .append(".rename(columns=")
                  .append(renameColumns(rhsRenameMap))
                  .append(", copy=False).drop_duplicates(ignore_index=True)\n")
                  .append(indent)
                  .append(outVar.emit())
                  .append(" = pd.concat([")
                  .append(lhsExpr.emit())
                  .append(".rename(columns=")
                  .append(renameColumns(lhsRenameMap))
                  .append(", copy=False).drop_duplicates(ignore_index=True), ")
                  .append(rhsDfTmp.emit())
                  .append(", ")
                  .append(rhsDfTmp.emit())
                  .append("]).drop_duplicates(keep=False, ignore_index=True)\n")
                  .toString()));
    }

    return outputOperations;
  }

  /**
   * Helper function that inserts a new column of cumulative counts into an existing dataframe. Used
   * in INTERSECT ALL and EXCEPT ALL codegen. Cumulative counts are stored in a new column named
   * getDummyColNameBase().
   *
   * @param outVar The output variable.
   * @param inputDfVar The expression of the input dataframe
   * @param colNames The names of columns in the input dataframe
   * @param bodoVisitorClass Bodo Visitor used to create temps and global variables
   */
  private static Op generateCumcountDf(
      Variable outVar,
      Variable inputDfVar,
      List<String> colNames,
      BodoCodeGenVisitor bodoVisitorClass) {
    StringBuilder cumcountBuilder = new StringBuilder();
    final String indent = getBodoIndent();

    // Since cumcount currently lacks JIT support, we use cumsum on a column of
    // one's instead.
    // We use init_dataframe to efficiently add columns in place to an existing
    // table.
    // Since BodoSQL never uses Index values, we replace the index with a dummy
    // RangeIndex:
    // this avoids MultiIndex issues and allows Bodo to optimize more.
    // We also replace the Groupby index to inform the compiler that len(inputDfVar)
    // ==
    // len(groupby).
    //
    // Example Codegen:
    // colOnes = np.ones((len(inputDfVar),), dtype=np.int64)
    // tableOnes = logical_table_to_table(get_dataframe_all_data(inputDfVar),
    // (colOnes,),
    // MetaType(...dummyColIdxs...), inputDfVar.shape[1])
    // dfOnes = init_dataframe((tableOnes,), init_range_index(0, len(inputDfVar), 1, None),
    // ColNamesMetaType(...colNames..., "__bodo_dummy__"))
    // colCnt = dfOnes.groupby([...colNames...],
    // dropna=False).cumsum()["__bodo_dummy__"]
    // tableCnt = logical_table_to_table(get_dataframe_all_data(inputDfVar),
    // (colCnt,),
    // MetaType(...dummyColIdxs...), inputDfVar.shape[1])
    // dfCnt = init_dataframe((tableCnt,), init_range_index(0, len(inputDfVar), 1, None),
    // ColNamesMetaType(...colNames..., "__bodo_dummy__"))

    // Generate temp vars
    final Variable colOnes = bodoVisitorClass.genSeriesVar();
    final Variable tableOnes = bodoVisitorClass.genTableVar();
    final Variable dfOnes = bodoVisitorClass.genDfVar();
    final Variable colCnt = bodoVisitorClass.genSeriesVar();
    final Variable tableCnt = bodoVisitorClass.genTableVar();
    final Variable dfCnt = outVar;

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

    Variable dummyColIdxsGlobal = bodoVisitorClass.lowerAsMetaType(dummyColIdxsTuple);
    Variable dummyColNamesGlobal = bodoVisitorClass.lowerAsColNamesMetaType(dummyColNamesTuple);

    // TODO: Refactor to use Exprs
    // Compute dfOnes
    cumcountBuilder
        .append(indent)
        .append(colOnes.emit())
        .append(" = np.ones((len(")
        .append(inputDfVar.emit())
        .append("),), dtype=np.int64)\n")
        .append(indent)
        .append(tableOnes.emit())
        .append(" = bodo.hiframes.table.logical_table_to_table(")
        .append("bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(")
        .append(inputDfVar.emit())
        .append("), (")
        .append(colOnes.emit())
        .append(",), ")
        .append(dummyColIdxsGlobal.getName())
        .append(", ")
        .append(inputDfVar.emit())
        .append(".shape[1])\n")
        .append(indent)
        .append(dfOnes.emit())
        .append(" = bodo.hiframes.pd_dataframe_ext.init_dataframe((")
        .append(tableOnes.emit())
        .append(",), bodo.hiframes.pd_index_ext.init_range_index(0, len(")
        .append(inputDfVar.emit())
        .append("), 1, None), ")
        .append(dummyColNamesGlobal.getName())
        .append(")\n");

    // Compute dfCnt
    cumcountBuilder
        .append(indent)
        .append(colCnt.emit())
        .append(" = ")
        .append(dfOnes.emit())
        .append(".groupby(")
        .append(colNamesList.emit())
        .append(", dropna=False).cumsum()[")
        .append(makeQuoted(getDummyColNameBase()))
        .append("]\n")
        .append(indent)
        .append(tableCnt.emit())
        .append(" = bodo.hiframes.table.logical_table_to_table(")
        .append("bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(")
        .append(inputDfVar.emit())
        .append("), (")
        .append(colCnt.emit())
        .append(",), ")
        .append(dummyColIdxsGlobal.getName())
        .append(", ")
        .append(inputDfVar.emit())
        .append(".shape[1])\n")
        .append(indent)
        .append(dfCnt.emit())
        .append(" = bodo.hiframes.pd_dataframe_ext.init_dataframe((")
        .append(tableCnt.emit())
        .append(",), bodo.hiframes.pd_index_ext.init_range_index(0, len(")
        .append(inputDfVar.emit())
        .append("), 1, None), ")
        .append(dummyColNamesGlobal.getName())
        .append(")\n");

    return new Op.Code(cumcountBuilder.toString());
  }
}
