package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.utils.AggHelpers.getDummyColName;
import static com.bodosql.calcite.application.utils.JoinHelpers.preventColumnCollision;
import static com.bodosql.calcite.application.utils.Utils.makeQuoted;

import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.Variable;
import java.util.*;

public class JoinCodeGen {

  /* Counter used to generate dummy column names. */
  public static int dummyCounter = 0;

  /**
   * Function that returns the necessary generated code for a Join expression, and fills the input
   * list, allColNames, with the column names of the table resulting from the join. If the join
   * condition was on True, or on a single equality, no additional code will be needed. Otherwise,
   * an additional filter will need to table produced by this code.
   *
   * @param outVar The output variable.
   * @param joinType The type of the join being performed.
   * @param rightTable The name of the dataframe on the right hand side of the join
   * @param leftTable The name of the dataframe on the left hand side of the join
   * @param rightColNames The names of the columns in the right table
   * @param leftColNames The names of the columns in the left table
   * @param expectedOutColumns The expected names of the columns in the output table
   * @param joinCond Expression to be passed to Bodo for pd.merge.
   * @param mergeCols Set of column names that overlap for merging. Used for renamed columns.
   * @param tryRebalanceOutput Should we set a flag to try and rebalance the output if there is
   *     skew.
   * @return The code generated for the Join expression.
   */
  public static Op.Assign generateJoinCode(
      Variable outVar,
      String joinType,
      Variable rightTable,
      Variable leftTable,
      List<String> rightColNames,
      List<String> leftColNames,
      List<String> expectedOutColumns,
      Expr joinCond,
      HashSet<String> mergeCols,
      boolean tryRebalanceOutput) {

    List<String> allColNames = new ArrayList<>();

    HashMap<String, Boolean> seenColNames = new HashMap<String, Boolean>();
    preventColumnCollision(leftColNames, mergeCols, seenColNames);
    preventColumnCollision(rightColNames, mergeCols, seenColNames);
    String onStr = "";

    // Do we rename keys. Note: We could possibly avoid renaming keys if
    // we have an inner join, but that complicates the logic for renaming
    // columns that would require avoiding suffixes, so we skip that case.
    boolean hasDuplicateKeys = !mergeCols.isEmpty();
    // Do we rename anything
    boolean hasDuplicateNames = hasDuplicateKeys;

    // Determine and update the output column names
    for (String name : leftColNames) {
      if (seenColNames.containsKey(name) && seenColNames.get(name)) {
        // If we have a column that conflicts between tables, always
        // rename to avoid handling possible edge cases with suffixes
        // and multiple joins.
        allColNames.add(getDummyColName(dummyCounter) + name + "_left");
        // Add the column to merge columns so we generate a rename.
        // We only need to do this once since left/right have the same
        // name.
        mergeCols.add(name);
        hasDuplicateNames = true;
      } else if (hasDuplicateKeys && mergeCols.contains(name)) {
        allColNames.add(getDummyColName(dummyCounter) + name + "_left");
      } else {
        allColNames.add(name);
      }
    }
    for (String name : rightColNames) {
      if (seenColNames.containsKey(name) && seenColNames.get(name)) {
        // If we have a column that conflicts between tables, always
        // rename to avoid handling possible edge cases with suffixes
        // and multiple joins.
        allColNames.add(getDummyColName(dummyCounter) + name + "_right");
        // Add the column to merge columns so we generate a rename.
        // We only need to do this once since left/right have the same
        // name.
      } else if (hasDuplicateKeys && mergeCols.contains(name)) {
        allColNames.add(getDummyColName(dummyCounter) + name + "_right");
      } else {
        allColNames.add(name);
      }
    }

    boolean updateOnStr = false;
    if (!joinType.equals("cross")) {
      // Use Expr.StringLiteral in case there are nested quotes
      onStr = new Expr.StringLiteral(joinCond.emit()).emit();
      updateOnStr = true;
    }

    Expr leftTableExpr;
    Expr rightTableExpr;

    // If we have an outer join we need to create duplicate columns
    if (hasDuplicateNames) {
      // Give the key columns separate names to create duplicate columns
      // Need to use a TreeSet so we generate the same code on diff nodes.
      TreeSet<String> sortedMergeCols = new TreeSet<>();
      sortedMergeCols.addAll(mergeCols);
      StringBuilder leftRename = new StringBuilder();
      StringBuilder rightRename = new StringBuilder();
      leftRename.append(".rename(columns={");
      rightRename.append(".rename(columns={");
      for (String name : sortedMergeCols) {
        // Need to update both the condition expr and the actual column tables
        String leftName = getDummyColName(dummyCounter) + name + "_left";
        String rightName = getDummyColName(dummyCounter) + name + "_right";
        leftRename.append(makeQuoted(name)).append(": ").append(makeQuoted(leftName)).append(", ");
        rightRename
            .append(makeQuoted(name))
            .append(": ")
            .append(makeQuoted(rightName))
            .append(", ");
        if (updateOnStr) {
          String prevLeft = "left." + '`' + name + '`';
          String newLeft = "left." + '`' + leftName + '`';
          onStr = onStr.replace(prevLeft, newLeft);
          String prevRight = "right." + '`' + name + '`';
          String newRight = "right." + '`' + rightName + '`';
          onStr = onStr.replace(prevRight, newRight);
        }
      }
      leftRename.append("}, copy=False)");
      rightRename.append("}, copy=False)");
      leftTableExpr = new Expr.Raw(leftTable.emit() + leftRename.toString());
      rightTableExpr = new Expr.Raw(rightTable.emit() + rightRename.toString());
    } else {
      leftTableExpr = leftTable;
      rightTableExpr = rightTable;
    }

    if (joinType.equals("full")) {
      joinType = "outer";
    }
    onStr = onStr.equals("") ? "" : ", on=" + onStr;
    StringBuilder joinBuilder = new StringBuilder();
    joinBuilder
        .append(leftTableExpr.emit())
        .append(".merge(")
        .append(rightTableExpr.emit())
        .append(onStr)
        .append(", how=")
        .append(makeQuoted(joinType))
        .append(", _bodo_na_equal=False");
    if (tryRebalanceOutput) {
      joinBuilder.append(", _bodo_rebalance_output_if_skewed=True");
    }
    joinBuilder.append(")");

    // Determine if we need to do a rename to convert names
    // back.
    if (hasDuplicateNames) {
      joinBuilder.append(".rename(columns={");
      for (int i = 0; i < expectedOutColumns.size(); i++) {
        String expectedColumn = expectedOutColumns.get(i);
        String actualColumn = allColNames.get(i);
        if (!expectedColumn.equals(actualColumn)) {
          joinBuilder
              .append(makeQuoted(actualColumn))
              .append(": ")
              .append(makeQuoted(expectedColumn))
              .append(", ");
        }
      }
      joinBuilder.append("}, copy=False)");
    }

    Op.Assign outputAssign = new Op.Assign(outVar, new Expr.Raw(joinBuilder.toString()));

    // Increment the counter for future joins.
    dummyCounter += 1;
    return outputAssign;
  }
}
