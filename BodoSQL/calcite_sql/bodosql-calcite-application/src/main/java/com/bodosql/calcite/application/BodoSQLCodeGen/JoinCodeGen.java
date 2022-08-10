package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.AggHelpers.getDummyColName;
import static com.bodosql.calcite.application.Utils.JoinHelpers.preventColumnCollision;
import static com.bodosql.calcite.application.Utils.Utils.getBodoIndent;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import java.util.*;

public class JoinCodeGen {

  /* Counter used to generate dummy column names. */
  private static int dummyCounter = 0;
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
   * @param joinCond Expression to be passed to Bodo for pd.merge. If this expression is True we
   *     cannot support it directly and need to perform a cross join instead.
   * @return The code generated for the Join expression.
   */
  public static String generateJoinCode(
      String outVar,
      String joinType,
      String rightTable,
      String leftTable,
      List<String> rightColNames,
      List<String> leftColNames,
      List<String> expectedOutColumns,
      String joinCond,
      boolean hasEquals,
      HashSet<String> mergeCols) {

    List<String> allColNames = new ArrayList<>();
    StringBuilder generatedJoinCode = new StringBuilder();

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

    /* Do we generate a dummy column to do the merge. */
    boolean hasDummy = false;
    boolean updateOnStr = false;

    if (joinCond.equals("True") || !hasEquals) {
      /*
       * Temporary workaround to support cross join until it's supported in
       * the engine. Creates a temporary column and merges on that column.
       */
      hasDummy = true;
      generatedJoinCode.append(
          String.format("  left_arr = np.ones(len(%s), np.int8)\n", leftTable));
      generatedJoinCode.append(
          String.format("  right_arr = np.ones(len(%s), np.int8)\n", rightTable));
      /* Generate a unique column name to avoid possible collisions. */
      String dummyColumn = getDummyColName(dummyCounter) + "_key";
      leftTable =
          String.format(
              "pd.concat((%s, pd.DataFrame({\"%s\": left_arr})), axis=1)", leftTable, dummyColumn);
      rightTable =
          String.format(
              "pd.concat((%s, pd.DataFrame({\"%s\": right_arr})), axis=1)",
              rightTable, dummyColumn);
      onStr = makeQuoted(dummyColumn);
    } else {
      onStr = makeQuoted(joinCond);
      updateOnStr = true;
    }
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
      leftTable = leftTable + leftRename.toString();
      rightTable = rightTable + rightRename.toString();
    }

    if (joinType.equals("full")) joinType = "outer";
    StringBuilder joinBuilder = new StringBuilder();
    joinBuilder
        .append(leftTable)
        .append(".merge(")
        .append(rightTable)
        .append(", on=")
        .append(onStr)
        .append(", how=")
        .append(makeQuoted(joinType))
        .append(", _bodo_na_equal=False)");
    /* Drop the dummy column if it exists. */
    if (hasDummy) {
      joinBuilder.append(".loc[:, [");
      for (String name : allColNames) {
        joinBuilder.append(makeQuoted(name)).append(", ");
      }
      joinBuilder.append("]]");
    }
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

    generatedJoinCode
        .append(getBodoIndent())
        .append(outVar)
        .append(" = ")
        .append(joinBuilder.toString())
        .append('\n');
    // Increment the counter for future joins.
    dummyCounter += 1;
    return generatedJoinCode.toString();
  }
}
