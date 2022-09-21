package com.bodosql.calcite.application;

import java.util.HashMap;
import java.util.Set;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.logical.LogicalValues;
import org.apache.calcite.rex.*;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.fun.*;

/** Class of static methods used to determine the ExprType of each node. */
public class ExprTypeVisitor {

  // Set of functions that use the arg0 exprType and SQLKind
  private static Set<SqlKind> arg0KindFunctions = Set.of(SqlKind.CEIL, SqlKind.FLOOR, SqlKind.MOD);
  // Set of function that use the arg0 exprType and fnName
  private static Set<String> arg0NamedFunctions =
      Set.of(
          "ABS",
          "ACOS",
          "ACOSH",
          "ASCII",
          "ASIN",
          "ASINH",
          "ATAN",
          "ATANH",
          "CBRT",
          "CHAR",
          "CHAR_LENGTH",
          "CHARACTER_LENGTH",
          "CHR",
          "CONV",
          "COS",
          "COSH",
          "COT",
          "DATE_FORMAT",
          "DAYNAME",
          "DAYOFMONTH",
          "DAYOFWEEK",
          "DAYOFYEAR",
          "DEGREES",
          "EXP",
          "FACTORIAL",
          "TO_TIME",
          "TIME",
          "FROM_DAYS",
          "FROM_UNIXTIME",
          "HOUR",
          "LAST_DAY",
          "LCASE",
          "LEN",
          "LENGTH",
          "LN",
          "LOG10",
          "LOG2",
          "LOWER",
          "MICROSECOND",
          "MINUTE",
          "MONTH",
          "MONTHNAME",
          "NEXT_DAY",
          "NULLIF",
          "ORD",
          "PREVIOUS_DAY",
          "QUARTER",
          "RADIANS",
          "REVERSE",
          "SECOND",
          "SIGN",
          "SIN",
          "SINH",
          "SPACE",
          "SQUARE",
          "STR_TO_DATE",
          "TAN",
          "TANH",
          "TO_DATE",
          "TO_DAYS",
          "TO_SECONDS",
          "UCASE",
          "UPPER",
          "WEEK",
          "WEEKDAY",
          "WEEKISO",
          "WEEKOFYEAR",
          "YEAR");
  // Set of functions that meet their exprType
  private static Set<String> meetFunctions =
      Set.of(
          "ADDDATE",
          "ATAN2",
          "BITAND",
          "BITNOT",
          "BITOR",
          "BITSHIFTLEFT",
          "BITSHIFTRIGHT",
          "BITXOR",
          "BOOLAND",
          "BOOLNOT",
          "BOOLOR",
          "BOOLXOR",
          "COALESCE",
          "CONCAT",
          "CONCAT_WS",
          "CONTAINS",
          "DATE_ADD",
          "DATE_SUB",
          "DATEADD",
          "DATEDIFF",
          "DECODE",
          "DIV0",
          "EDITDISTANCE",
          "EQUAL_NULL",
          "FORMAT",
          "GETBIT",
          "HAVERSINE",
          "IF",
          "IFF",
          "IFNULL",
          "INITCAP",
          "INSTR",
          "LEFT",
          "LOG",
          "LPAD",
          "LTRIM",
          "MAKEDATE",
          "MID",
          "NULLIF",
          "NULLIFZERO",
          "NVL",
          "NVL2",
          "POW",
          "POWER",
          "REGEXP",
          "REGEXP_COUNT",
          "REGEXP_INSTR",
          "REGEXP_LIKE",
          "REGEXP_REPLACE",
          "REGEXP_SUBSTR",
          "REGR_VALX",
          "REGR_VALY",
          "REPEAT",
          "REPLACE",
          "RIGHT",
          "RLIKE",
          "ROUND",
          "RPAD",
          "RTRIM",
          "SPLIT_PART",
          "STRCMP",
          "STRTOK",
          "SUBDATE",
          "SUBSTR",
          "SUBSTRING_INDEX",
          "TRANSLATE3",
          "TRIM",
          "TRUNC",
          "TRUNCATE",
          "WEEKDAY",
          "WIDTH_BUCKET",
          "YEAROFWEEKISO",
          "YEARWEEK",
          "ZEROIFNULL");

  /**
   * Generates a unique key for RelNodes. This done with just RelNode.id. The idea
   *
   * @param node The RelNode that needs a key generated.
   * @return The string key for the Relnode.
   */
  public static String generateRelNodeKey(RelNode node) {
    return String.valueOf(node.getId());
  }

  /**
   * Generates a unique key for RexNode. This done with a RelNode.id for identify unique tables and
   * then the NAME + TYPE of the RexNode.
   *
   * @param node The RexNode that needs a key generated.
   * @param id The RelNode id used to uniquely identify the table.
   * @return The string key for the RexNode. This is id::type::name.
   */
  public static String generateRexNodeKey(RexNode node, int id) {
    return String.format("%d::%s::%s", id, node.getType().toString(), node.toString());
  }

  /**
   * Determine the exprTypes for each RelNode in the plan. This is different from the code
   * generated, determining the type of the output, not the type of the operators used. (i.e.) A AND
   * B should always return column if either A or B is a column, but this doesn't determine if AND
   * should generate `&` or `and`, which depends instead on if the code is inside an apply.
   *
   * @param node The current RelNode that needs its exprType determined.
   * @param exprTypes The hashmap of existing String -> ExprType, where String is the key for the
   *     node.
   * @param searchMap Mapping from the key of search nodes to their expanded implementation. This is
   *     done to avoid expanding twice. TODO: Properly update the structure of the Plan to actually
   *     replace the Search Nodes without a hashmap.
   */
  public static void determineRelNodeExprType(
      RelNode node,
      HashMap<String, BodoSQLExprType.ExprType> exprTypes,
      HashMap<String, RexNode> searchMap) {
    /** Always start by visiting the exprType of each child. */
    // TODO: Determine if Join Left/Right is visited
    for (RelNode child : node.getInputs()) {
      determineRelNodeExprType(child, exprTypes, searchMap);
    }
    /**
     * Visiting RexNodes requires special handling because they aren't stored in Inputs. Generate
     * special code for each support RelNode type.
     */
    visitRexNodesRelNode(node, exprTypes, searchMap);
    // Mark each RelNode as DataFrame
    String key = generateRelNodeKey(node);
    exprTypes.put(key, BodoSQLExprType.ExprType.DATAFRAME);
  }

  /**
   * Visit all the RexNodes stored within a RelNode.
   *
   * @param node RelNode whose children need to be visited.
   * @param exprTypes The hashmap of existing String -> ExprType, where String is the key for the
   *     node.
   * @param searchMap Mapping from the key of search nodes to their expanded implementation. This is
   *     done to avoid expanding twice. TODO: Properly update the structure of the Plan to actually
   *     replace the Search Nodes without a hashmap.
   */
  public static void visitRexNodesRelNode(
      RelNode node,
      HashMap<String, BodoSQLExprType.ExprType> exprTypes,
      HashMap<String, RexNode> searchMap) {
    RexBuilder builder = node.getCluster().getRexBuilder();
    if (node instanceof Join) {
      determineRexNodeExprType(
          ((Join) node).getCondition(), exprTypes, node.getId(), searchMap, builder);
    } else if (node instanceof LogicalProject) {
      for (RexNode child : ((LogicalProject) node).getProjects()) {
        determineRexNodeExprType(child, exprTypes, node.getId(), searchMap, builder);
      }
    } else if (node instanceof LogicalFilter) {
      determineRexNodeExprType(
          ((LogicalFilter) node).getCondition(), exprTypes, node.getId(), searchMap, builder);
    } else if (node instanceof LogicalValues) {
      LogicalValues logicalValuesNode = (LogicalValues) node;
      if (logicalValuesNode.getTuples().size() == 0) {
        return;
      }
      // We only support the case where logicalValuesNode is has a single Tuple
      assert logicalValuesNode.getTuples().size() == 1;
      for (RexLiteral child : logicalValuesNode.getTuples().get(0)) {
        determineRexNodeExprType(child, exprTypes, node.getId(), searchMap, builder);
      }
    }
    // All other RelNodes have no RexNodes to visit.
  }

  /**
   * Determine the exprTypes for each RexNode in the plan. This is different from the code
   * generated, determining the type of the output, not the type of the operators used. (i.e.) A AND
   * B should always return column if either A or B is a column, but this doesn't determine if AND
   * should generate `&` or `and`, which depends instead on if the code is inside an apply.
   *
   * @param node The current RexNode that needs its exprType determined.
   * @param exprTypes The hashmap of existing String -> ExprType, where String is the key for the
   *     node.
   * @param id The unique id for the parent RelNode.
   * @param searchMap Mapping from the key of search nodes to their expanded implementation. This is
   *     done to avoid expanding twice. TODO: Properly update the structure of the Plan to actually
   *     replace the Search Nodes without a hashmap.
   * @param builder RexBuilder used to replace any unsupported internal nodes. This runs at this
   *     stage after optimizations are finished.
   */
  public static void determineRexNodeExprType(
      RexNode node,
      HashMap<String, BodoSQLExprType.ExprType> exprTypes,
      int id,
      HashMap<String, RexNode> searchMap,
      RexBuilder builder) {
    String key = generateRexNodeKey(node, id);
    if (node instanceof RexInputRef) {
      // InputRef is always a column
      exprTypes.put(key, BodoSQLExprType.ExprType.COLUMN);
    } else if (node instanceof RexLiteral || node instanceof RexNamedParam) {
      // Literal or NamedParam is always a scalar
      exprTypes.put(key, BodoSQLExprType.ExprType.SCALAR);
    } else if (node instanceof RexCall) {
      // RexCall types vary depending on the contents.
      determineRexCallExprType((RexCall) node, exprTypes, id, searchMap, builder);
    } else if (node instanceof RexFieldAccess || node instanceof RexCorrelVariable) {
      throw new BodoSQLExprTypeDeterminationException(
          "Internal Error: BodoSQL does not support corelated Queries");
    } else {
      throw new BodoSQLExprTypeDeterminationException(
          "Internal Error: Calcite Plan Produced an Unsupported RexNode");
    }
  }

  /**
   * Determine the exprTypes for each RexCall in the plan. This is different from the code
   * generated, determining the type of the output, not the type of the operators used. (i.e.) A AND
   * B should always return column if either A or B is a column, but this doesn't determine if AND
   * should generate `&` or `and`, which depends instead on if the code is inside an apply.
   *
   * @param node The current RexCall that needs its exprType determined.
   * @param exprTypes The hashmap of existing String -> ExprType, where String is the key for the
   *     node.
   * @param id The unique id for the parent RelNode.
   * @param searchMap Mapping from the key of search nodes to their expanded implementation. This is
   *     done to avoid expanding twice. TODO: Properly update the structure of the Plan to actually
   *     replace the Search Nodes without a hashmap.
   * @param builder RexBuilder used to replace any unsupported internal nodes. This runs at this
   *     stage after optimizations are finished.
   */
  public static void determineRexCallExprType(
      RexCall node,
      HashMap<String, BodoSQLExprType.ExprType> exprTypes,
      int id,
      HashMap<String, RexNode> searchMap,
      RexBuilder builder) {
    String key = generateRexNodeKey(node, id);
    // Visit all children
    for (RexNode operand : node.operands) {
      determineRexNodeExprType(operand, exprTypes, id, searchMap, builder);
    }
    if (node.getOperator() instanceof SqlBinaryOperator
        || node.getOperator() instanceof SqlCaseOperator
        || node.getOperator() instanceof SqlSubstringFunction
        || node.getOperator() instanceof SqlDatetimePlusOperator
        || node.getOperator() instanceof SqlDatetimeSubtractionOperator) {
      // Binary, Case, and substring operators compute the meet of the operands.
      BodoSQLExprType.ExprType exprType = BodoSQLExprType.ExprType.SCALAR;
      for (RexNode operand : node.operands) {
        String operandKey = generateRexNodeKey(operand, id);
        exprType = BodoSQLExprType.meet_elementwise_op(exprType, exprTypes.get(operandKey));
      }
      exprTypes.put(key, exprType);
    } else if (node.getOperator() instanceof SqlPostfixOperator
        || node.getOperator() instanceof SqlPrefixOperator
        || node.getOperator() instanceof SqlCastFunction
        || node.getOperator() instanceof SqlLikeOperator) {
      // Postfix, Prefix, Like and Cast operators are the same as operand 0
      RexNode child = node.operands.get(0);
      String childKey = generateRexNodeKey(child, id);
      exprTypes.put(key, exprTypes.get(childKey));
    } else if (node.getOperator() instanceof SqlInternalOperator) {
      if (node.getOperator().getKind() == SqlKind.SEARCH) {
        // TODO: Replace this code with something more with an actual
        // update to the plan.
        // Ideally we can use RexRules when they are available
        // https://issues.apache.org/jira/browse/CALCITE-4559
        RexNode newNode = RexUtil.expandSearch(builder, null, node);
        determineRexNodeExprType(newNode, exprTypes, id, searchMap, builder);
        String newNodeKey = generateRexNodeKey(newNode, id);
        exprTypes.put(key, exprTypes.get(newNodeKey));
        searchMap.put(key, newNode);
      }
    } else if (node.getOperator() instanceof SqlExtractFunction) {
      // Extract's type is the output of the operand 1
      RexNode child = node.operands.get(1);
      String childKey = generateRexNodeKey(child, id);
      exprTypes.put(key, exprTypes.get(childKey));
    } else if (node.getOperator() instanceof SqlTimestampDiffFunction) {
      // TimestampDiff computes the meet
      BodoSQLExprType.ExprType exprType = BodoSQLExprType.ExprType.SCALAR;
      for (RexNode operand : node.operands) {
        String operandKey = generateRexNodeKey(operand, id);
        exprType = BodoSQLExprType.meet_elementwise_op(exprType, exprTypes.get(operandKey));
      }
      exprTypes.put(key, exprType);
    } else if (node.getOperator() instanceof SqlFunction) {
      String fnName = node.getOperator().toString();
      if (node.getOperator() instanceof SqlAggFunction) {
        if (node instanceof RexOver) {
          // Windowed Aggregation functions return columns
          exprTypes.put(key, BodoSQLExprType.ExprType.COLUMN);
          RexOver castedNode = (RexOver) node;
          if (!(castedNode.getWindow().getLowerBound().isUnbounded()
              || castedNode.getWindow().getLowerBound().isCurrentRow())) {
            determineRexNodeExprType(
                castedNode.getWindow().getLowerBound().getOffset(),
                exprTypes,
                id,
                searchMap,
                builder);
          }
          if (!(castedNode.getWindow().getUpperBound().isUnbounded()
              || castedNode.getWindow().getUpperBound().isCurrentRow())) {

            determineRexNodeExprType(
                castedNode.getWindow().getUpperBound().getOffset(),
                exprTypes,
                id,
                searchMap,
                builder);
          }

        } else {
          // Aggregation functions return scalars
          exprTypes.put(key, BodoSQLExprType.ExprType.SCALAR);
        }
      } else if (fnName.equals("RAND")
          || fnName.equals("PI")
          || fnName.equals("CURRENT_TIMESTAMP")
          || fnName.equals("LOCALTIME")
          || fnName.equals("LOCALTIMESTAMP")
          || fnName.equals("NOW")
          || fnName.equals("UTC_TIMESTAMP")
          || fnName.equals("UTC_DATE")
          || fnName.equals("CURDATE")
          || fnName.equals("CURRENT_DATE")
          || fnName.equals("UNIX_TIMESTAMP")
          || fnName.equals("TIME_FROM_PARTS")) {
        // PI/Rand take no arguments and output scalar.
        // TODO: Fix Rand as it should output a column in some cases
        exprTypes.put(key, BodoSQLExprType.ExprType.SCALAR);
      } else if (fnName.equals("DATE_TRUNC")) {
        // DATE_TRUNC's type is the output of the operand 1
        RexNode child = node.operands.get(1);
        String childKey = generateRexNodeKey(child, id);
        exprTypes.put(key, exprTypes.get(childKey));
      } else if (arg0KindFunctions.contains(node.getOperator().getKind())
          || arg0NamedFunctions.contains(fnName)) {
        // Functions that use arg0
        RexNode child = node.operands.get(0);
        String childKey = generateRexNodeKey(child, id);
        exprTypes.put(key, exprTypes.get(childKey));
      } else if (meetFunctions.contains(fnName)) {
        // Functions that compute the meet of args
        BodoSQLExprType.ExprType exprType = BodoSQLExprType.ExprType.SCALAR;
        for (RexNode operand : node.operands) {
          String operandKey = generateRexNodeKey(operand, id);
          exprType = BodoSQLExprType.meet_elementwise_op(exprType, exprTypes.get(operandKey));
        }
        exprTypes.put(key, exprType);
      } else {

        throw new BodoSQLExprTypeDeterminationException(
            "Internal Error: Function: " + fnName + " not supported");
      }
    } else if (node.getOperator() instanceof SqlDatetimePlusOperator
        || node.getOperator() instanceof SqlDatetimeSubtractionOperator) {
      // Datetime operators also compute the meet of the operands.
      BodoSQLExprType.ExprType exprType = BodoSQLExprType.ExprType.SCALAR;
      for (RexNode operand : node.operands) {
        String operandKey = generateRexNodeKey(operand, id);
        exprType = BodoSQLExprType.meet_elementwise_op(exprType, exprTypes.get(operandKey));
      }
      exprTypes.put(key, exprType);
    } else if (node.getOperator() instanceof SqlNullTreatmentOperator) {
      // This call only occurs wrapping the following windowed aggregations:
      // FIRST_VALUE, LAST_VALUE, NTH_Value, LEAD and LAG
      // The expr type output of this call is the same as the expr type of the wrapped call.
      assert node.operands.size() == 1;
      assert node.operands.get(0) instanceof RexCall;
      SqlKind childKind = ((RexCall) node.operands.get(0)).getOperator().getKind();
      assert childKind == SqlKind.LEAD
          || childKind == SqlKind.LAG
          || childKind == SqlKind.NTH_VALUE
          || childKind == SqlKind.FIRST_VALUE
          || childKind == SqlKind.LAST_VALUE;

      String childOperandKey = generateRexNodeKey(node.operands.get(0), id);
      BodoSQLExprType.ExprType childExprType = exprTypes.get(childOperandKey);
      exprTypes.put(key, childExprType);
    } else {
      throw new BodoSQLExprTypeDeterminationException(
          "Internal Error: Calcite Plan Produced an Unsupported RexCall: " + node.getOperator());
    }
  }
}
