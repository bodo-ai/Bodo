package com.bodosql.calcite.application.Utils;

import static com.bodosql.calcite.application.Utils.BodoArrayHelpers.sqlTypeToBodoArrayType;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.catalog.SnowflakeCatalogImpl;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.CatalogSchemaImpl;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.CatalogTableImpl;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexOver;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.*;

/** Class filled with static utility functions. */
public class Utils {

  // Name of Dummy Colnames for Bodo Intermediate operations
  private static final String dummyColNameBase = "__bodo_dummy__";

  // two space indent
  private static final String bodoIndent = "  ";

  /** Function used to return the standard indent used within BodoSql */
  public static String getBodoIndent() {
    return bodoIndent;
  }

  /** Function used to add multiple indents to a string buffer all at once */
  public static void addIndent(StringBuilder funcText, int numIndents) {
    for (int i = 0; i < numIndents; i++) {
      funcText = funcText.append(bodoIndent);
    }
  }

  /**
   * Function to return the baseDummyColumnName. This should be extended with a counter if an
   * operation requires multiple dummy columns. NOTE: We assume dummy columns do not persist between
   * operations.
   *
   * @return dummyColNameBase
   */
  public static String getDummyColNameBase() {
    return dummyColNameBase;
  }

  /**
   * Function to enclose string in quotes
   *
   * @param unquotedString string to be enclosed
   * @return single quoted string
   */
  public static String makeQuoted(String unquotedString) {
    return '"' + unquotedString + '"';
  }

  /**
   * Function to convert a Java Hashmap of names into a Python dictionary for use in a
   * DataFrame.rename(columns) calls.
   */
  public static String renameColumns(HashMap<String, String> colMap) {
    StringBuilder dictStr = new StringBuilder();
    dictStr.append("{");
    // Generate a sorted version of the map so the same code is always
    // generated on all nodes
    TreeMap<String, String> sortedMap = new TreeMap<>(colMap);
    for (String prv : sortedMap.keySet()) {
      dictStr.append(makeQuoted(prv));
      dictStr.append(": ");
      dictStr.append(makeQuoted(colMap.get(prv)));
      dictStr.append(", ");
    }
    dictStr.append("}");
    return dictStr.toString();
  }

  /**
   * Escapes " so Python interprets String correctly.
   *
   * @param inputStr String possibly containing "
   * @return String with quotes properly escaped.
   */
  public static String escapePythonQuotes(String inputStr) {
    return inputStr.replaceAll("(?<!\\\\)\"", "\\\\\"");
  }

  /**
   * Function to convert a SQL type to a matching Pandas type.
   *
   * @param typeName SQL Type.
   * @param outputScalar Should the output generate a type for converting scalars.
   * @return The pandas type
   */
  public static String sqlTypenameToPandasTypename(SqlTypeName typeName, boolean outputScalar) {
    String dtype;
    switch (typeName) {
      case BOOLEAN:
        if (outputScalar) {
          dtype = "bodosql.libs.generated_lib.sql_null_checking_scalar_conv_bool";
        } else {
          dtype = makeQuoted("boolean");
        }
        break;
      case TINYINT:
        if (outputScalar) {
          dtype = "bodosql.libs.generated_lib.sql_null_checking_scalar_conv_int8";
        } else {
          dtype = "pd.Int8Dtype()";
        }
        break;
      case SMALLINT:
        if (outputScalar) {
          dtype = "bodosql.libs.generated_lib.sql_null_checking_scalar_conv_int16";
        } else {
          dtype = "pd.Int16Dtype()";
        }
        break;
      case INTEGER:
        if (outputScalar) {
          dtype = "bodosql.libs.generated_lib.sql_null_checking_scalar_conv_int32";
        } else {
          dtype = "pd.Int32Dtype()";
        }
        break;
      case BIGINT:
        if (outputScalar) {
          dtype = "bodosql.libs.generated_lib.sql_null_checking_scalar_conv_int64";
        } else {
          dtype = "pd.Int64Dtype()";
        }
        break;
      case FLOAT:
        if (outputScalar) {
          dtype = "bodosql.libs.generated_lib.sql_null_checking_scalar_conv_float32";
        } else {
          dtype = "pd.Float32Dtype()";
        }
        break;
      case DOUBLE:
      case DECIMAL:
        if (outputScalar) {
          dtype = "bodosql.libs.generated_lib.sql_null_checking_scalar_conv_float64";
        } else {
          dtype = "pd.Float64Dtype()";
        }
        break;
      case DATE:
      case TIMESTAMP:
        if (outputScalar) {
          dtype = "bodosql.libs.generated_lib.sql_null_checking_scalar_conv_pd_to_datetime";
        } else {
          dtype = "np.dtype(\"datetime64[ns]\")";
        }
        break;
      case TIME:
        // TODO [BE-3649]: The precision needs to be handled here.
        throw new BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Unsupported TIME Type");
      case VARCHAR:
      case CHAR:
        if (outputScalar) {
          dtype = "bodosql.libs.generated_lib.sql_null_checking_scalar_conv_str";
        } else {
          dtype = "str";
        }
        break;
      case VARBINARY:
      case BINARY:
        if (outputScalar) {
          dtype = "bodosql.libs.generated_lib.sql_null_checking_scalar_conv_str";
        } else {
          // TODO: FIXME?
          dtype = "bodo.bytes_type";
        }
        break;
      case INTERVAL_DAY_HOUR:
      case INTERVAL_DAY_MINUTE:
      case INTERVAL_DAY_SECOND:
      case INTERVAL_HOUR_MINUTE:
      case INTERVAL_HOUR_SECOND:
      case INTERVAL_MINUTE_SECOND:
      case INTERVAL_HOUR:
      case INTERVAL_MINUTE:
      case INTERVAL_SECOND:
      case INTERVAL_DAY:
        if (outputScalar) {
          // pd.to_timedelta(None) returns None in standard python, but not in Bodo
          // This should likely be in the engine itself, to match pandas behavior
          // BE-2882
          dtype = "pd.to_timedelta";
        } else {
          dtype = "np.dtype(\"timedelta64[ns]\")";
        }
        break;
      case INTERVAL_YEAR:
      case INTERVAL_MONTH:
      case INTERVAL_YEAR_MONTH:
        // May later refactor this code to create DateOffsets, for now
        // causes an error
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Unsupported Type: " + typeName.getName());
    }
    return dtype;
  }

  /**
   * Calcite optimizes a large number of windowed aggregation functions into case statements, which
   * check if the window size is valid. This checks if the supplied node is one of those case
   * statements.
   *
   * <p>The rough location in which this occurs within calcite is here:
   * https://github.com/apache/calcite/blob/master/core/src/main/java/org/apache/calcite/sql2rel/SqlToRelConverter.java#L2081
   * I am still trying to find the exact location where this translation into case statments occurs.
   *
   * @param node the case node to check
   * @return true if it is a wrapped windowed aggregation function, and False if it is not
   */
  public static boolean isWindowedAggFn(RexCall node) {
    // First, we expect exactly three operands in the case statement
    if (node.getOperands().size() != 3) {
      return false;
    }
    return isEmptyWindowCheck(node) || windowLen1Check(node);
  }

  /**
   * Calcite optimizes a large number of windowed aggregation functions into case statements, which
   * check if the window size is valid. This checks if the rexcall is a windowed aggregation
   * function checking that the size of the window is 0.
   *
   * @param node the rexCall on which to perform the check
   * @return Boolean determining if a rexcall is in fact a windowed aggregation with an empty window
   *     check
   */
  public static boolean isEmptyWindowCheck(RexCall node) {
    // For arg0 (when case), we expect a comparison to the size of the window
    boolean arg0IsWindowSizeComparison =
        node.getOperands().get(0) instanceof RexCall
            && ((RexCall) node.getOperands().get(0)).getOperator().getKind() == SqlKind.GREATER_THAN
            && ((RexCall) node.getOperands().get(0)).getOperands().get(0) instanceof RexOver;
    // For arg1 (then case), we expect a windowed aggregation function
    boolean arg1IsWindowed = node.getOperands().get(1) instanceof RexOver;
    // For the else case, we expect NULL
    boolean arg2Null = node.getOperands().get(2) instanceof RexLiteral;

    return arg0IsWindowSizeComparison && arg1IsWindowed && arg2Null;
  }

  /**
   * Calcite optimizes a large number of windowed aggregation functions into case statements, which
   * check if the window size is valid. This checks if the input rexcall is a windowed aggregation
   * function checking that the size of the window is 1.
   *
   * @param node the rexCall on which to perform the check
   * @return Boolean determining if a rexcall is in fact a windowed aggregation with a window size 1
   *     check
   */
  public static boolean windowLen1Check(RexCall node) {
    // For arg0 (when case), we expect a comparison to the size of the window
    boolean arg0IsWindowSizeComparison =
        node.getOperands().get(0) instanceof RexCall
            && ((RexCall) node.getOperands().get(0)).getOperator().getKind() == SqlKind.EQUALS
            && ((RexCall) node.getOperands().get(0)).getOperands().get(0) instanceof RexOver;
    // For arg1 (then case), we expect NULL
    boolean arg1IsWindowed = node.getOperands().get(1) instanceof RexLiteral;
    // For the else case, we expect a windowed aggregation function
    //    boolean arg2Null = node.getOperands().get(2) instanceof RexOver;

    return arg0IsWindowSizeComparison && arg1IsWindowed;
  }

  /**
   * Helper function, takes the existing column names and a hashset of columns to add, and returns a
   * new dataframe, consisting of both the new and old columns. Generally used immediately before
   * generating code for CASE statements.
   *
   * @param inputVar The name of the input dataframe, to which we add the new columns.
   * @param colNames The list of the columns already present in inputVar, which need to be present
   *     in the output dataframe
   * @param colsToAddList The List of array variables that must be added to new dataframe.
   * @return
   */
  public static String generateCombinedDf(
      String inputVar, List<String> colNames, List<String> colsToAddList) {
    // TODO filter out the columns that don't need to be kept
    StringBuilder newDf = new StringBuilder("pd.DataFrame({");
    for (String curCol : colNames) {
      newDf
          .append(makeQuoted(curCol))
          .append(":")
          .append(inputVar)
          .append("[")
          .append(makeQuoted(curCol))
          .append("], ");
    }
    for (String preGeneratedCol : colsToAddList) {
      newDf.append(makeQuoted(preGeneratedCol)).append(":").append(preGeneratedCol).append(", ");
    }
    newDf.append("})");
    return newDf.toString();
  }

  /**
   * Checks if a string is a legal name for a Python identifier
   *
   * @param name the string name that needs to be checked
   * @return Boolean for if the name matches the regex [A-Za-z_]\w*
   */
  public static boolean isValidPythonIdentifier(String name) {
    final Pattern p = Pattern.compile("[a-zA-Z_]\\w*");
    Matcher m = p.matcher(name);
    return m.matches();
  }

  public static String getInputColumn(
      List<String> inputColumnNames, AggregateCall a, List<Integer> keyCols) {
    if (a.getArgList().isEmpty()) {
      // count(*) case
      // count(*) is turned into to count() by Calcite
      // in this case, we can use any column for aggregation, since inputColumnNames should
      // always contain at least one group by column, we use the first column for the
      // aggregation, and manually set the fieldname to *. However, count(*) includes
      // NULL values (whereas count does not).
      assert !inputColumnNames.isEmpty();
      if (keyCols.size() > 0) {
        // Use the key the list is not empty.
        return inputColumnNames.get(keyCols.get(0));
      }
      return inputColumnNames.get(0);
    } else {
      return inputColumnNames.get(a.getArgList().get(0));
    }
  }

  /***
   * Searches the input expression for table references to oldTableName, and replaces them to reference the new Table.
   * Only used inside CASE when there are window functions (so a new dataframe has to be created).
   * For example, table1_1[i] -> tmp_case_df2_1[i]
   *
   * @param expr The expression to replace table references
   * @param oldTableName The old table name, that the input expr uses for table references
   * @param newTableName The new table name, that the output expr will use for table references
   * @return
   */
  public static String renameTableRef(String expr, String oldTableName, String newTableName) {
    // check word boundary with \b to reduce chance of name conflicts with oldTableName
    return expr.replaceAll("\\b" + Pattern.quote(oldTableName + "_"), newTableName + "_");
  }

  /***
   *  Renames a list of codeExpressions that are input to CASE to use a new table reference.
   *  Used for handling window functions inside CASE where a new dataframe is created
   *  to include window function output as columns.
   *  For example, table1_1[i] -> tmp_case_df2_1[i]
   *
   *
   * @param codeExprs The list of expressions to replace table references
   * @param oldTableName The old table name, that the input expr uses for table references
   * @param newTableName The new table name, that the output expr will use for table references
   * @return
   */
  public static List<String> renameExprsList(
      List<String> codeExprs, String oldTableName, String newTableName) {
    List<String> outputExprs = new ArrayList<>();
    for (int i = 0; i < codeExprs.size(); i++) {
      outputExprs.add(renameTableRef(codeExprs.get(i), oldTableName, newTableName));
    }
    return outputExprs;
  }

  public static String generateDfApply(
      String inputVar,
      BodoCtx ctx,
      String lambdaFnStr,
      RelDataType outputType,
      PandasCodeGenVisitor pdVisitorClass) {
    // We assume, at this point, that the ctx.colsToAddList has been added to inputVar, and
    // the arguments have been renamed appropriately.

    // pass named parameters as kws to bodosql_case_placeholder()
    // sorting to make sure the same code is generated on each rank
    TreeSet<String> sortedParamSet = new TreeSet<>(ctx.getNamedParams());
    StringBuilder namedParamArgs = new StringBuilder();
    for (String param : sortedParamSet) {
      namedParamArgs.append(param + "=" + param + ", ");
    }

    // generate bodosql_case_placeholder() call with inputs:
    // 1) a tuple of necessary input arrays
    // 2) number of output rows (same as input rows, needed for allocation)
    // 3) initialization code for unpacking the input array tuple with the right array names
    // (MetaType global)
    // 4) body of the CASE loop (global constant)
    // 5) output array type
    // For example:
    // S5 = bodo.utils.typing.bodosql_case_placeholder(
    //   (bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df3, 0), ),
    //   len(df3),
    //   MetaType(('  df3_0 = arrs[0]',)),
    //   '((None if (pd.isna(bodo.utils.indexing.scalar_optional_getitem(df3_0)) ) else np.int64(1),
    //   IntegerArrayType(int64),
    // )
    StringBuilder inputDataStr = new StringBuilder();
    inputDataStr.append("(");
    StringBuilder initCode = new StringBuilder();
    initCode.append("(");

    int i = 0;
    TreeSet<Integer> sortedUsedColumns = new TreeSet<>(ctx.getUsedColumns());
    for (int colNo : sortedUsedColumns) {
      inputDataStr.append(
          String.format(
              "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(%s, %d), ", inputVar, colNo));
      initCode
          .append(makeQuoted(String.format("  %s = arrs[%d]", inputVar + "_" + colNo, i)))
          .append(", ");
      i++;
    }
    inputDataStr.append(")");
    initCode.append(")");
    String initGlobal = pdVisitorClass.lowerAsMetaType(initCode.toString());
    // have to use triple quotes here since lambdaFnStr can contain " or '
    String bodyGlobal = pdVisitorClass.lowerAsGlobal("\"\"\"" + lambdaFnStr + "\"\"\"");

    String outputArrayType = sqlTypeToBodoArrayType(outputType, false);
    String outputArrayTypeGlobal = pdVisitorClass.lowerAsGlobal(outputArrayType);

    return String.format(
        "bodo.utils.typing.bodosql_case_placeholder(%s, len(%s), %s, %s, %s, %s)",
        inputDataStr, inputVar, initGlobal, bodyGlobal, outputArrayTypeGlobal, namedParamArgs);
  }

  public static void assertWithErrMsg(boolean test, String msg) {
    if (!test) {
      throw new RuntimeException(msg);
    }
  }

  public static boolean isSnowflakeCatalogTable(BodoSqlTable table) {
    BodoSqlSchema schema = table.getSchema();
    if (table instanceof CatalogTableImpl && schema instanceof CatalogSchemaImpl) {
      CatalogSchemaImpl catalogSchema = (CatalogSchemaImpl) schema;
      return catalogSchema.getCatalog() instanceof SnowflakeCatalogImpl;
    }
    return false;
  }
}
