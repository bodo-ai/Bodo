package com.bodosql.calcite.application.utils;

import static com.bodosql.calcite.application.utils.IsScalar.isScalar;

import com.bodosql.calcite.adapter.pandas.ArrayRexToPandasTranslator;
import com.bodosql.calcite.adapter.pandas.PandasRel;
import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.BodoEngineTable;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.IntegerLiteral;
import com.bodosql.calcite.ir.Expr.StringLiteral;
import com.bodosql.calcite.ir.Module;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.SnowflakeCatalogTable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexOver;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.ArraySqlType;
import org.apache.calcite.sql.type.MapSqlType;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.VariantSqlType;

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
    if (unquotedString.length() > 1
        && unquotedString.charAt(0) == '"'
        && unquotedString.charAt(unquotedString.length() - 1) == '"') {
      return unquotedString;
    }
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

  public static void expectScalarArgument(RexNode argNode, String fnName, String argName) {
    if (!isScalar(argNode)) {
      throw new BodoSQLCodegenException(
          "Error: argument '" + argName + "' to function " + fnName + " must be a scalar.");
    }
  }

  /**
   * Function to convert a SQL type to a matching Pandas type.
   *
   * @param typeName SQL Type.
   * @param outputScalar Should the output generate a type for converting scalars.
   * @param precision The type precision for types that require it.
   * @return The pandas type
   */
  public static String sqlTypenameToPandasTypename(
      SqlTypeName typeName, boolean outputScalar, int precision) {
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
        if (outputScalar) {
          dtype = "bodosql.libs.generated_lib.sql_null_checking_scalar_conv_pd_to_date";
        } else {
          dtype = "bodo.datetime_date_type";
        }
        break;
      case TIMESTAMP:
        if (outputScalar) {
          dtype = "bodosql.libs.generated_lib.sql_null_checking_scalar_conv_pd_to_datetime";
        } else {
          dtype = "np.dtype(\"datetime64[ns]\")";
        }
        break;
      case TIMESTAMP_TZ:
        if (outputScalar) {
          dtype = "bodo.TimestampTZType";
        } else {
          dtype = "bodo.timestamptz_array_type";
        }
        break;
      case TIME:
        if (outputScalar) {
          dtype = String.format(Locale.ROOT, "bodo.TimeType(%d)", precision);
        } else {
          dtype = String.format(Locale.ROOT, "TimeArrayType(%d)", precision);
        }
        break;
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
   * Check if input type is VARIANT/MAP or has a VARIANT/MAP component (which means concrete type
   * not fully known). NOTE: MAP could be a struct array or map array in Bodo compiler and is not
   * fully concrete.
   *
   * @param type input type to check
   * @return true flag if there is VARIANT/MAP
   */
  public static boolean hasVariantType(RelDataType type) {
    if (type instanceof VariantSqlType) {
      return true;
    }
    if (type instanceof MapSqlType) {
      MapSqlType mapType = (MapSqlType) type;
      return hasVariantType(mapType.getKeyType()) || hasVariantType(mapType.getValueType());
    }
    if (type instanceof ArraySqlType) {
      ArraySqlType arrayType = (ArraySqlType) type;
      return hasVariantType(arrayType.getComponentType());
    }
    return false;
  }

  /**
   * Calcite optimizes a large number of windowed aggregation functions into case statements, which
   * check if the window size is valid. This checks if the supplied node is one of those case
   * statements.
   *
   * <p>The rough location in which this occurs within calcite is here:
   * https://github.com/apache/calcite/blob/master/core/src/main/java/org/apache/calcite/sql2rel/SqlToRelConverter.java#L2081
   * I am still trying to find the exact location where this translation into case statements
   * occurs.
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

  public static void assertWithErrMsg(boolean test, String msg) {
    if (!test) {
      throw new RuntimeException(msg);
    }
  }

  public static boolean isSnowflakeCatalogTable(BodoSqlTable table) {
    return table instanceof SnowflakeCatalogTable;
  }

  /**
   * Convert a list of strings to a list of string literals.
   *
   * @param arg The list of strings
   * @return A list of string literals
   */
  public static List<StringLiteral> stringsToStringLiterals(List<String> arg) {
    List<StringLiteral> output = new ArrayList<>(arg.size());
    for (int i = 0; i < arg.size(); i++) {
      output.add(new StringLiteral(arg.get(i)));
    }
    return output;
  }

  /**
   * Given a non-negative number stop, create a list of integer literals from [0, stop).
   *
   * @param stop The end of range.
   * @return A list of integer literals from [0, stop)
   */
  public static List<IntegerLiteral> integerLiteralArange(int stop) {
    List<IntegerLiteral> output = new ArrayList<>(stop);
    for (int i = 0; i < stop; i++) {
      output.add(new IntegerLiteral(i));
    }
    return output;
  }

  public static List<AggregateCall> literalAggPrunedAggList(List<AggregateCall> aggregateCallList) {
    return aggregateCallList.stream()
        .filter(x -> x.getAggregation().getKind() != SqlKind.LITERAL_AGG)
        .collect(Collectors.toList());
  }

  /**
   * Given a table that contains the outputs of an aggregate, this inserts any LITERAL_AGG values,
   * which are just literals added to the original locations.
   *
   * @param builder The builder used for generating intermediate code.
   * @param ctx The ctx used to generate code for literals.
   * @param inputTable The table with the aggregate output.
   * @param node The aggregation node.
   * @return A BodoEngineTable concatenating the literals into the table.
   */
  public static BodoEngineTable concatenateLiteralAggValue(
      final Module.Builder builder,
      PandasRel.BuildContext ctx,
      final BodoEngineTable inputTable,
      final Aggregate node) {
    final List<IntegerLiteral> indices = new ArrayList<>();
    final List<Expr> literalArrays = new ArrayList<>();
    final int numGroupCols = node.getGroupCount();
    for (int i = 0; i < numGroupCols; i++) {
      indices.add(new IntegerLiteral(i));
    }
    // Generate the literal arrays
    final List<AggregateCall> aggregateCallList = node.getAggCallList();
    final int numAggCols =
        node.getRowType().getFieldCount()
            - (aggregateCallList.size() - literalAggPrunedAggList(aggregateCallList).size());
    ArrayRexToPandasTranslator translator = ctx.arrayRexTranslator(inputTable);
    int seenLiterals = 0;
    int keptColumns = numGroupCols;
    for (int i = 0; i < aggregateCallList.size(); i++) {
      final AggregateCall call = aggregateCallList.get(i);
      final IntegerLiteral outputIndex;
      if (call.getAggregation().getKind() == SqlKind.LITERAL_AGG) {
        // Generate the code for the literal array.
        if (call.rexList.size() != 1) {
          throw new RuntimeException(
              "Internal Error: LITERAL_AGG encountered with more than 1 literal.");
        }
        literalArrays.add(translator.apply(call.rexList.get(0)));
        outputIndex = new IntegerLiteral(numAggCols + seenLiterals);
        seenLiterals += 1;
      } else {
        outputIndex = new IntegerLiteral(keptColumns);
        keptColumns += 1;
      }
      indices.add(outputIndex);
    }
    // Concatenate the arrays to the table.
    Variable outVar = builder.getSymbolTable().genTableVar();
    Variable buildIndices = ctx.lowerAsMetaType(new Expr.Tuple(indices));
    Expr tableExpr =
        new Expr.Call(
            "bodo.hiframes.table.logical_table_to_table",
            inputTable,
            new Expr.Tuple(literalArrays),
            buildIndices,
            new Expr.IntegerLiteral(numAggCols));
    builder.add(new Op.Assign(outVar, tableExpr));
    return new BodoEngineTable(outVar.emit(), node);
  }

  /**
   * Determine the conversion function name for a cast to a given type.
   *
   * @param outputType The output data type.
   * @param isSafe Is this a TRY_CAST instead of a regular cast?
   * @return String function name (in all uppercase)
   */
  public static String getConversionName(RelDataType outputType, boolean isSafe) {
    if (outputType instanceof VariantSqlType) {
      return "TO_VARIANT";
    } else {
      switch (outputType.getSqlTypeName()) {
        case CHAR:
        case VARCHAR:
          return "TO_VARCHAR";
        case BINARY:
        case VARBINARY:
          if (isSafe) {
            return "TRY_TO_BINARY";
          } else {
            return "TO_BINARY";
          }
        case BOOLEAN:
          if (isSafe) {
            return "TRY_TO_BOOLEAN";
          } else {
            return "TO_BOOLEAN";
          }
        case TINYINT:
        case SMALLINT:
        case INTEGER:
        case BIGINT:
        case DECIMAL:
          if (isSafe) {
            return "TRY_TO_NUMBER";
          } else {
            return "TO_NUMBER";
          }
        case FLOAT:
        case DOUBLE:
          if (isSafe) {
            return "TRY_TO_DOUBLE";
          } else {
            return "TO_DOUBLE";
          }
        case TIMESTAMP:
          if (isSafe) {
            return "TRY_TO_TIMESTAMP_NTZ";
          } else {
            return "TO_TIMESTAMP_NTZ";
          }
        case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
          if (isSafe) {
            return "TRY_TO_TIMESTAMP_LTZ";
          } else {
            return "TO_TIMESTAMP_LTZ";
          }
        case TIMESTAMP_TZ:
          if (isSafe) {
            throw new BodoSQLCodegenException("TRY_TO_TIMESTAMP_TZ not currently supported");
          } else {
            return "TO_TIMESTAMPTZ";
          }
        case DATE:
          if (isSafe) {
            return "TRY_TO_DATE";
          } else {
            return "TO_DATE";
          }
        case TIME:
          if (isSafe) {
            return "TRY_TO_TIME";
          } else {
            return "TO_TIME";
          }
        case ARRAY:
          return "TO_ARRAY";
        case MAP:
          return "TO_OBJECT";
        default:
          throw new RuntimeException(
              String.format(Locale.ROOT, "Unsupported cast to type %s", outputType));
      }
    }
  }
}
