package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.ir.ExprKt.bodoSQLKernel;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.BodoEngineTable;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.Tuple;
import com.bodosql.calcite.ir.ExprKt;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import kotlin.Pair;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;

public class StringFnCodeGen {

  // Hashmap of functions for which there is a one to one mapping between the SQL function call,
  // and a function call where any of the arguments can be scalars or vectors.
  // IE SQLFN(C1, s1, C2, s2) => FN(C1, s1, C2, s2)
  // EX LPAD(A, 10, S) => bodosql.kernels.lpad(A, 10, S)
  static HashMap<String, String> equivalentFnMapBroadcast;

  static {
    equivalentFnMapBroadcast = new HashMap<>();
    equivalentFnMapBroadcast.put("CHAR_LENGTH", "length");
    equivalentFnMapBroadcast.put("LENGTH", "length");
    equivalentFnMapBroadcast.put("LOWER", "lower");
    equivalentFnMapBroadcast.put("UPPER", "upper");
    equivalentFnMapBroadcast.put("CONTAINS", "contains");
    equivalentFnMapBroadcast.put("LPAD", "lpad");
    equivalentFnMapBroadcast.put("RPAD", "rpad");
    equivalentFnMapBroadcast.put("LEFT", "left");
    equivalentFnMapBroadcast.put("RIGHT", "right");
    equivalentFnMapBroadcast.put("ASCII", "ord_ascii");
    equivalentFnMapBroadcast.put("CHAR", "char");
    equivalentFnMapBroadcast.put("FORMAT", "format");
    equivalentFnMapBroadcast.put("REPEAT", "repeat");
    equivalentFnMapBroadcast.put("REVERSE", "reverse");
    equivalentFnMapBroadcast.put("REPLACE", "replace");
    equivalentFnMapBroadcast.put("RTRIMMED_LENGTH", "rtrimmed_length");
    equivalentFnMapBroadcast.put("JAROWINKLER_SIMILARITY", "jarowinkler_similarity");
    equivalentFnMapBroadcast.put("SPACE", "space");
    equivalentFnMapBroadcast.put("STRCMP", "strcmp");
    equivalentFnMapBroadcast.put("INSTR", "instr");
    equivalentFnMapBroadcast.put("SUBSTRING_INDEX", "substring_index");
    equivalentFnMapBroadcast.put("TRANSLATE3", "translate");
    equivalentFnMapBroadcast.put("SPLIT_PART", "split_part");
    equivalentFnMapBroadcast.put("STARTSWITH", "startswith");
    equivalentFnMapBroadcast.put("ENDSWITH", "endswith");
    equivalentFnMapBroadcast.put("INSERT", "insert");
    equivalentFnMapBroadcast.put("SPLIT", "split");
  }

  /**
   * Helper function that handles codegen for most string functions with no special dictionary
   * encoding support such as CHAR and FORMAT.
   *
   * @param fnName The name of the function
   * @param args The arguments for the expression.
   * @return The Expr corresponding to the function call
   */
  public static Expr getStringFnCode(String fnName, List<Expr> args) {
    if (equivalentFnMapBroadcast.containsKey(fnName)) {
      return ExprKt.bodoSQLKernel(equivalentFnMapBroadcast.get(fnName), args, List.of());
    } else {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }
  }

  /**
   * Helper function that handles codegen for most string functions with special dictionary encoding
   * support.
   *
   * @param fnName The name of the function
   * @param args The arguments for the expression.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr getOptimizedStringFnCode(
      String fnName, List<Expr> args, List<Pair<String, Expr>> streamingNamedArgs) {
    if (equivalentFnMapBroadcast.containsKey(fnName)) {
      return ExprKt.bodoSQLKernel(equivalentFnMapBroadcast.get(fnName), args, streamingNamedArgs);
    } else {
      throw new BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported");
    }
  }

  /**
   * Function that returns the rexInfo for a LPAD/RPAD Function call NOTE: Snowflake allows 3rd
   * argument to be optional only if base (1st argument) is a string From Snowflake spec: When base
   * is a string, the default pad string default is ‘ ‘ (a single blank space). When base is a
   * binary value, the pad argument must be provided explicitly.
   *
   * @param fnOperation
   * @param operands The arguments to LPAD/RPAD function
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr generatePadCode(
      RexCall fnOperation, List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    String fnName = fnOperation.getOperator().getName();
    List<Expr> args = new ArrayList<>(operands);
    // 2 arguments is only allowed if arg0 is string
    if (args.size() == 2) {
      // HA: Let me know if there's a better way to do this check.
      // Throw an error if 1st argument is binary and no pad is provided.
      if (SqlTypeName.BINARY_TYPES.contains(
          fnOperation.getOperands().get(0).getType().getSqlTypeName())) {
        throw new BodoSQLCodegenException(
            fnName
                + ": When base is a binary value, the pad argument must be provided explicitly.");
      }
      // the default pad string default is ‘ ‘ (a single blank space)
      args.add(new Expr.StringLiteral(" "));
    }
    if (args.size() != 3) {
      throw new BodoSQLCodegenException(
          "Error, invalid number of arguments passed to "
              + fnName
              + ". Expected 2 or 3, received "
              + args.size()
              + ".\n");
    }
    return bodoSQLKernel(fnName.toLowerCase(), args, streamingNamedArgs);
  }

  /**
   * Function that returns the rexInfo for the Concat Function Call.
   *
   * @param operands The Exprs for all the arguments to the Concat Call
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr generated that matches the Concat expression.
   */
  public static Expr generateConcatCode(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs, RelDataType dataType) {
    Expr separatorInfo =
        SqlTypeFamily.CHARACTER.contains(dataType)
            ? new Expr.StringLiteral("")
            : new Expr.BinaryLiteral("");
    return generateConcatWSCode(separatorInfo, operands, streamingNamedArgs);
  }

  /**
   * Function that returns the rexInfo for the Concat_ws Function Call.
   *
   * @param separator the Expr for the string used for the separator
   * @param operandsInfo The Exprs for the list of string arguments to be concatenated
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr generated that matches the Concat_ws expression.
   */
  public static Expr generateConcatWSCode(
      Expr separator, List<Expr> operandsInfo, List<Pair<String, Expr>> streamingNamedArgs) {
    if (operandsInfo.size() == 1) {
      // If we only have a single argument don't emit any additional codegen - just pass along the
      // input arg.
      return operandsInfo.get(0);
    }
    Expr.Tuple tupleArg = new Tuple(operandsInfo);
    return ExprKt.bodoSQLKernel("concat_ws", List.of(tupleArg, separator), streamingNamedArgs);
  }

  /**
   * Function that returns the rexInfo for a INITCAP Function call
   *
   * @param operands the information about the 1-2 arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr generateInitcapInfo(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    List<Expr> args = new ArrayList<>(operands);

    if (args.size() == 1) {
      // If 1 arguments was provided, provide a default delimiter string.
      // We don't use a string literal because of escape characters.
      args.add(new Expr.Raw("' \\t\\n\\r\\f\\u000b!?@\\\"^#$&~_,.:;+-*%/|\\[](){}<>'"));
    }
    assert args.size() == 2;
    return bodoSQLKernel("initcap", args, streamingNamedArgs);
  }

  /**
   * Function that returns the rexInfo for an STRTOK Function Call.
   *
   * @param operands the information about the 1-3 arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr generateStrtok(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    List<Expr> args = new ArrayList<>(operands);
    if (args.size() == 1) {
      args.add(new Expr.StringLiteral(" "));
    }
    if (args.size() == 2) {
      args.add(new Expr.IntegerLiteral(1));
    }
    assert args.size() == 3;
    return ExprKt.bodoSQLKernel("strtok", args, streamingNamedArgs);
  }

  /**
   * Function that returns the rexInfo for an STRTOK_TO_ARRAY Function Call.
   *
   * @param operands the information about the 1-2 arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr generateStrtokToArray(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    List<Expr> args = new ArrayList<>(operands);
    if (args.size() == 1) {
      args.add(new Expr.StringLiteral(" "));
    }
    assert args.size() == 2;
    return ExprKt.bodoSQLKernel("strtok_to_array", args, streamingNamedArgs);
  }

  /**
   * Function that returns the rexInfo for an EDITDISTANCE Function Call.
   *
   * @param operands The function arguments.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr generateEditdistance(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    String fnName;
    if (operands.size() == 2) {
      fnName = "editdistance_no_max";
    } else {
      assert operands.size() == 3;
      fnName = "editdistance_with_max";
    }
    return ExprKt.bodoSQLKernel(fnName, operands, streamingNamedArgs);
  }

  /**
   * Function that returns the rexInfo for a POSITION/CHARINDEX Function Call.
   *
   * @param operands the information about the two/three arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr generatePosition(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {

    if (!(2 <= operands.size() && operands.size() <= 3)) {
      throw new BodoSQLCodegenException("Error, invalid number of arguments passed to POSITION");
    }
    List<Expr> args = new ArrayList<>();
    args.addAll(operands);
    if (operands.size() == 2) {
      args.add(new Expr.IntegerLiteral(1));
    }
    return bodoSQLKernel("position", args, streamingNamedArgs);
  }

  /**
   * Function that returns the rexInfo of a L/R/Trim function Call. Extended to handle trimming non
   * whitespace characters
   *
   * @param trimName The argument that determines from which sides we trim characters
   * @param stringToBeTrimmed The rexInfo of the string to be trimmed
   * @param charactersToBeTrimmed The characters to trimmed from the string
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The rexVisitorInfo for the trim call
   */
  public static Expr generateTrimFnCode(
      String trimName,
      Expr stringToBeTrimmed,
      Expr charactersToBeTrimmed,
      List<Pair<String, Expr>> streamingNamedArgs) {
    return bodoSQLKernel(
        trimName.toLowerCase(Locale.ROOT),
        List.of(stringToBeTrimmed, charactersToBeTrimmed),
        streamingNamedArgs);
  }
  /**
   * Function that returns the rexInfo for a SUBSTRING Function call
   *
   * @param operands The arguments to Substring
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The Expr corresponding to the function call
   */
  public static Expr generateSubstringCode(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    String fnName;
    int argCount = operands.size();
    if (argCount == 3) {
      fnName = "substring";
    } else if (argCount == 2) {
      fnName = "substring_suffix";
    } else {
      throw new BodoSQLCodegenException("Error, invalid number of arguments passed to SUBSTRING");
    }
    return bodoSQLKernel(fnName, operands, streamingNamedArgs);
  }

  /**
   * Generate python code for REPLACE
   *
   * @param operands Input arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Generated code
   */
  public static Expr generateReplace(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    if (operands.size() == 2) {
      return ExprKt.bodoSQLKernel(
          "replace",
          List.of(operands.get(0), operands.get(1), new Expr.Raw("\"\"")),
          streamingNamedArgs);
    } else if (operands.size() == 3) {
      return ExprKt.bodoSQLKernel("replace", operands, streamingNamedArgs);
    } else {
      throw new BodoSQLCodegenException("Invalid number of arguments passed to REPLACE.");
    }
  }

  /**
   * Generate python code for SHA2
   *
   * @param operands Input arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Generated code
   */
  public static Expr generateSHA2(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    if (operands.size() == 1) {
      return ExprKt.bodoSQLKernel(
          "sha2", List.of(operands.get(0), new Expr.IntegerLiteral(256)), streamingNamedArgs);
    } else if (operands.size() == 2) {
      return ExprKt.bodoSQLKernel("sha2", operands, streamingNamedArgs);
    } else {
      throw new BodoSQLCodegenException("Invalid number of arguments passed to SHA2.");
    }
  }

  /**
   * Generate python code for HEX_ENCODE
   *
   * @param operands Input arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Generated code
   */
  public static Expr generateHexEncode(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    ArrayList<Expr> args = new ArrayList<>(operands);
    if (args.size() == 1) {
      args.add(new Expr.IntegerLiteral(1));
    } else {
      // Extract the second argument as a literal instead of a call with np.int32
      String lineArgStr = operands.get(1).emit();
      if (!(lineArgStr.startsWith("np.int"))) {
        throw new BodoSQLCodegenException(
            "Invalid second argument to BASE64_ENCODE: " + (operands.get(1).emit()));
      }
      Expr numericArg = new Expr.Raw(lineArgStr.split("\\(|\\)")[1]);
      args.set(1, numericArg);
    }
    assert args.size() == 2;
    return ExprKt.bodoSQLKernel("hex_encode", args, streamingNamedArgs);
  }

  /**
   * Generate python code for one of the decoding functions for HEX_ENCODE
   *
   * @param fnName Which decoding function is being used
   * @param operands Input arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Generated code
   */
  public static Expr generateHexDecodeFn(
      String fnName, List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    ArrayList<Expr> args = new ArrayList<>(operands);
    assert args.size() == 1;
    boolean tryMode;
    String kernel;
    switch (fnName) {
      case "HEX_DECODE_STRING":
        {
          tryMode = false;
          kernel = "hex_decode_string";
          break;
        }
      case "TRY_HEX_DECODE_STRING":
        {
          tryMode = true;
          kernel = "hex_decode_string";
          break;
        }
      case "HEX_DECODE_BINARY":
        {
          tryMode = false;
          kernel = "hex_decode_binary";
          break;
        }
      case "TRY_HEX_DECODE_BINARY":
        {
          tryMode = true;
          kernel = "hex_decode_binary";
          break;
        }
      default:
        {
          throw new BodoSQLCodegenException(
              "Unsupported function for generateHexDecodeFn: " + fnName);
        }
    }
    args.add(new Expr.BooleanLiteral(tryMode));
    return ExprKt.bodoSQLKernel(kernel, args, streamingNamedArgs);
  }

  /**
   * Generate python code for BASE64_ENCODE
   *
   * @param operands Input arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Generated code
   */
  public static Expr generateBase64Encode(
      List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    ArrayList<Expr> args = new ArrayList<>(operands);
    if (args.size() == 1) {
      args.add(new Expr.IntegerLiteral(0));
    } else {
      // Extract the second argument as a literal instead of a call with np.int32
      String lineArgStr = operands.get(1).emit();
      if (!(lineArgStr.startsWith("np.int"))) {
        throw new BodoSQLCodegenException(
            "Invalid second argument to BASE64_ENCODE: " + (operands.get(1).emit()));
      }
      Expr numericArg = new Expr.Raw(lineArgStr.split("\\(|\\)")[1]);
      args.set(1, numericArg);
    }
    if (args.size() == 2) {
      args.add(new Expr.StringLiteral("+/="));
    }
    assert args.size() == 3;
    return ExprKt.bodoSQLKernel("base64_encode", args, streamingNamedArgs);
  }

  /**
   * Generate python code for one of the decoding functions for BASE64_ENCODE
   *
   * @param fnName Which decoding function is being used
   * @param operands Input arguments
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return Generated code
   */
  public static Expr generateBase64DecodeFn(
      String fnName, List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    ArrayList<Expr> args = new ArrayList<>(operands);
    if (args.size() == 1) {
      args.add(new Expr.StringLiteral("+/="));
    }
    boolean tryMode;
    String kernel;
    switch (fnName) {
      case "BASE64_DECODE_STRING":
        {
          tryMode = false;
          kernel = "base64_decode_string";
          break;
        }
      case "TRY_BASE64_DECODE_STRING":
        {
          tryMode = true;
          kernel = "base64_decode_string";
          break;
        }
      case "BASE64_DECODE_BINARY":
        {
          tryMode = false;
          kernel = "base64_decode_binary";
          break;
        }
      case "TRY_BASE64_DECODE_BINARY":
        {
          tryMode = true;
          kernel = "base64_decode_binary";
          break;
        }
      default:
        {
          throw new BodoSQLCodegenException(
              "Unsupported function for generateBase64DecodeFn: " + fnName);
        }
    }
    args.add(new Expr.BooleanLiteral(tryMode));
    assert args.size() == 3;
    return ExprKt.bodoSQLKernel(kernel, args, streamingNamedArgs);
  }

  public static Expr generateUUIDString(
      BodoEngineTable input,
      boolean isSingleRow,
      List<Expr> operands,
      List<Pair<String, Expr>> streamingNamedArgs) {
    if (!operands.isEmpty()) {
      return ExprKt.bodoSQLKernel("uuid5", operands, streamingNamedArgs);
    } else {
      Expr arg;
      if (isSingleRow) {
        arg = Expr.None.INSTANCE;
      } else {
        arg = input;
      }
      return ExprKt.bodoSQLKernel("uuid4", List.of(arg), List.of());
    }
  }
}
