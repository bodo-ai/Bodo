package com.bodosql.calcite.application;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.calcite.tools.ValidationException;

public class SqlValidationException extends SqlException {
  private static final long serialVersionUID = -429299379494895888L;

  public SqlValidationException(
      final String queryString, final ValidationException validationException) {
    super(description(queryString, validationException));
  }

  private static Pattern pattern =
      Pattern.compile("From line (\\d+), column (\\d+) to line (\\d+), column (\\d+): .*");

  private static String description(
      final String queryString, final ValidationException validationException) {
    final StringBuilder builder = new StringBuilder();
    final String message = validationException.getLocalizedMessage();

    builder.append("SqlValidationException\n\n");

    Matcher m = pattern.matcher(message);

    try {
      if (m.find()) {
        int startLineNum = Integer.parseInt(m.group(1));
        int startColNum = Integer.parseInt(m.group(2));
        int endLineNum = Integer.parseInt(m.group(3));
        int endColNum = Integer.parseInt(m.group(4));
        SqlException.pointInQueryString(
            builder,
            queryString,
            new SqlPosition(startLineNum, startColNum, endLineNum, endColNum));

        builder.append('\n');
      }
    } catch (Exception e) {

    }

    builder.append(message);

    return builder.toString();
  }
}
