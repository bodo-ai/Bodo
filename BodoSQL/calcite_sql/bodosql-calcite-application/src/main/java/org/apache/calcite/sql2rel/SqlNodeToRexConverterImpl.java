/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.calcite.sql2rel;

import org.apache.calcite.avatica.util.ByteString;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlIntervalQualifier;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlTimeLiteral;
import org.apache.calcite.sql.SqlTimestampLiteral;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.util.BitString;
import org.apache.calcite.util.DateString;
import org.apache.calcite.util.NlsString;
import org.apache.calcite.util.TimeString;
import org.apache.calcite.util.TimestampString;
import org.apache.calcite.util.TimestampWithTimeZoneString;
import org.apache.calcite.util.Util;

import java.math.BigDecimal;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Standard implementation of {@link SqlNodeToRexConverter}.
 */
public class SqlNodeToRexConverterImpl implements SqlNodeToRexConverter {
  //~ Instance fields --------------------------------------------------------

  private final SqlRexConvertletTable convertletTable;

  //~ Constructors -----------------------------------------------------------

  SqlNodeToRexConverterImpl(SqlRexConvertletTable convertletTable) {
    this.convertletTable = convertletTable;
  }

  //~ Methods ----------------------------------------------------------------

  @Override public RexNode convertCall(SqlRexContext cx, SqlCall call) {
    final SqlRexConvertlet convertlet = convertletTable.get(call);
    if (convertlet != null) {
      return convertlet.convertCall(cx, call);
    }

    // No convertlet was suitable. (Unlikely, because the standard
    // convertlet table has a fall-back for all possible calls.)
    throw Util.needToImplement(call);
  }

  @Override public RexLiteral convertInterval(
      SqlRexContext cx,
      SqlIntervalQualifier intervalQualifier) {
    RexBuilder rexBuilder = cx.getRexBuilder();

    return rexBuilder.makeIntervalLiteral(intervalQualifier);
  }

  @Override public RexNode convertLiteral(
      SqlRexContext cx,
      SqlLiteral literal) {
    // Bodo Change: Include the type system when creating literals.
    final RelDataTypeFactory typeFactory = cx.getTypeFactory();
    final RelDataTypeSystem typeSystem = typeFactory.getTypeSystem();
    final RexBuilder rexBuilder = cx.getRexBuilder();
    if (literal.getValue() == null) {
      if (literal.getTypeName() == SqlTypeName.NULL) {
        RelDataType type = cx.getValidator().getValidatedNodeTypeIfKnown(literal);
        if (type == null) {
          type = rexBuilder.getTypeFactory().createSqlType(SqlTypeName.NULL);
        }
        return rexBuilder.makeNullLiteral(type);
      } else {
        RelDataType type = cx.getValidator().getValidatedNodeType(literal);
        return rexBuilder.makeNullLiteral(type);
      }
    }

    switch (literal.getTypeName()) {
    case DECIMAL:
      // exact number
      BigDecimal bd = literal.getValueAs(BigDecimal.class, typeSystem);
      RelDataType type = literal.createSqlType(cx.getTypeFactory());
      return rexBuilder.makeExactLiteral(bd, type);

    case DOUBLE:
      // approximate type
      // TODO:  preserve fixed-point precision and large integers
      return rexBuilder.makeApproxLiteral(literal.getValueAs(BigDecimal.class, typeSystem));

    case CHAR:
      return rexBuilder.makeCharLiteral(literal.getValueAs(NlsString.class, typeSystem));
    case BOOLEAN:
      return rexBuilder.makeLiteral(literal.getValueAs(Boolean.class, typeSystem));
    case BINARY:
      final BitString bitString = literal.getValueAs(BitString.class, typeSystem);
      checkArgument((bitString.getBitCount() % 8) == 0, "incomplete octet");

      // An even number of hexits (e.g. X'ABCD') makes whole number
      // of bytes.
      final ByteString byteString = new ByteString(bitString.getAsByteArray());
      return rexBuilder.makeBinaryLiteral(byteString);
    case SYMBOL:
      return rexBuilder.makeFlag(literal.getValueAs(Enum.class, typeSystem));
    case TIMESTAMP:
      return rexBuilder.makeTimestampLiteral(
          literal.getValueAs(TimestampString.class, typeSystem),
          ((SqlTimestampLiteral) literal).getPrec());
    case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
      return rexBuilder.makeTimestampWithLocalTimeZoneLiteral(
          literal.getValueAs(TimestampString.class),
          ((SqlTimestampLiteral) literal).getPrec());
    case TIMESTAMP_TZ:
      return rexBuilder.makeTimestampTzLiteral(
          literal.getValueAs(TimestampWithTimeZoneString.class),
          ((SqlTimestampLiteral) literal).getPrec());
    case TIME:
      return rexBuilder.makeTimeLiteral(
          literal.getValueAs(TimeString.class, typeSystem),
          ((SqlTimeLiteral) literal).getPrec());
    case DATE:
      return rexBuilder.makeDateLiteral(literal.getValueAs(DateString.class, typeSystem));

    case INTERVAL_YEAR:
    case INTERVAL_YEAR_MONTH:
    case INTERVAL_MONTH:
    case INTERVAL_DAY:
    case INTERVAL_DAY_HOUR:
    case INTERVAL_DAY_MINUTE:
    case INTERVAL_DAY_SECOND:
    case INTERVAL_HOUR:
    case INTERVAL_HOUR_MINUTE:
    case INTERVAL_HOUR_SECOND:
    case INTERVAL_MINUTE:
    case INTERVAL_MINUTE_SECOND:
    case INTERVAL_SECOND:
      SqlIntervalQualifier sqlIntervalQualifier =
          literal.getValueAs(SqlIntervalQualifier.class, typeSystem);
      return rexBuilder.makeIntervalLiteral(
          literal.getValueAs(BigDecimal.class, typeSystem),
          sqlIntervalQualifier);

    case UNKNOWN:
      return convertLiteral(cx, cx.getValidator().resolveLiteral(literal));

    default:
      throw Util.unexpected(literal.getTypeName());
    }
  }
}
