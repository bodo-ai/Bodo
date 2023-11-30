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
package org.apache.calcite.sql;

import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.util.Litmus;

import com.bodosql.calcite.rel.type.BodoRelDataTypeFactory;

import org.checkerframework.checker.nullness.qual.Nullable;
import java.util.Objects;

/**
 * A sql type name specification of a timezone aware sql type.
 *
 */
public class SqlTzAwareTypeNameSpec extends SqlTypeNameSpec {

  @Nullable
  public final BodoTZInfo origTz;

  public final int precision;

  /**
   * General SqlTzAwareTypeNameSpec constructor. This holds the information about the timezone
   * if a available, which is done when we construct a SqlTzAwareTypeNameSpec from a type.
   */
  public SqlTzAwareTypeNameSpec(SqlIdentifier identifier, int precision, @Nullable BodoTZInfo tzInfo) {
    super(Objects.requireNonNull(identifier), SqlParserPos.ZERO);
    this.origTz = tzInfo;
    this.precision = precision;
  }

  /**
   * SqlTzAwareTypeNameSpec constructor when not type information is available
   * (e.g. an explicit TIMESTAMP_LTZ cast).
   */
  public SqlTzAwareTypeNameSpec(SqlIdentifier identifier, int precision) {
    this(identifier, precision, null);
  }

  @Override public RelDataType deriveType(final SqlValidator validator) {
    final BodoTZInfo usedTz;
    if (origTz == null) {
      // If we don't have a timezone there was a cast like TIMESTAMP_LTZ, so we fetch
      // the default.
      RelDataTypeSystem typeSystem = validator.getTypeFactory().getTypeSystem();
      if (typeSystem instanceof BodoSQLRelDataTypeSystem) {
        usedTz = ((BodoSQLRelDataTypeSystem) typeSystem).getDefaultTZInfo();
      } else {
        throw new RuntimeException("Internal Error: TZ Aware Timezones require a BodoSQLRelDataTypeSystem");
      }
    } else {
      usedTz = origTz;
    }
    return BodoRelDataTypeFactory.createTZAwareSqlType(validator.getTypeFactory(), usedTz, precision);
  }

  @Override public void unparse(final SqlWriter writer, final int leftPrec, final int rightPrec) {
    // Unparse is used for generating Snowflake SQL text
    writer.keyword("TIMESTAMP_LTZ");
    final SqlWriter.Frame frame =
            writer.startList(SqlWriter.FrameTypeEnum.FUN_CALL, "(", ")");
    writer.print(precision);
    writer.endList(frame);
  }

  @Override public boolean equalsDeep(final SqlTypeNameSpec spec, final Litmus litmus) {
    // TODO: Does NULL matter here?
    if (!(spec instanceof SqlTzAwareTypeNameSpec) || this.origTz != ((SqlTzAwareTypeNameSpec) spec).origTz || this.precision != ((SqlTzAwareTypeNameSpec) spec).precision) {
      return litmus.fail("{} != {}", this, spec);
    }
    return litmus.succeed();
  }
}
