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
package com.bodosql.calcite.sql.parser;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;

import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.google.common.base.Throwables;
import java.util.*;
import kotlin.Pair;
import org.apache.calcite.sql.SqlDialect;
import org.apache.calcite.sql.dialect.MysqlSqlDialect;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.parser.SqlParserFixture;
import org.apache.calcite.sql.parser.SqlParserTest;
import org.apache.calcite.sql.parser.StringAndPos;
import org.apache.calcite.tools.Hoist;
import org.apache.commons.lang3.StringUtils;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Tests the Bodo SQL parser. We use a separate parser to allow us to reduce the amount of changes
 * needed to extend the parser.
 */
public class BodoParserTest extends SqlParserTest {
  public static Map<String, List<String>> timeUnitTestCases =
      Map.ofEntries(
          Map.entry("YEAR", List.of("Year", "Y", "yy", "YYY", "yyYy", "yr", "YEARS", "yrs")),
          Map.entry("MONTH", List.of("Month", "mm", "MON", "mons", "monThs")),
          Map.entry("DAY", List.of("Day", "d", "DD", "DaYs", "dayofmonth")),
          Map.entry("DAYOFMONTH", List.of("Day", "d", "DD", "DaYs", "dayofmonth")),
          Map.entry("DAYOFWEEK", List.of("Dayofweek", "WEEKDAY", "dow", "dW")),
          Map.entry("DAYOFWEEKISO", List.of("Dayofweekiso", "WEEKDAY_ISO", "dow_iso", "dW_iso")),
          Map.entry("DAYOFYEAR", List.of("Dayofyear", "yearday", "DOY", "dY")),
          Map.entry("WEEK", List.of("Week", "w", "WK", "weekOFYeAr", "woy", "wy")),
          Map.entry("WEEKISO", List.of("Weekiso", "week_iso", "WEEKOFYEARISO", "WEEKOFYeaRISO")),
          Map.entry("QUARTER", List.of("Quarter", "q", "QTR", "QtRs", "quarters")),
          Map.entry("YEAROFWEEK", List.of("YearofWeek")),
          Map.entry("YEAROFWEEKISO", List.of("YearofWeekIso")),
          Map.entry("HOUR", List.of("Hour", "H", "hH", "hr", "hours", "hRs")),
          Map.entry("MINUTE", List.of("Minute", "m", "MI", "min", "MinUtes", "mins")),
          Map.entry("SECOND", List.of("Second", "s", "SEC", "SECONDS", "seCs")),
          Map.entry("MILLISECOND", List.of("MILLISECOND", "MS", "MSEC", "MILLISECONDS")),
          Map.entry("MICROSECOND", List.of("MICROSECOND", "US", "USEC", "MICROSECONDS")),
          Map.entry(
              "NANOSECOND",
              List.of(
                  "NANOSECOND",
                  "NS",
                  "NSEC",
                  "NANOSEC",
                  "NSECOND",
                  "NANOSECONDS",
                  "NANOSECS",
                  "NSECONDS")),
          Map.entry("EPOCH_SECOND", List.of("epoch_second", "EPOCH", "epoch_secondS")),
          Map.entry("EPOCH_MILLISECOND", List.of("epoch_millisecond", "EPOCH_MILLISECONDS")),
          Map.entry("EPOCH_MICROSECOND", List.of("epoch_microsecond", "EPOCH_MICROSECONDS")),
          Map.entry("EPOCH_NANOSECOND", List.of("epoch_nanosecond", "EPOCH_NANOSECONDS")),
          Map.entry("TIMEZONE_HOUR", List.of("timezone_hour", "TZH")),
          Map.entry("TIMEZONE_MINUTE", List.of("timezone_minute", "TZM")));

  public SqlParserFixture fixture() {
    return SqlParserFixture.DEFAULT
        .withTester(new BodoTesterImpl())
        .withConfig(c -> c.withParserFactory(SqlBodoParserImpl.FACTORY));
  }

  protected SqlParserFixture sql(String sql) {
    return fixture().sql(sql);
  }

  @Test
  void testSelect() {
    final String sql = "select 1 from t";
    final String expected = "SELECT 1\n" + "FROM `T`";
    sql(sql).ok(expected);
  }

  // This test is a copy of SqlParserTest.testParensInFrom, but modified to change some expected
  // fails to passes.
  @Test
  void testParensInFrom() {
    // UNNEST may not occur within parentheses.
    // FIXME should fail at "unnest"
    sql("select *from (^unnest(x)^)").fails("Expected query or join");

    // Bodo change: allow table name in parens
    // <table-name> may not occur within parentheses.
    // TODO: Postgres gives "syntax error at ')'", which might be better
    sql("select * from (^emp^)");

    // <table-name> may not occur within parentheses.
    // TODO: Postgres gives "syntax error at ')'", which might be better
    sql("select * from (^emp as x^)").fails("Expected query or join");

    // Bodo change: allow table name in parens
    // <table-name> may not occur within parentheses.
    sql("select * from (^emp^) as x");

    // Parentheses around JOINs are OK, and sometimes necessary.
    String sql1 = "select *\n" + "from (emp join dept using (deptno))";
    String expected = "SELECT *\n" + "FROM `EMP`\n" + "INNER JOIN `DEPT` USING (`DEPTNO`)";
    sql(sql1).ok(expected);

    String sql2 = "select *\n" + "from (emp join dept using (deptno))\n" + "join foo using (x)";
    String expected2 =
        "SELECT *\n"
            + "FROM `EMP`\n"
            + "INNER JOIN `DEPT` USING (`DEPTNO`)\n"
            + "INNER JOIN `FOO` USING (`X`)";
    sql(sql2).ok(expected2);

    // In Postgres and Standard SQL, you can alias a join:
    //   "select x.i from (t cross join u) as x"
    // is syntactically and semantically valid; but
    //   "select t.i from (t cross join u) as x"
    // is semantically invalid.
    // TODO: Support this in Calcite.
    sql("select * from (t cross ^join^ u) as x")
        .fails("Join expression encountered in illegal context");
    sql("select *\n" + "from (t cross ^join^ u)\n" + "  tablesample substitute('medium')")
        .fails("Join expression encountered in illegal context");
    sql("select *\n"
            + "from (t cross ^join^ u)\n"
            + "PIVOT (sum(sal) AS sal FOR job in ('CLERK' AS c))")
        .fails("Join expression encountered in illegal context");
  }

  /**
   * This is a failure test making sure the LOOKAHEAD for WHEN clause is 2 in BODO, where in core
   * parser this number is 1.
   *
   * @see SqlParserTest#testCaseExpression()
   * @see <a href= "https://issues.apache.org/jira/browse/CALCITE-2847">[CALCITE-2847] Optimize
   *     global LOOKAHEAD for SQL parsers</a>
   */
  // @Disabled
  @Test
  void testCaseExpressionBodo() {
    sql("case x when 2, 4 then 3 ^when^ then 5 else 4 end")
        .fails("(?s)Encountered \"when then\" at .*");
  }

  /**
   * In Redshift, DATE is a function. It requires special treatment in the parser because it is a
   * reserved keyword. (Curiously, TIMESTAMP and TIME are not functions.)
   */
  @Test
  void testDateFunction() {
    final String expected = "SELECT `DATE`(`X`)\n" + "FROM `T`";
    sql("select date(x) from t").ok(expected);
  }

  @Test
  void testFloor() {
    expr("floor(1.5)").ok("FLOOR(1.5)");
    expr("floor(x)").ok("FLOOR(`X`)");
    expr("floor(3.1415926, 3)").ok("FLOOR(3.1415926, 3)");
    expr("floor(y, -1)").ok("FLOOR(`Y`, -1)");
  }

  @Test
  void testCeil() {
    expr("ceil(1.5)").ok("CEIL(1.5)");
    expr("ceil(x)").ok("CEIL(`X`)");
    expr("ceil(3.1415926, 3)").ok("CEIL(3.1415926, 3)");
    expr("ceil(y, -1)").ok("CEIL(`Y`, -1)");
  }

  @Test
  void testDollarStrings() {
    expr("$$Alphabet Soup$$").ok("'Alphabet Soup'");
    expr("$$$ $_$x$$").ok("'$ $_$x'");
    expr("$$\\n\\t$$").ok("'\\n\\t'");
  }

  /** Tests parsing PostgreSQL-style "::" cast operator. */
  @Test
  void testParseInfixCast() {
    // Numeric Types
    checkParseInfixCast("number", "decimal");
    checkParseInfixCast("decimal", "decimal");
    checkParseInfixCast("numeric", "decimal");
    checkParseInfixCast("int", "integer");
    checkParseInfixCast("integer", "integer");
    checkParseInfixCast("bigint", "bigint");
    checkParseInfixCast("smallint", "smallint");
    checkParseInfixCast("tinyint", "tinyint");
    checkParseInfixCast("byteint", "tinyint");
    checkParseInfixCast("float", "float");
    checkParseInfixCast("float4", "float");
    checkParseInfixCast("float8", "float");
    checkParseInfixCast("double", "double");
    checkParseInfixCast("real", "real");
    checkParseInfixCast("double precision", "double");

    // String & Binary Types
    checkParseInfixCast("char", "char");
    checkParseInfixCast("character", "char");
    checkParseInfixCast("nchar", "char");
    checkParseInfixCast("string", "varchar");
    checkParseInfixCast("text", "varchar");
    checkParseInfixCast("nvarchar", "varchar");
    checkParseInfixCast("nvarchar2", "varchar");
    checkParseInfixCast("char varying", "varchar");
    checkParseInfixCast("nchar varying", "varchar");
    checkParseInfixCast("binary", "binary");
    checkParseInfixCast("varbinary", "varbinary");

    // Logical Types
    checkParseInfixCast("boolean", "boolean");

    // Date & Time Types
    checkParseInfixCast("date", "date");
    checkParseInfixCast("datetime", "timestamp(9)");
    checkParseInfixCast("time", "time(9)");
    checkParseInfixCast("timestamp", "timestamp(9)");
    checkParseInfixCast("timestamp(3)", "timestamp(3)");
    checkParseInfixCast("timestamp_ntz", "timestamp(9)");
    checkParseInfixCast("timestamp_ntz(0)", "timestamp(0)");
    checkParseInfixCast("timestamp_ltz", "timestamp_ltz(9)");
    checkParseInfixCast("timestamp_ltz(4)", "timestamp_ltz(4)");
    // TODO: always use "with time zone" once enableTimestampTz is gone
    if (RelationalAlgebraGenerator.enableTimestampTz) {
      checkParseInfixCast("timestamp_tz", "timestamp(9) with time zone");
      checkParseInfixCast("timestamp_tz(6)", "timestamp(6) with time zone");
      checkParseInfixCast("timestamp with time zone", "timestamp(9) with time zone");
    } else {
      checkParseInfixCast("timestamp_tz", "timestamp_ltz(9)");
      checkParseInfixCast("timestamp_tz(6)", "timestamp_ltz(6)");
      checkParseInfixCast("timestamp with time zone", "timestamp_ltz(9)");
    }

    final String sql = "select -('12' || '.34')::VARCHAR(30)::INTEGER as x\n" + "from t";
    final String expected =
        "" + "SELECT (- ('12' || '.34') :: VARCHAR(30) :: INTEGER) AS `X`\n" + "FROM `T`";
    sql(sql).ok(expected);
  }

  private void checkParseInfixCast(String sqlTypeInput, String sqlTypeOutput) {
    String sql = "SELECT x::" + sqlTypeInput + " FROM (VALUES (1, 2)) as tbl(x,y)";
    String expected =
        "SELECT `X` :: "
            + sqlTypeOutput.toUpperCase(Locale.ROOT)
            + "\n"
            + "FROM (VALUES (ROW(1, 2))) AS `TBL` (`X`, `Y`)";
    sql(sql).ok(expected);
  }

  /** Tests parsing MySQL-style "<=>" equal operator. */
  @Test
  void testParseNullSafeEqual() {
    // x <=> y
    final String projectSql = "SELECT x <=> 3 FROM (VALUES (1, 2)) as tbl(x,y)";
    sql(projectSql).ok("SELECT (`X` <=> 3)\n" + "FROM (VALUES (ROW(1, 2))) AS `TBL` (`X`, `Y`)");
    final String filterSql = "SELECT y FROM (VALUES (1, 2)) as tbl(x,y) WHERE x <=> null";
    sql(filterSql)
        .ok(
            "SELECT `Y`\n"
                + "FROM (VALUES (ROW(1, 2))) AS `TBL` (`X`, `Y`)\n"
                + "WHERE (`X` <=> NULL)");
    final String joinConditionSql =
        "SELECT tbl1.y FROM (VALUES (1, 2)) as tbl1(x,y)\n"
            + "LEFT JOIN (VALUES (null, 3)) as tbl2(x,y) ON tbl1.x <=> tbl2.x";
    sql(joinConditionSql)
        .ok(
            "SELECT `TBL1`.`Y`\n"
                + "FROM (VALUES (ROW(1, 2))) AS `TBL1` (`X`, `Y`)\n"
                + "LEFT JOIN (VALUES (ROW(NULL, 3))) AS `TBL2` (`X`, `Y`) ON (`TBL1`.`X` <=>"
                + " `TBL2`.`X`)");
    // (a, b) <=> (x, y)
    final String rowComparisonSql =
        "SELECT y\n" + "FROM (VALUES (1, 2)) as tbl(x,y) WHERE (x,y) <=> (null,2)";
    sql(rowComparisonSql)
        .ok(
            "SELECT `Y`\n"
                + "FROM (VALUES (ROW(1, 2))) AS `TBL` (`X`, `Y`)\n"
                + "WHERE ((ROW(`X`, `Y`)) <=> (ROW(NULL, 2)))");
    // the higher precedence
    final String highPrecedenceSql = "SELECT x <=> 3 + 3 FROM (VALUES (1, 2)) as tbl(x,y)";
    sql(highPrecedenceSql)
        .ok("SELECT (`X` <=> (3 + 3))\n" + "FROM (VALUES (ROW(1, 2))) AS `TBL` (`X`, `Y`)");
    // the lower precedence
    final String lowPrecedenceSql = "SELECT NOT x <=> 3 FROM (VALUES (1, 2)) as tbl(x,y)";
    sql(lowPrecedenceSql)
        .ok("SELECT (NOT (`X` <=> 3))\n" + "FROM (VALUES (ROW(1, 2))) AS `TBL` (`X`, `Y`)");
  }

  /** Similar to testHoist() but using custom parser. */
  @Test
  void testHoistMySql() {
    // SQL contains back-ticks, which require MySQL's quoting,
    // and DATEADD, which requires Babel/Bodo.
    final String sql =
        "select 1 as x,\n"
            + "  'ab' || 'c' as y\n"
            + "from `my emp` /* comment with 'quoted string'? */ as e\n"
            + "where deptno < 40\n"
            + "and DATEADD(day, 1, hiredate) > date '2010-05-06'";
    final SqlDialect dialect = MysqlSqlDialect.DEFAULT;
    final Hoist.Hoisted hoisted =
        Hoist.create(
                Hoist.config()
                    .withParserConfig(
                        dialect
                            .configureParser(SqlParser.config())
                            .withParserFactory(SqlBodoParserImpl::new)))
            .hoist(sql);

    // Simple toString converts each variable to '?N'
    final String expected =
        "select ?0 as x,\n"
            + "  ?1 || ?2 as y\n"
            + "from `my emp` /* comment with 'quoted string'? */ as e\n"
            + "where deptno < ?3\n"
            + "and DATEADD(?4, ?5, hiredate) > ?6";
    assertThat(hoisted.toString(), is(expected));

    // Custom string converts variables to '[N:TYPE:VALUE]'
    final String expected2 =
        "select [0:DECIMAL:1] as x,\n"
            + "  [1:CHAR:ab] || [2:CHAR:c] as y\n"
            + "from `my emp` /* comment with 'quoted string'? */ as e\n"
            + "where deptno < [3:DECIMAL:40]\n"
            + "and DATEADD([4:SYMBOL:DAY], [5:DECIMAL:1], hiredate) > [6:DATE:2010-05-06]";
    assertThat(hoisted.substitute(SqlParserTest::varToStr), is(expected2));
  }

  /** Test DATEDIFF and its aliases TIMEDIFF and TIMESTAMPDIFF */
  @Test
  void testDateDiff() {
    for (String func : List.of("datediff", "timediff", "timestampdiff")) {
      // quoted time unit
      sql("select " + func + "('second', A, B) from emp")
          .ok("SELECT " + func.toUpperCase() + "(SECOND, `A`, `B`)\nFROM `EMP`");
      // unquoted time unit
      sql("select " + func + "(year, A, B) from emp")
          .ok("SELECT " + func.toUpperCase() + "(YEAR, `A`, `B`)\nFROM `EMP`");
    }
  }

  @Test
  void testLikeAnyAll() {
    for (String likeKind : List.of("like", "ilike")) {
      sql("select a " + likeKind + " any ('%a%', '%b%') from emp")
          .ok("SELECT (`A` " + likeKind.toUpperCase() + " ANY ('%a%', '%b%'))\nFROM `EMP`");
      sql("select a " + likeKind + " all ('%a%', '%b%') from emp")
          .ok("SELECT (`A` " + likeKind.toUpperCase() + " ALL ('%a%', '%b%'))\nFROM `EMP`");
      sql("select a " + likeKind + " any ('%a%', '%b%') escape '|' from emp")
          .ok(
              "SELECT (`A` "
                  + likeKind.toUpperCase()
                  + " ANY ('%a%', '%b%') ESCAPE '|')\nFROM `EMP`");
      sql("select a " + likeKind + " all ('%a%', '%b%') escape '|' from emp")
          .ok(
              "SELECT (`A` "
                  + likeKind.toUpperCase()
                  + " ALL ('%a%', '%b%') ESCAPE '|')\nFROM `EMP`");
    }

    // Precedence tests.
    // This is similar to some of the precedence tests in calcite for like,
    // but excludes some of the tests that don't make much sense for like any.
    // In particular, the tests for nested expressions don't make much sense
    // here because, unlike like, like any doesn't have a right hand side
    // that doesn't involve parenthesis.

    // LIKE has higher precedence than AND
    sql("values a and b like any ('%a%') escape d and e")
        .ok("VALUES (ROW(((`A` AND (`B` LIKE ANY ('%a%') ESCAPE `D`)) AND `E`)))");

    // LIKE has same precedence as '='; LIKE is right-assoc, '=' is left
    sql("values a = b like any ('%a%') = d")
        .ok("VALUES (ROW(((`A` = (`B` LIKE ANY ('%a%'))) = `D`)))");
  }

  @Test
  void testBackslashEscape() {
    // Intended to be \\ but need to escape because of Java escape sequences.
    sql("select '\\\\'")
        // Strange output but standard SQL doesn't include
        // backslash as a string literal escape sequence so this
        // is correct output.
        .ok("SELECT '\\'");
  }

  @Test
  void testCopyIntoDestination() {
    // Test the various destination formats for COPY INTO

    // Regular table (+ with namespace)
    sql("copy into t2 from @t1").ok("COPY INTO `T2` FROM @t1");
    sql("copy into ns.t2 from @t1").ok("COPY INTO `NS`.`T2` FROM @t1");

    // Internal Stage / External Stage: ~[<PATH>]
    sql("copy into @~ from t1").ok("COPY INTO @~ FROM `T1`");
    sql("copy into @~/path from t1").ok("COPY INTO @~/path FROM `T1`");
    sql("copy into @~/larger/path from t1").ok("COPY INTO @~/larger/path FROM `T1`");
    sql("copy into @~/path.csv from t1").ok("COPY INTO @~/path.csv FROM `T1`");
    sql("copy into @~/larger/path.txt from t1").ok("COPY INTO @~/larger/path.txt FROM `T1`");

    // Internal Stage / External Stage: name[<PATH>]
    sql("copy into @t2 from t1").ok("COPY INTO @t2 FROM `T1`");
    sql("copy into @t2/path from t1").ok("COPY INTO @t2/path FROM `T1`");
    sql("copy into @t2/path_to/file.csv from t1").ok("COPY INTO @t2/path_to/file.csv FROM `T1`");

    // Internal Stage / External Stage: namespace.name[<PATH>]
    sql("copy into @ns.t2 from t1").ok("COPY INTO @ns.t2 FROM `T1`");
    sql("copy into @ns.t2/path from t1").ok("COPY INTO @ns.t2/path FROM `T1`");
    sql("copy into @ns.t2/path_to/file.csv from t1")
        .ok("COPY INTO @ns.t2/path_to/file.csv FROM `T1`");

    // Internal Stage / External Stage: %name[<PATH>]
    sql("copy into @%t2 from t1").ok("COPY INTO @%t2 FROM `T1`");
    sql("copy into @%t2/path from t1").ok("COPY INTO @%t2/path FROM `T1`");
    sql("copy into @%t2/path_to/file.csv from t1").ok("COPY INTO @%t2/path_to/file.csv FROM `T1`");

    // Internal Stage / External Stage: namespace.%name[<PATH>]
    sql("copy into @ns.%t2 from t1").ok("COPY INTO @ns.%t2 FROM `T1`");
    sql("copy into @ns.%t2/path from t1").ok("COPY INTO @ns.%t2/path FROM `T1`");
    sql("copy into @ns.%t2/path_to/file.csv from t1")
        .ok("COPY INTO @ns.%t2/path_to/file.csv FROM `T1`");

    // External location (e.g. link to data stored on a CSP)
    sql("copy into 's3://mybucket/./../a.csv' from t1")
        .ok("COPY INTO 's3://mybucket/./../a.csv' FROM `T1`");
    sql("copy into 'gcs://mybucket/./../a.csv' from t1")
        .ok("COPY INTO 'gcs://mybucket/./../a.csv' FROM `T1`");
    sql("copy into 'azure://myaccount.blob.core.windows.net/mycontainer/./../a.csv' from t1")
        .ok("COPY INTO 'azure://myaccount.blob.core.windows.net/mycontainer/./../a.csv' FROM `T1`");
  }

  @Test
  void testCopyIntoSource() {
    // Test the various source formats for COPY INTO

    // Regular table (+ with namespace)
    sql("copy into @ns.%tname from t1").ok("COPY INTO @ns.%tname FROM `T1`");
    sql("copy into @ns.%tname from ns.t1").ok("COPY INTO @ns.%tname FROM `NS`.`T1`");

    // Regular query
    sql("copy into @ns.%tname from (select * from t1 where balance >= 0)")
        .ok(
            "COPY INTO @ns.%tname FROM\n"
                + "(SELECT *\n"
                + "FROM `T1`\n"
                + "WHERE (`BALANCE` >= 0))");

    // Internal Stage / External Stage: ~[<PATH>]
    sql("copy into t1 from @~").ok("COPY INTO `T1` FROM @~");
    sql("copy into t1 from @~/path").ok("COPY INTO `T1` FROM @~/path");
    sql("copy into t1 from @~/larger/path").ok("COPY INTO `T1` FROM @~/larger/path");
    sql("copy into t1 from @~/path.csv").ok("COPY INTO `T1` FROM @~/path.csv");
    sql("copy into t1 from @~/larger/path.txt").ok("COPY INTO `T1` FROM @~/larger/path.txt");

    // Internal Stage / External Stage: name[<PATH>]
    sql("copy into t1 from @t2").ok("COPY INTO `T1` FROM @t2");
    sql("copy into t1 from @t2/path").ok("COPY INTO `T1` FROM @t2/path");
    sql("copy into t1 from @t2/path_to/file.csv").ok("COPY INTO `T1` FROM @t2/path_to/file.csv");

    // Internal Stage / External Stage: namespace.name[<PATH>]
    sql("copy into t1 from @ns.t2").ok("COPY INTO `T1` FROM @ns.t2");
    sql("copy into t1 from @ns.t2/path").ok("COPY INTO `T1` FROM @ns.t2/path");
    sql("copy into t1 from @ns.t2/path_to/file.csv")
        .ok("COPY INTO `T1` FROM @ns.t2/path_to/file.csv");

    // Internal Stage / External Stage: %name[<PATH>]
    sql("copy into t1 from @%t2").ok("COPY INTO `T1` FROM @%t2");
    sql("copy into t1 from @%t2/path").ok("COPY INTO `T1` FROM @%t2/path");
    sql("copy into t1 from @%t2/path_to/file.csv").ok("COPY INTO `T1` FROM @%t2/path_to/file.csv");

    // Internal Stage / External Stage: namespace.%name[<PATH>]
    sql("copy into t1 from @ns.%t2").ok("COPY INTO `T1` FROM @ns.%t2");
    sql("copy into t1 from @ns.%t2/path").ok("COPY INTO `T1` FROM @ns.%t2/path");
    sql("copy into t1 from @ns.%t2/path_to/file.csv")
        .ok("COPY INTO `T1` FROM @ns.%t2/path_to/file.csv");

    // External location (e.g. link to data stored on a CSP)
    sql("copy into t1 from 's3://mybucket/./../a.csv'")
        .ok("COPY INTO `T1` FROM 's3://mybucket/./../a.csv'");
    sql("copy into t1 from 'gcs://mybucket/./../a.csv'")
        .ok("COPY INTO `T1` FROM 'gcs://mybucket/./../a.csv'");
    sql("copy into t1 from 'azure://myaccount.blob.core.windows.net/mycontainer/./../a.csv'")
        .ok("COPY INTO `T1` FROM 'azure://myaccount.blob.core.windows.net/mycontainer/./../a.csv'");

    // Transformation query
    sql("copy into t1 from (select $1 from @t2)")
        .ok("COPY INTO `T1` FROM\n" + "(SELECT `$1`\n" + "FROM @t2)");
    sql("copy into t1 from (select $1 from @t2 t)")
        .ok("COPY INTO `T1` FROM\n" + "(SELECT `$1`\n" + "FROM @t2 AS `T`)");
    sql("copy into t1 from (select T.$1 from @t2 t)")
        .ok("COPY INTO `T1` FROM\n" + "(SELECT `T.$1`\n" + "FROM @t2 AS `T`)");
    sql("copy into t1 from (select T.$1, $2, T.$3, $4 from @t2 t)")
        .ok("COPY INTO `T1` FROM\n" + "(SELECT `T.$1`, `$2`, `T.$3`, `$4`\n" + "FROM @t2 AS `T`)");

    sql("copy into t1 (A) from (select $1 from @t2)")
        .ok("COPY INTO `T1` (`A`) FROM\n" + "(SELECT `$1`\n" + "FROM @t2)");
    sql("copy into t1 (B) from (select $1 from @t2 t)")
        .ok("COPY INTO `T1` (`B`) FROM\n" + "(SELECT `$1`\n" + "FROM @t2 AS `T`)");
    sql("copy into t1 (C) from (select T.$1 from @t2 t)")
        .ok("COPY INTO `T1` (`C`) FROM\n" + "(SELECT `T.$1`\n" + "FROM @t2 AS `T`)");
    sql("copy into t1 (D, E, F, G) from (select T.$1, $2, T.$3, $4 from @t2 t)")
        .ok(
            "COPY INTO `T1` (`D`, `E`, `F`, `G`) FROM\n"
                + "(SELECT `T.$1`, `$2`, `T.$3`, `$4`\n"
                + "FROM @t2 AS `T`)");
  }

  @Test
  void testCopyIntoClauses() {
    // Test the various optional clauses

    // COPY INTO <table>
    sql("copy into t1 from @t2 pattern='.*/.*/.*[.]csv[.]gz'")
        .ok("COPY INTO `T1` FROM @t2 PATTERN = '.*/.*/.*[.]csv[.]gz'");
    sql("copy into t1 from @t2 file_format=(type='JSON')")
        .ok("COPY INTO `T1` FROM @t2 FILE_FORMAT = (TYPE = 'JSON')");
    sql("copy into t1 from @t2 file_format=(format_name='mycsv') pattern='*.csv'")
        .ok("COPY INTO `T1` FROM @t2 PATTERN = '*.csv' FILE_FORMAT = (FORMAT_NAME = 'mycsv')");

    // COPY INTO <location>
    sql("copy into @%t1 from t2 partition by year(dt)::varchar(30)")
        .ok("COPY INTO @%t1 FROM `T2` PARTITION BY YEAR(`DT`) :: VARCHAR(30)");
    sql("copy into @%t1 from t2 file_format=(format_name='ns.fmt1')")
        .ok("COPY INTO @%t1 FROM `T2` FILE_FORMAT = (FORMAT_NAME = 'ns.fmt1')");
    sql("copy into @%t1 from t2 partition by c file_format=(type='PARQUET')")
        .ok("COPY INTO @%t1 FROM `T2` PARTITION BY `C` FILE_FORMAT = (TYPE = 'PARQUET')");
  }

  @Test
  void testNamedParam() {
    // Tests uses of named param with @ syntax to make sure it does not
    // conflict with the parsing of internal/external stages for COPY INTO
    sql("select lpad(s, @n, 24) from table1").ok("SELECT `LPAD`(`S`, @n, 24)\nFROM `TABLE1`");
  }

  /**
   * Tests a statement with the various file format options appended to the end. Uses the format
   * type options documented on snowflake:
   * https://docs.snowflake.com/en/sql-reference/sql/copy-into-table#format-type-options-formattypeoptions
   *
   * @param queryPrefix the prefix of the statement before any file format options
   * @param expectedPrefix the prefix of the expected answer before any file format options
   */
  void testFileFormatOptions(String queryPrefix, String expectedPrefix) {
    // Test the various file format options on COPY INTO queries by generating a
    // hashmap
    // mapping each of the 6 accepted file types to a list of subsets of the
    // accepted options
    HashMap<String, List<Pair<List<String>, List<String>>>> formatTypeMap =
        new HashMap<String, List<Pair<List<String>, List<String>>>>();

    List<Pair<List<String>, List<String>>> argsList =
        new ArrayList<Pair<List<String>, List<String>>>();
    argsList.add(
        new Pair<List<String>, List<String>>(
            List.of(
                "compression=auto",
                "field_delimiter=none",
                "null_if=('A')",
                "record_delimiter='.'",
                "skip_header=100"),
            List.of(
                "COMPRESSION = AUTO",
                "FIELD_DELIMITER = NONE",
                "NULL_IF = ('A')",
                "RECORD_DELIMITER = '.'",
                "SKIP_HEADER = 100")));
    argsList.add(
        new Pair<List<String>, List<String>>(
            List.of(
                "binary_format=base64",
                "compression=gzip",
                "date_format=auto",
                "encoding=utf8",
                "escape='*'",
                "null_if=('A', 'B', 'C')",
                "parse_header=true",
                "record_delimiter=none",
                "skip_blank_lines=false"),
            List.of(
                "BINARY_FORMAT = BASE64",
                "COMPRESSION = GZIP",
                "DATE_FORMAT = AUTO",
                "ENCODING = UTF8",
                "ESCAPE = '*'",
                "NULL_IF = ('A', 'B', 'C')",
                "PARSE_HEADER = TRUE",
                "RECORD_DELIMITER = NONE",
                "SKIP_BLANK_LINES = FALSE")));
    argsList.add(
        new Pair<List<String>, List<String>>(
            List.of(
                "binary_format=hex",
                "compression=bz2",
                "encoding='S'",
                "error_on_column_count_mismatch=true",
                "field_delimiter=';'",
                "parse_header=false",
                "skip_blank_lines=true",
                "time_format='HH:MM:SS'",
                "trim_space=false"),
            List.of(
                "BINARY_FORMAT = HEX",
                "COMPRESSION = BZ2",
                "ENCODING = 'S'",
                "ERROR_ON_COLUMN_COUNT_MISMATCH = TRUE",
                "FIELD_DELIMITER = ';'",
                "PARSE_HEADER = FALSE",
                "SKIP_BLANK_LINES = TRUE",
                "TIME_FORMAT = 'HH:MM:SS'",
                "TRIM_SPACE = FALSE")));
    argsList.add(
        new Pair<List<String>, List<String>>(
            List.of(
                "compression=raw_deflate",
                "empty_field_as_null=true",
                "escape=none",
                "escape_unenclosed_field='/'",
                "field_optionally_enclosed_by=none",
                "replace_invalid_characters=false",
                "timestamp_format='???'"),
            List.of(
                "COMPRESSION = RAW_DEFLATE",
                "EMPTY_FIELD_AS_NULL = TRUE",
                "ESCAPE = NONE",
                "ESCAPE_UNENCLOSED_FIELD = '/'",
                "FIELD_OPTIONALLY_ENCLOSED_BY = NONE",
                "REPLACE_INVALID_CHARACTERS = FALSE",
                "TIMESTAMP_FORMAT = '???'")));
    argsList.add(
        new Pair<List<String>, List<String>>(
            List.of(
                "compression=none",
                "empty_field_as_null=false",
                "error_on_column_count_mismatch=false",
                "escape_unenclosed_field=none",
                "field_optionally_enclosed_by='&'",
                "replace_invalid_characters=true"),
            List.of(
                "COMPRESSION = NONE",
                "EMPTY_FIELD_AS_NULL = FALSE",
                "ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE",
                "ESCAPE_UNENCLOSED_FIELD = NONE",
                "FIELD_OPTIONALLY_ENCLOSED_BY = '&'",
                "REPLACE_INVALID_CHARACTERS = TRUE")));
    formatTypeMap.put("CSV", argsList);

    argsList = new ArrayList<Pair<List<String>, List<String>>>();
    argsList.add(
        new Pair<List<String>, List<String>>(
            List.of(
                "allow_duplicate=true",
                "binary_format=utf8",
                "compression=zstd",
                "date_format='$'",
                "enable_octal=true",
                "replace_invalid_characters=true",
                "strip_null_values=true",
                "strip_outer_array=true",
                "time_format=auto"),
            List.of(
                "ALLOW_DUPLICATE = TRUE",
                "BINARY_FORMAT = UTF8",
                "COMPRESSION = ZSTD",
                "DATE_FORMAT = '$'",
                "ENABLE_OCTAL = TRUE",
                "REPLACE_INVALID_CHARACTERS = TRUE",
                "STRIP_NULL_VALUES = TRUE",
                "STRIP_OUTER_ARRAY = TRUE",
                "TIME_FORMAT = AUTO")));
    argsList = new ArrayList<Pair<List<String>, List<String>>>();
    argsList.add(
        new Pair<List<String>, List<String>>(
            List.of(
                "allow_duplicate=false",
                "compression=deflate",
                "enable_octal=false",
                "replace_invalid_characters=false",
                "strip_null_values=false",
                "strip_outer_array=false",
                "timestamp_format=auto"),
            List.of(
                "ALLOW_DUPLICATE = FALSE",
                "COMPRESSION = DEFLATE",
                "ENABLE_OCTAL = FALSE",
                "REPLACE_INVALID_CHARACTERS = FALSE",
                "STRIP_NULL_VALUES = FALSE",
                "STRIP_OUTER_ARRAY = FALSE",
                "TIMESTAMP_FORMAT = AUTO")));
    formatTypeMap.put("JSON", argsList);

    argsList = new ArrayList<Pair<List<String>, List<String>>>();
    argsList.add(
        new Pair<List<String>, List<String>>(
            List.of("compression=brotli"), List.of("COMPRESSION = BROTLI")));
    formatTypeMap.put("AVRO", argsList);

    argsList = new ArrayList<Pair<List<String>, List<String>>>();
    argsList.add(
        new Pair<List<String>, List<String>>(
            List.of("trim_space=true"), List.of("TRIM_SPACE = TRUE")));
    formatTypeMap.put("ORC", argsList);

    argsList = new ArrayList<Pair<List<String>, List<String>>>();
    argsList.add(
        new Pair<List<String>, List<String>>(
            List.of("binary_as_text=true", "compression=snappy"),
            List.of("BINARY_AS_TEXT = TRUE", "COMPRESSION = SNAPPY")));
    formatTypeMap.put("PARQUET", argsList);

    argsList = new ArrayList<Pair<List<String>, List<String>>>();
    argsList.add(
        new Pair<List<String>, List<String>>(
            List.of(
                "compression=brotli",
                "disable_auto_convert=true",
                "disable_snowflake_data=true",
                "ignore_utf8_errors=true",
                "preserve_space=true",
                "skip_byte_order_mark=true",
                "strip_outer_element=true"),
            List.of(
                "COMPRESSION = BROTLI",
                "DISABLE_AUTO_CONVERT = TRUE",
                "DISABLE_SNOWFLAKE_DATA = TRUE",
                "IGNORE_UTF8_ERRORS = TRUE",
                "PRESERVE_SPACE = TRUE",
                "SKIP_BYTE_ORDER_MARK = TRUE",
                "STRIP_OUTER_ELEMENT = TRUE")));
    argsList.add(
        new Pair<List<String>, List<String>>(
            List.of(
                "compression=zstd",
                "disable_auto_convert=false",
                "disable_snowflake_data=false",
                "ignore_utf8_errors=false",
                "preserve_space=false",
                "skip_byte_order_mark=false",
                "strip_outer_element=false"),
            List.of(
                "COMPRESSION = ZSTD",
                "DISABLE_AUTO_CONVERT = FALSE",
                "DISABLE_SNOWFLAKE_DATA = FALSE",
                "IGNORE_UTF8_ERRORS = FALSE",
                "PRESERVE_SPACE = FALSE",
                "SKIP_BYTE_ORDER_MARK = FALSE",
                "STRIP_OUTER_ELEMENT = FALSE")));
    formatTypeMap.put("XML", argsList);

    for (String fileType : formatTypeMap.keySet()) {
      for (Pair<List<String>, List<String>> optArgs : formatTypeMap.get(fileType)) {
        for (String seperator : List.of(" ", ", ")) {
          String query =
              String.format(
                  "%s file_format=(type='%s' %s)",
                  queryPrefix, fileType, StringUtils.join(optArgs.getFirst(), seperator));
          String expected =
              String.format(
                  "%s FILE_FORMAT = (TYPE = '%s' %s)",
                  expectedPrefix, fileType, StringUtils.join(optArgs.getSecond(), ", "));
          sql(query).ok(expected);
        }
      }
    }
  }

  @Test
  void testCopyIntoFileFormatOptions() {
    testFileFormatOptions("copy into @t1 from t2", "COPY INTO @t1 FROM `T2`");
  }

  @Test
  void testDropTable() {
    // Test all combination of [IF EXISTS] [CASCADE | RESTRICT] [PURGE]
    sql("drop table mytable").ok("DROP TABLE `MYTABLE` CASCADE");
    sql("drop table if exists mytable").ok("DROP TABLE IF EXISTS `MYTABLE` CASCADE");
    sql("drop table mytable cascade").ok("DROP TABLE `MYTABLE` CASCADE");
    sql("drop table if exists mytable cascade").ok("DROP TABLE IF EXISTS `MYTABLE` CASCADE");
    sql("drop table mytable restrict").ok("DROP TABLE `MYTABLE` RESTRICT");
    sql("drop table if exists mytable restrict").ok("DROP TABLE IF EXISTS `MYTABLE` RESTRICT");
    sql("drop table mytable purge").ok("DROP TABLE `MYTABLE` CASCADE PURGE");
    sql("drop table if exists mytable purge").ok("DROP TABLE IF EXISTS `MYTABLE` CASCADE PURGE");
    sql("drop table mytable cascade purge").ok("DROP TABLE `MYTABLE` CASCADE PURGE");
    sql("drop table if exists mytable cascade purge")
        .ok("DROP TABLE IF EXISTS `MYTABLE` CASCADE PURGE");
    sql("drop table mytable restrict purge").ok("DROP TABLE `MYTABLE` RESTRICT PURGE");
    sql("drop table if exists mytable restrict purge")
        .ok("DROP TABLE IF EXISTS `MYTABLE` RESTRICT PURGE");

    // Test PURGE is non-reserved keyword on a subset of cases
    sql("drop table purge").ok("DROP TABLE `PURGE` CASCADE");
    sql("drop table if exists purge").ok("DROP TABLE IF EXISTS `PURGE` CASCADE");
    sql("drop table if exists purge restrict").ok("DROP TABLE IF EXISTS `PURGE` RESTRICT");
    sql("drop table purge purge").ok("DROP TABLE `PURGE` CASCADE PURGE");
    sql("drop table if exists purge cascade purge")
        .ok("DROP TABLE IF EXISTS `PURGE` CASCADE PURGE");
    sql("drop table if exists purge restrict purge")
        .ok("DROP TABLE IF EXISTS `PURGE` RESTRICT PURGE");
  }

  @Test
  void testDropView() {
    sql("drop view mytable").ok("DROP VIEW `MYTABLE`");
    sql("drop view if exists mytable").ok("DROP VIEW IF EXISTS `MYTABLE`");
  }

  @Test
  void testDescribeView() {
    sql("describe view myview").ok("DESCRIBE VIEW `MYVIEW`");
    sql("desc view myview").ok("DESCRIBE VIEW `MYVIEW`");
  }

  @Test
  void testDescribeSchema() {
    sql("describe schema myschema").ok("DESCRIBE SCHEMA `MYSCHEMA`");
    sql("describe schema mydb.myschema").ok("DESCRIBE SCHEMA `MYDB`.`MYSCHEMA`");
    sql("desc schema myschema").ok("DESCRIBE SCHEMA `MYSCHEMA`");
  }

  @Test
  void testAlterTableSwapRename() {
    sql("alter table t1 rename to t2").ok("ALTER TABLE `T1` RENAME TO `T2`");
    sql("alter table if exists t1 rename to t3").ok("ALTER TABLE IF EXISTS `T1` RENAME TO `T3`");
    sql("alter table schema1.t1 rename to schema1.t2")
        .ok("ALTER TABLE `SCHEMA1`.`T1` RENAME TO `SCHEMA1`.`T2`");
    sql("alter table db1.schema1.t1 rename to db1.schema1.t2")
        .ok("ALTER TABLE `DB1`.`SCHEMA1`.`T1` RENAME TO `DB1`.`SCHEMA1`.`T2`");
    sql("alter table t1 swap with t4").ok("ALTER TABLE `T1` SWAP WITH `T4`");
    sql("alter table if exists t1 swap with t5").ok("ALTER TABLE IF EXISTS `T1` SWAP WITH `T5`");
  }

  @Test
  void testAlterTableAddColumn() {
    sql("alter table t1 add column A integer").ok("ALTER TABLE `T1` ADD COLUMN `A` INTEGER");
    sql("alter table if exists t1 add column B VARCHAR(2000)")
        .ok("ALTER TABLE IF EXISTS `T1` ADD COLUMN `B` VARCHAR(2000)");
    sql("alter table t1 add C float").ok("ALTER TABLE `T1` ADD COLUMN `C` FLOAT");
    sql("alter table if exists t1 add D date").ok("ALTER TABLE IF EXISTS `T1` ADD COLUMN `D` DATE");
  }

  @Test
  void testAlterTableRenameColumn() {
    sql("alter table t1 rename column A TO B").ok("ALTER TABLE `T1` RENAME COLUMN `A` TO `B`");
    sql("alter table if exists t1 rename column C TO D")
        .ok("ALTER TABLE IF EXISTS `T1` RENAME COLUMN `C` TO `D`");
    sql("alter table t1 rename E TO F").ok("ALTER TABLE `T1` RENAME COLUMN `E` TO `F`");
    sql("alter table if exists t1 rename G TO H")
        .ok("ALTER TABLE IF EXISTS `T1` RENAME COLUMN `G` TO `H`");
  }

  @Test
  void testAlterTableDropColumn() {
    sql("alter table t1 drop column A").ok("ALTER TABLE `T1` DROP COLUMN `A`");
    sql("alter table t1 drop A").ok("ALTER TABLE `T1` DROP COLUMN `A`");
    sql("alter table t1 drop column if exists A").ok("ALTER TABLE `T1` DROP COLUMN IF EXISTS `A`");
    sql("alter table if exists t1 drop column B, C")
        .ok("ALTER TABLE IF EXISTS `T1` DROP COLUMN `B`, `C`");
    sql("alter table t1 drop D, E, F").ok("ALTER TABLE `T1` DROP COLUMN `D`, `E`, `F`");
    sql("alter table if exists t1 drop G, H, I, J")
        .ok("ALTER TABLE IF EXISTS `T1` DROP COLUMN `G`, `H`, `I`, `J`");
  }

  @Test
  void testAlterTableAlterColumnComment() {
    sql("alter table t1 alter column A comment 'this is a comment'")
        .ok("ALTER TABLE `T1` ALTER COLUMN `A` COMMENT 'this is a comment'");
    sql("alter table t1 alter A comment 'this is a comment'")
        .ok("ALTER TABLE `T1` ALTER COLUMN `A` COMMENT 'this is a comment'");
    sql("alter table t1 alter column A.B comment 'this is also a comment'")
        .ok("ALTER TABLE `T1` ALTER COLUMN `A`.`B` COMMENT 'this is also a comment'");
  }

  @Test
  void testAlterTableAlterColumnDropNotNull() {
    sql("alter table t1 alter column A drop not null")
        .ok("ALTER TABLE `T1` ALTER COLUMN `A` DROP NOT NULL");
  }

  @Test
  void testAlterTableSetProperty() {
    // Basic test
    sql("alter table t1 set property 'p1'='v1'").ok("ALTER TABLE `T1` SET PROPERTY 'p1' = 'v1'");
    // With spaces
    sql("alter table t1 set property 'p1'='This has spaces'")
        .ok("ALTER TABLE `T1` SET PROPERTY 'p1' = 'This has spaces'");
    // Compound Identifier
    sql("alter table t1 set property 'p1.pp1'='v1'")
        .ok("ALTER TABLE `T1` SET PROPERTY 'p1.pp1' = 'v1'");
    sql("alter table t1 set property 'this.property.has.dots'='v1'")
        .ok("ALTER TABLE `T1` SET PROPERTY 'this.property.has.dots' = 'v1'");
    // Multiple tag
    sql("alter table t1 set property 'p1'='v1', 'p2'='v2'")
        .ok("ALTER TABLE `T1` SET PROPERTY 'p1' = 'v1', 'p2' = 'v2'");
    // if exists option
    sql("alter table if exists t1 set property 'p1'='v1'")
        .ok("ALTER TABLE IF EXISTS `T1` SET PROPERTY 'p1' = 'v1'");
    sql("alter table if exists t1 set property 'p1'='v1', 'p2'='v2'")
        .ok("ALTER TABLE IF EXISTS `T1` SET PROPERTY 'p1' = 'v1', 'p2' = 'v2'");
    // aliases
    sql("alter table t1 set tag 'p1'='v1', 'p2'='v2'")
        .ok("ALTER TABLE `T1` SET PROPERTY 'p1' = 'v1', 'p2' = 'v2'");
    sql("alter table t1 set tags 'p1'='v1', 'p2'='v2'")
        .ok("ALTER TABLE `T1` SET PROPERTY 'p1' = 'v1', 'p2' = 'v2'");
    sql("alter table t1 set properties 'p1'='v1', 'p2'='v2'")
        .ok("ALTER TABLE `T1` SET PROPERTY 'p1' = 'v1', 'p2' = 'v2'");
    sql("alter table t1 set tblproperty 'p1'='v1', 'p2'='v2'")
        .ok("ALTER TABLE `T1` SET PROPERTY 'p1' = 'v1', 'p2' = 'v2'");
    sql("alter table t1 set tblproperties 'p1'='v1', 'p2'='v2'")
        .ok("ALTER TABLE `T1` SET PROPERTY 'p1' = 'v1', 'p2' = 'v2'");
    // Edge case property / value names
    sql("alter table t1 set property 'property1'='value1'")
        .ok("ALTER TABLE `T1` SET PROPERTY 'property1' = 'value1'");
    sql("alter table t1 set property 'property'='value'")
        .ok("ALTER TABLE `T1` SET PROPERTY 'property' = 'value'");
    sql("alter table t1 set property ' '=' '").ok("ALTER TABLE `T1` SET PROPERTY ' ' = ' '");
  }

  @Test
  void testAlterTableUnsetProperty() {
    // Basic test
    sql("alter table t1 unset property 'p1'").ok("ALTER TABLE `T1` UNSET PROPERTY 'p1'");
    // Spaces
    sql("alter table t1 unset property 'this has spaces'")
        .ok("ALTER TABLE `T1` UNSET PROPERTY 'this has spaces'");
    // Multiple tags
    sql("alter table t1 unset property 'p1', 'p2'")
        .ok("ALTER TABLE `T1` UNSET PROPERTY 'p1', 'p2'");
    // if exists option
    sql("alter table if exists t1 unset property 'p1'")
        .ok("ALTER TABLE IF EXISTS `T1` UNSET PROPERTY 'p1'");
    sql("alter table if exists t1 unset property  'p1', 'p2'")
        .ok("ALTER TABLE IF EXISTS `T1` UNSET PROPERTY 'p1', 'p2'");
    // aliases
    sql("alter table t1 unset tag 'p1', 'p2'").ok("ALTER TABLE `T1` UNSET PROPERTY 'p1', 'p2'");
    sql("alter table t1 unset tags 'p1', 'p2'").ok("ALTER TABLE `T1` UNSET PROPERTY 'p1', 'p2'");
    sql("alter table t1 unset properties 'p1', 'p2'")
        .ok("ALTER TABLE `T1` UNSET PROPERTY 'p1', 'p2'");
    sql("alter table t1 unset tblproperty 'p1', 'p2'")
        .ok("ALTER TABLE `T1` UNSET PROPERTY 'p1', 'p2'");
    sql("alter table t1 unset tblproperties 'p1', 'p2'")
        .ok("ALTER TABLE `T1` UNSET PROPERTY 'p1', 'p2'");
  }

  @Test
  void testAlterViewRename() {
    sql("alter view t1 rename to t2").ok("ALTER VIEW `T1` RENAME TO `T2`");
    sql("alter view if exists t1 rename to t3").ok("ALTER VIEW IF EXISTS `T1` RENAME TO `T3`");
  }

  @Test
  void testShowTables() {
    sql("show terse tables in dbname.schemaname").ok("SHOW TERSE TABLES IN `DBNAME`.`SCHEMANAME`");
    sql("show tables in dbname.schemaname").ok("SHOW TABLES IN `DBNAME`.`SCHEMANAME`");
  }

  @Test
  void testShowViews() {
    sql("show terse views in dbname.schemaname").ok("SHOW TERSE VIEWS IN `DBNAME`.`SCHEMANAME`");
    sql("show views in dbname.schemaname").ok("SHOW VIEWS IN `DBNAME`.`SCHEMANAME`");
  }

  @Test
  void testShowObjects() {
    sql("show terse objects in dbname.schemaname")
        .ok("SHOW TERSE OBJECTS IN `DBNAME`.`SCHEMANAME`");
    sql("show objects in dbname.schemaname").ok("SHOW OBJECTS IN `DBNAME`.`SCHEMANAME`");
  }

  @Test
  void testShowSchemas() {
    sql("show terse schemas in dbname").ok("SHOW TERSE SCHEMAS IN `DBNAME`");
    sql("show schemas in dbname").ok("SHOW SCHEMAS IN `DBNAME`");
  }

  @Test
  void testShowTblproperties() {
    sql("show tblproperties tablename ('propertyname')")
        .ok("SHOW TBLPROPERTIES `TABLENAME` ('propertyname')");
    sql("show tblproperties schemaname.tablename ('propertyname')")
        .ok("SHOW TBLPROPERTIES `SCHEMANAME`.`TABLENAME` ('propertyname')");
    sql("show tblproperties tablename").ok("SHOW TBLPROPERTIES `TABLENAME`");
    sql("show tblproperties schemaname.tablename")
        .ok("SHOW TBLPROPERTIES `SCHEMANAME`.`TABLENAME`");

    // aliases
    sql("show properties tablename").ok("SHOW TBLPROPERTIES `TABLENAME`");
    sql("show tags tablename").ok("SHOW TBLPROPERTIES `TABLENAME`");
  }

  /**
   * Parametrically tests CREATE TABLE statements with all 44 possible combinations of OR REPLACE,
   * table type and IF NOT EXISTS
   *
   * @param querySuffix The suffix of the `CREATE TABLE` statement after the table name
   * @param expectedSuffix The same as querySuffix but for the unparsed answer.
   * @param requireOrReplace Whether OR REPLACE should always be used
   */
  void testCreateTableFormat(String querySuffix, String expectedSuffix, boolean requireOrReplace) {
    List<Pair<String, String>> orReplaceList =
        new ArrayList<Pair<String, String>>(
            Arrays.asList(
                new Pair<String, String>("", ""),
                new Pair<String, String>("or replace ", "OR REPLACE ")));
    if (requireOrReplace) {
      orReplaceList.remove(0);
    }
    List<Pair<String, String>> tableTypeList =
        new ArrayList<Pair<String, String>>(
            Arrays.asList(
                new Pair<String, String>("", ""),
                new Pair<String, String>("transient ", "TRANSIENT "),
                new Pair<String, String>("temp ", "TEMPORARY "),
                new Pair<String, String>("temp ", "TEMPORARY "),
                new Pair<String, String>("local temp ", "TEMPORARY "),
                new Pair<String, String>("global temp ", "TEMPORARY "),
                new Pair<String, String>("temporary ", "TEMPORARY "),
                new Pair<String, String>("temporary ", "TEMPORARY "),
                new Pair<String, String>("local temporary ", "TEMPORARY "),
                new Pair<String, String>("global temporary ", "TEMPORARY "),
                new Pair<String, String>("volatile ", "TEMPORARY ")));
    List<Pair<String, String>> ifNotExistsList =
        new ArrayList<Pair<String, String>>(
            Arrays.asList(
                new Pair<String, String>("", ""),
                new Pair<String, String>("if not exists ", "IF NOT EXISTS ")));
    String queryFmt = "create %s%stable %sout_test %s";
    String expectedFmt = "CREATE %s%sTABLE %s`OUT_TEST` %s";
    for (Pair<String, String> orReplace : orReplaceList) {
      for (Pair<String, String> tableType : tableTypeList) {
        for (Pair<String, String> ifNotExists : ifNotExistsList) {
          String query =
              String.format(
                  queryFmt,
                  orReplace.getFirst(),
                  tableType.getFirst(),
                  ifNotExists.getFirst(),
                  querySuffix);
          String expected =
              String.format(
                  expectedFmt,
                  orReplace.getSecond(),
                  tableType.getSecond(),
                  ifNotExists.getSecond(),
                  expectedSuffix);
          sql(query).ok(expected);
        }
      }
    }
  }

  @Test
  void testCreateTableRegular() {
    // Tests CREATE TABLE without AS, LIKE or CLONE clauses

    // Test with no additional clauses
    String query = "(A integer, B integer, C integer)";
    String expected = "(`A` INTEGER, `B` INTEGER, `C` INTEGER)";
    testCreateTableFormat(query, expected, false);

    // Test with a CLUSTER BY clause
    query = "(A integer, B integer, C INTEGER) cluster by (A, C)";
    expected = "(`A` INTEGER, `B` INTEGER, `C` INTEGER) CLUSTER BY (`A`, `C`)";
    testCreateTableFormat(query, expected, false);

    // Test with a COPY GRANTS clause
    query = "(A integer, B integer, C INTEGER) copy grants";
    expected = "(`A` INTEGER, `B` INTEGER, `C` INTEGER) COPY GRANTS";
    testCreateTableFormat(query, expected, true);

    // Test with a COPY GRANTS clause and multiple CLUSTER BY clauses (the last one
    // should be
    // selected)
    query = "(A integer, B integer, C INTEGER) cluster by (a) copy grants cluster by (b)";
    expected = "(`A` INTEGER, `B` INTEGER, `C` INTEGER) CLUSTER BY (`B`) COPY GRANTS";
    testCreateTableFormat(query, expected, true);
  }

  @Test
  void testCreateTableAsWithoutColumns() {
    // Tests CREATE TABLE with an AS SELECT clause without any column names being
    // provided

    // Test with no additional clauses
    String query = "as select a, b, c from other_table";
    String expected = "AS\n" + "SELECT `A`, `B`, `C`\n" + "FROM `OTHER_TABLE`";
    testCreateTableFormat(query, expected, false);

    // Test with a COPY GRANTS clause
    query = "copy grants as select a, b, c from other_table";
    expected = "COPY GRANTS AS\n" + "SELECT `A`, `B`, `C`\n" + "FROM `OTHER_TABLE`";
    testCreateTableFormat(query, expected, true);
  }

  @Test
  void testCreateTableAsWithColumns() {
    // Tests CREATE TABLE with an AS SELECT clause with column names being provided

    // Test with no additional clauses
    String query = "(c1 integer, c2 integer) as select 1, 2 from other_table";
    String expected = "(`C1` INTEGER, `C2` INTEGER) AS\n" + "SELECT 1, 2\n" + "FROM `OTHER_TABLE`";
    testCreateTableFormat(query, expected, false);

    // Test with a COPY GRANTS clause
    query = "(c1 integer, c2 integer) copy grants as select 1, 2 from other_table";
    expected =
        "(`C1` INTEGER, `C2` INTEGER) COPY GRANTS AS\n" + "SELECT 1, 2\n" + "FROM `OTHER_TABLE`";
    testCreateTableFormat(query, expected, true);

    // Test with a CLUSTER BY clause
    query = "(c1 integer, c2 integer) cluster by (c1) as select 1, 2 from other_table";
    expected =
        "(`C1` INTEGER, `C2` INTEGER) CLUSTER BY (`C1`) AS\n"
            + "SELECT 1, 2\n"
            + "FROM `OTHER_TABLE`";
    testCreateTableFormat(query, expected, true);
  }

  @Test
  void testCreateTableAsWithComments() {
    // Tests CREATE TABLE with an AS SELECT clause with columns where there are comments for
    // some of the columns and for the overall table.
    String query =
        "(c1 integer comment 'hello', c2 integer, c3 varchar comment $$goodbye$$) comment ="
            + " $$Fizz\n"
            + "Buzz$$ as select 1, 2, 3 from other_table";
    String expected =
        "(`C1` INTEGER COMMENT 'hello', `C2` INTEGER, `C3` VARCHAR COMMENT 'goodbye') COMMENT ="
            + " 'Fizz\n"
            + "Buzz' AS\n"
            + "SELECT 1, 2, 3\n"
            + "FROM `OTHER_TABLE`";
    testCreateTableFormat(query, expected, false);
  }

  @Test
  void testCreateTableTblProperties() {
    // Tests CREATE TABLE with or without "COLUMN COMMENTS", "AS SELECT", "TBLPROPERTIES", "TABLE
    // COMMENTS"
    List<Pair<String, String>> Properties =
        new ArrayList<Pair<String, String>>(
            Arrays.asList(
                new Pair<String, String>("", ""),
                new Pair<String, String>(
                    " tags ('a' = 'c', 'd' = 'kotlin')",
                    " TBLPROPERTIES ('a' = 'c', 'd' = 'kotlin')"),
                new Pair<String, String>(
                    " property ('a' = 'c', 'd' = 'kotlin')",
                    " TBLPROPERTIES ('a' = 'c', 'd' = 'kotlin')"),
                new Pair<String, String>(
                    " tblproperties ('a' = 'c', 'd' = 'kotlin')",
                    " TBLPROPERTIES ('a' = 'c', 'd' = 'kotlin')")));
    List<Pair<String, String>> AsSelect =
        new ArrayList<Pair<String, String>>(
            Arrays.asList(
                new Pair<String, String>("", ""),
                new Pair<String, String>(
                    " as select a, b, c from other_table",
                    " AS\n" + "SELECT `A`, `B`, `C`\n" + "FROM `OTHER_TABLE`")));
    List<Pair<String, String>> TableComments =
        new ArrayList<Pair<String, String>>(
            Arrays.asList(
                new Pair<String, String>("", ""),
                new Pair<String, String>(" comment = $$A\nB$$", " COMMENT = 'A\nB'")));
    List<Pair<String, String>> Columns =
        new ArrayList<Pair<String, String>>(
            Arrays.asList(
                new Pair<String, String>(
                    "(c1 integer, c2 integer)", "(`C1` INTEGER, `C2` INTEGER)"),
                new Pair<String, String>(
                    "(c1 integer comment 'hello', c2 integer, c3 varchar comment $$goodbye$$)",
                    "(`C1` INTEGER COMMENT 'hello', `C2` INTEGER, `C3` VARCHAR COMMENT"
                        + " 'goodbye')")));
    for (Pair<String, String> properties : Properties) {
      for (Pair<String, String> asselect : AsSelect) {
        for (Pair<String, String> tableComment : TableComments) {
          for (Pair<String, String> column : Columns) {
            // Test different orders of optional arguments
            // table comments before properties and properties before table comments
            String query1 =
                column.getFirst()
                    + tableComment.getFirst()
                    + properties.getFirst()
                    + asselect.getFirst();
            String query2 =
                column.getFirst()
                    + properties.getFirst()
                    + tableComment.getFirst()
                    + asselect.getFirst();
            String expected =
                column.getSecond()
                    + tableComment.getSecond()
                    + properties.getSecond()
                    + asselect.getSecond();
            testCreateTableFormat(query1, expected, false);
            testCreateTableFormat(query2, expected, false);
          }
        }
      }
    }
  }

  @Test
  void testCreateTableLike() {
    // Tests CREATE TABLE with a LIKE clause

    // Test with no additional clauses
    String query = "like other_table";
    String expected = "LIKE `OTHER_TABLE`";
    testCreateTableFormat(query, expected, false);

    // Test with a CLUSTER BY clause
    query = "like other_table cluster by (mod(a, 10), mod(b, 10))";
    expected = "LIKE `OTHER_TABLE` CLUSTER BY (MOD(`A`, 10), MOD(`B`, 10))";
    testCreateTableFormat(query, expected, false);

    // Test with a COPY GRANTS clause
    query = "like other_table copy grants";
    expected = "LIKE `OTHER_TABLE` COPY GRANTS";
    testCreateTableFormat(query, expected, false);

    // Test with multiple COPY GRANTS clauses and a CLUSTER BY clause
    query = "like other_table copy grants copy grants cluster by (d) copy grants";
    expected = "LIKE `OTHER_TABLE` CLUSTER BY (`D`) COPY GRANTS";
    testCreateTableFormat(query, expected, false);
  }

  @Test
  void testCreateTableClone() {
    // Tests CREATE TABLE with a CLONE clause

    // Test with no additional clauses
    String query = "clone other_table";
    String expectedFmt = "CLONE `OTHER_TABLE`";
    testCreateTableFormat(query, expectedFmt, false);

    // Test with a COPY GRANTS clause
    query = "clone other_table copy grants";
    expectedFmt = "CLONE `OTHER_TABLE` COPY GRANTS";
    testCreateTableFormat(query, expectedFmt, false);
  }

  @Test
  void testCreateTableColumnDefault() {
    // Tests regular CREATE TABLE statements where the columns have default values
    // or
    // auto increments

    // Scalar default
    String query = "(A integer default 0)";
    String expected = "(`A` INTEGER DEFAULT 0)";
    testCreateTableFormat(query, expected, false);

    // Standard auto-increment syntax
    query = "(A integer autoincrement (0, 1))";
    expected = "(`A` INTEGER AUTOINCREMENT (0, 1))";
    testCreateTableFormat(query, expected, false);

    // START/INCREMENT auto-increment syntax
    query = "(A integer autoincrement start 100 increment -1)";
    expected = "(`A` INTEGER AUTOINCREMENT (100, -1))";
    testCreateTableFormat(query, expected, false);

    // Standard auto-increment syntax using IDENTITY
    query = "(A integer identity (0, 1))";
    expected = "(`A` INTEGER AUTOINCREMENT (0, 1))";
    testCreateTableFormat(query, expected, false);

    // START/INCREMENT auto-increment syntax using IDENTITY
    query = "(A integer identity start 100 increment -1)";
    expected = "(`A` INTEGER AUTOINCREMENT (100, -1))";
    testCreateTableFormat(query, expected, false);

    // Mixed combinations and orderings with AS SELECT
    query =
        "(A date not null default '1999-12-31'::date, B integer autoincrement(0,1) not null) as"
            + " select 1, 2, 3 from emp";
    expected =
        "(`A` DATE NOT NULL DEFAULT '1999-12-31' :: DATE, `B` INTEGER NOT NULL AUTOINCREMENT (0,"
            + " 1)) AS\n"
            + "SELECT 1, 2, 3\n"
            + "FROM `EMP`";
    testCreateTableFormat(query, expected, false);
  }

  @Test
  void testUpdateBasic() {
    // Tests basic UPDATE syntax
    final String update = "UPDATE T1 SET A = A + 1, B = B - 1";
    final String expected = "UPDATE `T1` SET `A` = (`A` + 1), `B` = (`B` - 1)";
    sql(update).ok(expected);
  }

  @Test
  void testUpdateWhere() {
    // Tests UPDATE using a WHERE clause
    final String update = "UPDATE T1 SET A = B WHERE A < B";
    final String expected = "UPDATE `T1` SET `A` = `B`\n" + "WHERE (`A` < `B`)";
    sql(update).ok(expected);
  }

  @Test
  void testUpdateFromWhere() {
    // Tests UPDATE using a FROM clause and a WHERE clause
    final String update = "UPDATE T1 SET A = T2.A FROM T2 WHERE T1.B = T2.B";
    final String expected =
        "UPDATE `T1` SET `A` = `T2`.`A`\n" + "FROM `T2`\n" + "WHERE (`T1`.`B` = `T2`.`B`)";
    sql(update).ok(expected);
  }

  @Test
  void testUpdateComplex() {
    // Tests a more complex UPDATE using a FROM clause and a WHERE clause
    final String update =
        "UPDATE table1 T1 SET B = T2.B FROM (SELECT A, COUNT(*) AS B FROM table1 GROUP BY A"
            + " HAVING COUNT(*) > 3) AS T2 WHERE T1.A = T2.A";
    final String expected =
        "UPDATE `TABLE1` AS `T1` SET `B` = `T2`.`B`\n"
            + "FROM (SELECT `A`, COUNT(*) AS `B`\n"
            + "FROM `TABLE1`\n"
            + "GROUP BY `A`\n"
            + "HAVING (COUNT(*) > 3)) AS `T2`\n"
            + "WHERE (`T1`.`A` = `T2`.`A`)";
    sql(update).ok(expected);
  }

  @Test
  void testCreateViewPrefixSyntax() {
    // Tests that various kinds of CREATE VIEW syntax are parsed to the point that we can use
    // the definition for inlining (this means we don't care about most of the features).
    List<Pair<String, String>> orReplacePairs =
        List.of(new Pair("", ""), new Pair("or replace ", "OR REPLACE "));
    // SECURE keyword currently has no effect
    List<Pair<String, String>> securePairs = List.of(new Pair("", ""), new Pair("secure ", ""));
    // View type keywords currently have no effect
    List<Pair<String, String>> typePairs =
        List.of(
            new Pair("", ""),
            new Pair("temp ", ""),
            new Pair("local temporary ", ""),
            new Pair("global volatile ", ""));
    // RECURSIVE keyword currently has no effect
    List<Pair<String, String>> recursivePairs =
        List.of(new Pair("", ""), new Pair("recursive ", ""));
    // IF NOT EXISTS keyword currently has no effect
    List<Pair<String, String>> ifNotExistsPairs =
        List.of(new Pair("", ""), new Pair("if not exists ", ""));
    for (Pair<String, String> orReplace : orReplacePairs) {
      for (Pair<String, String> secure : securePairs) {
        for (Pair<String, String> type : typePairs) {
          for (Pair<String, String> recursive : recursivePairs) {
            for (Pair<String, String> ifNotExists : ifNotExistsPairs) {
              String query =
                  "create "
                      + orReplace.getFirst()
                      + secure.getFirst()
                      + type.getFirst()
                      + recursive.getFirst()
                      + "view "
                      + ifNotExists.getFirst()
                      + "vname as select * from tname";
              String ans =
                  "CREATE "
                      + orReplace.getSecond()
                      + secure.getSecond()
                      + type.getSecond()
                      + recursive.getSecond()
                      + "VIEW "
                      + ifNotExists.getSecond()
                      + "`VNAME` AS\nSELECT *\nFROM `TNAME`";
              sql(query).ok(ans);
            }
          }
        }
      }
    }
  }

  @Test
  void testCreateViewColumnSyntax() {
    // Tests that various kinds of CREATE VIEW syntax for column declarations are parsed to
    // the point that we can use the definition for inlining (this means we don't care about
    // most of the features).
    String query;
    String answer;

    query =
        "CREATE OR REPLACE VIEW ANALYTICS.PRODUCTION.ACCOUNT_USERS (\"ID\", TOKEN, RETAILER_ID,"
            + " VERSION, BRAND_ID, STATUS, CREATED_AT, UPDATED_AT, USER_ID,"
            + " IS_SHARING_INSTRUMENTS) AS SELECT \"ID\"::NUMBER(38,0) AS \"ID\","
            + " TOKEN::VARCHAR(32) AS TOKEN, RETAILER_ID::NUMBER(38,0) AS RETAILER_ID ,"
            + " VERSION::NUMBER(38,0) AS VERSION , BRAND_ID::NUMBER(38,0) AS BRAND_ID ,"
            + " STATUS::VARCHAR(30) AS STATUS , TO_TIMESTAMP_NTZ(CREATED_AT) AS CREATED_AT ,"
            + " TO_TIMESTAMP_NTZ(UPDATED_AT) AS UPDATED_AT , USER_ID::NUMBER(38,0) AS USER_ID ,"
            + " IS_SHARING_INSTRUMENTS::BOOLEAN AS IS_SHARING_INSTRUMENTS         FROM"
            + " RAW.FIVETRAN__MYSQL__PRODUCTION__BACKEND_PRIORITY__INDIGOFAIR_PROD.ACCOUNT_USERS";

    answer =
        "CREATE OR REPLACE VIEW `ANALYTICS`.`PRODUCTION`.`ACCOUNT_USERS` (`ID` UNKNOWN, `TOKEN`"
            + " UNKNOWN, `RETAILER_ID` UNKNOWN, `VERSION` UNKNOWN, `BRAND_ID` UNKNOWN, `STATUS`"
            + " UNKNOWN, `CREATED_AT` UNKNOWN, `UPDATED_AT` UNKNOWN, `USER_ID` UNKNOWN,"
            + " `IS_SHARING_INSTRUMENTS` UNKNOWN) AS\n"
            + "SELECT `ID` :: DECIMAL(38, 0) AS `ID`, `TOKEN` :: VARCHAR(32) AS `TOKEN`,"
            + " `RETAILER_ID` :: DECIMAL(38, 0) AS `RETAILER_ID`, `VERSION` :: DECIMAL(38, 0) AS"
            + " `VERSION`, `BRAND_ID` :: DECIMAL(38, 0) AS `BRAND_ID`, `STATUS` :: VARCHAR(30) AS"
            + " `STATUS`, `TO_TIMESTAMP_NTZ`(`CREATED_AT`) AS `CREATED_AT`,"
            + " `TO_TIMESTAMP_NTZ`(`UPDATED_AT`) AS `UPDATED_AT`, `USER_ID` :: DECIMAL(38, 0) AS"
            + " `USER_ID`, `IS_SHARING_INSTRUMENTS` :: BOOLEAN AS `IS_SHARING_INSTRUMENTS`\n"
            + "FROM `RAW`.`FIVETRAN__MYSQL__PRODUCTION__BACKEND_PRIORITY__INDIGOFAIR_PROD`.`ACCOUNT_USERS`";
    sql(query).ok(answer);

    // Test columns with tags, masking policies, and a tag/row access policy at the end
    query =
        "create view vname ( ident number(38,0) masking policy bmp1 with tag"
            + " (bodo.tags.view_tag='h53172'), bar array with"
            + " tag(bodo.tags.semi_tag='array[string]') with masking policy bmp2 using (bar,"
            + " array_size(bar)), txt varchar not null, foo int with tag (country='USA', city='S')"
            + " with masking policy bmp3 using (foo, HASH(foo), lower(foo)) ) with row access"
            + " policy brap ON (ident, bar) copy grants as select * from tname";
    answer =
        "CREATE VIEW `VNAME` (`IDENT` DECIMAL(38, 0), `BAR` `ARRAY`, `TXT` VARCHAR NOT NULL, `FOO`"
            + " INTEGER) AS\n"
            + "SELECT *\n"
            + "FROM `TNAME`";
    sql(query).ok(answer);

    // Test providing a comment afterward
    query =
        "CREATE OR REPLACE VIEW \"ANALYTICS\".\"PRODUCTION\".\"MESSENGER_CONVERSATIONS\" copy"
            + " grants comment = 'A view on messenger_conversations (not sensitive)' AS SELECT *"
            + " FROM"
            + " RAW.FIVETRAN__MYSQL__PRODUCTION__MESSENGER__MESSENGER_PROD.MESSENGER_CONVERSATIONS";
    answer =
        "CREATE OR REPLACE VIEW `ANALYTICS`.`PRODUCTION`.`MESSENGER_CONVERSATIONS` AS\n"
            + "SELECT *\n"
            + "FROM `RAW`.`FIVETRAN__MYSQL__PRODUCTION__MESSENGER__MESSENGER_PROD`.`MESSENGER_CONVERSATIONS`";
    sql(query).ok(answer);
  }

  @Test
  void testIntervalKeywordsUnreserved() {
    // Test that interval units can be column names
    final String query =
        "SELECT t.YEAR AS YEARS, t.QUARTER AS QUARTERS, t.MONTH AS MONTHS, t.WEEKS AS WEEK, t.DAY"
            + " AS DAYS, t.HOUR AS HOURS, t.MINUTE AS MINUTES, t.SECOND AS SECONDS FROM table1 t";
    final String expected =
        "SELECT `T`.`YEAR` AS `YEARS`, `T`.`QUARTER` AS `QUARTERS`, "
            + "`T`.`MONTH` AS `MONTHS`, `T`.`WEEKS` AS `WEEK`, `T`.`DAY` AS `DAYS`, "
            + "`T`.`HOUR` AS `HOURS`, `T`.`MINUTE` AS `MINUTES`, `T`.`SECOND` AS `SECONDS`\n"
            + "FROM `TABLE1` AS `T`";
    sql(query).ok(expected);
  }

  @Test
  void testTypeNameKeywordsUnreserved() {
    // Test that type names can be column names
    final ArrayList<String> typeNames =
        new ArrayList<String>(
            Arrays.asList(
                "INTEGER",
                "INT",
                "BIGINT",
                "SMALLINT",
                "TINYINT",
                "BYTEINT",
                "NUMBER",
                "NUMERIC",
                "FLOAT",
                "FLOAT4",
                "FLOAT8",
                "DOUBLE",
                "DECIMAL",
                "CHAR",
                "CHARACTER",
                "NCHAR",
                "VARCHAR",
                "STRING",
                "TEXT",
                "NVARCHAR",
                "NVARCHAR2",
                "BINARY",
                "VARBINARY",
                "BOOLEAN",
                "DATE",
                "TIME",
                "TIMESTAMP",
                // TIMESTAMP_TZ is not registered in parser.jj yet
                "TIMESTAMP_LTZ",
                "TIMESTAMP_NTZ",
                // "TIMESTAMP_TZ",
                "VARIANT",
                "ARRAY",
                "OBJECT"));
    String query = "SELECT";
    String expected = "SELECT";
    for (String typeName : typeNames) {
      query += " t." + typeName + ",";
      expected += " `T`.`" + typeName + "`,";
    }
    // add an extra 1 to minimize additional processing of strings
    query += "1 FROM table1 t";
    expected += " 1\nFROM `TABLE1` AS `T`";
    sql(query).ok(expected);
  }

  @Test
  void testTruncateTable() {
    sql("truncate table if exists temp").ok("TRUNCATE TABLE IF EXISTS `TEMP`");
  }

  @Test
  void testNestedDataAccessRewrite() {
    // Basic field access
    sql("SELECT A:foo FROM T").ok("SELECT GET_PATH(`A`, 'foo')\nFROM `T`");
    sql("SELECT T.A:foo FROM T").ok("SELECT GET_PATH(`T`.`A`, 'foo')\nFROM `T`");
    sql("SELECT parse_json(A):foo FROM T")
        .ok("SELECT GET_PATH(`PARSE_JSON`(`A`), 'foo')\nFROM `T`");

    // Using a quoted identifier
    sql("SELECT A:\"bar\" FROM T").ok("SELECT GET_PATH(`A`, 'bar')\nFROM `T`");
    sql("SELECT T.A:\"bar\" FROM T").ok("SELECT GET_PATH(`T`.`A`, 'bar')\nFROM `T`");
    sql("SELECT parse_json(A):\"bar\" FROM T")
        .ok("SELECT GET_PATH(`PARSE_JSON`(`A`), 'bar')\nFROM `T`");

    // Multiple identifiers separated by dots (+ weird capitalization)
    sql("SELECT A:Fizz.bUzz FROM T").ok("SELECT GET_PATH(`A`, 'Fizz.bUzz')\nFROM `T`");
    sql("SELECT T.A:Fizz.bUzz FROM T").ok("SELECT GET_PATH(`T`.`A`, 'Fizz.bUzz')\nFROM `T`");
    sql("SELECT parse_json(A):Fizz.bUzz FROM T")
        .ok("SELECT GET_PATH(`PARSE_JSON`(`A`), 'Fizz.bUzz')\nFROM `T`");

    // Multiple identifiers separated by colons
    sql("SELECT A:x:y FROM T").ok("SELECT GET_PATH(`A`, 'x.y')\nFROM `T`");
    sql("SELECT T.A:x:y FROM T").ok("SELECT GET_PATH(`T`.`A`, 'x.y')\nFROM `T`");
    sql("SELECT parse_json(A):x:y FROM T")
        .ok("SELECT GET_PATH(`PARSE_JSON`(`A`), 'x.y')\nFROM `T`");

    // Multiple identifiers (quoted and unquoted) separated by colons and dots
    sql("SELECT A:\"A\".B:\"select\".D FROM T")
        .ok("SELECT GET_PATH(`A`, 'A.B.select.D')\nFROM `T`");
    sql("SELECT T.A:\"A\".B:\"C\".D FROM T").ok("SELECT GET_PATH(`T`.`A`, 'A.B.C.D')\nFROM `T`");
    sql("SELECT parse_json(A):\"A\".B:\"select\".D FROM T")
        .ok("SELECT GET_PATH(`PARSE_JSON`(`A`), 'A.B.select.D')\nFROM `T`");

    // Testing correct application of precedence
    sql("SELECT A+B:YAY FROM T").ok("SELECT (`A` + GET_PATH(`B`, 'YAY'))\nFROM `T`");
    sql("SELECT * FROM T WHERE A:action LIKE '%sing'")
        .ok("SELECT *\nFROM `T`\nWHERE (GET_PATH(`A`, 'action') LIKE '%sing')");
    sql("SELECT A:B+C:D FROM T").ok("SELECT (GET_PATH(`A`, 'B') + GET_PATH(`C`, 'D'))\nFROM `T`");

    // Field names that are also non-reserved keywords
    sql("SELECT A:rank FROM T").ok("SELECT GET_PATH(`A`, 'rank')\nFROM `T`");
    sql("SELECT A:Epoch FROM T").ok("SELECT GET_PATH(`A`, 'Epoch')\nFROM `T`");
    sql("SELECT A:size FROM T").ok("SELECT GET_PATH(`A`, 'size')\nFROM `T`");
    sql("SELECT A:tImE FROM T").ok("SELECT GET_PATH(`A`, 'tImE')\nFROM `T`");
    sql("SELECT A:text.hex FROM T").ok("SELECT GET_PATH(`A`, 'text.hex')\nFROM `T`");
    sql("SELECT A:day.width_bucket.insert FROM T")
        .ok("SELECT GET_PATH(`A`, 'day.width_bucket.insert')\nFROM `T`");

    // Strange case: following the END keyword of a CASE statement
    sql("SELECT CASE WHEN T.COND THEN parse_json(T.S1) ELSE parse_json(T.S2) END:key\n FROM T")
        .ok(
            "SELECT GET_PATH((CASE WHEN `T`.`COND` THEN `PARSE_JSON`(`T`.`S1`) ELSE"
                + " `PARSE_JSON`(`T`.`S2`) END), 'key')\n"
                + "FROM `T`");

    // Following array access
    sql("SELECT A[0]:foo FROM T").ok("SELECT GET_PATH(`A`[0], 'foo')\nFROM `T`");

    // Following cast to object
    sql("SELECT A::object:foo FROM T").ok("SELECT GET_PATH(`A` :: `OBJECT`, 'foo')\nFROM `T`");
  }

  /** Test the basic snowflake examples for lateral. */
  @Test
  void testLateral() {
    sql("SELECT * \n"
            + "    FROM departments AS d, LATERAL (SELECT * FROM employees AS e WHERE"
            + " e.department_ID = d.department_ID) AS iv2\n"
            + "    ORDER BY employee_ID")
        .ok(
            "SELECT *\n"
                + "FROM `DEPARTMENTS` AS `D`,\n"
                + "LATERAL (SELECT *\n"
                + "FROM `EMPLOYEES` AS `E`\n"
                + "WHERE (`E`.`DEPARTMENT_ID` = `D`.`DEPARTMENT_ID`)) AS `IV2`\n"
                + "ORDER BY `EMPLOYEE_ID`");
    sql("SELECT * \n"
            + "    FROM departments AS d INNER JOIN LATERAL (SELECT * FROM employees AS e WHERE"
            + " e.department_ID = d.department_ID) AS iv2\n"
            + "    ORDER BY employee_ID")
        .ok(
            "SELECT *\n"
                + "FROM `DEPARTMENTS` AS `D`\n"
                + "INNER JOIN LATERAL (SELECT *\n"
                + "FROM `EMPLOYEES` AS `E`\n"
                // Note: We add ON=TRUE in the parser, which is implicit for Snowflake inner
                // joins.
                // See the discussion of the condition
                // https://docs.snowflake.com/en/sql-reference/constructs/join#syntax
                + "WHERE (`E`.`DEPARTMENT_ID` = `D`.`DEPARTMENT_ID`)) AS `IV2` ON TRUE\n"
                + "ORDER BY `EMPLOYEE_ID`");
  }

  // Bodo Change: We update this join with no ON
  // to always output a default ON TRUE.
  @Test
  void testFullOuterJoin() {
    this.sql("select * from a full outer join b").ok("SELECT *\nFROM `A`\nFULL JOIN `B` ON TRUE");
  }

  // Bodo Change: We update these joins with no ON
  // to always output a default ON TRUE.
  @Test
  void testTableHintsInQuery() {
    final String hint = "/*+ PROPERTIES(K1 ='v1', K2 ='v2'), INDEX(IDX0, IDX1) */";
    final String sql1 = String.format(Locale.ROOT, "select * from t %s", hint);
    final String expected1 =
        "SELECT *\n"
            + "FROM `T`\n"
            + "/*+ `PROPERTIES`(`K1` = 'v1', `K2` = 'v2'), `INDEX`(`IDX0`, `IDX1`) */";
    sql(sql1).ok(expected1);
    final String sql2 =
        String.format(
            Locale.ROOT,
            "select * from\n" + "(select * from t %s union all select * from t %s )",
            hint,
            hint);
    final String expected2 =
        "SELECT *\n"
            + "FROM (SELECT *\n"
            + "FROM `T`\n"
            + "/*+ `PROPERTIES`(`K1` = 'v1', `K2` = 'v2'), `INDEX`(`IDX0`, `IDX1`) */\n"
            + "UNION ALL\n"
            + "SELECT *\n"
            + "FROM `T`\n"
            + "/*+ `PROPERTIES`(`K1` = 'v1', `K2` = 'v2'), `INDEX`(`IDX0`, `IDX1`) */)";
    sql(sql2).ok(expected2);
    final String sql3 = String.format(Locale.ROOT, "select * from t %s join t %s", hint, hint);
    final String expected3 =
        "SELECT *\n"
            + "FROM `T`\n"
            + "/*+ `PROPERTIES`(`K1` = 'v1', `K2` = 'v2'), `INDEX`(`IDX0`, `IDX1`) */\n"
            + "INNER JOIN `T`\n"
            + "/*+ `PROPERTIES`(`K1` = 'v1', `K2` = 'v2'), `INDEX`(`IDX0`, `IDX1`) */ ON TRUE";
    sql(sql3).ok(expected3);
  }

  // Bodo Change: We update this join with no ON
  // to always output a default ON TRUE.
  @Test
  void testQueryInFrom() {
    this.sql("select * from (select * from emp) as e join (select * from dept) d")
        .ok(
            "SELECT *\n"
                + "FROM (SELECT *\n"
                + "FROM `EMP`) AS `E`\n"
                + "INNER JOIN (SELECT *\n"
                + "FROM `DEPT`) AS `D` ON TRUE");
  }

  @Test
  void testObjectLiteral() {
    // 2 key-value pairs, both columns
    sql("select {'K_A': a, 'K_B': b} from t")
        .ok("SELECT OBJECT_CONSTRUCT('K_A', `A`, 'K_B', `B`)\nFROM `T`");
    // 1 key-value pair, as a literal
    sql("select {'id': 42} as J from t").ok("SELECT OBJECT_CONSTRUCT('id', 42) AS `J`\nFROM `T`");
    // 0 key-value pairs, inside of a case statement
    sql("select array_unique_agg(case when filters is not null and filters <> {} then filters end)"
            + " from t")
        .ok(
            "SELECT `ARRAY_UNIQUE_AGG`((CASE WHEN ((`FILTERS` IS NOT NULL) AND (`FILTERS` <>"
                + " OBJECT_CONSTRUCT())) THEN `FILTERS` ELSE NULL END))\n"
                + "FROM `T`");
    // Nested object and array literals
    sql("select [{'A': [{'D': 0, 'E': 1}]}, {}, {'B': [], 'C': [{'F': 6, 'G': 7, 'H': 9}, {'I':"
            + " 10}]}] as J from t")
        .ok(
            "SELECT ARRAY_CONSTRUCT(OBJECT_CONSTRUCT('A', ARRAY_CONSTRUCT(OBJECT_CONSTRUCT('D', 0,"
                + " 'E', 1))), OBJECT_CONSTRUCT(), OBJECT_CONSTRUCT('B', ARRAY_CONSTRUCT(), 'C',"
                + " ARRAY_CONSTRUCT(OBJECT_CONSTRUCT('F', 6, 'G', 7, 'H', 9),"
                + " OBJECT_CONSTRUCT('I', 10)))) AS `J`\n"
                + "FROM `T`");
  }

  @Test
  void testFlattenParse() {
    /** Test basic support for Flatten from the Snowflake docs. */
    this.sql("SELECT * FROM TABLE(FLATTEN(input => parse_json('[1, ,77]'))) f")
        .ok("SELECT *\n" + "FROM TABLE(`FLATTEN`(`INPUT` => `PARSE_JSON`('[1, ,77]'))) AS `F`");
  }

  @Test
  void testLateralFlattenParse() {
    /** Test basic support for Lateral Flatten from the Snowflake docs. */
    this.sql(
            "SELECT emp.employee_ID, emp.last_name, index, value AS project_name\n"
                + "    FROM employees AS emp, LATERAL FLATTEN(INPUT => emp.project_names) AS"
                + " proj_names\n"
                + "    ORDER BY employee_ID")
        .ok(
            "SELECT `EMP`.`EMPLOYEE_ID`, `EMP`.`LAST_NAME`, `INDEX`, `VALUE` AS `PROJECT_NAME`\n"
                + "FROM `EMPLOYEES` AS `EMP`,\n"
                + "LATERAL TABLE(FLATTEN(`INPUT` => `EMP`.`PROJECT_NAMES`)) AS `PROJ_NAMES`\n"
                + "ORDER BY `EMPLOYEE_ID`");
  }

  @Test
  void lateralFlattenMixedCallingConventions() {
    /**
     * Verify that when we mix calling conventions for flatten the positional arguments are
     * transformed to named arguments for validation.
     */
    sql("SELECT emp.empno, proj_names.*\n"
            + "    FROM emp, LATERAL FLATTEN(TO_ARRAY(emp.ename), OUTER => true) AS"
            + " proj_names\n"
            + "    ORDER BY empno")
        .ok(
            "SELECT `EMP`.`EMPNO`, `PROJ_NAMES`.*\n"
                + "FROM `EMP`,\n"
                + "LATERAL TABLE(FLATTEN(`INPUT` => `TO_ARRAY`(`EMP`.`ENAME`), `OUTER` => TRUE))"
                + " AS `PROJ_NAMES`\n"
                + "ORDER BY `EMPNO`");
  }

  @Test
  void testArrayGetParse() {
    /** Test basic support for accessing array items via the indexing operator []. */
    this.sql("SELECT ARRAY_CONSTRUCT(emp.employee_ID, 1, 2)[0]\n" + "    FROM employees")
        .ok("SELECT `ARRAY_CONSTRUCT`(`EMP`.`EMPLOYEE_ID`, 1, 2)[0]\n" + "FROM `EMPLOYEES`");
  }

  @Test
  void testArrayGetNonConstantExpr() {
    /** Test array indexing with a non-constant index. */
    this.sql(
            "SELECT ARRAY_CONSTRUCT(emp.employee_ID, 1, 2)[emp.employee_ID * 10 + 2]\n"
                + "    FROM employees")
        .ok(
            "SELECT `ARRAY_CONSTRUCT`(`EMP`.`EMPLOYEE_ID`, 1, 2)[((`EMP`.`EMPLOYEE_ID` * 10) +"
                + " 2)]\n"
                + "FROM `EMPLOYEES`");
  }

  @Test
  void testArrayGetRepeated() {
    /** Tests indexing into multi-level arrays, */
    this.sql(
            "SELECT ARRAY_CONSTRUCT(ARRAY_CONSTRUCT(emp.employee_ID))[0][0] AS project_name\n"
                + "    FROM employees")
        .ok(
            "SELECT `ARRAY_CONSTRUCT`(`ARRAY_CONSTRUCT`(`EMP`.`EMPLOYEE_ID`))[0][0] AS"
                + " `PROJECT_NAME`\n"
                + "FROM `EMPLOYEES`");
  }

  @Test
  void testArrayGetSubexpressionParse() {
    /** Tests indexing into arrays, when used as an subexpression input to other functions */
    this.sql(
            "SELECT ARRAY_CONSTRUCT("
                + "ARRAY_CONSTRUCT(emp.employee_ID, 1, 2)[0], "
                + "ARRAY_CONSTRUCT(ARRAY_CONSTRUCT(emp.employee_ID), ARRAY_CONSTRUCT(1))[1][0]"
                + ")[1] * 10 + 2 \n"
                + "    FROM employees")
        .ok(
            "SELECT ((`ARRAY_CONSTRUCT`(`ARRAY_CONSTRUCT`(`EMP`.`EMPLOYEE_ID`, 1, 2)[0],"
                + " `ARRAY_CONSTRUCT`(`ARRAY_CONSTRUCT`(`EMP`.`EMPLOYEE_ID`),"
                + " `ARRAY_CONSTRUCT`(1))[1][0])[1] * 10) + 2)\n"
                + "FROM `EMPLOYEES`");
  }

  @Test
  void testArrayGetIdxIsSubexpressionParse() {
    /**
     * Tests indexing into arrays, when the index itself is a subexpression containing another get
     */
    this.sql(
            "SELECT ARRAY_CONSTRUCT(ARRAY_CONSTRUCT(emp.employee_ID, 1, 2)[ARRAY_CONSTRUCT(2)[0]],"
                + " ARRAY_CONSTRUCT(ARRAY_CONSTRUCT(emp.employee_ID),"
                + " ARRAY_CONSTRUCT(1))[ARRAY_CONSTRUCT(ARRAY_CONSTRUCT(1))[0][0]][0], "
                + "ARRAY_CONSTRUCT(10)[ARRAY_CONSTRUCT(0)[ARRAY_CONSTRUCT(0)[ARRAY_CONSTRUCT(0)[ARRAY_CONSTRUCT(0)[ARRAY_CONSTRUCT(0)[0]]]]]])[1]"
                + " * 10 + 2 \n"
                + "    FROM employees")
        .ok(
            "SELECT ((`ARRAY_CONSTRUCT`(`ARRAY_CONSTRUCT`(`EMP`.`EMPLOYEE_ID`, 1,"
                + " 2)[`ARRAY_CONSTRUCT`(2)[0]],"
                + " `ARRAY_CONSTRUCT`(`ARRAY_CONSTRUCT`(`EMP`.`EMPLOYEE_ID`),"
                + " `ARRAY_CONSTRUCT`(1))[`ARRAY_CONSTRUCT`(`ARRAY_CONSTRUCT`(1))[0][0]][0],"
                + " `ARRAY_CONSTRUCT`(10)[`ARRAY_CONSTRUCT`(0)[`ARRAY_CONSTRUCT`(0)[`ARRAY_CONSTRUCT`(0)[`ARRAY_CONSTRUCT`(0)[`ARRAY_CONSTRUCT`(0)[0]]]]]])[1]"
                + " * 10) + 2)\n"
                + "FROM `EMPLOYEES`");
  }

  /**
   * Utility function for datePart that is used for testing each operator.
   *
   * <p>As an example if we were testing day, then values would be an array of valid day
   * representations like {"DAY", "d"} funcName would be DAYOFMONTH because everything maps to that
   * function. and isDatePartFunc would be True. Most functions will set SqlDatePartFunction, but
   * for example YEAROFWEEK does not. For more information look at the actual function declarations
   * in Bodo/Calcite.
   *
   * @param funcName The function all parts create.
   * @param isDatePartFunc Is the SQLFunction created a SqlDatePartFunction?
   */
  void baseDatePartTest(String funcName, Boolean isDatePartFunc) {
    String baseLiteralQuery = "Select DATE_PART(%s, '2013-05-08'::TIMESTAMP)";
    String baseStringQuery = "Select DATE_PART('%s', '2013-05-08'::TIMESTAMP)";
    String baseOutput;
    // Implementations that dispatch to DatePart have extra parentheses
    if (isDatePartFunc) {
      baseOutput = "SELECT (%s('2013-05-08' :: TIMESTAMP(9)))";
    } else {
      baseOutput = "SELECT %s('2013-05-08' :: TIMESTAMP(9))";
    }
    for (String value : timeUnitTestCases.get(funcName)) {
      String literalQuery = String.format(Locale.ROOT, baseLiteralQuery, value);
      String stringQuery = String.format(Locale.ROOT, baseStringQuery, value);
      String output = String.format(Locale.ROOT, baseOutput, funcName);
      this.sql(literalQuery).ok(output);
      this.sql(stringQuery).ok(output);
    }
  }

  // Test DATE_PART on all funcations that are SqlDatePartFunctions
  @ParameterizedTest
  @ValueSource(
      strings = {
        "YEAR",
        "MONTH",
        "DAYOFMONTH",
        "DAYOFWEEK",
        "DAYOFWEEKISO",
        "DAYOFYEAR",
        "WEEK",
        "WEEKISO",
        "QUARTER",
        "HOUR",
        "MINUTE",
        "SECOND",
        "NANOSECOND"
      })
  void testDatePart(String unit) {
    baseDatePartTest(unit, true);
  }

  // Test DATE_PART on all funcations that are NOT SqlDatePartFunctions
  @ParameterizedTest
  @ValueSource(
      strings = {
        "YEAROFWEEK",
        "YEAROFWEEKISO",
        "EPOCH_SECOND",
        "EPOCH_MILLISECOND",
        "EPOCH_MICROSECOND",
        "EPOCH_NANOSECOND",
        "TIMEZONE_HOUR",
        "TIMEZONE_MINUTE"
      })
  void testDatePartNonDatePartFuncs(String unit) {
    baseDatePartTest(unit, false);
  }

  // Test LAST_DAY on all supported units
  @ParameterizedTest
  @ValueSource(strings = {"YEAR", "MONTH", "WEEK", "QUARTER"})
  void testLastDay(String unit) {
    String baseLiteralQuery = "Select LAST_DAY('2013-05-08'::TIMESTAMP, %s)";
    String baseStringQuery = "Select LAST_DAY('2013-05-08'::TIMESTAMP, '%s')";
    String baseOutput = "SELECT LAST_DAY('2013-05-08' :: TIMESTAMP(9), %s)";
    for (String value : timeUnitTestCases.get(unit)) {
      String literalQuery = String.format(Locale.ROOT, baseLiteralQuery, value);
      String stringQuery = String.format(Locale.ROOT, baseStringQuery, value);
      String literalOutput = String.format(Locale.ROOT, baseOutput, unit);
      String stringOutput = String.format(Locale.ROOT, baseOutput, unit);
      this.sql(literalQuery).ok(literalOutput);
      this.sql(stringQuery).ok(stringOutput);
    }
  }

  /**
   * Utility function for temporal add/diff functions that is used for testing each operator. See
   * baseDatePartTest.
   *
   * @param func function to test.
   * @param unit The function all parts create.
   */
  void baseTemporalAddDiffTest(String func, String unit) {
    String baseLiteralQuery = "Select %s(%s, 1, '2013-05-08'::TIMESTAMP)";
    String baseStringQuery = "Select %s('%s', 1, '2013-05-08'::TIMESTAMP)";
    String baseOutput = "SELECT %s(%s, 1, '2013-05-08' :: TIMESTAMP(9))";
    for (String value : timeUnitTestCases.get(unit)) {
      String literalQuery = String.format(Locale.ROOT, baseLiteralQuery, func, value);
      String stringQuery = String.format(Locale.ROOT, baseStringQuery, func, value);
      String literalOutput = String.format(Locale.ROOT, baseOutput, func, unit);
      String stringOutput = String.format(Locale.ROOT, baseOutput, func, unit);
      this.sql(literalQuery).ok(literalOutput);
      this.sql(stringQuery).ok(stringOutput);
    }
  }

  // Test DATEADD/TIMEADD/TIMESTAMPADD an all supported units
  @ParameterizedTest
  @ValueSource(
      strings = {
        "YEAR",
        "MONTH",
        "DAY",
        "WEEK",
        "QUARTER",
        "HOUR",
        "MINUTE",
        "SECOND",
        "MILLISECOND",
        "MICROSECOND",
        "NANOSECOND"
      })
  void testTemporalAdd(String unit) {
    baseTemporalAddDiffTest("DATEADD", unit);
    baseTemporalAddDiffTest("TIMEADD", unit);
    baseTemporalAddDiffTest("TIMESTAMPADD", unit);
  }

  // Test DATEDIFF/TIMEDIFF/TIMESTAMPDIFF an all supported units
  @ParameterizedTest
  @ValueSource(
      strings = {
        "YEAR",
        "MONTH",
        "DAY",
        "WEEK",
        "QUARTER",
        "HOUR",
        "MINUTE",
        "SECOND",
        "MILLISECOND",
        "MICROSECOND",
        "NANOSECOND"
      })
  void testDateDiff(String unit) {
    List<String> values = timeUnitTestCases.get(unit);
    baseTemporalAddDiffTest("DATEDIFF", unit);
    baseTemporalAddDiffTest("TIMEDIFF", unit);
    baseTemporalAddDiffTest("TIMESTAMPDIFF", unit);
  }

  // Test DATEDTRUNC an all supported units
  @ParameterizedTest
  @ValueSource(
      strings = {
        "YEAR",
        "MONTH",
        "DAY",
        "WEEK",
        "QUARTER",
        "HOUR",
        "MINUTE",
        "SECOND",
        "MILLISECOND",
        "MICROSECOND",
        "NANOSECOND"
      })
  void testDateTrunc(String unit) {
    String baseLiteralQuery = "Select DATE_TRUNC(%s, '2013-05-08'::TIMESTAMP)";
    String baseStringQuery = "Select DATE_TRUNC('%s', '2013-05-08'::TIMESTAMP)";
    String baseDoubleQuotedStringQuery = "Select DATE_TRUNC(\"%s\", '2013-05-08'::TIMESTAMP)";
    String baseOutput = "SELECT DATE_TRUNC(%s, '2013-05-08' :: TIMESTAMP(9))";
    for (String value : timeUnitTestCases.get(unit)) {
      String literalQuery = String.format(Locale.ROOT, baseLiteralQuery, value);
      String stringQuery = String.format(Locale.ROOT, baseStringQuery, value);
      String doubleQuotedStringQuery =
          String.format(Locale.ROOT, baseDoubleQuotedStringQuery, value);
      String literalOutput = String.format(Locale.ROOT, baseOutput, unit);
      String stringOutput = String.format(Locale.ROOT, baseOutput, unit);
      this.sql(literalQuery).ok(literalOutput);
      this.sql(stringQuery).ok(stringOutput);
      this.sql(doubleQuotedStringQuery).ok(stringOutput);
    }
  }

  @Test
  void testTimestampTypeExpandedSyntax() {
    String query, output;

    // TODO: always use WITH TIME ZONE once enableTimestampTz is removed
    query = "SELECT TO_CHAR(CURRENT_DATE::TIMESTAMP WITH TIME ZONE, 'YYYYMMDD'::text)";
    if (RelationalAlgebraGenerator.enableTimestampTz) {
      output =
          "SELECT `TO_CHAR`(CURRENT_DATE :: TIMESTAMP(9) WITH TIME ZONE, 'YYYYMMDD' ::"
              + " VARCHAR)";
    } else {
      output = "SELECT `TO_CHAR`(CURRENT_DATE :: TIMESTAMP_LTZ(9), 'YYYYMMDD' ::" + " VARCHAR)";
    }
    sql(query).ok(output);

    query = "SELECT TO_CHAR(CURRENT_DATE::TIMESTAMP WITH LOCAL TIME ZONE, 'YYYYMMDD'::text)";
    output = "SELECT `TO_CHAR`(CURRENT_DATE :: TIMESTAMP_LTZ(9), 'YYYYMMDD' ::" + " VARCHAR)";
    sql(query).ok(output);

    query = "SELECT TO_CHAR(CURRENT_DATE::TIMESTAMP WITHOUT TIME ZONE, 'YYYYMMDD'::text)";
    output = "SELECT `TO_CHAR`(CURRENT_DATE :: TIMESTAMP(9), 'YYYYMMDD' :: VARCHAR)";
    sql(query).ok(output);
  }

  // Bodo Change: Default Time/Timestamp precision is 9 not -1
  @Test
  public void testCast() {
    expr("cast(x as boolean)").ok("CAST(`X` AS BOOLEAN)");
    expr("cast(x as integer)").ok("CAST(`X` AS INTEGER)");
    expr("cast(x as varchar(1))").ok("CAST(`X` AS VARCHAR(1))");
    expr("cast(x as date)").ok("CAST(`X` AS DATE)");
    expr("cast(x as time)").ok("CAST(`X` AS TIME(9))");
    expr("cast(x as time without time zone)").ok("CAST(`X` AS TIME(9))");
    expr("cast(x as timestamp without time zone)").ok("CAST(`X` AS TIMESTAMP(9))");
    expr("cast(x as timestamp with local time zone)").ok("CAST(`X` AS TIMESTAMP_LTZ(9))");
    expr("cast(x as time(0))").ok("CAST(`X` AS TIME(0))");
    expr("cast(x as time(0) without time zone)").ok("CAST(`X` AS TIME(0))");
    expr("cast(x as timestamp(0))").ok("CAST(`X` AS TIMESTAMP(0))");
    expr("cast(x as timestamp(0) without time zone)").ok("CAST(`X` AS TIMESTAMP(0))");
    expr("cast(x as timestamp(0) with local time zone)").ok("CAST(`X` AS TIMESTAMP_LTZ(0))");
    // TODO: always use WITH TIME ZONE once enableTimestampTz is removed
    if (RelationalAlgebraGenerator.enableTimestampTz) {
      expr("cast(x as timestamp(0) with time zone)").ok("CAST(`X` AS TIMESTAMP(0) WITH TIME ZONE)");
    } else {
      expr("cast(x as timestamp(0) with time zone)").ok("CAST(`X` AS TIMESTAMP_LTZ(0))");
    }
    expr("cast(x as timestamp)").ok("CAST(`X` AS TIMESTAMP(9))");
    expr("cast(x as decimal(1,1))").ok("CAST(`X` AS DECIMAL(1, 1))");
    expr("cast(x as char(1))").ok("CAST(`X` AS CHAR(1))");
    expr("cast(x as binary(1))").ok("CAST(`X` AS BINARY(1))");
    expr("cast(x as varbinary(1))").ok("CAST(`X` AS VARBINARY(1))");
    expr("cast(x as tinyint)").ok("CAST(`X` AS TINYINT)");
    expr("cast(x as smallint)").ok("CAST(`X` AS SMALLINT)");
    expr("cast(x as bigint)").ok("CAST(`X` AS BIGINT)");
    expr("cast(x as real)").ok("CAST(`X` AS REAL)");
    expr("cast(x as double)").ok("CAST(`X` AS DOUBLE)");
    expr("cast(x as decimal)").ok("CAST(`X` AS DECIMAL)");
    expr("cast(x as decimal(0))").ok("CAST(`X` AS DECIMAL(0))");
    expr("cast(x as decimal(1,2))").ok("CAST(`X` AS DECIMAL(1, 2))");

    expr("cast('foo' as bar)").ok("CAST('foo' AS `BAR`)");
  }

  // Bodo Change: Default Timestamp precision is 9 not -1
  @Test
  void testCastAsRowType() {
    expr("cast(a as row(f0 int, f1 varchar))").ok("CAST(`A` AS ROW(`F0` INTEGER, `F1` VARCHAR))");
    expr("cast(a as row(f0 int not null, f1 varchar null))")
        .ok("CAST(`A` AS ROW(`F0` INTEGER, `F1` VARCHAR NULL))");
    // test nested row type.
    expr("cast(a as row("
            + "f0 row(ff0 int not null, ff1 varchar null) null, "
            + "f1 timestamp not null))")
        .ok(
            "CAST(`A` AS ROW("
                + "`F0` ROW(`FF0` INTEGER, `FF1` VARCHAR NULL) NULL, "
                + "`F1` TIMESTAMP(9)))");
    // test row type in collection data types.
    expr("cast(a as row(f0 bigint not null, f1 decimal null) array)")
        .ok("CAST(`A` AS ROW(`F0` BIGINT, `F1` DECIMAL NULL) ARRAY)");
    expr("cast(a as row(f0 varchar not null, f1 timestamp null) multiset)")
        .ok("CAST(`A` AS ROW(`F0` VARCHAR, `F1` TIMESTAMP(9) NULL) MULTISET)");
  }

  @Test
  void testCastFails() {
    // "with time zone" is invalid in Calcite, but valid for BodoSQL
    // expr("cast(x as time with ^time^ zone)")
    //         .fails("(?s).*Encountered \"time\" at .*");
    // expr("cast(x as time(0) with ^time^ zone)")
    //        .fails("(?s).*Encountered \"time\" at .*");
    // expr("cast(x as timestamp with ^time^ zone)")
    //         .fails("(?s).*Encountered \"time\" at .*");
    // expr("cast(x as timestamp(0) with ^time^ zone)")
    //        .fails("(?s).*Encountered \"time\" at .*");
    expr("cast(x as varchar(10) ^with^ local time zone)")
        .fails("(?s).*Encountered \"with\" at line 1, column 23.\n.*");
    expr("cast(x as varchar(10) ^without^ time zone)")
        .fails("(?s).*Encountered \"without\" at line 1, column 23.\n.*");
  }

  @Test
  void testNullIf() {
    // We don't use the default parsing for NULLIF, and extend support to allow for variants
    expr("NULLIF(1, 1)").same();
    expr("NULLIF(1, TO_VARIANT(1))").ok("NULLIF(1, `TO_VARIANT`(1))");
    expr("NULLIF(TO_VARIANT(1), 1)").ok("NULLIF(`TO_VARIANT`(1), 1)");
    expr("NULLIF(TO_VARIANT(1), TO_VARIANT(1))").ok("NULLIF(`TO_VARIANT`(1), `TO_VARIANT`(1))");
  }

  @Test
  void testFormatClauseInCast() {
    this.expr("cast(date '2001-01-01' as varchar FORMAT 'YYYY Q MM')")
        .ok("CAST(DATE '2001-01-01' AS VARCHAR FORMAT 'YYYY Q MM')");
    this.expr("cast(time '1:30:00' as varchar format 'HH24')")
        .ok("CAST(TIME '1:30:00' AS VARCHAR FORMAT 'HH24')");
    this.expr("cast(timestamp '2008-12-25 12:15:00' as varchar format 'MON, YYYY')")
        .ok("CAST(TIMESTAMP '2008-12-25 12:15:00' AS VARCHAR FORMAT 'MON, YYYY')");
    this.expr("cast('18-12-03' as date format 'YY-MM-DD')")
        .ok("CAST('18-12-03' AS DATE FORMAT 'YY-MM-DD')");
    this.expr("cast('01:05:07.16' as time format 'HH24:MI:SS.FF4')")
        .ok("CAST('01:05:07.16' AS TIME(9) FORMAT 'HH24:MI:SS.FF4')");
    this.expr("cast('2020.06.03 12:42:53' as timestamp format 'YYYY.MM.DD HH:MI:SS')")
        .ok("CAST('2020.06.03 12:42:53' AS TIMESTAMP(9) FORMAT 'YYYY.MM.DD HH:MI:SS')");
  }

  @Disabled
  @Test
  void testStringLiteralDoubleQuoted() {}

  @Disabled
  @Test
  void testUnparseableIntervalQualifiers() {}

  @Disabled
  @Test
  void testRlike() {}

  @Disabled
  @Test
  @Override
  protected void testHoist() {}

  @Disabled
  @Test
  void testReverseSolidus() {}

  @Disabled
  @Test
  void testParseWithReader() {}

  @Disabled
  @Test
  void testSubstring() {}

  // These tests are disabled because they relate to Calcite support for JSON
  // functionality
  // that we do not support and/or differs from Snowflake syntax in ways that are
  // difficult
  // to reconcile
  @Disabled
  @Test
  void testJsonValueExpressionOperator() {}

  @Disabled
  @Test
  void testJsonExists() {}

  @Disabled
  @Test
  void testJsonValue() {}

  @Disabled
  @Test
  void testJsonType() {}

  @Disabled
  @Test
  void testJsonDepth() {}

  @Disabled
  @Test
  void testJsonLength() {}

  @Disabled
  @Test
  void testJsonKeys() {}

  @Disabled
  @Test
  void testJsonRemove() {}

  @Disabled
  @Test
  void testJsonObjectAgg() {}

  @Disabled
  @Test
  void testJsonArray() {}

  @Disabled
  @Test
  void testJsonPretty() {}

  @Disabled
  @Test
  void testJsonStorageSize() {}

  @Disabled
  @Test
  void testJsonArrayAgg1() {}

  @Disabled
  @Test
  void testJsonArrayAgg2() {}

  @Disabled
  @Test
  void testJsonPredicate() {}

  @Disabled
  @Test
  void testUnparseableIntervalQualifiers2() {}

  // Unreserving rollup as a keyword result in this test not passing
  @Disabled
  @Test
  void testGroupByRollup() {}

  // This test is checking words as keywords that Bodo now unreserves to match
  // Snowflake
  @Disabled
  @Test
  protected void testMetadata() {}

  // This test is testing cast(x as interval year) which is not supported in
  // Snowflake
  // and this now fails after unreserving `interval`
  @Disabled
  @Test
  void testCastToInterval() {}

  // This test is disabled because it uses FLOOR TO UNIT, which is not supported
  // in Snowflake/Bodo
  @Disabled
  @Test
  protected void testTimeUnitCodes() {}

  // This test is disabled because "describe X" isn't supported if X is both a
  // table alias and a keyword. In our example S is a non-reserved keyword
  @Disabled
  @Test
  void testDescribeTable() {}

  // Includes Unit names we don't support.
  @Disabled
  @Test
  void testTimestampAdd() {}

  // Includes Unit names we don't support.
  @Disabled
  @Test
  void testTimestampDiff() {}

  // Includes Unit names we don't support.
  @Disabled
  @Test
  void testTimeTrunc() {}

  // Includes Unit names we don't support.
  @Disabled
  @Test
  void testTimestampTrunc() {}

  // MsSqL only function that hits a gap in our parser.
  @Disabled
  @Test
  void testMssqlConvert() {}

  // We support the functionality that produces an "error" in this test.
  @Test
  void testFromValuesWithoutParens() {}

  @Test
  void testCurrentDatabase() {
    this.sql("select current_database()").ok("SELECT `CURRENT_DATABASE`()");
  }

  @Test
  void testCurrentAccount() {
    this.sql("select current_account()").ok("SELECT `CURRENT_ACCOUNT`()");
    this.sql("select current_account_name()").ok("SELECT `CURRENT_ACCOUNT_NAME`()");
  }

  /**
   * Bodo's parser's global {@code LOOKAHEAD} is larger than the core parser's. This causes
   * different parse error message between these two parsers. Here we define a looser error checker
   * for Bodo, so that we can reuse failure testing codes from {@link SqlParserTest}.
   *
   * <p>If a test case is written in this file -- that is, not inherited -- it is still checked by
   * {@link SqlParserTest}'s checker.
   */
  public static class BodoTesterImpl extends SqlParserTest.TesterImpl {
    @Override
    protected void checkEx(
        String expectedMsgPattern, StringAndPos sap, @Nullable Throwable thrown) {
      if (thrown != null && thrownByBodoTest(thrown)) {
        super.checkEx(expectedMsgPattern, sap, thrown);
      } else {
        checkExNotNull(sap, thrown);
      }
    }

    private boolean thrownByBodoTest(Throwable ex) {
      Throwable rootCause = Throwables.getRootCause(ex);
      StackTraceElement[] stackTrace = rootCause.getStackTrace();
      for (StackTraceElement stackTraceElement : stackTrace) {
        String className = stackTraceElement.getClassName();
        if (Objects.equals(className, BodoParserTest.class.getName())) {
          return true;
        }
      }
      return false;
    }

    private void checkExNotNull(StringAndPos sap, @Nullable Throwable thrown) {
      if (thrown == null) {
        throw new AssertionError(
            "Expected query to throw exception, " + "but it did not; query [" + sap.sql + "]");
      }
    }
  }

  // Test that all interval units can be parsed with quotes
  @ParameterizedTest
  @ValueSource(
      strings = {
        "YEAR",
        "MONTH",
        "DAY",
        "WEEK",
        "QUARTER",
        "HOUR",
        "MINUTE",
        "SECOND",
        "MILLISECOND",
        "MICROSECOND",
        "NANOSECOND"
      })
  void baseIntervalTest(String unit) {
    String baseQuery = "Select getdate() - INTERVAL '%s'";
    String baseOutput = "SELECT (`GETDATE`() - INTERVAL '' %s)";
    for (String value : timeUnitTestCases.get(unit)) {
      String query = String.format(Locale.ROOT, baseQuery, value);
      String output = String.format(Locale.ROOT, baseOutput, unit);
      this.sql(query).ok(output);
    }
  }

  // Test that all interval units can be parsed without quotes
  @ParameterizedTest
  @ValueSource(
      strings = {
        "YEAR",
        "MONTH",
        "DAY",
        "WEEK",
        "QUARTER",
        "HOUR",
        "MINUTE",
        "SECOND",
        "MILLISECOND",
        "MICROSECOND",
        "NANOSECOND"
      })
  void baseUnquotedIntervalTest(String unit) {
    String baseQuery = "Select GETDATE() - INTERVAL 1 %s";
    String baseOutput = "SELECT (`GETDATE`() - (INTERVAL 1 %s))";
    for (String value : timeUnitTestCases.get(unit)) {
      String query = String.format(Locale.ROOT, baseQuery, value);
      String output = String.format(Locale.ROOT, baseOutput, unit);
      this.sql(query).ok(output);
    }
  }

  // Test copied from Calcite to change the types to match our type system.
  @Test
  void testEmbeddedTime() {
    this.expr("{t '16:22:34'}").ok("TIME '16:22:34.000000000'");
  }

  // Test copied from Calcite to change the types to match our type system.
  @Test
  void testEmbeddedTimestamp() {
    this.expr("{ts '1998-10-22 16:22:34'}").ok("TIMESTAMP '1998-10-22 16:22:34.000000000'");
  }

  @ParameterizedTest
  @ValueSource(strings = {"1 year, 1 day, 1 second"})
  void baseIntervalWithComma(String expr) {
    String query = String.format(Locale.ROOT, "Select GETDATE() - INTERVAL '%s'", expr);
    String output =
        "SELECT (`GETDATE`() - COMBINE_INTERVALS(COMBINE_INTERVALS(INTERVAL '1' YEAR, INTERVAL '1'"
            + " DAY), INTERVAL '1' SECOND))";

    sql(query).ok(output);
  }

  /**
   * Test that TIMESTAMPNTZ can be parsed as a valid alias for TIMESTAMP_NTZ and is converted to a
   * standard format.
   */
  @Test
  void testTimestampNTZAlias() {
    String query1 = "SELECT 1::TIMESTAMPNTZ";
    String output1 = "SELECT 1 :: TIMESTAMP(9)";
    sql(query1).ok(output1);
    String query2 = "SELECT Cast(1 as TIMESTAMPNTZ)";
    String output2 = "SELECT CAST(1 AS TIMESTAMP(9))";
    sql(query2).ok(output2);
  }

  /**
   * Test that TIMESTAMPLTZ can be parsed as a valid alias for TIMESTAMP_LTZ and is converted to a
   * standard format.
   */
  @Test
  void testTimestampLTZAlias() {
    String query1 = "SELECT 1::TIMESTAMPLTZ";
    String output1 = "SELECT 1 :: TIMESTAMP_LTZ(9)";
    sql(query1).ok(output1);
    String query2 = "SELECT Cast(1 as TIMESTAMPLTZ)";
    String output2 = "SELECT CAST(1 AS TIMESTAMP_LTZ(9))";
    sql(query2).ok(output2);
  }

  @Test
  void testsSelectDollarSignIdentifier() {
    String query = "select METADATA$ACTION from my_table";
    String expected = "SELECT `METADATA$ACTION`\n" + "FROM `MY_TABLE`";
    sql(query).ok(expected);
  }

  @Test
  void testTrailingCommaSelect() {
    String query = "select A, B, C, from my_table";
    String expected = "SELECT `A`, `B`, `C`\n" + "FROM `MY_TABLE`";
    sql(query).ok(expected);
  }

  @Test
  void testsSelectDollarSignCompoundIdentifier() {
    String query = "select T.METADATA$ACTION from my_table T";
    String expected = "SELECT `T`.`METADATA$ACTION`\n" + "FROM `MY_TABLE` AS `T`";
    sql(query).ok(expected);
  }

  @ParameterizedTest
  @ValueSource(strings = {",", "JOIN", "LEFT JOIN", "RIGHT JOIN", "CROSS JOIN", "FULL OUTER JOIN"})
  void testImplicitLateralJoin(String joinType) {
    // Test that BodoSQL inserts an implicit lateral join where appropriate.

    String query =
        String.format(Locale.ROOT, "select * from t1 %s Table(Flatten(t1.js))", joinType);

    String output;
    // Formatting varies slightly between outputs, the simplest solution is to just hardcode the
    // responses
    switch (joinType) {
      case ",":
        output = "SELECT *\n" + "FROM `T1`,\n" + "LATERAL TABLE(`FLATTEN`(`T1`.`JS`))";
        break;
      case "JOIN":
        output =
            "SELECT *\n" + "FROM `T1`\n" + "INNER JOIN LATERAL TABLE(`FLATTEN`(`T1`.`JS`)) ON TRUE";
        break;
      case "FULL OUTER JOIN":
        output =
            "SELECT *\n" + "FROM `T1`\n" + "FULL JOIN LATERAL TABLE(`FLATTEN`(`T1`.`JS`)) ON TRUE";
        break;
      case "CROSS JOIN":
        output = "SELECT *\n" + "FROM `T1`\n" + "CROSS JOIN LATERAL TABLE(`FLATTEN`(`T1`.`JS`))";
        break;
      case "LEFT JOIN":
      case "RIGHT JOIN":
        output =
            String.format(
                Locale.ROOT,
                "SELECT *\n" + "FROM `T1`\n" + "%s LATERAL TABLE(`FLATTEN`(`T1`.`JS`)) ON TRUE",
                joinType);
        break;
      default:
        output = "Unhandled JoinType: " + joinType;
        break;
    }

    sql(query).ok(output);
  }

  /**
   * Test case for <a href="https://issues.apache.org/jira/browse/CALCITE-5997">[CALCITE-5997]
   * Modified to remove Bigquery dialect SAFE_ORDINAL is bigquery specific
   */
  @Test
  void testOffset() {
    sql("SELECT ARRAY[2,4,6][2]").ok("SELECT (ARRAY[2, 4, 6])[2]");
    sql("SELECT ARRAY[2,4,6][ORDINAL(2)]").ok("SELECT (ARRAY[2, 4, 6])[ORDINAL(2)]");
    sql("SELECT ARRAY[2,4,6][OFFSET(2)]").ok("SELECT (ARRAY[2, 4, 6])[OFFSET(2)]");
    sql("SELECT ARRAY[2,4,6][SAFE_OFFSET(2)]").ok("SELECT (ARRAY[2, 4, 6])[SAFE_OFFSET(2)]");
    sql("SELECT ARRAY[2,4,6][SAFE_ORDINAL(2)]").ok("SELECT (ARRAY[2, 4, 6])[2]");

    // All these tests work without BIG_QUERY as well.
    // The SQL parser accepts this syntax, so we need to be
    // able to unparse it into something.
    sql("SELECT ARRAY[2,4,6][ORDINAL(2)]").ok("SELECT (ARRAY[2, 4, 6])[ORDINAL(2)]");
    sql("SELECT ARRAY[2,4,6][OFFSET(2)]").ok("SELECT (ARRAY[2, 4, 6])[OFFSET(2)]");
    sql("SELECT ARRAY[2,4,6][SAFE_OFFSET(2)]").ok("SELECT (ARRAY[2, 4, 6])[SAFE_OFFSET(2)]");
    sql("SELECT ARRAY[2,4,6][SAFE_ORDINAL(2)]").ok("SELECT (ARRAY[2, 4, 6])[2]");
  }

  // Bodo Change: Disable the Oracle CONVERT function test because our parser
  // dialect does not support the Oracle-specific CONVERT syntax added in
  // Calcite 1.39 (CALCITE-6730).
  @Disabled
  @Test
  void testConvertOracle() {}
}
