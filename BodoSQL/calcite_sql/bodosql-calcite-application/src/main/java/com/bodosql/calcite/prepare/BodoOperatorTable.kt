package com.bodosql.calcite.prepare

import com.bodosql.calcite.application.BodoSQLOperatorTables.ArrayOperatorTable
import com.bodosql.calcite.application.BodoSQLOperatorTables.CastingOperatorTable
import com.bodosql.calcite.application.BodoSQLOperatorTables.CondOperatorTable
import com.bodosql.calcite.application.BodoSQLOperatorTables.DatetimeOperatorTable
import com.bodosql.calcite.application.BodoSQLOperatorTables.JsonOperatorTable
import com.bodosql.calcite.application.BodoSQLOperatorTables.NumericOperatorTable
import com.bodosql.calcite.application.BodoSQLOperatorTables.SinceEpochFnTable
import com.bodosql.calcite.application.BodoSQLOperatorTables.StringOperatorTable
import com.bodosql.calcite.application.BodoSQLOperatorTables.ThreeOperatorStringTable
import com.bodosql.calcite.sql.func.SqlBodoOperatorTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.util.ChainedSqlOperatorTable

object BodoOperatorTable : ChainedSqlOperatorTable(
    ImmutableList.of(
        // TODO(jsternberg): I suspect that this operator table
        // should be the last one as placing it first prevents us from
        // overloading any operators added to this.
        SqlBodoOperatorTable.instance(),
        SqlStdOperatorTable.instance(),
        DatetimeOperatorTable.instance(),
        NumericOperatorTable.instance(),
        StringOperatorTable.instance(),
        JsonOperatorTable.instance(),
        CondOperatorTable.instance(),
        SinceEpochFnTable.instance(),
        ThreeOperatorStringTable.instance(),
        CastingOperatorTable.instance(),
        ArrayOperatorTable.instance(),
    ),
)
