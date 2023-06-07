package com.bodosql.calcite.prepare

import com.bodosql.calcite.application.BodoSQLOperatorTables.*
import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.util.ChainedSqlOperatorTable

object BodoOperatorTable : ChainedSqlOperatorTable(
    ImmutableList.of(
        // TODO(jsternberg): I suspect that this operator table
        // should be the last one as placing it first prevents us from
        // overloading any operators added to this.
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
    )
)
