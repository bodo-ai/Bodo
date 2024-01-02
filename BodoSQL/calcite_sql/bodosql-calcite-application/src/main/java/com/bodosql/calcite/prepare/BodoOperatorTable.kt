package com.bodosql.calcite.prepare

import com.bodosql.calcite.application.operatorTables.ArrayOperatorTable
import com.bodosql.calcite.application.operatorTables.CastingOperatorTable
import com.bodosql.calcite.application.operatorTables.CondOperatorTable
import com.bodosql.calcite.application.operatorTables.ContextOperatorTable
import com.bodosql.calcite.application.operatorTables.DatetimeOperatorTable
import com.bodosql.calcite.application.operatorTables.JsonOperatorTable
import com.bodosql.calcite.application.operatorTables.NumericOperatorTable
import com.bodosql.calcite.application.operatorTables.SinceEpochFnTable
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.application.operatorTables.TableFunctionOperatorTable
import com.bodosql.calcite.application.operatorTables.ThreeOperatorStringTable
import com.bodosql.calcite.sql.func.SqlBodoOperatorTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.`fun`.SqlAggOperatorTable
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.util.ChainedSqlOperatorTable

object BodoOperatorTable : ChainedSqlOperatorTable(
    ImmutableList.of(
        // NOTE: SelectOperatorTable is not included since all of those
        // operators should be expanded during validation
        SqlBodoOperatorTable.instance(),
        DatetimeOperatorTable.instance(),
        NumericOperatorTable.instance(),
        StringOperatorTable.instance(),
        JsonOperatorTable.instance(),
        CondOperatorTable.instance(),
        SinceEpochFnTable.instance(),
        ThreeOperatorStringTable.instance(),
        CastingOperatorTable.instance(),
        ArrayOperatorTable.instance(),
        ContextOperatorTable.instance(),
        TableFunctionOperatorTable.instance(),
        SqlAggOperatorTable.instance(),
        // Note: we put SqlStdOperatorTable last so we can override
        // any functions it provides.
        SqlStdOperatorTable.instance(),
    ),
)
