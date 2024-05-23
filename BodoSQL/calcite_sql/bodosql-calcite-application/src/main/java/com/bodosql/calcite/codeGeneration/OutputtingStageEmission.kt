package com.bodosql.calcite.codeGeneration

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable

class OutputtingStageEmission(
    bodyFn: (BodoPhysicalRel.BuildContext, StateVariable, BodoEngineTable?) -> BodoEngineTable,
    reportOutTableSize: Boolean,
) : StageEmission(
        bodyFn,
        reportOutTableSize,
    )
