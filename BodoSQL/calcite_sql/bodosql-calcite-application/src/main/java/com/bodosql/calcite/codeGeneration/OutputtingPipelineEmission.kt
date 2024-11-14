package com.bodosql.calcite.codeGeneration

import org.apache.calcite.rel.RelNode

class OutputtingPipelineEmission(
    emittingStages: List<OutputtingStageEmission>,
    startPipeline: Boolean,
    child: RelNode?,
) : PipelineEmission(
        emittingStages,
        null,
        startPipeline,
        child,
    )
