package com.bodosql.calcite.codeGeneration

import org.apache.calcite.rel.RelNode

class TerminatingPipelineEmission(
    emittingStages: List<OutputtingStageEmission>,
    terminatingStage: TerminatingStageEmission,
    startPipeline: Boolean,
    child: RelNode?,
) : PipelineEmission(
        emittingStages,
        terminatingStage,
        startPipeline,
        child,
    )
