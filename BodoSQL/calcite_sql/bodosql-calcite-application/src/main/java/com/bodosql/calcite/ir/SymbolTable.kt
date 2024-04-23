package com.bodosql.calcite.ir

/**
 * Symbol table holds information about the defined variables
 * and allows programs to define new variables.
 *
 * NOTE(jsternberg): The intended use of this structure is defined
 * by the comment above, but the current use of it is just a class
 * dedicated to generating unique variables for code generation.
 * It'll likely be expanded in the future, but this is the
 * functionality that's presently needed at the time that I wrote this.
 */
class SymbolTable {
    private var dfVarId: Int = 1
    private var tableVarId: Int = 1
    private var seriesVarId: Int = 1
    private var arrVarId: Int = 1
    private var colVarId: Int = 1
    private var accVarId: Int = 1
    private var tempVarId: Int = 1
    private var streamingFlagId: Int = 1
    private var streamingWriterId: Int = 1
    private var groupByApplyFnId: Int = 1
    private var globalVarId: Int = 1
    private var idxVarId: Int = 1
    private var iterVarId: Int = 1
    private var outputControlId: Int = 1
    private var inputRequestId: Int = 1
    private var stateVarId: Int = 1
    private var funcIdCounter: Int = 1
    private var closureVarId: Int = 1

    companion object {
        private const val DUMMY_COL_NAME_BASE = "__bodo_dummy__"
    }

    fun genDfVar(): Variable {
        return Variable("df${dfVarId++}")
    }

    fun genTableVar(): Variable {
        return Variable("T${tableVarId++}")
    }

    fun genSeriesVar(): Variable {
        return Variable("S${seriesVarId++}")
    }

    fun genArrayVar(): Variable {
        return Variable("A${arrVarId++}")
    }

    fun genIndexVar(): Variable {
        return Variable("index_${idxVarId++}")
    }

    fun genIterVar(): Variable {
        return Variable("_iter_${iterVarId++}")
    }

    fun genOutputControlVar(): Variable {
        return Variable("_produce_output_${outputControlId++}")
    }

    fun genInputRequestVar(): Variable {
        return Variable("_input_request_${inputRequestId++}")
    }

    fun genWriterVar(): Variable {
        return Variable("__bodo_streaming_writer_${streamingWriterId++}")
    }

    fun genGenericTempVar(): Variable {
        return Variable("_temp${tempVarId++}")
    }

    // Variable for an accumulator for elapsed time of a stage of an operator
    fun genOperatorStageTimerVar(
        opID: Int,
        stageID: Int,
    ): Variable {
        return Variable("_op_stage_timer_${opID}_$stageID")
    }

    // Variable for start timestamp for an instance of an operator stage in a pipeline
    fun genOperatorStageTimerStartVar(
        opID: Int,
        stageID: Int,
    ): Variable {
        return Variable("_start_op_stage_${opID}_$stageID")
    }

    fun genOperatorStageRowCountVar(
        opID: Int,
        stageID: Int,
    ): Variable {
        return Variable("_op_stage_${opID}_${stageID}_output_len")
    }

    // Variable for end timestamp for an instance of an operator stage in a pipeline
    fun genOperatorStageTimerEndVar(
        opID: Int,
        stageID: Int,
    ): Variable {
        return Variable("_end_op_stage_${opID}_$stageID")
    }

    // Variable for time elapsed for an instance of an operator stage in a pipeline
    fun genOperatorStageTimerElapsedVar(
        opID: Int,
        stageID: Int,
    ): Variable {
        return Variable("_elapsed_op_stage_${opID}_$stageID")
    }

    fun genTempColumnVar(): Variable {
        return Variable("__bodo_generated_column__${colVarId++}")
    }

    fun genBatchAccumulatorVar(): Variable {
        return Variable("__bodo_streaming_batches_table_builder_${accVarId++}")
    }

    fun genFinishedStreamingFlag(): Variable {
        return Variable("__bodo_is_last_streaming_output_${streamingFlagId++}")
    }

    fun genWindowedAggDf(): Variable {
        return Variable("__bodo_windowfn_generated_df_${dfVarId++}")
    }

    fun genWindowedAggFnName(): Variable {
        return Variable("${DUMMY_COL_NAME_BASE}_sql_windowed_apply_fn_${groupByApplyFnId++}")
    }

    fun genGroupbyApplyAggFnName(): Variable {
        return Variable("${DUMMY_COL_NAME_BASE}_sql_groupby_apply_fn_${groupByApplyFnId++}")
    }

    fun genGlobalVar(): Variable {
        return Variable("global_${globalVarId++}")
    }

    fun genStateVar(): StateVariable {
        return StateVariable("state_${stateVarId++}")
    }

    fun genFuncID(): Expr.IntegerLiteral {
        return Expr.IntegerLiteral(funcIdCounter++)
    }

    fun genClosureVar(): Variable {
        return Variable("func_${closureVarId++}")
    }
}
