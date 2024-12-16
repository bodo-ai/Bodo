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

    fun genDfVar(): Variable = Variable("df${dfVarId++}")

    fun genTableVar(): Variable = Variable("T${tableVarId++}")

    fun genSeriesVar(): Variable = Variable("S${seriesVarId++}")

    fun genArrayVar(): Variable = Variable("A${arrVarId++}")

    fun genIndexVar(): Variable = Variable("index_${idxVarId++}")

    fun genIterVar(): Variable = Variable("_iter_${iterVarId++}")

    fun genOutputControlVar(): Variable = Variable("_produce_output_${outputControlId++}")

    fun genInputRequestVar(): Variable = Variable("_input_request_${inputRequestId++}")

    fun genWriterVar(): Variable = Variable("__bodo_streaming_writer_${streamingWriterId++}")

    fun genGenericTempVar(): Variable = Variable("_temp${tempVarId++}")

    // Variable for an accumulator for elapsed time of a stage of an operator
    fun genOperatorStageTimerVar(
        opID: OperatorID,
        stageID: Int,
    ): Variable = Variable("_op_stage_timer_${opID}_$stageID")

    // Variable for start timestamp for an instance of an operator stage in a pipeline
    fun genOperatorStageTimerStartVar(
        opID: OperatorID,
        stageID: Int,
    ): Variable = Variable("_start_op_stage_${opID}_$stageID")

    fun genOperatorStageRowCountVar(
        opID: OperatorID,
        stageID: Int,
    ): Variable = Variable("_op_stage_${opID}_${stageID}_output_len")

    // Variable for end timestamp for an instance of an operator stage in a pipeline
    fun genOperatorStageTimerEndVar(
        opID: OperatorID,
        stageID: Int,
    ): Variable = Variable("_end_op_stage_${opID}_$stageID")

    // Variable for time elapsed for an instance of an operator stage in a pipeline
    fun genOperatorStageTimerElapsedVar(
        opID: OperatorID,
        stageID: Int,
    ): Variable = Variable("_elapsed_op_stage_${opID}_$stageID")

    fun genTempColumnVar(): Variable = Variable("__bodo_generated_column__${colVarId++}")

    fun genBatchAccumulatorVar(): Variable = Variable("__bodo_streaming_batches_table_builder_${accVarId++}")

    fun genFinishedStreamingFlag(): Variable = Variable("__bodo_is_last_streaming_output_${streamingFlagId++}")

    fun genWindowedAggDf(): Variable = Variable("__bodo_windowfn_generated_df_${dfVarId++}")

    fun genWindowedAggFnName(): Variable = Variable("${DUMMY_COL_NAME_BASE}_sql_windowed_apply_fn_${groupByApplyFnId++}")

    fun genGroupbyApplyAggFnName(): Variable = Variable("${DUMMY_COL_NAME_BASE}_sql_groupby_apply_fn_${groupByApplyFnId++}")

    fun genGlobalVar(): Variable = Variable("global_${globalVarId++}")

    fun genStateVar(): StateVariable = StateVariable("state_${stateVarId++}")

    fun genFuncID(): Expr.IntegerLiteral = Expr.IntegerLiteral(funcIdCounter++)

    fun genClosureVar(): Variable = Variable("func_${closureVarId++}")
}
