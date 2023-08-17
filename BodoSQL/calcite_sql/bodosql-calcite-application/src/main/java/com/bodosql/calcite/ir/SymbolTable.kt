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
    private var streamingReaderId: Int = 1
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
        private const val dummyColNameBase = "__bodo_dummy__"
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

    fun genReaderVar(): Variable {
        return Variable("__bodo_streaming_reader_${streamingReaderId++}")
    }

    fun genWriterVar(): Variable {
        return Variable("__bodo_streaming_writer_${streamingWriterId++}")
    }

    fun genGenericTempVar(): Variable {
        return Variable("_temp${tempVarId++}")
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
        return Variable("${dummyColNameBase}_sql_windowed_apply_fn_${groupByApplyFnId++}")
    }

    fun genGroupbyApplyAggFnName(): Variable {
        return Variable("${dummyColNameBase}_sql_groupby_apply_fn_${groupByApplyFnId++}")
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
