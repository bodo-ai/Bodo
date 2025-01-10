package com.bodosql.calcite.ir

import java.lang.Exception
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class StreamingStateScopeTest {
    @Test
    fun testNoRegisteredOps() {
        val scope = StreamingStateScope()
        val ops = scope.genOpComptrollerInit()
        assertEquals(2, ops.size)
    }

    @Test
    fun testRegisterOneOp() {
        val scope = StreamingStateScope()
        val opID = OperatorID(0, true)
        scope.startOperator(opID, 0, OperatorType.UNKNOWN)
        scope.endOperator(opID, 0)

        val ops = scope.genOpComptrollerInit()
        assertEquals(3, ops.size)
        when (val registerOp = ops[1]) {
            is Op.Stmt -> assertTrue(registerOp.expr is Expr.Call)
            else -> assertTrue(false, "Expected a Op.Stmt")
        }
    }

    @Test
    fun testRegisterOpWithNoEnd() {
        val scope = StreamingStateScope()
        scope.startOperator(OperatorID(0, true), 0, OperatorType.UNKNOWN)
        var gotException = false
        try {
            scope.genOpComptrollerInit()
        } catch (e: Exception) {
            assertTrue(e.message!!.contains("StreamingStateScope"))
            gotException = true
        }
        assertTrue(gotException)
    }

    @Test
    fun testRegisterOpWithInvalidEndPipeline() {
        val scope = StreamingStateScope()
        val opID = OperatorID(0, true)
        // Start at pipeline id 1
        scope.startOperator(opID, 1, OperatorType.UNKNOWN)
        var gotException = false
        try {
            // end at pipeline id 0
            scope.endOperator(opID, 0)
        } catch (e: Exception) {
            assertTrue(e.message!!.contains("StreamingStateScope"))
            gotException = true
        }
        assertTrue(gotException)
    }

    @Test
    fun testRegisterOpTwice() {
        val scope = StreamingStateScope()
        val opID = OperatorID(0, true)
        scope.startOperator(opID, 0, OperatorType.UNKNOWN)
        var gotException = false
        try {
            scope.startOperator(opID, 1, OperatorType.UNKNOWN)
        } catch (e: Exception) {
            assertTrue(e.message!!.contains("StreamingStateScope"))
            gotException = true
        }
        assertTrue(gotException)
    }

    @Test
    fun testEndOpTwice() {
        val scope = StreamingStateScope()
        val opID = OperatorID(0, true)
        scope.startOperator(opID, 0, OperatorType.UNKNOWN)
        scope.endOperator(opID, 0)
        var gotException = false
        try {
            scope.endOperator(opID, 1)
        } catch (e: Exception) {
            assertTrue(e.message!!.contains("StreamingStateScope"))
            gotException = true
        }
        assertTrue(gotException)
    }

    @Test
    fun testEndUnknownOp() {
        val scope = StreamingStateScope()
        var gotException = false
        try {
            scope.endOperator(OperatorID(0, true), 1)
        } catch (e: Exception) {
            assertTrue(e.message!!.contains("StreamingStateScope"))
            gotException = true
        }
        assertTrue(gotException)
    }

    @Test
    fun testAddToFrame() {
        val frame = CodegenFrame()
        val op1 = Op.Assign(Variable("first"), Expr.IntegerLiteral(1))
        frame.add(op1)
        val op2 = Op.Assign(Variable("second"), Expr.IntegerLiteral(2))
        frame.add(op2)
        val ret = Op.ReturnStatement(Variable("first"))
        frame.add(ret)

        val scope = StreamingStateScope()
        val opID = OperatorID(0, false)
        scope.startOperator(opID, 0, OperatorType.UNKNOWN)
        scope.endOperator(opID, 0)
        scope.addToFrame(frame)

        val doc = Doc()
        frame.emit(doc)
        val actual = doc.toString()
        val expected =
            """
            bodo.libs.memory_budget.init_operator_comptroller()
            bodo.libs.memory_budget.register_operator(0, bodo.libs.memory_budget.OperatorType.UNKNOWN, 0, 0, -1)
            bodo.libs.memory_budget.compute_satisfiable_budgets()
            bodo.libs.query_profile_collector.init()
            first = 1
            second = 2
            bodo.libs.query_profile_collector.finalize()
            return first

            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testAddToFrameNoOperations() {
        val frame = CodegenFrame()
        val op1 = Op.Assign(Variable("first"), Expr.IntegerLiteral(1))
        frame.add(op1)
        val op2 = Op.Assign(Variable("second"), Expr.IntegerLiteral(2))
        frame.add(op2)

        val scope = StreamingStateScope()
        scope.addToFrame(frame)
        val doc = Doc()
        frame.emit(doc)
        val actual = doc.toString()
        val expected =
            """
            first = 1
            second = 2

            """.trimIndent()
        assertEquals(expected, actual)
    }
}
