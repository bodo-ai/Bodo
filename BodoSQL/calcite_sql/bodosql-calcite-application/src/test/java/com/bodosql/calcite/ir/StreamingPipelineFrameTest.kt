package com.bodosql.calcite.ir

import kotlin.test.Test
import kotlin.test.assertEquals

class StreamingPipelineFrameTest {
    @Test
    fun testEmit() {
        val frame = StreamingPipelineFrame(Variable("stop"), Variable("iter"), StreamingStateScope(), 1)
        frame.add(Op.Assign(Variable("v1"), Expr.Raw("1")))
        frame.add(Op.Assign(Variable("stop"), Expr.Raw("v1 > 1")))
        val doc = Doc()
        // Append the frame result
        frame.emit(doc)
        val actual = doc.toString()
        val expected =
            """
            stop = False
            iter = 0
            bodo.libs.query_profile_collector.start_pipeline(1)
            while not(stop):
              v1 = 1
              stop = v1 > 1
              iter = (iter + 1)
            bodo.libs.query_profile_collector.end_pipeline(1, iter)

            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testAppend() {
        val frame = StreamingPipelineFrame(Variable("stop"), Variable("iter"), StreamingStateScope(), 1)
        // Single line.
        frame.append("a = 1\n")
        // Test multiple lines being explicitly appended
        frame.append("b = [\n")
        frame.append("  1,\n")
        frame.append("  2,\n")
        frame.append("]\n")

        val doc = Doc()
        // Append the frame result
        frame.emit(doc)
        val actual = doc.toString()
        val expected =
            """
            stop = False
            iter = 0
            bodo.libs.query_profile_collector.start_pipeline(1)
            while not(stop):
              a = 1
              b = [
                1,
                2,
              ]
              iter = (iter + 1)
            bodo.libs.query_profile_collector.end_pipeline(1, iter)

            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testAddInitializationTermination() {
        val frame = StreamingPipelineFrame(Variable("stop"), Variable("iter"), StreamingStateScope(), 1)
        frame.add(Op.Assign(Variable("v1"), Expr.Raw("1")))
        frame.addInitialization(Op.Assign(Variable("state"), Expr.Call("aggregate_state")))
        frame.addTermination(Op.Stmt(Expr.Call("delete_state", listOf(Variable("state")))))
        frame.add(Op.Assign(Variable("stop"), Expr.Call("f", listOf(Variable("v1"), Variable("state")))))
        val doc = Doc()
        // Append the frame result
        frame.emit(doc)
        val actual = doc.toString()
        val expected =
            """
            stop = False
            iter = 0
            state = aggregate_state()
            bodo.libs.query_profile_collector.start_pipeline(1)
            while not(stop):
              v1 = 1
              stop = f(v1, state)
              iter = (iter + 1)
            delete_state(state)
            bodo.libs.query_profile_collector.end_pipeline(1, iter)

            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testEndSection() {
        val frame = StreamingPipelineFrame(Variable("stop"), Variable("iter"), StreamingStateScope(), 1)
        frame.add(Op.Assign(Variable("v1"), Expr.Raw("1")))
        frame.add(Op.Assign(Variable("stop"), Expr.Raw("v1 > 1")))
        frame.endSection(Variable("stop2"))
        frame.add(Op.Assign(Variable("stop2"), Expr.Call("g")))
        val doc = Doc()
        // Append the frame result
        frame.emit(doc)
        val actual = doc.toString()
        val expected =
            """
            stop = False
            iter = 0
            stop2 = False
            bodo.libs.query_profile_collector.start_pipeline(1)
            while not(stop2):
              v1 = 1
              stop = v1 > 1
              stop2 = g()
              iter = (iter + 1)
            bodo.libs.query_profile_collector.end_pipeline(1, iter)

            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testGenerateIOFlags() {
        val frame = StreamingPipelineFrame(Variable("stop"), Variable("iter"), StreamingStateScope(), 1)
        frame.addOutputControl(Variable("OutputControl1"))
        frame.addInputRequest(Variable("InputRequest1"))
        frame.addOutputControl(Variable("OutputControl2"))
        frame.addInputRequest(Variable("InputRequest2"))
        frame.addOutputControl(Variable("OutputControl3"))
        val doc = Doc()

        // Append the frame result
        frame.emit(doc)
        val actual = doc.toString()
        val expected =
            """
            stop = False
            iter = 0
            OutputControl1 = True
            OutputControl2 = True
            OutputControl3 = True
            bodo.libs.query_profile_collector.start_pipeline(1)
            while not(stop):
              iter = (iter + 1)
              OutputControl1 = (InputRequest2 and InputRequest1)
              OutputControl2 = InputRequest2
            bodo.libs.query_profile_collector.end_pipeline(1, iter)
            
            """.trimIndent()
        assertEquals(expected, actual)
    }
}
