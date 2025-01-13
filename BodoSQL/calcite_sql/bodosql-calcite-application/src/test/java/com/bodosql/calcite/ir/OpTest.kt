package com.bodosql.calcite.ir

import kotlin.test.Test
import kotlin.test.assertEquals

class OpTest {
    @Test
    fun testOpIfEmit() {
        // Test If emit with 1 if and 1 else
        val doc = Doc()
        // Create IF
        val ifFrame = CodegenFrame()
        ifFrame.add(Op.Assign(Variable("y"), Expr.Raw("7")))
        val elseFrame = CodegenFrame()
        elseFrame.add(Op.Assign(Variable("y"), Expr.Raw("-1")))
        val ifOp: Op.If = Op.If(Expr.Raw("x == 5"), ifFrame, elseFrame)
        ifOp.emit(doc)

        val actual = doc.toString().trimEnd()
        val expected =
            """
            if x == 5:
              y = 7
            else:
              y = -1
            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testOpIfNoElseEmit() {
        val doc = Doc()
        // Create IF
        val ifFrame = CodegenFrame()
        ifFrame.add(Op.Assign(Variable("y"), Expr.Raw("7")))
        val ifOp = Op.If(Expr.Raw("x == 5"), ifFrame)
        ifOp.emit(doc)

        val actual = doc.toString().trimEnd()
        val expected =
            """
            if x == 5:
              y = 7
            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testOpMultiIfEmit() {
        // Test If emits with several ifs and 1 else
        val doc = Doc()
        // Create the if statements in reverse order
        // Innermost If
        val ifFrame3 = CodegenFrame()
        ifFrame3.add(Op.Assign(Variable("y"), Expr.Raw("0")))
        val elseFrame3 = CodegenFrame()
        elseFrame3.add(Op.Assign(Variable("y"), Expr.Raw("-2")))
        val ifOp3 = Op.If(Expr.Raw("(x == 4) and (j > 0)"), ifFrame3, elseFrame3)
        // Middle If
        val ifFrame2 = CodegenFrame()
        ifFrame2.add(Op.Assign(Variable("y"), Expr.Raw("21")))
        val elseFrame2 = CodegenFrame()
        elseFrame2.add(ifOp3)
        val ifOp2 = Op.If(Expr.Raw("x != 4"), ifFrame2, elseFrame2)
        // Outermost If
        val ifFrame1 = CodegenFrame()
        ifFrame1.add(Op.Assign(Variable("y"), Expr.Raw("14")))
        val elseFrame1 = CodegenFrame()
        elseFrame1.add(ifOp2)
        val ifOp = Op.If(Expr.Raw("x == 5"), ifFrame1, elseFrame1)

        // Emit the code
        ifOp.emit(doc)

        val actual = doc.toString().trimEnd()
        val expected =
            """
            if x == 5:
              y = 14
            else:
              if x != 4:
                y = 21
              else:
                if (x == 4) and (j > 0):
                  y = 0
                else:
                  y = -2
            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testOpCodeEmit() {
        // Identical text should be the same.
        assertOpEmit(
            """def foo():
            |  return 4
            """.trimMargin(),
            Op.Code(
                """def foo():
            |  return 4
                """.trimMargin(),
            ),
        )

        // With consistent indentation.
        assertOpEmit(
            """def foo():
            |  return 4
            """.trimMargin(),
            Op.Code(
                """  def foo():
            |    return 4
                """.trimMargin(),
            ),
        )

        // Blank lines with random indentation.
        assertOpEmit(
            """def foo():
            |  return 4
            """.trimMargin(),
            Op.Code(
                """  def foo():
            |
            |
            |    return 4
                """.trimMargin(),
            ),
        )
    }

    @Test
    fun testWhileEmit() {
        val doc = Doc()
        val frame = CodegenFrame()
        frame.add(Op.Assign(Variable("x"), Expr.IntegerLiteral(1)))
        frame.add(Op.Stmt(Expr.Call("f", listOf(Variable("x")))))
        val whileOp = Op.While(Expr.Raw("y == 2"), frame)
        whileOp.emit(doc)
        val actual = doc.toString().trimEnd()
        val expected =
            """
            while y == 2:
              x = 1
              f(x)
            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testStreamingPipelineEmit() {
        val frame = StreamingPipelineFrame(Variable("stop"), Variable("iter"), StreamingStateScope(), 1)
        frame.add(Op.Assign(Variable("v1"), Expr.Raw("1")))
        frame.add(Op.Assign(Variable("stop"), Expr.Raw("v1 > 1")))
        val doc = Doc()
        val streamingPipeline = Op.StreamingPipeline(frame)
        streamingPipeline.emit(doc)
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

    private fun assertOpEmit(
        expected: String,
        op: Op,
    ) {
        val doc = Doc()
        op.emit(doc)

        val actual = doc.toString().trimEnd()
        assertEquals(expected, actual)
    }

    @Test
    fun testTupleAssign() {
        val doc = Doc()

        // Check empty edge case is handled (should be no-op)
        Op.TupleAssign(listOf(), Expr.IntegerLiteral(1)).emit(doc)
        // Check 1 element edge case is handled
        Op
            .TupleAssign(
                listOf(Variable("A")),
                Expr.Tuple(
                    listOf(Expr.IntegerLiteral(1)),
                ),
            ).emit(doc)
        // Check 1 element edge case is handled
        Op
            .TupleAssign(
                listOf(Variable("X"), Variable("Y"), Variable("Z")),
                Expr.Tuple(
                    listOf(
                        Expr.IntegerLiteral(-1),
                        Expr.IntegerLiteral(-2),
                        Expr.IntegerLiteral(-3),
                    ),
                ),
            ).emit(doc)

        val actual = doc.toString().trimEnd()
        val expected =
            """
            (A, ) = (1,)
            (X, Y, Z, ) = (-1, -2, -3)
            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testReturn() {
        val doc = Doc()
        // Check null is handled
        Op.ReturnStatement(null).emit(doc)
        Op.ReturnStatement(Variable("df2")).emit(doc)

        val actual = doc.toString().trimEnd()
        val expected =
            """
            return
            return df2
            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testSetItem() {
        val doc = Doc()
        Op.SetItem(Variable("arr"), Variable("i"), Expr.IntegerLiteral(7)).emit(doc)

        val actual = doc.toString().trimEnd()
        val expected =
            """
            arr[i] = 7
            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testContinue() {
        val doc = Doc()
        Op.Continue.emit(doc)
        val actual = doc.toString().trimEnd()
        val expected =
            """
            continue
            """.trimIndent()
        assertEquals(expected, actual)
    }
}
