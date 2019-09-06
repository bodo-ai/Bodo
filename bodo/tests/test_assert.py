"""
PYTEST_DONT_REWRITE
"""
import unittest
import bodo

# using separate file for assert tests to avoid pytest rewrite errors
class AssertTest(unittest.TestCase):

    @unittest.skip("TODO: fix static raise issue")
    def test_assert(self):
        # make sure assert in an inlined function works
        def g(a):
            assert a==0

        hpat_g = bodo.jit(g)
        def f():
            hpat_g(0)

        # TODO: check for static raise presence in IR
        hpat_f = bodo.jit(f)
        hpat_f()
