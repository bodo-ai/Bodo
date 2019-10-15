# Copyright (C) 2019 Bodo Inc.
import unittest
import bodo.tests

def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(bodo.tests))
    return suite

if __name__ == '__main__':
    unittest.main()
