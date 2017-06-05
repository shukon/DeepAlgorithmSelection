import unittest
import logging

from dlas.aslib.aslib_handler import ASlibHandler


class ASlibTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()
        self.logger = logging.getLogger('ASlibTest')
        self.logger.setLevel(logging.DEBUG)

        self.scen = "TestScen"
        self.ID = "TestID"
        # Load scenario
        self.aslib = ASlibHandler()
        self.aslib.load_scenario(self.scen, 'jpeg')

    def test_cvs(self):
        folds = self.aslib.getCVfolds(self.scen)
        folds_expected = [['test1'], ['test2'], ['test3'], ['test4'], ['test5'],
                          ['test6'], ['test7'], ['test8'], ['test9'], ['test10']]
        self.assertEqual(folds, folds_expected)

    def test_get_instances(self):
        insts = self.aslib.get_instances(self.scen)
        insts_expected = ['test1', 'test10', 'test2', 'test3', 'test4', 'test5',
                          'test6', 'test7', 'test8', 'test9']
        self.assertEqual(insts, insts_expected)

    def test_get_solvers(self):
        solvers = self.aslib.get_solvers(self.scen)
        solvers_expected = ['eax', 'eax.restart', 'lkh', 'lkh.restart', 'maos']
        self.assertEqual(solvers, solvers_expected)
