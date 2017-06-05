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

    def test_solvers_stats(self):
        stats = self.aslib.solver_stats(self.scen)
        stats_expected = ['Solver 0: (4.5074999999999994, 0.31346650538773674) PAR10 with 0.0 '
                            'unsuccessful runs.',
                            'Solver 1: (4.5054999999999996, 0.2419550578103298) PAR10 with 0.0 '
                            'unsuccessful runs.',
                            'Solver 2: (3630.6485000000007, 10789.88993237606) PAR10 with 0.1 '
                            'unsuccessful runs.',
                            'Solver 3: (44.980999999999995, 95.470006934115176) PAR10 with 0.0 '
                            'unsuccessful runs.',
                            'Solver 4: (7207.7865000000002, 14396.106753974675) PAR10 with 0.2 '
                            'unsuccessful runs.']
        self.assertEqual(stats, stats_expected)
