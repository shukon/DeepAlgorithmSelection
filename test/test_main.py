import pickle
import unittest
import logging

from dlas.aslib.aslib_handler import ASlibHandler
from dlas.config.config import Config
from dlas.main import prep, run_experiment


class MainTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig()
        self.logger = logging.getLogger('ImagePrepTest')
        self.logger.setLevel(logging.DEBUG)

        self.scen = "TSP"
        self.ID = "tsp-test"
        self.config = Config(self.scen, self.ID)
        self.aslib = ASlibHandler()
        with open("aslib_loaded.pickle", "rb") as f:
            self.aslib.data = pickle.load(f)

    def tearDown(self):
        pass

    def test_pipeline(self):
        prep(self.scen, self.config, "instances/", True)
        run_experiment(self.scen, self.ID, self.config, False)

