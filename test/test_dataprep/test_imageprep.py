import pickle
import unittest
import logging

from dlas.aslib.aslib_handler import ASlibHandler
from dlas.data_prep.data_prep import DataPreparer
from dlas.config.config import conf


class ImagePrepTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()
        self.logger = logging.getLogger('ImagePrepTest')
        self.logger.setLevel(logging.DEBUG)

        self.config = conf("TestScen", "TestID")
        self.aslib = ASlibHandler()
        with open("aslib_loaded.pickle", "rb") as f:
            self.aslib.data = pickle.load(f)

    def tearDown(self):
        pass

    def test_from_image(self):
        prep = DataPreparer(self.config, self.aslib, instance_path="test/test_files/",
                            img_dir="test/test_files/", label_dir="test/test_files/")
        scen = self.config["scen"]
        inst = self.aslib.get_instances(scen)
        img = prep.get_image_data(self.aslib.local_paths(scen, inst),
                recalculate=True)

