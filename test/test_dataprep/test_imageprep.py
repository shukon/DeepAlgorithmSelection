import pickle
import unittest
import logging

from dlas.aslib.aslib_handler import ASlibHandler
from dlas.data_prep.data_prep import DataPreparer
import dlas.config.config as conf


class ImagePrepTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig()
        self.logger = logging.getLogger('ImagePrepTest')
        self.logger.setLevel(logging.DEBUG)

        self.config = conf.conf("TestScen", "TestID")
        self.aslib = ASlibHandler()
        with open("aslib_loaded.pickle", "rb") as f:
            self.aslib.data = pickle.load(f)

    def tearDown(self):
        pass

    def test_from_image(self):
        c = conf.update(self.config, [("image-mode", "FromImage")])
        prep = DataPreparer(c, self.aslib,
                            instance_path="test/test_files/instances/fromimage",
                            img_dir="test/test_files/", label_dir="test/test_files/")
        scen = self.config["scen"]
        inst = self.aslib.get_instances(scen)
        img = prep.get_image_data(self.aslib.local_paths(scen, inst),
                recalculate=True)

    def test_text2image(self):
        c = conf.update(self.config, [("image-mode", "TextToImage")])
        prep = DataPreparer(c, self.aslib,
                            instance_path="test/test_files/instances/text2image",
                            img_dir="test/test_files/", label_dir="test/test_files/")
        scen = self.config["scen"]
        inst = self.aslib.get_instances(scen)
        img = prep.get_image_data(self.aslib.local_paths(scen, inst),
                recalculate=True)

