import os
import logging as log

import numpy as np

from dlas.data_prep.text_to_image import TextToImage
from dlas.data_prep.from_image import FromImage

from dlas.data_prep.multi_label_base import MultiLabelBase


class DataPreparer(object):
    """
    This class is responsible for providing the instances as a numpy-file to the
    main workflow.
    For both, image and label conversion, one can choose between using one of
    the provided methods or a customized one.

    Args:
        config -- dictionary with config-details
        instance_path -- path to instance-files to be read in
        img_dir -- where to put imagedata
        label_dir -- where to put labeldata
    """

    def __init__(self, config, instance_path=None, img_dir=None, label_dir=None):
        self.log = log.getLogger('DataPreparer')

        image_mode, label_mode = config["image_mode"], config["label_mode"]

        self.image_prep = None
        self.label_prep = None
        self._set_image_prep(image_mode)
        self._set_label_prep(label_mode)
        self.config = config

        if instance_path:
            self.inst_path = instance_path
        else:
            # Define instance dir (relative/root...), so you can have instances in a remote place
            #if os.getcwd().startswith("/home/marbenj"):
            #    #instDir = "/data/aad/benchmarks/"
            #    instDir = "/home/marbenj/instances/TSP/"
            #elif os.getcwd().startswith("/home/shuki"):
            #    #instDir = "/data/aad/benchmarks/"
            #    instDir = "/home/marbenj/ishukis/TSP/DLTSP/"
            #else:
            #    self.log.warning("Somethings wrong")
            #    instDir = os.path.join(os.getcwd(), "instances/")
            raise ValueError("No instance path specified!")
        self.img_dir = img_dir
        self.label_dir = label_dir
        #TODO

    def normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        if std.any == 0:
            self.log.warning("ALERT! std = 0")
        data = (data-mean)/std
        return data

    def float32(self, k):  # Necessary for theano-intern reasons
        return np.cast['float32'](k)

    def get_image_data(self, local_inst, recalculate=False):
        """
        Provides the image-data for a scenario scen with specific conversion-properties
        specified in config.

        Arguments:
            local_inst -- list of strings (ORDERED!)
                an ordered list of paths to local instances

        Returns:
            img_data

        Sideeffect:
            if self.img_output, writes data into output_dir
        """

        if (not recalculate and self.img_dir and os.path.isfile(self.img_dir)):
            image_data = np.load(self.img_dir)  # TODO catch?
        else:
            image_data, times = self.image_prep.get_image_data(local_inst)

        image_data = np.reshape(image_data, (-1, 1, self.config["image-dim"],
                                             self.config["image_dim"]))
        image_data = self.float32(image_data)
        image_data = self.normalize(image_data)
        #imageData = np.nan_to_num(imageData)
        assert(np.isnan(image_data).any(), False)

        np.save(self.img_dir, image_data)
        return image_data

    def get_label_data(self, local_inst, recalculate=False):
        """
        Provides the label-data for a scenario scen with specific conversion-properties
        specified in config.

        Returns label_data

        Sideeffect:
            if self.label_output, writes data into output_dir
        """
        num_solvers = None # TODO

        if (not recalculate and self.label_dir and os.path.isfile(self.label_dir)):
            label_data = np.load(self.label_dir)
        else:
            label_data = self.label_prep.get_label_data(local_inst)

        label_data = np.reshape(label_data, (-1, num_solvers))
        label_data = self.float32(label_data)

        if self.label_dir: np.save(self.label_dir, label_data)
        return label_data

    def _set_image_prep(self, image_mode):
        if image_mode == "TextToImage":
            self.image_prep = TextToImage()
        elif image_mode == "FromImage":
            self.image_prep = FromImage(self.config)
        else:
            raise Exception(image_mode + " not implemented.")

    def _set_label_prep(self, label_mode):
        if label_mode == "MultiLabelBase":
            self.label_prep = MultiLabelBase()
        else:
            raise Exception(label_mode + " not implemented.")
