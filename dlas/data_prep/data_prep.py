import os
import logging as log

import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreparer(object):
    """
    This class is responsible for providing the instances as a numpy-file to the
    main workflow.
    For both, image and label conversion, one can choose between using one of
    the provided methods or a customized one.

    Args:
        config: dictionary or Configuration
            specifies conversion details
        instance_path: string
            path to instance-files to be read in
        img_dir: string
            where to put imagedata
        label_dir: string
            where to put labeldata
    """

    def __init__(self, config, aslib, instance_path, img_dir=None, label_dir=None):
        self.log = log.getLogger('DataPreparer')

        self.config = config
        self.aslib = aslib
        image_mode, label_mode = config["image-mode"], config["label-mode"]

        self.image_prep = None
        self.label_prep = None
        self._set_image_prep(image_mode)
        self._set_label_prep(label_mode)

        self.inst_path = instance_path

        self.img_dir = img_dir
        self.label_dir = label_dir
        #TODO
        self.log.debug("Saving images in: \"{}\", labels in: \"{}\"".format(self.img_dir,
            self.label_dir))

    def norm(self, data):
        scale = StandardScaler()
        return scale.fit_transform(data)

    def float32(self, k):  # Necessary for theano-intern reasons
        return np.cast['float32'](k)

    def get_image_data(self, local_inst, recalculate=False):
        """
        Provides the image-data for a scenario scen with specific conversion-properties
        specified in config.

        Arguments:
            local_inst: list of strings (ORDERED!)
                an ordered list of paths to local instances

        Returns:
            img_data: np.array
                the (converted) image-instance data

        Sideeffect:
            if self.img_output, writes data into output_dir
        """
        path = os.path.join(self.img_dir, self.image_prep.id+".npy")
        self.log.debug("Checking for image-data in {}".format(path))

        if (not recalculate and self.img_dir and os.path.isfile(path)):
            self.log.debug("Load image-data from {}".format(path))
            return np.load(path)  # TODO catch?
        else:
            self.log.debug("Calculating image-data ... ")
            image_data, times = self.image_prep.get_image_data(local_inst)
            self.log.debug("done!")

        image_data = np.reshape(image_data, (-1, 1, self.config["image-dim"],
                                             self.config["image-dim"]))
        num_inst = image_data.shape[0]
        image_data = np.reshape(image_data, (num_inst, -1))
        image_data = self.float32(image_data)
        image_data = self.norm(image_data)
        image_data = np.reshape(image_data, (-1, 1, self.config["image-dim"],
                                             self.config["image-dim"]))

        if np.isnan(image_data).any():
            self.log.error("Something went wrong with preprocessing (NAN)")
            #image_data = np.nan_to_num(image_data)
        if ((image_data == 0).all()):
            self.log.error("Something went wrong with preprocessing (all equal 0)")

        self.log.debug("Saving image-data in {}.".format(path))
        np.save(path, image_data)
        return image_data

    def get_label_data(self, inst, recalculate=False):
        """
        Provides the label-data for a scenario scen with specific
        conversion-properties specified in config.
        Instances as in aslib-instances (not local-paths!).

        Returns
            label_data: np.array
                (converted) label-data

        Sideeffect:
            if self.label_output, writes data into output_dir
        """
        path = os.path.join(self.label_dir, self.label_prep.id+".npy")
        self.log.debug("Checking for label-data in {}".format(path))
        if (not recalculate and self.label_dir and os.path.isfile(path)):
            self.log.debug("Loading label-data from {}".format(path))
            return np.load(path)
        else:
            self.log.debug("Calculating label-data.")
            label_data = self.label_prep.get_label_data(inst)

        label_data = np.reshape(label_data, (len(inst), -1))
        label_data = self.float32(label_data)

        if self.label_dir:
            np.save(path, label_data)
        return label_data

    def get_class(self, kls):
        """
        Used to load and use customized preparation-classes.

        Args:
            kls: string
                path to class-file
        """
        parts = kls.split('.')
        module = ".".join(parts[:-1])
        m = __import__( module )
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m

    def _set_image_prep(self, image_mode):
        """
        Setting image-preparation-method by loading class manually.

        Args:
            image_mode: string
                class- (and file-) name of preparation unit
        """
        try:
            self.image_prep = self.get_class("dlas.data_prep."+image_mode+"."+image_mode)(self.config)
        except ImportError:
            self.log.error("No class with name {} found. Do file and class have "
                           "the same name?".format(image_mode))

    def _set_label_prep(self, label_mode):
        """
        Setting label-preparation-method by loading class manually.

        Args:
            label_mode: string
                class- (and file-) name of preparation unit
        """
        try:
            self.label_prep = self.get_class("dlas.data_prep."+label_mode+"."+label_mode)(self.config, self.aslib)
        except ImportError:
            self.log.error("No class with name {} found. Do file and class have "
                           "the same name?".format(label_mode))
