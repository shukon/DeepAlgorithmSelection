import os
import time

from PIL import Image
import numpy as np

from dlas.data_prep.image_prep import ImagePrep

class FromImage(ImagePrep):
    """
    Implements image conversion from text-file to image-file.
    """
    def __init__(self, config):
        super(FromImage, self).__init__(config)
        self.image_dim = config["image-dim"]
        if config["resize-method"] == 'LANCZOS':
            self.resize_method = Image.LANCZOS
        else:
            raise ValueError("{} is not a valid resize-option for FromImage-"
                             "conversion".format(config["resize-method"]))

        self.id = "-".join([config.scen,self.image_mode,
                       str(self.image_dim), config["resize-method"]])

    def get_image_data(self, local_inst):
        """
        Arguments:
            local_inst -- list of strings
                local paths to instance-pictures

        Returns:
            X -- numpy.array
                image-data
            times -- list of ints
                time to convert for each instance
        """
        data, times = np.array([]), []
        for i in local_inst:
            img, t = self._convert(i)
            data = np.append(data, np.array(img))
            times.append(t)
            if (np.isnan(np.array(img)).any()):
                self.log.warning("NAN: {}".format(i))
        return data, times

    def _convert(self, img_path, save=True):
        """
        Converts instance-image specified in img_path and returns modified
        image, conversion time and - if save - saves the image.
        """
        start = time.clock()
        img = Image.open(img_path)
        img = img.convert('L')
        img = img.resize((self.image_dim, self.image_dim), Image.LANCZOS)
        if save:
            img_path = os.path.splitext(img_path)[0]
            img.save("{}_resized-{}-{}.jpeg".format(img_path, self.image_dim,
                                self.resize_method)+".jpeg", "JPEG")
        stop = time.clock()
        return img, stop-start


