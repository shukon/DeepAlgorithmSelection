import time
import gzip, bz2
import math
import warnings

from PIL import Image
import numpy as np

from dlas.data_prep.image_prep import ImagePrep

class TextToImage(ImagePrep):
    """
    Implements image conversion from text-file to image-file.
    """
    def __init__(self, config):
        super(TextToImage, self).__init__(config)
        self.image_dim = config["image-dim"]
        self.resize_method = config["resize-method"]
        self.unpack = config["unpack"]
        self.round_method = config["round-method"]
        self.id = "-".join([config["scen"],self.image_mode,
                       str(self.image_dim), self.resize_method,
                       str(self.unpack), self.round_method])

    def get_image_data(self, local_inst):
        """
        Arguments:
            local_inst -- list of strings
                local paths to instances

        Returns:
            X -- numpy.array
                image-data
            times -- list of ints
                time to convert for each instance
        """
        # Make image data
        X = np.array([])  # Image data (sorted alphabetically for instance names)
        times = []        # Measure time for conversion (should be optional)

        # Conversion-loop
        for i in local_inst:
            image, time = self._convert(i)
            X = np.append(X, image)
            times.append(time)
            if len(times)%50 == 0 or local_inst.index(i) == len(local_inst)-1:
                self.log.debug("Converted {} of {} instances in a total time of {}".format(
                                            len(times), len(local_inst), sum(times)))

        self.log.info("Mean time conversion: {}, std: {}".format(
                        np.mean(np.array(times)), np.std(np.array(times))))
        return X, times

    def _convert(self, path):
        """
        Converts one text-file according to the options into an image (in
        numpy-format)

        Arguments:
            path -- str
                path to instance to be converted

        Returns:
            inst_img -- numpy array
                Image data for instance
        """
        start = time.clock()

        # Read in file
        if self.unpack and path.endswith(".gz"):
            with gzip.open(path, 'rb') as f:
                inst = f.readlines()
        elif self.unpack and path.endswith(".bz2"):
            with bz2.open(path, 'rb') as f:
                inst = f.readlines()
        else:
            with open(path, 'r') as f:
                inst = f.readlines()

        #TODO remove comments

        # Turn into ascii-vector
        inst = "".join(inst)
        inst = np.array([ord(c) for c in inst])

        # Reshape to sqrt(n) according to roundingMethod
        if self.round_method == "ceil":      edge = math.ceil(math.sqrt(len(inst)))
        elif self.round_method == "floor":   edge = math.floor(math.sqrt(len(inst)))
        elif self.round_method == "closest": edge = round(math.sqrt(len(inst)))
        else: raise ValueError("{} is not a valid option for rounding-method."
                               "Use \"ceil\", \"floor\" or \"closest\".")
        inst = np.resize(inst, (edge, edge))

        # Turn into greyscale-image
        image = Image.fromarray(inst.astype('uint8')).convert('L')

        # Resize to the needed dimension
        if self.resize_method == "LANCZOS":
            image = image.resize((self.image_dim, self.image_dim), Image.LANCZOS)
        elif self.resize_method == "BILINEAR":
            image = image.resize((self.image_dim, self.image_dim), Image.BILINEAR)
        elif self.resize_method == "BICUBIC":
            image = image.resize((self.image_dim, self.image_dim), Image.BICUBIC)
        elif self.resize_method == "NEAREST":
            image = image.resize((self.image_dim, self.image_dim), Image.NEAREST)
        elif self.resize_method == "NONE":
            pass
        else: raise ValueError("{} is not a valid option for resize-method. Use \"LANCZOS\", \"BILINEAR\", \"BICUBIC\" or \"NEAREST\".")

        stop = time.clock()

        return np.array(image), stop-start
