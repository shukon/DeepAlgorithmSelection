import time

from PIL import Image
import numpy as np

from dlas.data_prep.image_prep import ImagePrep

class FromImage(ImagePrep):
    """
    Implements image conversion from text-file to image-file.
    """
    # TODO: implement options
    def __init__(self, config):
        super(FromImage, self).__init__(config)
        self.image_dim = config["image-dim"]
        if config["resize-method"] == 'LANCZOS':
            self.resize_method = Image.LANCZOS
        else:
            raise ValueError("{} is not a valid resize-option for FromImage-conversion".format(config["resize-method"]))

        self.id = "-".join([self.image_mode,
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
            start = time.clock()
            img = Image.open(i)
            #self.log.debug("Image nan: " + str(np.isnan(np.array(img)).any()))
            img = img.convert('L')
            #self.log.debug("Image nan after convert: " + str(np.isnan(np.array(img)).any()))
            img = img.resize((self.image_dim, self.image_dim), Image.LANCZOS)
            #self.log.debug("Image nan after resize: " + str(np.isnan(np.array(img)).any()))
            if (np.isnan(np.array(img)).any()):
                self.log.debug("NAN: {}".format(i))
            #img_path, ext = os.path.splitext(i)
            #if save: tmp.save(f+"_resized-"+str(imgDim)+".jpeg","JPEG")
            stop = time.clock()
            data = np.append(data, np.array(img))
            times.append(stop-start)
        return data, times
