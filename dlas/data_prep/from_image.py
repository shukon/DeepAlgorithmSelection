import time

from PIL import Image
import numpy as np

from dlas.data_prep.image_prep import ImagePrep

class FromImage(ImagePrep):
    """
    Implements image conversion from text-file to image-file.
    """
    # TODO: implement options
    def __init__(self, image_dim=128):
        super(FromImage, self).__init__()
        self.image_dim = image_dim

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
            img = img.convert('L')
            img = img.resize((self.image_dim,self.image_dim) ,Image.LANCZOS)
            #img_path, ext = os.path.splitext(i)
            #if save: tmp.save(f+"_resized-"+str(imgDim)+".jpeg","JPEG")
            stop = time.clock()
            data = np.append(data, np.array(img))
            times.append(stop-start)
        return data, times
