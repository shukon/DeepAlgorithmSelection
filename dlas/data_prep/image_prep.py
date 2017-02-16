import logging as log

class ImagePrep(object):
    """
    Base class for image converter.
    """

    def __init__(self, config, output_dir=None):
        self.log = log.getLogger("ImagePrep")
        self.image_mode = config["image-mode"]

    def get_image_data(self, local_inst):
        """
        Arguments:
            local_inst -- list of strings
                local paths to instances

        Returns:
            image_data -- numpy.array
        """
        raise NotImplementedError()
