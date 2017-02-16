import logging as log

class LabelPrep(object):
    """
    Base class for Label constructor.
    """

    def __init__(self, config, aslib, output_dir=None):
        self.log = log.getLogger("LabelPrep")
        self.config = config
        self.aslib = aslib
        self.label_mode = config["label-mode"]

    def get_label_data(self, inst):
        """
        Arguments:
            inst -- list of strings
                aslib-instances

        Returns:
            y -- numpy.array
                image-data
        """
        raise NotImplementedError()
