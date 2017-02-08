import logging as log

class LabelPrep(object):
    """
    Base class for Label constructor.
    """

    def __init__(self, inst_path, output_dir=None):
        self.log = log.getLogger("LabelPrep")

        self.inst_path = inst_path
        self.output_dir = output_dir

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
