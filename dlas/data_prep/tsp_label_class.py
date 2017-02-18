import numpy as np

from dlas.data_prep.label_prep import LabelPrep

class TSPLabelClass(LabelPrep):
    """
    Implements labeling from instance-names
    """
    # TODO: implement options
    def __init__(self, config, aslib):
        """
        Base-class provides aslib-instance (self.aslib)
        """
        super(TSPLabelClass, self).__init__(config, aslib)

    def get_label_data(self, inst):
        """
        Arguments:
            inst -- list of strings
                aslib-instances

        Returns:
            y -- numpy.array
                image-data
        """
        y = np.array([])
        for i in inst:
            if "morphed" in self.aslib.local_path(i):
                labels = [1, 0, 0]
            elif "netgen" in self.aslib.local_path(i):
                labels = [0, 1, 0]
            elif "rue" in self.aslib.local_path(i):
                labels = [0, 0, 1]
            else:
                raise ValueError("{} does not fit any class".format(i))
            y = np.append(y, labels)
        return y
