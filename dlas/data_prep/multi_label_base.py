import numpy as np

from dlas.data_prep.label_prep import LabelPrep

class MultiLabelBase(LabelPrep):
    """
    Implements image conversion from text-file to image-file.
    """
    # TODO: implement options
    def __init__(self, label_mode="TimesGood"):
        """
        Base-class provides aslib-instance (self.aslib)
        """
        super(MultiLabelBase, self).__init__()
        self.label_mode = label_mode

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
        cutoff = None # TODO

        for i in inst:
            par10labels = aslib.getLabels(scen, i, label="PAR10")
            time_weighted = [cutoff*10-l for l in par10labels]

            if self.label_mode == "TimesGood":
                labels = [k/float(cutoff*10) for k in time_weighted]
                y = np.append(y, labels)
            else:
                raise ValueError("{} not recognized as labelID".format(self.label_mode))
            """
            elif self.label_mode == "base": y = np.append(y, aslib.getLabels(scen, i, label="status"))
            elif self.label_mode == "OnlyBest": y = np.append(y, [1 if l == min(par10labels) and l < aslib.scenInfo[scen]["cutoffTime"] else 0 for l in par10labels])
            elif self.label_mode == "TimesMax": y = np.append(y, norm("max", [aslib.scenInfo[scen]["cutoffTime"]*10-l for l in par10labels]))
            elif self.label_mode == "TimesSum": y = np.append(y, norm("sum", [aslib.scenInfo[scen]["cutoffTime"]*10-l for l in par10labels]))
            elif self.label_mode == "TimesLog": y = np.append(y, norm("log", [(aslib.scenInfo[scen]["cutoffTime"]*10)-l for l in par10labels]))
            """
        np.save(labelPath, y)
        return data, times
        """
        def norm(self):
            if mode == "log": return [k/float(aslib.scenInfo[scen]["cutoffTime"]) for k in l]
            if mode == "max": return [k/max(l) if max(l) != 0 else 0 for k in l]
            if mode == "sum": return [k/sum(l) if sum(l) != 0 else 0 for k in l]
        """
