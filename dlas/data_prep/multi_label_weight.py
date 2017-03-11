import numpy as np

from dlas.data_prep.multi_label_base import MultiLabelBase

class MultiLabelWeight(MultiLabelBase):
    """
    Implements labeling of instances, but uses weights on timeout and score.
    """
    # TODO: implement options
    def __init__(self, config, aslib):
        """
        Base-class provides aslib-instance (self.aslib)
        """
        super(MultiLabelWeight, self).__init__(config, aslib)
        self.label_norm = config["label-norm"]
        self.weight_timeout = config["label-weight-timeout"]
        self.weight_best = config["label-weight-best"]
        assert(self.weight_timeout+self.weight_best == 1)
        self.id = "-".join([config["scen"], self.label_mode, self.label_norm,
                            str(self.weight_timeout),
                            str(self.weight_best)])

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
        cutoff = self.aslib.scen_info[self.config["scen"]]["cutoffTime"]

        for i in inst:
            par10labels = self.aslib.get_labels(self.config["scen"], i, label="par10")
            times = np.array([cutoff*10-l for l in par10labels])
            if self.label_norm == "TimesGood":
                times_weighted = times/float(cutoff*10)
            else:
                raise ValueError("{} not recognized as normalization-strategy".format(self.label_norm))
            best = np.array([1 if l == min(par10labels) and l < cutoff else 0 for l in par10labels])
            print(best)
            labels = times_weighted * self.weight_timeout + best * self.weight_best
            print(labels)

            y = np.append(y, labels)
            """
            elif self.label_mode == "base": y = np.append(y, aslib.getLabels(scen, i, label="status"))
            elif self.label_mode == "OnlyBest": y = np.append(y, [1 if l == min(par10labels) and l < aslib.scenInfo[scen]["cutoffTime"] else 0 for l in par10labels])
            elif self.label_mode == "TimesMax": y = np.append(y, norm("max", [aslib.scenInfo[scen]["cutoffTime"]*10-l for l in par10labels]))
            elif self.label_mode == "TimesSum": y = np.append(y, norm("sum", [aslib.scenInfo[scen]["cutoffTime"]*10-l for l in par10labels]))
            elif self.label_mode == "TimesLog": y = np.append(y, norm("log", [(aslib.scenInfo[scen]["cutoffTime"]*10)-l for l in par10labels]))
            """
        return y
        """
        def norm(self):
            if mode == "log": return [k/float(aslib.scenInfo[scen]["cutoffTime"]) for k in l]
            if mode == "max": return [k/max(l) if max(l) != 0 else 0 for k in l]
            if mode == "sum": return [k/sum(l) if sum(l) != 0 else 0 for k in l]
        """
