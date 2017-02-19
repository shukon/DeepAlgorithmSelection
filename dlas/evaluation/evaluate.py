from __future__ import print_function

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging as log
import random
import re
import pickle
from scipy.stats import f_oneway
from matplotlib.backends.backend_pdf import PdfPages
from operator import itemgetter

from dlas.aslib.aslib_handler import ASlibHandler

class Evaluator:
    """ Evaluates experiments that have been performed and produces some nice
    overviews. Experiments are always defined as a tuple (scen, id)

    Provides: PAR10-scores, #of wrong classifications, % of solved instances
              as table or plot over time.
    """
    def __init__(self):
        self.basePath = "results/{}/{}/"

        self.aslib = ASlibHandler()
        with open("aslib_loaded.pickle", "rb") as f: self.aslib.data = pickle.load(f)

    def print_table(self, scen, ID):
        """ Returns a string which contains a table with the following
        information:
        Scen | ID | PAR10 (epoch) | misclass (epoch) | %solved (epoch)
        where PAR10, misclass and %solved are the best values over all epochs
        with the corresponding epoch in brackets.
        The underlying data over all instances is taken from the
        validation-folds of the crossvalidation. Each instance is validated
        against exactly once (assumption!). The per-instance performance is then
        averaged over the repetitions.

        Args:
            scen -- string, scenario-name (e.g. "SAT11-INDU")
            ID -- string, experiment-id (e.g. "test")

        Returns:
            list with values as documented in doc-string 
        """
        par10_per_epoch = self._get_par10_per_epoch(scen, ID)
        par10 = (min(par10_per_epoch), par10_per_epoch.index(min(par10_per_epoch)))
        percent_solved_per_epoch = self._get_percent_solved_per_epoch(scen, ID)
        percent_solved = (max(percent_solved_per_epoch),
                percent_solved_per_epoch.index(max(percent_solved_per_epoch)))
        misclassified_per_epoch = self._get_misclassified_per_epoch(scen, ID)
        misclassified = (min(misclassified_per_epoch), misclassified_per_epoch.index(min(misclassified_per_epoch)))

        return [scen, ID, round(par10[0][0],2), par10[1],
                round(percent_solved[0][0],4), percent_solved[1],
                round(misclassified[0][0], 2), misclassified[1]]

    def _get_num_epo(self, scen, ID):
        """Returns number of epochs"""
        config = pickle.load(open(self.basePath.format(scen,ID)+"0/config.p"))
        return int(config["nn-numEpochs"])

    def _get_par10_per_epoch(self, scen, ID):
        num_epo = self._get_num_epo(scen, ID)
        result = []
        for epoch in range(self._get_num_epo(scen, ID)):
            result.append(self._get_score_over_reps(scen, ID,"par10","val",epoch))
        return result

    def _get_percent_solved_per_epoch(self, scen, ID):
        num_epo = self._get_num_epo(scen, ID)
        result = []
        for epoch in range(self._get_num_epo(scen, ID)):
            result.append(self._get_score_over_reps(scen, ID,"percent_solved","val",epoch))
        return result

    def _get_misclassified_per_epoch(self, scen, ID):
        num_epo = self._get_num_epo(scen, ID)
        result = []
        for epoch in range(self._get_num_epo(scen, ID)):
            result.append(self._get_score_over_reps(scen, ID,"misclassified","val",epoch))
        return result

    def get_pred(self, scen, ID, rep=0, epoch=-1):
        """ Load and zip up instances and predictions for val and test. Flatten
        cv-folds, so that for validation and test there are lists which each
        contain all instances
        Returns two list in which each instance is mapped to its prediction,
        once for validation and once for test.

        Arguments:
            resultPath -- str
                Path to output_dir
            epoch -- int
                Epoch to be loaded (-1 as default for the LAST epoch)
        Returns:
            (val, test) -- (Tuple, Tuple)
                Two tuples, each containing two arrays in turn:
                (instance_names, prediction_array)
                e.g.: ("inst0021", [0.92, 0.34, 0.5])
        """
        resultPath = os.path.join(self.basePath.format(scen,ID), rep) + "/"
        trainPred =  [a for b in [fold[epoch] for fold in np.load(resultPath+"trainPred.npz")["trainPred"]] for a in b]
        valPred =  [a for b in [fold[epoch] for fold in np.load(resultPath+"valPred.npz")["valPred"]] for a in b]
        testPred = [a for b in [fold[epoch] for fold in np.load(resultPath+"testPred.npz")["testPred"]] for a in b]
        valInst =  np.load(resultPath+"instInFoldVal.npz")["valFolds"].flatten()
        testInst = np.load(resultPath+"instInFoldTest.npz")["folds"].flatten()
        val  = (valInst, valPred)
        test = (testInst, testPred)
        return val, test

    def _get_score_over_reps(self, scen, ID, metric="par10", mode="val", epoch=-1):
        """ Calculates PAR10-score over repetitions for a certain epoch.
        val for validation, test for test. """
        # Determine number of repetitions from folder-structure:
        path = "results/{}/{}/".format(scen,ID)
        repetitions = next(os.walk(path))[1]
        mean, std = [], []
        for rep in repetitions:
            # Zip instances to their (for now final) scores:
            val, test = self.get_pred(scen, ID, rep, epoch)
            if val:    inst,pred = val
            elif test: inst,pred = test
            else: raise ValueError("{} is not a valid instance-subset.".format(mode))
            # Turn prediction ([0.2, 0.4, 0.5]) into choice by index (-> 2)
            if not metric == "misclassified":
                pred = [np.argmax(x) for x in pred]
            rep_mean, rep_std = self.aslib.evaluate(scen, list(inst),
                    list(pred), mode=metric)
            mean.append(rep_mean)
            std.append(rep_std)
        # We average over the repetitions
        return np.mean(mean), np.mean(std)

    def scoresPerEpoch(scen, ID, mode="val"):
        basePath = "results/{}/{}/".format(scen, ID)
        with open(basePath + "0/config.p", 'rb') as f:
            config = pickle.load(f)
        return zip(*[getScore(scen, ID, epoch=e) for e in range(config["numEpochs"])])
    
    def getLoss(self, scen, ID):
        """ Calculates the loss-values for a given experiment.
        Always a list for all epochs averaged over folds.

        Returns:
            (trainLoss, valLoss, testLoss) -- three lists of losses averaged
            over folds and repetitions.
        """
        resultPath = self.basePath.format(scen, ID)
        reps = next(os.walk(resultPath))[1]
        trainLoss, valLoss, testLoss = [], [], []
        for rep in reps:
            trainLoss.append(np.load(resultPath+rep+"/"+"trainLoss.npz")["trainLoss"])
            valLoss.append(np.load(resultPath  +rep+"/"+"valLoss.npz")["valLoss"])
            testLoss.append(np.load(resultPath +rep+"/"+"testLoss.npz")["testLoss"])
        return np.mean(trainLoss, axis=0), np.mean(valLoss, axis=0), np.mean(testLoss, axis=0)

    def printLine(scen, ID, options=["VBS","net","BSS","rand","SOA"]):
        line = scen + " " + ID
        for o in options:
            line += " "
            if o == "VBS":    line += str(getVBS(scen)[0])
            elif o == "BSS":  line += str(getBSS(scen)[0])
            elif o == "rand": line += str(getRandom(scen)[0][0]) + " " + str(getRandom(scen)[1][0])
            elif o == "SOA":  line += str(getSOA(scen))
            elif o == "net":  line += str(min(scoresPerEpoch(scen, ID)[0]))
            else: raise Exception("{} is not an option.".format(o))
        return line
    
    def plot(scen, ID, train=True, val=True, test=False):
        """
        Plot train-/valLoss vs PAR10 over epochs.
        """
        # Get data
        trainLoss   = getLoss(scen, ID)[0]
        trainLossMean, trainLossStd   = np.mean(trainLoss, axis=0), np.std(trainLoss, axis=0)
        valLoss   = getLoss(scen, ID)[1]
        valLossMean, valLossStd   = np.mean(valLoss, axis=0), np.std(valLoss, axis=0)
        valPAR10mean, valPAR10std = scoresPerEpoch(scen, ID, mode="val")  # per epoch
    
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        numEpochs = range(len(valLossMean))
        ax1.plot(numEpochs, valLossMean, 'b-')
        ax1.plot(numEpochs, trainLossMean, 'b+')
        ax1.set_xlabel('epochs')
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('validation loss (-), train loss (+)', color='b')
        for tl in ax1.get_yticklabels(): tl.set_color('b')
        # Make twin for PAR10
        ax11 = ax1.twinx()
        ax11.plot(numEpochs, valPAR10mean, 'r')
        ax11.set_ylabel('PAR10', color='r')
        for tl in ax11.get_yticklabels(): tl.set_color('r')
        # Second plot
        ax2.plot(numEpochs, valLossMean, 'b-')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('validation loss (-), train loss (+)', color='b')
        for tl in ax2.get_yticklabels(): tl.set_color('b')
        # Make twin for PAR10
        ax21 = ax2.twinx()
        ax21.plot(numEpochs, valPAR10mean, 'r')
        ax21.set_ylabel('PAR10', color='r')
        fig.tight_layout()
        plt.show()
        return

    def permutationTest(self, *args):
        if len(args)==2 and len(args[0]) > 0 and len(args[1]) > 0: data1, data2 = args
        else: return 20
        def test(x,y): return abs((sum(x)/float(len(x))) - (sum(y)/float(len(y))))
        t = test(data1,data2)
        pool = zip(data1,data2)
        s = []
        for count in range(10000):
            X,Y = [],[]
            index = [0,1]
            for shark in pool:
                # PANIC ~~~/\~~~
                # ~~~~~~~~~~~~~~ 
                random.shuffle(index)
                X.append(shark[index[0]])
                Y.append(shark[index[1]])
            s.append(test(X,Y))
        p = len([z for z in s if z > t])/float(len(s))
        print(" "+str(p),end='')
        return p

if __name__ == "__main__":
    scen = "TSP-NO-EAXRESTART"
    scen = "TSP"
    ID = "test"
    eva = Evaluator()
    print(eva.print_table(scen, ID))
    #plot(scen, ID)
