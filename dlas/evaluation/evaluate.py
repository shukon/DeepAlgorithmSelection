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

import ASlibHandler

class Evaluator:
    """ Evaluates experiments that have been performed and produces some nice
    overviews. Experiments are always defined as a tuple (scen, id)
    """
    def __init__(self):
        self.basePath = "results/{}/{}/"

        aslib = ASlibHandler.ASlibHandler()
        with open("aslib_loaded.pickle", "rb") as f: aslib.data = pickle.load(f)

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

    def getPred(resultPath, epoch=-1):
        """ Load and zip up instances and predictions for val and test. Flatten cv-folds.
            Returns two list in which each instance is mapped to its prediction,
            once for validation and once for test.
            Arguments:
                resultPath -- str
                    Path to output_dir
                epoch -- int
                    Epoch to be loaded (-1 as default for the LAST epoch)
            Returns:
                (val, test) -- Tuple(Tuple, Tuple)
                    Two tuples, each containing two arrays in turn:
                    (instance_names, prediction_array)
                    e.g.: ("inst0021", [0.92, 0.34, 0.5])
        """
        trainPred =  [a for b in [fold[epoch] for fold in np.load(resultPath+"trainPred.npz")["trainPred"]] for a in b]
        valPred =  [a for b in [fold[epoch] for fold in np.load(resultPath+"valPred.npz")["valPred"]] for a in b]
        testPred = [a for b in [fold[epoch] for fold in np.load(resultPath+"testPred.npz")["testPred"]] for a in b]
        valInst =  np.load(resultPath+"instInFoldVal.npz")["valFolds"].flatten()
        testInst = np.load(resultPath+"instInFoldTest.npz")["folds"].flatten()
        val  = (valInst, valPred)
        test = (testInst, testPred)
        return val, test
    
    def getScore(scen, ID, mode="val", epoch=-1, inst2use=None):
        """ Calculates PAR10-score over repetitions for a certain epoch.
        val for validation, test for test. """
        # Determine number of repetitions from folder-structure:
        reps = next(os.walk(basePath))[1]
        mean, std = [], []
        for rep in reps:
            path = os.path.join(basePath, rep) + "/"
            # Zip instances to their (for now final) scores:
            val, test = getPred(path, epoch)
            if val:    inst = val
            elif test: inst = test
            else: raise Exception("{} not valid mode (getScore, evaluate)".format(mode))
            # Turn prediction ([0.2, 0.4, 0.5]) into choice by index (2)
            inst = zip(inst[0], inst[1])
            inst = [(x[0], np.argmax(x[1])) for x in inst[1]]
            # Use only specific instances
            if inst2use:
                inst = [x for x in inst if x[0] in inst2use]
            meanTMP, stdTMP = aslib.evaluate(scen, list(inst[0]), list(inst[1]))
            mean.append(meanTMP)
            std.append(stdTMP)
        # We average over the repetitions
        print(len(mean))
        return np.mean(mean), np.mean(std)
    
    def scoresPerEpoch(scen, ID, mode="val"):
        basePath = "results/{}/{}/".format(scen, ID)
        with open(basePath + "0/config.p", 'rb') as f:
            config = pickle.load(f)
        return zip(*[getScore(scen, ID, epoch=e) for e in range(config["numEpochs"])])
    
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
                random.shuffle(index)
                X.append(shark[index[0]])
                Y.append(shark[index[1]])
            s.append(test(X,Y))
        p = len([z for z in s if z > t])/float(len(s))
        print(" "+str(p),end='')
        return p


if __name__ == "__main__":
    scen = "TSP"
    scen = "TSP-NO-EAXRESTART"
    ID = "TSP-100-baseLabels"
    aslib.indiSolvers(scen)
    print(printLine(scen, ID))
    plot(scen, ID)
