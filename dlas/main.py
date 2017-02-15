from __future__ import print_function

import os
import sys
import logging as log
import pickle
import numpy as np

from dlas.neural_net.network import Network
from dlas.data_prep.data_prep import DataPreparer
import dlas.config.config as conf
import dlas.config.experiments as exp
from dlas.aslib.aslib_handler import ASlibHandler


log.basicConfig(level=log.DEBUG)

aslib = ASlibHandler()
with open("aslib_loaded.pickle", "rb") as f:
    aslib.data = pickle.load(f)

def prep(scen, config, instance_path, recalculate = False):
    """
    Responsible for preparing data, that is:
      - Generating the data (images and labels) according to specifics in config-file
      - Image conversion done by Converter-Class
      - Labels calculated using ASlib-data
      - Image = X, label = y, format in (#inst,#channels,dim1,dim2)
    """
    # Sorted, INCLUDE ALL INSTANCES, i.e. include timeouts etc.:
    inst = aslib.get_instances(scen, False)
    preparer = DataPreparer(config, instance_path, "images/", "labels/")

    local_inst = [aslib.local_path(scen,i) for i in inst]
    X = preparer.get_image_data(local_inst, recalculate)  #  t = conversion times
    y = preparer.get_label_data(local_inst, recalculate)

    return inst, X, y

def run_experiment(scen, ID, config, skipIfResultExists = True):
    """
    Run experiment (that is defined by name through scen and ID), which is
    further defined in config.
    """
    NET_CONF_ID = config["runID"]  # THIS IS VERY IMPORTANT - the individual experiment-ID to make sure results are not confused.
    config = conf.update(config, updates = [("scen",scen),("num-solvers",
                len(aslib.get_solvers(scen)))])

    rep = config["repetition"]
    resultPath = config["resultPath"].format(scen, ID, rep)
    if skipIfResultExists and os.path.exists(resultPath):
        log.info("Skipping experiment for scen {} with ID {} in repetition {}, because it seems that it has already been performed.".format(scen,
            ID, rep))
        return
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    # We assume ASlibHandler is matching from execdir
    data = prep(scen, config, ".", recalculate = False)
    inst, X, y = data

    scores = cross_validation(scen, ID, inst, X, y, config, resultPath, rep)
    return(np.mean(scores))

def cross_validation(scen, ID, inst, X, y, config, resultPath, rep = 1):
    """
    Performs a 10-fold crossvalidation for a given scenario and configuration
    on a bunch of instances. The results are then saved in a result-file.

    Args:
        scen -- string, name of scenario
        ID -- string, custom ID for experiment
        inst -- list of all instances in scenario
        X -- imagedata
        y -- labeldata
        config -- config-dict
        output_dir -- string, path to output-directory
        rep -- int, repetition-number
    """
    folds, valFolds = [], []  # Save test/val instances per fold as list: [[f1-i1, f1-i2, ...], [f2-i1, f2-i2, ...], ...]
    trainPred, valPred, testPred  = [], [], []  # Save the predicions on the validation folds (also as nested list)
    trainLoss, valLoss, testLoss  = [], [], []  # Save loss-history per fold [loss1, loss2, ...]
    valAcc, testAcc = [], []

    timesPerEpoch  = []  # Save times per epoch per fold, see above
    timesToPredict = []  # How long it takes for each fold to predict the values on average

    folds = aslib.getCVfolds(scen)  # Use folds from ASlib-scenario

    log.debug("Number of instances in list: {}, number of images in image-data: {}".format(len(inst), len(X)))
    assert(len(X)==len(inst))

    # Check if any folds are overlapping:
    for a in folds:
        for b in folds:
            if a == b: continue
            if not (set(a).isdisjoint(b)): raise ValueError
            if not len(a) == len(b): log.warning("Folds not equal size! {} vs {}".format(len(a), len(b)))

    for test_fold in folds:
        # We use the fold following the current testfold for validation (test 1 -> val 2, ... test 10 -> val 1)
        valInst = folds[(folds.index(test_fold)+1+config["repetition"])%len(folds)]
        trainInst = [i for i in inst if not (i in test_fold or i in valInst)]

        indices_test  = [inst.index(i) for i in test_fold]
        indices_val   = [inst.index(i) for i in valInst]
        indices_train = [inst.index(i) for i in trainInst]

        X_test   = np.array([X[i] for i in indices_test])
        y_test   = np.array([y[i] for i in indices_test])
        X_val    = np.array([X[i] for i in indices_val])
        y_val    = np.array([y[i] for i in indices_val])
        X_train  = np.array([X[i] for i in indices_train])
        y_train  = np.array([y[i] for i in indices_train])

        inst_test  = [inst[i] for i in indices_test]
        inst_val   = [inst[i] for i in indices_val]
        inst_train = [inst[i] for i in indices_train]

        assert(len(X_test)+len(X_val)+len(X_train)==len(inst))
        assert(len(set(inst_test+inst_val+inst_train)) == len(inst))
        assert(set(inst_test+inst_val+inst_train) == set(inst))

        valFolds.append(valInst)

        log.info("Now training with crossvalidation, test-fold {} of {}, use Validation: {}, repetition {}.".format(folds.index(test_fold),
                            len(folds), config["useValidationSet"], config["repetition"]))
        net = Network(config)
        # Is this really the best method? ...
        result = net.fit(X_train, y_train, X_val, y_val, X_test, y_test, config, [inst_train, inst_val, inst_test])
        # PE=per epoch,TP=times to predict,L=loss,P=prediction,A=accuracy
        if result: timesPE, timesTP, trL, vaL, teL, trP, vaP, teP, vaA, teA = result
        else:
            errorLog ="failedrun_{}_{}_{}.txt\"".format(scen, config["runID"], config["repetition"])
            log.error("Training failed. Saving {}.".format(errorLog))
            with open(errorLog, 'w') as f: f.write(errorLog)
            return

        # Append times, losses, predictions and accuracies to the according lists
        timesPerEpoch.append(timesPE)  # Times
        timesToPredict.append(timesTP)
        trainLoss.append(trL)          # Losses
        valLoss.append(vaL)
        testLoss.append(teL)
        trainPred.append(trP)          # Predictions
        valPred.append(vaP)
        testPred.append(teP)
        valAcc.append(vaA)             # Accuracies
        testAcc.append(teA)
    # Pickle config for saving in results
 
    np.savez(os.path.join(resultPath, "timesPerEpoch.npz"), timesPerEpoch=timesPerEpoch)
    np.savez(os.path.join(resultPath, "timesToPredict.npz"), timesToPredict=timesToPredict)
    np.savez(os.path.join(resultPath, "trainLoss.npz"), trainLoss=trainLoss)
    np.savez(os.path.join(resultPath, "valLoss.npz"), valLoss=valLoss)
    np.savez(os.path.join(resultPath, "testLoss.npz"), testLoss=testLoss)
    np.savez(os.path.join(resultPath, "trainPred.npz"), trainPred=trainPred)
    np.savez(os.path.join(resultPath, "valPred.npz"), valPred=valPred)
    np.savez(os.path.join(resultPath, "testPred.npz"), testPred=testPred)
    np.savez(os.path.join(resultPath, "instInFoldVal.npz"), valFolds=valFolds)
    np.savez(os.path.join(resultPath, "instInFoldTest.npz"), folds=folds)

    with open(os.path.join(resultPath, "config.p"), 'wb') as handle:
        pickle.dump(config, handle)
    return

if __name__ == "__main__":
    # Logger config
    log.basicConfig(level = log.DEBUG)
    log.addLevelName( log.WARNING, "\033[1;31m%s\033[1;0m" % log.getLevelName(log.WARNING))  # Color warnings
    log.addLevelName( log.ERROR, "\033[1;31m%s\033[1;0m" % log.getLevelName(log.ERROR))  # Color errors
    scen = sys.argv[2]
    ID = sys.argv[3]
    if sys.argv[1] == "exp":
        c = exp.getConfig(ID, scen)
        run_experiment(scen, c["runID"], c, skipIfResultExists=False)
    elif sys.argv[1] == "stat":
        # Print stats of scenario
        log.info("Scenario-statistics for {}:".format(scen))
        log.info("{} instances, {} solvable.".format(
            len(aslib.get_instances(scen, removeUnsolved=False)),
            len(aslib.get_instances(scen, removeUnsolved=True))))
        #log.info("Solver-Dist.: {}".format(aslib.indiSolvers(scen)))
