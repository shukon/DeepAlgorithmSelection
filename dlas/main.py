#!/usr/bin/env python

""" Main execution script for DLAS.
Here the basic experiment functions are defined and the interaction between user
and module happens. """
from __future__ import print_function

import os
import sys
import logging
import pickle
import numpy as np
import inspect

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from dlas.neural_net.network import Network
from dlas.data_prep.data_prep import DataPreparer
from dlas.config.config import Config
from dlas.aslib.aslib_handler import ASlibHandler
from dlas.evaluation.evaluate import Evaluator

ASLIB = ASlibHandler()

def setupLogging(output_dir, verbose_level):
    # Log to stream (console)
    logging.getLogger().setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)
    if verbose_level == "INFO":
        stdout_handler.setLevel(logging.INFO)
    else:
        stdout_handler.setLevel(logging.DEBUG)
        if verbose_level == "DEV_DEBUG":
            # Disable annoying boilerplate-debug-logs from foreign modules
            disable_loggers = []
            for logger in disable_loggers:
                logging.getLogger().debug("Setting logger \'%s\' on level INFO", logger)
                logging.getLogger(logger).setLevel(logging.INFO)
    logging.getLogger().addHandler(stdout_handler)
    # Log to file
    if not os.path.exists(os.path.join(output_dir, "debug")):
        os.makedirs(os.path.join(output_dir, "debug"))
    fh = logging.FileHandler(os.path.join(output_dir, "debug/debug.log"), "w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    # Color warnings and errors
    for lvl in [logging.WARNING, logging.ERROR]:
        logging.addLevelName(lvl, "\033[1;31m%s\033[1;0m" % logging.getLevelName(lvl))

def prep(scen, config, instance_path, recalculate = False):
    """
    Responsible for preparing data for experiment, that is:
      - Generating the data (images and labels) according to specifics in config-file
      - Image conversion done by Converter-Class
      - Labels calculated using ASlib-data
      - Image = X, label = y, image-format in (#inst,#channels,dim1,dim2)

    Args:
        scen : string
            -- specifies scenario
        config : dict
            -- options for this experiment
        instance_path : string
            -- path to local instances
        recalculate : bool
            -- if True, recalculate  image and labels

    Returns:
        inst : list of instances
        X    : image-data, format in (#inst,#channels,dim1,dim2)
        y    : label-data
    """
    # Load scenario-data
    if config["image-mode"] == "TextToImage":
        ASLIB.load_scenario(scen, extension="tsp")
    elif config["image-mode"] == "FromImage":
        ASLIB.load_scenario(scen, extension="jpeg")
    # Sorted, INCLUDE ALL INSTANCES, i.e. include timeouts etc.:
    inst = ASLIB.get_instances(scen, remove_unsolved=False)
    preparer = DataPreparer(config, ASLIB, instance_path, "images/", "labels/")

    for folder in ["images", "labels", "results"]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    local_inst = [ASLIB.local_path(scen, i) for i in inst]
    X = preparer.get_image_data(local_inst, recalculate)  #  t = conversion times
    y = preparer.get_label_data(inst, recalculate)

    return inst, X, y

def run_experiment(scen, ID, config, skip_if_result_exists = True):
    """
    Run experiment (that is defined by name through scen and ID), which is
    further defined in config.
    """
    # Load scenario-data
    ASLIB.load_scenario(scen)

    logging.debug(config.config)

    rep = config.rep
    result_path = config.result_path

    if skip_if_result_exists and os.path.exists(result_path):
        logging.info("Skipping experiment for scen {} with ID {} in repetition {}, "
                     "because it seems that it has already been performed.".format(
                      scen, ID, rep))
        return
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # We assume ASlibHandler is matching from execdir
    data = prep(scen, config, "instances/"+scen, recalculate = False)
    inst, X, y = data
    config.num_labels = len(y[0])

    cross_validation(scen, ID, inst, X, y, config, result_path, rep)

def cross_validation(scen, ID, inst, X, y, config, resultPath, rep = 1):
    """
    Performs a 10-fold crossvalidation for a given scenario and configuration
    on a bunch of instances. The results are then saved in multiple result-files.

    Args:
        scen -- string, name of scenario
        ID -- string, custom ID for experiment
        inst -- list of all instances in scenario
        X -- imagedata
        y -- labeldata
        config -- config-dict
        rep -- int, repetition-number
    """
    folds, valFolds = [], []  # Save test/val instances per fold as list: [[f1-i1, f1-i2, ...], [f2-i1, f2-i2, ...], ...]
    trainPred, valPred, testPred  = [], [], []  # Save the predicions on the validation folds (also as nested list)
    trainLoss, valLoss, testLoss  = [], [], []  # Save loss-history per fold [loss1, loss2, ...]
    valAcc, testAcc = [], []

    timesPerEpoch  = []  # Save times per epoch per fold, see above
    timesToPredict = []  # How long it takes for each fold to predict the values on average

    folds = ASLIB.getCVfolds(scen)  # Use folds from ASlib-scenario

    logging.debug("Number of instances in list: {}, number of images in image-data: {}".format(len(inst), len(X)))
    assert(len(X)==len(inst))

    # Check if any folds are overlapping:
    for a in folds:
        for b in folds:
            if a == b: continue
            if not (set(a).isdisjoint(b)):
                raise ValueError("Crossvalidation-folds are not disjoint.")
            if not len(a) == len(b):
                logging.warning("Folds not equal size! {} vs {}".format(len(a), len(b)))

    for test_fold in folds:
        # We use the fold following the current testfold for validation (test 1 -> val 2, ... test 10 -> val 1)
        valInst = folds[(folds.index(test_fold)+1+config.rep)%len(folds)]
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

        logging.info("Now training with crossvalidation, test-fold {} of {}, use Validation: {}, repetition {}.".format(folds.index(test_fold),
                            len(folds), config.use_validation, config.rep))
        net = Network(config)
        # Is this really the best method? ...
        result = net.fit([X_train, X_val, X_test], [y_train, y_val, y_test], [inst_train, inst_val, inst_test])
        # PE=per epoch,TP=times to predict,L=loss,P=prediction,A=accuracy
        if result: timesPE, timesTP, trL, vaL, teL, trP, vaP, teP, vaA, teA = result
        else:
            errorLog ="failedrun_{}_{}_{}.txt\"".format(scen, config.ID,
                    config.rep)
            logging.error("Training failed. Saving {}.".format(errorLog))
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
    logging.debug("Saving losses.")
    np.savez(os.path.join(resultPath, "trainLoss.npz"), trainLoss=trainLoss)
    np.savez(os.path.join(resultPath, "valLoss.npz"), valLoss=valLoss)
    np.savez(os.path.join(resultPath, "testLoss.npz"), testLoss=testLoss)
    logging.debug("Saving predictions.")
    np.savez(os.path.join(resultPath, "trainPred.npz"), trainPred=trainPred)
    np.savez(os.path.join(resultPath, "valPred.npz"), valPred=valPred)
    np.savez(os.path.join(resultPath, "testPred.npz"), testPred=testPred)
    logging.debug("Saving folds")
    np.savez(os.path.join(resultPath, "instInFoldVal.npz"), valFolds=valFolds)
    np.savez(os.path.join(resultPath, "instInFoldTest.npz"), folds=folds)

    with open(os.path.join(resultPath, "config.p"), 'wb') as handle:
        pickle.dump(config.get_dictionary(), handle)
    return

if __name__ == "__main__":

    eva = Evaluator()

    import argparse

    parser = argparse.ArgumentParser(description='Process input to dlas-cmdline.')
    parser.add_argument('--mode', action='store', default='exp',
                        help='Specifies what action should be performed.',
                        choices=['exp', 'eval', 'prep', 'stat'])
    parser.add_argument('--scen', action='store', default='TSP',
                        help='Defines the scenario to use.')
    parser.add_argument('--ID', action='store', default='tsp-default',
                        help='Defines experiment ID, by filename in'
                             '\"experiments/\" (without .txt).')

    args = parser.parse_args()
    args = vars(args)
    scen = args["scen"]
    ID = args["ID"]
    mode = args["mode"]

    setupLogging("tmpOut", "DEBUG")

    ASLIB.load_scenario(scen)
    if mode == "exp":
        if scen == "all":
            scenarios = ["TSP", "TSP-MORPHED", "TSP-NETGEN", "TSP-RUE", "TSP-NO-EAXRESTART", "TSP-MORPHED-NO-EAXRESTART", "TSP-NETGEN-NO-EAXRESTART", "TSP-RUE-NO-EAXRESTART"]
        else:
            scenarios = [scen]
        for s in scenarios:
            c = Config(s, ID)
            run_experiment(s, ID, c, skip_if_result_exists=False)
            print(eva.print_table(s, ID, string=True))
    elif mode == "eval":
        print("Evaluating {}.".format(scen))
        if ID:
            print(eva.print_table(scen, ID, string=True))
            eva.plot(scen, ID)
        else:
            for element in eva.compare_ids_for_scen(scen):
                print(element)
    elif mode == "stat":
        # Print stats of scenario
        ASLIB.load_scenario(scen)
        print("Scenario-statistics for {}:".format(scen))
        print("{} instances, {} solvable.".format(
            len(ASLIB.get_instances(scen, remove_unsolved=False)),
            len(ASLIB.get_instances(scen, remove_unsolved=True))))
        print("Virtual Best Solver: {}".format(ASLIB.baseline(scen)["vbs"]))
        chosen = [ASLIB.get_labels(scen, i, "par10").index(min(ASLIB.get_labels(scen, i, "par10"))) for i in ASLIB.get_instances(scen)]
        print(ASLIB.solver_distribution(chosen, scen))
    elif mode == "prep":
        # Prepare image and label
        c = Config(scen, ID)
        prep(scen, c, "instances/"+scen, recalculate = True)
