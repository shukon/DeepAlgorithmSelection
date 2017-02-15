# Config file. Run in python once to validate and make pickle-file.

import os
import pickle
import sys
import logging as log


def conf(NET_CONF_ID, updates = []):
    """ This function creates a configuration with basic parameters, which will
    (at the end of the function) be overwritten with the values in the updates-
    dictionary. """
    config = {}

    config["scen"] = None
    config["ID"]   = "default"
    config["runID"] = NET_CONF_ID  # Must be unique!

    config["repetition"] = 0

    # Image Conversion
    config["image-id"] = "text2image"
    config["unpack"] = True          # unpack before conversion
    config["remove-comments"] = True  # remove comments (enforced by data-prep)
    config["image-dim"] = 128
    config["rounding-method"]="ceil"
    config["resize-method"]="LANCZOS"

    config["scale"] = 1  # TODO ??

    # Neural network configs:
    config["nn-model"] = "cnn"
    config["nn-numEpochs"] = 100
    config["nn-batchsize"] = 128
    config["nn-update-method"] = "nesterov"
    config["nn-lossfunction"] = "binary_crossentropy"
    config["nn-learningrate-start"] = 0.3
    config["nn-learningrate-stop"] = 0.3
    config["nn-momentum-start"] = 0.9
    config["nn-momentum-stop"] = 0.9
    """
    config["epsilon"] = 1e-08
    config["rho"] = 0.95
    config["beta1"], config["beta2"] = 0.9, 0.999
    """


    config["useValidationSet"] = True
    config["useCVforValidation"] = False  # TODO ??

    # DO NOT CHANGE VALUES BELOW MANUALLY EXCEPT YOU KNOW WHAT YOU ARE DOING
    config = update(config, updates)
    return config

def update(config, updates = []):
    for k, v in updates:
        #log.debug(k, v)
        config[k] = v

    # results/{scen}/{expID}/{repitition}/
    config["resultPath"]      = "results/{}/{}/{}/"  # .format(scen,id,rep) (done in main)
    config["modelPath"]       = "results/{}_"+config["runID"]+"_{}of{}_rep{}_model.npz"  # .format(scen, currect-fold, total-folds, repetition)

    save(config)
    return config

def save(config):
    confPath = "configs/config_{}_{}.pickle".format(config["scen"], config["runID"])

    with open(confPath, "wb") as handle:
        pickle.dump(config, handle)
