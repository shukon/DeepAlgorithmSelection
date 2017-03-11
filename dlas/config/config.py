# Config file. Run in python once to validate and make pickle-file.

import os
import pickle
import sys
import logging as log


def conf(scen, ID, updates = []):
    """ This function creates a configuration with default parameters, which will
    (at the end of the function) be overwritten with the values in the updates-
    dictionary. """
    config = {}

    # An experiment is uniquely specified by the scenario-name and the ID
    config["scen"] = scen
    config["ID"]   = ID

    config["repetition"] = 0

    # Image Conversion
    config["image-mode"] = "FromImage"
    config["unpack"] = True          # unpack before conversion
    config["remove-comments"] = True  # remove comments (enforced by data-prep)
    config["image-dim"] = 128
    config["round-method"]="ceil"
    config["resize-method"]="LANCZOS"

    # Labels
    config["label-mode"] = "MultiLabelBase"
    config["label-norm"] = "TimesGood"
    config["num-labels"] = "num-solvers"
    # for "MultiLabelWeight":

    # Neural network configs:
    config["nn-model"] = "cnn"
    config["nn-conv-size-one"] = 3,
    config["nn-conv-size-two"] = 2
    config["nn-numEpochs"] = 100
    config["nn-batchsize"] = 128
    config["nn-update-method"] = "nesterov"
    config["nn-lossfunction"] = "binary_crossentropy"
    config["nn-learningrate-start"] = 0.3
    config["nn-learningrate-stop"] = 0.3
    config["nn-momentum-start"] = 0.9
    config["nn-momentum-stop"] = 0.9
    config["nn-regression"] = True
    config["nn-output-nonlinearity"] = "sigmoid"
    """
    config["epsilon"] = 1e-08
    config["rho"] = 0.95
    config["beta1"], config["beta2"] = 0.9, 0.999
    """
    config["useValidationSet"] = True

    # DO NOT CHANGE VALUES BELOW MANUALLY EXCEPT YOU KNOW WHAT YOU ARE DOING
    config = update(config, updates)
    return config

def update(config, updates = []):
    for k, v in updates:
        if not k in config:
            log.warning(str(k)+" = "+str(v)+" is not a standard option.")
        config[k] = v

    # If the number of labels is equal the number of solvers, it suffices to
    # set num-labels to "num-solvers"
    if config["num-labels"] == "num-solvers" and "num-solvers" in config:
        log.debug("Setting number of labels to number of solvers")
        config["num-labels"] = config["num-solvers"]

    # results/{scen}/{expID}/{repitition}/
    config["resultPath"] = "results/{}/{}/{}/".format(config["scen"], config["ID"],
                                       config["repetition"])

    save(config)
    return config

def save(config):

    if not os.path.exists(config["resultPath"]):
        os.makedirs(config["resultPath"])

    with open(os.path.join(config["resultPath"], "config.p"), "wb") as handle:
        pickle.dump(config, handle)
