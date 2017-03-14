# Config file. Run in python once to validate and make pickle-file.

import os
import pickle
import sys
import logging as log
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.io import pcs
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

class Config(object):
    """ This class is an extension of the ConfigurationSpace-Configuration,
    introducing module-members scenario, id, repetition etc. """
    def __init__(self, scen, ID, repetition=0, updates=""):
        """ This function creates a configuration with default parameters, which will
        (at the end of the function) be overwritten with the values in the updates-
        dictionary. """
        if isinstance(updates, str):
            updates = self.dict_from_file(scen, ID)
        elif not isinstance(updates, dict):
            raise ValueError("updates to Config must be of type str (for"
                             "filepath) or dict.")

        with open("dlas/dlas.pcs", 'r') as f:
            configspace = pcs.read(f.readlines())
        self.default_config = configspace.get_default_configuration()
        config_dict = self.default_config.get_dictionary()
        config_dict.update(updates)
        self.config = Configuration(configspace, config_dict)
        self.scen = scen
        self.ID = ID
        self.rep = repetition
        self.use_validation = True
        self.result_path = "results/{}/{}/{}/".format(self.scen, self.ID, self.rep)

    def __getitem__(self, attr):
        return self.config[attr]

    def dict_from_file(self, s, ID):
        with open("experiments/{}.txt".format(ID), 'r') as f:
            content = f.readlines()
            content = [tuple(line.strip("\n").split("=")) for line in content if line != "\n"]
            print(content)
            content = [(name.strip(), value.strip()) for name, value in content]
            content = dict(content)
            for c in content:
                try:
                    content[c] = float(content[c])
                    if content[c].is_integer():
                        print(content[c])
                        content[c] = int(content[c])
                except ValueError:
                    pass
            print(dict(content))
            return dict(content)

if __name__ == "__main__":
    c = Config("TestScen", "TestID")
    print(c["image-dim"])
    print(c.config)
