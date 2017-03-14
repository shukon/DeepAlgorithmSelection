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
    def __init__(self, scen, ID, updates=None):
        """ This function creates a configuration with default parameters, which will
        (at the end of the function) be overwritten with the values in the updates-
        dictionary. """
        if isinstance(updates, str):
            updates = self.dict_from_file(scen, ID)
        elif isinstance(updates, dict):
            pass

        with open("dlas/dlas.pcs", 'r') as f:
            configspace = pcs.read(f.readlines())
        self.default_config = configspace.get_default_configuration()
        config_dict = self.default_config.get_dictionary()
        config_dict.update(updates)
        self.config = Configuration(configspace, config_dict)
        self.scen = scen
        self.ID = ID

    def __getitem__(self, attr):
        return self.config[attr]

    def dict_from_file(self, s, ID):
        with open("experiments/{}.txt".format(ID), 'r') as f:
            content = [line.strip("\n").split("=") for line in f.readlines]
            content = [(name.strip(), value.strip()) for name, value in content]
            print(dict(content))
            return dict(content)

if __name__ == "__main__":
    c = Config("TestScen", "TestID")
    print(c["image-dim"])
    print(c.config)
