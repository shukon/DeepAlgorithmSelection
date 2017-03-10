DLAS aims to be a framework for experiments on Algorithm Selection, mainly using Deep Neural Networks.
DLAS is based on the Algorithm Selection Library (ASlib, https://github.com/coseal/aslib_data).

The framework is under heavy construction and may not be working correctly at all times, backwards-compatibility is not guaranteed. To perform experiments, define the options in an experiment in dlas/config/experiments.

General layout:
Experiments are defined via a number of options. The default-values for all those options are defined in dlas/config/config.py. To run an experiment, define it in dlas/config/experiments, you can use the dlas/main.py - interface, which provides several methods:
`python dlas/main.py exp SCENARIO ID` -- this starts an experiment, that you have to define in dlas/config/experiments.py
`python dlas/main.py eval SCENARIO ID` -- will attempt to evaluate the results including calculating PAR10 and percentage of solved instances, as well as plots on the loss and PAR10 over the epochs of the neural network.
`python dlas/main.py stat SCENARIO` -- prints a number of scenario-statistics, like the distribution of solvers in VBS and the general strength of solvers.
`python dlas/main.py prep SCENARIO ID` -- will convert the instances and create the labels according to specifications in the experiment
