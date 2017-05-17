DLAS aims to be a framework for experiments on Algorithm Selection, mainly using Deep Neural Networks.
DLAS is based on the Algorithm Selection Library (ASlib, https://github.com/coseal/aslib_data).

!! The framework is under heavy construction and may not be working correctly at all times, backwards-compatibility is not guaranteed. !!

**Performing Experiments:**
To perform experiments, define the options in an experiment in dlas/config/experiments.py. You can currently see the default-values in dlas/config/config.py.
Experiments consist of a number of options. The default-values for all those options are defined in dlas/config/config.py.
An experiment is uniquely identified by the scenario-name + a custom ID and can be invoked using the **dlas/main.py** interface, which provides the following functionalities:
`python dlas/main.py --mode exp --scen SCENARIO --ID ID` -- starts an experiment, that you have to define in dlas/config/experiments.py
`python dlas/main.py --mode eval --scen SCENARIO --ID ID` -- will attempt to evaluate the results including calculating PAR10 and percentage of solved instances, as well as plots on the loss and PAR10 over the epochs of the neural network.
`python dlas/main.py --mode prep --scen SCENARIO --ID ID` -- will convert the instances and create the labels according to specifications in the experiment
`python dlas/main.py --mode stat --scen SCENARIO --ID ID` -- prints a number of scenario-statistics, like the distribution of solvers in Virtual Best Solver (VBS) and the general strength of solvers.

The configurational setup mainly consists of three parts:
  - image conversion
  - labeling
  - neural architecture
Each part can be easily modified, partly through options and partly by writing the corresponding functions in the modular architecture.

[classdiagramm]

_Image-conversion_: You can choose between two options: `FromImage` and `TextToImage`. `FromImage` needs the instances to be available as images, it is mainly responsible for rescaling. `TextToImage` reads in the instance-files in a text-file-format and converts the ascii-symbols into a greyscale-image (see http://www.cs.toronto.edu/~horst/cogrobo/papers/DLforAP.pdf). For both methods you can define image-dimension and rescaling-method through the config-dictionary. To implement a new class you need only derive it as a class from `ImagePrep` in dlas/data_prep/image_prep.py. The file is required to have the same name as the class. To use your class, simply provide the name of your class as `image_mode`-option.

_Labeling_: Correct labeling is crucial for any learning. Currently, labels are a vector, containing a value between 0 and 1 for each solver, reflecting the solvers ability to solve an instance. `MultiLabelBase` implements this labeling. You can implement new classes simply by introducing a new class derived from `LabelPrep`. The file is required to have the same name as the class. To use your class, simply provide the name of your class as `label_mode`-option.

_Neural Network_: The neural network used so far is a Convolutional Neural Network. It is implemented in Lasagne. The network can be configured using the options starting with `nn-`. `nn-model` defines the class to use for building the network. Standard is `cnn`, which implements the cnn.py, but you can define your own network (simply write a class implementing the `build_network(net -> Network)`-function and set `nn-model` to the name of your class (which has to be located in dlas/neural_net/)
