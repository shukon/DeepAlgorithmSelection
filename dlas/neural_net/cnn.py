import lasagne

def build_cnn(self, input_var=None):
    # CNN as described in Loreggia et al (2016)
    # Consisting of:
    # Input layer 128x128
    # 32 conv. 3x3, Max pool 2x2, Dropout 0.1
    # 64 conv. 2x2, Max pool 2x2, Dropout 0.2
    # 128 conv. 2x2, Max pool 2x2, Dropout 0.3
    # Fully connected, 1000 nodes, Dropout 0.5
    # Fully connected, 200 nodes
    # Output layer, N solvers

    # Input layer, as usual:
    dim = self.config["image-dim"]
    network = lasagne.layers.InputLayer(shape=(None, 1, dim, dim),
                                        input_var=input_var)

    conv1 = self.config["nn-conv-size-one"]
    conv2 = self.config["nn-conv-size-two"]
    # Convolutional layer with 32 kernels of size 3x3.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(conv1, conv1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.dropout(network, p=0.1)

    # Convolutional layer with 64 kernels of size 2x2.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(conv2, conv2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.dropout(network, p=0.2)

    # Convolutional layer with 128 kernels of size 2x2.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(conv2, conv2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.dropout(network, p=0.3)

    # Fully connected layers:
    network = lasagne.layers.DenseLayer(
            network, num_units=1000,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.DenseLayer(
            network, num_units=200,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    if self.config["nn-output-nonlinearity"] == "sigmoid":
        conf_nonlinearity=lasagne.nonlinearities.sigmoid
    elif self.config["nn-output-nonlinearity"] == "softmax":
        conf_nonlinearity=lasagne.nonlinearities.softmax
    else: raise ValueError()

    network = lasagne.layers.DenseLayer(
            network, num_units=self.config["num-labels"],
            nonlinearity = conf_nonlinearity)

    # Set regresssion to true for multi-label-classification
    network.regression = self.config["nn-regression"]

    network.update_learning_rate=theano.shared(float32(self.config["nn-learningrate-start"]))
    network.update_momentum=theano.shared(float32(self.config["nn-momentum-start"]))

    return network
