import theano
import theano.tensor as T
import lasagne

def build_network(self):
    """
    Building a 1-dimensional convolutional neural network.

    Args:
        self -- Network
            network to be built
    """
    # Make input one-dimensional
    self.input_var = T.tensor3("inputs")

    # Input layer, one-dimensional:
    dim = self.config["imageDim"]**2
    network = lasagne.layers.InputLayer(shape=(None, 1, dim),
                                        input_var=self.input_var)

    # Convolutional layer with 32 kernels of size 9.
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=(9),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=(4))
    network = lasagne.layers.dropout(network, p=0.1)

    network = lasagne.layers.Conv1DLayer(
            network, num_filters=64, filter_size=(4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=(4))
    network = lasagne.layers.dropout(network, p=0.2)

    network = lasagne.layers.Conv1DLayer(
            network, num_filters=128, filter_size=(4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=(4))
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
    network = lasagne.layers.DenseLayer(
            network, num_units=self.config["num-labels"],
            nonlinearity=lasagne.nonlinearities.sigmoid)
            #nonlinearity=lasagne.nonlinearities.softmax)

    # Set regresssion to true for multi-label-classification
    network.regression=True

    return network
