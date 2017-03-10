import theano
import lasagne

def build_cnn1d(self, input_var=None):
    learningRate = self.config["nn-learningrate-start"]

    # Input layer, one-dimensional:
    dim = self.config["imageDim"]**2
    network = lasagne.layers.InputLayer(shape=(None, 1, dim),
                                        input_var=input_var)

    scale = self.config["scale"]

    # Convolutional layer with 32 kernels of size 3x3.
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=32/scale, filter_size=(9),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=(4))
    network = lasagne.layers.dropout(network, p=0.1)

    # Convolutional layer with 64 kernels of size 2x2.
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=64/scale, filter_size=(4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=(4))
    network = lasagne.layers.dropout(network, p=0.2)

    # Convolutional layer with 128 kernels of size 2x2.
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=128/scale, filter_size=(4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=(4))
    network = lasagne.layers.dropout(network, p=0.3)

    # Fully connected layers:
    network = lasagne.layers.DenseLayer(
            network, num_units=1000/scale,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.DenseLayer(
            network, num_units=200/scale,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network, num_units=self.config["num-labels"],
            nonlinearity=lasagne.nonlinearities.sigmoid)
            #nonlinearity=lasagne.nonlinearities.softmax)

    # Set regresssion to true for multi-label-classification
    network.regression=True

    # Update learning rate and momentum to be shared variables
    network.update_learning_rate=theano.shared(float32(learningRate)),
    network.update_momentum=theano.shared(float32(0.9)),

    return network
