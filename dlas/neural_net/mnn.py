import lasagne

def build_mnn(self, input_var=None):
    """ Builds a multilayer-neural-network without convolution. """
    learningRate = self.config["nn-learningrate-start"]
    dim = self.config["image-dim"]

    hiddenLayers = int(self.config["nn-mnn-layer"])
    hiddenNodes = int((((dim*dim)+self.config["num-labels"])/2)/hiddenLayers)

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, dim, dim),
                                        input_var=input_var)

    # Fully connected layers:
    for layer in range(hiddenLayers):
        network = lasagne.layers.dropout(network, p=(layer+1)/10.0)
        network = lasagne.layers.DenseLayer(
            network, num_units=hiddenNodes,
            nonlinearity=lasagne.nonlinearities.rectify)

    # Set regresssion to true for multi-label-classification
    network.regression=False

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network, num_units=self.config["num-labels"],
            nonlinearity=lasagne.nonlinearities.softmax)

    return network
