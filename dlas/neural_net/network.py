#!/usr/bin/env python

"""
This file is an altered version from:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import pkg_resources
#pkg_resources.require("Theano==0.8.0")

import sys
import os
import time
import logging as log
import pickle
import random
import math

import numpy as np
import warnings
import theano
import theano.tensor as T
import lasagne

from dlas.aslib.aslib_handler import ASlibHandler

class AdjustVariable(object):
    """ Helper class for update of learning rate and momentum.
    (from http://danielnouri.org/notes/category/machine-learning/)
    """
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        # Create array according to start-stop-epochs...
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        #... and choose new values according to current epoch.
        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

def float32(k):
    return np.cast['float32'](k)

class Network:

    network = None
    config = None
    aslib = None

    input_var = None
    target_var = None
    prediction = None
    loss = None
    test_prediction = None
    test_loss = None
    params = None
    updates = None
    train_fn = None
    val_fn = None
    predict_fn = None
    classify_fn = None

    def __init__(self, config, aslib=None):
        """
        Constructor. Needs config and aslib.
        """
        self.config = config
        self.network = self.buildTheanoFunctions()
        # Load aslib if provided.
        if aslib:
            self.aslib = aslib
        else:
            self.aslib = ASlibHandler()
            with open("aslib_loaded.pickle", "rb") as f: self.aslib.data = pickle.load(f)

    def build_mnn(self, input_var=None):
        """ Builds a multilayer-neural-network without convolution. """

        learningRate = self.config["learningRate"]
        dim = self.config["imageDim"]
        out = self.config["numSolvers"]
        hiddenLayers = int(config["mnnLayer"])
        hiddenNodes = int((((dim*dim)+out)/2)/hiddenLayers)

        # Input layer, as usual:
        network = lasagne.layers.InputLayer(shape=(None, 1, dim, dim),
                                            input_var=input_var)

        # Fully connected layers:
        for layer in range(hiddenLayers):
            network = lasagne.layers.dropout(network, p=(layer+1)/10.0)
            network = lasagne.layers.DenseLayer(
                network, num_units=hiddenNodes,
                nonlinearity=lasagne.nonlinearities.rectify)

        # And, finally, the 10-unit output layer with 50% dropout on its inputs:
        network = lasagne.layers.DenseLayer(
                network, num_units=out,
                nonlinearity=lasagne.nonlinearities.sigmoid)
                #nonlinearity=lasagne.nonlinearities.softmax)

        # Set regresssion to true for multi-label-classification
        network.regression=True

        return network

    def build_cnn1D(self, input_var=None):
        # CNN as described in Loreggia et al (2016)
        # Consisting of:
        # Input layer 128x128
        # 32 conv. 3x3, Max pool 2x2, Dropout 0.1
        # 64 conv. 2x2, Max pool 2x2, Dropout 0.2
        # 128 conv. 2x2, Max pool 2x2, Dropout 0.3
        # Fully connected, 1000 nodes, Dropout 0.5
        # Fully connected, 200 nodes
        # Output layer, N solvers

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

        # Note: at the moment all nodes are reduced by half

        # Input layer, as usual:
        dim = self.config["image-dim"]
        network = lasagne.layers.InputLayer(shape=(None, 1, dim, dim),
                                            input_var=input_var)

        # Convolutional layer with 32 kernels of size 3x3.
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        network = lasagne.layers.dropout(network, p=0.1)

        # Convolutional layer with 64 kernels of size 2x2.
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=64, filter_size=(2, 2),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        network = lasagne.layers.dropout(network, p=0.2)

        # Convolutional layer with 128 kernels of size 2x2.
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=128, filter_size=(2, 2),
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
        network = lasagne.layers.DenseLayer(
                network, num_units=self.config["num-solvers"],
                nonlinearity=lasagne.nonlinearities.sigmoid)
                #nonlinearity=lasagne.nonlinearities.softmax)

        # Set regresssion to true for multi-label-classification
        network.regression=True

        network.update_learning_rate=theano.shared(float32(self.config["nn-learningrate-start"]))
        network.update_momentum=theano.shared(float32(self.config["nn-momentum-start"]))

        return network


    # ############################# Batch iterator ###############################
    # This is just a simple helper function iterating over training data in
    # mini-batches of a particular size, optionally in random order. It assumes
    # data is available as numpy arrays. For big datasets, you could load numpy
    # arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
    # own custom data iteration function. For small datasets, you can also copy
    # them to GPU at once for slightly improved performance. This would involve
    # several changes in the main program, though, and is not demonstrated here.
    # Notice that this function returns only mini-batches of size `batchsize`.
    # If the size of the data is not a multiple of `batchsize`, it will not
    # return the last (remaining) mini-batch.

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def iterate_minibatches_test(self, inputs, batchsize):
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt]

    # ############################## Main program ################################
    # Everything else will be handled in our main program now. We could pull out
    # more functions to better separate the code, but it wouldn't make it any
    # easier to read.

    def buildTheanoFunctions(self):
        # Prepare Theano variables for inputs and targets
        self.input_var, self.target_var = T.tensor4("inputs"), T.matrix("targets")

        # Build network
        if   self.config["nn-model"] == "cnn": network = self.build_cnn(self.input_var)
        elif self.config["nn-model"] == "mnn": network = self.build_mnn(self.input_var)
        elif self.config["nn-model"] == "cnn1D":
            self.input_var = T.tensor3("inputs")
            network = self.build_cnn1D(self.input_var)
        else: raise ValueError("{} not a modelchoice!".format(self.config["model"]))

        # We adjust learning rate and momentum:
        if not self.config["nn-update-method"] == "adam":
            network.on_epoch_finished=[
              AdjustVariable('update_learning_rate',
                                start=self.config["nn-learningrate-start"],
                                stop=self.config["nn-learningrate-stop"]),
              AdjustVariable('update_momentum',
                                start=self.config["nn-momentum-start"],
                                stop=self.config["nn-momentum-stop"]),
            ]

        # Create a loss expression for training and validation/testing.
        # The crucial difference here is that we do a deterministic
        # forward pass through the network, disabling dropout layers.
        self.prediction = lasagne.layers.get_output(network)
        self.test_prediction = lasagne.layers.get_output(network, deterministic=True)
        if (self.config["nn-lossfunction"] == "categorical_crossentropy"):
            self.loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.target_var)
            self.test_loss = lasagne.objectives.categorical_crossentropy(self.test_prediction, self.target_var)
        elif (self.config["nn-lossfunction"] == "binary_crossentropy"):
            self.loss = lasagne.objectives.binary_crossentropy(self.prediction, self.target_var)
            self.test_loss = lasagne.objectives.binary_crossentropy(self.test_prediction, self.target_var)
        elif (self.config["nn-lossfunction"] == "squared_error"):
            self.loss = lasagne.objectives.squared_error(self.prediction, self.target_var)
            self.test_loss = lasagne.objectives.squared_error(self.test_prediction, self.target_var)
        else:
            raise ValueError("Unrecognized loss function %r." % self.config["nn-lossfunction"])

        # No matter what, normalize!
        self.loss = self.loss.mean()
        self.test_loss = self.test_loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.
        # As a bonus, also create an expression for the classification accuracy:
        self.test_acc = T.mean(T.eq(self.test_prediction, self.target_var), dtype=theano.config.floatX)

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step.
        self.params = lasagne.layers.get_all_params(network, trainable=True)
        update_meth = self.config["nn-update-method"]
        learningRate, momentum = self.config["nn-learningrate-start"], self.config["nn-momentum-start"]
        if update_meth == "sgd": self.updates = lasagne.updates.sgd(self.loss, self.params, learningRate)
        elif update_meth == "momentum": self.updates = lasagne.updates.nesterov_momentum(self.loss, self.params, learningRate, momentum=0.9)
        elif update_meth == "nesterov": self.updates = lasagne.updates.nesterov_momentum(self.loss, self.params, learningRate, momentum=0.9)
        elif update_meth == "adam"    : self.updates = lasagne.updates.adam(self.loss, self.params, learning_rate=learningRate)
        else: raise ValueError("{} not a legal option for update-method in neural network.".format(update_meth))
        #updates = lasagne.updates.adam(loss, params)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.train_fn = theano.function([self.input_var, self.target_var], self.loss, updates=self.updates)

        # Compile a second function computing the validation loss and accuracy:
        self.val_fn = theano.function([self.input_var, self.target_var], [self.test_loss, self.test_acc])

        # Function that outputs highest ranked class
        self.classify_fn = theano.function([self.input_var],T.argmax(self.test_prediction, axis=1))

        # Function that outputs an array with probabilities for each label
        self.predict_fn = theano.function([self.input_var],self.test_prediction)

        return network


    def fit(self, X, y, X_val, y_val, X_test, y_test, inst, save=False):
        """ inst = [[train], [val], [test]] """
        log.info("Fitting neural network training set for scenario {}, {} epochs.".format(
                     self.config ["scen"], self.config["nn-numEpochs"]))

        log.info("Starting training...")
        trainloss, valloss, testloss = [], [], []
        trainpred, valpred, testpred = [], [], []
        valAcc, testAcc = [], []
        timesPerEpoch = []
        timesToPredict = []

        batchsize = self.config["nn-batchsize"]
        numEpochs = self.config["nn-numEpochs"]
        scen = self.config["scen"]

        useValidation = self.config["useValidationSet"]

        if self.config["nn-model"] == "cnn1D":
            X = X.reshape(-1,1,self.config["imageDim"]*self.config["imageDim"])
            X_val = X_val.reshape(-1,1,self.config["imageDim"]*self.config["imageDim"])
            X_test = X_test.reshape(-1,1,self.config["imageDim"]*self.config["imageDim"])

        # Define custom batchsizes for val and test to evalutate whole sets
        batchsize_val = X_val.shape[0]/8  # Number of instances in set/ 2. TODO: This still might lead to MemOut, do something about it. Also Integer Division, possibly skipping one instance
        batchsize_test = X_test.shape[0]/8  # Number of instances in set/ 2. TODO: This still might lead to MemOut, do something about it. Also Integer Division, possibly skipping one instance

        # We iterate over epochs:
        for epoch in range(numEpochs):
            # Output format
            if epoch%25==0:
                print("Epoch"+8*" "+"Time"+5*" "+"Training Loss  " + "Validation Loss  " + "Validation Accuracy")
            # Measure time per epoch
            start_time = time.time()
            # In each epoch, we do a full pass over the training data:
            train_err, train_batches = 0, 0
            if batchsize > 0 and X.shape[0] > batchsize:
                for batch in self.iterate_minibatches(X, y, batchsize, shuffle=True):
                    inputs, targets = batch
                    train_err += self.train_fn(inputs, targets)
                    train_batches += 1
                train_err = train_err/train_batches
            else:
                train_batches = 1
                train_err = self.train_fn(X, y)

            if useValidation:
                # And validation
                val_err, val_acc, val_batches = 0, 0, 0
                for batch in self.iterate_minibatches(X_val, y_val, batchsize_val, shuffle=False):
                    inputs, targets = batch
                    err, acc = self.val_fn(inputs, targets)
                    val_err += err
                    val_acc += acc
                    val_batches += 1
                val_err = val_err/val_batches
                val_acc = val_acc/val_batches*100

            # And test
            test_err, test_acc, test_batches = 0, 0, 0
            for batch in self.iterate_minibatches(X_test, y_test, batchsize_test, shuffle=False):
                inputs, targets = batch
                err, acc = self.val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            test_err = test_err/test_batches
            test_acc = test_acc/test_batches*100

            timesPerEpoch.append(time.time()-start_time)

            # Save losses
            trainloss.append(train_err)
            if useValidation: valloss.append(val_err)
            testloss.append(test_err)

            # Save accuracies
            if useValidation: valAcc.append(val_acc)
            testAcc.append(test_acc)

            # Save predictions
            start_time = time.time()
            trainpred.append(self.predict_fn(X))
            if useValidation: valpred.append(self.predict_fn(X_val))
            testpred.append(self.predict_fn(X_test))
            timesToPredict.append(time.time()-start_time)

            # Skip really bad settings right away
            if math.isnan(trainloss[-1]) or (useValidation and math.isnan(valloss[-1])):
                log.info("Not training, error is nan.")
                #return False

            # Then we print the results for this epoch:
            res = "{:03d} of {}".format(epoch+1, numEpochs)
            res += (13 - len(res))*" "
            res += "{:.5}s".format(timesPerEpoch[-1])
            res += (22 - len(res))*" "
            res += "{:.12}".format(train_err)
            if useValidation:
                res += (36 - len(res))*" "
                res += " {:.12}".format(val_err)
                res += (50 - len(res))*" "
                res += " {:.12}".format(val_acc)
            log.info(res)

        # Saving the network weights
        #if save:
        #    path = self.config["modelPath"].format(scen, cv[0], cv[1], self.config["repetition"])
        #    log.info("Save network in {}".format(path))
        #    np.savez(path, *lasagne.layers.get_all_param_values(self.network))

        return timesPerEpoch, timesToPredict, trainloss, valloss, testloss, trainpred, valpred, testpred, valAcc, testAcc

    def get_params(self, deep=True):
        return {"config": self.config}

    def classify(self, X):
        log.info("Classifying best solver.")
        res = np.array(self.classify_fn(X))
        return res

    def predict(self, X):
        res = np.array(self.predict_fn(X))
        return res

        # And load them again later on like this:
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)



if __name__ == '__main__':
    # log.basicConfig(level=log.INFO)
    """for i in range(5):
        for j in range(10):
        #batchsize = 2 * batchsize
    """
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['numEpochs'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['scen'] = sys.argv[3]
        if len(sys.argv) > 4:
            kwargs['batchsize'] = int(sys.argv[4])
        main(**kwargs)
