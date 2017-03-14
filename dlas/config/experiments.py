import sys
import random

import dlas.config.config as conf

def dict_from_file(s, ID):
    with open("experiments/{}.txt".format(ID), 'r') as f:
        content = [line.strip("\n").split("=") for line in f.readlines]
        content = [(name.strip(), value.strip()) for name, value in content]
        print(dict(content))
        return dict(content)

def getConfig(s, ID):
    x = dict_from_file(s, ID)
    return conf.conf(s, ID, x)

def tsp_weights(s, ID):
    c = tsp_default(s, ID)
    c["label-mode"] = "MultiLabelWeight"
    c["label-weight-timeout"] = 0.9
    c["label-weight-best"] = 0.1
    return c

def tsp_from_txt(s, ID):
    c = tsp_default(s, ID)
    c["image-mode"] = "TextToImage"
    return c

def tsp_conv(s, ID):
    c = tsp_default(s,ID)
    c["nn-conv-size-one"] = 40
    c["nn-conv-size-two"] = 9
    return c

def tsp_adagrad(s, ID):
    c = tsp_default(s,ID)
    c["nn-learningrate-start"] = 0.001
    c["nn-update-method"] = "adagrad"
    c["nn-numEpochs"] = 20
    return c

def tsp_idea(s, ID):
    c = tsp_default(s,ID)
    c["nn-regression"] = True
    #c["nn-output-nonlinearity"] = "softmax"
    c["lossFunction"] = "categorical_crossentropy"
    c["nn-update-method"] = "adam"
    c["label-mode"] = "MultiLabelBase"
    c["num-labels"] = "num-solvers"
    c["nn-learningrate-start"] = 0.001
    c["nn-learningrate-stop"] = 0.01
    return c

def tsp_default(s, ID):
    c = conf.conf(s, ID, updates=[
                    ("nn-learningrate-start",0.1),
                    ("nn-learningrate-stop",0.0001),
                    ("nn-momentum-start",0.9),
                    ("nn-momentum-stop",0.999),
                    ("image-mode","FromImage"),("label-mode", "MultiLabelBase"),
                    ("num-labels","num-solvers"),
                    ("image-dim",100),("nn-model","cnn"),
                    ("nn-numEpochs",30),("nn-batchsize",64),
                    #("lossFunction","squared_error")
                    ("nn-lossfunction","binary_crossentropy"),
                    #("nn-lossfunction","categorical_crossentropy"),
                    ("nn-conv-size-one", 3),
                    ("nn-conv-size-two", 2)
                    ])
    return c

def tsp_inst_name_cnn(s, ID):
    c = tsp_default(s, ID)
    c["nn-learningrate-start"] = 0.0001
    c["nn-learningrate-stop"] = 0.01
    c["label-mode"] = "TSPLabelClass"
    c["num-labels"] = 3
    c["nn-regression"] = False
    c["nn-output-nonlinearity"] = "softmax"
    c["lossFunction"] = "categorical_crossentropy"
    c["nn-update-method"] = "adagrad"
    return c

def tsp_inst_name(s, ID):
    c = conf.conf(s, ID, updates=[
                    ("nn-learningrate-start",0.001),
                    ("image-mode","FromImage"),("label-mode", "TSPLabelClass"),
                    ("num-labels",3),
                    ("image-dim",100),("nn-model","mnn"),
                    ("nn-numEpochs",30),("nn-batchsize",64),
                    ("nn-lossfunction","binary_crossentropy"),
                    ("nn-mnn-layer",3)
                    ])
    return c

def tspRand(s):
    lr = random.randint(1,1000)/1000.0
    lf = random.choice(["squared_error","binary_crossentropy",
                        "categorical_crossentropy"])
    dim = random.choice([50,75,100,128])
    batch = random.choice([32,64])

    c = conf.conf("TSP-{}".format(random.randint(1,10000)),
                  updates=[("learningRate",lr),
                    ("convID","base-"+str(dim)),("labelID", "base"),
                    ("imageDim",dim),("model","cnn"),
                    ("numEpochs",30),("batchsize",batch),
                    ("lossFunction",lf),
                    #("weightTimeOut", 0.2),("weightBest",0.8)
                    ])
    return c

