import sys
import random

import dlas.config.config as conf

def getConfig(e, s):
    if e == "TSP":
        return tsp(s)
    elif e == "random":
        return tspRand(s)
    else:
        print("{} is not defined as an experiment!".format(e))

def tsp(s):
    c = conf.conf("TSP-resize-100-base", updates=[("learningRate",0.01),
                    ("image-id","FromImage"),("labelID", "MultiLabelBase"),
                    ("imageDim",100),("model","cnn"),
                    ("numEpochs",30),("batchsize",64),
                    #("lossFunction","squared_error")
                    ("lossFunction","binary_crossentropy"),
                    #("lossFunction","categorical_crossentropy")
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

