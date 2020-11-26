import numpy as np
import rwlib as rw
import nnlib
import random
import sys
import multiprocessing as mp
import matplotlib.pyplot as plt
import datalib as dlib

nn_num_nodes_hidden = [2, 2]
d = nnlib.get_num_params(nn_num_nodes_hidden)

def U(p):
    return nnlib.loss(dlib.X_dataset, dlib.y_dataset, p, nn_num_nodes_hidden)

def ACC(p):
    return nnlib.accuracy(dlib.X_dataset, dlib.y_dataset, p,nn_num_nodes_hidden)


# The following parameters are exclusively for the Monte Carlo exploration
h = 3
nsamples = 10000
thin = 3
L = 10

# Numbers of complete chains to produce, each used to compute expectations
# in order to check convergence
nsimu_convergence = 600

# When using the multichain approach, each complete chain is obtained as
# a random mix of nchain independent chains - to reduce correlation
nchains = 15


SAMPLING_SINGLE_CHAIN = True #False #True
SAMPLING_TO_CHECK_CONVERGENCE = True #True #False #True
SIMPLE_RW = 0 #True # When false, performs the more efficient multichain


if SAMPLING_SINGLE_CHAIN:
    print("Constructing a single full chain")
    if SIMPLE_RW:
        print("...simple single RW (DEBUG)")
        startx = np.random.uniform(-L, L, d)
        print("Starting accuracy: ", ACC(startx))
        print("Starting loss: ", U(startx))
#        input("PRESS ENTER")
        X, info_str, arate, _ = rw.chainRW(startx, h, U, nsamples, thin, 
                                                                L, verbose = 2)
        print("Classifiation accuracy using the last sample: ", ACC(X[-1]))

    # Ignore the following, temporearely
    else:
        print("...multichain RW approach")
        X, arate, _  = rw.multiRW(d, h, U, nsamples, nchains, thin, L)
        info_str = "INFOSIMU: Multichain RW\n"
        print("Classifiation accuracy using the last sample: ", ACC(X[-1]))


    # Store the samples into a separate file, modular approach
    filename = "markovchain_" + str(dlib.theta) +".smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1]) + "_chain.smp"
    samples_file = open(filename, "w")
    samples_file.write(info_str)
    for x in X:
        for i in range(len(x)):
            if i < (len(x) - 1):
                print(x[i], file = samples_file, end = ' ')
            else:
                print(x[i], file = samples_file, end = '\n')
    samples_file.close()
    print("Samples and information stored in " + filename)

if SAMPLING_TO_CHECK_CONVERGENCE:
    print("Sampling now to check convergence.")
    print("PRESS ENTER TO CONTINUE")
    X = rw.convRW(nsimu_convergence, d, h, U, nsamples, nchains, thin, L)
    info_str = "CONVERGENCE of: Multichain RW\n"
    # Store the samples into a separate file, to incentivate a chain approach
    filename = "expectations_" + str(dlib.theta) + ".smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1]) + "_convergence.smp"
    samples_file = open(filename, "w")
    samples_file.write(info_str)
    for x in X:
        for i in range(len(x)):
            if i < (len(x) - 1):
                print(x[i], file = samples_file, end = ' ')
            else:
                print(x[i], file = samples_file, end = '\n')
    samples_file.close()
    print("Expectations and information stored in " + filename)
