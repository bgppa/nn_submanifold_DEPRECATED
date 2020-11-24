import numpy as np
import rw
import nnlib
import random
import sys
import multiprocessing as mp
import matplotlib.pyplot as plt
from numpy import cos, sin


theta = 120 # 240

def rotate(x, theta):
    # x in R^2, theta rotational angle
    R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return np.dot(R, x)


# Part 1: determine the model to train
# First of all, consider the Neural Network architecture. It is understood
# to be sequential and fully connected, taking in input
# 2-dimensional points, and giving an scalar output according to the
# point classification. You need to specify a list containing the number
# of nodes for each hidden layer.
nn_num_nodes_hidden = [3, 3]

# Generate the random points to classify
N_points = 10
# Generate always the same points, but rotate w.r.t to theta
# the starting seed is stored and then set again in order not to
# interfere with the later Monte Carlo algorithm
bak_state = np.random.get_state()
# I like the seed number 2 for generating points
np.random.seed(2)
X_dataset = np.random.uniform(-1, 1, size = [N_points, 2])
np.random.set_state(bak_state)
#X_dataset = np.array([[0.1, 0.1], [0.9, 0.9]])
X_dataset = np.array([rotate(x, theta) for x in X_dataset])
y_dataset = np.zeros(N_points * 2)
for i in range(N_points):
    if (i < N_points/2):
        y_dataset[2 * i] = 1
    else:
        y_dataset[2 * i + 1] = 1
y_dataset.shape  = (N_points, 2)

#nnlib.plot_data(X_dataset, y_dataset)
#quit()

# Define now the model and the loss function
d = nnlib.get_num_params(nn_num_nodes_hidden)

def U(x):
    return nnlib.loss(X_dataset, y_dataset, x, nn_num_nodes_hidden)

def ACC(x):
    return nnlib.accuracy(X_dataset, y_dataset, x, nn_num_nodes_hidden)


# The following parameters are exclusively for the Monte Carlo exploration
h = 0.8
nsamples = 20000
thin = 4
nsimu_convergence = 500
L = 10
nchains = 48


SAMPLING_SINGLE_CHAIN = True
SAMPLING_TO_CHECK_CONVERGENCE = False#True #True #False #True
SIMPLE_RW = 1 #True # When false, performs the more efficient multichain


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
        info_str += '\n'
        print("Starting point: ", startx)
        print("Classifiation accuracy using the last sample: ", ACC(X[-1]))

    # Ignore the following, temporearely
    else:
        print("...multichain RW approach")
        X, arate, _  = rw.multiRW(d, h, U, nsamples, nchains, thin, L)
        info_str = "INFOSIMU: Multichain RW\n"
        print("Classifiation accuracy using the last sample: ", ACC(X[-1]))



    # Store the samples into a separate file, modular approach
    filename = "biNNary_chain_" + str(theta) +".smp"
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
    X = rw.convRW(nsimu_convergence, d, h, U, nsamples, nchains, thin, L)
    info_str = "CONVERGENCE of: Multichain RW\n"
    # Store the samples into a separate file, to incentivate a chain approach
    filename = "biNNary_conv_" + str(theta) + ".smp"
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
