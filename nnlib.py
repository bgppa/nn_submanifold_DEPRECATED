# The goals of this small library is just to give the tools to create 
# various kind of fully connected neural networks and define their loss
# functions w.r.t dataset given by other sources.
# It prepared the loss function, whose geometry will be explored
# by using Monte Carlo methods (in another script)


## TO CHECK BETTER:
# - forward evaluation
# - binary entropy

# This is a very simple library to generate naive Neural Networks
import numpy as np
from numpy import log, exp
import matplotlib.pyplot as plt


# Given the nn architecture, return the number of parameters needed
# to define it. I.e. it's the loss function domain's dimension.
def get_num_params(num_nodes_hidden, num_inputs = 2, num_output = 2):
    num_hidden_layers = len(num_nodes_hidden)
    tot = num_inputs * num_nodes_hidden[0] + num_nodes_hidden[0]
    for i in range(1, num_hidden_layers + 1):
        if (i == num_hidden_layers):
            tot += num_nodes_hidden[i-1] * num_output + num_output
        else:
            tot += num_nodes_hidden[i-1] * num_nodes_hidden[i] + \
                   num_nodes_hidden[i]
    print("This model requires ", tot, "parameters")
    return tot


# Create a NN with default structure from R^2 to R^2
def init_network(params, num_nodes_hidden, num_inputs = 2, num_output = 2):
    num_hidden_layers = len(num_nodes_hidden)
    num_nodes_previous = num_inputs # number of nodes in the previous layer
    network = {}
    offset = 0
    # loop through each layer and initialize the weights and biases 
    # associated with each layer
    for layer in range(num_hidden_layers + 1):
        # Start by giving names to each layer
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = num_nodes_hidden[layer]
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': \
                      np.asanyarray(params[offset:offset+num_nodes_previous]),
                'bias'   : np.asanyarray(params[offset + num_nodes_previous])
            }
            offset = offset + num_nodes_previous + 1
        num_nodes_previous = num_nodes
    return network



### CHECK BETTER
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias


# Standard sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def relu(x):
    return max(0., x)

# ReLU function
def node_activation(weighted_sum):
    return relu(weighted_sum)
#   return sigmoid(weighted_sum)


# Softmax function for 2-dim output classification 
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def forward_propagate(network, inputs):
    # start with the input layer as the input to the first hidden layer
    layer_inputs = list(inputs)     
    for layer in network:
        layer_data = network[layer]
        layer_outputs = []
        for layer_node in layer_data:
            node_data = layer_data[layer_node]
            # compute the weighted sum and the output 
            # of each node at the same time
            node_output = node_activation(compute_weighted_sum(layer_inputs, 
                                    node_data['weights'], node_data['bias']))
            layer_outputs.append(node_output)

#        if layer != 'output':
#            print("Node output nodes hlayer number {}: {}"\
#                            .format(layer.split('_')[1], layer_outputs))

        # set the output of this layer to be the input to next layer
        layer_inputs = layer_outputs     
        network_predictions = softmax(layer_outputs)
    return network_predictions


### Now we care about loss functions
# Mean Squared Error
def mse(ytrue, ypred):
    n = len(ytrue)
    sm = 0.
    for i in range(n):
        sm += (ytrue[i] - ypred[i]) ** 2
    return (np.sqrt(sm) / n) * 100


# Binary Cross Entropy
def bce(yt, yp):
    n = len(yt)
    sm = 0
    for i in range(n):
        if (yp[i] > 0 and yp[i] < 1):
            sm += yt[i]*log(yp[i]) + (1-yt[i])*log(1-yp[i])
    return (-sm * 100) / n


def loss(X, y, p, num_nodes_hidden, num_inputs = 2, num_output = 2):
    n = len(X)
    # Create a Network
    loc_nn = init_network(p, num_nodes_hidden, num_inputs, num_output)
    # Evaluate each datapoint on the created newtork
    y_hat = np.array([forward_propagate(loc_nn, X[i]) for i in range(n)])
    yt = y[:,0]
    yp = y_hat[:,0]
    #return mse(yt, yp)
    return bce(yt, yp)


# Compute the accuracy of the model
def old_from_prob_to_01(yyy):
    yy = np.copy(yyy)
    for i in range(len(yy)):
        if yy[i][0] < yy[i][1]:
            yy[i][0] = 0.
            yy[i][1] = 1.
        else:
            yy[i][0] = 1.
            yy[i][1] = 0.
    return yy


# Compute the accuracy of the model
def from_prob_to_01(yyy):
    yy = np.copy(yyy[:, 0])
    for i in range(len(yy)):
        if yy[i] < 0.5:
            yy[i] = 0.
        else:
            yy[i] = 1.
    return yy

def accuracy(X, y, p, num_nodes_hidden, num_inputs = 2, num_output = 2):
    len_dataset = len(X)
    loc_nn = init_network(p, num_nodes_hidden, num_inputs, num_output)
    # Evaluate each datapoint on the created newtork
    y_hat = np.array([forward_propagate(loc_nn, X[i]) \
                                                for i in range(len_dataset)])
#    print("--- accuracy debug ---")
#    print("Parameters: ", p)
#    print("y_hat (R): ", y_hat)
    y_hat = from_prob_to_01(y_hat)
#    print("y_hat (01): ", y_hat)
#    print("true y: ", y)
    correct = 0
    for i in range(len(y_hat)):
        # Since are 1/0, to check equality is enough using the first coordinate
#        if (y[i][0] == y_hat[i][0]):
        if (y[i][0] == y_hat[0]):
            correct += 1
    acc = correct * 100 / len(y_hat)
#    print("..acc: ", acc)
#    input("OK")
    return acc


#def from_R_to_prob(yy):
#    yyy = np.copy(yy)
#    for i in range(len(yyy)):
#        yyy[i] = sigmoid(yyy[i])
#    return yyy


if __name__ == '__main__':
    print("Entering debug mode, data are imported from datalib.py")
    from datalib import X_dataset, y_dataset
    nn_num_nodes_hidden = [2, 2]
    d = get_num_params(nn_num_nodes_hidden)
    L = 3
    def ACC(x):
        return accuracy(X_dataset, y_dataset, x, nn_num_nodes_hidden)
    def U(x):
        return loss(X_dataset, y_dataset, x, nn_num_nodes_hidden)
    for i in range(10):
        p = (np.random.uniform(-L, L, d))
        print(ACC(p))
        print(U(p))
