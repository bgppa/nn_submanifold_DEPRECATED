## TO CHECK BETTER:
# - forward evaluation
# - binary entropy


# This is a very simple library to generate naive Neural Networks
import numpy as np
from numpy import log, exp
import matplotlib.pyplot as plt

def ber():
    if np.random.uniform() < 0.5:
        return 0.
    return 1.

# Global variables that defines my Newtwork on which to experiment
#NUM_INPUTS = 2
#NUM_HIDDEN_LAYERS = 3
#NUM_NODES_HIDDEN = [4, 4, 2]
#NUM_NODES_OUTPUT = 1
#global_N_points = 10
#old_seed = np.random.get_state()
#np.random.seed(3)
#global_X = np.random.uniform(size = [global_N_points, 2])
#global_y = np.array([ber() for i in range(global_N_points)])
#np.random.set_state(old_seed)

def plot_data(X, y):
    X1 = []
    X0 = []
    for i in range(len(y)):
        if y[i][0] == 1:
            X1.append(X[i])
        else:
            X0.append(X[i])
    X1 = np.asanyarray(X1)
    X0 = np.asanyarray(X0)
    plt.scatter(X0[:, 0], X0[:, 1], color = 'red')
    plt.scatter(X1[:, 0], X1[:, 1], color = 'blue')
    plt.title("Data to classify")
    plt.show()


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


# Create a NN with default structure from R^2 to R.
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


def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

#def sigmoid(x):
#    return 1.0 / (1.0 + exp(-x))

# Define the ReLU function
def node_activation(weighted_sum):
    return max(0., weighted_sum)

# 
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
#    v = np.copy(vv)
#    for i in range(len(v)):
#        v[i] = np.exp(v[i])
#    return v / np.sum(v)


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



def l2square(ytrue, ypred):
    n = len(ytrue)
    sm = 0.
    for i in range(n):
        sm += (ytrue[i] - ypred[i]) ** 2
    return (np.sqrt(sm) / n) * 100


# STEP 3: define the loss function
# THE LOSS AND ACCURACY  FUNCTION MUST BE THE ONLY 
# ONE DEPENDING ON GLOBAL VARIABLES
def loss(X, y, p, num_nodes_hidden, num_inputs = 2, num_output = 2):
    # Create a Network
    n = len(X)
    loc_nn = init_network(p, num_nodes_hidden, num_inputs, num_output)
     # Evaluate each datapoint on the created newtork
    y_hat = np.array([forward_propagate(loc_nn, X[i]) for i in range(n)])
    yt = y[:,0]
    yp = y_hat[:,0]
    sm = 0
    for i in range(n):
        if (yp[i] > 0 and yp[i] < 1):
                sm += yt[i]*log(yp[i]) + (1-yt[i])*log(1-yp[i])
    #sm = np.sum([yt[i]*log(yp[i]) + (1-yt[i])*log(1-yp[i]) for i in range(n)])
    return (-sm * 100) / n
#    return l2square(y, y_hat)


def from_prob_to_01(yyy):
    yy = np.copy(yyy)
    for i in range(len(yy)):
        if yy[i][0] < yy[i][1]:
            yy[i][0] = 0.
            yy[i][1] = 1.
        else:
            yy[i][0] = 1.
            yy[i][1] = 0.
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
        if (y[i][0] == y_hat[i][0]):
            correct += 1
    return correct * 100 / len(y_hat)


#def from_R_to_prob(yy):
#    yyy = np.copy(yy)
#    for i in range(len(yyy)):
#        yyy[i] = sigmoid(yyy[i])
#    return yyy


if __name__ == '__main__':
    print("Entering debug mode")
    N_points = 10
    #X_dataset = np.array([[0.1, 0.1], [9, 9]])
    X_dataset = np.random.uniform(size = [N_points, 2])
    y_dataset = np.zeros(N_points * 2)
    for i in range(N_points):
        if (i < N_points/2):
            y_dataset[2 * i] = 1
        else:
            y_dataset[2 * i + 1] = 1
    y_dataset.shape  = (N_points, 2)
    nn_num_nodes_hidden = [3, 4]
    d = get_num_params(nn_num_nodes_hidden)
    L = 1
    def ACC(x):
        return accuracy(X_dataset, y_dataset, x, nn_num_nodes_hidden)
    def U(x):
        return loss(X_dataset, y_dataset, x, nn_num_nodes_hidden)
    for i in range(10):
        p = (np.random.uniform(-L, L, d))
        print(ACC(p))
        print(U(p))
