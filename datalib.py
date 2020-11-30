# This script is used to fix the dataset to use when training the network
# I chose random points in the 2-dimensional square [-1,1]^2
# and fix the random seed for reproducibility
# For the experiments it is useful to have the same data, this time
# subjected to a rotation of angle theta.
import numpy as np
from numpy import log, exp
import matplotlib.pyplot as plt
from numpy import cos, sin


my_seed = 2
N_points = 10
theta = 0 #120 #240 #120 #0 #120 # 240


###############################
def rotate(x, theta):
    # x in R^2, theta rotational angle
    R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return np.dot(R, x)


# Generate random points and rotate them
bak_state = np.random.get_state()
np.random.seed(my_seed)
X_dataset = np.random.uniform(-1, 1, size = [N_points, 2])
np.random.set_state(bak_state)
X_dataset = np.array([rotate(x, theta) for x in X_dataset])
# Classify them, labels in y
y_dataset = np.zeros(N_points * 2)
for i in range(N_points):
    if (i < N_points/2):
        y_dataset[2 * i] = 1
    else:
        y_dataset[2 * i + 1] = 1
y_dataset.shape  = (N_points, 2)


def plot_data(X = X_dataset, y = y_dataset, theta = theta, xserver = 0):
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
    plt.title("Dataset for the classification. Rotation angle: " + str(theta))
    if xserver:
        plt.show()
    else:
        plt.savefig("points_wseed" + str(my_seed) + "angle" +str(theta)+".png")


if __name__ == '__main__':
    plot_data(X_dataset, y_dataset, theta, True)
 #   plot_data(X_dataset, y_dataset, theta, False)
