import numpy as np
from numpy import cos, sin
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import mcmc
from sklearn.decomposition import PCA, KernelPCA
import nnlib

STUDY_CONVERGENCE = False
DETECT_SUBMANIFOLD = True


if STUDY_CONVERGENCE:
    print("Reading the results about CONVERGENCE")
    # Open the file containing the list of samples
    filename = "biNNary_conv_120.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1])
    print("Loading:", filename)
    samples_file = open(filename, "r")

    # Its first line contains information about the simulation
    info_str = samples_file.readline()
    print(info_str[:-1])

    # Collect all the samples into X
    X = []
    for x in samples_file:
        X.append(np.array(x[0:-1].split(' ')).astype("float64"))
    X = np.asanyarray(X)
    samples_file.close()
    print("Read", len(X), "samples of dimension", len(X[0]))
    m = len(X[0])
    # Computing the confidence interval for each marginal
    for i in range(m):
        # Compute the 95% confidence interval
        mean, sigma = mcmc.mean_variance1d(X[:,i])
        print("Merginal number #", i)
        print("Mean: ", mean, "sigma: ", sigma)
        print("95% Confidence Interval: [",
                        mean-2.*sigma, " ", mean+2.*sigma, "]")

    # Plot the expectations' distribution
    for i in range(m):
        plt.subplot(int(m / 4) + 1, 4, i+1)
        plt.hist(X[:,i], 30, density=True)
    plt.suptitle("Convergence analysis. Gaussian = WIN")
    plt.show()


if DETECT_SUBMANIFOLD:
    print("Searching for a SUBMANIFOLD on the CHAIN SAMPLES")
    # Open the file containing the list of samples
    filename = "biNNary_chain_120.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1])
    print("Loading:", filename)
    samples_file = open(filename, "r")

    # Its first line contains information about the simulation
    info_str = samples_file.readline()
    print(info_str[:-1])

    # Collect all the samples into X
    X = []
    for x in samples_file:
        X.append(np.array(x[0:-1].split(' ')).astype("float64"))
    X = np.asanyarray(X)
    samples_file.close()
    print("Read", len(X), "samples of dimension", len(X[0]))
    m = len(X[0])
    # start the kernel PCA to find low energy submanifold
    kpcaRBF = KernelPCA(n_components = 3, kernel = "rbf",
                    fit_inverse_transform=True, n_jobs = -1)
    reducedXrbf = kpcaRBF.fit_transform(X)
    # Let's plot the reduced 3D space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reducedXrbf[:,0], reducedXrbf[:,1], reducedXrbf[:,2])
    plt.title("3D reduction of the " + str(m) + "D parameter with rbf kernel")
    plt.savefig("ALLDATA.png")
#    plt.show()    

    reconstructedX = kpcaRBF.inverse_transform(reducedXrbf)
    print("Reconstructing error: ", norm(X - reconstructedX))

    print("Let's find the surface of minimal energy")
    # Information for defining the potential
    bak_state = np.random.get_state()
    np.random.seed(2)
    N_points = 10
    theta = 120 # 240
    nn_num_nodes_hidden = [3, 3]

    def rotate(x, theta):
        # x in R^2, theta rotational angle
        R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        return np.dot(R, x)

    X_dataset = np.random.uniform(-1, 1, size = [N_points, 2])
    np.random.set_state(bak_state)
    X_dataset = np.array([rotate(x, theta) for x in X_dataset])
    y_dataset = np.zeros(N_points * 2)
    for i in range(N_points):
        if (i < N_points/2):
            y_dataset[2 * i] = 1
        else:
            y_dataset[2 * i + 1] = 1
    y_dataset.shape  = (N_points, 2)

    def U(x):
        return nnlib.loss(X_dataset, y_dataset, x, nn_num_nodes_hidden) 

    def ACC(x):
        return nnlib.accuracy(X_dataset, y_dataset, x, nn_num_nodes_hidden)


    max_energy = 10
    labels = np.zeros(len(reducedXrbf))
    for i in range(len(reconstructedX)):
        Ui = U(reconstructedX[i])
        print("Energy of point", i, ":", Ui)
        print("its accuracy: ", ACC(reconstructedX[i]))
        if Ui < max_energy:
            labels[i] = 1 
            print("Energy of point", i, ":", Ui)
            print("its accuracy: ", ACC(reconstructedX[i]))
            input("OK?")
    input("OK?")

    # Subdivide the points into two classes, then print each of them
    lowX, highX = [], []
    for i in range(len(labels)):
        if labels[i]:
            lowX.append(reducedXrbf[i])
        else:
            highX.append(reducedXrbf[i])
    print("Amout of low energy points: ", len(lowX))
    print("...of not-so-low: ", len(highX))

    # Convert the points into a numpy array
    lowX = np.asanyarray(lowX)
    highX = np.asanyarray(highX)

    # Plot, in 3D, only the points with right energy
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(lowX[:,0], lowX[:,1], lowX[:,2], color='red')
    plt.title("Distribution of the low energy points in" +\
          "the 3D reduced space")
    plt.savefig("MANIFOLD.png")
#    plt.show()

    # Perform a classic PCA reduction on the low-energy space,
    # which seem to be located on a line...
    pcaLOW = PCA(n_components=1)
    pcaLOW.fit(lowX)
    pcaLOW.explained_variance_ratio_
    print("How well do the points fit a line? ",
                        sum(pcaLOW.explained_variance_ratio_)* 100, "%")

    minimumX = pcaLOW.transform(lowX)
    print("PCA error: ", norm(minimumX - pcaLOW.inverse_transform(minimumX)))



    pcaLOW = PCA(n_components=2)
    pcaLOW.fit(lowX)
    pcaLOW.explained_variance_ratio_
    print("How well do the points fit a plane? ",
                        sum(pcaLOW.explained_variance_ratio_)* 100, "%")

    minimumX = pcaLOW.transform(lowX)
    print("PCA error: ", norm(minimumX - pcaLOW.inverse_transform(minimumX)))

