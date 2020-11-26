import numpy as np
from numpy import cos, sin
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from sklearn.decomposition import PCA, KernelPCA
import nnlib
import datalib as dlib


STUDY_CONVERGENCE = True #False

# this section will require nnlib and datalib to access the potential
DETECT_SUBMANIFOLD = True


# Given a list of 1-dimensional samples, return the confidence interval
def mean_variance1d(samples1d):
    mean = np.mean(samples1d)
    sigma = 0.
    n = len(samples1d)
    for i in range(n):
        sigma += (samples1d[i] - mean) ** 2.
    sigma = np.sqrt(sigma / (n - 1))
    return mean, sigma


if STUDY_CONVERGENCE:
    print("Reading the results about CONVERGENCE")
    # Open the file containing the list of samples
    filename = "expectations_0.smp"
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
        mean, sigma = mean_variance1d(X[:,i])
        print("Merginal number #", i)
        print("Mean: ", mean, "sigma: ", sigma)
        print("95% Confidence Interval: [",
                        mean-2.*sigma, " ", mean+2.*sigma, "]")

    # Plot the expectations' distribution
    for i in range(m):
        plt.subplot(int(m / 4) + 1, 4, i+1)
        plt.hist(X[:,i], 30, density=True)
    plt.suptitle("Convergence analysis. Gaussian = WIN")
    plt.savefig("are_gaussians.png")
#    plt.show()

if DETECT_SUBMANIFOLD:
    print("Searching for a SUBMANIFOLD on the CHAIN SAMPLES")
    # Open the file containing the list of samples
    filename = "markov_chain_0.smp"
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
    plt.savefig("tmp_ALLDATA.png")
#    plt.show()    

    reconstructedX = kpcaRBF.inverse_transform(reducedXrbf)
    print("Reconstructing error: ", norm(X - reconstructedX))

    print("Let's find the surface of minimal energy")
    # Information for defining the potential
    # Import now the dataset
    nn_num_nodes_hidden = [2, 2]
    d = nnlib.get_num_params(nn_num_nodes_hidden)
    def U(x):
        return nnlib.loss(dlib.X_dataset, dlib.y_dataset, 
                x, nn_num_nodes_hidden) 
    def ACC(x):
        return nnlib.accuracy(dlib.X_dataset, dlib.y_dataset, 
                x, nn_num_nodes_hidden)

    max_energy = 1.0
    labels = np.zeros(len(reducedXrbf))
    for i in range(len(reconstructedX)):
        Ui = U(reconstructedX[i])
#        print("Energy of point", i, ":", Ui)
#        print("its accuracy: ", ACC(reconstructedX[i]))
        if Ui < max_energy:
            labels[i] = 1 
            print("Energy of point", i, ":", Ui)
            print("its accuracy: ", ACC(reconstructedX[i]))
#            input("OK?")
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
    plt.savefig("tmp_MANIFOLD.png")
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

   # minimumX = pcaLOW.transform(lowX)
   # print("PCA error: ", norm(minimumX - pcaLOW.inverse_transform(minimumX)))
