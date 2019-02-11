"""
Module to plot our data
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math

def scatterPlot (X, Y, Xlabel="X", Ylabel="Y", marker="ro", fileName="Q4/plots/raw.png", title="Scatter Plot"):
    """
    Plots Y vs X as scatter plot
    """
    circles = []
    crosses = []
    for x, label in zip(X, Y):
        if label == 0:
            circles.append(x)
        else:
            crosses.append(x)
    circles = np.array(circles)
    crosses = np.array(crosses)

    fig = plt.figure(1)
    cir = plt.plot(circles[:,0], circles[:,1], 'go', label="Alaska (0)")
    cro = plt.plot(crosses[:,0], crosses[:,1], 'rx', label="Canada (1)")
    plt.suptitle(title)
    plt.title("Please close the figure for next figure", fontsize=6)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.legend()
    fig.savefig(fileName)
    plt.show()


def GDAPlot (X, Y, Mu0, Mu1, Sigma0, Sigma1, Xlabel="X", Ylabel="Y", marker="ro", fileName="Q4/plots/guassian.png", title="Guassian Plot"):
    """
    Plots Y vs X as scatter plot
    """
    A = Sigma0_inv = np.matrix(np.linalg.inv(Sigma0))
    B = Sigma1_inv = np.matrix(np.linalg.inv(Sigma1))
    Mu0 = np.matrix(Mu0).T
    Mu1 = np.matrix(Mu1).T
    # print (Mu0, Sigma0)
    # exit()
    detA = np.linalg.det(Sigma0)
    detB = np.linalg.det(Sigma1)
    phi = 0.5

    print (Sigma0_inv, type(Sigma0_inv), Sigma0_inv[0][0], Mu0[0])
    # exit(0)

    circles = []
    crosses = []
    for x, label in zip(X, Y):
        if label == 0:
            circles.append(x)
        else:
            crosses.append(x)
    circles = np.array(circles)
    crosses = np.array(crosses)

    fig = plt.figure(2)
    cir = plt.plot(circles[:,0], circles[:,1], 'go', label="Alaska (0)")
    cro = plt.plot(crosses[:,0], crosses[:,1], 'rx', label="Canada (1)")
    plt.plot(Mu0[0], Mu0[1], 'bP')
    plt.plot(Mu1[0], Mu1[1], 'bP')

    x = np.linspace(50,180,300)
    y = np.linspace(300,500,200)
    # x,y = np.meshgrid(x, y)

    # x0 = x - Mu0[0]
    # y0 = y - Mu0[1]
    # x1 = x - Mu1[0]
    # y1 = y - Mu1[1]

    print (x.shape, y.shape)
    # print (x.shape, y.shape, x0.shape, y1.shape, x0.shape)
    # print (x)
    # print(Mu0)
    # print (x0)

    # LHS = A[0][0] * (x0**2) + (A[0][1]+A[1][0]) * (x0*y0) + A[1][1] * (y0**2)
    # LHS = (1-phi) * np.exp(-0.5 * LHS) / math.sqrt(detA)
    # RHS = B[0][0] * (x1*2) + (B[0][1]+B[1][0]) * (x1*y1) + B[1][1] * (y1**2)
    # RHS = (1-phi) * np.exp(-0.5 * RHS) / math.sqrt(detB)

    # Z = LHS - RHS
    # Z = A[0][0] * (x0**2) + (A[0][1]+A[1][0]) * (x0*y0) + A[1][1] * (y0**2) 
    # - (B[0][0] * (x1**2) + (B[0][1]+B[1][0]) * (x1*y1) + B[1][1] * (y1**2))

    # print (Z)
    # Z = np.zeros(shape=(x1.shape[0], x2.shape[0]))
    Z = np.zeros(shape=(y.shape[0], x.shape[0]))
    print (type(Mu0), type(Sigma0_inv))
    print (Z.shape)
    for i in range(y.shape[0]):
        for j in range(x.shape[0]):
            X = np.matrix([x[j], y[i]]).T
            Tmp = (X-Mu0).T * Sigma0_inv * (X-Mu0) - (X-Mu1).T * Sigma1_inv * (X-Mu1)
            Z[i][j] = float(Tmp)

    plt.contour(x,y,Z,[0])
    plt.suptitle(title)
    plt.title("Please close the figure for next figure", fontsize=6)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.legend()
    fig.savefig(fileName)
    plt.show()
