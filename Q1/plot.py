"""
Module to plot our data
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def scatterPlot (X, Y, Xlabel="X", Ylabel="Y", marker="ro"):
    """
    Plots Y vs X as scatter plot
    """
    plt.plot(X, Y, marker)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.show()


def regressionPlot (X, Y, slope, intercept, Xlabel="X", Ylabel="Y", marker="ro", linecolor="r", fileName="plots/test.png"):
    """
    Plots Y vs X as scatter plot, joined by the line
    """
    fig = plt.figure(1)
    plt.plot(X, Y, marker)
    plt.plot(X, X*slope + intercept, linecolor)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    fig.savefig(fileName)
    plt.show()


def costPlot (X, Y, Z, Xlabel="X", Ylabel="Y", Zlabel="Z"):
    """
    Plots a 3D curve of XYZ
    """
    print('hoa')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(Xlabel)
    ax.set_ylabel(Ylabel)
    ax.set_zlabel(Zlabel)

    X, Y = np.meshgrid(X, Y)
    print (X.shape, Y.shape)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5)
    plt.show()
