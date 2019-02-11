"""
Module to plot our data
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def scatterPlot (X, Y, Xlabel="X", Ylabel="Y", marker="ro", fileName="Q1/plots/raw.png", title="Scatter Plot"):
    """
    Plots Y vs X as scatter plot
    """
    fig = plt.figure(1)
    plt.plot(X, Y, marker)
    plt.suptitle(title)
    plt.title("Please close the figure for next figure", fontsize=6)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    fig.savefig(fileName)
    plt.show()


def regressionPlot (X, Y, slope, intercept, Xlabel="X", Ylabel="Y", marker="ro", linecolor="r", fileName="Q1/plots/regression.png", title="Regression"):
    """
    Plots Y vs X as scatter plot, joined by the line
    """
    fig = plt.figure(2)
    plt.plot(X, Y, marker)
    plt.suptitle(title)
    plt.title("Please close the figure for next figure", fontsize=6)
    plt.plot(X, X*slope + intercept, linecolor)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    fig.savefig(fileName)
    plt.show()


def costPlot (X, Y, Z, costData, delay=0.2, Xlabel="X", Ylabel="Y", Zlabel="Z", title="Cost Plot"):
    """
    Plots a 3D curve of XYZ
    """
    fig = plt.figure(3)
    ax = fig.gca(projection='3d')
    plt.suptitle(title)
    ax.set_xlabel(Xlabel)
    ax.set_ylabel(Ylabel)
    ax.set_zlabel(Zlabel)

    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha=0.5)

    E = costData.shape[0]
    for i in range(E):
        ax.scatter(costData[i][1], costData[i][2], costData[i][0], marker='o', color='k')
        plt.pause(delay)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5)
    plt.show()


def contourPlot (X, Y, Z, costData, delay=0.2, Xlabel="X", Ylabel="Y", Zlabel="Z", title="Contour Plot"):
    """
    Plots a contour curve of XYZ
    """
    fig = plt.figure(4)
    plt.suptitle(title)
    plt.title("Please don't close the figure until animation completes", fontsize=6)
    plt.contour(X,Y,Z,[0.00001, 0.00005, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1])
    
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    
    E = costData.shape[0]
    for i in range(E):
        plt.scatter(costData[i][1], costData[i][2], marker='o', color='b')
        plt.pause(delay)

    plt.show(block=False)
