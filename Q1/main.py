"""
Main controller
"""

import plot as plot
import read as ld
import model as model

if __name__ == '__main__':

    # Load data
    data = ld.loadData()

    # visualise data
    plot.scatterPlot(data[0], data[1], "Acidity", "Density of Wine", marker='bx')

    # Train data
    Theta = model.train(data[0], data[1])
    print("theta0: %.6f | theta1: %.6f" % (Theta[0], Theta[1]))