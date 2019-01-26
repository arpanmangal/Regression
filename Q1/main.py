"""
Main controller
"""

import plot as plot
import read as ld
import model as model
import sys

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Please run the program with proper command line arguments")
        exit(1)

    # Load data
    data = ld.loadData()

    # visualise data
    # plot.scatterPlot(data[0], data[1], "Acidity", "Density of Wine", marker='bx')

    if (sys.argv[1] == 'curve'):
        # Train data
        Theta = model.train(data[0], data[1])
        print("theta0: %.6f | theta1: %.6f" % (Theta[0], Theta[1]))
    elif (sys.argv[1] == 'bowl'):
        # Plot J(theta) curve
        model.bowlCurve(data[0], data[1])
    else:
        print("Please run the program with proper command line arguments")
        exit(1)