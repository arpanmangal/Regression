"""
Main controller
"""

import plot as plot
import read as ld
import model as model
import sys

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print("Please run the program with proper command line arguments")
        exit(1)

    # Load data
    (X, Y) = ld.loadData(sys.argv[1], sys.argv[2])

    # visualise data
    plot.scatterPlot(X, Y, "x1", "x2", title='Raw Data', fileName='Q3/plots/raw.png')

    # Train the logistic regresssion
    Theta = model.trainLogistic(X, Y)
    print ("Theta0: %f | Theta1: %f | Theta2: %f" % (Theta[0], Theta[1], Theta[2]))

    slope = -Theta[1,0]/Theta[2,0]
    intercept = -Theta[0,0]/Theta[2,0]

    plot.logisticPlot(X, Y, slope, intercept, "x1", "x2", title="Logistic Regression", fileName='Q3/plots/boundary.png')
