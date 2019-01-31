"""
Main controller
"""

import plot as plot
import read as ld
import model as model
import sys

if __name__ == '__main__':
    # Load data
    (X, Y) = ld.loadData()

    # visualise data
    plot.scatterPlot(X, Y, "x1", "x2", title='Raw Data', fileName='plots/raw.png')

    # Train the logistic regresssion
    Theta = model.trainLogistic(X, Y)
    print(Theta)

    slope = -Theta[1,0]/Theta[2,0]
    intercept = -Theta[0,0]/Theta[2,0]

    plot.logisticPlot(X, Y, slope, intercept, "x1", "x2", title="Logistic Regression", fileName='plots/boundary.png')
