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
    plot.scatterPlot(data[0], data[1], "X", "Y", marker='bx', title='Raw Data')

    if (sys.argv[1] == '1'):
        # Unweighted linear regression
        Theta = model.trainUnweighted(data[0], data[1])
        plot.regressionPlot(data[0], data[1], Theta[1,0], Theta[0,0], "X", "Y", "bx", fileName="plots/curve.png", title="Unweighted")
    elif (sys.argv[1] == '2'):
        # Weighted linear regression with tau = 0.8
        model.displayWeighted(data[0], data[1])
