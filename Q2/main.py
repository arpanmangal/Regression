"""
Main controller
"""

import plot as plot
import read as ld
import model as model
import sys

if __name__ == '__main__':
    if (len(sys.argv) < 4):
        print("Please run the program with proper command line arguments")
        exit(1)

    # Load data
    data = ld.loadData(sys.argv[1], sys.argv[2])

    # visualise data
    plot.scatterPlot(data[0], data[1], "X", "Y", marker='bx', title='Raw Data')

    # Unweighted linear regression
    Theta = model.trainUnweighted(data[0], data[1])
    plot.regressionPlot(data[0], data[1], Theta[1,0], Theta[0,0], "X", "Y", "bx", fileName="Q2/plots/unweighted.png", title="Unweighted")
    
    # Weighted linear regression with tau = 0.8
    model.displayWeighted(data[0], data[1], float(sys.argv[3]))