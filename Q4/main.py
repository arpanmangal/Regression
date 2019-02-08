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
    (X, Y) = ld.loadData(sys.argv[1], sys.argv[2])

    # visualise data
    # plot.scatterPlot(X, Y, "Fresh", "Marine", title='Raw Data', fileName='Q4/plots/raw.png')

    # Train the GDA model
    model.trainGDA (X, Y, int(sys.argv[3]))