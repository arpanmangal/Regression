"""
Main controller
"""

import plot as plot
import read as ld
import model as model
import sys

if __name__ == '__main__':
    if (len(sys.argv) < 5):
        print("Please run the program with proper command line arguments")
        exit(1)

    # Load data
    data = ld.loadData(sys.argv[1], sys.argv[2])

    # visualise data
    plot.scatterPlot(data[0], data[1], "Acidity", "Density of Wine", marker='bx', title="Raw Data")

    # Train data
    Theta = model.train(data[0], data[1], learning_rate=float(sys.argv[3]), delay=float(sys.argv[4]))
